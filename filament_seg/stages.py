"""Pipeline stages. Each stage reads and writes artifacts in ``cfg["paths"].out`` under fixed names (ART).

Training order (stages marked GPU need a CUDA device):
  folds -> prep -> semantic (GPU, 5 folds + hold-out) -> yolo (GPU, 5 folds + hold-out) -> candidates
  -> crops -> stage1 (GPU, 5 folds) -> gong + ssl (GPU) -> residual (GPU, 5 folds) -> verifier -> evaluate
Inference of this base pipeline: infer (CPU is enough; about 2.5 h for the 180 test frames with 4 workers, see the README).
The final model (configs/final_yellow.yaml) does not use infer; its last two stages, refine-test (GPU) and final-cpu, are in
filament_seg/nlv.py (all stages: README, "Final model: stages, commands and checks").
"""
import json
import os
import pickle
import time
from multiprocessing import Pool
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import candidates, features, folds, io, preprocess, refiner, verifier
from .metrics import PQ

ART = dict(
    prep="prep1024.npz",
    v1_oof_prob="seg_v1oof_valprob.npy", v1_oof_names="seg_v1oof.json",
    yolo_oof="inst_a_s1536_oof_preds.pkl",
    cands="verifier_cands.parquet", cand_rles="verifier_cands_rles.pkl", gt_rles="verifier_gt_rles.pkl",
    crops="refiner_crops.npy", teacher="refiner_teacher.npy", crop_meta="refiner_meta.parquet",
    refined_rles="verifier_cands_refined_{tag}_rles.pkl", refined_table="verifier_cands_refined_{tag}.parquet",
    verifier_oof="verifier_gbdt_refined_{tag}_oof.parquet", verifier_full="verifier_gbdt_refined_{tag}_full.joblib",
    verifier_oof_orig="verifier_gbdt_oof.parquet", verifier_full_orig="verifier_gbdt_full.joblib",
    test_names="test_names.npy", test_v1_prob="test_v1_prob.npy", test_yolo="inst_yolo_a_test_preds.pkl",
    test_cands="verifier_test_cands_v1.parquet", test_cand_rles="verifier_test_rles_v1.pkl",
    test_refiner_prob="refiner_{tag}_test_prob.npy", test_refined_rles="refiner_{tag}_test_rles.pkl",
)
G = {}  # per-worker state of multiprocessing pools


def art(cfg, key, **kw) -> Path:
    return cfg["paths"].out / ART[key].format(tag=cfg["refiner"]["tag"], **kw)


def _map(fn, jobs, workers, init=None, initargs=()):
    """Ordered parallel map; workers <= 1 runs in-process (useful for tests and debugging)."""
    if workers <= 1:
        if init:
            init(*initargs)
        return [fn(j) for j in jobs]
    with Pool(workers, initializer=init, initargs=initargs) as pool:
        return pool.map(fn, jobs, chunksize=2)


# ----------------------------------------------------------------------------- data preparation

def stage_folds(cfg):
    """Write the 5-fold split and the hold-out split of the physical frames (folds.make_kfold / make_holdout) to out/."""
    stems = sorted(io.group_by_physical(io.load_coco(cfg["paths"].train_json)))
    cfg["paths"].out.mkdir(parents=True, exist_ok=True)
    folds.make_kfold(stems, cfg["paths"].out)
    folds.make_holdout(stems, cfg["paths"].out)


def stage_prep(cfg):
    """Physical 3-channel inputs and annotator-mean soft targets for all 707 train images."""
    p = cfg["paths"]
    phys = io.group_by_physical(io.load_coco(p.train_json))
    names = sorted(phys)
    res = cfg["semantic"]["res"]
    images = np.zeros((len(names), 3, res, res), np.uint8)
    soft = np.zeros((len(names), res, res), np.uint8)
    for i, n in enumerate(names):
        images[i] = preprocess.make_input(io.read_gray(p.train_images / phys[n]["file_name"]), res)
        soft[i] = preprocess.soft_target(phys[n], res)
    np.savez(p.out / ART["prep"], images=images, soft=soft, names=np.array(names))


# ----------------------------------------------------------------------------- base models

def stage_semantic(cfg, split_name: str, tag: str):
    """Train the U-Net++ on one split (split_f{k}.json or split_v1.json)."""
    from . import semantic
    p, c = cfg["paths"], cfg["semantic"]
    d = np.load(p.out / ART["prep"])
    names = [str(n) for n in d["names"]]
    pos = {n: i for i, n in enumerate(names)}
    split = folds.load_split(p.out, split_name)
    tr = np.array([pos[n] for n in split["train"]])
    va = np.array([pos[n] for n in split["val"]])
    phys = io.group_by_physical(io.load_coco(p.train_json))
    va_gt = [io.gt_sets(phys[names[i]]) for i in va]
    best, probs = semantic.train(d["images"], d["soft"], tr, va, va_gt, p.out / f"seg_{tag}.pt", encoder=c["encoder"],
                                 epochs=c["epochs"], bs=c["bs"], lr=c["lr"], eval_every=c["eval_every"])
    np.save(p.out / f"seg_{tag}_valprob.npy", probs.astype(np.float16))
    json.dump({"tag": tag, "best_pq": best, "val_names": [names[i] for i in va]}, open(p.out / f"seg_{tag}.json", "w"))


def stage_semantic_oof(cfg, prefix="v1f"):
    """Stack the five fold-validation probability maps into one OOF array sorted by image name."""
    out = cfg["paths"].out
    names, probs = [], []
    for k in range(5):
        names += json.load(open(out / f"seg_{prefix}{k}.json"))["val_names"]
        probs.append(np.load(out / f"seg_{prefix}{k}_valprob.npy"))
    P = np.concatenate(probs)
    order = np.argsort(names)
    np.save(out / ART["v1_oof_prob"], P[order].astype(np.float16))
    json.dump({"val_names": [names[i] for i in order]}, open(out / ART["v1_oof_names"], "w"))


def stage_yolo(cfg, split_name: str, run: str, device="0"):
    """Base pipeline: train YOLO11s-seg on one split (GPU) as run `run` and store its predictions on the validation frames."""
    from . import yolo
    p, c = cfg["paths"], cfg["yolo"]
    split = folds.load_split(p.out, split_name)
    data = yolo.make_dataset(p.train_json, p.train_images, split, p.out / f"yolo_ds_{run}")
    w = yolo.train(data, p.out / "yolo", run, c["model"], c["imgsz"], c["epochs"], c["batch"], device)
    phys = io.group_by_physical(io.load_coco(p.train_json))
    preds = yolo.predict(w, [p.train_images / phys[n]["file_name"] for n in split["val"]], c["imgsz"], c["conf"], c["iou"], c["max_det"], device)
    pickle.dump(preds, open(p.out / f"yolo_{run}_val_preds.pkl", "wb"))


def stage_yolo_oof(cfg, prefix="a_s1536_f"):
    """Base pipeline: merge the validation predictions of the five fold runs (yolo_<prefix>k_val_preds.pkl) into the OOF predictions."""
    out = cfg["paths"].out
    allp = {}
    for k in range(5):
        allp.update(pickle.load(open(out / f"yolo_{prefix}{k}_val_preds.pkl", "rb")))
    pickle.dump(allp, open(out / ART["yolo_oof"], "wb"))


# ----------------------------------------------------------------------------- candidates and crops (OOF)

def _init_train(cfg):
    p = cfg["paths"]
    G["cfg"] = cfg
    G["phys"] = io.group_by_physical(io.load_coco(p.train_json))
    G["probs"] = np.load(p.out / ART["v1_oof_prob"], mmap_mode="r")
    G["pos"] = {n: i for i, n in enumerate(json.load(open(p.out / ART["v1_oof_names"]))["val_names"])}
    for key in ("yolo_oof", "cand_rles", "gt_rles"):
        f = p.out / ART[key]
        G[key] = pickle.load(open(f, "rb")) if f.exists() else None


def _train_raw(name):
    return io.read_gray(G["cfg"]["paths"].train_images / G["phys"][name]["file_name"])


def _oof_prob(name):
    p = np.asarray(G["probs"][G["pos"][name]], dtype=np.float32)
    return p / 255.0 if p.max() > 1.5 else p


def _cands_work(name):
    c = G["cfg"]["candidates"]
    p2048 = candidates.upsample_prob(_oof_prob(name))
    raw = _train_raw(name)
    cl = candidates.collect(p2048, G["yolo_oof"].get(name, []), tuple(c["bins"]), c["min_area"], c["yolo_conf_min"])
    gts = io.gt_sets(G["phys"][name])
    rows, rles, _ = features.featurize(name, p2048, raw, cl, gts, cluster_iou=c["cluster_iou"])
    return name, rows, rles, gts


def stage_candidates(cfg, workers=4):
    """OOF candidate union of the 707 train images with features and verifier targets."""
    out = cfg["paths"].out
    names = json.load(open(out / ART["v1_oof_names"]))["val_names"]
    res = _map(_cands_work, names, workers, _init_train, (cfg,))
    fold = folds.fold_of(out)
    df = pd.DataFrame([r for _, rows, _, _ in res for r in rows])
    df["fold"] = df.image.map(fold)
    df.to_parquet(out / ART["cands"])
    pickle.dump({n: rl for n, _, rl, _ in res}, open(out / ART["cand_rles"], "wb"))
    pickle.dump({n: g for n, _, _, g in res}, open(out / ART["gt_rles"], "wb"))


def _crops_work(args):
    name, g = args
    c = G["cfg"]["refiner"]
    R = G["cand_rles"][name]
    rles = [R[i] for i in g.idx]
    seeds_src = g.source.tolist()
    teachers, best_gt_iou = refiner.teacher_for_seeds(rles, G["gt_rles"][name])
    t_masks = [io.decode(t).astype(np.uint8) * 255 if t is not None else np.zeros((io.H, io.W), np.uint8) for t in teachers]
    bboxes = list(zip(g.x, g.y0, g.w, g.h))
    X, boxes, pidx, (T,) = refiner.make_crops(_train_raw(name), _oof_prob(name), rles, seeds_src, bboxes, c["crop_size"],
                                              c["crop_scale"], c["crop_min_side"], extra=(t_masks,))
    _, piou, _ = refiner.partners(rles, seeds_src)
    meta = []
    areas, max_ious = g.area.values, g.max_iou.values
    for k in range(len(g)):
        t_area = int(io.mask_utils.area(teachers[k])) if teachers[k] is not None else 0
        pk = int(pidx[k])
        meta.append(dict(x0=int(boxes[k, 0]), y0=int(boxes[k, 1]), side=int(boxes[k, 2]), partner_idx=int(g.idx.values[pk]) if pk >= 0 else -1,
                         partner_iou=float(piou[k]), teacher_area=t_area,
                         w_hard=refiner.stage1_weights(int(areas[k]), float(max_ious[k]), float(best_gt_iou[k]), pk,
                                                       float(best_gt_iou[pk]) if pk >= 0 else 0.0, int(areas[pk]) if pk >= 0 else 0, t_area)))
    return g.index.values, X, T, meta


def stage_crops(cfg, workers=4):
    """256 x 256 refiner crops, teachers and stage-1 weights for every OOF candidate (row order = candidate table)."""
    out = cfg["paths"].out
    df = pd.read_parquet(out / ART["cands"]).reset_index(drop=True)
    S = cfg["refiner"]["crop_size"]
    X = np.lib.format.open_memmap(out / ART["crops"], mode="w+", dtype=np.uint8, shape=(len(df), 4, S, S))
    T = np.lib.format.open_memmap(out / ART["teacher"], mode="w+", dtype=np.uint8, shape=(len(df), S, S))
    meta = [None] * len(df)
    jobs = list(df.groupby("image", sort=False))
    for pos, x, t, m in _map(_crops_work, jobs, workers, _init_train, (cfg,)):
        X[pos], T[pos] = x, t
        for p_, mm in zip(pos, m):
            meta[p_] = mm
    X.flush()
    T.flush()
    md = pd.concat([df[["image", "idx", "source", "fold", "conf", "area", "y", "max_iou", "cluster", "cluster_n_src"]], pd.DataFrame(meta)], axis=1)
    md.to_parquet(out / ART["crop_meta"])


# ----------------------------------------------------------------------------- refiners (GPU)

def _fold_inference(cfg, meta, va, prob, tag, fold):
    """Paste the fold's refined crops back and record seed / refined IoU with the GT."""
    out = cfg["paths"].out
    R = pickle.load(open(out / ART["cand_rles"], "rb"))
    gt = pickle.load(open(out / ART["gt_rles"], "rb"))
    rles, rows = {}, []
    for k, gi in enumerate(va):
        m = meta.iloc[gi]
        seed_rle = R[m.image][int(m.idx)]
        ref, status = refiner.paste_back(prob[k], int(m.x0), int(m.y0), int(m.side), io.decode(seed_rle), cfg["refiner"].get("paste_thr", 0.5))
        rle = io.encode(ref) if status == "ok" else seed_rle
        rles.setdefault(m.image, {})[int(m.idx)] = rle
        rows.append(dict(image=m.image, idx=int(m.idx), source=m.source, status=status,
                         maxiou_seed=refiner.gt_metrics(seed_rle, gt[m.image])[0], maxiou_ref=refiner.gt_metrics(rle, gt[m.image])[0]))
    pickle.dump(rles, open(out / f"refiner_{tag}_f{fold}_rles.pkl", "wb"))
    pd.DataFrame(rows).to_parquet(out / f"refiner_{tag}_f{fold}_cands.parquet")


def _load_rows(cfg, idx, extra=True):
    """Crops (4 planes, plus the planes listed in refiner.extra_crops, e.g. the nnU-Net map of configs/nlv.yaml (nlv.stage_nnr_crops)
    or the two NNRP planes of configs/stk.yaml (refplanes.stage_nnrp_planes, (N, 2, S, S))) and teachers."""
    import torch
    out = cfg["paths"].out
    Xm = np.load(out / ART["crops"], mmap_mode="r")
    Tm = np.load(out / ART["teacher"], mmap_mode="r")
    X = np.ascontiguousarray(Xm[idx])
    if extra and cfg["refiner"].get("extra_crops"):
        ex = [np.ascontiguousarray(np.load(out / f, mmap_mode="r")[idx]) for f in cfg["refiner"]["extra_crops"]]
        X = np.concatenate([X] + [e[:, None] if e.ndim == 3 else e for e in ex], 1)
    return torch.from_numpy(X), torch.from_numpy(np.ascontiguousarray(Tm[idx]))


def stage_stage1(cfg, fold: int):
    """Base pipeline: train the stage-1 refiner ("A2", plain 4-channel U-Net) of one fold on the crops of the other folds (GPU); its
    masks on the held-out fold define two terms of the sample weight of the residual refiner (refiner.residual_weights)."""
    import torch
    c = cfg["refiner"]["stage1"]
    meta = pd.read_parquet(cfg["paths"].out / ART["crop_meta"])
    tr = np.where(meta.fold.values != fold)[0]
    va = np.where(meta.fold.values == fold)[0]
    X, T = _load_rows(cfg, tr, extra=False)
    model = refiner.train_stage1(X, T, meta.w_hard.values[tr], c["epochs"], c["bs"], c["lr"], c["max_minutes"])
    torch.save(dict(state=model.state_dict(), fold=fold), cfg["paths"].out / f"refiner_{c['tag']}_f{fold}.pt")
    Xv, _ = _load_rows(cfg, va, extra=False)
    prob = refiner.predict_stage1(model, Xv, torch.device("cuda"))
    _fold_inference(cfg, meta, va, prob, c["tag"], fold)


def stage_gong(cfg):
    """Download the unlabeled GONG H-alpha frames for SSL to <data>/../gong_unlabeled, excluding every train and test frame."""
    from . import gong
    p = cfg["paths"]
    exclude = {Path(i["file_name"]).stem for i in io.load_coco(p.train_json)["images"]} | {f.stem for f in io.list_images(p.test_images)}
    gong.download(p.data.parent / "gong_unlabeled", exclude)


def stage_ssl(cfg):
    """SimMIM pre-training on the unlabeled GONG frames of stage `gong` (GPU); the checkpoint initialises the encoder of the residual refiner."""
    from . import ssl
    c = cfg["ssl"]
    from . import gong
    files = sorted((cfg["paths"].data.parent / "gong_unlabeled").glob("*.jpg"))
    files = gong.clean_frames(files, [f.stem for f in io.list_images(cfg["paths"].test_images)])  # host rule: no frame within +-1 h of test
    ssl.pretrain(files, cfg["paths"].out / c["encoder"], c["steps"], c["bs"], c["lr"])


def _residual_meta(cfg):
    out = cfg["paths"].out
    meta = pd.read_parquet(out / ART["crop_meta"])
    s1 = (cfg["refiner"].get("stage1") or {}).get("tag")
    if s1 is None:  # final-model configs: no stage-1 refiner was trained on this candidate table, so both stage-1 terms of residual_weights are 0
        return meta.assign(maxiou_seed=meta.max_iou.values, maxiou_ref=meta.max_iou.values)
    a2 = pd.concat([pd.read_parquet(out / f"refiner_{s1}_f{k}_cands.parquet") for k in range(5)])[["image", "idx", "maxiou_seed", "maxiou_ref"]]
    m = meta.merge(a2, on=["image", "idx"], how="left")
    assert len(m) == len(meta)
    return m


def residual_rows(meta: pd.DataFrame, fold: int) -> tuple:
    """(training rows, validation rows) of the residual refiner: fold k >= 0 = rows of the other folds / of fold k;
    fold -1 = full-data refiner (configs/final_yellow.yaml, the final model; configs/final_fullfit.yaml) = every row of the 707 training frames, no validation rows."""
    if fold < 0:
        return np.arange(len(meta)), np.zeros(0, np.int64)
    return np.where(meta.fold.values != fold)[0], np.where(meta.fold.values == fold)[0]


def stage_residual(cfg, fold: int, seed: int = -1, tag: str = None, member: str = None):
    """Residual refiner of one fold (GPU) -> out/refiner_<tag>_f<fold>.pt and its OOF masks; fold -1 = full-data refiner
    (all 707 frames, checkpoint refiner_<tag>_f-1.pt, no OOF output). seed / tag: see refiner.train_residual.
    member: tag of an entry of refiner.members (configs/final_yellow.yaml, e.g. FFNNRPx16; configs/final_fullfit.yaml): its `seed` is used, and a member without
    test_feats (NNRP) is trained without the S15F feature crops."""
    import torch
    c = cfg["refiner"]
    if member:
        m = [x for x in c.get("members") or [] if x.get("tag") == member]
        assert len(m) == 1, f"refiner.members has no single member {member}"
        tag, seed = member, int(m[0].get("seed", -1))
        if not m[0].get("test_feats"):
            c = dict(c, feat_crops=None)
    tag = tag or c["tag"]
    meta = _residual_meta(cfg)
    w, cpq, S2 = refiner.residual_weights(meta, c["protect"])
    tr, va = residual_rows(meta, fold)
    X, T = _load_rows(cfg, tr)
    FT = FV = None
    if c.get("feat_crops"):  # S15F (configs/s15f.yaml): Swin-L 1536 feature crops of the same rows (filament_seg/s15f.py)
        from . import s15f
        fpath = cfg["paths"].out / c["feat_crops"]
        FT = s15f.load_feats(fpath, tr, lazy=c.get("feat_lazy", False))
        FV = s15f.load_feats(fpath, va, lazy=c.get("feat_lazy", False)) if len(va) else None
    init = cfg["paths"].out / cfg["ssl"]["encoder"] if c["ssl_init"] else None
    model = refiner.train_residual(X, T, torch.from_numpy(S2[tr]), w[tr], cpq[tr], c["alpha"], c["epochs"], c["bs"], c["lr"], init, FT=FT,
                                   seed=seed)
    torch.save(dict(state=model.state_dict(), fold=fold, epochs=c["epochs"]), cfg["paths"].out / f"refiner_{tag}_f{fold}.pt")
    if not len(va):  # full-data refiner: no held-out rows (test path: nlv.refine_test with a `full` member)
        return
    Xv, _ = _load_rows(cfg, va)
    prob = refiner.predict_residual(model, Xv, torch.from_numpy(S2[va]), torch.device("cuda"), bs=64, FT=FV)
    if c.get("oof_prob_u8"):  # the S15F OOF masks were pasted back from probabilities stored as uint8 (round(p * 255))
        prob = np.round(prob * 255).astype(np.uint8).astype(np.float32) / 255.0
    _fold_inference(cfg, meta, va, prob, tag, fold)


# ----------------------------------------------------------------------------- verifier and OOF evaluation

def _init_refined(cfg, ref):
    _init_train(cfg)
    G["ref"] = ref


def _refined_feat_work(args):
    name, items = args  # [(idx, source, conf)]
    p2048 = candidates.upsample_prob(_oof_prob(name))
    R, ref = G["cand_rles"][name], G["ref"].get(name, {})
    cl, out_rles = [], []
    for i, s, c in items:
        r = ref.get(i, R[i])
        if io.decode(r).sum() == 0:
            r = R[i]
        cl += candidates.recollect(p2048, [(s, c, r)])
        out_rles.append(r)
    rows, _, _ = features.featurize(name, p2048, _train_raw(name), cl, G["gt_rles"][name])
    for row, (i, _, _) in zip(rows, items):
        row["idx"] = i
    return name, rows, out_rles


def stage_verifier(cfg, workers=4):
    """Collect the refined OOF masks and fit the GBDT verifier on image folds (OOF) and on all 707 images (test).

    verifier.features = "original": the GBDT sees the features of the candidates before refinement (champion);
    "refined": the features are recomputed on the refined masks (the earlier V2bS submission).
    """
    out, tag = cfg["paths"].out, cfg["refiner"]["tag"]
    ref = {}
    for k in range(5):
        ref.update(pickle.load(open(out / f"refiner_{tag}_f{k}_rles.pkl", "rb")))
    df = pd.read_parquet(out / ART["cands"])
    jobs = [(n, [(int(r.idx), r.source, float(r.conf)) for r in g.itertuples()]) for n, g in df.groupby("image", sort=False)]
    res = _map(_refined_feat_work, jobs, workers, _init_refined, (cfg, ref))
    rf = pd.DataFrame([r for _, rows, _ in res for r in rows])
    rf["fold"] = rf.image.map(folds.fold_of(out))
    rf.to_parquet(art(cfg, "refined_table"))
    pickle.dump({n: {int(r["idx"]): rl for r, rl in zip(rows, rls)} for n, rows, rls in res}, open(art(cfg, "refined_rles"), "wb"))
    gbdt = cfg["verifier"]["gbdt"]
    if cfg["verifier"]["features"] == "original":
        df["yhat_gbdt"], _ = verifier.fit_oof(df, gbdt)
        df[["image", "idx", "source", "fold", "y", "max_iou", "yhat_gbdt"]].to_parquet(out / ART["verifier_oof_orig"])
        joblib.dump(verifier.fit_full(df, gbdt), out / ART["verifier_full_orig"])
    else:
        rf["yhat_new"], _ = verifier.fit_oof(rf, gbdt)
        rf[["image", "idx", "source", "fold", "y", "max_iou", "yhat_new"]].to_parquet(art(cfg, "verifier_oof"))
        joblib.dump(verifier.fit_full(rf, gbdt), art(cfg, "verifier_full"))


def oof_scores(cfg) -> pd.DataFrame:
    """OOF verifier scores (columns image, idx, score) in candidate order, for the configured verifier."""
    out = cfg["paths"].out
    if cfg["verifier"]["features"] == "original":
        return pd.read_parquet(out / ART["verifier_oof_orig"])[["image", "idx", "yhat_gbdt"]].rename(columns={"yhat_gbdt": "score"})
    return pd.read_parquet(art(cfg, "verifier_oof"))[["image", "idx", "yhat_new"]].rename(columns={"yhat_new": "score"})


def _init_eval(rles, gt):
    G["rl"], G["gt"] = rles, gt


def _eval_work(args):
    name, idx, scores, tau, d = args
    pred = verifier.decode_image([G["rl"][name][i] for i in idx], scores, tau, d["nms_iou"], d["keep_frac"])
    acc = PQ()
    for g in G["gt"][name]:
        acc.add(g, pred)
    return acc


def stage_evaluate(cfg, taus=None, workers=4) -> dict:
    """OOF PQ of the full pipeline on the 707 train images (official-equivalent metric, records pooled)."""
    out, d = cfg["paths"].out, cfg["decode"]
    rl = pickle.load(open(art(cfg, "refined_rles"), "rb"))
    gt = pickle.load(open(out / ART["gt_rles"], "rb"))
    oof = oof_scores(cfg)
    results = {}
    for tau in taus or [d["tau"]]:
        jobs = [(n, g.idx.astype(int).tolist(), g.score.astype(float).tolist(), tau, d) for n, g in oof.groupby("image", sort=False)]
        total = PQ()
        for acc in _map(_eval_work, jobs, workers, _init_eval, (rl, gt)):
            total.tp_iou += acc.tp_iou
            total.fp += acc.fp
            total.fn += acc.fn
        for n in set(gt) - set(oof.image):  # images without any candidate: all their GT are misses
            for g in gt[n]:
                total.add(g, [])
        results[tau] = total.result()
    return results


# ----------------------------------------------------------------------------- test-time inference

def _init_test(cfg, probs, yolo_preds, cand_rles=None):
    G["cfg"], G["probs"], G["yolo"], G["cand_rles"] = cfg, probs, yolo_preds, cand_rles


def _test_raw(name):
    test_dir = G["cfg"]["paths"].test_images
    f = test_dir / f"{name}.jpeg"
    return io.read_gray(f if f.exists() else test_dir / f"{name}.jpg")


def _test_cands_work(name):
    c = G["cfg"]["candidates"]
    p2048 = candidates.upsample_prob(G["probs"][name])
    cl = candidates.collect(p2048, G["yolo"].get(name, []), tuple(c["bins"]), c["min_area"], c["yolo_conf_min"])
    rows, rles, _ = features.featurize(name, p2048, _test_raw(name), cl, [], cluster_iou=c["cluster_iou"])
    return name, rows, rles


def _test_crops_work(args):
    name, g = args
    c = G["cfg"]["refiner"]
    R = G["cand_rles"][name]
    X, boxes, _, _ = refiner.make_crops(_test_raw(name), G["probs"][name], [R[i] for i in g.idx], g.source.tolist(),
                                        list(zip(g.x, g.y0, g.w, g.h)), c["crop_size"], c["crop_scale"], c["crop_min_side"])
    return g.index.values, X, boxes


def _test_refined_feat_work(args):
    name, items = args  # [(idx, source, conf, refined RLE)]
    p2048 = candidates.upsample_prob(G["probs"][name])
    cl = candidates.recollect(p2048, [(s, c, r) for _, s, c, r in items])
    rows, _, _ = features.featurize(name, p2048, _test_raw(name), cl, [])
    for row, it in zip(rows, items):
        row["idx"] = it[0]
    return rows


def efficiency(cfg) -> dict:
    """Speed options of the base pipeline (`efficiency` section). Missing section or keys = the submitted runs, bit for bit."""
    e = dict(yolo_conf=None, refiner_bs=16)
    e.update(cfg.get("efficiency") or {})
    return e


def _refine_chunk(args):
    X, S2, cks, kind, alpha, device, bs = args
    import torch
    torch.set_num_threads(1)
    if kind == "stage1":  # plain 4-channel refiner (the A2 submission)
        return refiner.ensemble_stage1(cks, X, device)
    return refiner.ensemble_residual(cks, X, S2, alpha, device, bs)


def infer(cfg, weights: Path, out_csv: Path, workers=4, device="cpu", limit=0, log=print, return_state=False, extra_planes=None, refiner_device=None):
    """Test frames -> submission CSV. weights: folder holding the checkpoints listed in configs (see weights/README.md).

    return_state=True stops before the decode and returns the intermediate test artifacts.
    extra_planes(df, boxes) -> (N, k, S, S) uint8: further crop planes of the refiner.
    refiner_device: device of the refiner (default: device). With refiner.infer_bs set, the refiner runs in this process
    over all candidates in batches of infer_bs; otherwise in CPU worker chunks.
    """
    import torch
    from . import semantic, submission, yolo
    p, c = cfg["paths"], cfg["refiner"]
    files = io.list_images(p.test_images)[: limit or None]
    names = [f.stem for f in files]
    t0, timings = time.time(), {}

    def lap(key):
        timings[key] = round(time.time() - t0 - sum(timings.values()), 1)
        log(f"{key}: {timings[key]} s")

    # 1. full-disk U-Net++ (the hold-out model, no TTA, as in training); probabilities are stored in fp16 as during development
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
    model = semantic.build_model(cfg["semantic"]["encoder"], pretrained=False)
    model.load_state_dict(torch.load(weights / cfg["weights"]["semantic"], map_location="cpu"))
    dev = torch.device(device)
    probs = {}
    for f in files:
        x = preprocess.make_input(io.read_gray(f), cfg["semantic"]["res"])[None]
        probs[f.stem] = semantic.predict(model.to(dev), x, dev, amp=False)[0].astype(np.float16)  # fp32 forward pass (amp=False), stored as fp16, as submitted
    lap("semantic")
    # 2. YOLO11s-seg candidates
    y = cfg["yolo"]
    eff = efficiency(cfg)
    yolo_preds = yolo.predict(weights / cfg["weights"]["yolo"], files, y["imgsz"], eff["yolo_conf"] or y["conf"], y["iou"], y["max_det"], device)
    lap("yolo")
    # 3. candidate union + features (cluster_n_src feeds the refiner's source planes)
    res = _map(_test_cands_work, names, workers, _init_test, (cfg, probs, yolo_preds))
    df = pd.DataFrame([r for _, rows, _ in res for r in rows])
    cand_rles = {n: rl for n, _, rl in res}
    lap("candidates")
    # 4. crops and the 5-fold refiner (flip TTA x4; residual by default, plain stage-1 model for the A2 run)
    S = c["crop_size"]
    X = np.zeros((len(df), 4, S, S), np.uint8)
    boxes = np.zeros((len(df), 3), np.int64)
    for pos, x, b in _map(_test_crops_work, list(df.groupby("image", sort=False)), workers, _init_test, (cfg, probs, yolo_preds, cand_rles)):
        X[pos], boxes[pos] = x, b
    if extra_planes is not None:
        X = np.concatenate([X, extra_planes(df, boxes)], 1)
    is_y = (df.source == "yolo").values
    S2 = np.stack([is_y, is_y & (df.cluster_n_src.values == 1)], 1).astype(np.float32)
    cks = [weights / f for f in cfg["weights"]["refiner"]]
    if c.get("infer_bs"):
        prob = refiner.ensemble_residual(cks, X, S2, c["alpha"], refiner_device or device, c["infer_bs"])
    else:
        chunks = np.array_split(np.arange(len(df)), max(workers, 1))
        prob = np.concatenate(_map(_refine_chunk, [(X[ch], S2[ch], cks, c.get("kind", "residual"), c["alpha"], device, eff["refiner_bs"]) for ch in chunks], workers))
    lap("refiner")
    # 5. paste back, verify (on the original or the re-computed features), decode
    ref = {}
    for i, r in enumerate(df.itertuples()):
        seed_rle = cand_rles[r.image][int(r.idx)]
        m, st = refiner.paste_back(prob[i], int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), io.decode(seed_rle), c.get("paste_thr", 0.5))
        ref.setdefault(r.image, {})[int(r.idx)] = io.encode(m) if st == "ok" else seed_rle
    gbdt = joblib.load(weights / cfg["weights"]["verifier"])
    if cfg["verifier"]["features"] == "original":
        df["yhat_ref"] = verifier.predict(gbdt, df)
    else:
        jobs = [(n, [(int(r.idx), r.source, float(r.conf), ref[n][int(r.idx)]) for r in g.itertuples()]) for n, g in df.groupby("image", sort=False)]
        rf = pd.DataFrame([r for rows in _map(_test_refined_feat_work, jobs, workers, _init_test, (cfg, probs, yolo_preds)) for r in rows])
        rf["yhat"] = verifier.predict(gbdt, rf)
        df = df.merge(rf[["image", "idx", "yhat"]].rename(columns={"yhat": "yhat_ref"}), on=["image", "idx"], how="left")
    if return_state:
        return dict(names=names, files=files, probs=probs, cands=df, cand_rles=cand_rles, refined=ref, timings=timings, lap=lap)
    d = cfg["decode"]
    preds = {}
    for n in names:
        g = df[df.image == n]
        if len(g):
            preds[n] = [r["counts"].decode("ascii") for r in
                        verifier.decode_image([ref[n][int(i)] for i in g.idx], g.yhat_ref.astype(float).tolist(), d["tau"], d["nms_iou"], d["keep_frac"])]
    submission.write({n: v for n, v in preds.items() if v}, out_csv)
    lap("verify_decode")
    json.dump(timings, open(Path(out_csv).with_suffix(".timings.json"), "w"), indent=1)
    return out_csv


# ----------------------------------------------------------------------------- own2 maps (map features for the NLv selector of the final model)

def own_map_dir(cfg, mode: str, fold: int = None, views: int = 8) -> Path:
    """Our own maps: <out>/own_maps[_v1]/oof or .../test_f<k> (P-MATCH: one map per fold model at test)."""
    base = cfg["paths"].out / ("own_maps" if views == 8 else f"own_maps_v{views}")
    return base / ("oof" if mode == "oof" else f"test_f{fold}")


def stage_own_maps(cfg, mode: str, weights_dir: Path, device="cuda", train=False, views=8, limit=0):
    """Own 2048-px maps of the final model and its variants (filament_seg/ownmap.py). mode "oof": each training frame by the fold model that did
    not see it (with --train, missing fold models are trained first); mode "testfold": every test frame by every fold model,
    one folder per fold. Weights: <weights_dir>/own/own_<family>_f<k>.pt. Frames are read with OpenCV (grayscale), as in the
    submitted runs."""
    import cv2
    import torch
    from . import ownmap
    p = cfg["paths"]
    dev = torch.device(device)
    wd = weights_dir / "own"
    if mode == "oof":
        fold = folds.fold_of(p.out)
        recs = io.load_records(p.train_json)
        stems = sorted(fold)[: limit or None]
        imgs = {s: cv2.imread(str(p.train_images / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE) for s in sorted(fold)}
        for fam in ownmap.ENC:
            for k in range(5):
                wp = wd / f"own_{fam}_f{k}.pt"
                if not wp.exists():
                    assert train, f"missing {wp} (run with --train)"
                    tr = [s for s in sorted(fold) if fold[s] != k]
                    ownmap.train(imgs, {s: list(recs[s].values()) for s in tr}, tr, fam, wp)
                m = ownmap.load_model(wp, fam, dev)
                ownmap.write_maps(m, fam, imgs, [s for s in stems if fold[s] == k], own_map_dir(cfg, "oof", views=views), dev, views)
    else:
        files = io.list_images(p.test_images)[: limit or None]
        imgs = {f.stem: cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files}
        for fam in ownmap.ENC:
            for k in range(5):
                m = ownmap.load_model(wd / f"own_{fam}_f{k}.pt", fam, dev)
                ownmap.write_maps(m, fam, imgs, sorted(imgs), own_map_dir(cfg, "testfold", k, views), dev, views)
