"""NLv + NNR (configs/nlv.yaml): the CPU stages (verifier table, V-ENS, decode) and the stage `final-cpu`. The final model
(configs/final_yellow.yaml) uses the same candidates, votes, map features, verifier and decode; it differs in the refiner
(six full-data members), the C0 rule, tau 0.25, the order bonus and mask NMS 0.1, all set by its config.

Stage order of NLv + NNR (GPU stages produce the cached model outputs read here):
  1. candidates  nnU-Net map components (thresholds 0.3 / 0.4, area >= 50; filament_seg/nnunet_cands.py) united with the
                 Mask DINO Swin-L instances (filament_seg/maskdino_cands.py, arch swinl). Candidate features = FEATS_ALL
                 (filament_seg/features.py) with the nnU-Net components in place of the U-Net++ components.
  2. votes       8 features from the three Mask DINO R50 models s0 / s1 / s2 (filament_seg/votes.py).
  3. map feats   14 features on the own2 maps (filament_seg/mapfeats.py). Training rows: OOF maps; test rows: the map of
                 each of the five fold models (five feature tables per test candidate).
  4. verifier    V-ENS: HistGradientBoosting, CatBoost, XGBoost and ExtraTrees fitted once on all training rows
                 (FEATS_ALL + DFEATS + VFEATS = 70 features, target y). On the stacked five test tables the four scores
                 are turned into rank / N, averaged and mapped to the quantiles of the HistGradientBoosting scores;
                 the five values of a candidate (one per fold map) are averaged -> yhat.
  5. refiner     NNR: residual U-Net r34 with a 7th input plane = nnU-Net map (filament_seg/refiner.py, plane7 nnunet).
  6. decode      candidates with yhat > tau (0.28; final model 0.25) in decreasing yhat, mask NMS (IoU <= 0.3 with every kept mask; final model 0.1), then
                 pixels already owned are removed and a mask keeping < 50 % of its pixels is dropped (no overlaps).
                 Optional settings of the config section `decode` (DECODE_KNOBS, all off by default): order_bonus
                 (configs/final_yellow.yaml) and iom_max.

STK-avg5 + NNRP (configs/stk_nnrp.yaml; configs/stk.yaml is the same with the S15F refiner) uses the same stages with four additions, all switched on by the
config: a 4th voter (Mask DINO Swin-L 1536, votes.VFEATS_STK), the 7 g7_m2 columns (filament_seg/evofeats.py), V-ENS fitted
with seeds 0-4 and the five test scores averaged (verifier.seeds), and the C0 rule on the refined masks (refiner.c0_ratio,
apply_c0). Its refiner is NNRP (filament_seg/refplanes.py): NNR plus two planes with the best-matching Swin-L 1536 and
RF-DETR-Seg masks.

Port of scripts/pm_select_mdnlv.py, scripts/vens_test2.py (members) and scripts/mk_csv_tau.py of the development
repository; numbers are identical on the submitted runs (tests/test_golden_nlv.py, tests/test_golden_stk.py).
Everything here is single-process (no multiprocessing pool).
"""
import hashlib
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mU
from scipy.stats import rankdata

from . import evofeats, mapfeats, votes
from .features import FEATS_ALL

GB = dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0, random_state=0)
ENS = dict(
    cat=dict(depth=6, iterations=800, learning_rate=0.05, loss_function="RMSE", random_seed=0, thread_count=2, verbose=0),
    xgb=dict(tree_method="hist", max_depth=6, n_estimators=800, learning_rate=0.04, subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=2),
    et=dict(n_estimators=500, min_samples_leaf=20, max_features=0.5, random_state=0, n_jobs=2),
)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def feature_list(vote_cols=None, evo=False) -> list:
    """Verifier columns in the order of the submitted runs: NLv 70 (FEATS_ALL + DFEATS + VFEATS); STK 80 (VFEATS_STK + evofeats.FEATS)."""
    return FEATS_ALL + mapfeats.DFEATS + list(vote_cols or votes.VFEATS) + (evofeats.FEATS if evo else [])


def is_stk(c: dict) -> bool:
    """True for the STK configs (4th voter "sl" = Mask DINO Swin-L 1536 in votes.voters)."""
    return "sl" in c.get("voters", votes.VOTERS)


def cfg_feature_list(c: dict) -> list:
    """Verifier columns of this config: feature_list with the config's vote columns and, if `nlv.evo` is set, the g7_m2 columns."""
    return feature_list(votes.VFEATS_STK if is_stk(c) else votes.VFEATS, evo="evo" in c)


# ----------------------------------------------------------------------------- inputs

def resolve(cfg) -> dict:
    """The `nlv` section of the config with ${VAR} expanded (also in nested sections); SOLAR_ROOT defaults to the development
    folder above release/, SOLAR_REPO to this repository (the final CSV is stored in its outputs/)."""
    os.environ.setdefault("SOLAR_ROOT", str(Path(__file__).resolve().parents[3]))
    os.environ.setdefault("SOLAR_REPO", str(Path(__file__).resolve().parents[1]))
    if "nlv" not in cfg:
        raise SystemExit("this stage needs a config with an `nlv` section: pass --config configs/final_yellow.yaml "
                         "(the default configs/champion.yaml is the base pipeline and has none)")

    def ex(v):
        return os.path.expandvars(v) if isinstance(v, str) else {k: ex(x) for k, x in v.items()} if isinstance(v, dict) else v
    c = ex(cfg["nlv"])
    c["voters"] = tuple(cfg.get("votes", {}).get("voters", votes.VOTERS))
    return c


def train_table(c: dict, recompute_votes=False, recompute_maps=False, recompute_evo=False) -> pd.DataFrame:
    """Training rows = OOF candidate table + own2 OOF map features + OOF votes (+ g7_m2 columns for STK; merge order of the submitted runs)."""
    tr = pd.read_parquet(c["train_cands"])
    if recompute_maps:
        R = pickle.load(open(c["train_rles"], "rb"))
        F = pd.DataFrame([r for n, g in tr.groupby("image", sort=True)
                          for r in mapfeats.image_features(Path(c["train_maps"]), n, R[n], g.idx.astype(int).tolist(), fast=True)])
    else:
        F = pickle.load(open(c["train_mapfeats"], "rb"))["F"]
    tr = tr.merge(F, on=["image", "idx"])
    if recompute_votes:
        R = pickle.load(open(c["train_rles"], "rb"))
        P = {k: {} for k in c["voters"]}
        for k in c["voters"]:
            for f in range(5):
                P[k].update(pickle.load(open(os.path.expandvars(c["oof_voters"][k]).format(k=f), "rb")))
        V = votes.oof_votes(tr[["image", "idx"]], R, P, c["voters"])
    else:
        V = pd.read_parquet(c["train_votes"])
    tr = tr.merge(V, on=["image", "idx"])
    if "evo" in c:
        n0 = len(tr)
        E = evofeats.oof_features(c) if recompute_evo else pd.read_parquet(c["evo"]["train"])
        tr = tr.merge(evofeats.as_verifier_cols(E.drop_duplicates(["image", "idx"])), on=["image", "idx"], how="left")
        assert len(tr) == n0 and tr[evofeats.FEATS].notna().all().all(), "g7_m2 columns of the training rows"
    return tr


def test_tables(c: dict, fast=True, recompute_evo=False) -> pd.DataFrame:
    """Five stacked test tables (column k = fold of the own2 map): candidate features + votes (+ g7_m2 columns) + map features."""
    T0 = pd.read_parquet(c["test_cands"]); TR = pickle.load(open(c["test_rles"], "rb"))
    names = [str(x) for x in np.load(c["test_names"], allow_pickle=True)]
    P = {k: [pickle.load(open(os.path.expandvars(c["test_voters"][k]).format(k=f), "rb")) for f in range(5)] for k in c["voters"]}
    VT = votes.test_votes(T0, TR, names, P, c["voters"])
    T0 = T0.merge(VT, on=["image", "idx"])
    vf = votes.VFEATS_STK if is_stk(c) else votes.VFEATS; ef = []
    if "evo" in c:
        m0 = len(T0)
        E = evofeats.test_features(c) if recompute_evo else pd.read_parquet(c["evo"]["test"])
        T0 = T0.merge(evofeats.as_verifier_cols(E), on=["image", "idx"], how="left"); ef = evofeats.FEATS
        assert len(T0) == m0 and T0[ef].notna().all().all(), "g7_m2 columns of the test rows"
    base = T0[["image", "idx"] + FEATS_ALL + vf + ef]; TE = []
    for k in range(5):
        md = Path(c["test_maps"].format(k=k))
        fr = pd.DataFrame([r for n, g in T0.groupby("image") for r in mapfeats.image_features(md, n, TR[n], g.idx.astype(int).tolist(), fast=fast)])
        te = base.merge(fr[["image", "idx"] + mapfeats.DFEATS], on=["image", "idx"]); assert len(te) == len(T0), (len(te), len(T0))
        te["k"] = k; TE.append(te)
        log(f"[nlv] test map features fold {k}: d_f99 mean {te.d_f99.mean():.4f}")
    return pd.concat(TE, ignore_index=True)


def candidates_image(name, p1024, raw, nn_map, md_items, gt_sets, cfg) -> tuple:
    """Candidate union + FEATS_ALL of one frame (nnU-Net components U Mask DINO Swin-L), as src/verifier.build_image(comp_p=...).

    p1024: U-Net++ probability (1024; uint8 0..255 or float); raw: 2048 frame; nn_map: nnU-Net probability (2048, [0, 1]);
    md_items: [(score, RLE)]; gt_sets: annotator records ([] at test time). Returns (rows, rles)."""
    from . import candidates, features, nnunet_cands
    c = cfg["candidates"]
    p2048 = candidates.upsample_prob(p1024)
    cl = nnunet_cands.collect(p2048, nn_map, md_items, tuple(c["bins"]), c["min_area"], c["conf_min"])
    rows, rles, _ = features.featurize(name, p2048, raw, cl, gt_sets, cluster_iou=c["cluster_iou"])
    return rows, rles


def oof_candidates(cfg, names) -> tuple:
    """OOF candidate table of some training frames from the cached OOF model outputs (U-Net++ OOF probabilities,
    nnU-Net OOF maps, Swin-L OOF instances) and the training images (paths.data)."""
    import json
    from . import io, nnunet_cands
    c = resolve(cfg)
    vn = json.load(open(c["train_v1_names"]))["val_names"]; P = np.load(c["train_v1_prob"], mmap_mode="r")
    phys = io.group_by_physical(io.load_coco(cfg["paths"].train_json))
    SL = {}
    for k in range(5):
        d = pickle.load(open(c["train_swinl"].format(k=k), "rb")); SL.update({n: d[n] for n in names if n in d})
    rows, rles = [], {}
    for n in names:
        raw = io.read_gray(cfg["paths"].train_images / phys[n]["file_name"])
        r, rl = candidates_image(n, np.asarray(P[vn.index(n)]), raw, nnunet_cands.load_map(c["train_nnmaps"], n),
                                 [(float(s), x) for s, x in SL.get(n, [])], io.gt_sets(phys[n]), cfg)
        rows += r; rles[n] = rl
    return pd.DataFrame(rows), rles


# ----------------------------------------------------------------------------- V-ENS

def fit_members(Xtr, ytr, ens=None, seed=0):
    """The four V-ENS members fitted on all training rows.

    seed 0 = the submitted NLv members; seed s > 0 sets random_state / random_seed of all four members to s
    (STK-avg5 averages the test scores of seeds 0-4)."""
    # import order of the submitted run (sklearn before catboost / xgboost): the other order segfaults on macOS (two OpenMP runtimes)
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor  # noqa: I001
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    e = {**ENS, **(ens or {})}
    gb = GB
    if seed:
        gb = dict(GB, random_state=seed)
        e = dict(cat=dict(e["cat"], random_seed=seed, allow_writing_files=False), xgb=dict(e["xgb"], random_state=seed),
                 et=dict(e["et"], random_state=seed))
    return dict(gbdt=HistGradientBoostingRegressor(**gb).fit(Xtr, ytr), cat=CatBoostRegressor(**e["cat"]).fit(Xtr, ytr),
                xgb=XGBRegressor(**e["xgb"]).fit(Xtr, ytr), et=ExtraTreesRegressor(**e["et"]).fit(Xtr, ytr))


def predict_vens(models: dict, X) -> np.ndarray:
    """Rank average of the four members on the rows X, mapped to the quantiles of the clipped HistGradientBoosting scores."""
    P = {"gbdt": np.clip(models["gbdt"].predict(X), 0, 1)}
    for m in ("cat", "xgb", "et"):
        P[m] = models[m].predict(X)
    n = len(X); r = np.mean([rankdata(P[m]) / n for m in ("gbdt", "cat", "xgb", "et")], axis=0)
    return np.quantile(P["gbdt"], np.clip(r, 0, 1))


def save_members(models: dict, out_dir: Path) -> dict:
    """Final verifier weights: verifier_nlv_{gbdt.joblib, cat.cbm, xgb.json, et.joblib}."""
    import joblib
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True); p = {}
    for k, m in models.items():
        p[k] = out_dir / f"verifier_nlv_{k}{dict(gbdt='.joblib', cat='.cbm', xgb='.json', et='.joblib')[k]}"
        if k in ("gbdt", "et"):
            joblib.dump(m, p[k], compress=3)
        else:
            m.save_model(str(p[k]))
    return p


def load_members(d: Path) -> dict:
    """Load the four fitted V-ENS members written by save_members from directory d."""
    import joblib
    import sklearn.ensemble  # noqa: F401  (sklearn before catboost / xgboost, see fit_members)
    from catboost import CatBoostRegressor
    from xgboost import XGBRegressor
    d = Path(d); cat = CatBoostRegressor(); cat.load_model(str(d / "verifier_nlv_cat.cbm")); xgb = XGBRegressor(); xgb.load_model(str(d / "verifier_nlv_xgb.json"))
    return dict(gbdt=joblib.load(d / "verifier_nlv_gbdt.joblib"), cat=cat, xgb=xgb, et=joblib.load(d / "verifier_nlv_et.joblib"))


def score_test(models: dict, te: pd.DataFrame, feats: list) -> pd.DataFrame:
    """yhat per test candidate = mean over the five fold-map tables (ranks taken over all five tables together)."""
    te = te.copy(); te["yhat_k"] = predict_vens(models, te[feats].values)
    return te.groupby(["image", "idx"], as_index=False).yhat_k.mean().rename(columns={"yhat_k": "yhat"})


# ----------------------------------------------------------------------------- C0 (STK)

def apply_c0(cands: pd.DataFrame, refined: dict, raw: dict, ratio: float = 0.6) -> tuple:
    """C0: a Mask DINO Swin-L candidate (source 'yolo') whose refined mask is smaller than ratio x its raw area keeps the raw mask.

    Same rule on the OOF refiner masks (training) and the test refiner masks. Returns ({image: {idx: RLE}} copy, n replaced)."""
    out = {n: dict(enumerate(v)) if isinstance(v, list) else dict(v) for n, v in refined.items()}; nrep = 0
    for n, i in zip(cands.image[cands.source == "yolo"], cands.idx[cands.source == "yolo"]):
        i = int(i)
        if n in out and i in out[n] and mU.area(out[n][i]) < ratio * max(mU.area(raw[n][i]), 1):
            out[n][i] = raw[n][i]; nrep += 1
    return out, nrep


# ----------------------------------------------------------------------------- decode

# Optional decode settings (config section `decode`) and their defaults = switched off, i.e. the decode of every earlier
# config is unchanged. A new setting is one entry here plus one keyword of decode_image; decode_knobs passes it on from the
# config to the test decode (stage_final_cpu) and to the OOF decode (oof_eval), so the two cannot drift apart.
#   order_bonus  added to yhat of the Mask DINO Swin-L candidates (source 'yolo') for the processing ORDER only; the
#                acceptance test yhat > tau uses the unchanged yhat (configs/final_yellow.yaml: 0.20)
#   iom_max      extra suppression next to mask NMS: a mask whose intersection with an already kept mask exceeds
#                iom_max x the smaller of the two areas is dropped (None = off)
# Settings that already exist elsewhere: decode.nms_iou (mask NMS), decode.keep_frac, refiner.c0_ratio (C0) and
# refiner.paste_thr (binarisation threshold of the refiner probability, used by refine-test).
DECODE_KNOBS = dict(order_bonus=0.0, iom_max=None)


def decode_knobs(cfg) -> dict:
    """The optional decode settings of the config (DECODE_KNOBS) as keywords of decode / decode_image."""
    return {k: cfg["decode"].get(k, v) for k, v in DECODE_KNOBS.items()}


def with_source(Y: pd.DataFrame, cands: pd.DataFrame) -> pd.DataFrame:
    """Y (image, idx, yhat) in its row order plus the column `swinl` (candidate source 'yolo' = Mask DINO Swin-L), which
    decode_image needs for order_bonus."""
    src = cands[["image", "idx", "source"]].drop_duplicates(["image", "idx"])
    out = Y.merge(src, on=["image", "idx"], how="left")
    assert len(out) == len(Y) and out.source.notna().all(), "candidate source of every scored row"
    out["swinl"] = (out.source == "yolo").values
    return out.drop(columns="source")


def decode_image(g: pd.DataFrame, R: dict, tau=0.28, nms_iou=0.3, keep_frac=0.5, lo=1.0, min_add_area=0, order_bonus=0.0, iom_max=None) -> list:
    """Instance RLEs of one image (decreasing yhat). g: rows (idx, yhat) of the image; R: {idx: refined RLE}.

    order_bonus / iom_max: see DECODE_KNOBS (g needs the column `swinl` for order_bonus, with_source)."""
    hi = g[g.yhat > tau]
    if order_bonus:   # order key = yhat + bonus on the Swin-L rows; stable, so equal keys keep the row order
        hi = hi.iloc[np.argsort(-(hi.yhat.values + order_bonus * hi.swinl.values.astype(float)), kind="stable")]
    else:
        hi = hi.sort_values("yhat", ascending=False)
    ad = g[(g.yhat > lo) & (g.yhat <= tau)].sort_values("yhat", ascending=False)
    sd = pd.concat([hi, ad]); ms = [R[int(i)] for i in sd.idx]; ar = [mU.area(m) for m in ms]; nh = len(hi); sel = []
    I = np.asarray(mU.iou(ms, ms, [0] * len(ms))).reshape(len(ms), len(ms)) if len(ms) > 1 else None
    M = None
    if iom_max is not None and I is not None:   # intersection = IoU x (area a + area b) / (1 + IoU); M = intersection / smaller area
        af = np.asarray(ar, float); M = I * (af[:, None] + af[None, :]) / (1 + I) / np.maximum(np.minimum(af[:, None], af[None, :]), 1)
    for a in range(len(ms)):
        if a >= nh and ar[a] < min_add_area:
            continue
        if (I is None or all(I[a, b] <= nms_iou for b in sel)) and (M is None or all(M[a, b] <= iom_max for b in sel)):
            sel.append(a)
    occ = np.zeros((2048, 2048), bool); out = []
    for a in sel:
        m = mU.decode(ms[a]).astype(bool); new = m & ~occ
        if new.sum() < 1 or new.sum() < keep_frac * m.sum():
            continue
        occ |= new; out.append(mU.encode(np.asfortranarray(new.astype(np.uint8))))
    return out


def decode(Y: pd.DataFrame, rles: dict, tau=0.28, nms_iou=0.3, keep_frac=0.5, lo=1.0, min_add_area=0, **knobs) -> list:
    """CSV lines (with header). Y: image, idx, yhat (+ swinl for order_bonus); rles: {image: {idx: refined RLE}}.

    lo / min_add_area: optional low-score add of candidates with lo < yhat <= tau and area >= min_add_area
    (lo = 1 disables it; the final model uses no add). knobs: the optional settings of DECODE_KNOBS (decode_knobs)."""
    assert set(knobs) <= set(DECODE_KNOBS), sorted(set(knobs) - set(DECODE_KNOBS))
    lines = ["filament_id,segmentation_rle"]
    for n, g in Y.groupby("image"):
        out = decode_image(g, rles[n], tau, nms_iou, keep_frac, lo, min_add_area, **knobs)
        lines += [f"{n}_{k + 1},{r['counts'].decode('ascii')}" for k, r in enumerate(out)]
    return lines


# ----------------------------------------------------------------------------- OOF self-evaluation

def pq_counts(pred: dict, gt: dict, names) -> np.ndarray:
    """(sum IoU of matches, TP, FP, FN) over every annotator record (IoU > 0.5 = match; the competition's PQ per record)."""
    c = np.zeros(4)
    for n in names:
        pr = pred.get(n, [])
        for r in gt[n]:
            if not pr:
                c[3] += len(r); continue
            if not r:
                c[2] += len(pr); continue
            I = np.asarray(mU.iou(pr, r, [0] * len(r))).reshape(len(pr), len(r)); h = I > 0.5
            c += [I[h].sum(), h.sum(), (h.sum(1) == 0).sum(), (h.sum(0) == 0).sum()]
    return c


def pq(c) -> float:
    """PQ from the counts c = (matched-IoU sum, TP, and the two unmatched counts): c[0] / (c[1] + c[2] / 2 + c[3] / 2)."""
    return float(c[0] / (c[1] + 0.5 * c[2] + 0.5 * c[3]))


def oof_eval(cfg, tau=None, keep=0.77, seeds=5) -> dict:
    """OOF PQ of the 707 training frames: OOF V-ENS scores + OOF NNR masks, decode as on the test set.

    rec = against the annotator records as they are; sim = mean over `seeds` draws keeping each GT filament with
    probability `keep` (calibrated on the public LB, development scripts/sim_eval.py)."""
    c = resolve(cfg); d = cfg["decode"]; tau = d["tau"] if tau is None else tau
    gt = pickle.load(open(c["gt_rles"], "rb"))
    Y = pd.read_parquet(c["oof_yhat"]).rename(columns={"yhat_ens": "yhat"})
    R = {}
    for k in range(5):
        R.update(pickle.load(open(c["oof_refiner_rles"].format(k=k), "rb")))
    if cfg["refiner"].get("c0_ratio"):
        R, _ = apply_c0(pd.read_parquet(c["train_cands"], columns=["image", "idx", "source"]), R,
                        pickle.load(open(c["train_rles"], "rb")), cfg["refiner"]["c0_ratio"])
    kn = decode_knobs(cfg)
    if kn["order_bonus"]:
        Y = with_source(Y, pd.read_parquet(c["train_cands"], columns=["image", "idx", "source"]))
    G = {n: g for n, g in Y.groupby("image")}
    P = {n: decode_image(G[n], R[n], tau, d["nms_iou"], d["keep_frac"], **kn) for n in sorted(gt) if n in G}
    names = sorted(P); rec = pq(pq_counts(P, gt, names)); sim = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        g2 = {n: [[m for m in rec_ if rng.random() < keep] for rec_ in recs] for n, recs in gt.items()}
        sim.append(pq(pq_counts(P, g2, names)))
    return dict(tau=tau, frames=len(names), rec_pq=round(rec, 4), sim_pq=round(float(np.mean(sim)), 4))


def write_csv(lines: list, path: Path) -> str:
    """Write the CSV lines (header first) with a final newline and return the md5 of the file."""
    Path(path).write_text("\n".join(lines) + "\n")
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- NNR refiner planes (GPU stages; not run in the tests)

def stage_nnr_crops(cfg):
    """NNR training crops of the 7th plane: the nnU-Net OOF map (nlv.train_nnmaps, <stem>_rep.png from the fold model that
    validated on the frame) cropped with every refiner crop box of the NL candidates (refiner_meta.parquet of `crops`)
    -> refiner_nnr_crops.npy, listed in refiner.extra_crops so that `residual --fold k` trains the 7-plane model."""
    from . import nnunet_cands, refiner
    out = cfg["paths"].out; name = cfg["refiner"]["extra_crops"][0]
    meta = pd.read_parquet(out / "refiner_meta.parquet")
    S = cfg["refiner"]["crop_size"]
    X = np.lib.format.open_memmap(out / (name + ".tmp.npy"), mode="w+", dtype=np.uint8, shape=(len(meta), S, S))
    map_dir = Path(resolve(cfg)["train_nnmaps"])
    for n, g in meta.groupby("image", sort=False):
        X[g.index.values] = refiner.crop_plane(nnunet_cands.load_map_u8(map_dir, n), g[["x0", "y0", "side"]].values, S)
    X.flush()
    del X
    os.replace(out / (name + ".tmp.npy"), out / name)


def nnr_planes(cfg, df, boxes):
    """plane7 for refiner.ensemble_residual_pmatch: plane7(k) = crops of the nnU-Net test map of fold model k (nlv.test_nnmaps)."""
    from . import nnunet_cands, refiner
    S = cfg["refiner"]["crop_size"]; pat = resolve(cfg)["test_nnmaps"]

    def plane7(k):
        P = np.zeros((len(df), 1, S, S), np.uint8)
        for n, g in df.groupby("image", sort=False):
            P[g.index.values, 0] = refiner.crop_plane(nnunet_cands.load_map_u8(Path(pat.format(k=k)), n), boxes[g.index.values], S)
        return P
    return plane7


def refiner_members(cfg, weights_dir=None) -> list:
    """Test-time refiner members from refiner.members (configs/stk.yaml: one member S15F; an ensemble of several refiners is
    a longer list, averaged with equal weights): [dict(tag, checkpoints=[5 paths], feats=None or callable k -> feature crops)].
    checkpoints: pattern with {k}, relative to weights_dir (default: the out folder); test_feats: pattern with {k} relative
    to the out folder (filament_seg/s15f.py stage_test). Without refiner.members: one member refiner_<tag>_f{k}.pt.
    full: true (configs/final_fullfit.yaml) = one full-data checkpoint (pattern without {k}, `residual --fold -1`) applied to
    each of the five fold-k inputs; its five probabilities are averaged like those of five fold models."""
    c = cfg["refiner"]; out = cfg["paths"].out; wd = Path(weights_dir) if weights_dir else out
    ex = lambda v, k: Path(os.path.expandvars(v).format(k=k))  # noqa: E731
    mem = c.get("members") or [dict(tag=c["tag"], checkpoints=f"refiner_{c['tag']}_f{{k}}.pt")]
    res = []
    for m in mem:
        assert bool(m.get("full")) == ("{k}" not in m["checkpoints"]), f"member {m.get('tag')}: full: true <=> checkpoint without {{k}}"
        cks = [wd / ex(m["checkpoints"], k) for k in range(5)]
        feats = None
        if m.get("test_feats"):
            from . import s15f
            pat = m["test_feats"]
            feats = (lambda p, lazy: (lambda k: s15f.load_feats(out / ex(p, k), lazy=lazy)))(pat, bool(c.get("feat_lazy", True)))
        res.append(dict(tag=m.get("tag", ""), full=bool(m.get("full")), checkpoints=cks, feats=feats, boxes=out / ex(m["test_boxes"], 0) if m.get("test_boxes") else None))
    return res


def refine_test(cfg, df: pd.DataFrame, cand_rles: dict, probs: dict, raws: dict, checkpoints, device="cuda") -> dict:
    """NNR on the test candidates: 4 base crop planes (stages.infer step 4) + plane 7 = nnU-Net map of fold k for fold model k,
    flip TTA x4, mean of the five fold models, paste-back -> {image: {idx: refined RLE}} (the format of refiner_rles).
    With refiner.best_match_planes (NNRP, configs/stk.yaml) planes 8-9 = best-matching Swin-L 1536 / RF-DETR-Seg masks of the
    fold-k detectors (filament_seg/refplanes.py).

    df: candidate table (image, idx, source, x, y0, w, h, cluster_n_src) in a fixed row order; probs: {image: U-Net++ 1024};
    raws: {image: 2048 frame}. checkpoints: list of fold checkpoints (one refiner) or the member list of refiner_members
    (S15F feature crops, several refiners averaged). Checked on the GPU against the submitted test masks (README, GPU stages)."""
    from . import refiner
    c = cfg["refiner"]; S = c["crop_size"]; df = df.reset_index(drop=True)
    X = np.zeros((len(df), 4, S, S), np.uint8); boxes = np.zeros((len(df), 3), np.int64)
    for n, g in df.groupby("image", sort=False):
        R = cand_rles[n]
        x, b, _, _ = refiner.make_crops(raws[n], probs[n], [R[int(i)] for i in g.idx], g.source.tolist(), list(zip(g.x, g.y0, g.w, g.h)),
                                        S, c["crop_scale"], c["crop_min_side"])
        X[g.index.values], boxes[g.index.values] = x, b
    is_y = (df.source == "yolo").values
    S2 = np.stack([is_y, is_y & (df.cluster_n_src.values == 1)], 1).astype(np.float32)
    if c.get("best_match_planes"):
        from . import refplanes
        planes = refplanes.nnrp_planes(cfg, df, boxes, cand_rles)
    else:
        planes = nnr_planes(cfg, df, boxes)
    members = [dict(checkpoints=list(checkpoints))] if checkpoints and not isinstance(checkpoints[0], dict) else checkpoints
    for m in members:
        if m.get("boxes") is not None:
            assert (np.load(m["boxes"]) == boxes).all(), "refiner windows differ from the windows of the feature crops"
    prob = refiner.ensemble_members(members, X, S2, planes, c["alpha"], device, c.get("infer_bs", 64), log=log)
    return paste_refined(cfg, df, cand_rles, prob, boxes)


def paste_refined(cfg, df: pd.DataFrame, cand_rles: dict, prob, boxes) -> dict:
    """Refiner probabilities (N, S, S) of the rows of df -> {image: {idx: refined RLE}} (paste-back; the seed is kept when the
    refined mask is empty or does not touch the seed). Also used on the per-member probability sums of the development run
    (`member_sum_mean`)."""
    from . import io, refiner
    c = cfg["refiner"]; df = df.reset_index(drop=True); ref = {}
    for i, r in enumerate(df.itertuples()):
        seed = cand_rles[r.image][int(r.idx)]
        m, st = refiner.paste_back(prob[i], int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), io.decode(seed), c.get("paste_thr", 0.5))
        ref.setdefault(r.image, {})[int(r.idx)] = io.encode(m) if st == "ok" else seed
    return ref


def member_sum_mean(sums, K: int = 5):
    """Equal-weight member mean from per-member sums over the K fold inputs (float32, in member order): sum / (K x M),
    the value refiner.ensemble_members returns."""
    total, m = None, 0
    for s in sums:  # any iterable (a generator keeps one sum in memory at a time)
        s = np.asarray(s, np.float32)
        total = s.copy() if total is None else total + s
        m += 1
    total /= K * m
    return total


# ----------------------------------------------------------------------------- stage

def stage_refine_test(cfg, out_pkl: Path, weights_dir=None, device="cuda", limit=0) -> dict:
    """GPU (CPU works, slower): refined masks of all test candidates -> {image: {idx: RLE}} pickle (the nlv.refiner_rles input
    of final-cpu, before C0). Inputs: nlv.test_cands / test_rles, U-Net++ test probabilities nlv.test_v1_prob (+ test_names),
    the test frames, nnU-Net test maps of fold k, the NNRP detector outputs of fold k and the members' feature crops."""
    from . import io
    c = resolve(cfg); t0 = time.time()
    df = pd.read_parquet(c["test_cands"]).reset_index(drop=True)
    if limit:
        keep = list(dict.fromkeys(df.image))[:limit]
        df = df[df.image.isin(keep)].reset_index(drop=True)
    cand = pickle.load(open(c["test_rles"], "rb"))
    names = [str(n) for n in np.load(c["test_names"])]
    P = np.load(c["test_v1_prob"], mmap_mode="r"); pos = {n: i for i, n in enumerate(names)}
    imgs = list(dict.fromkeys(df.image))
    probs = {n: np.asarray(P[pos[n]], np.float32) for n in imgs}
    raws = {n: io.read_gray(cfg["paths"].test_images / f"{n}.jpeg") for n in imgs}
    members = refiner_members(cfg, weights_dir)
    if limit:  # feature crops are stored for the full table: select the rows of the kept frames
        full = pd.read_parquet(c["test_cands"]).reset_index(drop=True)
        rows = np.where(full.image.isin(set(imgs)).values)[0]
        for m in members:
            if m["feats"]:
                m["feats"] = (lambda f: (lambda k: np.ascontiguousarray(f(k)[rows])))(m["feats"])
            m["boxes"] = None
    log(f"[refine-test] {len(df)} candidates / {len(imgs)} frames, members {[m['tag'] for m in members]}")
    ref = refine_test(cfg, df, cand, probs, raws, members, device)
    pickle.dump(ref, open(out_pkl, "wb"))
    return dict(out=str(out_pkl), candidates=len(df), frames=len(imgs), members=[m["tag"] for m in members], minutes=round((time.time() - t0) / 60, 1))


def stage_final_cpu(cfg, out_csv: Path, yhat_cached=False, recompute_votes=False, recompute_maps=False, save_weights=None, load_weights=None,
                    recompute_evo=False) -> dict:
    """From the cached GPU outputs: train table -> V-ENS (fit) -> test tables -> yhat -> decode with the refined masks -> CSV.

    yhat_cached=True skips the verifier and decodes the stored yhat (decode-only check);
    load_weights=<dir> loads the four fitted members (save_members) instead of fitting them.
    verifier.seeds (STK: [0, 1, 2, 3, 4]) fits one V-ENS per seed and averages the test scores; with several seeds the
    weights live in <dir>/seed<s>/. refiner.c0_ratio applies C0 to the refined test masks before decoding."""
    c = resolve(cfg); d = cfg["decode"]; t0 = time.time(); res = {}
    missing = [p for p in [c["refiner_rles"]] + ([c["yhat"]] if yhat_cached else []) if not Path(p).exists()]
    if missing:
        raise SystemExit("final-cpu: stored model outputs not found:\n  " + "\n  ".join(missing) + f"\nSOLAR_ROOT = {os.environ['SOLAR_ROOT']}. "
                         "Set SOLAR_ROOT to the folder that holds outputs/ (README, CPU path: the unpacked cache/final_cache_v1.tar).")
    seeds = list(cfg.get("verifier", {}).get("seeds", [0]))
    if yhat_cached:
        Y = pd.read_parquet(c["yhat"])
    else:
        feats = cfg_feature_list(c)
        wdir = lambda root, s: Path(root) / f"seed{s}" if len(seeds) > 1 else Path(root)  # noqa: E731
        tr = None if load_weights else train_table(c, recompute_votes, recompute_maps, recompute_evo)
        if tr is not None:
            log(f"[nlv] train rows {len(tr)}, features {len(feats)}, seeds {seeds}")
        te = test_tables(c, fast=cfg.get("efficiency", {}).get("fast_map_features", True), recompute_evo=recompute_evo)
        Ys = []
        for s in seeds:
            if load_weights:
                models = load_members(wdir(load_weights, s)); log(f"[nlv] V-ENS seed {s} loaded from {wdir(load_weights, s)}")
            else:
                models = fit_members(tr[feats].values, tr.y.values, cfg.get("verifier", {}).get("ens"), seed=s); log(f"[nlv] V-ENS seed {s} fitted")
            if save_weights:
                save_members(models, wdir(save_weights, s))
            Ys.append(score_test(models, te, feats))
            log(f"[nlv] seed {s}: yhat > {d['tau']}: {(Ys[-1].yhat > d['tau']).mean():.4f}")
        Y = Ys[0][["image", "idx"]].copy()
        for y in Ys[1:]:
            assert (y[["image", "idx"]].values == Y[["image", "idx"]].values).all()
        Y["yhat"] = np.mean([y.yhat.values for y in Ys], 0)
        out_y = Path(out_csv).with_suffix(".yhat.parquet"); Y.to_parquet(out_y); res["yhat"] = str(out_y)
        res["yhat_max_abs_diff_vs_submitted"] = None
        if c.get("yhat") and Path(c["yhat"]).exists():   # reference scores of the submitted run (development folder only)
            ref = pd.read_parquet(c["yhat"]).merge(Y, on=["image", "idx"], suffixes=("_ref", ""))
            res["yhat_max_abs_diff_vs_submitted"] = float(np.abs(ref.yhat_ref - ref.yhat).max()) if len(ref) else None
        log(f"[nlv] yhat > {d['tau']}: {(Y.yhat > d['tau']).mean():.4f}; max |diff| vs submitted {res['yhat_max_abs_diff_vs_submitted']}")
    R = pickle.load(open(c["refiner_rles"], "rb"))
    if cfg["refiner"].get("c0_ratio"):
        R, res["c0_replaced"] = apply_c0(pd.read_parquet(c["test_cands"], columns=["image", "idx", "source"]), R,
                                         pickle.load(open(c["test_rles"], "rb")), cfg["refiner"]["c0_ratio"])
    kn = decode_knobs(cfg)
    if kn["order_bonus"]:
        Y = with_source(Y, pd.read_parquet(c["test_cands"], columns=["image", "idx", "source"]))
    lines = decode(Y, R, d["tau"], d["nms_iou"], d["keep_frac"], d.get("low_add_lo", 1.0), d.get("low_add_min_area", 0), **kn)
    res.update(csv=str(out_csv), md5=write_csv(lines, out_csv), rows=len(lines) - 1, minutes=round((time.time() - t0) / 60, 1))
    return res
