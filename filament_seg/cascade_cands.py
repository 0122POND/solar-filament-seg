"""Cascade Mask R-CNN R50-FPN (detectron2, Apache-2.0): one of the seven detector families of the g7_m2 columns
(filament_seg/evofeats.py, key `cascade`). Not a candidate source; only its instances are read.

Recipe of the submitted run (configs/stk.yaml, section detectors.cascade):
  detectron2 model zoo Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml, COCO weights (model zoo checkpoint of that config), one class,
  anchor aspect ratios 0.25 / 0.5 / 1 / 2 / 4 (thin structures), pixel mean / std 127.5 / 64 on the grayscale frame resized to
  1024 (INTER_AREA) and replicated to 3 channels, 100 detections per image, score >= 0.05.
  Training: each sample uses the masks of one annotator record drawn at random (resized to 1024, > 0.25 after area
  interpolation, >= 10 px), D4 augmentation, SGD lr 0.0025 (momentum 0.9, wd 1e-4, 500 warm-up steps, x0.1 at 70 % and
  x0.01 at 90 % of the steps), gradient clip 5, batch 2, 40 epochs, seed 0. Inference: masks at 2048 (detectron2 pastes them
  at the requested height / width), kept with score >= 0.05 and >= 30 px.
  Training frames get the fold model that did not see them (fold-validation pickle cmrcnn_f{k}_val.pkl), test frames the
  output of every fold model k on all 180 frames (cmrcnn_test_f{k}.pkl); format {stem: [(score, RLE 2048)]}.
Setup: detectron2 as in filament_seg/maskdino_cands.py (commit 247691a). Port of the development scripts cmrcnn/train.py
and cmrcnn/cmrcnn_test.py (same numbers; paths come from the config).

usage (GPU):
  python -m filament_seg.cascade_cands --config configs/stk.yaml train --fold k       -> out/cmrcnn_f{k}.pt, cmrcnn_f{k}_val.pkl
  python -m filament_seg.cascade_cands --config configs/stk.yaml test --folds 0 1 2 3 4 [--weights DIR] -> out/cmrcnn_test_f{k}.pkl
"""
import argparse
import pickle
import random
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mU

from . import config, folds, io

S = 1024
ZOO = "Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
DEFAULTS = dict(size=S, aspect_ratios=[0.25, 0.5, 1.0, 2.0, 4.0], detections=100, score_min=0.05, min_area=30,
                epochs=40, bs=2, lr=0.0025, momentum=0.9, weight_decay=1e-4, warmup=500, clip=5.0, seed=0)


def settings(cfg) -> dict:
    return {**DEFAULTS, **(cfg.get("detectors", {}).get("cascade", {}) or {})}


def prep(img: np.ndarray, size: int = S) -> np.ndarray:
    return np.repeat(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)[None].astype(np.float32), 3, 0)


def masks_of(rec, size: int = S) -> list:
    out = []
    for r in rec:
        m = cv2.resize(mU.decode(r).astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.25
        if m.sum() >= 10:
            out.append(m)
    return out


def d2_config(s: dict, weights: str = ""):
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(ZOO))
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [list(s["aspect_ratios"])]
    cfg.MODEL.PIXEL_MEAN = [127.5] * 3; cfg.MODEL.PIXEL_STD = [64.0] * 3; cfg.INPUT.FORMAT = "RGB"
    cfg.TEST.DETECTIONS_PER_IMAGE = s["detections"]; cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = s["score_min"]
    cfg.freeze()
    return cfg


def predict(model, x: np.ndarray, s: dict) -> list:
    """(3, S, S) input -> [(score, RLE 2048)]."""
    import torch
    with torch.no_grad():
        o = model([{"image": torch.from_numpy(x), "height": 2048, "width": 2048}])[0]["instances"].to("cpu")
    items = []
    for sc, m in zip(o.scores.numpy(), o.pred_masks.numpy()):
        m = m.astype(np.uint8)
        if sc >= s["score_min"] and m.sum() >= s["min_area"]:
            items.append((float(sc), mU.encode(np.asfortranarray(m))))
    return items


def load(path, s: dict = None, device="cuda"):
    """Trained fold model (state dict of `train`)."""
    import torch
    from detectron2.modeling import build_model
    model = build_model(d2_config(s or DEFAULTS)); model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.to(device).eval()


def predict_image(model, img: np.ndarray, s: dict = None) -> list:
    """2048 gray frame -> [(score, RLE 2048)]."""
    s = s or DEFAULTS
    return predict(model, prep(img, s["size"]), s)


def train(cfg, fold: int, log=print):
    import torch
    from detectron2 import model_zoo
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from detectron2.structures import BitMasks, Instances
    from detectron2.utils.events import EventStorage
    s = settings(cfg); size = s["size"]; out = cfg["paths"].out
    torch.manual_seed(s["seed"]); rng = random.Random(s["seed"]); np.random.seed(s["seed"]); t0 = time.time()
    init = s.get("coco_weights") or model_zoo.get_checkpoint_url(ZOO)
    dcfg = d2_config(s, str(init))
    model = build_model(dcfg); DetectionCheckpointer(model).load(dcfg.MODEL.WEIGHTS); model.to("cuda")
    sp = folds.load_split(out, f"split_f{fold}.json"); tr, va = sp["train"], sp["val"]
    gt = io.load_records(cfg["paths"].train_json)
    X = {n: prep(io.read_gray(cfg["paths"].train_images / f"{n}.jpeg"), size) for n in tr + va}
    recs = {n: [r for r in (masks_of(rec, size) for rec in gt.get(n, {}).values() if rec) if r] for n in tr}
    tr = [n for n in tr if recs[n]]
    log(f"[cascade] fold {fold}: train {len(tr)} val {len(va)} ({time.time() - t0:.0f}s)")
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=s["lr"], momentum=s["momentum"], weight_decay=s["weight_decay"])
    steps = s["epochs"] * max(len(tr) // s["bs"], 1)
    lr_at = lambda t: min(1.0, (t + 1) / s["warmup"]) * (0.01 if t >= 0.9 * steps else 0.1 if t >= 0.7 * steps else 1.0)  # noqa: E731
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at); model.train(); st = 0; tl = 0.0
    es = EventStorage(0); es.__enter__()
    while st < steps:
        order = tr[:]; rng.shuffle(order)
        for b in range(0, len(order) - s["bs"] + 1, s["bs"]):
            batch = []
            for n in order[b:b + s["bs"]]:
                x = X[n]; ms = np.stack(rng.choice(recs[n])); k = rng.randrange(4); fl = rng.random() < 0.5
                x = np.rot90(x, k, (1, 2)); ms = np.rot90(ms, k, (1, 2))
                if fl:
                    x, ms = x[:, :, ::-1], ms[:, :, ::-1]
                t = torch.from_numpy(np.ascontiguousarray(ms)); inst = Instances((size, size))
                inst.gt_classes = torch.zeros(len(ms), dtype=torch.long); inst.gt_masks = BitMasks(t); inst.gt_boxes = BitMasks(t).get_bounding_boxes()
                batch.append({"image": torch.from_numpy(np.ascontiguousarray(x)), "instances": inst, "height": size, "width": size})
            ld = model(batch); loss = sum(ld.values())
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), s["clip"]); opt.step(); sched.step()
            st += 1; tl += float(loss)
            if st % 100 == 0:
                log(f"[cascade] fold {fold} step {st}/{steps} loss {tl / 100:.3f} ({time.time() - t0:.0f}s)"); tl = 0.0
            if st >= steps:
                break
        es.iter = st
    es.__exit__(None, None, None)
    torch.save(model.state_dict(), out / f"cmrcnn_f{fold}.pt")
    model.eval()
    res = {n: predict(model, X[n], s) for n in va}
    pickle.dump(res, open(out / f"cmrcnn_f{fold}_val.pkl", "wb"))
    log(f"[cascade] fold {fold} done ({(time.time() - t0) / 60:.1f} min)")


def test(cfg, folds_=(0, 1, 2, 3, 4), weights_dir=None, log=print):
    import torch
    s = settings(cfg); out = cfg["paths"].out; wd = Path(weights_dir) if weights_dir else out
    files = io.list_images(cfg["paths"].test_images)
    X = {f.stem: prep(io.read_gray(f), s["size"]) for f in files}
    for k in folds_:
        w = wd / s.get("weights", "cmrcnn_f{k}.pt").format(k=k)
        model = load(w, s)
        res = {n: predict(model, X[n], s) for n in sorted(X)}
        pickle.dump(res, open(out / f"cmrcnn_test_f{k}.pkl", "wb"))
        log(f"[cascade] test fold model {k}: {len(res)} frames, {np.mean([len(v) for v in res.values()]):.1f} instances / frame")
        del model; torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m filament_seg.cascade_cands", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=config.REPO / "configs" / "stk.yaml")
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("train").add_argument("--fold", type=int, required=True)
    t = sub.add_parser("test"); t.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); t.add_argument("--weights", type=Path, default=None)
    a = ap.parse_args(argv); cfg = config.load(a.config)
    train(cfg, a.fold) if a.mode == "train" else test(cfg, a.folds, a.weights)


if __name__ == "__main__":
    main()
