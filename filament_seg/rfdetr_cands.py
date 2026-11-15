"""RF-DETR-Seg Large (Roboflow `rfdetr` 1.11.1, Apache-2.0; DINOv2 encoder Apache-2.0): one of the seven detector families
of the g7_m2 columns (filament_seg/evofeats.py, key `rfdetrseg`) and the source of NNRP / S15F refiner plane 7
(filament_seg/refplanes.py). Not a candidate source.

Recipe of the submitted run (configs/stk.yaml, section detectors.rfdetrseg):
  Dataset (`prep`): for fold k, every training frame of the other folds gives `ncrop` = 2 crops of 1024 px, each centred
  (with jitter) on one instance of the annotator record with the most instances; the validation frames of fold k become
  non-overlapping 2 x 2 tiles (valid split; test split = the first 10 %). Masks -> external contours (>= 30 px) in a
  Roboflow COCO folder; images written as 3-channel JPEG q95 of the grayscale frame. numpy default_rng(42).
  Training (`train`): RFDETRSegLarge from the released COCO checkpoint, resolution 1008 (multiple of patch x windows),
  batch 2 x gradient accumulation 4, lr 1e-4, 15 epochs, seed 42, no early stopping; the EMA checkpoint
  checkpoint_best_ema.pth is used.
  Inference (`infer`): 2 x 2 tiles of 1088 px starting at 0 / 960, predict(threshold 0.01), masks pasted on the 2048 canvas,
  tile duplicates removed greedily (IoU > 0.7 with a higher-confidence mask); the stored outputs keep score >= 0.05 and
  >= 30 px. Training frames: fold-validation pickles rfdetrseg_f{k}_val.pkl; test: rfdetrseg_test_f{k}.pkl (every fold model on
  all 180 frames); format {stem: [(score, RLE 2048)]}.
Port of the development scripts rfdetr_crop_prep.py, rfdetr_pilot_train.py, rfdetr_d_infer.py and rfdetr_test_infer.py.
The images are grayscale; RF-DETR's PIL input is the single channel replicated to RGB (no colour information is used).

usage (GPU except prep):
  python -m filament_seg.rfdetr_cands --config configs/stk.yaml prep --fold k            -> out/rfdetr_crops_f{k}/
  python -m filament_seg.rfdetr_cands --config configs/stk.yaml train --fold k           -> out/rfdetr/rf_f{k}/checkpoint_best_ema.pth
  python -m filament_seg.rfdetr_cands --config configs/stk.yaml infer --fold k --split val|test [--weights FILE]
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mU

from . import config, folds, io

CROP = 1024
FULL = 2048
DEFAULTS = dict(ncrop=2, seed=42, size="large", resolution=1008, batch=2, accum=4, lr=1e-4, epochs=15, workers=6,
                tile=1088, starts=[0, 960], threshold=0.01, dup_iou=0.7, score_min=0.05, min_area=30)


def settings(cfg) -> dict:
    return {**DEFAULTS, **(cfg.get("detectors", {}).get("rfdetrseg", {}) or {})}


# ----------------------------------------------------------------------------- dataset

def polys_from_mask(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [[float(v) for v in c.reshape(-1)] for c in cnts if cv2.contourArea(c) >= 30 and len(c) >= 3]


def write_split(d: Path, records):
    """records: [(file name, uint8 gray crop, [bool masks])] -> d/*.jpg + d/_annotations.coco.json."""
    d.mkdir(parents=True, exist_ok=True); images, anns = [], []; aid = 1
    for i, (fn, img, masks) in enumerate(records):
        cv2.imwrite(str(d / fn), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        images.append({"id": i + 1, "file_name": fn, "height": img.shape[0], "width": img.shape[1]})
        for m in masks:
            if m.sum() < 30:
                continue
            ys, xs = np.nonzero(m); x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            segs = polys_from_mask(m)
            if segs:
                anns.append({"id": aid, "image_id": i + 1, "category_id": 1, "iscrowd": 0, "area": float(m.sum()),
                             "bbox": [x0, y0, x1 - x0, y1 - y0], "segmentation": segs}); aid += 1
    json.dump({"info": {"description": d.name}, "licenses": [], "categories": [{"id": 1, "name": "filament", "supercategory": "none"}],
               "images": images, "annotations": anns}, open(d / "_annotations.coco.json", "w"))
    return len(images), len(anns)


def prep(cfg, fold: int, log=print):
    s = settings(cfg); p = cfg["paths"]; rng = np.random.default_rng(s["seed"])
    fold_of = folds.fold_of(p.out)
    phys = io.group_by_physical(io.load_coco(p.train_json)); root = p.out / f"rfdetr_crops_f{fold}"

    def masks(name):
        anns = max(phys[name]["sets"].values(), key=len)
        return [io.decode(io.polygon_to_rle(a["segmentation"])) for a in anns]
    recs = []
    for name in sorted(n for n, f in fold_of.items() if f != fold):
        raw = io.read_gray(p.train_images / phys[name]["file_name"]); ms = [m for m in masks(name) if m.any()]
        for c in range(s["ncrop"]):
            if ms:
                m = ms[rng.integers(len(ms))]; ys, xs = np.nonzero(m); cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
                w, h = xs.max() - xs.min(), ys.max() - ys.min(); jx = (CROP - min(w, CROP - 64)) / 2 * 0.8; jy = (CROP - min(h, CROP - 64)) / 2 * 0.8
                x0 = int(np.clip(cx - CROP / 2 + rng.uniform(-jx, jx), 0, FULL - CROP)); y0 = int(np.clip(cy - CROP / 2 + rng.uniform(-jy, jy), 0, FULL - CROP))
            else:
                x0, y0 = int(rng.integers(0, FULL - CROP + 1)), int(rng.integers(0, FULL - CROP + 1))
            recs.append((f"{name}_c{c}_{x0}_{y0}.jpg", raw[y0:y0 + CROP, x0:x0 + CROP], [mm[y0:y0 + CROP, x0:x0 + CROP] for mm in ms]))
    log(f"[rfdetr] train {write_split(root / 'train', recs)}")
    recs = []
    for name in sorted(n for n, f in fold_of.items() if f == fold):
        raw = io.read_gray(p.train_images / phys[name]["file_name"]); ms = masks(name)
        for ty in (0, CROP):
            for tx in (0, CROP):
                recs.append((f"{name}_t{tx}_{ty}.jpg", raw[ty:ty + CROP, tx:tx + CROP], [mm[ty:ty + CROP, tx:tx + CROP] for mm in ms]))
    log(f"[rfdetr] valid {write_split(root / 'valid', recs)} test {write_split(root / 'test', recs[:max(8, len(recs) // 10)])}")


# ----------------------------------------------------------------------------- training / inference

def _model_class(size):
    import rfdetr
    return getattr(rfdetr, {"medium": "RFDETRSegMedium", "large": "RFDETRSegLarge", "xlarge": "RFDETRSegXLarge"}[size])


def train(cfg, fold: int, resume=None, log=print):
    s = settings(cfg); out = cfg["paths"].out
    m = _model_class(s["size"])(); mc = m.model_config; unit = mc.patch_size * mc.num_windows; res = (s["resolution"] // unit) * unit
    kw = dict(dataset_dir=str(out / f"rfdetr_crops_f{fold}"), epochs=s["epochs"], batch_size=s["batch"], grad_accum_steps=s["accum"], lr=s["lr"],
              resolution=res, output_dir=str(out / "rfdetr" / f"rf_f{fold}"), num_workers=s["workers"], device="cuda:0", run_test=False,
              tensorboard=False, wandb=False, early_stopping=False, checkpoint_interval=1, seed=s["seed"])
    if resume:
        kw["resume"] = str(resume)
    log(f"[rfdetr] train fold {fold}: res {res}, batch {s['batch']} x {s['accum']}, {s['epochs']} epochs")
    m.train(**kw)


def predict_frame(m, raw: np.ndarray, s: dict) -> list:
    """2048 gray frame -> [(score, RLE 2048)] after tiling, duplicate removal and the score / area filter."""
    import torch
    from PIL import Image
    T = s["tile"]; items = []
    for ty in s["starts"]:
        for tx in s["starts"]:
            tile = Image.fromarray(raw[ty:ty + T, tx:tx + T]).convert("RGB")
            with torch.no_grad():
                det = m.predict(tile, threshold=s["threshold"])
            if det.mask is None or len(det) == 0:
                continue
            for mk, c in zip(det.mask, det.confidence):
                if c < s["score_min"]:   # never kept, and (lower score) never suppresses a kept mask in the duplicate removal
                    continue
                mk = np.asarray(mk).astype(np.uint8)
                if mk.shape != (T, T):
                    mk = cv2.resize(mk, (T, T), interpolation=cv2.INTER_NEAREST)
                if mk.sum() < 1:
                    continue
                canvas = np.zeros((FULL, FULL), bool); canvas[ty:ty + T, tx:tx + T] = mk.astype(bool); items.append((float(c), io.encode(canvas)))
    items.sort(key=lambda t: -t[0]); keep = []
    if items:
        rl = [r for _, r in items]; iou = np.asarray(mU.iou(rl, rl, [0] * len(rl))).reshape(len(rl), len(rl))
        for i in range(len(items)):
            if all(iou[i, j] <= s["dup_iou"] for j in keep):
                keep.append(i)
    return [items[i] for i in keep if items[i][0] >= s["score_min"] and mU.area(items[i][1]) >= s["min_area"]]


def load(path, s: dict = None):
    """Trained fold model (checkpoint_best_ema.pth of `train`)."""
    s = s or DEFAULTS
    return _model_class(s["size"])(pretrain_weights=str(path), resolution=s["resolution"])


def infer(cfg, fold: int, split="test", weights=None, log=print):
    s = settings(cfg); p = cfg["paths"]; t0 = time.time()
    w = Path(weights) if weights else p.out / "rfdetr" / f"rf_f{fold}" / "checkpoint_best_ema.pth"
    m = load(w, s)
    if split == "val":
        files = [p.train_images / f"{n}.jpeg" for n in sorted(folds.load_split(p.out, f"split_f{fold}.json")["val"])]
        dst = p.out / f"rfdetrseg_f{fold}_val.pkl"
    else:
        files = io.list_images(p.test_images); dst = p.out / f"rfdetrseg_test_f{fold}.pkl"
    preds = {f.stem: predict_frame(m, io.read_gray(f), s) for f in files}
    pickle.dump(preds, open(dst, "wb"))
    log(f"[rfdetr] {split} fold model {fold}: {len(preds)} frames, {np.mean([len(v) for v in preds.values()]):.1f} / frame ({time.time() - t0:.0f}s)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m filament_seg.rfdetr_cands", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=config.REPO / "configs" / "stk.yaml")
    sub = ap.add_subparsers(dest="mode", required=True)
    sub.add_parser("prep").add_argument("--fold", type=int, required=True)
    t = sub.add_parser("train"); t.add_argument("--fold", type=int, required=True); t.add_argument("--resume", type=Path, default=None)
    t = sub.add_parser("infer"); t.add_argument("--fold", type=int, required=True); t.add_argument("--split", choices=["val", "test"], default="test")
    t.add_argument("--weights", type=Path, default=None)
    a = ap.parse_args(argv); cfg = config.load(a.config)
    if a.mode == "prep":
        prep(cfg, a.fold)
    elif a.mode == "train":
        train(cfg, a.fold, a.resume)
    else:
        infer(cfg, a.fold, a.split, a.weights)


if __name__ == "__main__":
    main()
