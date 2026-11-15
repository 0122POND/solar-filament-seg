"""NNRP refiner planes (configs/stk.yaml, the final refiner): NNR plus two crop planes holding the best-matching mask of two
detectors that are not candidate sources, Mask DINO Swin-L at 1536 px and RF-DETR-Seg Large.

For every refiner seed (candidate) and every detector: the instance with the largest IoU with the seed among the detector's
instances with score >= cmin (Swin-L 1536 0.05, RF-DETR-Seg 0.10); if that IoU >= 0.3 the plane is the instance mask (0 / 255)
cropped with the seed's refiner window, otherwise the plane is 0. Crop and resize as refiner.crop_plane (INTER_AREA when the
window is larger than the crop, INTER_LINEAR otherwise). Only model outputs are read (no GT, no metadata).

Training rows use the out-of-fold detector outputs (the fold of the detector = the refiner fold of the frame; asserted);
the refiner of fold k reads, at test time, the outputs of the fold-k detectors (as the 7th plane reads the fold-k nnU-Net
map), and the five refiner probabilities are averaged. Input order of the 9-channel NNRP model: 4 base planes, nnU-Net map,
Swin-L 1536 plane, RF-DETR-Seg plane, 2 source planes; the first-layer weights of the three added planes start at 0.
Port of the development scripts that built the training planes and the test planes.
"""
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mU

from . import refiner

SOURCES = (("swinl1536", 0.05), ("rfdetrseg", 0.10))   # (detector key in refplanes.oof / refplanes.test, min score); plane order
MIN_IOU = 0.3


def _rle(r):
    r = dict(r)
    if isinstance(r["counts"], str):
        r["counts"] = r["counts"].encode()
    return r


def best_match_planes(seed_rles: list, preds: list, boxes, size=256, min_iou=MIN_IOU) -> tuple:
    """(n, size, size) uint8 planes and the best IoU per seed. seed_rles: n RLEs; preds: detector RLEs (already score-filtered);
    boxes: (n, 3) refiner windows (x0, y0, side)."""
    n = len(seed_rles)
    X = np.zeros((n, size, size), np.uint8)
    best_iou = np.zeros(n, np.float32)
    if not preds or not n:
        return X, best_iou
    sr = [_rle(r) for r in seed_rles]
    I = np.asarray(mU.iou(sr, preds, [0] * len(preds)), dtype=np.float64).reshape(n, len(preds))
    best = I.argmax(1)
    bi = I[np.arange(n), best]
    best_iou[:] = bi
    dec = {}
    for row in range(n):
        if bi[row] < min_iou:
            continue
        b = int(best[row])
        if b not in dec:
            dec[b] = mU.decode(preds[b]).astype(np.uint8) * 255
        x0, y0, side = (int(v) for v in boxes[row])
        X[row] = cv2.resize(refiner.pad_crop(dec[b], x0, y0, side), (size, size),
                            interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR)
    return X, best_iou


def _preds(pat, k, cmin):
    d = pickle.load(open(os.path.expandvars(pat).format(k=k), "rb"))
    return {n: [_rle(r) for c, r in v if c >= cmin] for n, v in d.items()}


def stage_nnrp_planes(cfg, out_name="refiner_nnrp_planes.npy"):
    """Training planes of NNRP: (N, 2, 256, 256) uint8 in the row order of refiner_meta.parquet (stage `crops`), listed after
    refiner_nnr_crops.npy in refiner.extra_crops. Detector outputs: nlv.refplanes.oof.<key> (fold-validation pickles)."""
    from . import nlv
    c = nlv.resolve(cfg); pats = c["refplanes"]["oof"]; out = cfg["paths"].out; S = cfg["refiner"]["crop_size"]; t0 = time.time()
    meta = pd.read_parquet(out / "refiner_meta.parquet")
    seeds = pickle.load(open(c["train_rles"], "rb"))
    fold_of = meta.groupby("image").fold.first()
    P = {}
    for key, cmin in SOURCES:
        P[key] = {}
        for k in range(5):
            for n, v in _preds(pats[key], k, cmin).items():
                assert n not in P[key], (key, n)
                P[key][n] = (k, v)
        miss = [n for n in fold_of.index if n not in P[key]]
        bad = [n for n in fold_of.index if n in P[key] and P[key][n][0] != fold_of[n]]
        assert not miss and not bad, (key, len(miss), len(bad))   # OOF: detector fold == refiner fold
    tmp = out / (out_name + ".tmp.npy")
    X = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(meta), len(SOURCES), S, S))
    for n, g in meta.groupby("image", sort=False):
        rows = g.index.values
        for s, (key, _) in enumerate(SOURCES):
            X[rows, s] = best_match_planes([seeds[n][int(i)] for i in g.idx.values], P[key][n][1], g[["x0", "y0", "side"]].values, S)[0]
    X.flush()
    del X
    os.replace(tmp, out / out_name)
    nlv.log(f"[refplanes] {len(meta)} rows -> {out / out_name} ({time.time() - t0:.0f}s)")


def nnrp_planes(cfg, df: pd.DataFrame, boxes, cand_rles: dict):
    """Planes 5-7 for refiner.ensemble_residual_pmatch: planes(k) = (N, 3, S, S) = nnU-Net map of fold k (nlv.nnr_planes) +
    best-matching masks of the fold-k Swin-L 1536 / RF-DETR-Seg test outputs (nlv.refplanes.test.<key>)."""
    from . import nlv
    c = nlv.resolve(cfg); pats = c["refplanes"]["test"]; S = cfg["refiner"]["crop_size"]
    plane7 = nlv.nnr_planes(cfg, df, boxes)

    def planes(k):
        X = np.zeros((len(df), 1 + len(SOURCES), S, S), np.uint8)
        X[:, :1] = plane7(k)
        for s, (key, cmin) in enumerate(SOURCES):
            P = _preds(pats[key], k, cmin)
            for n, g in df.groupby("image", sort=False):
                X[g.index.values, 1 + s] = best_match_planes([cand_rles[n][int(i)] for i in g.idx], P.get(n, []), boxes[g.index.values], S)[0]
        return X
    return planes
