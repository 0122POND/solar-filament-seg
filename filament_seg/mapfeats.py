"""Map features for the NLv selector of the final model: 14 features of every candidate on the "own2" probability maps.

own2 = two U-Nets trained by us on the competition train set (smp U-Net r34 and tu-tf_efficientnetv2_s, ImageNet
initialisation, 2048 px; filament_seg/ownmap.py, `own-maps`), saved as <stem>_r34.png / <stem>_eff.png (uint8).
The verifier map is the mean of the two. Training frames use the OOF maps, test frames the map of each fold model
(the verifier is applied five times, once per fold map, and the five scores are averaged; see filament_seg/nlv.py).

Features d_* (DFEATS): statistics of the map inside and around the candidate mask, and its overlap with the kept
connected components of the map.
"""
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi

from .io import encode

KEEP = dict(t_lo=0.35, t_hi=0.99, min_area=200, min_meanp=0.85)  # component keep rule (features d_cc_*)
DFEATS = ["d_mean", "d_max", "d_q25", "d_f99", "d_f85", "d_f50", "d_ring", "d_contrast", "d_cc_iou", "d_cc_keep", "d_cc_meanp",
          "d_cover", "d_mean_r34", "d_mean_eff"]
_S8 = np.ones((3, 3), bool)
_RING = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


def load_map(map_dir: Path, stem: str):
    """(verifier map = mean of the two own2 maps, r34 map, EfficientNetV2-S map), all uint8."""
    a = cv2.imread(str(map_dir / f"{stem}_r34.png"), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(str(map_dir / f"{stem}_eff.png"), cv2.IMREAD_GRAYSCALE)
    return np.clip(np.round((a.astype(np.float32) + b) / 2), 0, 255).astype(np.uint8), a, b


def _u8(t):
    return int(np.floor(t * 255.0 + 1e-9))


def components(p: np.ndarray):
    """Components: 8-connected p > t_lo, with area / mean p / keep flag."""
    lab, n = ndi.label(p > _u8(KEEP["t_lo"]), structure=_S8)
    if n == 0:
        return lab, []
    idx = np.arange(1, n + 1)
    area = ndi.sum_labels(np.ones_like(p, np.int64), lab, idx)
    meanp = ndi.mean(p.astype(np.float64), lab, idx) / 255.0
    pmax = ndi.maximum(p, lab, idx)
    return lab, [dict(label=j + 1, area=int(area[j]), meanp=float(meanp[j]),
                      keep=bool(area[j] >= KEEP["min_area"] and meanp[j] >= KEEP["min_meanp"] and pmax[j] > _u8(KEEP["t_hi"])))
                 for j in range(n)]


def mask_features(m: np.ndarray, p, a, b, cc_rles, ccs) -> dict:
    """DFEATS of one candidate mask m (bool, 2048) on the maps (p, a, b)."""
    pv = p[m].astype(np.float32) / 255.0
    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = max(0, ys.min() - 10), ys.max() + 11, max(0, xs.min() - 10), xs.max() + 11
    sm = m[y0:y1, x0:x1].astype(np.uint8)
    ring = (cv2.dilate(sm, _RING) > 0) & (sm == 0)
    rv = p[y0:y1, x0:x1][ring].astype(np.float32) / 255.0 if ring.any() else pv
    f = dict(d_mean=float(pv.mean()), d_max=float(pv.max()), d_q25=float(np.percentile(pv, 25)), d_f99=float((pv > 0.99).mean()),
             d_f85=float((pv > 0.85).mean()), d_f50=float((pv > 0.5).mean()), d_ring=float(rv.mean()), d_contrast=float(pv.mean() - rv.mean()),
             d_mean_r34=float(a[m].mean() / 255.0), d_mean_eff=float(b[m].mean() / 255.0))
    if cc_rles:
        iou = np.asarray(mask_utils.iou([encode(m)], cc_rles, [0] * len(cc_rles))).reshape(-1)
        j = int(iou.argmax())
        f.update(d_cc_iou=float(iou[j]), d_cc_keep=float(ccs[j]["keep"]), d_cc_meanp=ccs[j]["meanp"])
    else:
        f.update(d_cc_iou=0.0, d_cc_keep=0.0, d_cc_meanp=0.0)
    f["d_cover"] = float((p[m] > _u8(KEEP["t_lo"])).mean())
    return f


def mask_features_fast(rle, p, a, b, lab, ccs, area) -> dict:
    """Same values as mask_features, computed on the candidate's bounding box (pixels in the same row-major order) and,
    for the component IoU, from a label histogram instead of one full-frame RLE per component (same inter / union)."""
    x, y, w, h = [int(v) for v in mask_utils.toBbox(rle)]
    full = mask_utils.decode(rle)
    y0, y1, x0, x1 = max(0, y - 10), y + h + 10, max(0, x - 10), x + w + 10
    sm = full[y0:y1, x0:x1].astype(bool)
    pc = p[y0:y1, x0:x1]
    pv = pc[sm].astype(np.float32) / 255.0
    smu = sm.astype(np.uint8)
    ring = (cv2.dilate(smu, _RING) > 0) & (smu == 0)
    rv = pc[ring].astype(np.float32) / 255.0 if ring.any() else pv
    f = dict(d_mean=float(pv.mean()), d_max=float(pv.max()), d_q25=float(np.percentile(pv, 25)), d_f99=float((pv > 0.99).mean()),
             d_f85=float((pv > 0.85).mean()), d_f50=float((pv > 0.5).mean()), d_ring=float(rv.mean()), d_contrast=float(pv.mean() - rv.mean()),
             d_mean_r34=float(a[y0:y1, x0:x1][sm].mean() / 255.0), d_mean_eff=float(b[y0:y1, x0:x1][sm].mean() / 255.0))
    if len(ccs):
        inter = np.bincount(lab[y0:y1, x0:x1][sm], minlength=len(ccs) + 1)[1:].astype(np.float64)
        iou = inter / (float(sm.sum()) + area - inter)
        j = int(iou.argmax())
        f.update(d_cc_iou=float(iou[j]), d_cc_keep=float(ccs[j]["keep"]), d_cc_meanp=ccs[j]["meanp"])
    else:
        f.update(d_cc_iou=0.0, d_cc_keep=0.0, d_cc_meanp=0.0)
    f["d_cover"] = float((pc[sm] > _u8(KEEP["t_lo"])).mean())
    return f


def image_features(map_dir: Path, stem: str, cand_rles: dict, idx: list, fast: bool = False) -> list:
    """DFEATS rows (with image, idx) of the candidates idx of one image; features use the original candidate masks.

    fast=True gives the same values about 6x faster (efficiency.fast_map_features; tests/test_efficiency.py)."""
    p, a, b = load_map(map_dir, stem)
    lab, ccs = components(p)
    if fast:
        area = np.array([c["area"] for c in ccs], np.int64)
        return [dict(image=stem, idx=int(i), **mask_features_fast(cand_rles[int(i)], p, a, b, lab, ccs, area)) for i in idx]
    cc_rles = [encode(lab == c["label"]) for c in ccs]
    return [dict(image=stem, idx=int(i), **mask_features(mask_utils.decode(cand_rles[int(i)]).astype(bool), p, a, b, cc_rles, ccs)) for i in idx]
