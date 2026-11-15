"""Candidate generation: connected components of the U-Net++ probability map united with YOLO masks.

A candidate is a tuple ``(source, conf, mask2048, bbox)`` with source in {"v1_b03", "v1_b04", "yolo"}:
- ``v1_b03`` / ``v1_b04``: 8-connected components of the 2048 probability map thresholded at 0.3 / 0.4
  (area >= min_area); conf = mean probability inside the component.
- ``yolo``: YOLO11-seg instance masks with confidence >= conf_min; conf = detector confidence.
Both thresholds are kept on purpose: the union is a high-recall seed pool; the verifier decides later.
"""
import cv2
import numpy as np
from pycocotools import mask as mask_utils

from .io import H, W

SOURCES = {"v1_b03": 0, "v1_b04": 1, "yolo": 2}


def upsample_prob(p1024: np.ndarray) -> np.ndarray:
    """(1024, 1024) probability -> (2048, 2048) float32 by bilinear interpolation (values in [0, 1])."""
    p = np.asarray(p1024, dtype=np.float32)
    if p.max() > 1.5:  # stored as 0..255
        p = p / 255.0
    return cv2.resize(p, (W, H), interpolation=cv2.INTER_LINEAR)


def bbox_of(m: np.ndarray) -> tuple:
    ys, xs = np.nonzero(m)
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def prob_components(p2048: np.ndarray, bin_thr: float, min_area: int) -> list:
    """[(bool mask, (x, y, w, h))] for the 8-connected components of p2048 >= bin_thr."""
    b = (p2048 >= bin_thr).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    out = []
    for lb in range(1, n):
        if int(stats[lb, cv2.CC_STAT_AREA]) < min_area:
            continue
        x, y, w, h = (int(stats[lb, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        m = np.zeros((H, W), bool)
        m[y:y + h, x:x + w] = labels[y:y + h, x:x + w] == lb
        out.append((m, (x, y, w, h)))
    return out


def collect(p2048: np.ndarray, yolo_items: list, bins=(0.3, 0.4), min_area: int = 50, conf_min: float = 0.05) -> list:
    """Union of probability components and YOLO masks. yolo_items: [(conf, RLE)] sorted by decreasing conf."""
    cands = []
    for bt in bins:
        for m, bb in prob_components(p2048, bt, min_area):
            cands.append((f"v1_b{int(round(bt * 10)):02d}", float(p2048[m].mean()), m, bb))
    for c, r in yolo_items:
        if c < conf_min:
            continue
        m = mask_utils.decode(r).astype(bool)
        if m.sum() < 1:
            continue
        cands.append(("yolo", float(c), m, bbox_of(m)))
    return cands


def recollect(p2048: np.ndarray, items: list) -> list:
    """Rebuild candidate tuples from refined masks. items: [(source, yolo_conf, RLE)].

    Probability-component candidates take the mean probability inside the *refined* mask as conf.
    """
    out = []
    for s, c, r in items:
        m = mask_utils.decode(r).astype(bool)
        out.append((s, float(c) if s == "yolo" else float(p2048[m].mean()), m, bbox_of(m)))
    return out
