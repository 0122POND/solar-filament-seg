"""PQ-aware verifier (gradient-boosted regressor on candidate features) and overlap-free decoding.

The regressor predicts y_hat ~ E[IoU * 1(hit)] averaged over annotators (see features.py). Decoding keeps
candidates with y_hat > tau in decreasing y_hat order. tau = 0.25 sits next to PQ / 2 of our OOF score,
the break-even point at which an extra prediction stops paying for its 0.5 in the PQ denominator.
"""
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import FEATS_ALL
from .io import H, W, encode

DEFAULT_GBDT = dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0, random_state=0)


def make_model(params: dict = None) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(**(params or DEFAULT_GBDT))


def predict(model, table: pd.DataFrame, feats=FEATS_ALL) -> np.ndarray:
    return np.clip(model.predict(table[feats]), 0, 1)


def fit_oof(table: pd.DataFrame, params: dict = None, n_folds: int = 5, feats=FEATS_ALL):
    """Image-fold OOF predictions (the table needs a ``fold`` column) -> (y_hat, [model per fold])."""
    yhat = np.full(len(table), np.nan)
    models = []
    for k in range(n_folds):
        tr = table.fold != k
        va = table.fold == k
        m = make_model(params).fit(table.loc[tr, feats], table.loc[tr, "y"])
        yhat[va.values] = m.predict(table.loc[va, feats])
        models.append(m)
    return np.clip(yhat, 0, 1), models


def fit_full(table: pd.DataFrame, params: dict = None, feats=FEATS_ALL) -> HistGradientBoostingRegressor:
    return make_model(params).fit(table[feats], table.y)


def decode_image(rles: list, scores: list, tau: float, nms_iou: float = 0.3, keep_frac: float = 0.5, min_area: int = 1) -> list:
    """Overlap-free decoding of one image (the submission forbids overlapping pixels).

    1. keep candidates with score > tau, in decreasing score order;
    2. mask-NMS: drop a candidate whose IoU with an already kept one exceeds nms_iou;
    3. pixel assignment in score order: each mask keeps only still-free pixels and is dropped when
       less than keep_frac of it survives.
    Returns the RLEs to submit, best first.
    """
    order = [i for i in np.argsort(-np.asarray(scores)) if scores[i] > tau]
    if not order:
        return []
    if nms_iou < 1.0 and len(order) > 1:
        sub = [rles[i] for i in order]
        iou = np.asarray(mask_utils.iou(sub, sub, [0] * len(sub))).reshape(len(sub), len(sub))
        sel = []
        for a in range(len(order)):
            if all(iou[a, b] <= nms_iou for b in sel):
                sel.append(a)
        order = [order[a] for a in sel]
    occupied = np.zeros((H, W), bool)
    out = []
    for i in order:
        m = mask_utils.decode(rles[i]).astype(bool)
        new = m & ~occupied
        if new.sum() < max(min_area, 1) or new.sum() < keep_frac * m.sum():
            continue
        occupied |= new
        out.append(encode(new))
    return out
