"""STK agreement / shape features "g7_m2" (7 columns, configs/stk.yaml): one more view of every candidate from detectors
that are not candidate sources. No GT or metadata is read: only model outputs, candidate RLEs and the nnU-Net map.

Columns (prefixed c_ in the verifier table, nlv.feature_list(evo=True)):
  e_sw_dup / e_sw_dupn         Mask DINO Swin-L (1024) duplicate queries around the candidate: sum of IoU x score over the
                               other queries with 0.3 <= IoU < 0.98 / number with IoU >= 0.5 and score >= 0.05
  e_c_siou_fam                 family mean of the best score x IoU (families: R50 seeds 0-3 | Swin-L 1536 | Cascade Mask
                               R-CNN | RF-DETR-Seg L)
  e_c_qsiou_rw                 reliability-weighted calibrated score x IoU (score -> quantile of that model's scores; the
                               weight of a model = its correlation with the other models, R50 seeds share one family weight)
  e_end_ext                    max nnU-Net probability just beyond the skeleton end points of the candidate (1024 px)
  e_h_ioa / e_h_nsplit         largest fraction of the candidate inside another nnU-Net component / number of distinct Swin-L
                               pieces that split that host component

Folds: a training frame uses the fold models that did not see it (fold k = its validation fold); test frame i (sorted
test_names) uses fold models i % 5, the same assignment as the candidates and the votes.
Calibration: training fold k uses the scores of the fold-k models on the fold-k validation frames; test group k uses the scores
of the fold-k models on all 180 test frames, and the model weights are fitted on each group's candidates (as submitted).
This needs every fold model on every test frame and makes a test frame's columns depend on the other test frames (no labels).
The 7 columns were selected by an evolutionary search on the same pooled OOF (not nested; the threshold tau is nested).

Port of the development scripts evo/g3_m1, g4_m3, g1_m5, g6_m3, g7_m2 and evoc0/g7_m2_test (single process; values identical,
tests/test_golden_stk.py).
"""
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pycocotools import mask as mU
from skimage.morphology import skeletonize

FAMILY_MODELS = ["r50_s0", "r50_s1", "r50_s2", "r50_s3", "swinl1536", "cascade", "rfdetrseg"]   # order matters (weights)
FAM = [[0, 1, 2, 3], [4], [5], [6]]
SMIN = 0.1
K8 = np.ones((3, 3), np.uint8)
COLS = ["e_sw_dup", "e_sw_dupn", "e_c_siou_fam", "e_c_qsiou_rw", "e_end_ext", "e_h_ioa", "e_h_nsplit"]
FEATS = ["c_" + c for c in COLS]
NN_SOURCES = ("v1_b03", "v1_b04")   # candidate sources of the nnU-Net components (names kept from the base pipeline)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _rle(r):
    c = r["counts"]
    return dict(size=list(r["size"]), counts=c.encode() if isinstance(c, str) else c)


def _fam_mean(A):
    return np.stack([np.nanmean(A[f], 0) if len(f) > 1 else A[f[0]] for f in FAM])


def agreement_image(name, cr, P, CAL):
    """Per model j and candidate: best score x IoU (siou), max calibrated score x IoU (qsiou), IoU and score of the best match."""
    n = len(cr)
    F = {f: np.full((len(FAMILY_MODELS), n), np.nan) for f in ("siou", "qsiou", "iou", "s")}
    for j, m in enumerate(FAMILY_MODELS):
        if name not in P[m]:
            continue
        pr = [(s, _rle(r)) for s, r in P[m][name] if s >= SMIN]
        for f in F:
            F[f][j] = 0.0
        if not pr:
            continue
        sc = np.array([s for s, _ in pr])
        rr = [r for _, r in pr]
        q = np.searchsorted(CAL[m], sc, side="right") / len(CAL[m])
        iou = mU.iou(cr, rr, [0] * len(rr)).reshape(n, len(rr))
        ar = np.arange(n)
        b = np.argmax(iou * sc[None, :], axis=1)
        ib = iou[ar, b]
        F["siou"][j] = ib * sc[b]
        F["qsiou"][j] = (iou * q[None, :]).max(1)
        F["iou"][j] = ib
        F["s"][j] = np.where(ib >= 0.2, sc[b], 0.0)
    return F


def family_weights(rows):
    """Weight of model j = max(0, corr(qsiou_j, mean qsiou of the other models)) over the group's candidates; R50 seeds / 4."""
    Q = np.concatenate([F["qsiou"] for F in rows], axis=1)
    w = np.zeros(len(FAMILY_MODELS))
    for j in range(len(FAMILY_MODELS)):
        o = np.nanmean(np.delete(Q, j, 0), 0)
        a = Q[j]
        mk = np.isfinite(a) & np.isfinite(o)
        w[j] = max(np.corrcoef(a[mk], o[mk])[0, 1], 0.0) if mk.sum() > 10 else 0.0
    w[:4] /= 4.0
    return w / max(w.sum(), 1e-9)


def agreement_cols(F, w):
    fs = _fam_mean(F["siou"])
    qs = F["qsiou"]
    mk = np.isfinite(qs)
    rw = np.where(mk, qs, 0.0).T @ w / np.maximum(mk.T.astype(float) @ w, 1e-9)
    with np.errstate(all="ignore"):
        return np.nanmean(fs, 0), rw


def duplicate_queries(cr, pr_all):
    n = len(cr)
    dup = np.zeros(n)
    dupn = np.zeros(n)
    if pr_all:
        sc = np.array([s for s, _ in pr_all])
        rr = [_rle(r) for _, r in pr_all]
        iou = mU.iou(cr, rr, [0] * len(rr)).reshape(n, len(rr))
        oth = iou < 0.98
        dup = (np.where(oth & (iou >= 0.3), iou, 0.0) * sc[None, :]).sum(1)
        dupn = (oth & (iou >= 0.5) & (sc[None, :] >= 0.05)).sum(1).astype(float)
    return dup, dupn


def _n_distinct(sel, ss, pp):
    ids = np.where(sel)[0]
    if len(ids) == 0:
        return 0
    ids = ids[np.argsort(-ss[ids])]
    kept = []
    for i in ids:
        if all(pp[i, j] < 0.3 for j in kept):
            kept.append(i)
    return len(kept)


def host_split(cr, ca, isnn, pr_all):
    """e_h_ioa / e_h_nsplit: containment in another nnU-Net component and the number of Swin-L pieces splitting it."""
    n = len(cr)
    ioa = np.zeros(n)
    nsplit = np.zeros(n)
    hid = np.where(isnn & (ca > 0))[0]
    has = np.zeros(n, bool)
    bh = np.zeros(n, int)
    if len(hid):
        hr = [cr[i] for i in hid]
        ha = ca[hid]
        iou_h = mU.iou(cr, hr, [0] * len(hr)).reshape(n, len(hr))
        inter_h = iou_h * (ca[:, None] + ha[None, :]) / (1 + iou_h)
        ioa_c = inter_h / np.maximum(ca[:, None], 1)
        ioa_c[hid, np.arange(len(hid))] = -1.0
        ioa_c[iou_h >= 0.98] = -1.0
        bh = ioa_c.argmax(1)
        bv = ioa_c[np.arange(n), bh]
        ioa = np.clip(bv, 0, 1)
        has = bv >= 0.3
    pr = [(s, _rle(r)) for s, r in pr_all if s >= SMIN]
    if pr and len(hid):
        ss = np.array([s for s, _ in pr])
        rr = [r for _, r in pr]
        pa = mU.area(rr).astype(np.float64)
        m = len(rr)
        pp = mU.iou(rr, rr, [0] * m).reshape(m, m)
        iou_c = mU.iou(cr, rr, [0] * m).reshape(n, m)
        ioa_p = iou_c * (ca[:, None] + pa[None, :]) / (1 + iou_c) / np.maximum(pa[None, :], 1)
        ioa_ph = ioa_p[hid]
        iou_ph = iou_c[hid]
        for i in np.where(has & (ca > 0))[0]:
            h = bh[i]
            nsplit[i] = _n_distinct((ioa_ph[h] >= 0.5) & (iou_ph[h] < 0.3) & (iou_c[i] < 0.3), ss, pp)
    return ioa, nsplit


def end_ext(m, P):
    """Max of the map P in a 13 px disc around the skeleton end points of mask m, outside m (both 1024 px)."""
    ys, xs = np.nonzero(m)
    if len(ys) < 3:
        return np.nan
    y0, y1, x0, x1 = max(ys.min() - 8, 0), min(ys.max() + 9, m.shape[0]), max(xs.min() - 8, 0), min(xs.max() + 9, m.shape[1])
    mc = m[y0:y1, x0:x1].astype(np.uint8)
    pc = P[y0:y1, x0:x1]
    sk = skeletonize(mc > 0)
    if sk.sum() == 0:
        sk = mc > 0
    nb = cv2.filter2D(sk.astype(np.uint8), -1, K8, borderType=cv2.BORDER_CONSTANT)
    ends = sk & (nb == 2)
    if ends.any():
        ed = (cv2.dilate(ends.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))) > 0) & (mc == 0)
        return float(pc[ed].max()) if ed.any() else 0.0
    return 0.0


def load_map_1024(path):
    """nnU-Net map PNG (2048) -> 1024 float in [0, 1] (INTER_AREA)."""
    P = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert P is not None, path
    return cv2.resize(P, (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def calibration(P, images):
    """Sorted scores (>= SMIN) of every model over the given frames."""
    return {m: np.sort(np.array([s for n in images for s, _ in P[m].get(n, []) if s >= SMIN] or [1.0])) for m in FAMILY_MODELS}


def run(groups, P_of, SW_of, CAL_of, MAP_of, RL):
    """groups: {k: DataFrame(image, idx, source)}; P_of(k) -> {model: {image: [(score, RLE)]}}; SW_of(k) -> Swin-L 1024 queries;
    CAL_of(k, P) -> calibration; MAP_of(k, image) -> 1024 nnU-Net map; RL: {image: {idx: RLE}}."""
    out = []
    t0 = time.time()
    for k, Ck in groups.items():
        P = P_of(k)
        SW = SW_of(k)
        CAL = CAL_of(k, P)
        rows = []
        for name, g in Ck.groupby("image", sort=True):
            idx = g.idx.values.astype(int)
            cr = [_rle(RL[name][i]) for i in idx]
            ca = mU.area(cr).astype(np.float64)
            F = agreement_image(name, cr, P, CAL)
            dup, dupn = duplicate_queries(cr, SW.get(name, []))
            ioa, nsp = host_split(cr, ca, g.source.isin(NN_SOURCES).values, SW.get(name, []))
            Pm = MAP_of(k, name)
            ee = np.array([end_ext(mU.decode(RL[name][i])[::2, ::2], Pm) for i in idx], np.float32)
            rows.append((name, idx, ca > 0, F, dup, dupn, ioa, nsp, ee))
        w = family_weights([r[3] for r in rows])
        log(f"[evofeats] group {k} weights", dict(zip(FAMILY_MODELS, np.round(w, 3).tolist())), f"{time.time() - t0:.0f}s")
        for name, idx, okc, F, dup, dupn, ioa, nsp, ee in rows:
            fam, rw = agreement_cols(F, w)
            df = pd.DataFrame(dict(image=name, idx=idx, e_sw_dup=dup, e_sw_dupn=dupn, e_c_siou_fam=fam, e_c_qsiou_rw=rw,
                                   e_end_ext=ee, e_h_ioa=ioa, e_h_nsplit=nsp))
            df.loc[~okc, COLS] = np.nan
            out.append(df)
    return pd.concat(out, ignore_index=True)


def _load(pat, k):
    return pickle.load(open(os.path.expandvars(pat).format(k=k), "rb"))


def oof_features(c: dict, folds=range(5)) -> pd.DataFrame:
    """Training columns. c = nlv.resolve(cfg): train_cands (with fold), train_rles, evo.oof_preds, train_swinl, train_nnmaps."""
    e = c["evo"]
    C = pd.read_parquet(c["train_cands"], columns=["image", "idx", "fold", "source"])
    RL = pickle.load(open(c["train_rles"], "rb"))
    groups = {k: C[C.fold == k] for k in folds}
    return run(groups,
               lambda k: {m: _load(e["oof_preds"][m], k) for m in FAMILY_MODELS},
               lambda k: _load(c["train_swinl"], k),
               lambda k, P: calibration(P, sorted(groups[k].image.unique())),
               lambda k, n: load_map_1024(Path(c["train_nnmaps"]) / f"{n}_rep.png"), RL)


def test_features(c: dict) -> pd.DataFrame:
    """Test columns. c = nlv.resolve(cfg): test_cands, test_rles, test_names, evo.test_preds, evo.test_swinl_queries, test_nnmaps."""
    e = c["evo"]
    T = pd.read_parquet(c["test_cands"], columns=["image", "idx", "source"])
    RL = pickle.load(open(c["test_rles"], "rb"))
    names = [str(x) for x in np.load(c["test_names"], allow_pickle=True)]
    T["k"] = T.image.map({n: i % 5 for i, n in enumerate(names)})
    assert T.k.notna().all()
    groups = {k: T[T.k == k] for k in range(5)}
    X = run(groups,
            lambda k: {m: _load(e["test_preds"][m], k) for m in FAMILY_MODELS},
            lambda k: _load(e["test_swinl_queries"], k),
            lambda k, P: calibration(P, names),
            lambda k, n: load_map_1024(Path(c["test_nnmaps"].format(k=k)) / f"{n}_rep.png"), RL)
    assert len(X) == len(T)
    return X


def as_verifier_cols(X: pd.DataFrame) -> pd.DataFrame:
    """image, idx + the 7 columns renamed c_e_* (names of the submitted verifier table)."""
    return X[["image", "idx"] + COLS].rename(columns=dict(zip(COLS, FEATS)))
