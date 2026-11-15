"""Per-candidate features and the PQ-aware training target of the verifier.

Target: y = mean over annotator records of (IoU with the best-matching GT if IoU > 0.5 else 0).
This is exactly the expected contribution of the candidate to the PQ numerator, so a candidate is worth
emitting when E[y] exceeds about PQ / 2 (it also adds 0.5 to the denominator when it misses).

Feature groups (FEATS_ALL, 48 columns): source and confidence, cluster statistics of overlapping candidates
(IoU > 0.7 union-find), size and shape (elongation, perimeter / area, solidity, fill), U-Net++ probability
statistics inside and around the mask, raw-intensity and local-contrast statistics, agreement with the
other source (best IoU with a YOLO / probability candidate), and image-level context.
"""
import cv2
import numpy as np
from pycocotools import mask as mask_utils

from .candidates import SOURCES
from .io import H, W, encode
from .preprocess import local_contrast

FEATS_ALL = ["src_id", "conf", "cluster_size", "cluster_n_src", "cluster_max_conf", "area", "log_area", "w", "h", "aspect", "r_disk",
             "elong", "per_area", "solidity", "n_parts", "fill", "p_mean", "p_max", "p_std", "p_q25", "p_q75", "f30", "f50", "f70", "f90", "mass",
             "ring_p_mean", "ring_f30", "raw_mean", "raw_std", "raw_q10", "raw_ring", "raw_contrast", "lc_mean", "lc_q75", "lc_f10",
             "best_yolo_iou", "best_yolo_conf", "best_v1_iou", "n_yolo_ovl", "n_v1_ovl", "v1_cov04", "v1_cov03",
             "img_pmean", "img_f30", "n_v1_04", "n_yolo25", "n_cands"]


def shape_feats(sub: np.ndarray) -> dict:
    """Bounding-box sub-mask -> elongation, perimeter / area, solidity, number of parts."""
    ys, xs = np.nonzero(sub)
    if len(xs) > 2:
        ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([xs, ys]))))[::-1]
        elong = float(np.sqrt(max(ev[0], 1e-6) / max(ev[1], 1e-6)))
    else:
        elong = 1.0
    cnts, _ = cv2.findContours(sub.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    per = float(sum(cv2.arcLength(c, True) for c in cnts))
    hull = float(sum(cv2.contourArea(cv2.convexHull(c)) for c in cnts))
    area = int(sub.sum())
    return dict(elong=elong, per_area=per / max(area, 1), solidity=area / max(hull, 1.0), n_parts=len(cnts))


def clusters(iou_cc: np.ndarray, thr: float) -> np.ndarray:
    """Union-find over candidate pairs with IoU > thr -> cluster id per candidate (in order of first appearance)."""
    n = len(iou_cc)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if iou_cc[i, j] > thr:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    cid = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    return np.array([cid[r] for r in roots], dtype=int)


def featurize(name: str, p: np.ndarray, raw: np.ndarray, cands: list, gt_sets: list, lc=None, cluster_iou: float = 0.7):
    """Features, labels and clusters for the candidates of one image.

    p: 2048 probability map, raw: 2048 frame, cands: [(source, conf, mask2048, bbox)],
    gt_sets: one GT RLE list per annotator record ([] at test time, then y = 0).
    Returns (rows, rles, gt_recall): one feature dict and one RLE per candidate, and per-GT best IoUs.
    """
    if lc is None:
        lc = local_contrast(raw)
    n = len(cands)
    n_sets = len(gt_sets)
    rles = [encode(c[2]) for c in cands]
    iou_cc = np.asarray(mask_utils.iou(rles, rles, [0] * n)).reshape(n, n) if n else np.zeros((0, 0))
    src = np.array([SOURCES[c[0]] for c in cands])
    is_yolo = src == 2
    is_v1 = ~is_yolo
    confs = np.array([c[1] for c in cands]) if n else np.zeros(0)
    cid = clusters(iou_cc, cluster_iou)

    # labels
    y = np.zeros(n)
    mx = np.zeros(n)
    nhit = np.zeros(n, dtype=int)
    gt_recall = []
    for si, g in enumerate(gt_sets):
        if not g:
            continue
        if n:
            iou = np.asarray(mask_utils.iou(rles, g, [0] * len(g))).reshape(n, len(g))  # (n_cand, n_gt)
            best = iou.max(axis=1)
            y += np.where(best > 0.5, best, 0.0)
            mx = np.maximum(mx, best)
            nhit += best > 0.5
            bv1 = iou[is_v1].max(axis=0) if is_v1.any() else np.zeros(len(g))
            byo = iou[is_yolo].max(axis=0) if is_yolo.any() else np.zeros(len(g))
        else:
            bv1 = byo = np.zeros(len(g))
        for gi, gr in enumerate(g):
            gt_recall.append(dict(image=name, set=si, gt_area=int(mask_utils.area(gr)), best_v1=float(bv1[gi]),
                                  best_yolo=float(byo[gi]), best_union=float(max(bv1[gi], byo[gi]))))
    y = y / max(n_sets, 1)

    # image-level context
    n_v1_04 = int((src == 1).sum())
    n_yolo25 = int((is_yolo & (confs >= 0.25)).sum()) if n else 0
    img_pmean = float(p.mean())
    img_f30 = float((p > 0.3).mean())

    rows = []
    for i, (s, conf, m, (x, yb, w, h)) in enumerate(cands):
        area = int(m.sum())
        sf = shape_feats(m[yb:yb + h, x:x + w])
        ps = p[m]
        rs = raw[m].astype(np.float32)
        ls = lc[m].astype(np.float32)
        # ring = bbox enlarged by half its size on each side, minus the mask
        X0, Y0, X1, Y1 = max(0, x - w // 2), max(0, yb - h // 2), min(W, x + w + w // 2), min(H, yb + h + h // 2)
        ring_p = p[Y0:Y1, X0:X1]
        ring_m = m[Y0:Y1, X0:X1]
        ring_raw = raw[Y0:Y1, X0:X1][~ring_m].astype(np.float32) if (~ring_m).any() else rs
        r_disk = float(np.hypot(x + w / 2 - W / 2, yb + h / 2 - H / 2) / (W / 2))
        # agreement with the other candidates
        others_yolo = is_yolo.copy()
        others_yolo[i] = False
        others_v1 = is_v1.copy()
        others_v1[i] = False
        iou_yolo = iou_cc[i][others_yolo]
        iou_v1 = iou_cc[i][others_v1]
        members = cid == cid[i]
        rows.append(dict(
            image=name, idx=i, source=s, src_id=int(src[i]), conf=float(conf), y=float(y[i]), max_iou=float(mx[i]),
            n_hit_sets=int(nhit[i]), n_sets=n_sets,
            cluster=int(cid[i]), cluster_size=int(members.sum()), cluster_n_src=int(len(set(src[members].tolist()))),
            cluster_max_conf=float(confs[members & is_yolo].max()) if (members & is_yolo).any() else 0.0,
            area=area, log_area=float(np.log(area)), x=x, y0=yb, w=w, h=h, aspect=max(w, h) / max(min(w, h), 1), r_disk=r_disk,
            elong=sf["elong"], per_area=sf["per_area"], solidity=sf["solidity"], n_parts=sf["n_parts"], fill=area / max(w * h, 1),
            p_mean=float(ps.mean()), p_max=float(ps.max()), p_std=float(ps.std()),
            p_q25=float(np.percentile(ps, 25)), p_q75=float(np.percentile(ps, 75)),
            f30=float((ps > 0.3).mean()), f50=float((ps > 0.5).mean()), f70=float((ps > 0.7).mean()), f90=float((ps > 0.9).mean()),
            mass=float(ps.sum()), ring_p_mean=float(ring_p.mean()), ring_f30=float((ring_p > 0.3).mean()),
            raw_mean=float(rs.mean()), raw_std=float(rs.std()), raw_q10=float(np.percentile(rs, 10)), raw_ring=float(ring_raw.mean()),
            raw_contrast=float(ring_raw.mean() - rs.mean()), lc_mean=float(ls.mean()), lc_q75=float(np.percentile(ls, 75)),
            lc_f10=float((ls > 10).mean()),
            best_yolo_iou=float(iou_yolo.max()) if iou_yolo.size else 0.0,
            best_yolo_conf=float(confs[others_yolo][iou_yolo.argmax()]) if iou_yolo.size else 0.0,
            best_v1_iou=float(iou_v1.max()) if iou_v1.size else 0.0,
            n_yolo_ovl=int((iou_yolo > 0.3).sum()), n_v1_ovl=int((iou_v1 > 0.3).sum()),
            v1_cov04=float((ps >= 0.4).mean()), v1_cov03=float((ps >= 0.3).mean()),
            img_pmean=img_pmean, img_f30=img_f30, n_v1_04=n_v1_04, n_yolo25=n_yolo25, n_cands=n,
        ))
    return rows, rles, gt_recall
