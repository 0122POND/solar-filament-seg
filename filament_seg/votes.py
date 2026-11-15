"""Vote features for the NLv selector of the final model: agreement of each candidate with three Mask DINO R50 models (seeds 0 / 1 / 2).

The R50 models are not candidate sources; they only vote. For every candidate mask m and every voter k:
  v_<k>_iou = max IoU of m with the voter's instances, v_<k>_sc = score of that instance (0 if the IoU is 0).
  v_agree2 = number of voters s0, s1 with IoU >= 0.5; v_agree3 = v_agree2 + (s2 with IoU >= 0.5).
8 features (VFEATS). Training frames use the out-of-fold voters (the fold model that did not see the frame);
test frame i (in test_names order) uses the fold model i % 5 of every seed, as the Mask DINO candidates.
STK (configs/stk.yaml) adds Mask DINO Swin-L at 1536 px as a 4th voter "sl" (not a candidate source):
v_sl_iou, v_sl_sc and v_agree4 = v_agree3 + (sl with IoU >= 0.5) -> 11 features (VFEATS_STK).
Port of scripts/sel_vote_rs.py (vote) and scripts/vote_build.py of the development repository.
"""
import numpy as np
import pandas as pd
from pycocotools import mask as mU

VOTERS = ("s0", "s1", "s2")
VFEATS = ["v_s0_iou", "v_s0_sc", "v_s1_iou", "v_s1_sc", "v_s2_iou", "v_s2_sc", "v_agree2", "v_agree3"]
VOTERS_STK = VOTERS + ("sl",)
VFEATS_STK = VFEATS[:6] + ["v_sl_iou", "v_sl_sc", "v_agree2", "v_agree3", "v_agree4"]   # column order of the submitted STK run


def rle(r):
    """Mask DINO pickles hold RLEs or bool arrays; votes work on RLEs."""
    return r if isinstance(r, dict) else mU.encode(np.asfortranarray(r.astype(np.uint8)))


def vote(name: str, ids: list, masks: list, voters: dict) -> list:
    """Rows (image, idx, v_<k>_iou, v_<k>_sc, v_agree, v_ssum) of the candidates ids / masks (RLE) of one image.

    voters: {key: [(score, RLE)]} in the order of VOTERS."""
    out = []
    for i, m in zip(ids, masks):
        d = dict(image=name, idx=int(i)); agree = 0; ssum = 0.
        for k, L in voters.items():
            if not L:
                d[f"v_{k}_iou"] = 0.; d[f"v_{k}_sc"] = 0.; continue
            iou = np.asarray(mU.iou([m], [r for _, r in L], [0] * len(L))).reshape(-1); j = int(iou.argmax())
            d[f"v_{k}_iou"] = float(iou[j]); d[f"v_{k}_sc"] = float(L[j][0]) if iou[j] > 0 else 0.
            if iou[j] >= 0.5:
                agree += 1; ssum += float(L[j][0])
        d["v_agree"] = agree; d["v_ssum"] = ssum; out.append(d)
    return out


def finish(rows: list) -> pd.DataFrame:
    """Vote rows -> table with v_agree2 / v_agree3 (+ v_agree4 with the 4th voter sl; v_agree / v_ssum dropped, as in the submitted runs)."""
    D = pd.DataFrame(rows)
    D["v_agree2"] = (D.v_s0_iou >= 0.5).astype(int) + (D.v_s1_iou >= 0.5).astype(int)
    D["v_agree3"] = D.v_agree2 + (D.v_s2_iou >= 0.5).astype(int)
    if "v_sl_iou" in D.columns:
        D["v_agree4"] = D.v_agree3 + (D.v_sl_iou >= 0.5).astype(int)
    return D.drop(columns=["v_agree", "v_ssum"])


def oof_votes(cands: pd.DataFrame, cand_rles: dict, oof_preds: dict, voters=VOTERS) -> pd.DataFrame:
    """Training votes. oof_preds: {voter: {image: [(score, mask)]}} merged over the five fold-validation pickles."""
    rows = []
    for n, g in cands.groupby("image"):
        ids = g.idx.astype(int).tolist()
        rows += vote(n, ids, [cand_rles[n][i] for i in ids], {k: [(float(s), rle(r)) for s, r in oof_preds[k].get(n, [])] for k in voters})
    return finish(rows)


def test_votes(cands: pd.DataFrame, cand_rles: dict, names: list, test_preds: dict, voters=VOTERS) -> pd.DataFrame:
    """Test votes. test_preds: {voter: [5 dicts {image: [(score, mask)]}, one per fold model]}; frame i uses fold i % 5."""
    rows = []
    for i, n in enumerate(names):
        g = cands[cands.image == n]
        if not len(g):
            continue
        ids = g.idx.astype(int).tolist()
        rows += vote(n, ids, [cand_rles[n][j] for j in ids], {k: [(float(s), rle(r)) for s, r in test_preds[k][i % 5].get(n, [])] for k in voters})
    return finish(rows)
