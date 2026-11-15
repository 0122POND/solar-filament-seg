"""Panoptic Quality exactly as the host's Self-Evaluation notebook computes it.

- Ground truth is scored per annotator record; the same prediction set is compared with each record.
- hit = IoU > 0.5 (strict). No one-to-one constraint: every hit cell adds its IoU to the TP sum.
- FP = predictions that hit no GT of the record; FN = GT that no prediction hits.
- PQ = sum(IoU_TP) / (|TP| + 0.5|FP| + 0.5|FN|), pooled over all records (not an image average).
"""
from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_utils

IOU_THR = 0.5


def iou_matrix(gt: list, pred: list) -> np.ndarray:
    """(n_gt, n_pred) IoU matrix between two lists of RLEs."""
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)))
    return np.asarray(mask_utils.iou(pred, gt, [0] * len(gt))).reshape(len(pred), len(gt)).T


@dataclass
class PQ:
    """Accumulate records with add(); read the pooled score with result()."""
    tp_iou: list = field(default_factory=list)
    fp: int = 0
    fn: int = 0

    def add(self, gt: list, pred: list):
        if not gt:
            self.fp += len(pred)
            return
        if not pred:
            self.fn += len(gt)
            return
        hit = (iou := iou_matrix(gt, pred)) > IOU_THR
        self.tp_iou.extend(iou[hit].tolist())
        self.fp += int((hit.sum(axis=0) == 0).sum())
        self.fn += int((hit.sum(axis=1) == 0).sum())

    def result(self) -> dict:
        tp = len(self.tp_iou)
        den = tp + 0.5 * self.fp + 0.5 * self.fn
        return {"PQ": sum(self.tp_iou) / den if den else 0.0,
                "SQ": float(np.mean(self.tp_iou)) if tp else 0.0,
                "RQ": tp / den if den else 0.0,
                "TP": tp, "FP": self.fp, "FN": self.fn}


def score(records: dict, predictions: dict) -> dict:
    """records: {stem: {record_id: [RLE]}}, predictions: {stem: [RLE]} -> pooled PQ."""
    acc = PQ()
    for stem, recs in records.items():
        for gt in recs.values():
            acc.add(gt, predictions.get(stem, []))
    return acc.result()


# ----------------------------------------------------------------------------- rubric metrics (quantitative part of the judging)

def union_mask(rles: list, shape=(2048, 2048)) -> np.ndarray:
    """Union of a list of RLEs as a uint8 mask (all zeros for an empty list)."""
    return mask_utils.decode(mask_utils.merge(list(rles), intersect=False)).astype(np.uint8) if rles else np.zeros(shape, np.uint8)


def rubric(pairs) -> tuple:
    """Rubric metrics of (GT RLEs of one annotator record, prediction RLEs of the frame) pairs.

    Returns (summary, per-record Dice, IoU of every overlapping pair):
      PQ / SQ / RQ / TP / FP / FN    official pooled PQ (as `score`)
      dice_mean / dice_median        torchmetrics.segmentation.DiceScore (the host's mean Dice) of the union masks, per record;
                                     records where both GT and prediction are empty give NaN and are skipped (dice_nan_records)
      pair_iou_mean / pair_dice_mean IoU / Dice distribution of all (GT, prediction) pairs with IoU > 0
      gt_missed / gt_one_to_many     share of GT instances overlapped by 0 / by >= 2 predictions (fragmentation)
      pred_unmatched / pred_many_to_one  share of predictions overlapping 0 / >= 2 GT instances (over-merging)
    Annotator-vs-annotator pairs give the human reference with the same function. Needs torch and torchmetrics.
    Port of the development script rubric_eval.record_metrics."""
    from collections import Counter

    import torch
    from torchmetrics.segmentation import DiceScore
    dsc = DiceScore(num_classes=1, include_background=True, average="macro", aggregation_level="samplewise", input_format="one-hot")
    acc = PQ(); tm, ious, gdeg, pdeg = [], [], [], []
    for g, p in pairs:
        acc.add(g, p)
        tm.append(float(dsc(torch.from_numpy(union_mask(p))[None, None], torch.from_numpy(union_mask(g))[None, None])))
        if g and p:
            iou = iou_matrix(g, p); h = iou > 0
            ious += iou[h].tolist(); gdeg += h.sum(1).tolist(); pdeg += h.sum(0).tolist()
        else:
            gdeg += [0] * len(g); pdeg += [0] * len(p)
    ious = np.array(ious); tm = np.array(tm); gd, pd_ = Counter(gdeg), Counter(pdeg); ng, npd = max(len(gdeg), 1), max(len(pdeg), 1)
    res = dict(**acc.result(), dice_mean=float(np.nanmean(tm)) if len(tm) else 0.0, dice_median=float(np.nanmedian(tm)) if len(tm) else 0.0,
               dice_nan_records=int(np.isnan(tm).sum()),
               pair_iou_mean=float(ious.mean()) if len(ious) else 0.0, pair_dice_mean=float((2 * ious / (1 + ious)).mean()) if len(ious) else 0.0,
               gt_missed=gd[0] / ng, gt_one_to_many=sum(v for k, v in gd.items() if k >= 2) / ng,
               pred_unmatched=pd_[0] / npd, pred_many_to_one=sum(v for k, v in pd_.items() if k >= 2) / npd)
    return res, tm, ious
