"""Sanity tests for the PQ implementation and the RLE orientation."""
import numpy as np

from filament_seg.io import encode, decode_str, encode_str
from pycocotools import mask as mU

from filament_seg import metrics
from filament_seg.metrics import PQ, score


def box(y0, y1, x0, x1):
    m = np.zeros((2048, 2048), np.uint8)
    m[y0:y1, x0:x1] = 1
    return m


def test_rle_roundtrip_is_column_major():
    m = box(10, 20, 100, 300)
    assert (decode_str(encode_str(m)) == m).all()
    assert not (decode_str(encode_str(m.T.copy())) == m).all()  # a transposed encoder would fail here


def test_perfect_prediction_scores_one():
    g = [encode(box(0, 50, 0, 50)), encode(box(100, 150, 100, 200))]
    assert score({"s": {"a-s": g}}, {"s": g})["PQ"] == 1.0


def test_hand_computed_example():
    gt = [encode(box(0, 100, 0, 100))]
    pred = [encode(box(0, 100, 0, 60)), encode(box(500, 520, 500, 520))]  # IoU 0.6 hit + 1 FP
    r = score({"s": {"a-s": gt}}, {"s": pred})
    assert (r["TP"], r["FP"], r["FN"]) == (1, 1, 0)
    assert abs(r["PQ"] - 0.6 / 1.5) < 1e-9


def test_records_are_pooled_not_averaged():
    gt = [encode(box(0, 100, 0, 100))]
    r = score({"s": {"a-s": gt, "b-s": []}}, {"s": gt})  # second annotator drew nothing -> our mask is an FP there
    assert (r["TP"], r["FP"], r["FN"]) == (1, 1, 0)
    assert abs(r["PQ"] - 1.0 / 1.5) < 1e-9


def test_strict_threshold():
    acc = PQ()
    acc.add([encode(box(0, 100, 0, 100))], [encode(box(0, 100, 0, 50))])  # IoU exactly 0.5 is not a hit
    assert acc.result()["TP"] == 0


def _rle_bar(x0, x1):
    m = np.zeros((2048, 2048), np.uint8); m[100:120, x0:x1] = 1
    return mU.encode(np.asfortranarray(m))


def test_rubric_counts_fragmentation_and_merging():
    g = [_rle_bar(0, 100), _rle_bar(300, 400)]
    res, tm, iou = metrics.rubric([(g, [_rle_bar(0, 40), _rle_bar(50, 100), _rle_bar(300, 400)])])
    assert (res["TP"], res["FP"], res["FN"]) == (1, 2, 1) and res["gt_one_to_many"] == 0.5 and res["pred_many_to_one"] == 0.0
    res2, _, _ = metrics.rubric([([_rle_bar(0, 40), _rle_bar(50, 100)], [_rle_bar(0, 100)])])
    assert res2["pred_many_to_one"] == 1.0 and res2["gt_one_to_many"] == 0.0
    assert 0.9 < res["dice_mean"] <= 1.0 and len(iou) == 3
