"""Unit tests of the STK-avg5 + NNRP building blocks on synthetic masks (no data, no GPU)."""
import numpy as np
import pandas as pd
import pytest

from filament_seg import evofeats, io, nlv, refplanes, votes


def box(y0, y1, x0, x1):
    m = np.zeros((2048, 2048), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_c0_keeps_raw_mask_of_shrunk_swinl_candidates():
    raw = {"a": {0: io.encode(box(0, 100, 0, 100)), 1: io.encode(box(0, 100, 0, 100)), 2: io.encode(box(0, 100, 0, 100))}}
    ref = {"a": {0: io.encode(box(0, 50, 0, 100)),    # 0.5 x raw, Swin-L source -> raw
                 1: io.encode(box(0, 70, 0, 100)),    # 0.7 x raw -> stays refined
                 2: io.encode(box(0, 50, 0, 100))}}   # 0.5 x raw, nnU-Net source -> stays refined
    cands = pd.DataFrame(dict(image=["a"] * 3, idx=[0, 1, 2], source=["yolo", "yolo", "v1_b03"]))
    out, n = nlv.apply_c0(cands, ref, raw, 0.6)
    assert n == 1 and out["a"][0] == raw["a"][0] and out["a"][1] == ref["a"][1] and out["a"][2] == ref["a"][2]
    assert ref["a"][0] != raw["a"][0]   # input not modified


def test_fourth_voter_columns_and_order():
    m = io.encode(box(0, 100, 0, 100))
    near, far = io.encode(box(0, 100, 0, 90)), io.encode(box(1000, 1100, 1000, 1100))
    cands = pd.DataFrame(dict(image=["a"], idx=[0]))
    P = {"s0": {"a": [(0.9, near)]}, "s1": {"a": [(0.8, far)]}, "s2": {"a": [(0.7, near)]}, "sl": {"a": [(0.6, near)]}}
    D = votes.oof_votes(cands, {"a": {0: m}}, P, votes.VOTERS_STK)
    assert [c for c in D.columns if c.startswith("v_")] == votes.VFEATS_STK
    r = D.iloc[0]
    assert r.v_agree2 == 1 and r.v_agree3 == 2 and r.v_agree4 == 3 and r.v_sl_sc == 0.6
    D3 = votes.oof_votes(cands, {"a": {0: m}}, {k: P[k] for k in votes.VOTERS})
    assert [c for c in D3.columns if c.startswith("v_")] == votes.VFEATS   # NLv unchanged


def test_feature_lists():
    assert len(nlv.feature_list()) == 70
    stk = nlv.feature_list(votes.VFEATS_STK, evo=True)
    assert len(stk) == 80 and stk[-7:] == evofeats.FEATS


def test_best_match_planes():
    seeds = [io.encode(box(100, 140, 100, 300)), io.encode(box(1500, 1520, 1500, 1520))]
    preds = [io.encode(box(100, 140, 100, 280)), io.encode(box(600, 640, 600, 700))]
    boxes = np.array([[0, 0, 400], [1400, 1400, 128]])
    X, iou = refplanes.best_match_planes(seeds, preds, boxes, size=256)
    assert X.shape == (2, 256, 256) and X.dtype == np.uint8
    assert iou[0] == pytest.approx(0.9) and iou[1] == 0.0
    assert X[0].max() == 255 and X[1].max() == 0      # second seed has no match >= 0.3 -> empty plane
    assert X[0][:, 200:].max() == 0                    # the matched mask ends at x = 280 of the 400-px window (-> 179 px)


def test_end_ext_range_and_empty_mask():
    # The end-point test uses cv2.filter2D on uint8 with a uint8 kernel. On the development build (OpenCV 4.13.0, macOS arm64)
    # that call does not return the 3 x 3 neighbour count for arrays wider than 7 px, so the submitted column is tied to that
    # build (README, "STK reproducibility"). Only build-independent properties are tested here.
    m = np.zeros((1024, 1024), np.uint8)
    m[502, 400:600] = 1
    P = np.zeros((1024, 1024), np.float32)
    P[500:505, 600:605] = 0.7
    v = evofeats.end_ext(m, P)
    assert 0.0 <= v <= 1.0
    assert np.isnan(evofeats.end_ext(np.zeros_like(m), P))


def test_predict_amp_flag_cpu():
    torch = pytest.importorskip("torch")
    from filament_seg import semantic
    torch.manual_seed(0)
    model = torch.nn.Conv2d(3, 1, 3, padding=1)
    x = np.random.default_rng(0).integers(0, 255, (2, 3, 32, 32), dtype=np.uint8)
    cpu = torch.device("cpu")
    p_default = semantic.predict(model, x, cpu)
    p_fp32 = semantic.predict(model, x, cpu, amp=False)
    p_bf16 = semantic.predict(model, x, cpu, amp=True)
    assert np.array_equal(p_default, p_fp32)               # CPU default = fp32 (the submitted test maps)
    assert not np.array_equal(p_fp32, p_bf16) and np.abs(p_fp32 - p_bf16).max() < 0.02
