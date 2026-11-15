"""Unit tests of the optional decode settings nlv.DECODE_KNOBS (order_bonus, iom_max) on synthetic masks (CPU, no weights)."""
import numpy as np
import pandas as pd
import pytest
from pycocotools import mask as mU

from filament_seg import config, nlv


def _rle(x0, x1, y0=100, y1=110):
    m = np.zeros((2048, 2048), np.uint8); m[y0:y1, x0:x1] = 1
    return mU.encode(np.asfortranarray(m))


def _areas(out):
    return [int(mU.area(r)) for r in out]


# three candidates in one image: 0 = nnU-Net component, 1 = Swin-L, 2 = nnU-Net component inside candidate 0
R = {0: _rle(0, 100), 1: _rle(60, 160), 2: _rle(10, 30, 102, 108)}
G = pd.DataFrame(dict(image="a", idx=[0, 1, 2], yhat=[0.50, 0.49, 0.30], swinl=[False, True, False]))


def test_defaults_are_off_and_every_config_keeps_the_plain_decode():
    assert nlv.DECODE_KNOBS == dict(order_bonus=0.0, iom_max=None)
    for name in ("nlv", "stk", "stk_nnrp", "final_fullfit"):
        assert nlv.decode_knobs(config.load(config.REPO / "configs" / f"{name}.yaml")) == nlv.DECODE_KNOBS, name
    assert nlv.decode_knobs(config.load(config.REPO / "configs" / "final_yellow.yaml")) == dict(order_bonus=0.10, iom_max=None)


def test_order_bonus_changes_the_order_only():
    plain = nlv.decode_image(G, R, tau=0.25)                                 # candidate 0 (0.50) first; candidate 1 loses the 400 shared pixels
    assert _areas(plain) == [1000, 600] and mU.decode(plain[0])[105, 10] == 1 and mU.decode(plain[0])[105, 150] == 0
    bonus = nlv.decode_image(G, R, tau=0.25, order_bonus=0.02)               # candidate 1 (0.49 + 0.02) first: it keeps all its pixels
    assert _areas(bonus) == [1000, 600] and mU.decode(bonus[0])[105, 150] == 1 and mU.decode(bonus[0])[105, 10] == 0
    assert nlv.decode_image(G, R, tau=0.25, order_bonus=0.0) == plain
    assert nlv.decode_image(G, R, tau=0.25, order_bonus=0.005) == plain      # 0.49 + 0.005 < 0.50: order unchanged


def test_order_bonus_does_not_change_which_candidates_are_accepted():
    g = G.assign(yhat=[0.50, 0.24, 0.30])                                    # the Swin-L candidate is below tau: 0.24 + 0.02 must not admit it
    assert nlv.decode_image(g, R, tau=0.25, order_bonus=0.02) == nlv.decode_image(g, R, tau=0.25)
    assert len(nlv.decode_image(g, {0: R[0], 1: R[1], 2: _rle(500, 520)}, tau=0.25, order_bonus=0.02)) == 2


def test_order_bonus_needs_the_source_column():
    with pytest.raises(AttributeError):
        nlv.decode_image(G.drop(columns="swinl"), R, tau=0.25, order_bonus=0.02)


def test_iom_max_drops_a_mostly_contained_mask():
    g = G.iloc[[0, 2]]
    r = {0: R[0], 2: _rle(10, 50, 85, 110)}                                  # 1000 px, 400 inside candidate 0: IoU 0.25, IoM 0.4, keeps 60 %
    assert _areas(nlv.decode_image(g, r, tau=0.25)) == [1000, 600]           # passes mask NMS 0.3 and the 50 % rule
    assert _areas(nlv.decode_image(g, r, tau=0.25, iom_max=0.5)) == [1000, 600]
    assert _areas(nlv.decode_image(g, r, tau=0.25, iom_max=0.3)) == [1000]


def test_with_source_keeps_the_row_order():
    Y = pd.DataFrame(dict(image=["b", "a", "a"], idx=[0, 1, 0], yhat=[0.1, 0.2, 0.3]))
    cands = pd.DataFrame(dict(image=["a", "a", "b"], idx=[0, 1, 0], source=["v1_b03", "yolo", "yolo"]))
    out = nlv.with_source(Y, cands)
    assert out[["image", "idx", "yhat"]].equals(Y) and out.swinl.tolist() == [True, True, False]
    with pytest.raises(AssertionError):
        nlv.with_source(Y, cands.iloc[:2])


def test_decode_passes_the_settings_and_rejects_unknown_ones():
    rles = {"a": R}
    assert nlv.decode(G, rles, 0.25, order_bonus=0.02)[1:] == [f"a_{k + 1},{r['counts'].decode('ascii')}"
                                                               for k, r in enumerate(nlv.decode_image(G, R, 0.25, order_bonus=0.02))]
    with pytest.raises(AssertionError):
        nlv.decode(G, rles, 0.25, order_bonus_typo=0.02)
