"""Unit tests of the pipeline building blocks on synthetic masks (no data, no GPU)."""
import numpy as np
import pytest

from filament_seg import candidates, features, io, preprocess, refiner, verifier


def box(y0, y1, x0, x1):
    m = np.zeros((2048, 2048), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_make_input_channels():
    img = np.full((2048, 2048), 120, np.uint8)
    img[1000:1010, 900:1100] = 40  # a dark filament
    x = preprocess.make_input(img)
    assert x.shape == (3, 1024, 1024) and x.dtype == np.uint8
    assert x[1, 502, 500] > 128  # local contrast is positive on dark structures


def test_decode_is_overlap_free_and_ordered():
    a, b, c = box(0, 100, 0, 100), box(0, 100, 50, 150), box(500, 520, 500, 520)
    rles = [io.encode(m) for m in (a, b, c)]
    out = verifier.decode_image(rles, [0.9, 0.8, 0.1], tau=0.25, nms_iou=0.5, keep_frac=0.5)
    assert len(out) == 2  # c is below tau
    m0, m1 = io.decode(out[0]), io.decode(out[1])
    assert (m0 == a).all() and not (m0 & m1).any() and m1.sum() == 100 * 50


def test_decode_nms_drops_near_duplicates():
    a, b = box(0, 100, 0, 100), box(0, 100, 0, 95)
    out = verifier.decode_image([io.encode(a), io.encode(b)], [0.9, 0.8], tau=0.25, nms_iou=0.3)
    assert len(out) == 1


def test_crop_geometry_at_the_border():
    x0, y0, side = refiner.crop_box(0, 0, 40, 10)
    assert side == 128 and (x0, y0) == (-44, -59)
    img = np.arange(2048 * 2048, dtype=np.int64).reshape(2048, 2048)
    c = refiner.pad_crop(img, x0, y0, side)
    assert c.shape == (128, 128) and c[0, 0] == 0 and c[59, 44] == img[0, 0]


def test_paste_back_keeps_the_component_touching_the_seed():
    seed = box(100, 110, 100, 200)
    prob = np.zeros((256, 256), np.float32)
    x0, y0, side = 0, 0, 512
    prob[48:58, 48:110] = 0.9   # overlaps the seed (2x scale)
    prob[200:210, 200:210] = 0.9  # an unrelated blob
    m, status = refiner.paste_back(prob, x0, y0, side, seed)
    assert status == "ok" and m[105, 150] and not m[410, 410]
    m, status = refiner.paste_back(np.zeros((256, 256), np.float32), x0, y0, side, seed)
    assert status == "empty" and (m == seed).all()


def test_residual_refiner_starts_as_identity():
    torch = pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    model = refiner.ResidualRefiner(alpha=1.0, pretrained=False).eval()
    X = np.zeros((2, 4, 64, 64), np.uint8)
    X[:, 2, 20:40, 10:50] = 255
    S2 = np.zeros((2, 2), np.float32)
    p = refiner.predict_residual(model, torch.from_numpy(X), torch.from_numpy(S2), torch.device("cpu"))
    assert np.allclose(p > 0.5, X[:, 2] > 127)


def test_featurize_target_is_mean_matched_iou():
    gt = [io.encode(box(0, 100, 0, 100))]
    cand = box(0, 100, 0, 80)  # IoU 0.8 with the GT
    p = np.zeros((2048, 2048), np.float32)
    p[cand] = 0.7
    raw = np.full((2048, 2048), 100, np.uint8)
    cl = [("v1_b04", 0.7, cand, candidates.bbox_of(cand))]
    rows, _, _ = features.featurize("s", p, raw, cl, [gt, []])  # second annotator drew nothing
    assert abs(rows[0]["y"] - 0.8 / 2) < 1e-9 and rows[0]["n_hit_sets"] == 1
    assert set(features.FEATS_ALL) <= set(rows[0])
