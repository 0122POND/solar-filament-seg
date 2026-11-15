"""Speed options: the defaults reproduce the submitted runs and the fast map features equal the original ones.
Synthetic maps and masks; no data, no GPU."""
from pathlib import Path

import cv2
import numpy as np

from filament_seg import config, io, mapfeats, stages

REPO = Path(__file__).resolve().parents[1]


def _maps(rng):
    p = cv2.GaussianBlur(rng.integers(0, 256, (2048, 2048), dtype=np.uint8), (0, 0), 6)
    p = cv2.normalize(p, None, 0, 255, cv2.NORM_MINMAX)
    a = np.clip(p.astype(int) + rng.integers(-8, 9, p.shape), 0, 255).astype(np.uint8)
    b = np.clip(p.astype(int) + rng.integers(-8, 9, p.shape), 0, 255).astype(np.uint8)
    return np.clip(np.round((a.astype(np.float32) + b) / 2), 0, 255).astype(np.uint8), a, b


def test_defaults_are_the_submitted_runs():
    e = stages.efficiency(config.load(REPO / "configs" / "champion.yaml"))
    assert e == dict(yolo_conf=None, refiner_bs=16)
    assert config.load(REPO / "configs" / "nlv.yaml")["efficiency"]["fast_map_features"] is True  # the submitted NLv run


def test_fast_map_features_are_identical(tmp_path):
    rng = np.random.default_rng(0)
    p, a, b = _maps(rng)
    cv2.imwrite(str(tmp_path / "x_r34.png"), a)
    cv2.imwrite(str(tmp_path / "x_eff.png"), b)
    rles = {}
    for i in range(25):  # boxes, thin lines and one candidate touching the frame border
        m = np.zeros((2048, 2048), np.uint8)
        y, x = rng.integers(0, 1900, 2)
        if i % 3 == 0:
            cv2.line(m, (int(x), int(y)), (int(x) + 140, int(y) + 60), 1, 4)
        else:
            m[y:y + rng.integers(5, 120), x:x + rng.integers(5, 120)] = 1
        if i == 24:
            m[:] = 0
            m[0:30, 2000:2048] = 1
        rles[i] = io.encode(m.astype(bool))
    slow = mapfeats.image_features(tmp_path, "x", rles, list(rles))
    fast = mapfeats.image_features(tmp_path, "x", rles, list(rles), fast=True)
    assert slow == fast
