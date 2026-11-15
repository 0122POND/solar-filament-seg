"""detect-test: which test frames each Mask DINO fold model predicts (no GPU, no detectron2: load / predict are replaced)."""
import pickle

import cv2
import numpy as np

from filament_seg import __main__ as cli
from filament_seg import maskdino_cands


def _frames(folder, n=12):
    folder.mkdir(parents=True)
    for i in range(n):
        cv2.imwrite(str(folder / f"2015010{i:02d}Mh.jpeg"), np.full((16, 16), i, np.uint8))
    return sorted(folder.glob("*.jpeg"))


def _fake(monkeypatch):
    calls = []
    monkeypatch.setattr(maskdino_cands, "load", lambda root, path, device="cuda", arch="r50": (str(path), arch))
    monkeypatch.setattr(maskdino_cands, "predict", lambda model, img, size=1024: calls.append((model[0], size)) or [(0.5, int(img.mean().round()))])
    return calls


def test_fold_frames():
    assert maskdino_cands.fold_frames(12, 2) == [2, 7]
    assert maskdino_cands.fold_frames(12, 2, all_folds=True) == list(range(12))
    own = sorted(i for k in range(5) for i in maskdino_cands.fold_frames(180, k))
    assert own == list(range(180))   # every frame once, by fold model i % 5


def test_predict_test_own_fold_and_all_folds(tmp_path, monkeypatch):
    files = _frames(tmp_path / "img"); calls = _fake(monkeypatch)
    w = {k: tmp_path / f"m{k}.pt" for k in range(5)}
    n = maskdino_cands.predict_test(tmp_path, w, files, str(tmp_path / "own_f{k}.pkl"), log=lambda *a: None)
    assert n == len(files) == len(calls)
    own = [pickle.load(open(tmp_path / f"own_f{k}.pkl", "rb")) for k in range(5)]
    for k in range(5):
        assert list(own[k]) == [f.stem for i, f in enumerate(files) if i % 5 == k]
    n = maskdino_cands.predict_test(tmp_path, w, files, str(tmp_path / "all_f{k}.pkl"), all_folds=True, log=lambda *a: None)
    assert n == 5 * len(files)
    for k in range(5):
        full = pickle.load(open(tmp_path / f"all_f{k}.pkl", "rb"))
        assert list(full) == [f.stem for f in files]
        assert all(full[s] == v for s, v in own[k].items())   # the own-fold file is the part of the all-folds file that is read


def test_detect_test_cli_defaults(tmp_path, monkeypatch, capsys):
    files = _frames(tmp_path / "test" / "test_images"); calls = _fake(monkeypatch)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"paths:\n  data: {tmp_path}\n  out: {tmp_path / 'out'}\n")
    cli.main(["--config", str(cfg), "detect-test", "--model", "r50", "--weights", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    assert len(calls) == 3 * len(files)   # 3 voter seeds, one fold model per frame
    assert sorted(p.name for p in (tmp_path / "o").glob("*.pkl")) == [f"maskdino_test_s{s}_f{k}.pkl" for s in range(3) for k in range(5)]
    assert {c[0] for c in calls} == {str(tmp_path / "w" / f"nlv/maskdino_r50_s{s}_f{k}.pt") for s in range(3) for k in range(5)}
    del calls[:]
    cli.main(["--config", str(cfg), "detect-test", "--model", "swinl1024", "--all-folds", "--weights", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    assert len(calls) == 5 * len(files)
    del calls[:]
    cli.main(["--config", str(cfg), "detect-test", "--model", "swinl1536", "--weights", str(tmp_path / "w"), "--out", str(tmp_path / "o")])
    assert len(calls) == 5 * len(files) and {c[1] for c in calls} == {1536}   # refiner planes / S15F: every fold model on every frame
    assert '"all_folds": true' in capsys.readouterr().out
