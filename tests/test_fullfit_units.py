"""Unit tests of the full-data refiner path (configs/final_fullfit.yaml and the final model configs/final_yellow.yaml; synthetic data, CPU, no weights)."""
import numpy as np
import pandas as pd
import pytest
import torch

from filament_seg import config, nlv, refiner, stages

CFG_PATH = config.REPO / "configs" / "final_fullfit.yaml"
TAGS = ["FFNNRPo", "FFNNRPx11", "FFNNRPx12", "FFS15Fo", "FFS15Fx11", "FFS15Fx12"]


def _model(feat, seed):
    torch.manual_seed(seed)
    m = refiner.ResidualRefiner(1.0, pretrained=False, in_channels=9)
    if feat:
        m.add_feat(16)
    with torch.no_grad():  # the head starts at 0 (output = seed): give it weights so that the fold inputs matter
        torch.nn.init.normal_(m.net.segmentation_head[0].weight, std=0.5)
        if feat:
            torch.nn.init.normal_(m.fproj.weight, std=0.5)
    return m.eval()


def _inputs():
    X = np.random.default_rng(1).integers(0, 256, (3, 4, 64, 64)).astype(np.uint8)
    planes = lambda k: np.random.default_rng(10 + k).integers(0, 256, (3, 3, 64, 64)).astype(np.uint8)  # noqa: E731
    S2 = np.zeros((3, 2), np.float32)
    FT = {k: np.random.default_rng(20 + k).integers(0, 256, (3, 16, 16, 16)).astype(np.uint8) for k in range(5)}
    return X, planes, S2, FT


def test_full_member_is_the_mean_over_the_five_fold_inputs(tmp_path):
    pytest.importorskip("segmentation_models_pytorch")   # GPU path only (requirements.txt); not in requirements-cpu.txt
    X, planes, S2, FT = _inputs()
    torch.save({"state": _model(True, 3).state_dict()}, tmp_path / "full.pt")
    full = dict(checkpoints=[tmp_path / "full.pt"] * 5, feats=FT.__getitem__, full=True)
    p = refiner.ensemble_members([full], X, S2, planes)
    m = refiner.load_residual(tmp_path / "full.pt", 9, 1.0, 16)
    ref = np.zeros_like(p)
    for k in range(5):
        ref += refiner.predict_residual(m, torch.from_numpy(np.concatenate([X, planes(k)], 1)), torch.from_numpy(S2),
                                        torch.device("cpu"), 16, FT=FT[k])
    assert np.array_equal(p, ref / 5)
    assert np.abs(p - refiner.predict_residual(m, torch.from_numpy(np.concatenate([X, planes(0)], 1)), torch.from_numpy(S2),
                                               torch.device("cpu"), 16, FT=FT[0])).max() > 1e-4   # the fold inputs differ


def test_member_mean_order_is_sum_over_members_divided_by_k_m(tmp_path):
    pytest.importorskip("segmentation_models_pytorch")   # GPU path only (requirements.txt); not in requirements-cpu.txt
    X, planes, S2, FT = _inputs()
    mem, sums = [], []
    for j, feat in enumerate((False, False, True)):
        torch.save({"state": _model(feat, j).state_dict()}, tmp_path / f"m{j}.pt")
        mem.append(dict(checkpoints=[tmp_path / f"m{j}.pt"] * 5, feats=FT.__getitem__ if feat else None))
        sums.append(refiner.ensemble_members([mem[-1]], X, S2, planes) * np.float32(5))
    p = refiner.ensemble_members(mem, X, S2, planes)
    assert np.allclose(p, np.mean(sums, 0) / 5, atol=1e-6)
    s = [np.zeros_like(p) for _ in mem]   # exact sums over k, as the development run stored them (sum_<member>.npy)
    for k in range(5):
        Xk = torch.from_numpy(np.concatenate([X, planes(k)], 1))
        for j, m in enumerate(mem):
            s[j] += refiner.predict_residual(refiner.load_residual(m["checkpoints"][k], 9, 1.0), Xk, torch.from_numpy(S2),
                                             torch.device("cpu"), 16, FT=m["feats"](k) if m["feats"] else None)
    assert np.array_equal(p, nlv.member_sum_mean(iter(s), K=5))


def test_residual_rows_full_data():
    meta = pd.DataFrame(dict(fold=[0, 1, 2, 3, 4, 0, 1]))
    tr, va = stages.residual_rows(meta, -1)
    assert tr.tolist() == list(range(7)) and len(va) == 0
    tr, va = stages.residual_rows(meta, 0)
    assert va.tolist() == [0, 5] and tr.tolist() == [1, 2, 3, 4, 6]


def test_final_fullfit_config():
    cfg = config.load(CFG_PATH)
    c = nlv.resolve(cfg)
    assert cfg["decode"]["tau"] == 0.25 and cfg["refiner"]["c0_ratio"] == 0.6 and cfg["decode"]["nms_iou"] == 0.3
    assert cfg["verifier"]["seeds"] == [0] and c["voters"] == ("s0", "s1", "s2") and "evo" not in c
    assert "sl" not in c["test_voters"] and "sl" not in c["oof_voters"] and not nlv.is_stk(c)
    assert len(nlv.cfg_feature_list(c)) == 70   # the NLv verifier columns
    assert c["yhat"].endswith("mdtest_nl/yhat_v.parquet") and c["vens_weights"].endswith("nlv_weights")
    assert cfg["refiner"]["best_match_planes"] == ["swinl1536", "rfdetrseg"]   # NNRP planes and S15F features as configs/stk.yaml
    mem = cfg["refiner"]["members"]
    assert [m["tag"] for m in mem] == TAGS and all(m["full"] for m in mem)
    assert [m.get("seed", -1) for m in mem] == [-1, 11, 12] * 2
    rm = nlv.refiner_members(cfg, "/w")
    for m, r in zip(mem, rm):
        assert r["full"] and len(set(r["checkpoints"])) == 1 and str(r["checkpoints"][0]) == f"/w/fullfit/refiner_{m['tag']}_f-1.pt"
        assert (r["feats"] is None) == ("S15F" not in m["tag"])


def test_full_flag_must_match_the_checkpoint_pattern():
    cfg = config.load(CFG_PATH)
    cfg["refiner"]["members"] = [dict(tag="x", full=True, checkpoints="r_f{k}.pt")]
    with pytest.raises(AssertionError):
        nlv.refiner_members(cfg)


def test_stage_residual_member_sets_seed_and_features(monkeypatch, tmp_path):
    cfg = config.load(CFG_PATH)
    cfg["paths"].out = tmp_path
    meta = pd.DataFrame(dict(fold=[0, 1, 2, 3, 4, 0], source="v1_b04", cluster_n_src=2, area=100, teacher_area=100,
                             maxiou_seed=0.6, maxiou_ref=0.6))
    calls = []
    monkeypatch.setattr(stages, "_residual_meta", lambda cfg: meta)
    monkeypatch.setattr(stages, "_load_rows", lambda cfg, idx, extra=True: (torch.zeros(len(idx), 7, 8, 8, dtype=torch.uint8), torch.zeros(len(idx), 8, 8, dtype=torch.uint8)))
    import filament_seg.s15f as s15f
    monkeypatch.setattr(s15f, "load_feats", lambda path, rows=None, lazy=False: np.zeros((len(rows), 16, 2, 2), np.uint8))

    def fake_train(X, T, S2, w, c, *a, FT=None, seed=-1, **k):
        calls.append(dict(n=len(X), feats=FT is not None, seed=seed))
        return torch.nn.Linear(1, 1)
    monkeypatch.setattr(refiner, "train_residual", fake_train)
    stages.stage_residual(cfg, -1, member="FFNNRPx11")
    stages.stage_residual(cfg, -1, member="FFS15Fo")
    assert calls == [dict(n=6, feats=False, seed=11), dict(n=6, feats=True, seed=-1)]
    assert (tmp_path / "refiner_FFNNRPx11_f-1.pt").exists() and (tmp_path / "refiner_FFS15Fo_f-1.pt").exists()
