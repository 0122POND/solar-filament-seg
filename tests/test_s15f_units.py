"""Unit tests of the S15F building blocks, the refiner member ensemble and config includes (synthetic data, CPU, no weights)."""
import numpy as np
import pytest
import torch

from filament_seg import config, refiner, s15f


def test_config_include_and_null_removes(tmp_path):
    (tmp_path / "a.yaml").write_text("x: {p: 1, q: 2}\ny: 3\n")
    (tmp_path / "b.yaml").write_text("include: [a.yaml]\nx: {q: 5, r: 6}\ny: null\n")
    c = config.read_yaml(tmp_path / "b.yaml")
    assert c == {"x": {"p": 1, "q": 5, "r": 6}}


def test_final_configs_resolve():
    stk = config.load(config.REPO / "configs" / "stk.yaml")
    assert stk["refiner"]["tag"] == "S15F" and stk["refiner"]["members"][0]["test_feats"] and "s15f" in stk
    nnrp = config.load(config.REPO / "configs" / "stk_nnrp.yaml")
    assert nnrp["refiner"]["tag"] == "NNRP" and "feat_crops" not in nnrp["refiner"]
    assert "NNRP" in nnrp["nlv"]["refiner_rles"] and "S15F" in stk["nlv"]["refiner_rles"]


def _model(feat=True, seed=0):
    torch.manual_seed(seed)
    m = refiner.ResidualRefiner(1.0, pretrained=False, in_channels=9)
    return m.add_feat(16).eval() if feat else m.eval()


def test_zero_fproj_is_the_plane_model_and_keeps_the_rng():
    pytest.importorskip("segmentation_models_pytorch")   # GPU path only (requirements.txt); not in requirements-cpu.txt
    torch.manual_seed(1); a = refiner.ResidualRefiner(1.0, pretrained=False, in_channels=9); r1 = torch.rand(1)
    torch.manual_seed(1); b = refiner.ResidualRefiner(1.0, pretrained=False, in_channels=9).add_feat(16); r2 = torch.rand(1)
    assert torch.equal(r1, r2)                       # fproj is created under fork_rng
    b.load_state_dict({**a.state_dict(), **{k: v for k, v in b.state_dict().items() if k.startswith("fproj")}})
    x = torch.rand(2, 9, 64, 64); seed = (torch.rand(2, 1, 64, 64) > 0.5).float(); f = torch.randn(2, 16, 16, 16)
    with torch.no_grad():
        assert torch.allclose(a.eval()(x, seed), b.eval()(x, seed, f=f), atol=1e-6)   # zero-initialised projection


def test_predict_with_features_lazy_equals_in_ram_and_tta_flips_features():
    pytest.importorskip("segmentation_models_pytorch")   # GPU path only (requirements.txt); not in requirements-cpu.txt
    m = _model()
    with torch.no_grad():   # the segmentation head starts at 0 (output = seed); give it and fproj weights
        torch.nn.init.normal_(m.fproj.weight, std=0.5); torch.nn.init.normal_(m.net.segmentation_head[0].weight, std=0.5)
    X = torch.randint(0, 256, (3, 7, 64, 64), dtype=torch.uint8); S2 = torch.zeros(3, 2)
    FT = np.random.default_rng(0).integers(0, 256, (3, 16, 16, 16)).astype(np.uint8)
    dev = torch.device("cpu")
    a = refiner.predict_residual(m, X, S2, dev, bs=2, FT=FT)
    b = refiner.predict_residual(m, X, S2, dev, bs=2, FT=s15f.RowView(FT, np.arange(3)))
    c = refiner.predict_residual(m, X, S2, dev, bs=2, FT=np.full_like(FT, 128))
    assert np.array_equal(a, b) and np.abs(a - c).max() > 1e-4


def test_ensemble_members_equal_weights(tmp_path):
    pytest.importorskip("segmentation_models_pytorch")   # GPU path only (requirements.txt); not in requirements-cpu.txt
    for j, s in enumerate((0, 1)):
        for k in range(2):
            torch.save({"state": _model(feat=j == 1, seed=10 * s + k).state_dict()}, tmp_path / f"m{j}_f{k}.pt")
    X = np.random.default_rng(1).integers(0, 256, (2, 4, 64, 64)).astype(np.uint8)
    planes = lambda k: np.full((2, 3, 64, 64), 40 * k, np.uint8)  # noqa: E731
    S2 = np.zeros((2, 2), np.float32)
    FT = np.random.default_rng(2).integers(0, 256, (2, 16, 16, 16)).astype(np.uint8)
    m0 = dict(checkpoints=[tmp_path / f"m0_f{k}.pt" for k in range(2)])
    m1 = dict(checkpoints=[tmp_path / f"m1_f{k}.pt" for k in range(2)], feats=lambda k: FT)
    p0 = refiner.ensemble_members([m0], X, S2, planes)
    p1 = refiner.ensemble_members([m1], X, S2, planes)
    p01 = refiner.ensemble_members([m0, m1], X, S2, planes)
    assert np.allclose(p01, (p0 + p1) / 2, atol=1e-6)
    assert np.array_equal(p0, refiner.ensemble_residual_pmatch(m0["checkpoints"], X, S2, planes))
    with pytest.raises(AssertionError):   # a feature checkpoint without feature crops
        refiner.ensemble_members([dict(checkpoints=m1["checkpoints"])], X, S2, planes)


def test_fit_alignment_recovers_a_shared_space():
    rng = np.random.default_rng(0)
    T = rng.normal(size=(4000, 16)).astype(np.float32)
    F = np.stack([T @ rng.normal(size=(16, 256)).astype(np.float32) + 0.01 * rng.normal(size=(4000, 256)).astype(np.float32) for _ in range(5)])
    al = s15f.fit_alignment(F, log=lambda *a: None)
    assert al["W"].shape == (5, 256, 16) and al["b"].shape == (5, 16) and al["mu"].shape == (5, 256)
    assert al["R2"].min() > 0.95


def test_crop_features_constant_and_outside_frame():
    pytest.importorskip("torchvision")   # GPU path only (requirements.txt); not in requirements-cpu.txt
    z = torch.full((16, s15f.FS, s15f.FS), 0.5)
    X = s15f.crop_features(z, np.array([[100, 200, 256], [-4000, -4000, 256]]))
    assert X.shape == (2, 16, 64, 64) and (X[0] == 144).all() and (X[1] == 128).all()   # round(0.5 * 32 + 128); outside = 0 -> 128


def test_common_features_matches_formula():
    rng = np.random.default_rng(0)
    al = dict(W=rng.normal(size=(5, 256, 16)).astype(np.float32), b=rng.normal(size=(5, 16)).astype(np.float32),
              mu=rng.normal(size=(5, 256)).astype(np.float32), sd=np.abs(rng.normal(size=(5, 256))).astype(np.float32) + 0.5)
    mf = torch.from_numpy(rng.normal(size=(256, 4, 4)).astype(np.float32))
    z = s15f.common_features(mf, al, 2).numpy()
    ref = np.einsum("chw,cd->dhw", (mf.half().float().numpy() - al["mu"][2][:, None, None]) / al["sd"][2][:, None, None], al["W"][2]) + al["b"][2][:, None, None]
    assert np.allclose(z, ref, rtol=2e-3, atol=2e-3)   # fp16 rounding of z
