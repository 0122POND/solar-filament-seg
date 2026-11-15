"""scripts/export_weights.py on synthetic nnU-Net checkpoints (no real weights): optimizer state dropped, network weights and
the predictor's metadata kept, trainer-folder files copied; the manifest lists every component of NLv + NNR."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("export_weights", REPO / "scripts" / "export_weights.py")
EW = importlib.util.module_from_spec(spec); spec.loader.exec_module(EW)


def _trainer_dir(tmp_path):
    d = tmp_path / "nnUNetTrainer__nnUNetResEncUNetLPlans__2d"; (d / "fold_0").mkdir(parents=True)
    for f in EW.FOLDER_FILES:
        (d / f).write_text(json.dumps({"name": f}))
    return d


def test_strip_pth(tmp_path):
    d = _trainer_dir(tmp_path)
    net = torch.nn.Conv2d(1, 2, 3)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9); net(torch.zeros(1, 1, 4, 4)).sum().backward(); opt.step()
    ck = dict(network_weights=net.state_dict(), optimizer_state=opt.state_dict(), grad_scaler_state=None, logging={"loss": [1.0] * 1000},
              _best_ema=0.5, current_epoch=1000, init_args={"configuration": "2d"}, trainer_name="nnUNetTrainer", inference_allowed_mirroring_axes=(0, 1))
    torch.save(ck, d / "fold_0" / "checkpoint_final.pth")
    dst = tmp_path / "out" / d.name / "fold_0" / "checkpoint_final.pth"
    r = EW.strip_nnunet(d / "fold_0" / "checkpoint_final.pth", dst)
    out = torch.load(dst, map_location="cpu", weights_only=False)
    assert set(out) == set(EW.KEEP_KEYS) and "optimizer_state" in r["dropped"]
    assert all(torch.equal(out["network_weights"][k], v) for k, v in net.state_dict().items())
    assert out["trainer_name"] == "nnUNetTrainer" and out["init_args"] == {"configuration": "2d"}
    assert all((dst.parent.parent / f).exists() for f in EW.FOLDER_FILES)
    assert r["bytes_out"] < r["bytes_in"]


def test_strip_safetensors(tmp_path):
    pytest.importorskip("safetensors")   # GPU path only (requirements.txt); not in requirements-cpu.txt
    from safetensors.torch import load_file, save_file
    d = _trainer_dir(tmp_path)
    w = {"network_weights.conv.weight": torch.randn(2, 1, 3, 3), "optimizer_state.state.0.momentum_buffer": torch.randn(2, 1, 3, 3)}
    save_file(w, str(d / "fold_0" / "checkpoint_final.safetensors"), metadata={"trainer_name": "nnUNetTrainer"})
    dst = tmp_path / "out" / d.name / "fold_0" / "checkpoint_final.safetensors"
    EW.strip_nnunet(d / "fold_0" / "checkpoint_final.safetensors", dst)
    out = load_file(str(dst))
    assert list(out) == ["network_weights.conv.weight"] and torch.equal(out["network_weights.conv.weight"], w["network_weights.conv.weight"])
    from safetensors import safe_open
    with safe_open(str(dst), framework="pt") as f:
        assert f.metadata() == {"trainer_name": "nnUNetTrainer"}


def test_manifest_components(tmp_path):
    rows = EW.manifest(tmp_path / "no_root", REPO, tmp_path / "m.tsv")
    comps = " ".join(r["component"] for r in rows)
    for c in ("nnU-Net", "Swin-L", "R50 seed 0", "R50 seed 1", "R50 seed 2", "own2 map U-Net r34", "own2 map U-Net eff", "U-Net++",
              "NNR refiner", "V-ENS member gbdt", "V-ENS member cat", "V-ENS member xgb", "V-ENS member et"):
        assert c in comps, c
    assert sum("NNR refiner" in r["component"] for r in rows) == 5
