"""Final model configs/final_yellow.yaml (NLv x full-data B6 with refiner seeds 16 / 17 / 18 + C0 + order bonus 0.10): the CPU stages
reproduce sub_NLv_FF6CC0_t025_ORDp10.csv (md5 540c8863..., public LB 0.41) byte for byte.

Inputs are the cached model outputs of the run (section `nlv` of the config; SOLAR_ROOT). Skipped when missing (except the config test).
  1. decode only: stored NLv scores + test masks of the six seed 16-18 members (before C0), C0 and the order bonus applied
     here -> CSV (under 1 min)
  2. the same with decode.order_bonus 0 -> sub_NLv_FF6CC0_t025.csv (md5 678c2482...), i.e. the bonus is the only difference
  3. saved NLv V-ENS members (Hugging Face nlv/), test votes and map features rebuilt -> CSV identical (3 min)
  4. oof-eval of the fold version (B6 fold refiners, seeds none / 11 / 12; the seed 16-18 members have no fold version)
     with the order bonus: rec 0.4759 / simulated 0.4144 (without the bonus 0.4758 / 0.4142)
The decode settings themselves (order_bonus, iom_max) are unit-tested on synthetic masks in tests/test_decode_knobs.py.
Not covered here: the six seed 16-18 checkpoints -> test masks (refine-test on the GPU; README).
final-cpu runs in a subprocess (CatBoost / XGBoost segfault on macOS next to torch).
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from filament_seg import config, nlv

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "configs" / "final_yellow.yaml"
CFG = config.load(CFG_PATH)
C = nlv.resolve(CFG)
YELLOW_MD5 = "540c886326cff3320dd341c0dedd2cd1"      # order_bonus 0.10 (the final submission)
BONUS_002_MD5 = "257bb97ad9cbe854007832b9725a3cff"   # order_bonus 0.02 (our best public score until Nov 14)
NO_BONUS_MD5 = "678c248214a5a61210c7fe1c6fd0e65b"    # order_bonus 0
NEEDED = ("test_cands", "test_rles", "refiner_rles", "yhat", "submitted_csv", "submitted_csv_bonus_002", "submitted_csv_no_bonus")
needs_cache = pytest.mark.skipif(not all(Path(C[k]).exists() for k in NEEDED), reason="cached model outputs not found (set SOLAR_ROOT)")


def _run(tmp_path, cfg_path, stage, *flags):
    out = tmp_path / "yellow.csv"
    args = ["final-cpu", "--out", str(out)] if stage == "final-cpu" else [stage]
    r = subprocess.run([sys.executable, "-m", "filament_seg", "--config", str(cfg_path), *args, *flags],
                       cwd=REPO, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    assert r.returncode == 0, r.stderr[-3000:]
    return out, json.loads(r.stdout[r.stdout.index("{"):])


@needs_cache
def test_reference_csvs():
    assert hashlib.md5(Path(C["submitted_csv"]).read_bytes()).hexdigest() == YELLOW_MD5
    assert hashlib.md5(Path(C["submitted_csv_bonus_002"]).read_bytes()).hexdigest() == BONUS_002_MD5
    assert hashlib.md5(Path(C["submitted_csv_no_bonus"]).read_bytes()).hexdigest() == NO_BONUS_MD5


def test_config_differs_from_final_fullfit_in_members_and_bonus_only():
    base = config.load(REPO / "configs" / "final_fullfit.yaml")
    assert CFG["decode"] == {**base["decode"], "order_bonus": 0.10} and nlv.decode_knobs(base) == nlv.DECODE_KNOBS
    mem = CFG["refiner"]["members"]
    assert [m["tag"] for m in mem] == [f"FF{t}x{s}" for t in ("NNRP", "S15F") for s in (16, 17, 18)]
    assert [m["seed"] for m in mem] == [16, 17, 18] * 2 and all(m["full"] for m in mem)
    strip = lambda m: {k: v for k, v in m.items() if k not in ("tag", "seed", "checkpoints")}  # noqa: E731
    assert [strip(m) for m in mem] == [strip(m) for m in base["refiner"]["members"]]
    assert {k: v for k, v in CFG["refiner"].items() if k != "members"} == {k: v for k, v in base["refiner"].items() if k != "members"}
    same = set(CFG) - {"decode", "refiner", "nlv", "paths"}
    assert all(CFG[k] == base[k] for k in same) and set(CFG) == set(base)
    diff = {k for k in set(CFG["nlv"]) | set(base["nlv"]) if CFG["nlv"].get(k) != base["nlv"].get(k)}
    assert diff == {"refiner_rles", "submitted_csv", "submitted_csv_bonus_002", "submitted_csv_no_bonus"}


@needs_cache
def test_decode_with_c0_and_order_bonus_byte_identical(tmp_path):
    out, res = _run(tmp_path, CFG_PATH, "final-cpu", "--yhat-cached")
    assert res["md5"] == YELLOW_MD5 and res["rows"] == 1209 and res["c0_replaced"] == 914
    assert out.read_bytes() == Path(C["submitted_csv"]).read_bytes()


@needs_cache
def test_order_bonus_002_is_the_previous_best(tmp_path):
    cfg2 = tmp_path / "bonus_002.yaml"
    cfg2.write_text(f"include: [{CFG_PATH}]\ndecode:\n  order_bonus: 0.02\n")
    out, res = _run(tmp_path, cfg2, "final-cpu", "--yhat-cached")
    assert res["md5"] == BONUS_002_MD5 and res["rows"] == 1209 and res["c0_replaced"] == 914
    assert out.read_bytes() == Path(C["submitted_csv_bonus_002"]).read_bytes()


@needs_cache
def test_order_bonus_zero_is_the_plain_decode(tmp_path):
    cfg0 = tmp_path / "no_bonus.yaml"
    cfg0.write_text(f"include: [{CFG_PATH}]\ndecode:\n  order_bonus: 0.0\n")
    out, res = _run(tmp_path, cfg0, "final-cpu", "--yhat-cached")
    assert res["md5"] == NO_BONUS_MD5 and res["rows"] == 1210 and res["c0_replaced"] == 914
    assert out.read_bytes() == Path(C["submitted_csv_no_bonus"]).read_bytes()


@needs_cache
def test_saved_nlv_vens_byte_identical(tmp_path):
    if not (Path(C["vens_weights"]) / "verifier_nlv_et.joblib").exists():
        pytest.skip("saved NLv V-ENS weights not found")
    out, res = _run(tmp_path, CFG_PATH, "final-cpu", "--load-weights", C["vens_weights"])
    assert res["yhat_max_abs_diff_vs_submitted"] == 0.0
    assert out.read_bytes() == Path(C["submitted_csv"]).read_bytes()


@needs_cache
def test_oof_eval_fold_version_with_order_bonus(tmp_path):
    need = [C["oof_refiner_rles"].format(k=k) for k in range(5)] + [C[k] for k in ("oof_yhat", "gt_rles", "train_cands", "train_rles")]
    if not all(Path(p).exists() for p in need):
        pytest.skip("OOF masks / scores / annotator records not found")
    _, res = _run(tmp_path, CFG_PATH, "oof-eval")   # without the bonus (configs/final_fullfit.yaml): 0.4758 / 0.4142
    assert (res["tau"], res["frames"], res["rec_pq"], res["sim_pq"]) == (0.25, 707, 0.4759, 0.4144)
