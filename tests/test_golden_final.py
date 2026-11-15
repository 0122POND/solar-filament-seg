"""NLv x full-data B6 refiners + C0 with refiner seeds none / 11 / 12 (configs/final_fullfit.yaml; the final model
configs/final_yellow.yaml is checked by tests/test_golden_yellow.py): the CPU stages reproduce sub_NLv_FF6C0_t025.csv
(md5 31e07084..., public LB 0.41) byte for byte.

Inputs are the cached model outputs of the run (section `nlv` of the config; SOLAR_ROOT). Skipped when missing.
  1. decode only: stored NLv scores + full-data B6 test masks (refine-test output, before C0), C0 applied here -> CSV (15 s)
  2. saved NLv V-ENS members (Hugging Face nlv/), test votes and map features rebuilt -> scores identical, CSV identical (3 min)
  3. oof-eval of the fold version (NLv OOF scores x B6 fold-refiner OOF masks, C0, tau 0.25): rec 0.4758 / simulated 0.4142
  4. GOLDEN_FINAL_FULL=1: V-ENS refitted on the OOF table -> CSV byte-identical (4 min)
The test masks themselves (refine-test with the six full-data refiners) were checked separately: repository member mean +
paste-back on the per-member probability sums of the development run gives the same pickle (README, "Final model").
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
CFG_PATH = REPO / "configs" / "final_fullfit.yaml"
CFG = config.load(CFG_PATH)
C = nlv.resolve(CFG)
FINAL_MD5 = "31e0708424a321eb19d996a2a2868dcf"
NEEDED = ("train_cands", "train_mapfeats", "train_votes", "test_cands", "test_rles", "test_names", "refiner_rles", "yhat", "submitted_csv")
pytestmark = pytest.mark.skipif(not all(Path(C[k]).exists() for k in NEEDED), reason="cached model outputs not found (set SOLAR_ROOT)")


def _run(tmp_path, stage, *flags):
    out = tmp_path / "final.csv"
    args = ["final-cpu", "--out", str(out)] if stage == "final-cpu" else [stage]
    r = subprocess.run([sys.executable, "-m", "filament_seg", "--config", str(CFG_PATH), *args, *flags],
                       cwd=REPO, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    assert r.returncode == 0, r.stderr[-3000:]
    return out, json.loads(r.stdout[r.stdout.index("{"):])


def test_final_csv_is_the_reference():
    assert hashlib.md5(Path(C["submitted_csv"]).read_bytes()).hexdigest() == FINAL_MD5


def test_decode_with_c0_byte_identical(tmp_path):
    out, res = _run(tmp_path, "final-cpu", "--yhat-cached")
    assert res["md5"] == FINAL_MD5 and res["rows"] == 1210 and res["c0_replaced"] == 950
    assert out.read_bytes() == Path(C["submitted_csv"]).read_bytes()


def test_saved_nlv_vens_byte_identical(tmp_path):
    if not (Path(C["vens_weights"]) / "verifier_nlv_et.joblib").exists():
        pytest.skip("saved NLv V-ENS weights not found")
    out, res = _run(tmp_path, "final-cpu", "--load-weights", C["vens_weights"])
    assert res["yhat_max_abs_diff_vs_submitted"] == 0.0
    assert out.read_bytes() == Path(C["submitted_csv"]).read_bytes()


def test_oof_eval_fold_version(tmp_path):
    if not all(Path(C["oof_refiner_rles"].format(k=k)).exists() for k in range(5)) or not Path(C["oof_yhat"]).exists():
        pytest.skip("OOF masks / scores not found")
    _, res = _run(tmp_path, "oof-eval")
    assert (res["tau"], res["frames"], res["rec_pq"], res["sim_pq"]) == (0.25, 707, 0.4758, 0.4142)


@pytest.mark.skipif(os.environ.get("GOLDEN_FINAL_FULL") != "1", reason="set GOLDEN_FINAL_FULL=1 (about 4 min)")
def test_vens_refit_and_decode_byte_identical(tmp_path):
    out, res = _run(tmp_path, "final-cpu")
    assert res["yhat_max_abs_diff_vs_submitted"] == 0.0
    assert out.read_bytes() == Path(C["submitted_csv"]).read_bytes()
