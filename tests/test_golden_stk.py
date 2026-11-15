"""Alternative STK-avg5 + C0 + S15F (configs/stk.yaml): the CPU stages reproduce sub_STKavg5_C0_S15F_t028.csv (md5 a2089e62...) byte for byte;
the previous submission STK-avg5 + C0 + NNRP (configs/stk_nnrp.yaml, md5 25ac8b05...) as well (decode test).

Inputs are the cached model outputs of the submitted run (configs/stk.yaml, section `nlv`; SOLAR_ROOT). Skipped when missing.
  1. decode only: stored STK-avg5 scores + S15F (NNRP) masks, C0 applied here -> CSV (about 30 s each)
  2. g7_m2 test columns rebuilt from the 7 x 5 detector pickles equal the stored table (about 1.5 min)
  3. GOLDEN_STK_FULL=1: five V-ENS seeds refitted on the OOF table, test votes (4 voters) and map features rebuilt,
     scores equal the stored ones and the CSV is byte-identical (about 8 min)
  4. GOLDEN_STK_OOF=1: g7_m2 OOF columns rebuilt equal the training table (about 4 min)
The final-cpu runs go through a subprocess (CatBoost / XGBoost segfault on macOS next to torch).
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from filament_seg import config, evofeats, nlv

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "configs" / "stk.yaml"
CFG = config.load(CFG_PATH)
C = nlv.resolve(CFG)
SUBMITTED_MD5 = "a2089e625c58f5c7675dedd249a5d35a"
NNRP_CFG_PATH = REPO / "configs" / "stk_nnrp.yaml"
NNRP_MD5 = "25ac8b051aed6c86e552230edf6e63c1"
NEEDED = ("train_cands", "train_mapfeats", "train_votes", "test_cands", "test_rles", "test_names", "refiner_rles", "yhat", "submitted_csv")
pytestmark = pytest.mark.skipif(not all(Path(C[k]).exists() for k in NEEDED) or not Path(C["evo"]["train"]).exists(),
                                reason="cached STK model outputs not found (set SOLAR_ROOT)")


def _final_cpu(tmp_path, *flags, cfg_path=CFG_PATH):
    out = tmp_path / "stk.csv"
    r = subprocess.run([sys.executable, "-m", "filament_seg", "--config", str(cfg_path), "final-cpu", "--out", str(out), *flags],
                       cwd=REPO, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    assert r.returncode == 0, r.stderr[-3000:]
    return out, json.loads(r.stdout[r.stdout.index("{"):])


def _same_columns(a: pd.DataFrame, b: pd.DataFrame, tol=1e-12):
    m = b.merge(a, on=["image", "idx"], suffixes=("_ref", ""))
    assert len(m) == len(a) == len(b)
    for c in evofeats.COLS:
        assert (m[c + "_ref"].isna() == m[c].isna()).all(), c
        assert np.nanmax(np.abs(m[c + "_ref"] - m[c])) <= tol, c


def test_submitted_csv_is_the_reference():
    assert hashlib.md5(Path(C["submitted_csv"]).read_bytes()).hexdigest() == SUBMITTED_MD5


def test_decode_with_c0_byte_identical(tmp_path):
    out, res = _final_cpu(tmp_path, "--yhat-cached")
    assert res["md5"] == SUBMITTED_MD5 and res["rows"] == 1155 and res["c0_replaced"] == 1027
    assert out.read_bytes() == Path(C["submitted_csv"]).read_bytes()


def test_previous_submission_nnrp_decode_byte_identical(tmp_path):
    cn = nlv.resolve(config.load(NNRP_CFG_PATH))
    if not (Path(cn["refiner_rles"]).exists() and Path(cn["submitted_csv"]).exists()):
        pytest.skip("NNRP test masks not found")
    out, res = _final_cpu(tmp_path, "--yhat-cached", cfg_path=NNRP_CFG_PATH)
    assert res["md5"] == NNRP_MD5 and res["rows"] == 1155 and res["c0_replaced"] == 854
    assert out.read_bytes() == Path(cn["submitted_csv"]).read_bytes()


def test_saved_vens_weights_byte_identical(tmp_path):
    """The five saved V-ENS seeds (final-cpu --save-weights; Hugging Face final/vens/) give the submitted CSV (about 2 min)."""
    if not (Path(C["vens_weights"]) / "seed4" / "verifier_nlv_et.joblib").exists():
        pytest.skip("saved V-ENS weights not found")
    out, res = _final_cpu(tmp_path, "--load-weights", C["vens_weights"])
    assert res["yhat_max_abs_diff_vs_submitted"] == 0.0
    assert out.read_bytes() == Path(C["submitted_csv"]).read_bytes()


def test_evo_test_columns_identical():
    _same_columns(evofeats.test_features(C), pd.read_parquet(C["evo"]["test"]))


@pytest.mark.skipif(os.environ.get("GOLDEN_STK_FULL") != "1", reason="set GOLDEN_STK_FULL=1 (about 8 min)")
def test_five_seed_vens_and_decode_byte_identical(tmp_path):
    out, res = _final_cpu(tmp_path)
    assert res["yhat_max_abs_diff_vs_submitted"] == 0.0
    assert out.read_bytes() == Path(C["submitted_csv"]).read_bytes()


@pytest.mark.skipif(os.environ.get("GOLDEN_STK_OOF") != "1", reason="set GOLDEN_STK_OOF=1 (about 4 min)")
def test_evo_oof_columns_identical():
    _same_columns(evofeats.oof_features(C), pd.read_parquet(C["evo"]["train"]))
