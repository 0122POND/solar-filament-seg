"""Minimal variant NLv + NNR (configs/nlv.yaml): the CPU stages reproduce the submitted CSV sub_mdNLv_NN_t028.csv (md5 9eb43f91...) byte for byte.

Inputs are the cached model outputs of the submitted run, read from the development folder (configs/nlv.yaml, section
`nlv`; SOLAR_ROOT, default = the folder above release/). Skipped when they are missing.
  1. decode only: stored verifier scores + NNR masks -> CSV (about 15 s)
  2. full CPU path: V-ENS fitted on the OOF table, five test tables (votes + own2 map features of each fold model),
     yhat identical to the stored one (max |diff| 0), CSV identical (about 4 min; GOLDEN_NLV_FULL=0 skips it)
  3. GOLDEN_NLV_RECOMPUTE=1: also recompute the training votes and map features from the OOF pickles / maps (about 6 min)
  4. building blocks on a few OOF frames: nnU-Net + Swin-L candidates, votes, map features equal the stored tables.
The final-cpu runs go through a subprocess: CatBoost / XGBoost segfault on macOS in a process that has imported torch.
"""
import hashlib
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from filament_seg import config, nlv

REPO = Path(__file__).resolve().parents[1]
CFG_PATH = REPO / "configs" / "nlv.yaml"
CFG = config.load(CFG_PATH)
C = nlv.resolve(CFG)
SUBMITTED_MD5 = "9eb43f91f77263f7a38e3a8c0dce8ba2"
NEEDED = ("train_cands", "train_mapfeats", "train_votes", "test_cands", "test_rles", "test_names", "refiner_rles", "yhat", "submitted_csv")
pytestmark = pytest.mark.skipif(not all(Path(C[k]).exists() for k in NEEDED), reason="cached NLv model outputs not found (set SOLAR_ROOT)")


def _final_cpu(tmp_path, *flags):
    out = tmp_path / "nlv.csv"
    r = subprocess.run([sys.executable, "-m", "filament_seg", "--config", str(CFG_PATH), "final-cpu", "--out", str(out), *flags],
                       cwd=REPO, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    assert r.returncode == 0, r.stderr[-3000:]
    return out, json.loads(r.stdout[r.stdout.index("{"):])


def test_submitted_csv_is_the_reference():
    assert hashlib.md5(Path(C["submitted_csv"]).read_bytes()).hexdigest() == SUBMITTED_MD5


def test_decode_byte_identical(tmp_path):
    out, res = _final_cpu(tmp_path, "--yhat-cached")
    assert res["md5"] == SUBMITTED_MD5 and res["rows"] == 1114
    assert out.read_bytes() == Path(C["submitted_csv"]).read_bytes()


@pytest.mark.skipif(os.environ.get("GOLDEN_NLV_FULL", "1") == "0", reason="GOLDEN_NLV_FULL=0")
def test_vens_and_decode_byte_identical(tmp_path):
    out, res = _final_cpu(tmp_path)
    assert res["yhat_max_abs_diff_vs_submitted"] == 0.0
    assert out.read_bytes() == Path(C["submitted_csv"]).read_bytes()


@pytest.mark.skipif(os.environ.get("GOLDEN_NLV_RECOMPUTE") != "1", reason="set GOLDEN_NLV_RECOMPUTE=1")
def test_recomputed_training_table_byte_identical(tmp_path):
    out, res = _final_cpu(tmp_path, "--recompute-votes", "--recompute-maps")
    assert res["yhat_max_abs_diff_vs_submitted"] == 0.0
    assert out.read_bytes() == Path(C["submitted_csv"]).read_bytes()


# ----------------------------------------------------------------------------- building blocks on OOF frames

@pytest.fixture(scope="module")
def oof():
    tr = pd.read_parquet(C["train_cands"])
    names = sorted(tr.image.unique())
    pick = [names[i] for i in (0, 1, 2, len(names) // 2, len(names) - 1)]
    return dict(tr=tr[tr.image.isin(pick)], rles=pickle.load(open(C["train_rles"], "rb")), names=pick)


def test_votes_match_stored(oof):
    from filament_seg import votes
    P = {k: {} for k in votes.VOTERS}
    for k in votes.VOTERS:
        for f in range(5):
            d = pickle.load(open(os.path.expandvars(C["oof_voters"][k]).format(k=f), "rb"))
            P[k].update({n: d[n] for n in oof["names"] if n in d})
    V = votes.oof_votes(oof["tr"][["image", "idx"]], oof["rles"], P)
    ref = pd.read_parquet(C["train_votes"]).merge(V, on=["image", "idx"], suffixes=("_ref", ""))
    assert len(ref) == len(oof["tr"])
    for c in votes.VFEATS:
        assert (ref[c + "_ref"].values == ref[c].values).all(), c


def test_map_features_match_stored(oof):
    from filament_seg import mapfeats
    F = pd.DataFrame([r for n, g in oof["tr"].groupby("image") for r in
                      mapfeats.image_features(Path(C["train_maps"]), n, oof["rles"][n], g.idx.astype(int).tolist(), fast=True)])
    ref = pickle.load(open(C["train_mapfeats"], "rb"))["F"].merge(F, on=["image", "idx"], suffixes=("_ref", ""))
    assert len(ref) == len(oof["tr"])
    for c in mapfeats.DFEATS:
        assert (ref[c + "_ref"].values == ref[c].values).all(), c


def test_candidates_match_stored(oof):
    """nnU-Net components (conf from the U-Net++ OOF map) U Swin-L instances = the OOF candidate table (masks, order, source, conf)."""
    from pycocotools import mask as mU
    from filament_seg import candidates, nnunet_cands
    root = Path(os.environ["SOLAR_ROOT"])
    probs, vn = root / "outputs" / "seg_v1oof_valprob.npy", root / "outputs" / "seg_v1oof.json"
    if not (probs.exists() and vn.exists() and Path(C["train_nnmaps"]).exists()):
        pytest.skip("U-Net++ OOF probabilities / nnU-Net OOF maps not found")
    vn = json.load(open(vn))["val_names"]; P = np.load(probs, mmap_mode="r")
    SL = {}
    for k in range(5):
        d = pickle.load(open(C["train_swinl"].format(k=k), "rb")); SL.update({n: d[n] for n in oof["names"] if n in d})
    for n in oof["names"]:
        cs = nnunet_cands.collect(candidates.upsample_prob(np.asarray(P[vn.index(n)])), nnunet_cands.load_map(C["train_nnmaps"], n),
                                  [(float(s), r) for s, r in SL.get(n, [])])
        g = oof["tr"][oof["tr"].image == n].sort_values("idx")
        assert len(cs) == len(g)
        assert [c[0] for c in cs] == g.source.tolist()
        assert np.array_equal(np.array([c[1] for c in cs]), g.conf.values)
        assert all(mU.encode(np.asfortranarray(c[2].astype(np.uint8)))["counts"] == oof["rles"][n][int(i)]["counts"] for c, i in zip(cs, g.idx))


def test_oof_candidate_features_match_stored(oof):
    """Candidate union + all FEATS_ALL / targets of 2 OOF frames rebuilt from the training images = the stored OOF table."""
    import copy
    cfg = copy.deepcopy(CFG)
    if not cfg["paths"].train_json.exists():  # SOLAR_DATA unset: the data folder of the development tree
        cfg["paths"].data = Path(os.environ["SOLAR_ROOT"]) / "data" / "MAGFiLO_1.0_Kaggle_2026"
    if not (cfg["paths"].train_json.exists() and Path(C["train_v1_prob"]).exists()):
        pytest.skip("training images / U-Net++ OOF probabilities not found")
    names = oof["names"][:2]
    df, _ = nlv.oof_candidates(cfg, names)
    ref = oof["tr"][oof["tr"].image.isin(names)].sort_values(["image", "idx"]).reset_index(drop=True)
    df = df.sort_values(["image", "idx"]).reset_index(drop=True)
    assert len(df) == len(ref) and (df.source.values == ref.source.values).all()
    for c in [c for c in ref.columns if c in df.columns and pd.api.types.is_numeric_dtype(ref[c])]:
        assert np.array_equal(df[c].values.astype(float), ref[c].values.astype(float), equal_nan=True), c
