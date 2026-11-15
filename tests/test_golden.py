"""Numerical equivalence with the artifacts of a submitted run (default: the champion V2bSC yorig, public LB 0.39).

Skipped unless FILAMENT_SEG_GOLDEN points at the development artifact folder (the files listed in GOLDEN)
and SOLAR_DATA at the competition data. Checks, on the first GOLDEN_N test frames that have candidates:
  1. candidate union + features from the stored U-Net++ / YOLO outputs,
  2. the 5-fold residual refiner (CPU, flip TTA x4) against the stored probabilities, and the pasted-back
     masks against the stored refined masks (exact),
  3. verifier scores (on the original or the refined features, as configured),
  4. the decoded masks against the submitted CSV rows (byte-identical RLE strings).
FILAMENT_SEG_GOLDEN_OOF=1 additionally re-scores the 707-image OOF (GOLDEN_OOF_PQ, default 0.4473 at tau 0.25).
The earlier V2bS submission is checked with GOLDEN_TAG=V2bS GOLDEN_FEATURES=refined GOLDEN_OOF_PQ=0.4488, the A2 one with
SOLAR_CONFIG=configs/a2_refined.yaml GOLDEN_OOF_PQ=0.4439 (its test probabilities were not stored, so check 2 is skipped).
"""
import os
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from filament_seg import candidates, config, features, io, refiner, stages, verifier

GOLD = Path(os.environ.get("FILAMENT_SEG_GOLDEN", "/nonexistent"))
N = int(os.environ.get("GOLDEN_N", "3"))
pytestmark = pytest.mark.skipif(not GOLD.exists(), reason="set FILAMENT_SEG_GOLDEN to the development artifacts")
CFG = config.load(os.environ.get("SOLAR_CONFIG", config.DEFAULT_CONFIG))
TAG = os.environ.get("GOLDEN_TAG", CFG["refiner"]["tag"])  # refiner run to compare with
FEATURES = os.environ.get("GOLDEN_FEATURES", CFG["verifier"]["features"])  # original (yorig) or refined (ynew)
SUBMITTED = f"sub_refiner_{TAG}_union_{'yorig' if FEATURES == 'original' else 'ynew'}.csv"
OOF_PQ = float(os.environ.get("GOLDEN_OOF_PQ", "0.4473"))


@pytest.fixture(scope="module")
def gold():
    cfg = CFG
    names = [str(n) for n in np.load(GOLD / "test_names.npy")]
    cands = pd.read_parquet(GOLD / "verifier_test_cands_v1.parquet")
    picked = [n for n in names if (cands.image == n).any()][:N]
    probs = np.load(GOLD / "test_v1_prob.npy", mmap_mode="r")
    return dict(
        cfg=cfg, names=picked,
        prob={n: np.asarray(probs[names.index(n)]) for n in picked},
        yolo=pickle.load(open(GOLD / "inst_yolo_a_test_preds.pkl", "rb")),
        cands=cands.reset_index(drop=True),
        cand_rles=pickle.load(open(GOLD / "verifier_test_rles_v1.pkl", "rb")),
        ref_prob=np.load(GOLD / f"refiner_{TAG}_test_prob.npy", mmap_mode="r") if (GOLD / f"refiner_{TAG}_test_prob.npy").exists() else None,
        ref_rles=pickle.load(open(GOLD / f"refiner_{TAG}_test_rles.pkl", "rb")),
        ref_cands=pd.read_parquet(GOLD / f"refiner_{TAG}_test_cands.parquet"),
        gbdt=joblib.load(GOLD / ("verifier_gbdt_full.joblib" if FEATURES == "original" else f"verifier_gbdt_refined_{TAG}_full.joblib")),
    )


def _raw(cfg, name):
    f = cfg["paths"].test_images / f"{name}.jpeg"
    return io.read_gray(f if f.exists() else cfg["paths"].test_images / f"{name}.jpg")


def test_candidates_and_features(gold):
    c = gold["cfg"]["candidates"]
    for n in gold["names"]:
        p2048 = candidates.upsample_prob(gold["prob"][n])
        cl = candidates.collect(p2048, gold["yolo"].get(n, []), tuple(c["bins"]), c["min_area"], c["yolo_conf_min"])
        rows, rles, _ = features.featurize(n, p2048, _raw(gold["cfg"], n), cl, [])
        ref = gold["cands"][gold["cands"].image == n]
        assert len(rows) == len(ref)
        got = pd.DataFrame(rows)[features.FEATS_ALL].to_numpy(float)
        assert np.allclose(got, ref[features.FEATS_ALL].to_numpy(float), rtol=1e-6, atol=1e-6)
        assert [r["counts"] for r in rles] == [r["counts"] for r in gold["cand_rles"][n]]


def test_refiner_crops_and_probabilities(gold):
    if gold["ref_prob"] is None:
        pytest.skip(f"no stored test probabilities for {TAG}")
    cfg, c = gold["cfg"], gold["cfg"]["refiner"]
    df = gold["cands"]
    for n in gold["names"][:1]:
        g = df[df.image == n]
        R = gold["cand_rles"][n]
        X, boxes, _, _ = refiner.make_crops(_raw(cfg, n), gold["prob"][n], [R[i] for i in g.idx], g.source.tolist(),
                                            list(zip(g.x, g.y0, g.w, g.h)), c["crop_size"], c["crop_scale"], c["crop_min_side"])
        is_y = (g.source == "yolo").values
        S2 = np.stack([is_y, is_y & (g.cluster_n_src.values == 1)], 1).astype(np.float32)
        cks = [GOLD / f"refiner_{TAG}_f{k}.pt" for k in range(5)]
        if c.get("kind", "residual") == "stage1":
            prob = refiner.ensemble_stage1(cks, X, "cpu")
        else:
            prob = refiner.ensemble_residual(cks, X, S2, c["alpha"], "cpu")
        stored = np.asarray(gold["ref_prob"][g.index.values], dtype=np.float32)
        assert np.abs(prob - stored).max() < 2e-3  # stored in fp16
        # the submitted masks were pasted from the float32 probabilities: they must match exactly
        for k, idx in enumerate(g.idx):
            m, st = refiner.paste_back(prob[k], *map(int, boxes[k]), io.decode(R[idx]))
            rle = io.encode(m) if st == "ok" else R[idx]
            assert rle["counts"] == gold["ref_rles"][n][int(idx)]["counts"], (n, idx)


def test_paste_features_scores_and_decoding(gold):
    cfg, d = gold["cfg"], gold["cfg"]["decode"]
    df = gold["cands"]
    sub = pd.read_csv(GOLD / SUBMITTED) if (GOLD / SUBMITTED).exists() else None
    for n in gold["names"]:
        g = df[df.image == n]
        R = gold["cand_rles"][n]
        stored = [gold["ref_rles"][n][int(i)] for i in g.idx]
        if gold["ref_prob"] is not None:
            _, boxes, _, _ = refiner.make_crops(_raw(cfg, n), gold["prob"][n], [R[i] for i in g.idx], g.source.tolist(),
                                                list(zip(g.x, g.y0, g.w, g.h)))
            refined = []
            for k, (gi, idx) in enumerate(zip(g.index.values, g.idx)):
                prob = np.asarray(gold["ref_prob"][gi], dtype=np.float32)
                m, st = refiner.paste_back(prob, *map(int, boxes[k]), io.decode(R[idx]))
                refined.append(io.encode(m) if st == "ok" else R[idx])
            # the stored probabilities are fp16-rounded, so a few boundary pixels may flip at the 0.5 threshold
            ious = [float(np.asarray(io.mask_utils.iou([a], [b], [0]))[0, 0]) for a, b in zip(refined, stored)]
            assert min(ious) > 0.95 and np.mean(ious) > 0.995, (n, min(ious), np.mean(ious))
        # verifier scores and decoding from the stored refined masks
        if FEATURES == "original":
            yhat = verifier.predict(gold["gbdt"], g)
            col = "yhat"
        else:
            p2048 = candidates.upsample_prob(gold["prob"][n])
            cl = candidates.recollect(p2048, [(s, c, r) for s, c, r in zip(g.source, g.conf, stored)])
            rows, _, _ = features.featurize(n, p2048, _raw(cfg, n), cl, [])
            yhat = verifier.predict(gold["gbdt"], pd.DataFrame(rows))
            col = "yhat_v2" if "yhat_v2" in gold["ref_cands"].columns else "yhat_new"
        ref_yhat = gold["ref_cands"].set_index(["image", "idx"]).loc[[(n, int(i)) for i in g.idx], col].to_numpy()
        assert np.allclose(yhat, ref_yhat, atol=1e-9)
        pred = verifier.decode_image(stored, yhat.tolist(), d["tau"], d["nms_iou"], d["keep_frac"])
        if sub is not None:
            rows_sub = sub[sub.filament_id.str.rsplit("_", n=1).str[0] == n].segmentation_rle.tolist()
            assert [r["counts"].decode("ascii") for r in pred] == rows_sub


@pytest.mark.skipif(os.environ.get("FILAMENT_SEG_GOLDEN_OOF") != "1", reason="slow: set FILAMENT_SEG_GOLDEN_OOF=1")
def test_oof_score_reproduces():
    cfg = config.load(os.environ.get("SOLAR_CONFIG", config.DEFAULT_CONFIG))
    cfg["paths"].out = GOLD
    cfg["refiner"]["tag"] = TAG
    cfg["verifier"]["features"] = FEATURES
    r = stages.stage_evaluate(cfg, [0.25], workers=int(os.environ.get("GOLDEN_WORKERS", "1")))[0.25]
    assert abs(r["PQ"] - OOF_PQ) < 5e-5, r


def test_semantic_probabilities(gold):
    torch = pytest.importorskip("torch")
    from filament_seg import preprocess, semantic
    torch.set_num_threads(1)
    n = gold["names"][0]
    model = semantic.build_model("resnet34", pretrained=False)
    model.load_state_dict(torch.load(GOLD / "seg_v1.pt", map_location="cpu"))
    p = semantic.predict(model, preprocess.make_input(_raw(gold["cfg"], n))[None], torch.device("cpu"))[0]
    assert np.abs(p - gold["prob"][n].astype(np.float32)).max() < 2e-3  # stored in fp16


def test_yolo_candidates(gold):
    pytest.importorskip("ultralytics")
    from filament_seg import yolo
    w = GOLD / "yolo" / "a_m1536" / "weights" / "best.pt"  # hold-out YOLO11s-seg run (legacy run name)
    n = gold["names"][0]
    got = yolo.predict(w, [gold["cfg"]["paths"].test_images / f"{n}.jpeg"], device="cpu")[n]
    ref = gold["yolo"][n]
    assert len(got) == len(ref)
    assert np.allclose([c for c, _ in got], [c for c, _ in ref], atol=1e-3)
    ious = [float(np.asarray(io.mask_utils.iou([a], [b], [0]))[0, 0]) for (_, a), (_, b) in zip(got, ref)]
    assert np.mean(ious) > 0.99
