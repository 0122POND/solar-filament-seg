"""stage1 (`python -m filament_seg stage1 --fold k`) must write the OOF table that the residual refiners read
(`refiner_<stage1 tag>_f<k>_cands.parquet`, stages._residual_meta). Synthetic rows, CPU, no weights: the two GPU calls
(refiner.train_stage1 / predict_stage1) are replaced, everything else is the code of the stage."""
import builtins
import dis
import pickle
import types

import numpy as np
import pandas as pd
import pytest

from filament_seg import config, io, stages


def _codes(co):
    yield co
    for c in co.co_consts:
        if isinstance(c, types.CodeType):
            yield from _codes(c)


def test_no_function_reads_an_undefined_global():
    """Every global a function of the package reads is defined in its module (a builtin, an import or a module-level name)."""
    bad = []
    for path in sorted((config.REPO / "filament_seg").glob("*.py")):
        co = compile(path.read_text(), str(path), "exec")
        known = {i.argval.split(".")[0] for c in _codes(co) for i in dis.get_instructions(c)
                 if i.opname in ("STORE_NAME", "STORE_GLOBAL", "IMPORT_NAME")} | set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
        bad += [f"{path.name}:{i.positions.lineno} {i.argval} ({c.co_name})" for c in _codes(co) if c is not co
                for i in dis.get_instructions(c) if i.opname == "LOAD_GLOBAL" and i.argval not in known]
    assert not bad, bad


def test_stage1_writes_the_oof_table_of_its_tag(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    S = 256
    cfg = config.read_yaml(config.REPO / "configs" / "champion.yaml")
    cfg["paths"] = config.Paths(data=tmp_path, out=tmp_path)
    tag = cfg["refiner"]["stage1"]["tag"]
    sq = np.zeros((io.H, io.W), bool); sq[100:140, 100:140] = True      # seed, and the ground truth of both frames
    meta = pd.DataFrame(dict(image=list("aabbab"), idx=[0, 1, 0, 1, 2, 2], source="yolo", fold=[0, 1, 2, 3, 4, 0], w_hard=1.0, x0=56, y0=56, side=128))
    meta.to_parquet(tmp_path / stages.ART["crop_meta"])
    pickle.dump({n: {i: io.encode(sq) for i in range(3)} for n in "ab"}, open(tmp_path / stages.ART["cand_rles"], "wb"))
    pickle.dump({n: [[io.encode(sq)]] for n in "ab"}, open(tmp_path / stages.ART["gt_rles"], "wb"))
    prob = np.zeros((S, S), np.float32); prob[88:168, 88:168] = 1.0     # = the seed square in crop coordinates (x 2)
    monkeypatch.setattr(stages, "_load_rows", lambda cfg, idx, extra=True: (torch.zeros(len(idx), 4, S, S), torch.zeros(len(idx), S, S)))
    monkeypatch.setattr(stages.refiner, "train_stage1", lambda *a, **k: torch.nn.Conv2d(4, 1, 1))
    monkeypatch.setattr(stages.refiner, "predict_stage1", lambda model, X, dev, **k: np.repeat(prob[None], len(X), 0))
    for fold in range(5):
        stages.stage_stage1(cfg, fold)
        assert (tmp_path / f"refiner_{tag}_f{fold}.pt").exists()
        t = pd.read_parquet(tmp_path / f"refiner_{tag}_f{fold}_cands.parquet")
        assert len(t) == (meta.fold == fold).sum() and (t.status == "ok").all() and (t.maxiou_ref > 0.9).all()
        assert set(pickle.load(open(tmp_path / f"refiner_{tag}_f{fold}_rles.pkl", "rb"))) == set(meta.image[meta.fold == fold])
    m = stages._residual_meta(cfg)   # what `residual --fold k` / `--fold -1` reads
    assert len(m) == len(meta) and m.maxiou_ref.notna().all()
