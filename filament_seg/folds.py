"""Image-level splits. All annotator records of a physical image always fall into the same split.

- ``split_f{k}.json`` (k = 0..4): 5-fold cross-validation over the 707 physical images (seed 42). Every
  out-of-fold (OOF) number in the report comes from these folds.
- ``split_v1.json``: one 80/20 hold-out stratified by observatory (the last two letters of the stem).
  The full-disk U-Net++ and the YOLO model used at test time were trained on its train part.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def make_kfold(stems: list, out: Path, n_folds: int = 5, seed: int = 42) -> list:
    stems = sorted(stems)
    perm = np.random.default_rng(seed).permutation(len(stems))
    paths = []
    for k, idx in enumerate(np.array_split(perm, n_folds)):
        val = sorted(stems[i] for i in idx)
        train = sorted(set(stems) - set(val))
        paths.append(out / f"split_f{k}.json")
        json.dump({"train": train, "val": val, "seed": seed, "fold": k}, open(paths[-1], "w"))
    return paths


def make_holdout(stems: list, out: Path, val_ratio: float = 0.2, seed: int = 42) -> Path:
    by_obs = defaultdict(list)
    for s in sorted(stems):
        by_obs[s[-2:]].append(s)
    rng = random.Random(seed)
    train, val = [], []
    for obs in sorted(by_obs):
        names = by_obs[obs][:]
        rng.shuffle(names)
        k = max(1, round(len(names) * val_ratio))
        val += names[:k]
        train += names[k:]
    path = out / "split_v1.json"
    json.dump({"train": sorted(train), "val": sorted(val), "seed": seed}, open(path, "w"), indent=1)
    return path


def load_split(out: Path, name: str) -> dict:
    return json.load(open(out / name))


def fold_of(out: Path, n_folds: int = 5) -> dict:
    """{stem: k} where k is the fold whose validation part holds the image."""
    return {s: k for k in range(n_folds) for s in load_split(out, f"split_f{k}.json")["val"]}
