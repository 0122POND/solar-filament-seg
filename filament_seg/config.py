"""Experiment configuration: one YAML file holds every hyper-parameter of the pipeline.

Paths may reference environment variables (``${SOLAR_DATA}``); unset variables fall back to the
defaults below so that ``data/`` and ``out/`` next to the repository work out of the box.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "configs" / "champion.yaml"


@dataclass
class Paths:
    data: Path   # MAGFiLO_1.0_Kaggle_2026/ (train/, test/)
    out: Path    # every artifact (probabilities, candidates, checkpoints, CSV)

    @property
    def train_json(self) -> Path:
        return self.data / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

    @property
    def train_images(self) -> Path:
        return self.data / "train" / "train_images"

    @property
    def test_images(self) -> Path:
        return self.data / "test" / "test_images"


def merge(base: dict, over: dict) -> dict:
    """Deep merge: dicts are merged key by key, everything else in `over` replaces `base`; a null value removes the key."""
    out = dict(base)
    for k, v in over.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = v
    return out


def read_yaml(path: Path) -> dict:
    """YAML with optional ``include: [other.yaml, ...]`` (paths relative to this file): the included files are read first
    (recursively, in order) and this file is deep-merged on top (configs/stk.yaml includes configs/s15f.yaml;
    configs/stk_nnrp.yaml includes configs/stk.yaml and switches the refiner back to NNRP)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    inc = cfg.pop("include", None) or []
    base = {}
    for f in [inc] if isinstance(inc, str) else inc:
        base = merge(base, read_yaml(path.parent / f))
    return merge(base, cfg)


def load(path: Path = DEFAULT_CONFIG) -> dict:
    """Read the YAML config (with includes) and resolve ``paths`` into a Paths object under key ``"paths"``."""
    cfg = read_yaml(path)
    os.environ.setdefault("SOLAR_DATA", str(REPO / "data" / "MAGFiLO_1.0_Kaggle_2026"))
    os.environ.setdefault("SOLAR_OUT", str(REPO / "out"))
    p = cfg["paths"]
    cfg["paths"] = Paths(data=Path(os.path.expandvars(p["data"])), out=Path(os.path.expandvars(p["out"])))
    return cfg
