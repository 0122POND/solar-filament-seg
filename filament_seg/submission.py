"""Write and validate the submission CSV (filament_id, segmentation_rle)."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from .io import rle_from_str

ID_RE = re.compile(r"^(\d{14}[A-Z][a-z])_(\d+)$")


def write(predictions: dict, path: Path):
    """predictions: {stem: [RLE counts str]} in decreasing confidence; ids are <stem>_<k> from 1."""
    rows = [(f"{stem}_{k + 1}", c) for stem, cs in predictions.items() for k, c in enumerate(cs)]
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(path, index=False)


def validate(path: Path, test_stems: set) -> dict:
    """Checks the server enforces: ids, no quotes, every stem in the test set, zero overlap within an image."""
    text = Path(path).read_text()
    df = pd.read_csv(path)
    stems = df.filament_id.str.extract(ID_RE)[0]
    max_overlap = 0.0
    for _, g in df.groupby(stems):
        rles = [rle_from_str(c) for c in g.segmentation_rle]
        if len(rles) > 1:
            inter = np.asarray(mask_utils.iou(rles, rles, [1] * len(rles)))  # iscrowd=1 -> intersection / area
            np.fill_diagonal(inter, 0)
            max_overlap = max(max_overlap, float(inter.max()))
    return {"header_ok": list(df.columns) == ["filament_id", "segmentation_rle"],
            "no_quotes": '"' not in text and "'" not in text,
            "ids_ok": bool(stems.notna().all()) and not df.filament_id.duplicated().any(),
            "stems_in_test": set(stems.dropna()) <= set(test_stems),
            "no_overlap": max_overlap == 0.0,
            "rows": len(df), "images": int(stems.nunique())}
