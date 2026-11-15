"""MAGFiLO COCO annotations, annotator records, images and RLE helpers.

Every image is 2048 x 2048 grayscale. A physical image can carry several independent
annotator records ("<annotator>-<image_stem>"); the official metric scores each record separately.
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

H = W = 2048
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def encode(mask: np.ndarray) -> dict:
    """Binary mask (H, W) -> COCO RLE. pycocotools expects column-major (Fortran) order."""
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def encode_str(mask: np.ndarray) -> str:
    """Binary mask -> RLE counts string, the format of the submission column."""
    return encode(mask)["counts"].decode("ascii")


def decode(rle: dict) -> np.ndarray:
    """COCO RLE -> boolean mask (H, W)."""
    return mask_utils.decode(rle).astype(bool)


def decode_str(counts: str) -> np.ndarray:
    """RLE counts string -> binary mask (H, W)."""
    return mask_utils.decode(rle_from_str(counts))


def rle_from_str(counts: str) -> dict:
    return {"size": [H, W], "counts": counts.encode("ascii")}


def polygon_to_rle(segmentation: list) -> dict:
    """COCO polygon list ([[x0, y0, ...]]) -> merged RLE."""
    return mask_utils.merge(mask_utils.frPyObjects(segmentation, H, W))


def load_coco(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def group_by_physical(coco: dict) -> dict:
    """{image_stem: {"sets": {record_id: [annotation, ...]}, "file_name": str}}.

    Records with no filament keep an empty list: under PQ every prediction is a false positive there.
    """
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[str(a["image_id"])].append(a)
    phys = {}
    for im in coco["images"]:
        rid = str(im["id"])
        stem = rid.split("-", 1)[1] if "-" in rid else rid
        entry = phys.setdefault(stem, {"sets": {}, "file_name": im["file_name"]})
        entry["sets"][rid] = anns.get(rid, [])
    return phys


def gt_sets(entry: dict) -> list:
    """One list of GT RLEs per annotator record of a physical image (empty polygons are dropped)."""
    out = []
    for anns in entry["sets"].values():
        rles = [polygon_to_rle(a["segmentation"]) for a in anns]
        out.append([r for r in rles if mask_utils.area(r) > 0])
    return out


def load_records(train_json: Path) -> dict:
    """{image_stem: {annotator_record_id: [RLE, ...]}}, the structure filament_seg.metrics.score expects."""
    phys = group_by_physical(load_coco(train_json))
    return {stem: dict(zip(e["sets"], gt_sets(e))) for stem, e in phys.items()}


def read_gray(path: Path) -> np.ndarray:
    """Read an H-alpha frame as a single 8-bit channel (the images are grayscale; never treat them as RGB)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def list_images(folder: Path) -> list:
    """Image files of a folder in sorted order; their stems are the submission image names."""
    return [f for f in sorted(Path(folder).glob("*")) if f.suffix.lower() in IMAGE_SUFFIXES]
