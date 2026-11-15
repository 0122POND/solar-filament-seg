"""YOLO11s-seg instance candidates (high-recall second source of seeds).

Training labels: for an image with several annotator records, the record with the most filaments is used
(one record avoids duplicated, overlapping instances). Images are the original 2048 frames (symlinked).
Inference keeps every instance with confidence >= 0.01 at 2048 resolution (retina masks); the candidate
union later keeps conf >= 0.05.
"""
import os
from pathlib import Path

import cv2
import numpy as np

from .io import H, W, encode, group_by_physical, load_coco


def make_dataset(train_json: Path, train_images: Path, split: dict, root: Path) -> Path:
    """Write an Ultralytics segmentation dataset for split {"train": [...], "val": [...]} -> data.yaml path."""
    phys = group_by_physical(load_coco(train_json))
    for s in ("train", "val"):
        (root / "images" / s).mkdir(parents=True, exist_ok=True)
        (root / "labels" / s).mkdir(parents=True, exist_ok=True)
        for stem in split[s]:
            e = phys[stem]
            best = max(e["sets"].values(), key=len)
            lines = []
            for a in best:
                for poly in a["segmentation"]:
                    if len(poly) < 6:
                        continue
                    xy = [min(max(v / 2048.0, 0.0), 1.0) for v in poly]
                    lines.append("0 " + " ".join(f"{v:.5f}" for v in xy))
            dst = root / "images" / s / e["file_name"]
            if not dst.exists():
                os.symlink((train_images / e["file_name"]).resolve(), dst)
            (root / "labels" / s / (Path(e["file_name"]).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    yaml_path = root / "data.yaml"
    yaml_path.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: filament\n")
    return yaml_path


def train(data_yaml: Path, project: Path, name: str, model="yolo11s-seg.pt", imgsz=1536, epochs=40, batch=4, device="0", workers=8):
    from ultralytics import YOLO
    YOLO(model).train(data=str(data_yaml), imgsz=imgsz, epochs=epochs, batch=batch, device=device, workers=workers,
                      project=str(project), name=name, exist_ok=True, mask_ratio=2, overlap_mask=True, cos_lr=True,
                      close_mosaic=10, patience=100, fliplr=0.5, flipud=0.5, degrees=0.0, scale=0.3, mosaic=1.0,
                      amp=True, plots=False, verbose=True)
    return project / name / "weights" / "best.pt"


def _no_nms_time_limit():
    """Lift Ultralytics' NMS wall-clock limit (2 s + 0.05 s per image), which stops NMS early on a loaded CPU.

    With one image per batch (as here) the limit only triggers a warning after the image is done, so results do not
    change; lifting it keeps them independent of machine load for any batch size.
    """
    import functools
    from ultralytics.utils import nms
    if not getattr(nms.non_max_suppression, "_no_time_limit", False):
        patched = functools.partial(nms.non_max_suppression, max_time_img=1e9)
        patched._no_time_limit = True
        nms.non_max_suppression = patched


def predict(weights: Path, paths: list, imgsz=1536, conf=0.01, iou=0.6, max_det=200, device="cpu") -> dict:
    """{stem: [(conf, RLE)]} sorted by decreasing confidence, masks at 2048 x 2048."""
    from ultralytics import YOLO
    _no_nms_time_limit()
    model = YOLO(str(weights))
    preds = {}
    results = model.predict([str(p) for p in paths], imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, retina_masks=True,
                            device=device, stream=True, verbose=False, half=(device != "cpu"), batch=1)
    for p, r in zip(paths, results):
        items = []
        if r.masks is not None:
            for m, c in zip(r.masks.data.bool().cpu().numpy(), r.boxes.conf.cpu().numpy()):
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                if m.sum() == 0:
                    continue
                items.append((float(c), encode(m)))
        preds[Path(p).stem] = sorted(items, key=lambda t: -t[0])
    return preds
