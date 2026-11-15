"""Physical 3-channel input built from the single grayscale H-alpha frame, and the soft training target.

The three channels are all derived from one 8-bit grayscale image (no colour information is used):
  0. the frame resized to ``res`` (INTER_AREA),
  1. local contrast = 31 px median background minus the frame (filaments are dark, so they become positive),
  2. CLAHE (clip 3.0, 16 x 16 tiles), which equalises the limb darkening and observatory differences.
"""
import cv2
import numpy as np

from .io import H, W, decode, polygon_to_rle


def make_input(img: np.ndarray, res: int = 1024) -> np.ndarray:
    """2048 grayscale -> (3, res, res) uint8."""
    small = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(small, 31)
    diff = bg.astype(np.int16) - small.astype(np.int16)
    diff = np.clip(diff * 4 + 128, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16)).apply(small)
    return np.stack([small, diff, clahe], axis=0)


def local_contrast(raw: np.ndarray) -> np.ndarray:
    """Full-resolution local contrast (int16, dark structures positive); used by the verifier features."""
    bg = cv2.medianBlur(raw, 31)
    return bg.astype(np.int16) - raw.astype(np.int16)


def soft_target(entry: dict, res: int = 1024) -> np.ndarray:
    """Annotator-mean soft mask: union of the filaments of each record, averaged over records -> uint8 (res, res)."""
    acc = np.zeros((H, W), np.float32)
    for anns in entry["sets"].values():
        m = np.zeros((H, W), bool)
        for a in anns:
            m |= decode(polygon_to_rle(a["segmentation"]))
        acc += m
    acc /= max(len(entry["sets"]), 1)
    return np.clip(cv2.resize(acc, (res, res), interpolation=cv2.INTER_AREA) * 255, 0, 255).astype(np.uint8)
