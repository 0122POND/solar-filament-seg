"""nnU-Net candidates for the NLv selector of the final model: the nnU-Net probability map replaces the U-Net++ map as the source of the
connected-component candidates (and is the 7th input plane of the NNR refiner).

nnU-Net v2 (GitHub master 288572b, Apache-2.0), dataset Dataset101_Solar: 2048-px grey PNG, one channel, label 1 =
union of all annotator records; splits_final.json = our 5 folds (fold k validates on the frames of split_f{k}).
Plans nnUNetPlannerResEncL (nnUNetResEncUNetLPlans), configuration 2d, default trainer settings (augmentation, deep
supervision, mirroring TTA). Epochs of the submitted run: fold 0 1000 (nnUNetTrainer), folds 2-4 250
(nnUNetTrainer_250epochs), fold 1 = checkpoint_latest of the 1000-epoch run at epoch 373 (the submitted weights; README).
Commands (GPU, about 13 h per fold for 250 epochs on an RTX PRO 6000; not run in this repository's tests):
  nnUNetv2_plan_and_preprocess -d 101 -pl nnUNetPlannerResEncL
  nnUNetv2_train 101 2d <fold> -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_250epochs --npz
Maps: softmax probability of class 1 at 2048 px -> uint8 round(p * 255) -> <stem>_rep.png. Training frames use the
fold model that validated on them (OOF). Test frames are predicted by every fold model (one map folder per fold, f{k}/):
the candidates of test frame i come from map i % 5 (as the Mask DINO candidates), and the refiner of fold k reads map k.

Candidates (replacement mode of the development code, src/verifier.build_image(..., comp_p=nn_map)):
8-connected components of nn_map >= 0.3 and >= 0.4 with area >= 50, kept under the source names v1_b03 / v1_b04;
their conf and all p-features still come from the U-Net++ map (upsampled to 2048), so the U-Net++ model is still a
part of the final model. Mask DINO Swin-L instances (score >= 0.05) are the "yolo" source.
Checked on OOF frames: masks, order, sources and conf equal outputs/fuseD/verifier_cands_mdNL (tests/test_golden_nlv.py).
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from .candidates import bbox_of, prob_components

DATASET = "Dataset101_Solar"


# ----------------------------------------------------------------------------- dataset (nnU-Net raw layout)

def export_dataset(raw_root: Path, images: dict, union_masks: dict, splits: dict):
    """images: {stem: 2048 grey uint8}; union_masks: {stem: 2048 bool (union of all records)}; splits: {stem: fold}.

    Writes imagesTr/<stem>_0000.png, labelsTr/<stem>.png, dataset.json and splits_final.json (to be copied into
    nnUNet_preprocessed/Dataset101_Solar/ after planning)."""
    d = Path(raw_root) / DATASET
    (d / "imagesTr").mkdir(parents=True, exist_ok=True); (d / "labelsTr").mkdir(exist_ok=True)
    for n, img in images.items():
        cv2.imwrite(str(d / "imagesTr" / f"{n}_0000.png"), img)
        cv2.imwrite(str(d / "labelsTr" / f"{n}.png"), union_masks[n].astype(np.uint8))
    json.dump(dict(channel_names={"0": "Halpha"}, labels={"background": 0, "filament": 1}, numTraining=len(images), file_ending=".png"),
              open(d / "dataset.json", "w"), indent=1)
    folds = sorted(set(splits.values()))
    sp = [dict(train=sorted(n for n, f in splits.items() if f != k), val=sorted(n for n, f in splits.items() if f == k)) for k in folds]
    json.dump(sp, open(d / "splits_final.json", "w"))
    return d


# ----------------------------------------------------------------------------- maps (GPU)

def predict_maps(model_dir: Path, fold: int, in_dir: Path, out_dir: Path, checkpoint="checkpoint_final.pth", device="cuda"):
    """nnU-Net inference of one fold model on a folder of <stem>_0000.png -> <stem>_rep.png (uint8 class-1 probability).

    model_dir = nnUNet_results/Dataset101_Solar/<trainer>__nnUNetResEncUNetLPlans__2d."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    os.environ.setdefault("nnUNet_compile", "f")
    # same settings as `nnUNetv2_predict` (step 0.5, Gaussian, mirroring). nnU-Net sets cudnn.benchmark = True, so the
    # uint8 map can differ by 1-3 / 255 between processes / GPUs (checked on an RTX PRO 6000 vs the submitted GTX 1080 Ti maps)
    pred = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=True, device=torch.device(device), allow_tqdm=False)
    pred.initialize_from_trained_model_folder(str(model_dir), use_folds=(fold,), checkpoint_name=checkpoint)
    tmp = Path(out_dir) / "_npz"; tmp.mkdir(parents=True, exist_ok=True)
    pred.predict_from_files(str(in_dir), str(tmp), save_probabilities=True, overwrite=True, num_processes_preprocessing=2, num_processes_segmentation_export=2)
    for f in sorted(tmp.glob("*.npz")):
        p = np.load(f)["probabilities"]; p = p[1].reshape(p.shape[-2], p.shape[-1])
        cv2.imwrite(str(Path(out_dir) / f"{f.stem}_rep.png"), np.clip(np.round(p * 255), 0, 255).astype(np.uint8))
    import shutil
    shutil.rmtree(tmp)   # float probabilities of nnU-Net (about 10 MB per frame): only the uint8 maps are read afterwards
    return out_dir


def network_weights_only(checkpoint: Path, out: Path):
    """Inference-only weights for distribution (optimizer state etc. dropped): scripts/export_weights.py strip-nnunet,
    which reads .safetensors checkpoints of nnU-Net master with nnU-Net's own loader."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("export_weights", Path(__file__).resolve().parents[1] / "scripts" / "export_weights.py")
    ew = importlib.util.module_from_spec(spec); spec.loader.exec_module(ew)
    return ew.strip_nnunet(checkpoint, out)


def load_map(map_dir: Path, stem: str) -> np.ndarray:
    """nnU-Net map of one frame as float32 in [0, 1] (2048), as read by the development code (cv2 grey / 255)."""
    m = cv2.imread(str(Path(map_dir) / f"{stem}_rep.png"), cv2.IMREAD_GRAYSCALE)
    assert m is not None, f"no nnU-Net map for {stem} in {map_dir}"
    return m.astype(np.float32) / 255.0


def load_map_u8(map_dir: Path, stem: str) -> np.ndarray:
    """uint8 map (the NNR refiner's 7th plane)."""
    m = cv2.imread(str(Path(map_dir) / f"{stem}_rep.png"), cv2.IMREAD_GRAYSCALE)
    assert m is not None, f"no nnU-Net map for {stem} in {map_dir}"
    return m


# ----------------------------------------------------------------------------- candidates

def collect(p2048: np.ndarray, nn_map: np.ndarray, md_items: list, bins=(0.3, 0.4), min_area=50, conf_min=0.05) -> list:
    """Candidate union of the final model: nnU-Net components (conf = mean U-Net++ probability inside) + Mask DINO Swin-L.

    p2048: U-Net++ probability (2048, float); nn_map: nnU-Net probability (2048, float in [0, 1]);
    md_items: [(score, RLE)] of the Swin-L model. Returns [(source, conf, mask, bbox)] for features.featurize."""
    cands = []
    for bt in bins:
        for m, bb in prob_components(nn_map, bt, min_area):
            cands.append((f"v1_b{int(round(bt * 10)):02d}", float(p2048[m].mean()), m, bb))
    for c, r in md_items:
        if c < conf_min:
            continue
        m = mask_utils.decode(r).astype(bool)
        if m.sum() < 1:
            continue
        cands.append(("yolo", float(c), m, bbox_of(m)))
    return cands
