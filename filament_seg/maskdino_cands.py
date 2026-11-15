"""Mask DINO candidates of the final model and its variants (second candidate source instead of YOLO11-seg).

Setup (once; see README "Mask DINO"):
  pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   (Blackwell GPUs: torch 2.7.1 + cu128)
  git clone https://github.com/facebookresearch/detectron2 && git -C detectron2 checkout 247691a
  CUDA_VISIBLE_DEVICES="" FORCE_CUDA=0 pip install --no-build-isolation -e detectron2        (CPU ops are enough)
  git clone https://github.com/IDEA-Research/MaskDINO && git -C MaskDINO checkout 3831d85
  One-line change in MaskDINO/maskdino/modeling/pixel_decoder/ops/functions/ms_deform_attn_func.py: replace
  ``raise ModuleNotFoundError(info_string)`` by ``MSDA = None`` so that the multi-scale deformable attention falls back to
  its pure-PyTorch implementation (same numbers, slower; no CUDA compiler needed).
  COCO weights (Apache-2.0): maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth from
  https://github.com/IDEA-Research/detrex-storage/releases/tag/maskdino-v0.1.0

Model: Mask DINO R50, config configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml, one class,
100 queries, 50 denoising queries, pixel mean / std 127.5 / 64 on the grayscale frame resized to 1024 and replicated to 3
channels. Training: each sample uses the masks of one annotator record drawn at random (resized to 1024, > 0.25 after
area interpolation, >= 10 px), D4 augmentation, AdamW lr 1e-4 (backbone x0.1), wd 0.05, OneCycle (5 % warm-up), gradient
clip 0.01, batch 1, 40 epochs, seed 0. Inference: masks resized to 2048 by the model, kept with score >= 0.05 and >= 30 px.
Training frames get candidates from the fold model that did not see them; test frame i (sorted names) uses fold model i % 5.

NLv selector (configs/nlv.yaml; also the final model): arch "swinl" = the candidate source, arch "r50" with seeds 0 / 1 / 2 = the three voters
(filament_seg/votes.py; they are not candidates). Swin-L: config
configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml of the MaskDINO repository (the
Swin-L backbone despite the file name; 4 scales), COCO weights
maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth (same release, Apache-2.0; stored on the
training server as maskdino_swinl_coco.pth). Inference checked on 6 test frames: scores and masks identical to the
submitted run for Swin-L and R50 (README, "GPU stages vs the submitted run"); every
other setting as R50 (1024, batch 1, 100 queries, 50 DN, 40 epochs, seed 0; about 32 GB GPU memory).
size: input side (MD_SIZE of the development scripts; 1024 in the submitted run, 1536 = the fine-tuning trial).
"""
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mU

S = 1024  # default input side
CONFIGS = dict(
    r50="configs/coco/instance-segmentation/maskdino_R50_bs16_50ep_3s_dowsample1_2048.yaml",
    swinl="configs/coco/instance-segmentation/swin/maskdino_R50_bs16_50ep_4s_dowsample1_2048.yaml",  # Swin-L (file name as in the repository)
)
COCO_WEIGHTS = dict(
    r50="maskdino_r50_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask46.3ap_box51.7ap.pth",
    swinl="maskdino_swinl_50ep_300q_hid2048_3sd1_instance_maskenhanced_mask52.3ap_box59.0ap.pth",  # release name; the training server holds it as maskdino_swinl_coco.pth
)


def _cfg(maskdino_root: Path, weights: str = "", device: str = "cuda", arch: str = "r50"):
    sys.path.insert(0, str(maskdino_root))
    from detectron2.config import get_cfg
    from detectron2.projects.deeplab import add_deeplab_config
    from maskdino import add_maskdino_config
    cfg = get_cfg(); add_deeplab_config(cfg); add_maskdino_config(cfg)
    cfg.merge_from_file(str(maskdino_root / CONFIGS[arch]))
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1; cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD = 0.05
    cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES = 100; cfg.MODEL.MaskDINO.DN_NUM = 50
    cfg.MODEL.PIXEL_MEAN = [127.5] * 3; cfg.MODEL.PIXEL_STD = [64.0] * 3; cfg.INPUT.FORMAT = "RGB"
    cfg.MODEL.WEIGHTS = weights; cfg.MODEL.DEVICE = device; cfg.freeze()
    return cfg


def prep(img: np.ndarray, size: int = S) -> np.ndarray:
    """2048 grayscale uint8 -> (3, size, size) float32 in 0..255 (normalised inside the model)."""
    return np.repeat(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)[None].astype(np.float32), 3, 0)


def _masks_of(record, size: int = S):
    out = []
    for r in record:
        m = cv2.resize(mU.decode(r).astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.25
        if m.sum() >= 10:
            out.append(m)
    return out


def train(maskdino_root: Path, coco_weights: Path, images: dict, records: dict, names: list, out_path: Path, epochs=40, lr=1e-4, seed=0, log=print,
          arch: str = "r50", size: int = S):
    """images: {stem: 2048 gray}; records: {stem: [annotator record = [RLE]]}. Saves the state dict to out_path.

    arch r50 (seeds 0 / 1 / 2 = the voters of NLv) or swinl (the candidate source of NLv); size = input side."""
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.modeling import build_model
    from detectron2.structures import BitMasks, Instances
    torch.manual_seed(seed); rng = random.Random(seed); np.random.seed(seed)
    cfg = _cfg(maskdino_root, str(coco_weights), arch=arch); model = build_model(cfg); DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS); model.to("cuda")
    X = {n: prep(images[n], size) for n in names}
    recs = {n: [r for r in (_masks_of(rec, size) for rec in records[n] if rec) if r] for n in names}
    tr = [n for n in names if recs[n]]
    bb = [p for k, p in model.named_parameters() if k.startswith("backbone.") and p.requires_grad]
    ot = [p for k, p in model.named_parameters() if not k.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": lr * 0.1}, {"params": ot, "lr": lr}], weight_decay=0.05)
    steps = epochs * len(tr); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr * 0.1, lr], total_steps=steps, pct_start=0.05)
    model.train(); st = 0; t0 = time.time()
    while st < steps:
        order = tr[:]; rng.shuffle(order)
        for n in order:
            x = X[n]; ms = np.stack(rng.choice(recs[n])); k = rng.randrange(4); fl = rng.random() < 0.5
            x = np.rot90(x, k, (1, 2)); ms = np.rot90(ms, k, (1, 2))
            if fl:
                x, ms = x[:, :, ::-1], ms[:, :, ::-1]
            t = torch.from_numpy(np.ascontiguousarray(ms)); inst = Instances((size, size)); inst.gt_classes = torch.zeros(len(ms), dtype=torch.long)
            inst.gt_masks = t; inst.gt_boxes = BitMasks(t).get_bounding_boxes()
            with torch.autocast("cuda", dtype=torch.float16):
                ld = model([{"image": torch.from_numpy(np.ascontiguousarray(x)), "instances": inst, "height": size, "width": size}])
            loss = sum(ld.values()); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01); opt.step(); sched.step(); st += 1
            if st % 500 == 0:
                log(f"[maskdino] step {st}/{steps} loss {float(loss):.3f} ({time.time() - t0:.0f}s)")
            if st >= steps:
                break
    out_path.parent.mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), out_path)
    return model.eval()


def load(maskdino_root: Path, path: Path, device="cuda", arch: str = "r50"):
    import torch
    from detectron2.modeling import build_model
    model = build_model(_cfg(maskdino_root, "", device, arch)); model.load_state_dict(torch.load(path, map_location="cpu")); return model.to(device).eval()


def predict(model, img: np.ndarray, size: int = S) -> list:
    """[(score, RLE 2048)] with score >= 0.05 and >= 30 px."""
    import torch
    with torch.no_grad():
        o = model([{"image": torch.from_numpy(prep(img, size)), "height": 2048, "width": 2048}])[0]["instances"]
        keep = o.scores >= 0.05   # select and binarise on the GPU: only the kept masks (uint8) are copied to the CPU
        sc = o.scores[keep].cpu().numpy(); ms = (o.pred_masks[keep] > 0.5).to(torch.uint8).cpu().numpy()
    return [(float(s), mU.encode(np.asfortranarray(m))) for s, m in zip(sc, ms) if m.sum() >= 30]


# Test-time fold models (python -m filament_seg detect-test): weight file below the weights folder, output pickle
TEST_MODELS = dict(
    swinl1024=dict(arch="swinl", size=1024, weights="nlv/maskdino_swinl_f{k}.pt", out="maskdino_test_swinl_f{k}.pkl"),          # candidates
    r50=dict(arch="r50", size=1024, weights="{d}/maskdino_r50_s{s}_f{k}.pt", out="maskdino_test_s{s}_f{k}.pkl"),                # voters (seeds 0 / 1 / 2)
    swinl1536=dict(arch="swinl", size=1536, weights="final/maskdino_swinl1536_f{k}.pt", out="maskdino_test_swinl1536_f{k}.pkl"),  # refiner planes, S15F
)


def fold_frames(n: int, k: int, all_folds: bool = False) -> list:
    """Indices of the n test frames (sorted names) that fold model k predicts. Default: frame i belongs to fold model i % 5,
    the only prediction the candidates and the votes of frame i read. all_folds: every frame (g7_m2 of configs/stk.yaml and
    the refiner planes / S15F features, which use every fold model on every frame)."""
    return [i for i in range(n) if all_folds or i % 5 == k]


def predict_test(maskdino_root: Path, weights: dict, files: list, out_pattern: str, arch: str = "r50", size: int = S, all_folds: bool = False,
                 device: str = "cuda", log=print) -> int:
    """Fold models on the test frames -> out_pattern.format(k=k) = pickle {stem: [(score, RLE)]} of fold model k.
    weights: {k: checkpoint}; files: test frames sorted by name; frames per model as in fold_frames. Returns the number of
    forward passes (len(files) with the default, 5 x len(files) with all_folds and five models)."""
    import pickle
    from . import io
    n = 0
    for k, w in sorted(weights.items()):
        t0 = time.time(); model = load(maskdino_root, w, device, arch)
        out = {files[i].stem: predict(model, io.read_gray(files[i]), size) for i in fold_frames(len(files), k, all_folds)}
        dst = Path(str(out_pattern).format(k=k)); dst.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(out, open(dst, "wb")); del model; n += len(out)
        log(f"{dst.name}: fold model {k}, {len(out)} frames, {sum(len(v) for v in out.values())} instances, {time.time() - t0:.0f}s")
    return n


def stage_test(cfg, model: str, weights: Path, out: Path = None, seeds=(0, 1, 2), folds_=(0, 1, 2, 3, 4), all_folds: bool = False,
               maskdino_root: Path = None, device: str = "cuda", limit: int = 0) -> dict:
    """GPU: test predictions of one model family (TEST_MODELS) -> <out>/maskdino_test_*_f{k}.pkl, the files the `nlv` section
    reads (test_voters, refplanes.test.swinl1536) and the Swin-L 1024 instances of the test candidates.
    swinl1024 / r50: frame i is predicted by fold model i % 5 only (36 of 180 frames per model; the final model reads nothing
    else), all_folds=True restores every model on every frame (5 x the passes; needed by g7_m2 of configs/stk.yaml).
    swinl1536 always predicts every frame with every fold model (the refiner reads the five fold-k inputs)."""
    import os
    from . import io
    m = TEST_MODELS[model]; t0 = time.time()
    files = sorted(io.list_images(cfg["paths"].test_images), key=lambda f: f.stem)
    if limit:
        files = files[:limit]
    root = Path(maskdino_root or os.environ.get("MASKDINO_ROOT", "MaskDINO")); out = Path(out or cfg["paths"].out)
    every = all_folds or model == "swinl1536"; n = 0; pats = []
    for s in (seeds if model == "r50" else [None]):
        w = {k: Path(weights) / m["weights"].format(d="final" if s == 3 else "nlv", s=s, k=k) for k in folds_}   # seed 3 (g7_m2 only): final/
        pats.append(str(out / m["out"]).replace("{s}", str(s)))
        n += predict_test(root, w, files, pats[-1], m["arch"], m["size"], every, device)
    return dict(model=model, frames=len(files), all_folds=every, passes=n, out=pats, seconds=round(time.time() - t0, 1))
