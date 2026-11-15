"""own2 maps of the final model: two native-resolution U-Nets per fold, ResNet-34 and EfficientNetV2-S, trained by us.

The maps are written as ``<stem>_r34.png`` / ``<stem>_eff.png`` (uint8); ``mapfeats.load_map`` reads them (the verifier
map is the mean of the two families) for the 14 map features.

Training (one family, one fold): 2048-px frames standardised by the mean and SD inside the solar disk (radius 950 px);
target = the mean over annotator records of each record's union mask (soft, uint8); random 512-px crops, 70 % centred on
a target pixel; gamma / contrast / brightness jitter, rot90 and flip; ImageNet initialisation (smp / timm); AdamW lr 3e-4,
wd 1e-4, batch 16, 10,500 steps, 300 warm-up steps then cosine to 1 %; loss BCE + soft Dice; fp16; seed 0.

Inference: D4 test-time augmentation over 8 views, fp16 autocast, views in the order flip (outer) x rot90 (inner),
probabilities summed in float32 and divided by 8, rounded to uint8. This is the exact computation of the submitted runs
(byte-identical maps need this order).

Training frames get maps only from the fold model that did not see them (OOF). Test frames get one map per fold model
(``<out>/own_maps/test_f<k>/``): the final model and its variants compute features and the refiner plane per fold map and average the outputs,
the same construction as the OOF ("P-MATCH").
"""
import math
from pathlib import Path

import cv2
import numpy as np

FULL = 2048
R_DISK = 950
ENC = {"r34": "resnet34", "eff": "tu-tf_efficientnetv2_s"}
_YY, _XX = np.ogrid[:FULL, :FULL]
DISK = ((_YY - (FULL - 1) / 2) ** 2 + (_XX - (FULL - 1) / 2) ** 2) <= R_DISK ** 2


def disk_stats(img: np.ndarray):
    v = img[DISK].astype(np.float32) / 255.0
    return float(v.mean()), max(float(v.std()), 1e-3)


def soft_target(records) -> np.ndarray:
    """Mean over annotator records of each record's union mask (an empty record counts as an all-zero mask), uint8."""
    from pycocotools import mask as mU
    acc = np.zeros((FULL, FULL), np.float32)
    n = 0
    for rec in records:
        if rec:
            acc += mU.decode(mU.merge(list(rec), intersect=False)).astype(np.float32)
        n += 1
    return np.round(acc / max(n, 1) * 255).astype(np.uint8)


def build_model(family: str, pretrained: bool):
    import segmentation_models_pytorch as smp
    return smp.Unet(ENC[family], encoder_weights="imagenet" if pretrained else None, in_channels=1, classes=1)


def load_model(path: Path, family: str, device):
    import torch
    m = build_model(family, False)
    m.load_state_dict(torch.load(path, map_location="cpu")["state"])
    return m.to(device).eval()


def infer_d4(model, img: np.ndarray, device, views: int = 8) -> np.ndarray:
    """uint8 probability map (2048). views 8 = D4 (the submitted runs), 1 = the light variant (no transform)."""
    import torch
    m, s = disk_stats(img)
    x = ((torch.from_numpy(img).float() / 255.0 - m) / s)[None, None].to(device)
    p = torch.zeros((1, 1, FULL, FULL), device=device)
    amp = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else torch.autocast("cpu", enabled=False)
    with torch.no_grad(), amp:
        for fl in ((False, True) if views == 8 else (False,)):
            xf = x.flip(-1) if fl else x
            for k in (range(4) if views == 8 else (0,)):
                q = torch.sigmoid(model(torch.rot90(xf, k, (-2, -1))).float())
                q = torch.rot90(q, -k, (-2, -1))
                p += q.flip(-1) if fl else q
    return (p[0, 0] / views * 255).round().clamp(0, 255).byte().cpu().numpy()


def _sample(rng, img, tgt, pos, ms, crop, pos_frac):
    r = rng.random() if len(pos) else 1.0
    if len(pos) and r < pos_frac:
        cy, cx = divmod(int(pos[rng.integers(len(pos))]), FULL)
        y0 = int(np.clip(cy - crop // 2, 0, FULL - crop)); x0 = int(np.clip(cx - crop // 2, 0, FULL - crop))
    else:
        y0, x0 = (int(v) for v in rng.integers(0, FULL - crop + 1, 2))
    x = img[y0:y0 + crop, x0:x0 + crop].astype(np.float32) / 255.0
    t = tgt[y0:y0 + crop, x0:x0 + crop].astype(np.float32) / 255.0
    x = np.clip(np.power(x, rng.uniform(0.8, 1.25)) * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
    x = (x - ms[0]) / ms[1]
    k = int(rng.integers(0, 4)); x, t = np.rot90(x, k), np.rot90(t, k)
    if rng.random() < 0.5:
        x, t = x[:, ::-1], t[:, ::-1]
    return np.ascontiguousarray(x), np.ascontiguousarray(t)


def train(images: dict, records: dict, train_names: list, family: str, out_path: Path, steps=10500, warmup=300, bs=16,
          crop=512, pos_frac=0.7, lr=3e-4, log=print):
    """images: {stem: uint8 2048 gray}; records: {stem: [annotator record = [RLE]]}. Saves {'state': ...} to out_path."""
    import torch
    import torch.nn.functional as F
    dev = torch.device("cuda"); torch.manual_seed(0); rng = np.random.default_rng(0)
    ms = {n: disk_stats(images[n]) for n in train_names}
    tgt = {n: soft_target(records[n]) for n in train_names}
    pos = {n: np.flatnonzero(tgt[n].ravel() > 0) for n in train_names}
    model = build_model(family, True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4); scaler = torch.amp.GradScaler()
    lr_at = lambda s: (s + 1) / warmup if s < warmup else 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at); model.train(); tl = 0.0
    for s in range(steps):
        bx, bt = zip(*(_sample(rng, images[n], tgt[n], pos[n], ms[n], crop, pos_frac) for n in (train_names[i] for i in rng.integers(0, len(train_names), bs))))
        x = torch.from_numpy(np.stack(bx))[:, None].to(dev, non_blocking=True); t = torch.from_numpy(np.stack(bt))[:, None].to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            lo = model(x).float()
        p = torch.sigmoid(lo); dice = 1 - (2 * (p * t).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)
        loss = F.binary_cross_entropy_with_logits(lo, t) + dice.mean()
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); tl += loss.item()
        if (s + 1) % 500 == 0:
            log(f"[ownmap {family}] step {s + 1}/{steps} loss {tl / 500:.4f}"); tl = 0.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=model.state_dict(), family=family, steps=steps), out_path)
    return model.eval()


def write_maps(model, family: str, images: dict, names: list, out_dir: Path, device, views: int = 8):
    """Writes <out_dir>/<stem>_<family>.png for each name (family r34 or eff, the names mapfeats.load_map reads)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        cv2.imwrite(str(out_dir / f"{n}_{family}.png"), infer_d4(model, images[n], device, views))
