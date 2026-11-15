"""Full-disk semantic model: U-Net++ (ResNet-34 encoder, ImageNet init) on the 1024 physical 3-channel input.

The target is the annotator-mean soft mask (preprocess.soft_target); the loss is BCE + soft Dice.
Model selection: the epoch with the best validation PQ of a simple decoding (components at 0.5,
mean probability >= 0.3, area >= 200), evaluated every ``eval_every`` epochs.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from .candidates import upsample_prob
from .io import encode
from .metrics import PQ

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_model(encoder="resnet34", pretrained=True):
    import segmentation_models_pytorch as smp
    return smp.UnetPlusPlus(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None, in_channels=3, classes=1)


class Dataset(torch.utils.data.Dataset):
    """Flips, 90 degree rotations and a global gain / offset jitter (a proxy for observatory differences)."""

    def __init__(self, images, soft, idx, train):
        self.images, self.soft, self.idx, self.train = images, soft, idx, train

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]
        x = self.images[k].astype(np.float32) / 255.0
        y = self.soft[k].astype(np.float32) / 255.0
        if self.train:
            if np.random.rand() < 0.5:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if np.random.rand() < 0.5:
                x, y = x[:, ::-1, :], y[::-1, :]
            r = np.random.randint(4)
            x, y = np.rot90(x, r, axes=(1, 2)), np.rot90(y, r)
            x = np.clip(x * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05), 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def loss_fn(logit, y):
    p = torch.sigmoid(logit)
    inter = (p * y).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return F.binary_cross_entropy_with_logits(logit, y) + dice.mean()


@torch.no_grad()
def predict(model, images, dev, bs=4, tta=False, amp=None):
    """(N, 3, R, R) uint8 -> (N, R, R) float32 probabilities. tta = mean over identity, h-flip and v-flip.

    amp: None = bf16 autocast on CUDA and fp32 on CPU (the training-time validation pass that made the OOF maps);
    True / False force it. The submitted runs pair OOF maps from that validation pass (CUDA, bf16) with test maps
    computed on CPU in fp32, so test-time callers pass amp=False and a GPU run gives the submitted test maps.
    """
    use_amp = (dev.type == "cuda") if amp is None else bool(amp)
    model.eval()
    probs = np.zeros((len(images), images.shape[-2], images.shape[-1]), np.float32)
    for s in range(0, len(images), bs):
        x = torch.from_numpy(images[s:s + bs].astype(np.float32) / 255.0).to(dev)
        x = (x - MEAN.to(dev)) / STD.to(dev)
        with torch.autocast(device_type="cuda" if dev.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=use_amp):
            p = torch.sigmoid(model(x))
            if tta:
                p = (p + torch.sigmoid(model(x.flip(-1))).flip(-1) + torch.sigmoid(model(x.flip(-2))).flip(-2)) / 3
        probs[s:s + bs] = p[:, 0].float().cpu().numpy()
    return probs


def simple_instances(p1024, bin_thr=0.5, min_area=200, tau=0.3):
    """Validation-time decoding used only for model selection: components with mean probability >= tau."""
    import cv2
    p = upsample_prob(p1024)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((p >= bin_thr).astype(np.uint8), connectivity=8)
    out = []
    for lb in range(1, n):
        if int(stats[lb, cv2.CC_STAT_AREA]) < min_area:
            continue
        m = labels == lb
        if float(p[m].mean()) >= tau:
            out.append(encode(m))
    return out


def train(images, soft, tr_idx, va_idx, va_gt, ckpt_path, encoder="resnet34", epochs=30, bs=4, lr=3e-4,
          eval_every=2, workers=4, log=print):
    """Train on images[tr_idx]; keep the checkpoint with the best validation PQ. Returns (best PQ, its val probabilities)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(encoder).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    dl = torch.utils.data.DataLoader(Dataset(images, soft, tr_idx, True), batch_size=bs, shuffle=True,
                                     num_workers=workers, drop_last=True, pin_memory=dev.type == "cuda")
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(dl), pct_start=0.1)
    best, best_probs, t0 = -1.0, None, time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for x, y in dl:
            x = (x.to(dev) - MEAN.to(dev)) / STD.to(dev)
            y = y.to(dev).unsqueeze(1)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit = model(x)
            loss = loss_fn(logit.float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            sched.step()
            tot += loss.item()
        msg = f"ep{ep} loss {tot / len(dl):.4f} | {time.time() - t0:.0f}s"
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            probs = predict(model, images[va_idx], dev)
            acc = PQ()
            for p, gts in zip(probs, va_gt):
                pred = simple_instances(p)
                for g in gts:
                    acc.add(g, pred)
            s = acc.result()
            msg += f" | val PQ {s['PQ']:.4f}"
            if s["PQ"] > best:
                best, best_probs = s["PQ"], probs
                torch.save(model.state_dict(), ckpt_path)
                msg += " *best*"
        log(msg)
    return best, best_probs
