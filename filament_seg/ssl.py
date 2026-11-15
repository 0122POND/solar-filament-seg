"""SimMIM-style self-supervised pre-training of the refiner encoder on unlabeled GONG H-alpha frames.

Data: 3,000 downloaded full-disk GONG frames (2011-2022, image files only, no labels or metadata), minus every frame
within 24 h or on the same UT day of a test frame (gong.clean_frames; 2,780 remain). Frames that appear in
the competition train / test sets are excluded as well. Each sample is a random on-disk square crop (side 128..1024,
log-uniform) resized to 256, randomly flipped / rotated; 50 % of the 16 x 16 patches are zeroed and the
U-Net r34 (1 input channel = raw, ImageNet init) reconstructs them (L1 on masked pixels only).
Only the encoder is kept; refiner.load_ssl_encoder copies it into input channel 0 of the refiner.
"""
import math
import time

import cv2
import numpy as np
import torch


class MaskedCrops(torch.utils.data.IterableDataset):
    def __init__(self, files, patch=16, ratio=0.5):
        self.files, self.patch, self.ratio = files, patch, ratio

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(None)
        files = self.files[wi.id::wi.num_workers] if wi else self.files
        imgs = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) for f in files]
        imgs = [m for m in imgs if m is not None and m.shape == (2048, 2048)]
        g = 256 // self.patch
        while True:
            im = imgs[rng.integers(len(imgs))]
            side = int(np.exp(rng.uniform(np.log(128), np.log(1024))))
            for _ in range(10):  # retry until the crop lies on the disk
                y0, x0 = rng.integers(0, 2048 - side, 2)
                c = im[y0:y0 + side, x0:x0 + side]
                if c.mean() > 40 and (c > 20).mean() > 0.9:
                    break
            c = cv2.resize(c, (256, 256), interpolation=cv2.INTER_AREA if side > 256 else cv2.INTER_LINEAR)
            if rng.random() < 0.5:
                c = c[:, ::-1]
            if rng.random() < 0.5:
                c = c[::-1]
            c = np.rot90(c, int(rng.integers(4))).copy()
            m = (rng.random((g, g)) < self.ratio).repeat(self.patch, 0).repeat(self.patch, 1)
            yield torch.from_numpy(c), torch.from_numpy(m)


def pretrain(files, enc_path, steps=16000, bs=48, lr=3e-4, workers=3, log=print):
    """Warm-up 500 steps then cosine decay; AdamW (wd 0.01); fp16 autocast. Saves the encoder state_dict."""
    import segmentation_models_pytorch as smp
    dl = torch.utils.data.DataLoader(MaskedCrops(files), batch_size=bs, num_workers=workers, persistent_workers=True)
    dev = torch.device("cuda")
    model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler()

    def lr_at(t):
        return lr * (t / 500 if t < 500 else 0.5 * (1 + math.cos(math.pi * (t - 500) / max(steps - 500, 1))))
    t0, acc, it = time.time(), [], iter(dl)
    for step in range(steps):
        c, m = next(it)
        x = (c.to(dev).float()[:, None] / 255.0 - 0.5) / 0.25
        mk = m.to(dev)[:, None].float()
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x * (1 - mk)).float()
        loss = ((out - x).abs() * mk).sum() / mk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        acc.append(loss.item())
        if (step + 1) % 500 == 0:
            log(f"step {step + 1} masked L1 {np.mean(acc):.4f} ({time.time() - t0:.0f}s)")
            acc = []
    torch.save(model.encoder.state_dict(), enc_path)
    return enc_path
