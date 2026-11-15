"""Residual crop refiner: re-draws each seed candidate at full resolution so that it crosses IoU 0.5.

Crop: square around the seed bounding box (side = 2 x max(w, h), at least 128 px, 2048 coordinates),
resized to 256. Input channels (uint8):
  0 raw H-alpha, 1 U-Net++ probability, 2 seed mask, 3 best partner mask from the other source
  (for a probability seed the best YOLO mask with IoU > 0.3, for a YOLO seed the best 0.4-component),
  4 is_yolo plane, 5 is_yolo_only plane (YOLO seed whose cluster has no probability candidate).
  Source names are kept from the base pipeline: in the final model the "YOLO" source is the Mask DINO Swin-L instances (README, naming table).

Optional 7th crop plane (in_channels 7 incl. the two source planes -> 4 + 1 + 2): configs/nlv.yaml (NNR, refiner.plane7
nnunet) = nnU-Net probability map (filament_seg/nnunet_cands.load_map_u8); training
crops from the OOF maps (nlv.stage_nnr_crops, a Python function without a CLI stage), test crops from the map of the matching fold model (ensemble_residual_pmatch).

Model (ResidualRefiner): logit = alpha * (2 * seed - 1) + U-Net_r34(x). The segmentation head starts at zero,
so the untrained model returns the seed itself; the network only learns corrections.
Stage 1 ("A2") is a plain 4-channel U-Net trained with hard-example weights; in the champion it is only
used to define the PQ-shaped sample weights of the residual model (which seeds stage 1 broke or fixed).
Paste-back keeps the single connected component that overlaps the seed most; if none, the seed is kept.

Optional feature input (S15F, configs/s15f.yaml, filament_seg/s15f.py): a (C, S/4, S/4) uint8 crop of the Swin-L 1536 Mask DINO
mask_features in the common 16-channel coordinates (round(z * 32 + 128)) is added, through a zero-initialised 1x1 conv
(`fproj`, C -> 64), to the stride-4 encoder stage (features[2] = ResNet layer1 output). At the start of training the model is
therefore the 9-plane NNRP function; flips / rotations of training and the flip TTA are applied to the features as to the
planes. Port of the development script refevo/s15feat/refiner_feat.py (--feat_ch).

Several refiners can be averaged at test time (refiner.members in the config, ensemble_members): every member is a list of
fold checkpoints (and optionally its fold-k feature crops); the probability is the equal-weight mean over the members of the
mean over their fold models.

Full-data refiners (the final model configs/final_yellow.yaml, and configs/final_fullfit.yaml): `residual --fold -1` trains on the crops of all 707 training
frames (no held-out rows, checkpoint refiner_<tag>_f-1.pt). At test time the one full-data checkpoint takes the place of every
fold model: it is applied to each of the five fold-k inputs (nnU-Net map, NNRP planes and S15F features of fold model k) and
the five probabilities are averaged, exactly as for the fold models (member `full: true`, checkpoint pattern without {k}).
`seed` (train_residual / residual --seed, or the `seed` of a refiner.members entry) reproduces the seeded members: x16 / x17 / x18
of the final model (seeds 16 / 17 / 18) and x11 / x12 of configs/final_fullfit.yaml; without it the torch / numpy streams are
not reset (member o of configs/final_fullfit.yaml).
"""
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pycocotools import mask as mask_utils

from .io import H, W, encode


# ----------------------------------------------------------------------------- crops

def crop_box(x, y, w, h, scale=2.0, min_side=128):
    side = int(max(max(w, h) * scale, min_side))
    cx, cy = x + w / 2, y + h / 2
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def pad_crop(img, x0, y0, side):
    """side x side window at (x0, y0); parts outside the image are zero."""
    out = np.zeros((side, side), img.dtype)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(img.shape[1], x0 + side), min(img.shape[0], y0 + side)
    if xs1 > xs0 and ys1 > ys0:
        out[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0] = img[ys0:ys1, xs0:xs1]
    return out


def partners(rles: list, sources: list, min_iou: float = 0.3):
    """Best other-source candidate per seed -> (partner index or -1, partner IoU, pairwise IoU matrix)."""
    n = len(rles)
    iou_cc = np.asarray(mask_utils.iou(rles, rles, [0] * n)).reshape(n, n) if n else np.zeros((0, 0))
    src = np.array(sources)
    is_yolo = src == "yolo"
    is_b04 = src == "v1_b04"
    idx = np.full(n, -1)
    piou = np.zeros(n)
    for k in range(n):
        cand = np.where(is_b04 if is_yolo[k] else is_yolo)[0]
        if len(cand):
            j = cand[iou_cc[k, cand].argmax()]
            if iou_cc[k, j] > min_iou:
                idx[k], piou[k] = int(j), float(iou_cc[k, j])
    return idx, piou, iou_cc


def make_crops(raw, p1024, rles, sources, bboxes, size=256, scale=2.0, min_side=128, extra=()):
    """4-channel crops for the seeds of one image.

    raw: 2048 frame; p1024: U-Net++ probability (1024); rles / sources / bboxes: the seeds, in candidate order;
    extra: further full-resolution uint8 masks per seed to crop the same way (e.g. the training teacher).
    Returns X (n, 4, size, size) uint8, boxes (n, 3) = (x0, y0, side), partner indices, extra crops.
    """
    p = np.asarray(p1024, dtype=np.float32)
    if p.max() > 1.5:
        p = p / 255.0
    p8 = cv2.resize((p * 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
    pidx, _, _ = partners(rles, sources)
    n = len(rles)
    X = np.zeros((n, 4, size, size), np.uint8)
    boxes = np.zeros((n, 3), np.int64)
    ex = [np.zeros((n, size, size), np.uint8) for _ in extra]
    for k, (x, y, w, h) in enumerate(bboxes):
        x0, y0, side = crop_box(x, y, w, h, scale, min_side)
        seed = mask_utils.decode(rles[k]).astype(np.uint8) * 255
        part = mask_utils.decode(rles[pidx[k]]).astype(np.uint8) * 255 if pidx[k] >= 0 else np.zeros((H, W), np.uint8)
        interp = cv2.INTER_AREA if side > size else cv2.INTER_LINEAR
        X[k] = np.stack([cv2.resize(pad_crop(a, x0, y0, side), (size, size), interpolation=interp) for a in (raw, p8, seed, part)])
        for e, masks in zip(ex, extra):
            e[k] = cv2.resize(pad_crop(masks[k], x0, y0, side), (size, size), interpolation=interp)
        boxes[k] = (x0, y0, side)
    return X, boxes, pidx, ex


def crop_plane(full, boxes, size=256):
    """Crop one full-resolution uint8 plane (e.g. the nnU-Net probability map) with the crop boxes of make_crops."""
    out = np.zeros((len(boxes), size, size), np.uint8)
    for k, (x0, y0, side) in enumerate(np.asarray(boxes)):
        out[k] = cv2.resize(pad_crop(full, int(x0), int(y0), int(side)), (size, size), interpolation=cv2.INTER_AREA if side > size else cv2.INTER_LINEAR)
    return out


def paste_back(prob_crop, x0, y0, side, seed_mask, thr: float = 0.5):
    """256 probability -> side x side -> 2048 canvas (> thr); keep the component overlapping the seed most.

    Returns (mask, status) with status "ok", "empty" (nothing above 0.5) or "disjoint" (no overlap with the seed);
    in the last two cases the seed is returned unchanged.
    """
    pr = cv2.resize(prob_crop, (side, side), interpolation=cv2.INTER_LINEAR)
    b = (pr > thr).astype(np.uint8)
    canvas = np.zeros((H, W), np.uint8)
    xs0, ys0 = max(0, x0), max(0, y0)
    xs1, ys1 = min(W, x0 + side), min(H, y0 + side)
    if xs1 > xs0 and ys1 > ys0:
        canvas[ys0:ys1, xs0:xs1] = b[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0]
    n, lab = cv2.connectedComponents(canvas, connectivity=8)
    if n <= 1:
        return seed_mask, "empty"
    ov = np.bincount(lab[seed_mask], minlength=n)
    ov[0] = 0
    if ov.max() == 0:
        return seed_mask, "disjoint"
    return lab == int(ov.argmax()), "ok"


def gt_metrics(rle, gt_sets):
    """(max IoU with any GT, verifier target y) of one mask."""
    mx, y = 0.0, 0.0
    for g in gt_sets:
        if not g:
            continue
        b = float(np.asarray(mask_utils.iou([rle], g, [0] * len(g))).reshape(-1).max())
        mx = max(mx, b)
        y += b if b > 0.5 else 0.0
    return mx, y / max(len(gt_sets), 1)


def teacher_for_seeds(rles, gt_sets, min_iou=0.1):
    """Teacher of each seed = the single GT instance (over all records) with the highest IoU, if IoU >= min_iou."""
    gt_all = [r for g in gt_sets for r in g]
    n = len(rles)
    if not (n and gt_all):
        return [None] * n, np.zeros(n)
    iou = np.asarray(mask_utils.iou(rles, gt_all, [0] * len(gt_all))).reshape(n, len(gt_all))
    best, best_iou = iou.argmax(1), iou.max(1)
    return [gt_all[best[k]] if best_iou[k] >= min_iou else None for k in range(n)], best_iou


def stage1_weights(area, max_iou, best_gt_iou, partner, partner_gt_iou, partner_area, teacher_area):
    """Hard-example sampling weight of stage 1 (one seed)."""
    ratio = partner_area / max(area, 1) if partner >= 0 else 1.0
    w = 1.0 + (area > 5000) + (0.2 <= max_iou <= 0.5) + (partner >= 0 and partner_gt_iou > best_gt_iou + 0.02) \
        + (partner >= 0 and (ratio < 0.7 or ratio > 1.3)) + (teacher_area > 1.3 * area)
    return 0.2 if teacher_area == 0 else float(w)


def residual_weights(meta: pd.DataFrame, protect: float = 1.5):
    """PQ-shaped sample weight w and hinge coefficient c of the residual refiner.

    meta columns: source, cluster_n_src, area, teacher_area, maxiou_seed (seed vs GT), maxiou_ref (stage-1 output vs GT).
    w = 1 + 2[0.4 <= seed IoU <= 0.7] + 3[stage 1 turned a TP into a miss] + 2[stage 1 crossed 0.5 upwards]
          + 1[YOLO-only seed] + 1[teacher > 1.3 x seed area];  0.2 when the teacher is empty.
    c = protect for seeds that already hit (do not break TPs), 2 for 0.3..0.5 (push them over), 0.5 otherwise.
    """
    is_y = (meta.source == "yolo").values
    yo = is_y & (meta.cluster_n_src.values == 1)
    si, ri = meta.maxiou_seed.values, meta.maxiou_ref.values
    w = 1 + 2 * ((si >= 0.4) & (si <= 0.7)) + 3 * ((si > 0.5) & (ri <= 0.5)) + 2 * ((si <= 0.5) & (ri > 0.5)) + 1 * yo \
        + 1 * (meta.teacher_area.values > 1.3 * meta.area.values)
    w = np.where(meta.teacher_area.values == 0, 0.2, w).astype(float)
    c = np.where(si > 0.5, protect, np.where(si >= 0.3, 2.0, 0.5)) * (meta.teacher_area.values > 0)
    return w, c, np.stack([is_y, yo], 1).astype(np.float32)


# ----------------------------------------------------------------------------- models

def _unet(in_channels, pretrained):
    import segmentation_models_pytorch as smp
    return smp.Unet("resnet34", encoder_weights="imagenet" if pretrained else None, in_channels=in_channels, classes=1)


class ResidualRefiner(nn.Module):
    """in_channels = 4 crop planes (+ extra crop planes, e.g. the nnU-Net map of configs/nlv.yaml) + 2 source planes."""

    def __init__(self, alpha: float = 1.0, pretrained: bool = True, in_channels: int = 6):
        super().__init__()
        self.alpha = alpha
        self.net = _unet(in_channels, pretrained)
        head = self.net.segmentation_head[0]
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        if in_channels > 6:  # extra crop planes start with zero first-layer weights: training starts from the 6-channel function
            with torch.no_grad():
                self.net.encoder.conv1.weight[:, 4:in_channels - 2] = 0.0

    def add_feat(self, cin: int = 16):
        """Zero-initialised 1x1 projection of cin feature channels onto the stride-4 encoder stage (S15F).
        Created under fork_rng so that the torch random stream (and hence training) is the same as without features."""
        with torch.random.fork_rng(devices=[]):
            self.fproj = nn.Conv2d(cin, self.net.encoder.out_channels[2], 1)
        nn.init.zeros_(self.fproj.weight)
        nn.init.zeros_(self.fproj.bias)
        return self

    def forward(self, x, seed, f=None):
        if f is None:
            return self.alpha * (2 * seed - 1) + self.net(x)
        fs = list(self.net.encoder(x))
        assert fs[2].shape[-2:] == f.shape[-2:], (fs[2].shape, f.shape)
        fs[2] = fs[2] + self.fproj(f)
        return self.alpha * (2 * seed - 1) + self.net.segmentation_head(self.net.decoder(fs))


def load_ssl_encoder(encoder: nn.Module, path):
    """Copy a 1-channel SimMIM encoder into a multi-channel encoder: conv1 goes to input channel 0 (raw)."""
    sd = torch.load(path, map_location="cpu")
    cw = encoder.conv1.weight.detach().cpu().clone()
    cw[:, 0:1] = sd.pop("conv1.weight")
    sd["conv1.weight"] = cw
    return nn.Module.load_state_dict(encoder, sd, strict=False)


def normalize(xb_u8):
    """uint8 crops -> float; the raw channel is standardised, masks stay in [0, 1]."""
    x = xb_u8.float() / 255.0
    x[:, 0] = (x[:, 0] - 0.5) / 0.25
    return x


def fnorm(fb_u8, dev):
    """uint8 feature crops round(z * 32 + 128) -> z (float)."""
    return (torch.as_tensor(np.ascontiguousarray(fb_u8) if isinstance(fb_u8, np.ndarray) else fb_u8).to(dev).float() - 128.0) / 32.0


def build_input(xb_u8, src2):
    """(B, 4 + extra, S, S) uint8 + (B, 2) source flags -> (6 + extra)-channel input and the binary seed."""
    x = normalize(xb_u8)
    seed = (x[:, 2:3] > 0.5).float()
    planes = src2[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
    return torch.cat([x, planes], 1), seed


def _flip_rot_gain(tensors, x_index=0):
    """Random flips / 90 degree rotations shared by all tensors, then gain and offset jitter on the raw channel."""
    if torch.rand(1).item() < 0.5:
        tensors = [t.flip(-1) for t in tensors]
    if torch.rand(1).item() < 0.5:
        tensors = [t.flip(-2) for t in tensors]
    k = int(torch.randint(0, 4, (1,)).item())
    if k:
        tensors = [torch.rot90(t, k, (-2, -1)) for t in tensors]
    x = tensors[x_index]
    B = x.shape[0]
    x[:, 0] = x[:, 0] * (1 + 0.15 * (torch.rand(B, 1, 1, device=x.device) - 0.5)) + 0.1 * (torch.rand(B, 1, 1, device=x.device) - 0.5)
    return tensors


TTA_FLIPS = ([], [-1], [-2], [-1, -2])


@torch.no_grad()
def predict_residual(model, X, S2, dev, bs=16, FT=None):
    """Flip-TTA x4 probabilities (N, S, S) of the residual refiner. X: uint8 tensor, S2: float tensor.
    FT: optional (N, C, S/4, S/4) uint8 feature crops (S15F; a numpy memmap is read batch by batch = lazy loading)."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x, seed = build_input(X[i:i + bs].to(dev), S2[i:i + bs].to(dev))
        f = fnorm(FT[i:i + bs], dev) if FT is not None else None
        p = 0
        for fl in TTA_FLIPS:
            xx, ss = (x.flip(fl), seed.flip(fl)) if fl else (x, seed)
            ff = (f.flip(fl) if fl else f) if f is not None else None
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                q = torch.sigmoid(model(xx, ss, f=ff).float())
            p = p + (q.flip(fl) if fl else q)
        out.append((p / 4)[:, 0].cpu())
    return torch.cat(out).numpy()


@torch.no_grad()
def predict_stage1(model, X, dev, bs=64, tta=True):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = normalize(X[i:i + bs].to(dev))
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            p = torch.sigmoid(model(xb).float())
            if tta:
                for fl in TTA_FLIPS[1:]:
                    p = p + torch.sigmoid(model(xb.flip(fl)).float()).flip(fl)
                p = p / 4
        out.append(p[:, 0].cpu())
    return torch.cat(out).numpy()


def ensemble_residual(checkpoints, X, S2, alpha=1.0, device="cpu", bs=16):
    """Mean of the fold models' TTA probabilities (the test-time refiner). X has 4 or more crop planes."""
    dev = torch.device(device)
    Xt, St = torch.from_numpy(X), torch.from_numpy(S2)
    prob = np.zeros((len(X), X.shape[2], X.shape[3]), np.float32)
    for ck in checkpoints:
        m = ResidualRefiner(alpha, pretrained=False, in_channels=X.shape[1] + 2)
        m.load_state_dict(torch.load(ck, map_location="cpu")["state"])
        prob += predict_residual(m.to(dev), Xt, St, dev, bs)
    return prob / len(checkpoints)


def ensemble_residual_pmatch(checkpoints, X, S2, plane7, alpha=1.0, device="cpu", bs=16):
    """Test-time refiner with a fold-matched 7th plane ("P-MATCH"): fold model k reads plane7(k), the crops of the map of
    fold model k (configs/nlv.yaml NNR: the nnU-Net map of fold k), and the five
    TTA probabilities are averaged. X: (N, 4, S, S) base planes; plane7(k) -> (N, 1, S, S) uint8."""
    return ensemble_members([dict(checkpoints=checkpoints)], X, S2, plane7, alpha, device, bs)


def load_residual(ck, in_channels, alpha=1.0, feat_ch=0):
    """Fold checkpoint -> ResidualRefiner (with the S15F feature projection when the checkpoint has one)."""
    sd = torch.load(ck, map_location="cpu")["state"]
    m = ResidualRefiner(alpha, pretrained=False, in_channels=in_channels)
    if "fproj.weight" in sd:
        assert feat_ch in (0, sd["fproj.weight"].shape[1]), (ck, feat_ch)
        m.add_feat(sd["fproj.weight"].shape[1])
    m.load_state_dict(sd)
    return m


def ensemble_members(members, X, S2, planes, alpha=1.0, device="cpu", bs=16, log=None):
    """Equal-weight mean over refiner members of the mean over their fold models (flip TTA x4 each).

    members: [dict(checkpoints=[fold 0..K-1 paths], feats=None or callable k -> (N, C, S/4, S/4) uint8 (memmap is fine))];
    planes(k) -> (N, P, S, S) uint8 fold-matched extra planes appended to the base planes X (N, 4, S, S) for fold model k.
    One member = ensemble_residual_pmatch. Plane arrays of fold k are built once and shared by all members.
    A full-data member lists the same checkpoint for every k (nlv.refiner_members, `full: true`): it is loaded once and
    applied to each fold-k input. Mean = sum over members of the sum over k, divided by K x (number of members), the order of
    the development runs (divens/ens_test_dv.py; for one member the same value as sum / K)."""
    dev = torch.device(device)
    St = torch.from_numpy(S2)
    total = np.zeros((len(X), X.shape[2], X.shape[3]), np.float32)
    K = {len(m["checkpoints"]) for m in members}
    assert len(K) == 1, "every member needs the same number of fold models"
    K = K.pop()
    acc = [np.zeros_like(total) for _ in members]
    cache = {}  # member index -> (checkpoint path, model): a full-data checkpoint is loaded once for the five fold inputs
    for k in range(K):
        Xk = torch.from_numpy(np.concatenate([X, planes(k)], 1))
        for j, mem in enumerate(members):
            FT = mem["feats"](k) if mem.get("feats") else None
            if FT is not None:
                assert len(FT) == len(X), (len(FT), len(X))
            ck = mem["checkpoints"][k]
            if j in cache and cache[j][0] == ck:
                m = cache[j][1]
            else:
                m = load_residual(ck, Xk.shape[1] + 2, alpha, 0 if FT is None else FT.shape[1]).to(dev)
                cache.pop(j, None)
            assert (FT is None) == (getattr(m, "fproj", None) is None), "feature crops and checkpoint disagree"
            acc[j] += predict_residual(m, Xk, St, dev, bs, FT=FT)
            if log:
                log(f"[refiner] member {j} fold {k} done")
            if len(set(map(str, mem["checkpoints"]))) == 1:
                cache[j] = (ck, m)
            del m
    for a in acc:
        total += a
    total /= K * len(members)
    return total


def ensemble_stage1(checkpoints, X, device="cpu", bs=64):
    """Mean of the stage-1 fold models' TTA probabilities (test-time path of the A2 submission)."""
    dev = torch.device(device)
    Xt = torch.from_numpy(X)
    prob = np.zeros((len(X), X.shape[2], X.shape[3]), np.float32)
    for ck in checkpoints:
        m = _unet(4, False)
        m.load_state_dict(torch.load(ck, map_location="cpu")["state"])
        prob += predict_stage1(m.to(dev), Xt, dev, bs)
    return prob / len(checkpoints)


def _dice(p, t):
    inter = (p * t).sum((1, 2, 3))
    return (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + t.sum((1, 2, 3)) + 1)).mean()


def train_stage1(X, T, w, epochs=24, bs=32, lr=3e-4, max_minutes=60, init_ssl=None, log=print):
    """Plain 4-channel U-Net r34 with BCE + Dice, hard-example weighted sampling. X, T: uint8 tensors (train rows)."""
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model = _unet(4, True)
    if init_ssl:
        load_ssl_encoder(model.encoder, init_ssl)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    spe = len(X) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * spe, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    W_ = torch.tensor(w, dtype=torch.double)
    t_start = time.time()
    for ep in range(epochs):
        model.train()
        samp = torch.multinomial(W_, spe * bs, replacement=True).numpy()
        tl = 0.0
        for i in range(spe):
            j = samp[i * bs:(i + 1) * bs]
            xb = normalize(X[j].to(dev, non_blocking=True))
            tb = (T[j].to(dev, non_blocking=True).float() / 255.0).unsqueeze(1)
            xb, tb = _flip_rot_gain([xb, tb])
            with torch.autocast("cuda", dtype=torch.float16):
                lo = model(xb).float()
            loss = F.binary_cross_entropy_with_logits(lo, tb) + _dice(torch.sigmoid(lo), tb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
        log(f"[stage1] ep{ep:02d} loss {tl / spe:.4f}")
        if (time.time() - t_start) / 60 > max_minutes:  # the champion's stage-1 runs used a 60 min budget
            log("[stage1] time budget reached")
            break
    return model


def train_residual(X, T, S2, w, c, alpha=1.0, epochs=24, bs=32, lr=3e-4, init_ssl=None, log=print, FT=None, seed=-1):
    """Residual refiner: BCE + Dice + 0.5 * PQ hinge, PQ-shaped weighted sampling (see residual_weights).

    hinge = softplus(8 * (0.55 - softIoU(pred, teacher))) * c, which pushes masks over IoU 0.5 and protects
    seeds that already hit. Tensors hold the training rows only.
    FT: optional (N, C, S/4, S/4) uint8 feature crops of the same rows (S15F): adds the zero-initialised fproj; the features
    get the same flips / rotations as the planes (same random draws, so FT=None is the unchanged recipe).
    seed >= 0: torch / numpy are seeded before the model is built and the sampler stream of epoch ep is 1000 + ep + 10007 * seed
    (the --seed of the development script; the final model's members x16 / x17 / x18 use seeds 16 / 17 / 18, the members x11 / x12
    of configs/final_fullfit.yaml seeds 11 / 12). seed < 0 = unchanged recipe.
    """
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    if seed >= 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
    soff = 10007 * seed if seed >= 0 else 0
    model = ResidualRefiner(alpha, pretrained=True, in_channels=X.shape[1] + 2)
    if FT is not None:
        model.add_feat(FT.shape[1])
    if init_ssl:
        log(f"SSL init {init_ssl}: {load_ssl_encoder(model.net.encoder, init_ssl)}")
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    spe = len(X) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * spe, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    W_ = torch.tensor(w, dtype=torch.double)
    C = torch.tensor(c, dtype=torch.float32)
    for ep in range(epochs):
        model.train()
        samp = torch.multinomial(W_, spe * bs, replacement=True, generator=torch.Generator().manual_seed(1000 + ep + soff)).numpy()
        tl = 0.0
        for i in range(spe):
            j = samp[i * bs:(i + 1) * bs]
            # from here on `seed` is the binary seed-mask tensor of the batch, not the integer RNG seed (soff was derived from it above)
            x, seed = build_input(X[j].to(dev, non_blocking=True), S2[j].to(dev))
            t = (T[j].to(dev).float() / 255.0).unsqueeze(1)
            if FT is None:
                x, seed, t = _flip_rot_gain([x, seed, t])
                fb = None
            else:
                x, seed, t, fb = _flip_rot_gain([x, seed, t, fnorm(FT[j], dev)])
            with torch.autocast("cuda", dtype=torch.float16):
                lo = model(x, seed, f=fb).float()
            p = torch.sigmoid(lo)
            inter = (p * t).sum((1, 2, 3))
            siou = inter / (p + t - p * t).sum((1, 2, 3)).clamp(min=1)
            hinge = (F.softplus(8 * (0.55 - siou)) * C[j].to(dev)).mean()
            loss = F.binary_cross_entropy_with_logits(lo, t) + _dice(p, t) + 0.5 * hinge
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if sched.last_epoch + 1 < epochs * spe:
                sched.step()
            tl += loss.item()
        log(f"[residual] ep{ep:02d} loss {tl / spe:.4f}")
    return model
