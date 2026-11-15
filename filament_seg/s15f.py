"""S15F refiner features (configs/s15f.yaml, included by configs/stk.yaml and, through it, by the final model): the pixel-decoder mask_features of the
Mask DINO Swin-L 1536 fold models, mapped into one common 16-channel space and cropped with the refiner windows.

1. Hook. MaskDINO's pixel decoder returns (mask_features, ...) from `forward_features`; the hook keeps mask_features of the
   last forward pass: 256 channels at stride 4 of the 1536 input = 384 x 384 for a 2048 frame (`attach_hook`).
2. Common coordinates (`fit_alignment`, CPU). The five fold models are different networks, so their 256 channels are not
   comparable. A reference set of 125 training frames (25 drawn from every fold's validation list, numpy default_rng(0)) is
   run through all five models; at 3,000 random pixels per frame each model's features are standardised (mu_k, sd_k), the
   five standardised vectors are concatenated (1,280 dims) and their first 16 principal components (unit variance) are the
   common target T. Per model a ridge regression W_k, b_k (256 -> 16, lambda = 0.01 x samples, bias not penalised) maps
   its standardised features onto T. No ground truth or annotation is used. Saved as s15f_align.npz (W, b, mu, sd, R2).
3. Feature crops (`common_features` + `crop_features`). z_k = W_k^T (mf - mu_k) / sd_k + b_k with the fp16 roundings of the
   submitted run (mask_features stored as fp16, z stored as fp16), then torchvision roi_align of every refiner window
   (x0, y0, side in 2048 coordinates) to 64 x 64 (spatial_scale 384 / 2048, sampling_ratio 2, aligned; outside the frame =
   0 = the mean of the common space), stored as uint8 round(z * 32 + 128).
   Training rows (refiner_meta.parquet of stage `crops`) use the fold model that did not see the frame (asserted against the
   refiner fold); the refiner of fold k reads, at test time, the features of the fold-k Swin-L 1536 model.
The refiner side (zero-initialised 1x1 projection onto the stride-4 encoder stage) is in filament_seg/refiner.py.

Port of the development scripts refevo/s15feat/s15_feats.py (modes pk / fit / oof / crops) and s15_test_feats.py. The
development run stored the raw 256-channel features of every validation frame between the modes; here they are mapped
and cropped in the same pass (same values: the raw features were stored as fp16, which `common_features` reproduces).
GPU for the Mask DINO forward passes (about 0.3 s per frame on a recent GPU); `fit_alignment` and the crops are cheap.
"""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import folds, io, maskdino_cands, refiner

SIZE = 1536          # Mask DINO input side of the Swin-L 1536 models
FS = SIZE // 4       # mask_features side (stride 4)
NREF_PER_FOLD = 25   # reference frames per fold validation list
NPIX = 3000          # sampled pixels per reference frame
NPC = 16             # common channels
CROP = 64            # feature crop side (= refiner crop 256 / 4)
ART = dict(align="s15f_align.npz", ref="s15f_ref_f{k}.npy", ref_names="s15f_ref_names.npy", train="s15f_feat_crops.npy",
           test="s15f_test_feat_f{k}.npy", test_boxes="s15f_test_boxes.npy")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- model + hook

def attach_hook(model) -> dict:
    """Wrap pixel_decoder.forward_features so that box["mf"] holds mask_features (B, 256, FS, FS) of the last forward."""
    pd_ = model.sem_seg_head.pixel_decoder
    orig = pd_.forward_features
    box = {}

    def ff(*args, **kw):
        r = orig(*args, **kw)
        box["mf"] = r[0].detach()
        return r
    pd_.forward_features = ff
    return box


def load_model(maskdino_root: Path, weights: Path, device="cuda"):
    """Swin-L Mask DINO fold model (maskdino_cands config: one class, 100 queries, 50 DN, mean / std 127.5 / 64) + hook."""
    model = maskdino_cands.load(Path(maskdino_root), Path(weights), device, arch="swinl")
    return model, attach_hook(model)


def raw_features(model, box, img: np.ndarray):
    """2048 gray frame -> (mask_features (256, FS, FS) float tensor on the model device, instances on the CPU)."""
    import torch
    with torch.no_grad():
        o = model([{"image": torch.from_numpy(maskdino_cands.prep(img, SIZE)), "height": 2048, "width": 2048}])[0]["instances"].to("cpu")
    mf = box["mf"][0].float()
    assert mf.shape == (256, FS, FS), mf.shape
    return mf, o


# ----------------------------------------------------------------------------- alignment

def reference_set(val_lists: dict) -> tuple:
    """(sorted reference names, {name: (NPIX, 2) pixel (y, x)}) from the five fold validation lists; numpy default_rng(0)."""
    rng = np.random.default_rng(0)
    ref = sorted(n for k in range(5) for n in sorted(rng.choice(sorted(val_lists[k]), NREF_PER_FOLD, replace=False)))
    pix = {n: rng.integers(0, FS, (NPIX, 2)) for n in ref}
    return ref, pix


def reference_sample(model, box, images: dict, ref: list, pix: dict) -> np.ndarray:
    """(len(ref) * NPIX, 256) fp16 features of one model at the reference pixels."""
    out = np.zeros((len(ref) * NPIX, 256), np.float16)
    for j, n in enumerate(ref):
        mf, _ = raw_features(model, box, images[n])
        yy, xx = pix[n][:, 0], pix[n][:, 1]
        out[j * NPIX:(j + 1) * NPIX] = mf[:, yy, xx].T.cpu().numpy()
    return out


def fit_alignment(F: np.ndarray, log=log) -> dict:
    """F: (K, M, 256) reference features of the K fold models at the same M pixels -> dict(W (K, 256, 16), b (K, 16),
    mu (K, 256), sd (K, 256), R2 (K, 16)). PCA16 of the concatenated standardised features = common target; ridge per model."""
    F = F.astype(np.float32)
    K, M, C = F.shape
    mu = F.mean(1, keepdims=True)
    sd = F.std(1, keepdims=True) + 1e-6
    Z = ((F - mu) / sd).transpose(1, 0, 2).reshape(M, K * C)
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc[::4], full_matrices=False)
    T = Zc @ Vt[:NPC].T
    T = T / T.std(0)
    log(f"[s15f] PCA{NPC} explained variance (of {K * C} dims) {float((S[:NPC] ** 2).sum() / (S ** 2).sum()):.3f}")
    W = np.zeros((K, C, NPC), np.float32)
    b = np.zeros((K, NPC), np.float32)
    R2 = []
    for k in range(K):
        A = (F[k] - mu[k]) / sd[k]
        A1 = np.concatenate([A, np.ones((M, 1), np.float32)], 1)
        lam = 1e-2 * M
        G = A1.T @ A1 + lam * np.eye(C + 1, dtype=np.float32)
        G[-1, -1] -= lam
        Wk = np.linalg.solve(G, A1.T @ T)
        W[k], b[k] = Wk[:-1], Wk[-1]
        pred = A1 @ Wk
        r2 = 1 - ((pred - T) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0)
        R2.append(r2)
        log(f"[s15f] fold model {k}: R2 mean {r2.mean():.3f} (min {r2.min():.3f})")
    P = np.stack([((F[k] - mu[k]) / sd[k]) @ W[k] + b[k] for k in range(K)])
    cc = [np.mean([np.corrcoef(P[i][:, c], P[j][:, c])[0, 1] for c in range(NPC)]) for i in range(K) for j in range(i + 1, K)]
    log(f"[s15f] correlation of the {NPC} common channels between models: mean {np.mean(cc):.3f}, min {np.min(cc):.3f}")
    return dict(W=W, b=b, mu=mu[:, 0], sd=sd[:, 0], R2=np.array(R2))


def common_features(mf, al, k: int):
    """mask_features (256, FS, FS) of fold model k -> (16, FS, FS) common features (torch, device of mf), with the fp16
    roundings of the submitted run (raw features and z were stored as fp16)."""
    import torch
    dev = mf.device
    W, b, mu, sd = (torch.from_numpy(np.asarray(al[n][k], np.float32)).to(dev) for n in ("W", "b", "mu", "sd"))
    z = torch.einsum("chw,cd->dhw", (mf.half().float() - mu[:, None, None]) / sd[:, None, None], W) + b[:, None, None]
    return z.half().float()


def crop_features(z, boxes) -> np.ndarray:
    """z (16, FS, FS) torch; boxes (n, 3) refiner windows (x0, y0, side) in 2048 coordinates -> (n, 16, 64, 64) uint8."""
    import torch
    from torchvision.ops import roi_align
    boxes = np.asarray(boxes)
    bb = torch.tensor(np.stack([np.zeros(len(boxes)), boxes[:, 0], boxes[:, 1], boxes[:, 0] + boxes[:, 2], boxes[:, 1] + boxes[:, 2]], 1),
                      dtype=torch.float32).to(z.device)
    r = roi_align(z[None], bb, output_size=CROP, spatial_scale=FS / 2048, sampling_ratio=2, aligned=True)
    return torch.clamp(torch.round(r * 32 + 128), 0, 255).to(torch.uint8).cpu().numpy()


def test_boxes(df: pd.DataFrame, scale=2.0, min_side=128) -> np.ndarray:
    """Refiner windows of the test candidate table (row order), as refiner.make_crops computes them."""
    return np.array([refiner.crop_box(int(r.x), int(r.y0), int(r.w), int(r.h), scale, min_side) for r in df.itertuples()], np.int64)


def frame_crops(model, box, al, k: int, img: np.ndarray, boxes) -> np.ndarray:
    """One frame through fold model k -> (n, 16, 64, 64) uint8 feature crops of its n refiner windows."""
    mf, _ = raw_features(model, box, img)
    return crop_features(common_features(mf, al, k), boxes)


def test_crops(model, box, al, k: int, df: pd.DataFrame, boxes, read) -> np.ndarray:
    """All test candidates (df row order, windows `boxes`) through fold model k; read(stem) -> 2048 gray frame."""
    X = np.zeros((len(df), NPC, CROP, CROP), np.uint8)
    for n in dict.fromkeys(df.image):
        g = df.index[df.image == n].values
        X[g] = frame_crops(model, box, al, k, read(n), boxes[g])
    return X


# ----------------------------------------------------------------------------- stages

def _settings(cfg) -> dict:
    from . import nlv
    s = dict(cfg.get("s15f", {}))
    os.environ.setdefault("SOLAR_WEIGHTS", str(Path(__file__).resolve().parents[1] / "weights"))
    s = nlv.resolve({"nlv": s})
    s.pop("voters", None)
    return s


def _weights(cfg, k: int) -> Path:
    s = _settings(cfg)
    return Path(s["swinl1536_weights"].format(k=k))


def _val_lists(cfg) -> dict:
    return {k: folds.load_split(cfg["paths"].out, f"split_f{k}.json")["val"] for k in range(5)}


def _read_train(cfg, n):
    return io.read_gray(cfg["paths"].train_images / f"{n}.jpeg")


def stage_ref(cfg, fold: int, device="cuda"):
    """GPU: reference sample of Swin-L 1536 fold model `fold` -> s15f_ref_f{fold}.npy (+ s15f_ref_names.npy)."""
    out = cfg["paths"].out; t0 = time.time()
    ref, pix = reference_set(_val_lists(cfg))
    model, box = load_model(Path(_settings(cfg)["maskdino_root"]), _weights(cfg, fold), device)
    F = reference_sample(model, box, {n: _read_train(cfg, n) for n in ref}, ref, pix)
    np.save(out / ART["ref"].format(k=fold), F)
    np.save(out / ART["ref_names"], np.array(ref))
    log(f"[s15f] reference sample fold model {fold}: {F.shape} ({time.time() - t0:.0f}s)")


def stage_fit(cfg):
    """CPU: s15f_align.npz from the five reference samples."""
    out = cfg["paths"].out
    al = fit_alignment(np.stack([np.load(out / ART["ref"].format(k=k)) for k in range(5)]))
    np.savez(out / ART["align"], **al)
    log(f"[s15f] wrote {out / ART['align']}")


def stage_train_crops(cfg, folds_=(0, 1, 2, 3, 4), device="cuda"):
    """GPU: training feature crops s15f_feat_crops.npy (N, 16, 64, 64) uint8 in the row order of refiner_meta.parquet; the
    rows of the validation frames of fold k come from fold model k (OOF). Folds can be run one at a time (the file is
    created on the first call and filled per fold)."""
    out = cfg["paths"].out; t0 = time.time()
    meta = pd.read_parquet(out / "refiner_meta.parquet")
    vl = _val_lists(cfg)
    fold_of = {n: k for k, v in vl.items() for n in v}
    bad = [n for n, g in meta.groupby("image") if fold_of[n] != int(g.fold.iloc[0])]
    assert not bad, ("refiner fold != Swin-L 1536 fold", len(bad))
    al = dict(np.load(out / ART["align"]))
    dst = out / ART["train"]
    X = np.lib.format.open_memmap(dst, mode="r+" if dst.exists() else "w+", dtype=np.uint8, shape=(len(meta), NPC, CROP, CROP))
    for k in folds_:
        model, box = load_model(Path(_settings(cfg)["maskdino_root"]), _weights(cfg, k), device)
        for n, g in meta[meta.fold == k].groupby("image", sort=False):
            X[g.index.values] = frame_crops(model, box, al, k, _read_train(cfg, n), g[["x0", "y0", "side"]].values)
        X.flush()
        del model
        log(f"[s15f] training crops of fold {k} ({time.time() - t0:.0f}s)")
    del X


def stage_test(cfg, folds_=(0, 1, 2, 3, 4), device="cuda", limit=0):
    """GPU: test feature crops s15f_test_feat_f{k}.npy (N, 16, 64, 64) uint8 of fold model k for every test candidate
    (nlv.test_cands row order) and the windows s15f_test_boxes.npy. limit = only the first frames (smoke test; rows of the
    other frames stay 0)."""
    from . import nlv
    out = cfg["paths"].out; t0 = time.time(); c = cfg["refiner"]
    df = pd.read_parquet(nlv.resolve(cfg)["test_cands"]).reset_index(drop=True)
    box = test_boxes(df, c["crop_scale"], c["crop_min_side"])
    bf = out / ART["test_boxes"]
    if bf.exists():
        assert (np.load(bf) == box).all(), "test windows changed"
    else:
        np.save(bf, box)
    af = out / ART["align"]   # written by s15f-fit; otherwise the released file (weights/MANIFEST_final.tsv: final/s15f_align.npz)
    if not af.exists():
        af = Path(os.path.expandvars("${SOLAR_WEIGHTS}")) / "final" / ART["align"]
    al = dict(np.load(af))
    names = list(dict.fromkeys(df.image.tolist()))
    if limit:
        names = names[:limit]
    for k in folds_:
        model, bx = load_model(Path(_settings(cfg)["maskdino_root"]), _weights(cfg, k), device)
        dst = out / ART["test"].format(k=k)
        X = np.lib.format.open_memmap(str(dst) + ".tmp.npy", mode="w+", dtype=np.uint8, shape=(len(df), NPC, CROP, CROP))
        for n in names:
            g = df.index[df.image == n].values
            X[g] = frame_crops(model, bx, al, k, io.read_gray(cfg["paths"].test_images / f"{n}.jpeg"), box[g])
        X.flush()
        del X, model
        os.replace(str(dst) + ".tmp.npy", dst)
        log(f"[s15f] test crops of fold model {k}: {len(names)} frames ({time.time() - t0:.0f}s)")
    return json.dumps(dict(rows=len(df), frames=len(names)))


# ----------------------------------------------------------------------------- refiner inputs

class RowView:
    """Lazy row subset of a (memory-mapped) array: v[j] = a[rows[j]] (read on access; used for the training feature crops)."""

    def __init__(self, a, rows):
        self.a, self.rows = a, np.asarray(rows)
        self.shape = (len(self.rows),) + tuple(a.shape[1:])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, j):
        return np.asarray(self.a[self.rows[j]])


def load_feats(path, rows=None, lazy=False):
    """Feature crops (N, C, 64, 64) uint8: in RAM (default, as the development run) or memory-mapped (lazy=True)."""
    a = np.load(path, mmap_mode="r")
    if rows is None:
        return a if lazy else np.ascontiguousarray(a)
    return RowView(a, rows) if lazy else np.ascontiguousarray(a[np.asarray(rows)])
