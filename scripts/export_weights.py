"""NLv + NNR (configs/nlv.yaml): inference-only weights and the weight manifest (weights/MANIFEST_nlv.tsv).

  python scripts/export_weights.py manifest [--root $SOLAR_ROOT] [--out weights/MANIFEST_nlv.tsv]
      every weight file the NLv + NNR pipeline needs, with its path inside the weights folder (`weights/nlv/...`, the layout
      the demo notebook expects), size and md5 for the files present on this machine, "on server" for the others.
  python scripts/export_weights.py strip-nnunet --src <trainer dir>/fold_k/checkpoint_final.safetensors --dst <trainer dir>/fold_k/checkpoint_final.pth
      nnU-Net checkpoint -> network weights only (optimizer / grad-scaler state and logs dropped, about half the size);
      plans.json / dataset.json / dataset_fingerprint.json of the trainer folder are copied next to fold_k/.
      Works on .pth (torch pickle) and .safetensors checkpoints (nnU-Net's own loader when nnunetv2 is installed;
      the output format follows the suffix of --dst). If src and dst share the trainer folder, nothing is copied.
  python scripts/export_weights.py collect --root ... --dst <weights folder>  (files land in <weights folder>/nlv/...)
      copy the files present locally into the layout of the manifest (nothing else is downloaded or created).

The predictor (nnunetv2 nnUNetPredictor.initialize_from_trained_model_folder) reads network_weights, init_args,
trainer_name and inference_allowed_mirroring_axes from the checkpoint and plans.json / dataset.json from the folder;
these are kept. Tested on a synthetic checkpoint only (tests/test_export_weights.py).
"""
import argparse
import csv
import hashlib
import shutil
import sys
from pathlib import Path

KEEP_KEYS = ("network_weights", "init_args", "trainer_name", "inference_allowed_mirroring_axes")
DROP_PREFIXES = ("optimizer", "grad_scaler", "logging", "_best_ema", "current_epoch")
FOLDER_FILES = ("plans.json", "dataset.json", "dataset_fingerprint.json")
NN_PLANS = "nnUNetResEncUNetLPlans__2d"
# nnU-Net folds of the submitted run (training length differs per fold; README "Minimal variant (NLv + NNR)")
NN_FOLDS = {0: ("nnUNetTrainer", "checkpoint_final.pth", "1000 epochs"),
            1: ("nnUNetTrainer", "checkpoint_latest.pth", "checkpoint_latest of the 1000-epoch run at epoch 373"),
            2: ("nnUNetTrainer_250epochs", "checkpoint_final.pth", "250 epochs"),
            3: ("nnUNetTrainer_250epochs", "checkpoint_final.pth", "250 epochs"),
            4: ("nnUNetTrainer_250epochs", "checkpoint_final.pth", "250 epochs")}


# ----------------------------------------------------------------------------- nnU-Net stripping

def strip_nnunet(src: Path, dst: Path) -> dict:
    """Write the inference-only checkpoint dst from src; returns {kept, dropped, bytes_in, bytes_out}."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:  # nnU-Net master (288572b+) writes .safetensors with a JSON state tensor; its loader reads both formats
        from nnunetv2.utilities.checkpoints import load_checkpoint_file
    except ImportError:
        load_checkpoint_file = None
    if load_checkpoint_file is not None:
        import torch
        ck = load_checkpoint_file(str(src), map_location="cpu", keys=KEEP_KEYS) if src.suffix == ".safetensors" else torch.load(src, map_location="cpu", weights_only=False)
        out = {k: ck[k] for k in KEEP_KEYS if k in ck}
        assert "network_weights" in out, f"not an nnU-Net checkpoint: {src}"
        if dst.suffix == ".safetensors":
            from nnunetv2.utilities.checkpoints import save_checkpoint_file
            save_checkpoint_file(out, str(dst))
        else:
            torch.save(out, dst)
        kept, dropped = sorted(out), sorted(set(ck) - set(out))
    elif src.suffix == ".safetensors":  # plain safetensors state (no nnU-Net installed): drop the training-state tensors
        from safetensors import safe_open
        from safetensors.torch import save_file
        with safe_open(str(src), framework="pt") as f:
            meta = f.metadata() or {}
            keys = list(f.keys())
            keep = {k: f.get_tensor(k) for k in keys if not k.startswith(DROP_PREFIXES)}
        save_file(keep, str(dst), metadata=meta)
        kept, dropped = sorted(keep), sorted(set(keys) - set(keep))
    else:
        import torch
        ck = torch.load(src, map_location="cpu", weights_only=False)
        assert "network_weights" in ck, f"not an nnU-Net checkpoint: {src}"
        out = {k: ck[k] for k in KEEP_KEYS if k in ck}
        torch.save(out, dst)
        kept, dropped = sorted(out), sorted(set(ck) - set(out))
    for name in FOLDER_FILES:  # trainer folder = parent of fold_k/
        f = src.parent.parent / name
        if f.exists() and f.resolve() != (dst.parent.parent / name).resolve():
            shutil.copy2(f, dst.parent.parent / name)
    return dict(kept=kept, dropped=dropped, bytes_in=src.stat().st_size, bytes_out=dst.stat().st_size)


# ----------------------------------------------------------------------------- manifest

def entries(root: Path, repo: Path) -> list:
    """(component, file in weights/, role, local source or None, source note, licence)."""
    E = []
    for k, (tr, ck, note) in NN_FOLDS.items():
        d = f"nlv/nnunet/Dataset101_Solar/{tr}__{NN_PLANS}"
        E.append(("nnU-Net ResEnc-L 2D (candidates 1, NNR plane 7)", f"{d}/fold_{k}/checkpoint_final.pth", f"inference, fold {k}", None,
                  f"{ck}, {note}; stripped with strip-nnunet", "Apache-2.0 (nnU-Net code); weights ours"))
    for tr in sorted({v[0] for v in NN_FOLDS.values()}):
        for f in FOLDER_FILES:
            E.append(("nnU-Net plans", f"nlv/nnunet/Dataset101_Solar/{tr}__{NN_PLANS}/{f}", "inference", None, "trainer folder of the server", "ours"))
    for k in range(5):
        E.append(("Mask DINO Swin-L (candidates 2)", f"nlv/maskdino_swinl_f{k}.pt", f"inference, fold {k}", None,
                  "Mask DINO Swin-L, COCO init, 1024, 40 epochs", "Apache-2.0 (Mask DINO, COCO weights)"))
    for s in range(3):
        for k in range(5):
            E.append((f"Mask DINO R50 seed {s} (votes)", f"nlv/maskdino_r50_s{s}_f{k}.pt", f"inference, fold {k}", None,
                      "Mask DINO R50, COCO init, 1024, 40 epochs", "Apache-2.0 (Mask DINO, COCO weights)"))
    for fam in ("r34", "eff"):
        for k in range(5):
            E.append((f"own2 map U-Net {fam} (map features)", f"nlv/own2_{fam}_f{k}.pt", f"inference, fold {k}", None,
                      "own-maps, ImageNet init", "MIT (ours); smp MIT, timm Apache-2.0"))
    E.append(("U-Net++ seg_v1 (p-features, candidate conf, refiner plane 1)", "seg_v1.pt", "inference (test, hold-out model)",
              repo / "weights" / "seg_v1.pt", "U-Net++ hold-out model of the base pipeline", "MIT"))
    for k in range(5):
        E.append(("U-Net++ seg_v1 fold models (OOF maps of the training frames)", f"nlv/train/seg_v1f{k}.pt", f"training only, fold {k}",
                  root / "outputs" / "evac_v1" / f"seg_v1f{k}.pt", "outputs/evac_v1", "MIT"))
    for k in range(5):
        E.append(("NNR refiner (U-Net r34, 7th plane = nnU-Net map)", f"nlv/refiner_V2bSCrdmNNR_f{k}.pt", f"inference, fold {k}", None,
                  "residual refiner NNR, GONG SimMIM init, 24 epochs", "MIT (ours)"))
    E.append(("GONG SimMIM encoder (NNR initialisation)", "nlv/train/ssl_mimC_r34_enc.pt", "training only",
              root / "outputs" / "ssl_mimC_r34_enc.pt", "outputs/", "MIT (ours); GONG data"))
    for m, ext in (("gbdt", "joblib"), ("cat", "cbm"), ("xgb", "json"), ("et", "joblib")):
        E.append((f"V-ENS member {m}", f"nlv/verifier_nlv_{m}.{ext}", "inference", root / "outputs" / "nlv_weights" / f"verifier_nlv_{m}.{ext}",
                  "final-cpu --save-weights (refit = submitted scores, max abs diff 0)", "MIT (ours)"))
    return E


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(root: Path, repo: Path, out: Path) -> list:
    rows = []
    for comp, file, role, src, note, lic in entries(root, repo):
        ok = src is not None and Path(src).exists()
        rows.append(dict(component=comp, file=file, role=role, size_bytes=Path(src).stat().st_size if ok else "",
                         md5=md5(src) if ok else "", status="local" if ok else "on server", source=note, licence=lic))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
    return rows


def collect(root: Path, repo: Path, dst: Path) -> int:
    n = 0
    for _, file, _, src, _, _ in entries(root, repo):
        if src is not None and Path(src).exists() and file.startswith("nlv/"):
            t = Path(dst) / file
            t.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, t); n += 1
    return n


def main(argv=None):
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("manifest"); s.add_argument("--root", type=Path, default=repo.parents[1]); s.add_argument("--out", type=Path, default=repo / "weights" / "MANIFEST_nlv.tsv")
    s = sub.add_parser("strip-nnunet"); s.add_argument("--src", type=Path, required=True); s.add_argument("--dst", type=Path, required=True)
    s = sub.add_parser("collect"); s.add_argument("--root", type=Path, default=repo.parents[1]); s.add_argument("--dst", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "manifest":
        rows = manifest(a.root, repo, a.out)
        print(f"{a.out}: {len(rows)} files, {sum(r['status'] != 'on server' for r in rows)} local")
    elif a.cmd == "strip-nnunet":
        print(strip_nnunet(a.src, a.dst))
    else:
        print(collect(a.root, repo, a.dst), "files copied")


if __name__ == "__main__":
    sys.exit(main())
