"""Writes notebooks/demo_nlv.ipynb (the demo notebook of the minimal variant NLv + NNR). Edit the cells here, then
  python scripts/make_demo_nlv.py && jupyter nbconvert --to notebook --execute --inplace notebooks/demo_nlv.ipynb
"""
import argparse
from pathlib import Path

import nbformat as nbf

argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()  # --help only

REPO = Path(__file__).resolve().parents[1]
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


md("""
# Solar filament segmentation: minimal variant NLv + NNR (demo)

One notebook from the test frames to `submission.csv`, with the trained weights loaded automatically:
preprocessing -> candidates (nnU-Net, Mask DINO Swin-L) -> votes (Mask DINO R50 x 3) -> map features (own2 U-Nets)
-> verifier V-ENS -> refiner NNR -> decode -> CSV -> self-evaluation.

* **GPU stages** (U-Net++, nnU-Net, Mask DINO, own2 maps, NNR) need a CUDA GPU (measured on the submitted run: about
  3.5 h for the 180 test frames on one GTX 1080 Ti / RTX 3080 Ti class GPU, mostly nnU-Net; see each cell). Mask DINO Swin-L
  needs about 32 GB for training, less for inference. Without a GPU they are skipped and the notebook uses the cached outputs of the
  submitted run (`CACHE`), so the CPU part (verifier, decode, evaluation) still runs end to end.
* **Training** is optional: `TRAIN_FROM_SCRATCH = False` (default) loads the trained models;
  `True` refits the verifier here and prints the training command of every other stage with its time.
* Numbers of NLv + NNR: OOF rec PQ **0.4701**, simulated test PQ **0.4116** (707 training frames, tau 0.28),
  public LB **0.40**; the submitted CSV has md5 `9eb43f91f77263f7a38e3a8c0dce8ba2`.
""")

code("""
# 1. Setup: repository, packages, devices
import os, sys, json, shutil, subprocess, time, hashlib
from pathlib import Path

REPO_URL = "https://github.com/0122POND/solar-filament-seg"   # public after the competition deadline
INSTALL = False            # True on a fresh machine (Kaggle / Colab): installs the pinned packages below
here = Path.cwd()
REPO = next((p for p in (here, here.parent) if (p / "filament_seg").exists()), None)
if REPO is None:
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "solar-filament-seg"], check=True)
    REPO = Path("solar-filament-seg").resolve()
if INSTALL:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")], check=True)
    # GPU stages only: nnU-Net v2 (GitHub master 288572b) and Mask DINO (detectron2 247691a, MaskDINO 3831d85, see filament_seg/maskdino_cands.py)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "git+https://github.com/MIC-DKFZ/nnUNet.git@288572b"], check=True)
sys.path.insert(0, str(REPO))
os.chdir(REPO)
HAS_GPU = shutil.which("nvidia-smi") is not None and subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
RUN_GPU_STAGES = HAS_GPU   # set False to use the cached model outputs even on a GPU machine
TRAIN_FROM_SCRATCH = False
CACHE = Path(os.environ.get("SOLAR_ROOT", REPO.parents[1]))   # development tree with the cached outputs of the submitted run
os.environ["SOLAR_ROOT"] = str(CACHE)
DATA = Path(os.environ.get("SOLAR_DATA", CACHE / "data" / "MAGFiLO_1.0_Kaggle_2026"))
os.environ["SOLAR_DATA"] = str(DATA)
WORK = REPO / "out" / "demo_nlv"; WORK.mkdir(parents=True, exist_ok=True)
print("repo", REPO.name, "| GPU", HAS_GPU, "| GPU stages", RUN_GPU_STAGES, "| cache", (CACHE / "outputs" / "mdtest_nl").exists(), "| data", DATA.exists())
""")

code("""
# 2. Trained weights: downloaded automatically (one variable), then checked against weights/MANIFEST_nlv.tsv
WEIGHTS_DATASET = "deepernet/solar-filament-nlv-weights"   # Kaggle Dataset (public after the deadline); or set WEIGHTS_URL
WEIGHTS_URL = ""                                           # alternative: public URL of a zip with the same layout
WEIGHTS = Path(os.environ.get("SOLAR_WEIGHTS", REPO / "weights"))
import csv
manifest = list(csv.DictReader(open(REPO / "weights" / "MANIFEST_nlv.tsv"), delimiter="\\t"))
need = [r for r in manifest if r["role"].startswith("inference")]
missing = [r["file"] for r in need if not (WEIGHTS / r["file"]).exists()]
if missing and RUN_GPU_STAGES:
    if WEIGHTS_URL:
        import urllib.request, zipfile
        z = WORK / "weights.zip"; urllib.request.urlretrieve(WEIGHTS_URL, z); zipfile.ZipFile(z).extractall(WEIGHTS)
    else:
        subprocess.run(["kaggle", "datasets", "download", "-d", WEIGHTS_DATASET, "-p", str(WEIGHTS), "--unzip"], check=True)
    missing = [r["file"] for r in need if not (WEIGHTS / r["file"]).exists()]
# V-ENS: the weights folder, else the members saved from the submitted run (final-cpu --save-weights)
VENS = WEIGHTS / "nlv" if (WEIGHTS / "nlv" / "verifier_nlv_gbdt.joblib").exists() else CACHE / "outputs" / "nlv_weights"
bad = [r["file"] for r in need if r["md5"] and (WEIGHTS / r["file"]).exists() and hashlib.md5((WEIGHTS / r["file"]).read_bytes()).hexdigest() != r["md5"]]
print(f"inference files {len(need)}, present {len(need) - len(missing)}, md5 mismatches {len(bad)}; V-ENS from {VENS.relative_to(CACHE) if VENS.is_relative_to(CACHE) else VENS}")
if missing and not RUN_GPU_STAGES:
    print("no GPU stages: the missing model weights are not needed, the cached model outputs of the submitted run are used")
""")

md("""
## 3. Preprocessing
Test frames: 2048 x 2048 grey JPEG, sorted by name (`test_names`; frame i uses fold model i % 5 everywhere).
U-Net++ (`seg_v1.pt`, hold-out model) gives the 1024-px probability used by the candidate features, the candidate
confidence and the refiner's second plane (CPU with 4 workers: about 1,500 s for the 180 frames; GPU: minutes).
""")

code("""
import numpy as np, pandas as pd, pickle
from filament_seg import config, io, nlv
cfg = config.load(REPO / "configs" / "nlv.yaml")
C = nlv.resolve(cfg)
files = io.list_images(DATA / "test" / "test_images") if (DATA / "test" / "test_images").exists() else []
test_names = sorted(f.stem for f in files) or [str(x) for x in np.load(C["test_names"], allow_pickle=True)]
assert test_names == sorted(test_names)
print(len(test_names), "test frames, e.g.", test_names[:2])
if RUN_GPU_STAGES:
    import torch
    from filament_seg import preprocess, semantic
    dev = torch.device("cuda")
    m = semantic.build_model("resnet34", pretrained=False); m.load_state_dict(torch.load(WEIGHTS / "seg_v1.pt", map_location="cpu"))
    m = m.to(dev)
    V1 = {f.stem: semantic.predict(m, preprocess.make_input(io.read_gray(f), 1024)[None], dev, amp=False)[0].astype(np.float16) for f in files}
    print("U-Net++ maps", len(V1))
""")

md("""
## 4. Candidates: nnU-Net components U Mask DINO Swin-L
* nnU-Net ResEnc-L 2D, one model per fold, mirroring TTA; class-1 probability -> uint8 map (measured: about 29 min per
  fold model for the 180 frames on a GTX 1080 Ti, i.e. about 2.4 h for 5 folds). Components at 0.3 / 0.4, area >= 50.
* Mask DINO Swin-L (1024 input), fold model i % 5, instances with score >= 0.05 and >= 30 px (measured: 6-8 min per fold
  model for 180 frames; here each fold model only predicts its 36 frames).
* Candidate features FEATS_ALL (48) with the U-Net++ map.
""")

code("""
if RUN_GPU_STAGES:
    from filament_seg import maskdino_cands, nnunet_cands
    MDROOT = Path(os.environ.get("MASKDINO_ROOT", "MaskDINO"))
    nn_in = WORK / "nn_in"; nn_in.mkdir(exist_ok=True)
    for f in files:
        cv2_img = io.read_gray(f); import cv2; cv2.imwrite(str(nn_in / f"{f.stem}_0000.png"), cv2_img)
    for k in range(5):
        tr = [r["file"] for r in manifest if r["file"].endswith(f"fold_{k}/checkpoint_final.pth")][0]
        nnunet_cands.predict_maps(WEIGHTS / Path(tr).parent.parent, k, nn_in, WORK / "nnunet_test_maps" / f"f{k}", checkpoint="checkpoint_final.pth")
    SWL = [maskdino_cands.load(MDROOT, WEIGHTS / f"nlv/maskdino_swinl_f{k}.pt", "cuda", arch="swinl") for k in range(5)]
    rows, rles = [], {}
    for i, f in enumerate(sorted(files, key=lambda f: f.stem)):
        n = f.stem; raw = io.read_gray(f)
        md_items = maskdino_cands.predict(SWL[i % 5], raw)
        nn_map = nnunet_cands.load_map(WORK / "nnunet_test_maps" / f"f{i % 5}", n)
        r, rl = nlv.candidates_image(n, V1[n].astype(np.float32), raw, nn_map, md_items, [], cfg)
        rows += r; rles[n] = rl
    pd.DataFrame(rows).to_parquet(WORK / "test_cands.parquet"); pickle.dump(rles, open(WORK / "test_rles.pkl", "wb"))
    np.save(WORK / "test_names.npy", np.array(test_names))
    C.update(test_cands=str(WORK / "test_cands.parquet"), test_rles=str(WORK / "test_rles.pkl"), test_names=str(WORK / "test_names.npy"),
             test_nnmaps=str(WORK / "nnunet_test_maps" / "f{k}"))
T0 = pd.read_parquet(C["test_cands"])
print(f"test candidates {len(T0)} on {T0.image.nunique()} frames; sources {T0.source.value_counts().to_dict()}  ('yolo' = Mask DINO slot)")
""")

md("""
## 5. Votes: Mask DINO R50 seeds 0 / 1 / 2 (8 features)
The R50 models are not candidates; for each candidate: max IoU with each voter's instances, that instance's score,
and the number of voters with IoU >= 0.5 (GPU: about 1.6 s per frame and model; 3 x 180 frame predictions, about 15 min).
""")

code("""
from filament_seg import votes
if RUN_GPU_STAGES:
    for s in range(3):
        R50 = [maskdino_cands.load(MDROOT, WEIGHTS / f"nlv/maskdino_r50_s{s}_f{k}.pt", "cuda", arch="r50") for k in range(5)]
        for k in range(5):
            mine = [f for i, f in enumerate(sorted(files, key=lambda f: f.stem)) if i % 5 == k]
            pickle.dump({f.stem: maskdino_cands.predict(R50[k], io.read_gray(f)) for f in mine}, open(WORK / f"r50_s{s}_f{k}.pkl", "wb"))
        C["test_voters"]["s%d" % s] = str(WORK / f"r50_s{s}_f{{k}}.pkl")
TR = pickle.load(open(C["test_rles"], "rb"))
P = {k: [pickle.load(open(os.path.expandvars(C["test_voters"][k]).format(k=f), "rb")) for f in range(5)] for k in votes.VOTERS}
VT = votes.test_votes(T0, TR, test_names, P)
print(VT[votes.VFEATS].describe().loc[["mean", "max"]].round(3).to_string())
""")

md("""
## 6. Map features: own2 U-Nets (r34 + EfficientNetV2-S), 14 features per candidate and per fold map
Each fold model writes one map per test frame (D4 TTA x 8; 10 models x 180 frames, not timed here). The verifier
is applied once per fold map and the five scores are averaged (computed inside `final-cpu` below).
""")

code("""
if RUN_GPU_STAGES:
    from filament_seg import ownmap
    imgs = {f.stem: io.read_gray(f) for f in files}
    for k in range(5):
        for fam in ("r34", "eff"):
            mdl = ownmap.load_model(WEIGHTS / f"nlv/own2_{fam}_f{k}.pt", fam, torch.device("cuda"))
            ownmap.write_maps(mdl, fam, imgs, test_names, WORK / "own2_test" / f"f{k}", torch.device("cuda"))
    C["test_maps"] = str(WORK / "own2_test" / "f{k}")
from filament_seg import mapfeats
n0 = test_names[0]; g0 = T0[T0.image == n0]
f0 = pd.DataFrame(mapfeats.image_features(Path(C["test_maps"].format(k=0)), n0, TR[n0], g0.idx.astype(int).tolist(), fast=True))
print(f"{n0}: {len(f0)} candidates, map features of fold map 0:"); print(f0[mapfeats.DFEATS[:6]].head(3).round(3).to_string())
""")

md("""
## 7. Refiner NNR (residual U-Net r34, 7th plane = nnU-Net map of the matching fold)
Every candidate is redrawn in a 256-px crop; fold model k reads the nnU-Net map of fold k and the five probabilities
are averaged (flip TTA x 4). Measured: 22 min for the 10,189 test candidates on a GTX 1080 Ti (5 fold models).
The GPU branch of this cell (`nlv.refine_test`) was checked on 6 test frames (byte-identical masks), not on all 180.
""")

code("""
if RUN_GPU_STAGES:  # not yet run end to end (the submitted NNR masks come from the development script on the server)
    raws = {f.stem: io.read_gray(f) for f in files}
    cks = [WEIGHTS / f"nlv/refiner_V2bSCrdmNNR_f{k}.pt" for k in range(5)]
    REFD = nlv.refine_test(cfg, T0, TR, {n: V1[n].astype(np.float32) for n in V1}, raws, cks, "cuda")
    pickle.dump(REFD, open(WORK / "refiner_rles.pkl", "wb")); C["refiner_rles"] = str(WORK / "refiner_rles.pkl")
REF = pickle.load(open(C["refiner_rles"], "rb"))
print("refined masks for", len(REF), "frames,", sum(len(v) for v in REF.values()), "candidates")
""")

md("""
## 8. Verifier V-ENS + decode -> submission.csv
V-ENS = HistGradientBoosting / CatBoost / XGBoost / ExtraTrees on 70 features, rank-averaged and mapped to the
HistGBR scale. Default: the trained members are loaded. `TRAIN_FROM_SCRATCH = True` refits them on the 39,314 OOF
candidates of the 707 training frames (CPU, about 1 min). Decode: tau 0.28, mask NMS 0.3, no overlaps.
Runs in a subprocess (CatBoost / XGBoost must not share a process with torch on macOS). CPU: about 4 min.
""")

code("""
import yaml
cfg_run = yaml.safe_load(open(REPO / "configs" / "nlv.yaml")); cfg_run["nlv"].update({k: v for k, v in C.items() if isinstance(v, (str, dict))})
run_cfg = WORK / "nlv_run.yaml"; yaml.safe_dump(cfg_run, open(run_cfg, "w"))
OUT = WORK / "submission.csv"
cmd = [sys.executable, "-m", "filament_seg", "--config", str(run_cfg), "final-cpu", "--out", str(OUT)]
cmd += ["--save-weights", str(WORK / "vens")] if TRAIN_FROM_SCRATCH else ["--load-weights", str(VENS)]
t = time.time(); r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
assert r.returncode == 0, r.stderr[-2000:]
res = json.loads(r.stdout[r.stdout.index("{"):]); print(json.dumps({k: (Path(v).name if k in ("csv", "yhat") else v) for k, v in res.items()}, indent=1)); print(f"{time.time() - t:.0f} s")
""")

code("""
from filament_seg import submission
chk = submission.validate(OUT, set(test_names))
print(chk)
assert all(chk[k] for k in ("header_ok", "no_quotes", "ids_ok", "stems_in_test", "no_overlap"))
print("identical to the submitted CSV (md5 9eb43f91...):", res["md5"] == "9eb43f91f77263f7a38e3a8c0dce8ba2")
""")

md("""
## 9. Self-evaluation on the training frames (OOF)
1. OOF PQ of the whole pipeline: OOF V-ENS scores + OOF NNR masks, decoded as on the test set; rec = against the
   annotator records, sim = 77 % of the GT filaments kept at random (5 draws; calibrated on the public LB).
2. Golden check of the preprocessing and candidate stage: the candidate union and all 48 features of two training frames,
   rebuilt here from the images and the OOF model outputs, equal the table the verifier was trained on.
""")

code("""
r = subprocess.run([sys.executable, "-m", "filament_seg", "--config", str(REPO / "configs" / "nlv.yaml"), "oof-eval"], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-2000:]
ev = json.loads(r.stdout[r.stdout.index("{"):]); print(ev)
print("expected rec 0.4701 / sim 0.4116:", (ev["rec_pq"], ev["sim_pq"]) == (0.4701, 0.4116))
""")

code("""
cfg_tr = config.load(REPO / "configs" / "nlv.yaml"); cfg_tr["paths"].data = DATA
ref = pd.read_parquet(C["train_cands"]); names = sorted(ref.image.unique())[:2]
df, _ = nlv.oof_candidates(cfg_tr, names)
a = ref[ref.image.isin(names)].sort_values(["image", "idx"]).reset_index(drop=True); b = df.sort_values(["image", "idx"]).reset_index(drop=True)
cols = [c for c in a.columns if c in b.columns and pd.api.types.is_numeric_dtype(a[c])]
same = len(a) == len(b) and all(np.array_equal(a[c].values.astype(float), b[c].values.astype(float), equal_nan=True) for c in cols)
print(f"{len(b)} candidates on {names}: all {len(cols)} numeric columns identical to the training table: {same}")
""")

md("""
## 10. Training from scratch (`TRAIN_FROM_SCRATCH = True`)
The verifier is refitted in section 8. The other stages are GPU jobs of several hours each; the cell prints their
commands (5 folds each; training frames always get the output of the fold model that did not see them).
""")

code("""
STEPS = [
    ("U-Net++ (5 folds + hold-out)", "python -m filament_seg semantic --split split_f$k.json --tag v1f$k ; --split split_v1.json --tag v1", "36 min x 6 (RTX 3080 Ti)"),
    ("nnU-Net ResEnc-L 2D", "nnUNetv2_plan_and_preprocess -d 101 -pl nnUNetPlannerResEncL ; nnUNetv2_train 101 2d $k -p nnUNetResEncUNetLPlans -tr nnUNetTrainer_250epochs --npz",
     "about 6 h per fold at 250 epochs (89 s / epoch, RTX PRO 6000); the submitted fold 0 used 1000 epochs, fold 1 a 373-epoch checkpoint"),
    ("Mask DINO Swin-L", "filament_seg.maskdino_cands.train(..., arch='swinl', epochs=40)", "longer than R50 (32 GB GPU memory)"),
    ("Mask DINO R50 seeds 0 / 1 / 2", "filament_seg.maskdino_cands.train(..., arch='r50', seed=s, epochs=40)", "158 min per fold and seed (RTX 3080 Ti)"),
    ("own2 map U-Nets r34 + EffV2-S", "python -m filament_seg own-maps --mode oof --train ; --mode testfold", "about 1 GPU hour per family and fold"),
    ("candidates + crops (NL)", "filament_seg.nlv.candidates_image per frame ; python -m filament_seg crops", "about 1 h (CPU)"),
    ("NNR refiner", "python -m filament_seg --config configs/nlv.yaml (nlv.stage_nnr_crops) ; residual --fold $k", "30 min x 5"),
    ("V-ENS + decode", "python -m filament_seg --config configs/nlv.yaml final-cpu --out nlv.csv", "4 min (CPU)"),
]
if TRAIN_FROM_SCRATCH:
    for name, c, t in STEPS:
        print(f"{name:32s} {t}\\n    {c}")
else:
    print("trained models loaded; set TRAIN_FROM_SCRATCH = True to print the training commands and refit the verifier")
""")

nb = nbf.v4.new_notebook(); nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
nbf.write(nb, REPO / "notebooks" / "demo_nlv.ipynb")
print("wrote", REPO / "notebooks" / "demo_nlv.ipynb", len(C), "cells")
