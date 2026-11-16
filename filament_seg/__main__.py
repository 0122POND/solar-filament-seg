"""Single entry point: ``python -m filament_seg <stage> [options]``.

Base pipeline (configs/champion.yaml; U-Net++, the refiner crops, the residual training and the SSL encoder
are also parts of the final model and of NLv + NNR; stage1 is trained for the base pipeline only):
  python -m filament_seg folds && python -m filament_seg prep
  for k in 0 1 2 3 4: python -m filament_seg semantic --split split_f$k.json --tag v1f$k
  python -m filament_seg semantic --split split_v1.json --tag v1 && python -m filament_seg semantic-oof
  for k in 0 1 2 3 4: python -m filament_seg yolo --split split_f$k.json --run a_s1536_f$k
  python -m filament_seg yolo --split split_v1.json --run holdout && python -m filament_seg yolo-oof
  python -m filament_seg candidates && python -m filament_seg crops
  for k in 0 1 2 3 4: python -m filament_seg stage1 --fold $k
  python -m filament_seg gong && python -m filament_seg ssl
  for k in 0 1 2 3 4: python -m filament_seg residual --fold $k
  python -m filament_seg verifier && python -m filament_seg evaluate
  python -m filament_seg infer --weights weights --out submission.csv
Minimal variant NLv + NNR (configs/nlv.yaml; README), CPU stages from the cached model outputs:
  python -m filament_seg --config configs/nlv.yaml final-cpu --out nlv.csv            (V-ENS fit + test scores + decode)
  python -m filament_seg --config configs/nlv.yaml final-cpu --yhat-cached --out d.csv (decode only)
  python -m filament_seg --config configs/nlv.yaml oof-eval                            (OOF PQ of the 707 training frames)
  python -m filament_seg --config configs/nlv.yaml own-maps --mode oof|testfold        (own2 maps, GPU)
Alternative STK-avg5 + C0 + S15F (configs/stk.yaml; README), CPU stages:
  python -m filament_seg --config configs/stk.yaml evo-feats --mode oof|test --out g7.parquet   (g7_m2 columns)
  python -m filament_seg --config configs/stk.yaml nnrp-planes                                  (NNRP training planes)
  python -m filament_seg --config configs/stk.yaml final-cpu --out stk.csv                      (5 V-ENS seeds + C0 + decode)
  S15F refiner features and the test-time refiner (configs/s15f.yaml, included by configs/stk.yaml; GPU except s15f-fit):
  for k in 0 1 2 3 4: python -m filament_seg --config configs/stk.yaml s15f-ref --fold $k      (reference sample)
  python -m filament_seg --config configs/stk.yaml s15f-fit                                       (align.npz, CPU)
  python -m filament_seg --config configs/stk.yaml s15f-train-crops                               (OOF feature crops)
  for k in 0 1 2 3 4: python -m filament_seg --config configs/stk.yaml residual --fold $k     (S15F refiner)
  python -m filament_seg --config configs/stk.yaml s15f-test                                      (test feature crops)
  python -m filament_seg --config configs/stk.yaml refine-test --weights W --out refined.pkl      (refiner members on the test)
FINAL MODEL NLv x full-data B6 refiners + C0 + order bonus + mask NMS 0.1 (configs/final_yellow.yaml; README):
  python -m filament_seg --config configs/final_yellow.yaml residual --fold -1 --member FFNNRPx16       (x 6 members, see the config)
  python -m filament_seg --config configs/final_yellow.yaml detect-test --model swinl1024|r50|swinl1536 --weights W --out D
                                             (Mask DINO fold models on the test frames, GPU; swinl1024 / r50: frame i by fold model i % 5 only)
  python -m filament_seg --config configs/final_yellow.yaml refine-test --weights W --out refined.pkl   (each full refiner x 5 fold inputs)
  python -m filament_seg --config configs/final_yellow.yaml final-cpu --out final.csv                   (NLv V-ENS + C0 + tau 0.25 + order bonus + mask NMS 0.1)
  python -m filament_seg --config configs/final_yellow.yaml final-cpu --yhat-cached --out final.csv     (C0 + decode, under 1 min)
  Same model with the refiner seeds none / 11 / 12, no order bonus and mask NMS 0.3: configs/final_fullfit.yaml (same stages, members FFNNRPo ...).
  Detector wrappers of the g7_m2 / NNRP families: python -m filament_seg.cascade_cands, python -m filament_seg.rfdetr_cands
"""
import argparse
import json
from pathlib import Path

from . import config


def main(argv=None):
    """Command-line entry point: parse --config and the stage, load the config (config.load) and run that one stage."""
    ap = argparse.ArgumentParser(prog="python -m filament_seg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=config.DEFAULT_CONFIG,
                    help="YAML config; default configs/champion.yaml = the base pipeline. The final model is configs/final_yellow.yaml")
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("folds", help="5-fold and hold-out image splits")
    sub.add_parser("prep", help="physical 3-channel inputs and soft targets (train)")
    s = sub.add_parser("semantic", help="train the full-disk U-Net++ on one split (GPU)")
    s.add_argument("--split", required=True)
    s.add_argument("--tag", required=True)
    sub.add_parser("semantic-oof", help="stack fold-validation probabilities into the OOF array")
    s = sub.add_parser("yolo", help="train YOLO11s-seg on one split and predict its validation images (GPU)")
    s.add_argument("--split", required=True)
    s.add_argument("--run", required=True)
    s.add_argument("--device", default="0")
    sub.add_parser("yolo-oof", help="stack fold-validation YOLO predictions")
    for name, hlp in (("candidates", "OOF candidate union, features and targets"), ("crops", "refiner crops and teachers"),
                      ("verifier", "refined features and GBDT verifier (OOF + full)")):
        sub.add_parser(name, help=hlp).add_argument("--workers", type=int, default=4)
    sub.add_parser("stage1", help="stage-1 refiner of one fold (GPU)").add_argument("--fold", type=int, required=True)
    s = sub.add_parser("residual", help="residual refiner of one fold (GPU); --fold -1 = full-data refiner on all 707 frames")
    s.add_argument("--fold", type=int, required=True)
    s.add_argument("--seed", type=int, default=-1, help="seed of torch / numpy and the sampler (default -1 = not reset)")
    s.add_argument("--tag", default=None, help="checkpoint tag (default refiner.tag): out/refiner_<tag>_f<fold>.pt")
    s.add_argument("--member", default=None, help="train the refiner.members entry with this tag (its seed; no features without test_feats)")
    sub.add_parser("gong", help="download unlabeled GONG frames for SSL")
    sub.add_parser("ssl", help="SimMIM pre-training of the refiner encoder (GPU)")
    s = sub.add_parser("evaluate", help="OOF PQ of the full pipeline")
    s.add_argument("--tau", type=float, nargs="*")
    s.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("infer", help="test frames -> submission CSV")
    s.add_argument("--weights", type=Path, default=config.REPO / "weights")
    s.add_argument("--out", type=Path, default=Path("submission.csv"))
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--device", default="cpu")
    s.add_argument("--limit", type=int, default=0, help="only the first N test frames (smoke test)")
    s = sub.add_parser("own-maps", help="own 2048-px maps (GPU), mode oof (train frames, OOF) or testfold (one map per fold model)")
    s.add_argument("--mode", choices=["oof", "testfold"], required=True)
    s.add_argument("--weights", type=Path, default=config.REPO / "weights")
    s.add_argument("--device", default="cuda")
    s.add_argument("--train", action="store_true", help="train missing fold models first (about 1 GPU-hour each)")
    s.add_argument("--views", type=int, choices=[1, 8], default=8, help="8 = D4 (submitted), 1 = light variant")
    s.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("final-cpu", help="selector (V-ENS) + C0 + decode from the cached model outputs: configs/final_yellow.yaml (final model), final_fullfit.yaml, stk.yaml, nlv.yaml")
    s.add_argument("--out", type=Path, default=Path("submission_nlv.csv"))
    s.add_argument("--yhat-cached", action="store_true", help="decode the stored verifier scores (no verifier fit)")
    s.add_argument("--recompute-votes", action="store_true", help="training votes from the OOF Mask DINO R50 pickles instead of the stored table")
    s.add_argument("--recompute-maps", action="store_true", help="training map features from the own2 OOF maps instead of the stored table")
    s.add_argument("--save-weights", type=Path, default=None, help="folder for the four fitted V-ENS members")
    s.add_argument("--load-weights", type=Path, default=None, help="load the four V-ENS members from this folder instead of fitting them")
    s.add_argument("--recompute-evo", action="store_true", help="STK: g7_m2 columns from the detector pickles instead of the stored tables")
    s = sub.add_parser("evo-feats", help="STK (configs/stk.yaml): g7_m2 agreement / shape columns of the OOF or test candidates (CPU)")
    s.add_argument("--mode", choices=["oof", "test"], required=True)
    s.add_argument("--out", type=Path, required=True)
    sub.add_parser("nnrp-planes", help="STK (configs/stk.yaml): NNRP training planes refiner_nnrp_planes.npy from refiner_meta.parquet (CPU)")
    s = sub.add_parser("s15f-ref", help="S15F: reference sample of one Swin-L 1536 fold model (GPU)")
    s.add_argument("--fold", type=int, required=True)
    s.add_argument("--device", default="cuda")
    sub.add_parser("s15f-fit", help="S15F: common 16-channel alignment s15f_align.npz from the five reference samples (CPU)")
    for name, hlp in (("s15f-train-crops", "S15F: OOF feature crops of the refiner training rows (GPU)"),
                      ("s15f-test", "S15F: test feature crops of fold model k (GPU)")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
        s.add_argument("--device", default="cuda")
        s.add_argument("--limit", type=int, default=0, help="s15f-test: only the first N test frames (smoke test)")
    s = sub.add_parser("detect-test", help="Mask DINO fold models on the test frames (GPU) -> maskdino_test_*_f<k>.pkl; "
                                           "swinl1024 / r50: frame i by fold model i %% 5 only, unless --all-folds")
    s.add_argument("--model", choices=["swinl1024", "r50", "swinl1536"], required=True, help="candidates / voters / refiner planes and S15F")
    s.add_argument("--weights", type=Path, default=config.REPO / "weights")
    s.add_argument("--out", type=Path, default=None, help="output folder (default: paths.out)")
    s.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="r50: voter seeds")
    s.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    s.add_argument("--all-folds", action="store_true", help="every fold model on every frame (5 x the passes; g7_m2 of configs/stk.yaml "
                                                            "needs it; always on for swinl1536)")
    s.add_argument("--maskdino-root", type=Path, default=None, help="MaskDINO checkout (default: $MASKDINO_ROOT)")
    s.add_argument("--device", default="cuda")
    s.add_argument("--limit", type=int, default=0, help="only the first N test frames (smoke test)")
    s = sub.add_parser("refine-test", help="test-time refiner (refiner.members, e.g. S15F) on all test candidates -> refined masks pickle")
    s.add_argument("--weights", type=Path, default=None, help="folder the member checkpoint patterns are relative to (default: out)")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--device", default="cuda")
    s.add_argument("--limit", type=int, default=0, help="only the first N test frames (smoke test)")
    s = sub.add_parser("oof-eval", help="OOF PQ (rec and simulated) of the 707 training frames: configs/final_yellow.yaml (fold version of the final model), final_fullfit.yaml, stk.yaml, nlv.yaml")
    s.add_argument("--tau", type=float, default=None)
    a = ap.parse_args(argv)
    cfg = config.load(a.config)
    if a.stage == "final-cpu":  # no torch in this process: CatBoost / XGBoost segfault next to torch's OpenMP on macOS
        from . import nlv
        print(json.dumps(nlv.stage_final_cpu(cfg, a.out, a.yhat_cached, a.recompute_votes, a.recompute_maps, a.save_weights, a.load_weights,
                                             a.recompute_evo), indent=1))
        return
    if a.stage == "evo-feats":
        from . import evofeats, nlv
        c = nlv.resolve(cfg)
        X = evofeats.oof_features(c) if a.mode == "oof" else evofeats.test_features(c)
        X.to_parquet(a.out)
        print(json.dumps(dict(out=str(a.out), rows=len(X), nan={k: int(v) for k, v in X[evofeats.COLS].isna().sum().items()}), indent=1))
        return
    if a.stage == "nnrp-planes":
        from . import refplanes
        refplanes.stage_nnrp_planes(cfg)
        return
    if a.stage in ("s15f-ref", "s15f-fit", "s15f-train-crops", "s15f-test"):
        from . import s15f
        if a.stage == "s15f-ref":
            s15f.stage_ref(cfg, a.fold, a.device)
        elif a.stage == "s15f-fit":
            s15f.stage_fit(cfg)
        elif a.stage == "s15f-train-crops":
            s15f.stage_train_crops(cfg, a.folds, a.device)
        else:
            print(s15f.stage_test(cfg, a.folds, a.device, a.limit))
        return
    if a.stage == "detect-test":
        from . import maskdino_cands
        print(json.dumps(maskdino_cands.stage_test(cfg, a.model, a.weights, a.out, a.seeds, a.folds, a.all_folds, a.maskdino_root, a.device,
                                                   a.limit), indent=1))
        return
    if a.stage == "refine-test":
        from . import nlv
        print(json.dumps(nlv.stage_refine_test(cfg, a.out, a.weights, a.device, a.limit), indent=1))
        return
    if a.stage == "oof-eval":
        from . import nlv
        print(json.dumps(nlv.oof_eval(cfg, a.tau), indent=1))
        return
    from . import stages

    if a.stage == "folds":
        stages.stage_folds(cfg)
    elif a.stage == "prep":
        stages.stage_prep(cfg)
    elif a.stage == "semantic":
        stages.stage_semantic(cfg, a.split, a.tag)
    elif a.stage == "semantic-oof":
        stages.stage_semantic_oof(cfg)
    elif a.stage == "yolo":
        stages.stage_yolo(cfg, a.split, a.run, a.device)
    elif a.stage == "yolo-oof":
        stages.stage_yolo_oof(cfg)
    elif a.stage == "candidates":
        stages.stage_candidates(cfg, a.workers)
    elif a.stage == "crops":
        stages.stage_crops(cfg, a.workers)
    elif a.stage == "stage1":
        stages.stage_stage1(cfg, a.fold)
    elif a.stage == "gong":
        stages.stage_gong(cfg)
    elif a.stage == "ssl":
        stages.stage_ssl(cfg)
    elif a.stage == "residual":
        stages.stage_residual(cfg, a.fold, a.seed, a.tag, a.member)
    elif a.stage == "verifier":
        stages.stage_verifier(cfg, a.workers)
    elif a.stage == "evaluate":
        res = stages.stage_evaluate(cfg, a.tau, a.workers)
        print(json.dumps({str(k): v for k, v in res.items()}, indent=1))
    elif a.stage == "infer":
        print(stages.infer(cfg, a.weights, a.out, a.workers, a.device, a.limit))
    elif a.stage == "own-maps":
        stages.stage_own_maps(cfg, a.mode, a.weights, a.device, a.train, a.views, a.limit)


if __name__ == "__main__":
    main()
