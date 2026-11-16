---
license: other
license_name: mit-and-apache-2.0
license_link: LICENSE
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - solar-physics
  - h-alpha
  - instance-segmentation
  - kaggle
---

# Solar filament instance segmentation: trained weights

Weights of our solution to the IEEE BigData Cup 2026 / Kaggle competition *filament-segmentation-2026* (instance
segmentation of solar filaments in GONG H-alpha full-disk frames, 2048 x 2048, grayscale). Code, configuration, the
inference notebook and the reproduction checks: https://github.com/0122POND/solar-filament-seg (`notebooks/`,
`configs/final_yellow.yaml`, `weights/MANIFEST_final.tsv`).

**Final model NLv x full-data B6 + C0, decode order bonus 0.20 and mask NMS 0.1** (public leaderboard 0.41, `configs/final_yellow.yaml`): nnU-Net ResEnc-L and Mask DINO Swin-L candidates, three
Mask DINO R50 voters, map features of our own U-Nets, a V-ENS verifier (gradient boosting / CatBoost / XGBoost / extra
trees), and six residual crop refiners (NNRP and S15F, seeds 16 / 17 / 18) retrained on all 707 training frames
(`fullfit/`); S15F also reads Swin-L 1536 pixel-decoder features. Out-of-fold PQ of the fold version on the 707 training
frames: nested 0.4756, rec 0.4761, simulated 0.4146 (nested 0.4753 without the order bonus and with mask NMS 0.3). Documented alternatives: the same model with
refiner seeds none / 11 / 12, no order bonus and mask NMS 0.3 (`configs/final_fullfit.yaml`, leaderboard 0.41), STK-avg5 + C0 + S15F (`configs/stk.yaml`,
OOF nested 0.4759, leaderboard 0.40) and the minimal model NLv + NNR (`configs/nlv.yaml`, `nlv/` files and `seg_v1.pt` only).

## Files

`MANIFEST_final.tsv` lists every file with its size, sha256 (and md5 where it was already published), role, the
pipelines that use it (`variants`: `final` = the final model, `fullfit_a` = the six refiners of `configs/final_fullfit.yaml`, `stk` = STK-avg5 + C0 + S15F, `ore` = its slimmer documented variant, `nlv` = NLv + NNR)
and its licence. The demo notebook downloads them with
`https://huggingface.co/<this repository>/resolve/main/<file>` (no login needed) and checks
the sha256. `cache/final_cache_v1.tar` (157 MB, listed in `MANIFEST_cache.tsv` of the code repository) holds the stored
outputs of our models that the notebook reads on a machine without a GPU.

| folder | content |
|---|---|
| `nlv/nnunet/` | nnU-Net v2 ResEnc-L 2D, 5 folds (candidate source 1; refiner plane) |
| `nlv/maskdino_swinl_f*.pt` | Mask DINO Swin-L 1024, 5 folds (candidate source 2) |
| `nlv/maskdino_r50_s{0,1,2}_f*.pt`, `final/maskdino_r50_s3_f*.pt` | Mask DINO R50 seeds 0-3, 5 folds each (voters, agreement features) |
| `final/maskdino_swinl1536_f*.pt` | Mask DINO Swin-L 1536, 5 folds (4th voter, agreement features, refiner plane, S15F features) |
| `final/cmrcnn_r50fpn_f*.pt` | Cascade Mask R-CNN R50-FPN 3x, 5 folds (agreement features) |
| `final/rfdetrseg_l_f*.pth` | RF-DETR-Seg Large, resolution 1008, 5 folds (agreement features, refiner plane) |
| `nlv/own2_{r34,eff}_f*.pt` | U-Net r34 / EfficientNetV2-S probability maps, 5 folds (map features) |
| `seg_v1.pt`, `nlv/train/seg_v1f*.pt` | U-Net++ (candidate features, refiner plane); fold models for the training frames |
| `nlv/train/ssl_mimC_r34_enc.pt` | SimMIM ResNet-34 encoder pre-trained on unlabeled GONG frames (refiner initialisation) |
| `final/refiner_S15F_f*.pt`, `final/s15f_align.npz` | S15F refiner, 5 folds (STK-avg5); linear alignment of the Swin-L 1536 features (no labels used; also the final model) |
| `fullfit/refiner_FF{NNRP,S15F}{x16,x17,x18}_f-1.pt` | the six refiners of the final model, trained on all 707 frames with seeds 16 / 17 / 18 (NNRP: 9 planes; S15F: + feature projection) |
| `fullfit/refiner_FF{NNRP,S15F}{o,x11,x12}_f-1.pt` | the six refiners of `configs/final_fullfit.yaml` (same recipe, seeds none / 11 / 12) |
| `final/vens/seed*/` | V-ENS verifier members of the five seeds (80 features, STK-avg5) |
| `nlv/verifier_nlv_*` | V-ENS of NLv (70 features): the verifier of the final model and of NLv + NNR |
| `nlv/refiner_V2bSCrdmNNR_f*.pt` | NNR refiner of the minimal model NLv + NNR |
| `cache/final_cache_v1.tar` | stored outputs of our models for the CPU path of the demo notebook: candidates, masks, probability maps, refined masks and selector scores on the test frames, out-of-fold candidates, scores and refined masks on the training frames, the submitted CSVs (no image, no annotation polygon; the out-of-fold candidate table carries the IoU of each candidate with the annotator records) |

## Licences

- Our code and our trained weights: MIT (see `LICENSE` of the code repository).
- Several networks were initialised from public checkpoints, whose licences apply to those parts: Mask DINO R50 /
  Swin-L COCO checkpoints (Apache-2.0; Swin Transformer backbone MIT), detectron2 model zoo Cascade Mask R-CNN COCO weights
  (Apache-2.0), RF-DETR-Seg Large (Apache-2.0; DINOv2 encoder Apache-2.0), ImageNet weights of timm / segmentation-models-pytorch
  encoders (Apache-2.0 / MIT). nnU-Net (Apache-2.0) was trained from scratch. The COCO / ImageNet terms of access apply to
  the pre-training data of those checkpoints.
- The verifier files are scikit-learn (BSD-3-Clause) pickles, CatBoost and XGBoost (Apache-2.0) model files. Load the
  `.joblib` files only from this repository (pickles execute code when loaded) and with the library versions of
  `requirements.txt` (scikit-learn 1.8.0, catboost 1.2.10, xgboost 3.4.1).

## Training data

Every model was trained by us on the training set of the competition (MAGFiLO 1.0 Kaggle 2026: GONG H-alpha frames with
filament polygons from several independent annotators; 707 physical frames, 5-fold cross-validation by physical frame).
The SimMIM encoder additionally saw unlabeled GONG H-alpha frames (NSO), excluding frames within 24 h of any test frame.
No competition image and no annotation polygon is stored here; `cache/final_cache_v1.tar` holds outputs of our models on the
competition frames (masks and scores) and, for the out-of-fold candidates of the training frames, their IoU with the
annotator records (the training target of the selector). No external labelled data was used.

## Intended use and limitations

Reproducing and inspecting our competition submission, and research on filament segmentation in GONG H-alpha data.
The models were tuned for the competition metric (Panoptic Quality with one-to-many / many-to-one penalties) on GONG
frames of the competition period; other instruments, resolutions or seeing conditions are out of distribution. Not
validated for operational space-weather use. The annotators disagree substantially on filament extent (annotator vs
annotator PQ is reported in the code repository), which bounds what any model trained on these labels can reach.
