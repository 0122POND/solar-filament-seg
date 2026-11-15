# Third-party notices

This repository's own code and our trained weights are MIT (`LICENSE`). No third-party source file is vendored:
the packages below are installed (`requirements.txt`, setup notes in `README.md`) and keep their own licences.
Every model of the final submission was trained by us on the competition train set; the only pre-trained weights we
start from are the public checkpoints listed here. DINOv3 is not used anywhere.

## Models and frameworks

| component | used for | licence | NLv (`configs/nlv.yaml`) | STK-avg5 + C0 + S15F (`configs/stk.yaml`) |
|---|---|---|---|---|
| nnU-Net v2 (MIC-DKFZ) | candidates 1, refiner plane 5, g7_m2 `e_end_ext` / `e_h_*` | Apache-2.0 | yes | yes |
| Mask DINO (IDEA-Research) and its COCO checkpoints | Swin-L candidates, R50 voters, Swin-L 1536 voter / plane / S15F pixel-decoder features (`filament_seg/s15f.py` wraps `forward_features` at run time; MaskDINO is not modified beyond the CPU fallback line of `maskdino_cands.py`) | Apache-2.0 | yes | yes |
| detectron2 (Meta) and its COCO model-zoo weights (Cascade Mask R-CNN R50-FPN 3x, `Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml`) | Mask DINO runtime; Cascade family of g7_m2 (`filament_seg/cascade_cands.py`, imports detectron2) | Apache-2.0 | yes (runtime) | yes |
| Swin Transformer backbone (Microsoft), ImageNet-22k pre-training | inside the Mask DINO Swin-L checkpoints | MIT (code and weights); the ImageNet terms of access apply to the pre-training data | yes | yes |
| RF-DETR (Roboflow) `rfdetr` 1.11.1, RF-DETR-Seg Large checkpoint | RF-DETR-Seg family of g7_m2, refiner plane (`filament_seg/rfdetr_cands.py`, imports rfdetr) | Apache-2.0 (RF-DETR-Seg sizes; the PML-1.0 licence of the package applies only to the detection XL / 2XL models of `rfdetr[plus]`, not used) | no | yes |
| DINOv2 (Meta) | encoder inside RF-DETR | Apache-2.0 | no | yes |
| segmentation-models-pytorch | U-Net++, own2 maps, refiners | MIT | yes | yes |
| timm (+ ImageNet weights of ResNet-34 / EfficientNetV2-S) | encoders | Apache-2.0 | yes | yes |
| PyTorch / torchvision (incl. `torchvision.ops.roi_align` for the S15F crops) | everything on GPU | BSD-3-Clause | yes | yes |
| torchmetrics | rubric metric mean Dice (`metrics.rubric`, `DiceScore`) | Apache-2.0 | evaluation | evaluation |
| ultralytics (YOLO11s-seg) | base pipeline only (`filament_seg/yolo.py`) | AGPL-3.0 | no | no |

Weights hosting: the Hugging Face model repository (`weights/HF_MODEL_CARD.md`) holds only our fine-tuned weights; the
public initial checkpoints (COCO / ImageNet) are downloaded from their original sources by the respective packages.
The wrappers `cascade_cands.py` and `rfdetr_cands.py` contain only our training / inference loops around the installed
packages (no copied source).

`filament_seg/yolo.py` only imports ultralytics; the final model and the NLv / STK variants do not run it. Installing ultralytics is needed
only to reproduce the base pipeline (`configs/champion.yaml`).

## Libraries

| library | licence |
|---|---|
| scikit-learn, joblib, SciPy, scikit-image, NumPy, pandas, nbformat, nbconvert, nbclient, ipykernel (demo notebook) | BSD-3-Clause |
| Matplotlib (figures of the demo notebook) | Matplotlib licence (PSF-based) |
| CatBoost, XGBoost, OpenCV (opencv-python-headless), pyarrow, safetensors, huggingface_hub (upload script only) | Apache-2.0 |
| pycocotools | BSD-2-Clause |
| PyYAML | MIT |
| Pillow | MIT-CMU (HPND) |
| pytest | MIT |

## Data

| data | use | terms |
|---|---|---|
| MAGFiLO / GONG H-alpha frames and annotations (competition data) | training and evaluation | competition rules; not redistributed (no image, label or prediction of the training frames is stored here; the stored outputs of `notebooks/demo.ipynb` include one figure of a test frame at reduced size; the archive `cache/final_cache_v1.tar` on the Hugging Face Hub holds outputs of our models on the competition frames: no image and no annotation polygon; the out-of-fold candidate table carries, for each candidate, its IoU with the annotator records (the training target of the selector)) |
| unlabeled GONG H-alpha frames (NSO) | SimMIM pre-training of the refiner encoder only | NSO data policy; listed by name in `resources/gong_unlabeled_clean.txt`, not redistributed; frames within 24 h of a test frame excluded (`resources/gong_exclusion_audit.tsv`) |
| COCO (via the Mask DINO / detectron2 checkpoints) | initialisation only | CC BY 4.0 annotations; Flickr terms for the images |
