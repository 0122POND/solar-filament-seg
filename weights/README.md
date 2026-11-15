# Weights

The weight files are too large for Git (up to 895 MB each; GitHub accepts at most 100 MB per file). They are stored on
the Hugging Face Hub, and the demo notebook downloads and checks them without a manual step.

**All weights (Hugging Face Hub).** `MANIFEST_final.tsv` lists the 115 files of the final model (`variants` `final`,
`configs/final_yellow.yaml`), of `configs/final_fullfit.yaml` (`fullfit_a`: its six refiners; everything else is shared with
`final`), of STK-avg5 + C0 + S15F (`stk`), of its slim variant (`ore`) and of NLv + NNR (`nlv`): path, role, size,
sha256, md5 (where already published), licence and the URL `https://huggingface.co/${HF_REPO}/resolve/main/<file>`
(`HF_REPO=bizyu/solar-filament-seg`; public, no login needed). `notebooks/demo.ipynb` (`fetch`) downloads
and verifies them. `HF_MODEL_CARD.md` is the model card of the Hub repository; `tools/hf_stage_upload.sh` uploads the files
(sha256 checked before each upload) and verifies the Hub copy (`verify`).

**Stored model outputs for the CPU path.** `MANIFEST_cache.tsv` lists one archive in the same format,
`cache/final_cache_v1.tar` (157 MB): the outputs of our models that the CPU path of `notebooks/demo.ipynb` and
`tests/test_golden_yellow.py` read instead of running the GPU stages (test frames: candidates, masks, voter predictions,
probability maps, refined masks, selector scores, the submitted CSVs; training frames: the out-of-fold candidate table,
scores and refined masks). It holds no image and no annotation polygon; the out-of-fold candidate table carries, for each candidate, its IoU with the annotator records (the training target of the selector).

**NLv only (Kaggle Dataset).** The NLv files are also in the Kaggle Dataset
`deepernet/solar-filament-nlv-weights` (needs a Kaggle login; the final model does not need it):

```
kaggle datasets download -d deepernet/solar-filament-nlv-weights -p weights --unzip
cd weights && md5sum -c MD5SUMS.txt
```

- `MANIFEST_nlv.tsv`: every file of the minimal variant NLv + NNR, with its path inside `weights/`, role, size in bytes,
  md5, origin and licence. `python scripts/export_weights.py manifest` rebuilds it.
- `MD5SUMS.txt`: md5 of every file in the dataset. Besides the files of the manifest, the dataset holds the original
  nnU-Net `checkpoint_final.safetensors` next to each stripped `checkpoint_final.pth`.

Inference uses `nlv/` and `seg_v1.pt`. `nlv/train/` (the U-Net++ fold models and the SimMIM encoder) is needed only
for retraining.
