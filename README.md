# Image Captioning Project (Comparison of CNN-LSTM, Attention, and Transformers)

A PyTorch implementation comparing sequence generation architectures on the MS COCO dataset.

## Project Environment

* **Primary Python Version:** 3.10
* **Dataset:** MS COCO (Karpathy splits)
* **Hardware:** CPU/GPU (CUDA); cluster jobs supported via `scripts/train.sbatch`

### Dependencies

Dependencies are listed in `requirements.txt`. Highlights:

* **`torch` & `torchvision`:** Training and pretrained CNN encoder (ResNet).
* **`pycocotools` & `pycocoevalcap`:** Karpathy-style metrics (BLEU, METEOR, CIDEr, ROUGE-L). **METEOR requires Java** on `PATH` and the `meteor-1.5.jar` bundled with `pycocoevalcap`.
* **`Pillow`:** Image loading.
* **`pyyaml`:** Training and evaluation YAML configs.

---

## Setup

**Do not commit** the MS COCO images, `dataset_coco.json` weights, or large checkpoints to Git. Use local paths such as `data/` and `checkpoints/`.

### Cloud (e.g. Colab)

Clone the repo, then:

```bash
pip install -r requirements.txt
```

### Local

Create a Conda (or venv) environment with Python 3.10, install PyTorch for your CUDA version, then:

```bash
pip install -r requirements.txt
```

### Dataset layout

After downloading (script or manual), you should have:

```text
data/
  annotations/
    dataset_coco.json      # Karpathy annotations
  images/
    train2014/             # COCO train images
    val2014/               # COCO val images
```

**Linux script:** from repo root:

```bash
cd scripts
bash download_coco.sh
```

**Manual (Windows or if the script fails):** download and unzip:

1. [Karpathy splits (`caption_datasets.zip`)](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) — place `dataset_coco.json` under `data/annotations/`.
2. [COCO Train 2014](http://images.cocodataset.org/zips/train2014.zip) — unzip into `data/images/train2014/`.
3. [COCO Val 2014](http://images.cocodataset.org/zips/val2014.zip) — unzip into `data/images/val2014/`.

---

## Commands reference

Run these from the **repository root** unless noted. Default data root is `./data` (override with `--data_dir`).

### Training

Training reads a YAML config and writes checkpoints under `checkpoints/` as `{model_type}_{N}pct_latest.pth`, `{model_type}_{N}pct_best.pth`, plus `{model_type}_{N}pct_loss_curve.png`.

```bash
# Example: one config
python train.py --config configs/transformer_config.yaml

# Regime configs (train_fraction 10% / 25% / 50% / 100%) — examples:
python train.py --config configs/regimes/lstm_10pct.yaml
python train.py --config configs/regimes/attention_25pct.yaml
python train.py --config configs/regimes/transformer_100pct.yaml
```

**Cluster (SLURM example):** edit `PROJECT_ROOT`, `CONDA_ENV`, and partition in `scripts/train.sbatch`, then:

```bash
sbatch scripts/train.sbatch
# Optional: override config for one job
sbatch --export=ALL,CONFIG=configs/regimes/lstm_50pct.yaml scripts/train.sbatch
```

---

### Metrics (full: BLEU-1–4, METEOR, CIDEr, ROUGE-L)

Uses GPU if available for caption generation (unless you score from JSON only).

```bash
# Single checkpoint
python metrics.py --checkpoint checkpoints/lstm_10pct_best.pth --split val

# Many checkpoints (see configs/eval_manifest.example.yaml)
python metrics.py --manifest configs/eval_manifest.example.yaml \
  --output results/metrics_by_regime.csv --print_regime_table

# Optional: save captions while scoring
python metrics.py --checkpoint checkpoints/lstm_10pct_best.pth \
  --split val --predictions_json results/captions/val_lstm_10pct.json

# Skip ROUGE-L only
python metrics.py --checkpoint checkpoints/lstm_10pct_best.pth --skip_rouge
```

---

### Creating predictions only (GPU server / fast eval without metrics)

Writes JSON lists of `{ "image_id", "caption" }` and **does not** compute BLEU/METEOR/CIDEr/ROUGE. Requires `--predictions_json`.

```bash
# One model (pick any unique filename)
python metrics.py --checkpoint checkpoints/lstm_10pct_best.pth --split val \
  --captions_only --predictions_json results/captions/val_lstm_10pct.json

# All runs in eval manifest (base path must end in .json):
#   Run 1 → base path (e.g. val.json); runs 2+ → <base>_<run_name>.json
#   With --predictions_json results/captions/val.json you get val.json, val_lstm_25pct.json, …
python metrics.py --manifest configs/eval_manifest.example.yaml --split val \
  --captions_only --predictions_json results/captions/val.json --data_dir ./data
```

Download the JSON files (and keep the same `dataset_coco.json` / `--split` when scoring locally). Filenames are documented in `configs/eval_predictions_manifest.example.yaml`.

---

### Metrics from saved prediction JSON (no GPU)

Use when captions were generated elsewhere. Expects the same format as `--predictions_json` output. Paths in the manifest must match files under your machine (e.g. `results/captions/`).

```bash
# One file (example: first manifest run is saved as val.json)
python metrics.py --from_predictions results/captions/val.json \
  --data_dir ./data --split val \
  --model_type lstm --eval_train_fraction 0.1 --run_name lstm_10pct

# All saved caption files (see configs/eval_predictions_manifest.example.yaml)
python metrics.py --manifest configs/eval_predictions_manifest.example.yaml \
  --data_dir ./data --split val \
  --output results/metrics_from_captions.csv --print_regime_table
```

Manifest rows use `predictions:` instead of `checkpoint:` and include `model_type` for regime tables.

---

### Git workflow

Use feature branches and merge to `main` when work is complete.
