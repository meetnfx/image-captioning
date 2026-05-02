# Image Captioning Project (Comparison of CNN-LSTM, Attention, and Transformers)
A PyTorch implementation comparing sequence generation architectures on the MS COCO dataset.





## Project Environment
* **Primary Python Version:** 3.10
* **Dataset:** MS COCO (Karpathy Splits)
* **Hardware Support:** Multi-platform (Cloud TPU/GPU & Local CUDA)
# important Dependency 
We have different hardware environments, so we separated our dependencies from pytorch
* **`torch` & `torchvision`:** The core deep learning framework and pre-trained CNNs (ResNet).
* **`pycocotools` & `pycocoevalcap`:** Needed for parsing MS COCO's massive JSON annotations and calculating standard image captioning metrics (BLEU, METEOR, CIDEr).
* **`Pillow`:** Backend image processing required before feeding images to PyTorch.
* **`pyyaml`:** Used to read our `.yaml` hyperparameter configuration files.

# Setup
**Important:** Please do not commit the MS COCO dataset or `.pth` model weights to GitHub. Place datasets inside `/data` and weights inside `/checkpoints`.
### Cloud use ( Colab)
These platforms already have optimized GPU drivers, `torch`, and `torchvision` pre-installed. 
Clone this repository.
Run pip install -r requirements.txt
  
### Local
Clone the repo
create and activate a Conda environment with correct python version 3.10
install pytorch and cuda depending on gpu
pip install -r requirements.txt

### Datasets (either run script for Linux or Manually)


After downloading (script or manual), you should have:

```text
data/
  annotations/
    dataset_coco.json      # Karpathy annotations
  images/
    train2014/             # COCO train images
    val2014/               # COCO val images
```

script (linux)
cd scripts
bash download_coco.sh 
OR manually downlaod the datasets (windows or if the script doesnt work)
Click the links below to download the dataset zips via your browser. 
1. [Karpathy Splits JSON (caption_datasets.zip)](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip)
2. [MS COCO Train 2014 (train2014.zip)](http://images.cocodataset.org/zips/train2014.zip)
3. [MS COCO Val 2014 (val2014.zip)](http://images.cocodataset.org/zips/val2014.zip)
Move the dataset_coco.json to data/annotations
Unzip Train 2014 into data/images/train2014
Unzip Val 2014 into data/images/val2014



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

Run these from the **repository root** unless noted. Default data root is `./data` 

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

### Github
when working on this, create feature branches and then merge to main when done.

