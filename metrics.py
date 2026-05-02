"""
Evaluate one or more checkpoints on Karpathy val/test split using BLEU (1-4),
METEOR (optional Java), CIDEr, and ROUGE-L.

Compare training-data regimes (e.g. 10%, 20%, 100%) by training separate
checkpoints with training.train_fraction set in the YAML config, then list
those checkpoints in a manifest (see configs/eval_manifest.example.yaml).

Examples:
  python metrics.py --checkpoint checkpoints/transformer_best.pth
  python metrics.py --manifest configs/eval_manifest.example.yaml --output results/metrics.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.attention_lstm import LSTMAttention
from models.baseline_lstm import DecoderRNN
from models.encoder import EncoderCNN
from models.transformer import TransformerDecoder
from utils.caption_metrics import compute_caption_metrics
from utils.dataset import Vocabulary
from utils.transforms import get_transforms


def load_karpathy_split_index(
    ann_file: str,
    root_dir: str,
    split: str,
) -> Tuple[Dict[int, List[str]], List[Dict[str, Any]]]:
    with open(ann_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    references: Dict[int, List[str]] = {}
    items: List[Dict[str, Any]] = []

    for img in data["images"]:
        if img["split"] != split:
            continue
        img_id = int(img.get("imgid", img.get("cocoid")))
        path = os.path.join(root_dir, img["filepath"], img["filename"])
        refs = [s["raw"] for s in img["sentences"]]
        references[img_id] = refs
        items.append({"image_id": img_id, "path": path})

    items.sort(key=lambda x: x["image_id"])
    return references, items


class _EvalImageDataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], transform):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        image = Image.open(it["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, it["image_id"]


def _collate_eval(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    img_ids = [int(b[1]) for b in batch]
    return imgs, img_ids


def ids_to_caption(sampled, vocab: Vocabulary) -> str:
    """Decode decoder.sample(...) output: tensor [1, T] or tuple (tensor, extras)."""
    if isinstance(sampled, tuple):
        sampled = sampled[0]
    flat_ids = sampled.reshape(-1).tolist()
    words: List[str] = []
    for tid in flat_ids:
        w = vocab.itos[int(tid)]
        if w == "<START>":
            continue
        if w in ("<END>", "<PAD>"):
            break
        words.append(w)
    return " ".join(words)


def build_encoder_decoder(checkpoint: Dict[str, Any], device: torch.device):
    config = checkpoint["model_config"]
    arch_cfg = config["architecture"]
    model_type = config["model_type"]
    vocab_size = int(checkpoint["vocab_size"])
    vocab = Vocabulary.from_saved(checkpoint["vocab_stoi"], checkpoint["vocab_itos"])
    pad_idx = vocab.stoi["<PAD>"]

    encoder = EncoderCNN(arch_cfg["embed_size"]).to(device)

    if model_type == "lstm":
        decoder = DecoderRNN(**arch_cfg, vocab_size=vocab_size).to(device)
    elif model_type == "attention":
        decoder = LSTMAttention(
            **arch_cfg,
            vocab_size=vocab_size,
            start_token=vocab.stoi["<START>"],
            stop_token=vocab.stoi["<END>"],
            seq_length=20,
        ).to(device)
    elif model_type == "transformer":
        decoder = TransformerDecoder(
            **arch_cfg,
            vocab_size=vocab_size,
            pad_idx=pad_idx,
            start_idx=vocab.stoi["<START>"],
            end_idx=vocab.stoi["<END>"],
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    decoder.load_state_dict(checkpoint["decoder_state_dict"])
    encoder.eval()
    decoder.eval()
    return encoder, decoder, vocab, model_type


@torch.no_grad()
def generate_predictions(
    encoder,
    decoder,
    vocab: Vocabulary,
    items: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> List[Dict[str, Any]]:
    transform = get_transforms("val")
    loader = DataLoader(
        _EvalImageDataset(items, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_eval,
    )
    predictions: List[Dict[str, Any]] = []
    for imgs, img_ids in tqdm(loader, desc="Generating captions"):
        imgs = imgs.to(device)
        features = encoder(imgs)
        for i in range(features.size(0)):
            feat = features[i : i + 1]
            sampled = decoder.sample(feat, max_length=max_length)
            cap = ids_to_caption(sampled, vocab)
            predictions.append({"image_id": img_ids[i], "caption": cap})
    return predictions


def evaluate_checkpoint(
    checkpoint_path: str,
    ann_file: str,
    images_root: str,
    split: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
    skip_rouge: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    references, items = load_karpathy_split_index(ann_file, images_root, split)
    encoder, decoder, vocab, _ = build_encoder_decoder(ckpt, device)
    preds = generate_predictions(
        encoder, decoder, vocab, items, device, batch_size, max_length
    )
    metrics = compute_caption_metrics(
        references,
        preds,
        include_rouge=not skip_rouge,
    )
    return ckpt, preds, metrics


def _row_from_run(
    name: Optional[str],
    checkpoint_path: str,
    ckpt: Dict[str, Any],
    metrics: Dict[str, float],
    train_fraction_override: Optional[float],
) -> Dict[str, Any]:
    config = ckpt.get("model_config", {})
    model_type = config.get("model_type", "")
    train_fraction = train_fraction_override
    if train_fraction is None:
        train_fraction = ckpt.get("train_fraction")
    row: Dict[str, Any] = {
        "name": name or os.path.basename(checkpoint_path),
        "checkpoint": checkpoint_path,
        "model_type": model_type,
        "train_fraction": train_fraction if train_fraction is not None else "",
    }
    row.update(metrics)
    return row


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _json_printable(row: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(row)
    for k, v in list(out.items()):
        if isinstance(v, float) and math.isnan(v):
            out[k] = None
    return out


def print_regime_table(rows: List[Dict[str, Any]]) -> None:
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        mt = str(r.get("model_type", "unknown"))
        by_model[mt].append(r)

    metric_keys = ["Bleu_4", "METEOR", "CIDEr", "ROUGE_L"]

    def fmt_cell(v):
        if v == "" or v is None:
            return ""
        if isinstance(v, float) and math.isnan(v):
            return "nan"
        if isinstance(v, (int, float)):
            return f"{float(v):.4f}"
        return str(v)

    for mt, mrows in sorted(by_model.items()):
        print(f"\n=== {mt} ===")
        header = ["train_fraction"] + metric_keys
        print("\t".join(header))

        def frac_sort_key(r):
            f = r.get("train_fraction")
            if f == "" or f is None:
                return (1, 0.0)
            return (0, float(f))

        mrows_sorted = sorted(mrows, key=frac_sort_key)
        for r in mrows_sorted:
            frac = r.get("train_fraction", "")
            vals = [fmt_cell(r.get(k)) for k in metric_keys]
            print(f"{frac}\t" + "\t".join(vals))


def main():
    parser = argparse.ArgumentParser(
        description="Caption metrics (BLEU, METEOR, CIDEr, ROUGE-L)"
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Single .pth checkpoint")
    parser.add_argument("--manifest", type=str, default=None, help="YAML with a list of runs")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.getcwd(), "data"),
        help="Project data directory (expects images/ and annotations/)",
    )
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional CSV path for aggregated results (manifest mode)",
    )
    parser.add_argument(
        "--predictions_json",
        type=str,
        default=None,
        help="Optional path to save predictions for the last (or only) run",
    )
    parser.add_argument(
        "--skip_rouge",
        action="store_true",
        help="Skip ROUGE-L",
    )
    parser.add_argument(
        "--print_regime_table",
        action="store_true",
        help="Pretty-print metrics grouped by model_type vs train_fraction",
    )
    args = parser.parse_args()

    if not args.checkpoint and not args.manifest:
        parser.error("Provide --checkpoint or --manifest")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ann_file = os.path.join(args.data_dir, "annotations", "dataset_coco.json")
    images_root = os.path.join(args.data_dir, "images")

    all_rows: List[Dict[str, Any]] = []
    run_count = 0

    def run_one(cp_path: str, name: Optional[str], frac_override: Optional[float]):
        nonlocal run_count
        ckpt, preds, metrics = evaluate_checkpoint(
            cp_path,
            ann_file=ann_file,
            images_root=images_root,
            split=args.split,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            skip_rouge=args.skip_rouge,
        )
        row = _row_from_run(name, cp_path, ckpt, metrics, frac_override)
        all_rows.append(row)
        print(json.dumps(_json_printable(row), indent=2))
        run_count += 1
        if args.predictions_json:
            out_path = args.predictions_json
            if run_count > 1 and out_path.endswith(".json"):
                base, ext = os.path.splitext(out_path)
                tag = name or os.path.splitext(os.path.basename(cp_path))[0]
                out_path = f"{base}_{tag}{ext}"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(preds, f, indent=2)

    if args.checkpoint:
        run_one(args.checkpoint, name=None, frac_override=None)

    if args.manifest:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = yaml.safe_load(f)
        runs = manifest.get("runs", [])
        for run in runs:
            cp = run["checkpoint"]
            name = run.get("name")
            frac = run.get("train_fraction")
            run_one(cp, name=name, frac_override=frac)

    if args.output:
        write_csv(all_rows, args.output)

    if args.print_regime_table and all_rows:
        print_regime_table(all_rows)


if __name__ == "__main__":
    main()
