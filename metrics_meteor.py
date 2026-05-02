"""
Same CLI and checkpoint/manifest flow as metrics.py, but computes METEOR only.

Examples:
  python metrics_meteor.py --checkpoint checkpoints/lstm_10pct_best.pth
  python metrics_meteor.py --manifest configs/eval_manifest.example.yaml \\
    --output results/meteor_only.csv --print_regime_table

  # Score captions generated elsewhere (e.g. server GPU) — needs same --data_dir/--split
  # and Karpathy annotations for references:
  python metrics_meteor.py --predictions preds_val.json --split val

Requires Java on PATH for METEOR (pycocoevalcap meteor-1.5.jar).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from metrics import (
    _json_printable,
    _row_from_run,
    build_encoder_decoder,
    generate_predictions,
    load_karpathy_split_index,
    write_csv,
)
from utils.caption_metrics import compute_meteor_only


def evaluate_checkpoint_meteor(
    checkpoint_path: str,
    ann_file: str,
    images_root: str,
    split: str,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, float]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    references, items = load_karpathy_split_index(ann_file, images_root, split)
    encoder, decoder, vocab, _ = build_encoder_decoder(ckpt, device)
    preds = generate_predictions(
        encoder, decoder, vocab, items, device, batch_size, max_length
    )
    try:
        meteor = compute_meteor_only(references, preds)
    except Exception:
        meteor = float("nan")
    metrics = {"METEOR": meteor}
    return ckpt, preds, metrics


def print_regime_table(rows: List[Dict[str, Any]]) -> None:
    by_model: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        mt = str(r.get("model_type", "unknown"))
        by_model[mt].append(r)

    metric_keys = ["METEOR"]

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
        description="METEOR-only metrics (same flags as metrics.py, METEOR only)"
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
        "--print_regime_table",
        action="store_true",
        help="Pretty-print METEOR grouped by model_type vs train_fraction",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        metavar="PATH",
        help="JSON list [{image_id, caption}, ...] from metrics.py (no GPU; uses references from --data_dir)",
    )
    args = parser.parse_args()

    ann_file = os.path.join(args.data_dir, "annotations", "dataset_coco.json")
    images_root = os.path.join(args.data_dir, "images")

    if args.predictions:
        if args.checkpoint or args.manifest:
            parser.error("--predictions cannot be used with --checkpoint or --manifest")
        if not os.path.isfile(args.predictions):
            print(f"Not found: {args.predictions}", file=sys.stderr)
            sys.exit(1)
        references, _ = load_karpathy_split_index(ann_file, images_root, args.split)
        with open(args.predictions, "r", encoding="utf-8") as f:
            preds = json.load(f)
        try:
            meteor = compute_meteor_only(references, preds)
        except Exception as e:
            print(str(e), file=sys.stderr)
            meteor = float("nan")
        print(json.dumps(_json_printable({"METEOR": meteor}), indent=2))
        sys.exit(0)

    if not args.checkpoint and not args.manifest:
        parser.error("Provide --checkpoint, --manifest, or --predictions")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    all_rows: List[Dict[str, Any]] = []
    run_count = 0

    def run_one(cp_path: str, name: Optional[str], frac_override: Optional[float]):
        nonlocal run_count
        ckpt, preds, metrics = evaluate_checkpoint_meteor(
            cp_path,
            ann_file=ann_file,
            images_root=images_root,
            split=args.split,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
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
