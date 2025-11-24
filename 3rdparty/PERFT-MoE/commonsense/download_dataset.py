#!/usr/bin/env python
"""Download and format an individual commonsense dataset."""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from dataset_builders import DATASET_REGISTRY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_dataset(dataset: str, split: str, output_path: str) -> None:
    builder = DATASET_REGISTRY[dataset]["builder"]
    records: List[dict] = builder(split=split)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Wrote {len(records)} records to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASET_REGISTRY.keys()),
        help="Dataset key to download (one of %(choices)s).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="HF split to export (default: train).",
    )
    parser.add_argument(
        "--output",
        help="Destination JSON file. Defaults to commonsense/raw/<dataset>_<split>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or os.path.join(BASE_DIR, "raw", f"{args.dataset}_{args.split}.json")
    download_dataset(args.dataset, args.split, output)


if __name__ == "__main__":
    main()
