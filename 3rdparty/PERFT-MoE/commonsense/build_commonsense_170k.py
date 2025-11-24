#!/usr/bin/env python
"""Merge commonsense benchmarks into commonsense_170k.json."""

from __future__ import annotations

import argparse
import json
import os
from typing import List

from dataset_builders import DATASET_REGISTRY, ORDERED_DATASETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="commonsense_170k.json",
        help="Destination JSON file for the merged dataset.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Split to export from each dataset (default: train).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records: List[dict] = []

    for dataset_key in ORDERED_DATASETS:
        builder = DATASET_REGISTRY[dataset_key]["builder"]
        subset = builder(split=args.split)
        print(f"Collected {len(subset)} samples from {dataset_key} ({args.split}).")
        records.extend(subset)

    output_path = args.output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    print(f"Wrote {len(records)} examples to {output_path}")


if __name__ == "__main__":
    main()
