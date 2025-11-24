#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt


def load_records(path: str) -> Iterable[dict]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _expert_key(rec: dict) -> Tuple[int | None, int | None]:
    # Handle both bench_capacity (num_experts/top_k) and sweep outputs (experts/k).
    e = rec.get("experts", rec.get("num_experts"))
    k = rec.get("k", rec.get("top_k"))
    return e, k


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CF sweep metrics")
    parser.add_argument("--input", type=str, default="results/capacity_ke.jsonl")
    parser.add_argument("--metric", type=str, default="avg_drop_rate")
    parser.add_argument("--out", type=str, default="results/capacity_plot.png")
    args = parser.parse_args()

    grouped = defaultdict(list)
    for rec in load_records(args.input):
        key = _expert_key(rec)
        grouped[key].append((rec.get("capacity_factor"), rec.get(args.metric)))

    if not grouped:
        raise SystemExit(f"No records found in {args.input}")

    for (experts, k), points in sorted(grouped.items()):
        points = [(x, y) for x, y in points if x is not None and y is not None]
        if not points:
            continue
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=f"E={experts}, K={k}")

    plt.xlabel("Capacity Factor")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric} vs CF")
    plt.legend()
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
