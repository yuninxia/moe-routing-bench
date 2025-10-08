#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import matplotlib.pyplot as plt


def load_records(path: str):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CF sweep metrics")
    parser.add_argument("--input", type=str, default="results/capacity_ke.jsonl")
    parser.add_argument("--metric", type=str, default="avg_drop_rate")
    args = parser.parse_args()

    grouped = defaultdict(list)
    for rec in load_records(args.input):
        key = (rec.get("experts"), rec.get("k"))
        grouped[key].append((rec.get("capacity_factor"), rec.get(args.metric)))

    for (experts, k), points in sorted(grouped.items()):
        points.sort()
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", label=f"E={experts}, K={k}")

    plt.xlabel("Capacity Factor")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric} vs CF")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
