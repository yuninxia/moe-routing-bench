#!/usr/bin/env python3
"""Download TinyStories from HuggingFace and dump plain-text files."""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit("Missing dependency: pip install datasets") from exc


def export_split(split: str, out_path: Path, limit: int | None = None) -> None:
    ds = load_dataset("roneneldan/TinyStories", split=split, streaming=False)
    count = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            f.write(text.replace("\r\n", "\n") + "\n\n")
            count += 1
            if limit is not None and count >= limit:
                break
    print(f"wrote {count} samples to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export TinyStories to plain text")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--out", type=str, default="tinystories_train.txt")
    parser.add_argument("--limit", type=int, default=None, help="optional number of samples to dump")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_split(args.split, Path(args.out), args.limit)


if __name__ == "__main__":
    main()
