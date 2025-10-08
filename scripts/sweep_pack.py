#!/usr/bin/env python3
import argparse
import datetime
import itertools
import json
import os
import subprocess
import sys


def run_once(num_tokens, hidden_dim, num_experts, top_k, iters, dtype):
    cmd = [
        sys.executable,
        "scripts/bench_pack.py",
        "--num-tokens",
        str(num_tokens),
        "--hidden-dim",
        str(hidden_dim),
        "--num-experts",
        str(num_experts),
        "--top-k",
        str(top_k),
        "--iters",
        str(iters),
        "--dtype",
        dtype,
    ]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep pack+combine benchmark grid")
    parser.add_argument("--tokens", type=str, default="8192,16384,32768")
    parser.add_argument("--hidden", type=str, default="2048,4096")
    parser.add_argument("--experts", type=str, default="32,64,128")
    parser.add_argument("--topk", type=str, default="1,2,4")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--out", type=str, default="results/pack.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    grids = {
        "num_tokens": [int(x) for x in args.tokens.split(",") if x],
        "hidden_dim": [int(x) for x in args.hidden.split(",") if x],
        "num_experts": [int(x) for x in args.experts.split(",") if x],
        "top_k": [int(x) for x in args.topk.split(",") if x],
    }

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as handle:
        for nt, hd, ne, k in itertools.product(
            grids["num_tokens"], grids["hidden_dim"], grids["num_experts"], grids["top_k"]
        ):
            res = run_once(nt, hd, ne, k, args.iters, args.dtype)
            res["ts"] = datetime.datetime.now().isoformat()
            try:
                sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            except subprocess.CalledProcessError:
                sha = "unknown"
            res["git_sha"] = sha
            line = json.dumps(res)
            print(line)
            handle.write(line + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
