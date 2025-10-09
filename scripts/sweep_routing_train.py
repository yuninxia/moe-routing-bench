#!/usr/bin/env python3
"""Routing/Capacity sweep launcher (PyTorch-only)."""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
from pathlib import Path
from typing import List


def _comma_list(raw: str, cast_type):
    if not raw:
        return []
    return [cast_type(item.strip()) for item in raw.split(",") if item.strip()]


def build_launch_command(args: argparse.Namespace, cfg: dict, nproc: int) -> List[str]:
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "scripts/train_small.py",
        "--distributed",
        "--device",
        "cuda",
        "--data",
        args.data,
        "--seq-len",
        str(args.seq_len),
        "--batch-size",
        str(args.batch_per_gpu),
        "--dim",
        str(args.dim),
        "--layers",
        str(args.layers),
        "--heads",
        str(args.heads),
        "--ffn-mult",
        str(args.ffn_mult),
        "--num-experts",
        str(cfg["num_experts"]),
        "--strategy",
        cfg["strategy"],
        "--top-k",
        str(cfg["top_k"]),
        "--capacity-factor",
        str(cfg["capacity_factor"]),
        "--dtype",
        args.dtype,
        "--lr",
        str(args.lr),
        "--warmup-steps",
        str(args.warmup_steps),
        "--max-steps",
        str(args.max_steps),
        "--eval-interval",
        str(args.eval_interval),
        "--seed",
        str(cfg["seed"]),
        "--outdir",
        cfg["outdir"],
        "--num-workers",
        str(args.num_workers),
    ]
    if args.grad_clip is not None:
        cmd.extend(["--grad-clip", str(args.grad_clip)])
    if args.weight_decay is not None:
        cmd.extend(["--weight-decay", str(args.weight_decay)])
    if args.dropout is not None:
        cmd.extend(["--dropout", str(args.dropout)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch routing/capacity sweeps (DDP)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-per-gpu", type=int, default=128)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--num-experts-list", type=str, default="64,128")
    parser.add_argument("--router-list", type=str, default="top1,topk-hard,softk")
    parser.add_argument("--topk-list", type=str, default="1,2,4")
    parser.add_argument("--capacity-factors", type=str, default="1.0,1.25,1.5")
    parser.add_argument("--seeds", type=str, default="17,18,19")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--outdir-root", type=str, default="runs/routing_sweep")
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    Path(args.outdir_root).mkdir(parents=True, exist_ok=True)

    num_experts = _comma_list(args.num_experts_list, int)
    routers = [r.strip() for r in args.router_list.split(",") if r.strip()]
    topks = _comma_list(args.topk_list, int)
    capacity_factors = _comma_list(args.capacity_factors, float)
    seeds = _comma_list(args.seeds, int)

    if not routers:
        raise ValueError("router-list must not be empty")

    if args.nproc_per_node is not None:
        nproc = args.nproc_per_node
    else:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        nproc = len([x for x in cuda_visible.split(",") if x.strip()]) or 1

    combos = []
    for E, router, topk, cf, seed in itertools.product(num_experts, routers, topks, capacity_factors, seeds):
        strategy = router.replace("-", "_")
        if strategy == "top1" and topk != 1:
            continue
        if strategy != "top1" and topk < 1:
            continue
        tag = f"R{router.replace('-', '')}_E{E}_K{topk}_CF{cf}_S{seed}"
        outdir = os.path.join(args.outdir_root, tag)
        combos.append({
            "num_experts": E,
            "strategy": router,
            "top_k": 1 if strategy == "top1" else topk,
            "capacity_factor": cf,
            "seed": seed,
            "outdir": outdir,
        })

    for cfg in combos:
        cmd = build_launch_command(args, cfg, nproc)
        print(">>>", " ".join(cmd))
        if args.dry_run:
            continue
        Path(cfg["outdir"]).mkdir(parents=True, exist_ok=True)
        with open(Path(cfg["outdir"]) / "launch_cmd.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
