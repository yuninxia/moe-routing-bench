#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from itertools import product

import torch

from moe_routing_bench.routers.base import get_router
from moe_routing_bench.topk_impls import softk_indices_and_gates, top1_indices
from moe_routing_bench.utils import (
    BYTES_PER_SCALAR,
    bytes_per_token_pack_combine_strict,
    gib_per_s,
)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _append_jsonl(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _prepare_routing(args, logits: torch.Tensor, experts: int, k: int):
    if args.strategy == "top1":
        idx = top1_indices(logits)
        gates = None
        k = 1
    elif args.strategy == "topk_hard":
        idx = torch.topk(logits, k, dim=-1).indices
        gates = None
    else:
        idx, gates = softk_indices_and_gates(
            logits,
            k,
            temperature=args.temperature,
            normalize=args.soft_normalize,
        )
    return idx, gates, k


def run_case(args, tokens: int, experts: int, hidden: int, k: int, cf: float, backend: str):
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    device = torch.device(args.device)

    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    logits = torch.randn(tokens, experts, device=device, dtype=torch.float32)
    topk_idx, gates, k_eff = _prepare_routing(args, logits, experts, k)

    expected = tokens * k_eff / max(1, experts)
    capacity = max(1, int(cf * expected + 0.9999))

    router = get_router(backend)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.time()
    packed, route = router.pack(x, topk_idx, gates, capacity, renorm_after_drop=args.renorm_after_drop)
    y = packed
    out = router.combine(y, route, out_tokens=tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.time() - t0) * 1000.0

    avg_drop = 1.0 - (route.kept_mask.sum().item()) / max(1, tokens * k_eff)
    token_drop = float((~route.kept_mask).all(dim=1).float().mean().item()) if tokens > 0 else 0.0

    dtype_bytes = BYTES_PER_SCALAR[args.dtype]
    bytes_per_token = bytes_per_token_pack_combine_strict(hidden, k_eff, dtype_bytes)
    bw = gib_per_s(tokens, bytes_per_token, route.kept_mask.numel() and elapsed_ms or elapsed_ms)

    return {
        "timestamp": time.time(),
        "git_sha": _git_commit(),
        "device": args.device,
        "backend": backend,
        "strategy": args.strategy,
        "renorm_after_drop": args.renorm_after_drop,
        "tokens": tokens,
        "experts": experts,
        "hidden": hidden,
        "k": k_eff,
        "requested_k": k,
        "capacity_factor": cf,
        "capacity": capacity,
        "avg_drop_rate": avg_drop,
        "token_drop_rate": token_drop,
        "elapsed_ms": elapsed_ms,
        "bw_GiBps_strict": bw,
        "bytes_per_token_strict": bytes_per_token,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capacity/K/E sweep for routing backends")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--experts", type=str, default="32,64,128")
    parser.add_argument("--topk", type=str, default="1,2,4")
    parser.add_argument("--capacity-factors", type=str, default="1.0,1.1,1.25,1.5")
    parser.add_argument("--backend", type=str, default="torch_soft", choices=["torch_soft"])
    parser.add_argument("--strategy", type=str, default="softk", choices=["top1", "topk_hard", "softk"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--soft_normalize", type=str, default="softmax", choices=["softmax", "sum"])
    parser.add_argument("--renorm-after-drop", action="store_true", dest="renorm_after_drop")
    parser.add_argument("--output", type=str, default="results/capacity_ke.jsonl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokens = args.tokens
    hidden = args.hidden_dim
    experts_list = [int(x) for x in args.experts.split(",") if x]
    k_list = [int(x) for x in args.topk.split(",") if x]
    cf_list = [float(x) for x in args.capacity_factors.split(",") if x]

    for experts, k, cf in product(experts_list, k_list, cf_list):
        rec = run_case(args, tokens, experts, hidden, k, cf, args.backend)
        _append_jsonl(args.output, rec)
        print(json.dumps(rec))


if __name__ == "__main__":
    main()
