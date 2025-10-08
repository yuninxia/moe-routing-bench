#!/usr/bin/env python3
import argparse
import json
import time

import torch

from moe_routing_bench.routers.base import get_router
from moe_routing_bench.topk_impls import softk_indices_and_gates, top1_indices
from moe_routing_bench.utils import (
    BYTES_PER_SCALAR,
    bytes_per_token_pack_combine_strict,
    gib_per_s,
)


def prepare_topk(args, logits: torch.Tensor):
    if args.strategy == "top1":
        topk_idx = top1_indices(logits)
        gates = None
        k_eff = 1
    elif args.strategy == "topk_hard":
        topk_idx = torch.topk(logits, args.top_k, dim=-1).indices
        gates = None
        k_eff = args.top_k
    else:
        topk_idx, gates = softk_indices_and_gates(
            logits,
            args.top_k,
            temperature=args.temperature,
            normalize=args.soft_normalize,
        )
        gates = gates.to(logits.dtype)
        k_eff = args.top_k
    return topk_idx, gates, k_eff


def bench_once(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    tokens = args.num_tokens
    hidden = args.hidden_dim
    experts = args.num_experts
    expand = args.expand

    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    logits = torch.randn(tokens, experts, device=device, dtype=torch.float32)

    topk_idx, gates, k_eff = prepare_topk(args, logits)
    expected = tokens * k_eff / max(1, experts)
    capacity = max(1, int(args.capacity_factor * expected + 0.9999))

    router = get_router(args.backend)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.time()
    packed, route = router.pack(x, topk_idx, gates, capacity, renorm_after_drop=args.renorm_after_drop)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    pack_ms = (time.time() - start) * 1000.0

    w1 = torch.randn(hidden, expand * hidden, device=device, dtype=dtype) / (hidden ** 0.5)
    w2 = torch.randn(expand * hidden, hidden, device=device, dtype=dtype) / ((expand * hidden) ** 0.5)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_ffn0 = time.time()
    y = torch.nn.functional.gelu(packed @ w1)
    y = y @ w2
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ffn_ms = (time.time() - t_ffn0) * 1000.0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t_combine0 = time.time()
    out = router.combine(y, route, out_tokens=tokens)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    combine_ms = (time.time() - t_combine0) * 1000.0

    total_ms = pack_ms + ffn_ms + combine_ms

    entries = tokens * k_eff
    flops = 2 * entries * hidden * (expand * hidden) * 2
    tflops = flops / (total_ms / 1e3) / 1e12 if total_ms > 0 else 0.0

    dtype_bytes = BYTES_PER_SCALAR[args.dtype]
    bytes_per_token = bytes_per_token_pack_combine_strict(hidden, k_eff, dtype_bytes)
    bw_strict = gib_per_s(tokens, bytes_per_token, pack_ms + combine_ms)

    avg_drop = 1.0 - (route.kept_mask.sum().item()) / max(1, tokens * k_eff)
    token_drop = float((~route.kept_mask).all(dim=1).float().mean().item()) if tokens > 0 else 0.0

    counts = route.expert_counts.to(torch.float32)
    load_mean = counts.mean().item() if counts.numel() > 0 else 0.0
    load_std = counts.std(unbiased=False).item() if counts.numel() > 0 else 0.0
    load_cv = load_std / (load_mean + 1e-9) if load_mean > 0 else 0.0

    return {
        "name": "moe_ffn_layer_microbench",
        "backend": args.backend,
        "strategy": args.strategy,
        "renorm_after_drop": args.renorm_after_drop,
        "router_ms": 0.0,  # logits->topk 在本脚本中不计入，便于区分
        "pack_ms": pack_ms,
        "ffn_ms": ffn_ms,
        "combine_ms": combine_ms,
        "total_ms": total_ms,
        "routing_share": (pack_ms + combine_ms) / total_ms if total_ms > 0 else 0.0,
        "tflops_total": tflops,
        "bw_GiBps_strict": bw_strict,
        "bytes_per_token_strict": bytes_per_token,
        "avg_drop_rate": avg_drop,
        "token_drop_rate": token_drop,
        "load_mean": load_mean,
        "load_cv": load_cv,
        "num_tokens": tokens,
        "hidden_dim": hidden,
        "num_experts": experts,
        "top_k": k_eff,
        "expand": expand,
        "capacity_factor": args.capacity_factor,
        "capacity": capacity,
        "dtype": args.dtype,
        "device": args.device,
        "seed": args.seed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="MoE FFN microbenchmark")
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--strategy", type=str, default="softk", choices=["top1", "topk_hard", "softk"])
    parser.add_argument("--expand", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--backend", type=str, default="torch_soft", choices=["torch_soft", "quack"])
    parser.add_argument("--capacity-factor", type=float, default=1.25, dest="capacity_factor")
    parser.add_argument("--renorm-after-drop", action="store_true", dest="renorm_after_drop")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--soft-normalize", type=str, default="softmax", choices=["softmax", "sum"], dest="soft_normalize")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    res = bench_once(args)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
