import argparse
import json
import time
from typing import Dict

import torch

from moe_routing_bench.routers import pack_by_expert_torch, unpack_by_expert_torch
from moe_routing_bench.utils import BYTES_PER_SCALAR, bytes_per_token_pack_combine_strict, gib_per_s


def compute_bandwidth_metrics(
    tokens: int,
    hidden_dim: int,
    top_k: int,
    pack_ms: float,
    combine_ms: float,
    dtype: str,
) -> Dict[str, float]:
    total_ms = max(pack_ms + combine_ms, 1e-6)
    bytes_per_token = bytes_per_token_pack_combine_strict(hidden_dim, top_k, BYTES_PER_SCALAR[dtype])
    bw_gib = gib_per_s(tokens, bytes_per_token, total_ms)
    return {
        "bw_GiBps_strict": bw_gib,
        "bytes_per_token": bytes_per_token,
    }


def run_bench(args: argparse.Namespace) -> Dict[str, float]:
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    tokens, hidden, num_experts, top_k = (
        args.num_tokens,
        args.hidden_dim,
        args.num_experts,
        args.top_k,
    )
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    logits = torch.randn(tokens, num_experts, device=device, dtype=torch.float32)
    gate_vals, assign = torch.topk(logits, k=top_k, dim=-1)
    assign = assign.to(torch.int64)
    gate_weights = torch.softmax(gate_vals, dim=-1).to(dtype)

    for _ in range(args.warmup):
        packed, meta = pack_by_expert_torch(x, assign, num_experts)
        _ = unpack_by_expert_torch(packed, meta, gate_weights)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        packed, meta = pack_by_expert_torch(x, assign, num_experts)
        _ = unpack_by_expert_torch(packed, meta, gate_weights)
    torch.cuda.synchronize()
    t1 = time.time()

    avg_ms = (t1 - t0) * 1e3 / args.iters

    # For per-phase metrics, rerun once with explicit timing
    torch.cuda.synchronize()
    start_pack = torch.cuda.Event(True)
    end_pack = torch.cuda.Event(True)
    start_combine = torch.cuda.Event(True)
    end_combine = torch.cuda.Event(True)

    start_pack.record()
    packed, meta = pack_by_expert_torch(x, assign, num_experts)
    end_pack.record(); end_pack.synchronize()

    start_combine.record()
    _ = unpack_by_expert_torch(packed, meta, gate_weights)
    end_combine.record(); end_combine.synchronize()

    pack_ms = start_pack.elapsed_time(end_pack)
    combine_ms = start_combine.elapsed_time(end_combine)

    counts = meta[1].to(torch.float32)
    mean_load = counts.mean().item()
    std_load = counts.std(unbiased=False).item()
    cv = std_load / (mean_load + 1e-9)

    bw_metrics = compute_bandwidth_metrics(tokens, hidden, top_k, pack_ms, combine_ms, args.dtype)

    return {
        "name": "pack_combine_torch_baseline",
        "avg_ms": avg_ms,
        "pack_ms": pack_ms,
        "combine_ms": combine_ms,
        "tokens_per_s": tokens / (avg_ms / 1e3),
        **bw_metrics,
        "load_mean": mean_load,
        "load_std": std_load,
        "load_cv": cv,
        "num_tokens": tokens,
        "hidden_dim": hidden,
        "num_experts": num_experts,
        "top_k": top_k,
        "dtype": str(dtype).replace("torch.", ""),
        "bytes_per_token_strict": bw_metrics["bytes_per_token"],
        "device": str(device),
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch baseline pack+combine benchmark")
    parser.add_argument("--num-tokens", type=int, default=16384)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_bench(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
