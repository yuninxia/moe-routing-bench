import argparse
import json
import time
from typing import Dict

import torch

from moe_routing_bench.routers import pack_by_expert_torch
from moe_routing_bench.utils import BYTES_PER_SCALAR, bytes_per_token_pack_only, gib_per_s


def compute_pack_bandwidth(tokens: int, hidden_dim: int, top_k: int, ms: float, dtype: str) -> Dict[str, float]:
    bytes_per_token = bytes_per_token_pack_only(hidden_dim, top_k, BYTES_PER_SCALAR[dtype])
    return {
        "bw_GiBps_strict": gib_per_s(tokens, bytes_per_token, ms),
        "bytes_per_token": bytes_per_token,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Torch pack-only benchmark")
    parser.add_argument("--num-tokens", type=int, default=16384)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Dict[str, float]:
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    tokens, hidden, num_experts, top_k = (
        args.num_tokens,
        args.hidden_dim,
        args.num_experts,
        args.top_k,
    )

    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    logits = torch.randn(tokens, num_experts, device=device, dtype=torch.float32)
    _, assign = torch.topk(logits, k=top_k, dim=-1)
    assign = assign.to(torch.int64)

    for _ in range(args.warmup):
        _ = pack_by_expert_torch(x, assign, num_experts)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(args.iters):
        _ = pack_by_expert_torch(x, assign, num_experts)
    torch.cuda.synchronize()
    t1 = time.time()

    avg_ms = (t1 - t0) * 1e3 / args.iters

    counts = pack_by_expert_torch(x, assign, num_experts)[1][1].to(torch.float32)
    mean_load = counts.mean().item()
    std_load = counts.std(unbiased=False).item()
    cv = std_load / (mean_load + 1e-9)

    bw = compute_pack_bandwidth(tokens, hidden, top_k, avg_ms, args.dtype)

    return {
        "name": "pack_only_torch",
        "avg_ms": avg_ms,
        "tokens_per_s": tokens / (avg_ms / 1e3),
        **bw,
        "load_mean": mean_load,
        "load_std": std_load,
        "load_cv": cv,
        "num_tokens": tokens,
        "hidden_dim": hidden,
        "num_experts": num_experts,
        "top_k": top_k,
        "dtype": str(dtype).replace("torch.", ""),
        "bytes_per_token_strict": bw["bytes_per_token"],
        "device": str(device),
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
    }


def main() -> None:
    args = parse_args()
    result = run(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
