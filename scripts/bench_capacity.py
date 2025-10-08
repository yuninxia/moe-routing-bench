import argparse
import json
import time

import torch

from moe_routing_bench.routers.pack_capacity_torch import (
    combine_from_packed_torch,
    pack_by_expert_with_capacity_torch,
)
from moe_routing_bench.utils import (
    BYTES_PER_SCALAR,
    bytes_per_token_pack_combine_strict,
    gib_per_s,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack/combine benchmark with capacity factor and drop rate")
    parser.add_argument("--num-tokens", type=int, default=16384)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--capacity-factors", type=str, default="1.00,1.10,1.25,1.50")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    torch.manual_seed(17)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

    num_tokens = args.num_tokens
    hidden_dim = args.hidden_dim
    num_experts = args.num_experts
    top_k = args.top_k

    x = torch.randn(num_tokens, hidden_dim, device=args.device, dtype=dtype)
    logits = torch.randn(num_tokens, num_experts, device=args.device, dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    gate_vals, expert_idx = torch.topk(probs, k=top_k, dim=-1)

    cf_list = [float(x) for x in args.capacity_factors.split(",") if x]

    for _ in range(args.warmup):
        pack_by_expert_with_capacity_torch(x, expert_idx, num_experts=num_experts, capacity_factor=cf_list[0])

    dtype_bytes = BYTES_PER_SCALAR[args.dtype]
    bytes_per_token = bytes_per_token_pack_combine_strict(hidden_dim, top_k, dtype_bytes)

    results = []
    for cf in cf_list:
        torch.cuda.synchronize()
        t0 = time.time()
        drop_acc = 0.0
        for _ in range(args.iters):
            pack_res = pack_by_expert_with_capacity_torch(
                x,
                expert_idx,
                num_experts=num_experts,
                capacity_factor=cf,
            )
            combine_from_packed_torch(
                pack_res.x_packed,
                pack_res.token_ids,
                gate_w_kept=None,
                num_tokens=num_tokens,
                reduce="sum",
            )
            drop_acc += pack_res.drop_rate
        torch.cuda.synchronize()
        avg_ms = (time.time() - t0) * 1000.0 / args.iters
        result = {
            "name": "pack_capacity_torch",
            "avg_ms": avg_ms,
            "tokens_per_s": num_tokens / (avg_ms / 1000.0),
            "bw_GiBps_strict": gib_per_s(num_tokens, bytes_per_token, avg_ms),
            "num_tokens": num_tokens,
            "hidden_dim": hidden_dim,
            "num_experts": num_experts,
            "top_k": top_k,
            "dtype": args.dtype,
            "device": args.device,
            "capacity_factor": cf,
            "avg_drop_rate": drop_acc / args.iters,
            "bytes_per_token_strict": bytes_per_token,
        }
        print(json.dumps(result, ensure_ascii=False))
        results.append(result)


if __name__ == "__main__":
    main()
