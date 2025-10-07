import argparse
import json

from .bench import bench_routing_identity, bench_topk, save_jsonl
from .config import BenchConfig


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-tokens", type=int, default=16384)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--out", type=str, default="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="moe-routing-bench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    bench_topk_parser = sub.add_parser("bench-topk", help="Benchmark top-k selection")
    _add_common_args(bench_topk_parser)
    bench_topk_parser.add_argument("--impl", type=str, default="torch", choices=["torch", "quack"])

    bench_routing_parser = sub.add_parser("bench-routing", help="Benchmark dispatch and combine")
    _add_common_args(bench_routing_parser)
    bench_routing_parser.add_argument("--impl-topk", type=str, default="torch", choices=["torch", "quack"])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BenchConfig(
        num_tokens=args.num_tokens,
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        dtype=args.dtype,
        device=args.device,
        seed=args.seed,
        warmup=args.warmup,
        iters=args.iters,
    )

    if args.cmd == "bench-topk":
        record = bench_topk(cfg, impl=args.impl)
    elif args.cmd == "bench-routing":
        record = bench_routing_identity(cfg, impl_topk=args.impl_topk)
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")

    print(json.dumps(record, indent=2))
    if args.out:
        save_jsonl(record, args.out)


if __name__ == "__main__":
    main()
