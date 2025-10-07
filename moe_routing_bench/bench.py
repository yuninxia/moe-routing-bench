import json
from dataclasses import asdict
from typing import Any, Dict

import torch

from .config import BenchConfig
from .routing import (
    GatingLinear,
    make_fake_batch,
    pack_by_expert,
    topk_logits_then_softmax_over_k,
    unpack_to_tokens_and_combine,
)
from .utils import get_dtype, measure_latency_ms


def bench_topk(cfg: BenchConfig, impl: str = "torch") -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dtype = get_dtype(cfg.dtype)

    x = make_fake_batch(cfg.num_tokens, cfg.hidden_dim, cfg.dtype, device)
    gating = GatingLinear(cfg.hidden_dim, cfg.num_experts, dtype=dtype, device=device)
    logits = gating(x)

    def run_once():
        indices, weights = topk_logits_then_softmax_over_k(logits, cfg.top_k, impl=impl)
        torch._assert(indices.shape == (cfg.num_tokens, cfg.top_k), "unexpected index shape")
        return indices, weights

    avg_ms, std_ms = measure_latency_ms(run_once, warmup=cfg.warmup, iters=cfg.iters)
    tokens_per_s = cfg.num_tokens / (avg_ms / 1e3)

    return {
        "name": "topk",
        "impl": impl,
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "tokens_per_s": tokens_per_s,
        **asdict(cfg),
    }


def bench_routing_identity(cfg: BenchConfig, impl_topk: str = "torch") -> Dict[str, Any]:
    torch.manual_seed(cfg.seed)
    device = cfg.device
    dtype = get_dtype(cfg.dtype)

    x = make_fake_batch(cfg.num_tokens, cfg.hidden_dim, cfg.dtype, device)
    gating = GatingLinear(cfg.hidden_dim, cfg.num_experts, dtype=dtype, device=device)
    logits = gating(x)
    indices, weights = topk_logits_then_softmax_over_k(logits, cfg.top_k, impl=impl_topk)

    def run_once():
        packed, order, counts, offsets = pack_by_expert(x, indices, cfg.num_experts)
        _ = counts, offsets
        y = unpack_to_tokens_and_combine(packed, order, cfg.num_tokens, cfg.top_k, weights)
        return y

    avg_ms, std_ms = measure_latency_ms(run_once, warmup=cfg.warmup, iters=cfg.iters)
    tokens_per_s = cfg.num_tokens / (avg_ms / 1e3)

    return {
        "name": "routing_identity",
        "impl_topk": impl_topk,
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "tokens_per_s": tokens_per_s,
        **asdict(cfg),
    }


def save_jsonl(record: Dict[str, Any], path: str) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
