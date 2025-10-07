from typing import Tuple

import torch
import torch.nn as nn

from .topk_impls import run_topk
from .utils import get_dtype


class GatingLinear(nn.Module):
    """Simple gating layer producing expert logits."""

    def __init__(self, hidden_dim: int, num_experts: int, dtype=torch.float16, device: str = "cuda"):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_experts, bias=True, device=device, dtype=dtype)
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def topk_logits_then_softmax_over_k(logits: torch.Tensor, k: int, impl: str = "torch") -> Tuple[torch.Tensor, torch.Tensor]:
    """Select top-k experts per token and produce normalized combine weights."""

    values, indices = run_topk(logits, k, impl=impl)
    weights = torch.softmax(values, dim=-1).to(values.dtype)
    return indices, weights


@torch.no_grad()
def pack_by_expert(
    x: torch.Tensor,
    topk_idx: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized sort-and-pack dispatch for expert inputs."""

    tokens, hidden = x.shape
    k = topk_idx.shape[1]
    flat_eids = topk_idx.reshape(-1)
    order = torch.argsort(flat_eids)
    x_rep = x.repeat_interleave(k, dim=0)
    packed = x_rep.index_select(0, order)
    counts = torch.bincount(flat_eids, minlength=num_experts)
    offsets = torch.cumsum(counts, dim=0)
    return packed, order, counts, offsets


@torch.no_grad()
def unpack_to_tokens_and_combine(
    expert_outputs_sorted: torch.Tensor,
    order: torch.Tensor,
    num_tokens: int,
    k: int,
    combine_weights: torch.Tensor,
) -> torch.Tensor:
    """Undo expert packing and collapse expert dimension with combine weights."""

    entries, hidden = expert_outputs_sorted.shape
    assert entries == num_tokens * k

    inv = torch.empty_like(order)
    inv[order] = torch.arange(entries, device=order.device)
    unsorted = expert_outputs_sorted.index_select(0, inv)
    unsorted = unsorted.view(num_tokens, k, hidden)
    y = (unsorted * combine_weights.unsqueeze(-1)).sum(dim=1)
    return y


def route_identity_pipeline(
    x: torch.Tensor,
    num_experts: int,
    topk_idx: torch.Tensor,
    combine_weights: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    """Run routing path assuming identity experts for correctness checks."""

    num_tokens, _ = x.shape
    k = topk_idx.shape[1]
    packed, order, counts, offsets = pack_by_expert(x, topk_idx, num_experts)
    expert_outputs_sorted = packed
    y = unpack_to_tokens_and_combine(expert_outputs_sorted, order, num_tokens, k, combine_weights)
    stats = {
        "counts": counts,
        "offsets": offsets,
    }
    return y, stats


def end2end_identity_from_logits(
    x: torch.Tensor,
    logits: torch.Tensor,
    k: int,
    impl_topk: str = "torch",
) -> torch.Tensor:
    indices, weights = topk_logits_then_softmax_over_k(logits, k=k, impl=impl_topk)
    y, _ = route_identity_pipeline(x, logits.shape[1], indices, weights)
    return y


def make_fake_batch(num_tokens: int, hidden: int, dtype: str = "float16", device: str = "cuda") -> torch.Tensor:
    dt = get_dtype(dtype)
    generator = torch.Generator(device=device).manual_seed(42)
    return torch.randn(num_tokens, hidden, device=device, dtype=dt, generator=generator)
