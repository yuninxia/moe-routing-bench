from __future__ import annotations

from typing import Tuple

import torch


def pack_by_expert_torch(
    x: torch.Tensor,
    assign: torch.Tensor,
    num_experts: int,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int]]:
    """Vectorized pack that groups token entries by expert id using sorting."""

    tokens, hidden = x.shape
    top_k = assign.shape[1]
    expert_ids = assign.reshape(-1)
    x_repeat = x.repeat_interleave(top_k, dim=0)
    order = torch.argsort(expert_ids, stable=True)
    packed = x_repeat.index_select(0, order)
    counts = torch.bincount(expert_ids, minlength=num_experts)
    starts = torch.cumsum(counts, dim=0) - counts
    meta = (order, counts, starts, tokens, top_k, hidden, num_experts)
    return packed, meta


def unpack_by_expert_torch(
    packed_outputs: torch.Tensor,
    meta: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int],
    gate_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Inverse operation of :func:`pack_by_expert_torch` with optional combine weights."""

    order, counts, starts, tokens, top_k, hidden, num_experts = meta
    flat_size = tokens * top_k

    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)
    restored = packed_outputs.index_select(0, inv).reshape(tokens, top_k, hidden)
    if gate_weights is not None:
        restored = restored * gate_weights.unsqueeze(-1)
    return restored.sum(dim=1).contiguous()
