from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


def _exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x.new_zeros(1), torch.cumsum(x, dim=0)[:-1]])


@dataclass
class PackResult:
    x_packed: torch.Tensor
    token_ids: torch.Tensor
    expert_ids: torch.Tensor
    expert_ptr: torch.Tensor
    kept_counts: torch.Tensor
    counts: torch.Tensor
    cap: int
    drop_rate: float


@torch.no_grad()
def pack_by_expert_with_capacity_torch(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    num_experts: Optional[int] = None,
    capacity_factor: float = 1.0,
    sort_stable: bool = True,
) -> PackResult:
    """Torch reference pack that enforces per-expert capacity and returns compact CSR metadata."""

    assert x.dim() == 2, "x must be [N, H]"
    tokens, hidden = x.shape
    tokens2, top_k = expert_idx.shape
    assert tokens2 == tokens, "expert_idx must align with x"

    if num_experts is None:
        num_experts = int(expert_idx.max().item()) + 1

    device = x.device

    flat_e = expert_idx.reshape(-1).to(torch.long)
    flat_t = torch.arange(tokens, device=device, dtype=torch.long).repeat_interleave(top_k)

    counts = torch.bincount(flat_e, minlength=num_experts)
    avg_load = (tokens * top_k) / num_experts
    cap = int(torch.ceil(torch.tensor(capacity_factor * avg_load, device=device)).item())
    cap_vec = torch.full((num_experts,), cap, device=device, dtype=torch.long)

    order = torch.argsort(flat_e, stable=sort_stable)
    e_sorted = flat_e[order]
    t_sorted = flat_t[order]

    seg_starts = _exclusive_cumsum(counts)
    positions = torch.arange(e_sorted.numel(), device=device, dtype=torch.long) - seg_starts[e_sorted]

    keep_mask = positions < cap_vec[e_sorted]
    kept_order = order[keep_mask]
    drop_rate = float((~keep_mask).sum().item()) / float(flat_e.numel())

    kept_counts = torch.minimum(counts, cap_vec)
    expert_ptr = torch.cat([kept_counts.new_zeros(1), torch.cumsum(kept_counts, dim=0)])

    kept_tokens = t_sorted[keep_mask]
    kept_experts = e_sorted[keep_mask]
    x_packed = x[kept_tokens]

    return PackResult(
        x_packed=x_packed,
        token_ids=kept_tokens,
        expert_ids=kept_experts,
        expert_ptr=expert_ptr,
        kept_counts=kept_counts,
        counts=counts,
        cap=cap,
        drop_rate=drop_rate,
    )


@torch.no_grad()
def combine_from_packed_torch(
    y_packed: torch.Tensor,
    token_ids: torch.Tensor,
    gate_w_kept: Optional[torch.Tensor],
    num_tokens: int,
    reduce: Literal["sum", "weighted_sum"] = "sum",
) -> torch.Tensor:
    hidden = y_packed.shape[1]
    out = torch.zeros((num_tokens, hidden), device=y_packed.device, dtype=y_packed.dtype)
    if reduce == "sum" or gate_w_kept is None:
        out.index_add_(0, token_ids, y_packed)
    else:
        out.index_add_(0, token_ids, y_packed * gate_w_kept.unsqueeze(-1))
    return out
