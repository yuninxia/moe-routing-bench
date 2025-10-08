from __future__ import annotations

from typing import Optional, Tuple

import torch

from .base import PackCombine, RouteInfo, register_router

Tensor = torch.Tensor


def _assign_slots_vectorized(topk_idx: torch.LongTensor, num_experts: int, capacity: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
    device = topk_idx.device
    tokens, k = topk_idx.shape
    if tokens == 0:
        return torch.empty_like(topk_idx), torch.zeros(num_experts, dtype=torch.long, device=device)

    flat_e = topk_idx.reshape(-1)
    order = torch.argsort(flat_e, stable=True)
    e_sorted = flat_e[order]
    total = e_sorted.numel()
    pos = torch.arange(total, device=device, dtype=torch.long)
    is_first = torch.zeros(total, dtype=torch.bool, device=device)
    is_first[0] = True
    is_first[1:] = e_sorted[1:] != e_sorted[:-1]
    group_id = torch.cumsum(is_first.to(torch.int64), dim=0) - 1
    first_pos = torch.zeros(group_id.max().item() + 1, dtype=torch.long, device=device)
    first_pos.index_copy_(0, group_id[is_first], pos[is_first])
    slot_sorted = pos - first_pos[group_id]

    kept_sorted = (slot_sorted < capacity).to(torch.int64)
    expert_counts = torch.zeros(num_experts, dtype=torch.long, device=device)
    expert_counts.index_add_(0, e_sorted, kept_sorted)

    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(total, device=device)
    slots = slot_sorted[inv_order].view(tokens, k)
    return slots, expert_counts


class TorchSoftPackCombine(PackCombine):
    name = "torch_soft"

    def pack(
        self,
        x: Tensor,
        topk_idx: torch.LongTensor,
        gates: Optional[Tensor],
        capacity: int,
        renorm_after_drop: bool,
    ) -> Tuple[Tensor, RouteInfo]:
        tokens, dim = x.shape
        device = x.device
        k = topk_idx.shape[1] if tokens > 0 else 0
        num_experts = int(topk_idx.max().item() + 1) if tokens > 0 else 0
        capacity = max(1, capacity)

        slots, expert_counts = _assign_slots_vectorized(topk_idx, num_experts, capacity)
        kept_mask = slots < capacity
        slots_clamped = slots.clone()
        slots_clamped[~kept_mask] = -1

        flat_e = topk_idx.reshape(-1)
        flat_slots = slots_clamped.reshape(-1)
        flat_tokens = torch.arange(tokens, device=device).unsqueeze(1).expand(tokens, k).reshape(-1)
        kept = flat_slots >= 0

        flat_out_index = flat_e[kept] * capacity + flat_slots[kept]
        flat_token_idx = flat_tokens[kept]

        if gates is None:
            flat_weights = None
        else:
            flat_weights = gates.reshape(-1)[kept]
            if renorm_after_drop and tokens > 0:
                denom = torch.zeros(tokens, dtype=gates.dtype, device=device)
                denom.index_add_(0, flat_token_idx, flat_weights)
                safe = denom[flat_token_idx]
                nonzero = safe > 0
                scaled = flat_weights.clone()
                scaled[nonzero] = scaled[nonzero] / safe[nonzero]
                flat_weights = scaled

        packed = x.new_zeros((num_experts * capacity, dim)) if num_experts > 0 else x.new_zeros((0, dim))
        if flat_out_index.numel() > 0:
            packed.index_copy_(0, flat_out_index, x[flat_token_idx])

        route = RouteInfo(
            tokens=tokens,
            experts=num_experts,
            k=k,
            capacity=capacity,
            topk_idx=topk_idx,
            gates=gates,
            slots=slots_clamped,
            kept_mask=kept_mask,
            expert_counts=expert_counts,
            renorm_after_drop=bool(gates is not None and renorm_after_drop),
            flat_token_idx=flat_token_idx,
            flat_out_index=flat_out_index,
            flat_weights=flat_weights,
        )
        return packed, route

    def combine(self, y: Tensor, route: RouteInfo, out_tokens: int) -> Tensor:
        out = y.new_zeros((out_tokens, y.shape[1]))
        if route.flat_out_index.numel() == 0:
            return out
        contrib = y[route.flat_out_index]
        if route.flat_weights is not None:
            contrib = contrib * route.flat_weights.unsqueeze(1)
        out.index_add_(0, route.flat_token_idx, contrib)
        return out


register_router(TorchSoftPackCombine.name, TorchSoftPackCombine)
