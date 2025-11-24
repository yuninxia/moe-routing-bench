from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from moe_routing_bench.routers.base import PackCombine, RouteInfo, get_router
from moe_routing_bench.topk_impls import softk_indices_and_gates, top1_indices

from .experts import GroupFFNExperts

Tensor = torch.Tensor


class MoEFeedForward(nn.Module):
    """Mixture-of-Experts feed-forward block with pluggable routing backend."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int,
        ffn_mult: int = 4,
        router_name: str = "torch_soft",
        strategy: str = "softk",
        capacity_factor: float = 1.25,
        renorm_after_drop: bool = True,
        activation: str = "gelu",
        load_balance_alpha: float = 1e-2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_name = router_name
        self.strategy = strategy
        self.capacity_factor = capacity_factor
        self.renorm_after_drop = renorm_after_drop
        self.load_balance_alpha = load_balance_alpha

        self.router_linear = nn.Linear(hidden_dim, num_experts, bias=True)
        nn.init.normal_(self.router_linear.weight, mean=0.0, std=hidden_dim ** -0.5)
        nn.init.zeros_(self.router_linear.bias)

        self.experts = GroupFFNExperts(
            num_experts=num_experts,
            d_model=hidden_dim,
            ffn_mult=ffn_mult,
            activation=activation,
            bias=bias,
        )

    def _get_router(self) -> PackCombine:
        return get_router(self.router_name)

    def _capacity(self, tokens: int, k_eff: int) -> int:
        expected = tokens * k_eff / max(1, self.num_experts)
        cap = max(1, int(math.ceil(self.capacity_factor * expected)))
        return cap

    def _route(self, logits: Tensor) -> Tuple[torch.LongTensor, Optional[Tensor], int]:
        if self.strategy == "top1":
            indices = top1_indices(logits)
            gates = None
            k_eff = 1
        elif self.strategy == "topk_hard":
            indices = torch.topk(logits, self.top_k, dim=-1).indices
            gates = None
            k_eff = self.top_k
        elif self.strategy == "softk":
            indices, gates = softk_indices_and_gates(logits, self.top_k, temperature=1.0, normalize="softmax")
            gates = gates.to(logits.dtype)
            k_eff = self.top_k
        elif self.strategy == "expert_choice":
            indices, gates = self._route_expert_choice(logits)
            k_eff = self.top_k
        elif self.strategy == "hash":
            indices, gates = self._route_hash(logits)
            k_eff = self.top_k
        else:
            raise ValueError(f"Unknown routing strategy: {self.strategy}")
        return indices, gates, k_eff

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: tensor of shape [batch, seq, hidden].
        Returns:
            output tensor and a dict of logging stats (including aux loss).
        """
        orig_shape = x.shape
        tokens = x.shape[0] * x.shape[1]
        hidden = x.shape[-1]
        x_2d = x.reshape(tokens, hidden)

        logits = self.router_linear(x_2d)
        topk_idx, gates, k_eff = self._route(logits)
        capacity = self._capacity(tokens, k_eff)

        router = self._get_router()
        packed, route = router.pack(
            x_2d,
            topk_idx,
            gates,
            capacity=capacity,
            renorm_after_drop=self.renorm_after_drop,
        )

        expected_tokens = self.num_experts * capacity
        if packed.shape[0] != expected_tokens:
            if packed.shape[0] < expected_tokens:
                pad = packed.new_zeros((expected_tokens - packed.shape[0], hidden))
                packed = torch.cat([packed, pad], dim=0)
            else:
                packed = packed[:expected_tokens]
        if route.expert_counts.numel() != self.num_experts:
            counts = route.expert_counts
            if counts.numel() < self.num_experts:
                pad_counts = counts.new_zeros(self.num_experts)
                pad_counts[: counts.numel()] = counts
                route.expert_counts = pad_counts
            else:
                route.expert_counts = counts[: self.num_experts]
        route.experts = self.num_experts

        # reshape to [E, C, D]
        packed_view = packed.view(self.num_experts, capacity, hidden)
        expert_out = self.experts(packed_view, route.expert_counts)
        expert_flat = expert_out.view(self.num_experts * capacity, hidden)
        combined = router.combine(expert_flat, route, out_tokens=tokens)
        out = combined.view(*orig_shape)

        stats = self._compute_stats(route, capacity, tokens, k_eff, logits, gates)
        return out, stats

    def _route_expert_choice(self, logits: Tensor) -> Tuple[torch.LongTensor, Tensor]:
        """Simplified expert-choice routing (vectorized).

        Experts pick top tokens; tokens then keep up to top_k experts ranked by logits.
        Falls back to standard token->expert topk if not enough valid experts.
        """

        tokens, num_experts = logits.shape
        device = logits.device
        dtype = logits.dtype
        k = self.top_k

        if tokens == 0 or num_experts == 0:
            empty_idx = torch.empty(tokens, k, dtype=torch.long, device=device)
            empty_gates = torch.empty(tokens, k, dtype=dtype, device=device)
            return empty_idx, empty_gates

        expected = tokens * k / max(1, num_experts)
        tokens_per_expert = max(1, int(math.ceil(self.capacity_factor * expected)))
        tokens_per_expert = min(tokens_per_expert, tokens)
        # Clamp to keep the per-expert topk tractable on small models.
        tokens_per_expert = min(tokens_per_expert, 64)

        # Expert-first selection mask: top tokens per expert (shape [E, T] -> mask [T, E])
        scores_T = logits.transpose(0, 1)
        _, top_tok = torch.topk(scores_T, k=tokens_per_expert, dim=1)
        mask = torch.zeros_like(scores_T, dtype=torch.bool)
        mask.scatter_(1, top_tok, True)
        mask_token_expert = mask.transpose(0, 1)  # [T, E]

        masked_logits = logits.masked_fill(~mask_token_expert, float("-inf"))
        vals, idx = torch.topk(masked_logits, k=min(k, num_experts), dim=-1)

        # Fallback for rows with no valid experts selected
        mask_any = mask_token_expert.any(dim=1, keepdim=True)  # [T,1] bool
        if not mask_any.all():
            std_vals, std_idx = torch.topk(logits, k=min(k, num_experts), dim=-1)
            vals = torch.where(mask_any, vals, std_vals)
            idx = torch.where(mask_any, idx, std_idx)

        gates = torch.softmax(vals, dim=-1).to(dtype)
        return idx, gates

    def _route_hash(self, logits: Tensor) -> Tuple[torch.LongTensor, Tensor]:
        """Content-agnostic hash routing with uniform gates."""

        tokens, num_experts = logits.shape
        device = logits.device
        dtype = logits.dtype
        k = self.top_k

        if tokens == 0 or num_experts == 0:
            empty_idx = torch.empty(tokens, k, dtype=torch.long, device=device)
            empty_gates = torch.empty(tokens, k, dtype=dtype, device=device)
            return empty_idx, empty_gates

        token_ids = torch.arange(tokens, device=device, dtype=torch.long)
        A = 1315423911
        B = 2654435761
        C = 97

        base = (token_ids * A + B) % num_experts
        indices = torch.empty(tokens, k, dtype=torch.long, device=device)
        indices[:, 0] = base
        for j in range(1, k):
            indices[:, j] = (base + j * C) % num_experts

        gates = torch.full((tokens, k), 1.0 / float(k), dtype=dtype, device=device)
        return indices, gates

    def _compute_stats(
        self,
        route: RouteInfo,
        capacity: int,
        tokens: int,
        k_eff: int,
        logits: Tensor,
        gates: Optional[Tensor],
    ) -> Dict[str, torch.Tensor]:
        total_assign = max(1, tokens * k_eff)
        kept = route.kept_mask.sum().item()
        avg_drop = 1.0 - kept / total_assign
        token_drop = float((~route.kept_mask).all(dim=1).float().mean().item()) if tokens > 0 else 0.0
        counts = route.expert_counts.float()
        load_mean = counts.mean()
        load_std = counts.std(unbiased=False)
        load_cv = load_std / (load_mean + 1e-9)
        used_capacity = counts.max() / max(1, capacity)

        loss_bal = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if counts.sum() > 0:
            norm_counts = counts / counts.sum()
            target = 1.0 / self.num_experts
            loss_bal = ((norm_counts - target) ** 2).sum()

        aux_loss = self.load_balance_alpha * loss_bal

        gate_entropy = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        if gates is not None:
            gate_entropy = -(gates * (gates.clamp_min(1e-9).log())).sum(dim=-1).mean()

        return {
            "drop_rate": torch.tensor(avg_drop, device=logits.device),
            "token_drop_rate": torch.tensor(token_drop, device=logits.device),
            "load_cv": load_cv.to(logits.device),
            "used_capacity": used_capacity.to(logits.device),
            "aux_loss": aux_loss,
            "gate_entropy": gate_entropy,
            "expert_counts": counts,
        }
