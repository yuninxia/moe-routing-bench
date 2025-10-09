from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupFFNExperts(nn.Module):
    """Grouped FFN experts with per-expert parameters.

    Args:
        num_experts: number of experts.
        d_model: input/output hidden dimension.
        ffn_mult: expansion factor for FFN hidden size (default 4x).
        activation: name of activation ("gelu" or "silu").
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        ffn_mult: int = 4,
        activation: str = "gelu",
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.d_model = d_model
        self.hidden = ffn_mult * d_model

        self.w1 = nn.Parameter(torch.empty(num_experts, d_model, self.hidden))
        self.w2 = nn.Parameter(torch.empty(num_experts, self.hidden, d_model))
        if bias:
            self.b1 = nn.Parameter(torch.zeros(num_experts, self.hidden))
            self.b2 = nn.Parameter(torch.zeros(num_experts, d_model))
        else:
            self.register_parameter("b1", None)
            self.register_parameter("b2", None)

        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)

        if activation == "gelu":
            self._activation = F.gelu
        elif activation == "silu":
            self._activation = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    @torch.no_grad()
    def num_parameters(self) -> int:
        total = self.w1.numel() + self.w2.numel()
        if self.b1 is not None:
            total += self.b1.numel() + self.b2.numel()
        return total

    def forward(self, packed_inputs: torch.Tensor, valid_load: torch.Tensor) -> torch.Tensor:
        """Applies grouped FFN to packed expert inputs.

        Args:
            packed_inputs: tensor of shape [E, capacity, D].
            valid_load: tensor [E] or iterable indicating valid rows per expert.
        Returns:
            Tensor with same shape as packed_inputs containing expert outputs.
        """

        if packed_inputs.dim() != 3:
            raise ValueError("packed_inputs must be [experts, capacity, hidden]")
        experts, capacity, hidden = packed_inputs.shape
        if experts != self.num_experts or hidden != self.d_model:
            raise ValueError(
                f"Mismatch: inputs shape {packed_inputs.shape} vs model (E={self.num_experts}, D={self.d_model})"
            )

        device = packed_inputs.device
        valid = torch.as_tensor(valid_load, device=device, dtype=torch.long)
        if valid.shape != (experts,):
            raise ValueError("valid_load must have shape [num_experts]")

        outputs = packed_inputs.new_zeros(experts, capacity, hidden)
        for expert_id in range(experts):
            load = int(valid[expert_id].item())
            if load <= 0:
                continue
            x_e = packed_inputs[expert_id, :load]
            h = x_e @ self.w1[expert_id]
            if self.b1 is not None:
                h = h + self.b1[expert_id]
            h = self._activation(h)
            y_e = h @ self.w2[expert_id]
            if self.b2 is not None:
                y_e = y_e + self.b2[expert_id]
            outputs[expert_id, :load] = y_e
        return outputs

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, d_model={self.d_model}, "
            f"hidden={self.hidden}, has_bias={self.b1 is not None}"
        )
