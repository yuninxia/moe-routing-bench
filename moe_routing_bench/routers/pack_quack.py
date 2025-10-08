from __future__ import annotations

import importlib
from typing import Optional, Tuple

import torch

from .base import PackCombine, RouteInfo, register_router
from .pack_soft_torch import TorchSoftPackCombine

Tensor = torch.Tensor


class QuackPackCombine(PackCombine):
    name = "quack"

    def __init__(self) -> None:
        self._fallback = TorchSoftPackCombine()
        try:
            self._quack = importlib.import_module("quack")
        except Exception:
            self._quack = None
        self._pack_kernel = getattr(self._quack, "pack_kernel", None) if self._quack else None
        self._combine_kernel = getattr(self._quack, "combine_kernel", None) if self._quack else None

    def pack(
        self,
        x: Tensor,
        topk_idx: torch.LongTensor,
        gates: Optional[Tensor],
        capacity: int,
        renorm_after_drop: bool,
    ) -> Tuple[Tensor, RouteInfo]:
        if self._pack_kernel is None:
            return self._fallback.pack(x, topk_idx, gates, capacity, renorm_after_drop)
        raise NotImplementedError("Hook your QuACK/CuTe pack kernel here to match TorchSoft semantics")

    def combine(self, y: Tensor, route: RouteInfo, out_tokens: int) -> Tensor:
        if self._combine_kernel is None:
            return self._fallback.combine(y, route, out_tokens)
        raise NotImplementedError("Hook your QuACK/CuTe combine kernel here")


register_router(QuackPackCombine.name, QuackPackCombine)
