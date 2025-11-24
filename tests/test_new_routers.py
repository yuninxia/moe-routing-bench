import torch

from moe_routing_bench.modules import MoEFeedForward
from moe_routing_bench.routers.base import get_router


def _reconstruct(x: torch.Tensor, idx: torch.LongTensor, gates: torch.Tensor) -> torch.Tensor:
    return (x.unsqueeze(1) * gates.unsqueeze(-1)).sum(dim=1)


def test_hash_routing_identity_no_drop():
    torch.manual_seed(0)
    tokens, experts, hidden, top_k = 32, 8, 16, 2
    x = torch.randn(tokens, hidden)
    logits = torch.randn(tokens, experts)

    moe = MoEFeedForward(
        hidden_dim=hidden,
        num_experts=experts,
        top_k=top_k,
        strategy="hash",
        capacity_factor=10.0,  # not used in _route_hash, but keeps _capacity large if needed
    )
    topk_idx, gates, _ = moe._route(logits)  # type: ignore[attr-defined]

    # Large capacity to avoid drops
    capacity = tokens * top_k
    router = get_router("torch_soft")
    packed, route = router.pack(x, topk_idx, gates, capacity=capacity, renorm_after_drop=True)
    out = router.combine(packed, route, out_tokens=tokens)

    recon = _reconstruct(x, topk_idx, gates)
    torch.testing.assert_close(out, recon, rtol=1e-5, atol=1e-6)


def test_expert_choice_routing_identity_no_drop():
    torch.manual_seed(1)
    tokens, experts, hidden, top_k = 24, 6, 12, 2
    x = torch.randn(tokens, hidden)
    logits = torch.randn(tokens, experts)

    moe = MoEFeedForward(
        hidden_dim=hidden,
        num_experts=experts,
        top_k=top_k,
        strategy="expert_choice",
        capacity_factor=2.0,
    )
    topk_idx, gates, _ = moe._route(logits)  # type: ignore[attr-defined]

    capacity = tokens * top_k
    router = get_router("torch_soft")
    packed, route = router.pack(x, topk_idx, gates, capacity=capacity, renorm_after_drop=True)
    out = router.combine(packed, route, out_tokens=tokens)

    recon = _reconstruct(x, topk_idx, gates)
    torch.testing.assert_close(out, recon, rtol=1e-5, atol=1e-6)
