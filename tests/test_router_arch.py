import torch

from moe_routing_bench.modules import MoEFeedForward


def _run_once(router_arch: str) -> dict:
    torch.manual_seed(0)
    batch, seq, hidden, experts, top_k = 2, 3, 16, 8, 2
    x = torch.randn(batch, seq, hidden)
    moe = MoEFeedForward(
        hidden_dim=hidden,
        num_experts=experts,
        top_k=top_k,
        strategy="softk",
        capacity_factor=10.0,  # avoid drops so stats are stable
        router_arch=router_arch,
    )
    out, stats = moe(x)
    return {
        "out_shape": out.shape,
        "router_params": float(stats["router_params"]),
        "mean_topk_prob": float(stats["mean_topk_prob"]),
        "drop_rate": float(stats["drop_rate"]),
    }


def test_router_arch_outputs_and_stats():
    # Linear, MLP, and Hadamard variants should run and produce stats.
    linear = _run_once("linear")
    mlp = _run_once("mlp")
    hadamard = _run_once("mlp_hadamard")

    # Shapes preserved
    assert linear["out_shape"] == (2, 3, 16)
    assert mlp["out_shape"] == (2, 3, 16)
    assert hadamard["out_shape"] == (2, 3, 16)

    # Router parameter counts should be positive and differ across architectures.
    assert linear["router_params"] > 0
    assert mlp["router_params"] > linear["router_params"]
    assert hadamard["router_params"] > linear["router_params"]

    # mean_topk_prob should be in (0,1] for soft routing.
    for stats in (linear, mlp, hadamard):
        assert 0.0 < stats["mean_topk_prob"] <= 1.0
        # With large capacity factor we shouldn't drop tokens.
        assert stats["drop_rate"] == 0.0
