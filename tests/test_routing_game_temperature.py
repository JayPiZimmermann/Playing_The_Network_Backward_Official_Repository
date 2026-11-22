"""Hard-routing and low-temperature limits used by Routing Game hooks."""

import torch

from bwgame.xai.methods.routing_game import _power_routing_shares
from bwgame.xai.methods.routing_game import _RoutingGameHookManager


def test_zero_temperature_argmax_ties_and_outside_option():
    mass = torch.tensor([[1.5, 1.5, 0.9], [1.0, 1.0, 0.5], [0.4, 0.8, 0.2]])
    coeff = torch.tensor([[1.0, 3.0, 5.0], [1.0, 2.0, 4.0], [1.0, 1.0, 1.0]])
    got = _power_routing_shares(mass, coeff, tau=0.0, eps=0.1)
    expected = torch.tensor([[0.25, 0.75, 0.0], [1 / 3.1, 2 / 3.1, 0.0], [0.0, 0.0, 0.0]])
    torch.testing.assert_close(got, expected)


def test_zero_temperature_zero_epsilon_uses_positive_maximum():
    got = _power_routing_shares(
        torch.tensor([[0.2, 0.2, 0.1], [0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 3.0, 9.0], [1.0, 1.0, 1.0]]),
        tau=0.0, eps=0.0,
    )
    torch.testing.assert_close(got, torch.tensor([[0.25, 0.75, 0.0], [0.0, 0.0, 0.0]]))


def test_small_positive_temperature_is_log_stable_and_matches_power_limit():
    mass = torch.tensor([[0.8, 0.9, 1.1], [0.4, 0.6, 0.7]], dtype=torch.float64)
    coeff = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=torch.float64)
    got = _power_routing_shares(mass, coeff, tau=0.05, eps=0.1)
    powered = coeff * mass.pow(20)
    expected = powered / (powered.sum(dim=-1, keepdim=True) + 0.1)
    torch.testing.assert_close(got, expected)
    assert torch.isfinite(got).all()


def test_merged_zero_temperature_uses_attention_weighted_local_tie_counts():
    class ValueBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v_proj = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.v_proj.weight.fill_(1.0)

    model = ValueBlock().double()
    manager = _RoutingGameHookManager(model, routing_temperature=0.0, lrp_eps=0.1)
    x = torch.tensor([[[2.0, 2.0], [2.0, 0.0]]], dtype=torch.float64)
    manager._activations["v_proj"] = x
    attention = torch.tensor([[[0.2, 0.8], [0.2, 0.8]]], dtype=torch.float64)
    values = model.v_proj(x).reshape(1, 1, 2, 1).detach()
    relevance = torch.tensor([[[1.0], [0.0]]], dtype=torch.float64)

    manager._merged_value_backward("av", attention, values, relevance, eps=0.1)
    cached_pos, cached_neg = manager._merged_value_relevance["v_proj"]
    routed = manager._linear_power_route(
        x, model.v_proj.weight.detach(), None, cached_pos, tau=0.0, eps=0.0
    )
    expected = torch.tensor([[[1 / 6, 1 / 6], [2 / 3, 0.0]]], dtype=torch.float64)
    torch.testing.assert_close(routed, expected)
    torch.testing.assert_close(cached_neg, torch.zeros_like(cached_neg))


def test_merged_small_positive_temperature_matches_composite_edge_formula():
    class ValueBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v_proj = torch.nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                self.v_proj.weight.fill_(1.0)

    model = ValueBlock().double()
    tau, eps = 0.05, 0.1
    manager = _RoutingGameHookManager(model, routing_temperature=tau, lrp_eps=eps)
    x = torch.tensor([[[2.0, 2.0], [2.0, 0.0]]], dtype=torch.float64)
    manager._activations["v_proj"] = x
    attention = torch.tensor([[[0.2, 0.8], [0.2, 0.8]]], dtype=torch.float64)
    values = model.v_proj(x).reshape(1, 1, 2, 1).detach()
    relevance = torch.tensor([[[1.0], [0.0]]], dtype=torch.float64)

    manager._merged_value_backward("av", attention, values, relevance, eps=eps)
    cached_pos, _ = manager._merged_value_relevance["v_proj"]
    routed = manager._linear_power_route(
        x, model.v_proj.weight.detach(), None, cached_pos, tau=tau, eps=0.0
    )

    edge_mass = x[0] * model.v_proj.weight.detach()
    edge_mass = edge_mass.pow(1.0 / tau)
    denominator = (attention[0, 0, :, None] * edge_mass).sum() + eps
    expected = torch.zeros_like(x)
    expected[0] = attention[0, 0, :, None] * edge_mass / denominator
    torch.testing.assert_close(routed, expected, rtol=1e-12, atol=1e-12)


def test_tau_point_one_keeps_legacy_linear_arithmetic_bitwise():
    model = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=True)).double().eval()
    layer = model[0]
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([[0.3, -0.4, 0.2], [-0.1, 0.25, 0.15]], dtype=torch.float64))
        layer.bias.copy_(torch.tensor([0.2, -0.15], dtype=torch.float64))
    x = torch.tensor([[0.7, -0.6, 0.5]], dtype=torch.float64)
    tau, eps, alpha, beta = 0.1, 0.1, 2.0, 1.0
    manager = _RoutingGameHookManager(
        model, lrp_alpha=alpha, lrp_beta=beta, lrp_eps=eps,
        routing_temperature=tau,
    )
    manager.install()
    try:
        x_in = x.clone().requires_grad_(True)
        out = model(x_in)
        seed = torch.tensor([[0.4, -0.8]], dtype=torch.float64)
        out.backward(seed)

        power = 1 / tau
        xp, xn = torch.relu(x), torch.relu(-x)
        wp, wn = torch.relu(layer.weight.detach()), torch.relu(-layer.weight.detach())
        xp, xn, wp, wn = xp.pow(power), xn.pow(power), wp.pow(power), wn.pow(power)
        b = layer.bias.detach()
        z_pos = xp @ wp.t() + xn @ wn.t() + b.clamp_min(0).pow(power)
        z_neg = xp @ wn.t() + xn @ wp.t() + (-b).clamp_min(0).pow(power)
        s_pos, s_neg = seed / (z_pos + eps), seed / (z_neg + eps)
        r_alpha = xp * (s_pos @ wp) + xn * (s_pos @ wn)
        r_beta = xp * (s_neg @ wn) + xn * (s_neg @ wp)
        expected = alpha * r_alpha - beta * r_beta
        assert torch.equal(x_in.grad, expected)
    finally:
        manager.remove()


def test_zero_temperature_conv_route_supports_groups():
    conv = torch.nn.Conv2d(2, 2, kernel_size=1, groups=2, bias=True).double()
    with torch.no_grad():
        conv.weight.copy_(torch.tensor([[[[2.0]]], [[[0.5]]]], dtype=torch.float64))
        conv.bias.copy_(torch.tensor([0.0, 0.0], dtype=torch.float64))
    x = torch.tensor([[[[2.0, 0.5]], [[0.5, 2.0]]]], dtype=torch.float64)
    seed = torch.ones(1, 2, 1, 2, dtype=torch.float64)
    got = _RoutingGameHookManager._conv_power_route(
        x, conv.weight.detach(), conv.bias.detach(), seed, conv, tau=0.0, eps=0.1
    )
    # Each grouped output has one input edge.  Mass 1 ties the edge with eps;
    # a mass below 1 is absorbed by the outside option.
    expected = torch.tensor([[[[1.0, 10 / 11]], [[0.0, 10 / 11]]]], dtype=torch.float64)
    torch.testing.assert_close(got, expected)


def test_merged_zero_temperature_handles_multiple_heads():
    class ValueBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.v_proj = torch.nn.Linear(2, 2, bias=False)
            with torch.no_grad():
                self.v_proj.weight.copy_(torch.eye(2))

    model = ValueBlock().double()
    manager = _RoutingGameHookManager(model, routing_temperature=0.0, lrp_eps=0.1)
    x = torch.tensor([[[2.0, 3.0], [2.0, 3.0]]], dtype=torch.float64)
    manager._activations["v_proj"] = x
    values = model.v_proj(x).reshape(1, 2, 2, 1).transpose(1, 2).detach()
    attention = torch.full((1, 2, 2, 2), 0.5, dtype=torch.float64)
    relevance = torch.tensor([[[[1.0], [0.0]], [[0.0], [1.0]]]], dtype=torch.float64)
    manager._merged_value_backward("av", attention, values, relevance, eps=0.1)
    pos, neg = manager._merged_value_relevance["v_proj"]
    routed = manager._linear_power_route(x, model.v_proj.weight.detach(), None, pos, 0.0, 0.0)
    assert routed.shape == x.shape
    assert torch.isfinite(routed).all()
    assert torch.count_nonzero(neg) == 0
