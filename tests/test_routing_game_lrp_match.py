"""Tests verifying RoutingGame ↔ zennit AlphaBeta LRP equivalence.

At τ=1 and λ=0 the RoutingGame occupation measure is αβ-LRP (Theorem 2):
its input relevance equals that of zennit's AlphaBeta rule on linear models
and on ReLU networks with biases and max-pooling.
"""

import pytest
import torch
import torch.nn as nn

# ---- helpers ---------------------------------------------------------------

def _zennit_alphabeta_attribution(model, x, target_class, alpha=2.0, beta=1.0, eps=1e-6):
    """Run zennit AlphaBeta attribution and return input relevance."""
    from zennit.composites import SpecialFirstLayerMapComposite
    from zennit.rules import AlphaBeta

    layer_map = [
        (nn.Conv2d, AlphaBeta(alpha=alpha, beta=beta, stabilizer=eps)),
        (nn.Linear, AlphaBeta(alpha=alpha, beta=beta, stabilizer=eps)),
    ]
    composite = SpecialFirstLayerMapComposite(
        layer_map=layer_map,
        first_map=layer_map,
        canonizers=[],
    )

    x_in = x.detach().clone().requires_grad_(True)
    with composite.context(model) as m:
        out = m(x_in)
        one_hot = torch.zeros_like(out)
        one_hot[0, target_class] = 1.0
        out.backward(gradient=one_hot)
    return x_in.grad.detach()


def _routing_game_attribution(model, x, target_class, alpha=2.0, beta=1.0,
                               eps=1e-6, tau=1.0):
    """Run RoutingGame attribution and return input relevance."""
    from bwgame.xai.methods.routing_game import _RoutingGameHookManager

    mgr = _RoutingGameHookManager(
        model=model,
        lrp_alpha=alpha,
        lrp_beta=beta,
        lrp_eps=eps,
        routing_temperature=tau,
        adf_lambda=0.0,
    )
    mgr.install()
    try:
        x_in = x.detach().clone().requires_grad_(True)
        out = model(x_in)
        one_hot = torch.zeros_like(out)
        one_hot[0, target_class] = 1.0
        out.backward(gradient=one_hot)
        return x_in.grad.detach()
    finally:
        mgr.remove()


# ---- models ----------------------------------------------------------------

class LinearOnly(nn.Module):
    """Three-layer MLP with NO activations."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc3(self.fc2(self.fc1(x)))


class ConvOnly(nn.Module):
    """Two conv layers + linear head, NO activations."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class CNNWithReLUMaxPool(nn.Module):
    """Two conv + ReLU + max-pool blocks and a linear head."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(8 * 2 * 2, 4)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.fc(x.flatten(1))


class MLPWithReLU(nn.Module):
    """Three-layer MLP with ReLU activations."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        return self.fc3(self.relu2(self.fc2(self.relu1(self.fc1(x)))))


# ---- tests -----------------------------------------------------------------

@pytest.fixture
def seed():
    torch.manual_seed(42)


def test_linear_only_matches_zennit_alphabeta(seed):
    """Pure linear MLP: RoutingGame must exactly match zennit AlphaBeta."""
    model = LinearOnly().eval()
    x = torch.randn(1, 8)

    rg = _routing_game_attribution(model, x, target_class=0,
                                    alpha=2.0, beta=1.0, eps=1e-6)
    zn = _zennit_alphabeta_attribution(model, x, target_class=0,
                                        alpha=2.0, beta=1.0, eps=1e-6)

    assert torch.allclose(rg, zn, rtol=1e-4, atol=1e-6), (
        f"Max diff: {(rg - zn).abs().max().item():.2e}"
    )


def test_conv_only_matches_zennit_alphabeta(seed):
    """Pure conv model: RoutingGame must exactly match zennit AlphaBeta."""
    model = ConvOnly().eval()
    x = torch.randn(1, 3, 8, 8)

    rg = _routing_game_attribution(model, x, target_class=0,
                                    alpha=2.0, beta=1.0, eps=1e-6)
    zn = _zennit_alphabeta_attribution(model, x, target_class=0,
                                        alpha=2.0, beta=1.0, eps=1e-6)

    assert torch.allclose(rg, zn, rtol=1e-4, atol=1e-6), (
        f"Max diff: {(rg - zn).abs().max().item():.2e}"
    )


@pytest.mark.parametrize("eps", [1e-6, 0.5])
def test_relu_mlp_matches_zennit_alphabeta(seed, eps):
    """MLP with ReLU and biases: RoutingGame must match zennit AlphaBeta."""
    model = MLPWithReLU().eval()
    x = torch.randn(1, 8)

    rg = _routing_game_attribution(model, x, target_class=0,
                                    alpha=2.0, beta=1.0, eps=eps)
    zn = _zennit_alphabeta_attribution(model, x, target_class=0,
                                        alpha=2.0, beta=1.0, eps=eps)

    assert torch.allclose(rg, zn, rtol=1e-4, atol=1e-6), (
        f"Max diff: {(rg - zn).abs().max().item():.2e}"
    )


@pytest.mark.parametrize("eps", [1e-6, 0.5])
def test_relu_maxpool_cnn_matches_zennit_alphabeta(seed, eps):
    """Conv + ReLU + max-pool network: RoutingGame must match zennit AlphaBeta."""
    model = CNNWithReLUMaxPool().eval()
    x = torch.randn(1, 3, 8, 8)

    rg = _routing_game_attribution(model, x, target_class=0,
                                    alpha=2.0, beta=1.0, eps=eps)
    zn = _zennit_alphabeta_attribution(model, x, target_class=0,
                                        alpha=2.0, beta=1.0, eps=eps)

    assert torch.allclose(rg, zn, rtol=1e-4, atol=1e-6), (
        f"Max diff: {(rg - zn).abs().max().item():.2e}"
    )


def test_different_alpha_beta_differ(seed):
    """Different α/β values must produce different attributions."""
    model = LinearOnly().eval()
    x = torch.randn(1, 8)

    rg_21 = _routing_game_attribution(model, x, target_class=0,
                                       alpha=2.0, beta=1.0)
    rg_10 = _routing_game_attribution(model, x, target_class=0,
                                       alpha=1.0, beta=0.0)

    assert not torch.allclose(rg_21, rg_10, rtol=1e-3), (
        "α=2,β=1 and α=1,β=0 should produce different results"
    )


def test_tau_zero_no_crash(seed):
    """τ=0 should not crash (gets clamped to small positive)."""
    model = LinearOnly().eval()
    x = torch.randn(1, 8)

    rg = _routing_game_attribution(model, x, target_class=0, tau=0.0)
    assert rg is not None
    assert not torch.isnan(rg).any(), "NaN in attribution with τ=0"
