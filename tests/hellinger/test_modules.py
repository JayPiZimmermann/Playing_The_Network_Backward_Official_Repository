"""Per-module-type Hellinger tests.

Builds tiny networks containing each individual supported module type
and verifies that the backward sweep through it satisfies the
mass-conservation and Hellinger-identity invariants for both games.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from bwgame.xai.methods.hellinger import compute_hellinger_distance

from .conftest import assert_invariants


def _check(model, x_A, x_B=None, game="adf", tag=""):
    model.eval()
    r = compute_hellinger_distance(model, x_A, input_B=x_B, game=game)
    assert_invariants(r, name=f"{tag}/{game}")
    total = r["h_squared"] + r["bhattacharyya"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-3), (
        f"[{tag}/{game}] H^2 + BC != 1: {total}"
    )
    return r


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["adf", "rg"])
def test_module_linear(game):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 5))
    x = torch.rand(2, 8) * 2 + 0.05
    y = torch.rand(2, 8) * 2 + 0.05
    _check(model, x, y, game=game, tag="linear")


# ---------------------------------------------------------------------------
# Conv2d (with same padding)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["adf", "rg"])
def test_module_conv2d_padding(game):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(16, 5),
    )
    x = torch.rand(2, 3, 8, 8)
    y = torch.rand(2, 3, 8, 8)
    _check(model, x, y, game=game, tag="conv2d_pad")


@pytest.mark.parametrize("game", ["adf", "rg"])
def test_module_conv2d_no_padding(game):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=0), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=0), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(16, 5),
    )
    x = torch.rand(2, 3, 10, 10)
    y = torch.rand(2, 3, 10, 10)
    _check(model, x, y, game=game, tag="conv2d_nopad")


@pytest.mark.parametrize("game", ["adf", "rg"])
def test_module_conv2d_stride(game):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1, stride=2), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(16, 5),
    )
    x = torch.rand(2, 3, 8, 8)
    y = torch.rand(2, 3, 8, 8)
    _check(model, x, y, game=game, tag="conv2d_stride")


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["adf", "rg"])
def test_module_avgpool2d(game):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(16, 5),
    )
    x = torch.rand(2, 3, 8, 8)
    y = torch.rand(2, 3, 8, 8)
    _check(model, x, y, game=game, tag="avgpool")


@pytest.mark.parametrize("game", ["adf", "rg"])
def test_module_maxpool2d(game):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(16, 5),
    )
    x = torch.rand(2, 3, 8, 8)
    y = torch.rand(2, 3, 8, 8)
    _check(model, x, y, game=game, tag="maxpool")


@pytest.mark.parametrize("game", ["adf", "rg"])
def test_module_adaptive_avgpool2d_nontrivial(game):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d((2, 2)),  # nontrivial output spatial
        nn.Flatten(),
        nn.Linear(8 * 2 * 2, 5),
    )
    x = torch.rand(2, 3, 8, 8)
    y = torch.rand(2, 3, 8, 8)
    _check(model, x, y, game=game, tag="adaptive_avgpool")


# ---------------------------------------------------------------------------
# Identity-like layers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["adf", "rg"])
@pytest.mark.parametrize(
    "act_factory",
    [nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid,
     lambda: nn.LeakyReLU(0.1)],
)
def test_module_activation(game, act_factory):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 16), act_factory(), nn.Linear(16, 5))
    x = torch.rand(2, 8) * 2 + 0.05
    y = torch.rand(2, 8) * 2 + 0.05
    _check(model, x, y, game=game, tag=f"act/{act_factory.__name__ if hasattr(act_factory, '__name__') else 'leaky'}")


@pytest.mark.parametrize("game", ["adf", "rg"])
def test_module_layernorm(game):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16), nn.LayerNorm(16), nn.ReLU(),
        nn.Linear(16, 5),
    )
    model.eval()
    x = torch.rand(2, 8) * 2 + 0.05
    y = torch.rand(2, 8) * 2 + 0.05
    _check(model, x, y, game=game, tag="layernorm")


@pytest.mark.parametrize("game", ["adf", "rg"])
def test_module_batchnorm2d(game):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(8, 5),
    )
    model.eval()  # BN uses running stats
    x = torch.rand(2, 3, 8, 8)
    y = torch.rand(2, 3, 8, 8)
    _check(model, x, y, game=game, tag="batchnorm2d")


@pytest.mark.parametrize("game", ["adf", "rg"])
def test_module_dropout(game):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(8, 16), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(16, 5),
    )
    model.eval()  # disables dropout, so it's identity
    x = torch.rand(2, 8) * 2 + 0.05
    y = torch.rand(2, 8) * 2 + 0.05
    _check(model, x, y, game=game, tag="dropout")


# ---------------------------------------------------------------------------
# Flatten
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["adf", "rg"])
def test_module_flatten(game):
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 5),
    )
    x = torch.rand(2, 3, 8, 8)
    y = torch.rand(2, 3, 8, 8)
    _check(model, x, y, game=game, tag="flatten")
