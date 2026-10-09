"""Shared fixtures and helpers for the Hellinger path-distance tests.

Provides:
- Reproducible model factories (MLPs, conv nets, all activation/pool variants)
- Reference inputs (positive-valued, ReLU-friendly)
- Helper to assert mass conservation invariants on the result dict
"""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _seed_everything():
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def _seeded_apply(model: nn.Module, seed: int) -> nn.Module:
    """Re-initialize a model with a specific seed (for randomization tests)."""
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        if p.dim() >= 2:
            nn.init.kaiming_normal_(p, generator=g)
        else:
            nn.init.normal_(p, std=0.01, generator=g)
    return model


def make_simple_mlp(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(20, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 5),
    )


def make_mlp_with_dropout(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    m = nn.Sequential(
        nn.Linear(20, 32),
        nn.ReLU(),
        nn.Dropout(p=0.0),  # disabled in eval mode
        nn.Linear(32, 16),
        nn.GELU(),
        nn.Linear(16, 5),
    )
    m.eval()
    return m


def make_simple_conv(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 5),
    )


def make_conv_with_avgpool(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2),
        nn.Flatten(),
        nn.Linear(16 * 2 * 2, 5),
    )


def make_conv_with_maxpool(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 5),
    )


def make_mlp_with_layernorm(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(20, 32),
        nn.LayerNorm(32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.LayerNorm(16),
        nn.ReLU(),
        nn.Linear(16, 5),
    )


def make_randomized_copy(model: nn.Module, seed: int) -> nn.Module:
    """Deep-copy and re-initialize all parameters with a different seed."""
    cp = copy.deepcopy(model)
    return _seeded_apply(cp, seed)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def make_mlp_input(seed: int = 0, batch: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 20, generator=g) * 2.0 + 0.05


def make_image_input(seed: int = 0, batch: int = 1, size: int = 8) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, size, size, generator=g)


# ---------------------------------------------------------------------------
# Invariant assertions
# ---------------------------------------------------------------------------

def assert_invariants(result: dict, *, name: str = ""):
    """Assert generic Hellinger result invariants.

    Checks:
        - hellinger_distance and h_squared are non-negative
        - h_squared is in [0, 1] (boundedness)
        - per_pixel_h2 is non-negative everywhere
        - per_pixel_h2.sum + cemetery_h2 == h_squared
        - bhattacharyya is in [0, 1]
        - mu_A and mu_B at the input layer are non-negative
        - beta at the input layer is non-negative
        - cemetery coordinates are non-negative
    """
    h2 = result["h_squared"]
    h = result["hellinger_distance"]
    bc = result["bhattacharyya"]
    pp = result["per_pixel_h2"]
    cem = result["cemetery_h2"]
    mu_A = result["mu_A_input"]
    mu_B = result["mu_B_input"]
    beta = result["beta_input"]

    tag = f"[{name}] " if name else ""

    assert (h2 >= -1e-5).all(), f"{tag}h^2 must be >=0, got {h2}"
    assert (h >= -1e-5).all(),  f"{tag}H must be >=0, got {h}"
    assert (h2 <= 1.0 + 1e-3).all(), f"{tag}h^2 must be <=1, got {h2}"
    assert (bc >= -1e-3).all(), f"{tag}BC must be >=0, got {bc}"
    assert (bc <= 1.0 + 1e-3).all(), f"{tag}BC must be <=1, got {bc}"
    assert (pp >= -1e-5).all(), f"{tag}per_pixel_h2 must be >=0, got min {pp.min()}"
    assert (mu_A >= -1e-5).all(), f"{tag}mu_A must be >=0"
    assert (mu_B >= -1e-5).all(), f"{tag}mu_B must be >=0"
    assert (beta >= -1e-5).all(), f"{tag}beta must be >=0"
    assert (cem >= -1e-5).all(), f"{tag}cemetery_h2 must be >=0, got {cem}"

    # DP consistency: the ordinary-input map plus the single cemetery
    # coordinate must recover the full squared Hellinger distance.
    total = pp.flatten(start_dim=1).sum(dim=1) + cem
    diff = (total - h2).abs().max().item()
    assert diff < 1e-3, (
        f"{tag}per_pixel.sum + cemetery != H^2: "
        f"sum={total}, h2={h2}, max diff={diff:.3e}"
    )
