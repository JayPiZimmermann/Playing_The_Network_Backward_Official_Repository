"""Generic Hellinger invariants for both ADF and Active Routing games.

Tests the mathematical guarantees that must hold for ANY model, ANY input,
and ANY game choice:

* Mass conservation: sum of mu_A and mu_B at the input layer (plus cemetery)
  is exactly 1 for both models.
* Hellinger identity: H^2 + BC = 1 (per batch element).
* Boundedness: H^2 in [0, 1], BC in [0, 1], H = sqrt(H^2).
* Non-negativity: per_pixel_h2 >= 0 everywhere; mu and beta sub-masses >= 0.
* Decomposition consistency: per_pixel.sum + cemetery == h_squared.
* Identity case: H^2 == 0 when both models and inputs are the same.
* Symmetry: H(A, B) == H(B, A).
"""

from __future__ import annotations

import pytest
import torch

from bwgame.xai.methods.hellinger import compute_hellinger_distance

from .conftest import (
    assert_invariants,
    make_image_input,
    make_mlp_input,
    make_randomized_copy,
    make_simple_mlp,
    make_simple_conv,
    make_conv_with_avgpool,
    make_conv_with_maxpool,
    make_mlp_with_layernorm,
    make_mlp_with_dropout,
)


# Pairs (factory, input_factory) used to parametrize the invariant suite.
MODEL_FACTORIES = [
    ("mlp", make_simple_mlp, make_mlp_input),
    ("mlp_dropout", make_mlp_with_dropout, make_mlp_input),
    ("mlp_layernorm", make_mlp_with_layernorm, make_mlp_input),
    ("conv", make_simple_conv, make_image_input),
    ("conv_avgpool", make_conv_with_avgpool, make_image_input),
    ("conv_maxpool", make_conv_with_maxpool, make_image_input),
]


# ---------------------------------------------------------------------------
# Identity case (same model + same input must give H^2 == 0)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,model_factory,input_factory", MODEL_FACTORIES)
@pytest.mark.parametrize("game", ["adf", "rg"])
def test_identity_zero_distance(name, model_factory, input_factory, game):
    model = model_factory()
    model.eval()
    x = input_factory(batch=2)

    result = compute_hellinger_distance(model, x, game=game)
    h2 = result["h_squared"]
    assert (h2 < 1e-3).all(), (
        f"[{name}/{game}] H^2 must be 0 for same model + same input, got {h2}"
    )
    assert_invariants(result, name=f"{name}/{game}/identity")


# ---------------------------------------------------------------------------
# Mass conservation: mu_A and mu_B (input layer + cemetery) sum to 1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,model_factory,input_factory", MODEL_FACTORIES)
@pytest.mark.parametrize("game", ["adf", "rg"])
def test_mu_mass_conservation(name, model_factory, input_factory, game):
    model = model_factory()
    model.eval()
    x = input_factory(batch=2)

    result = compute_hellinger_distance(model, x, game=game)
    mu_A_total = (
        result["mu_A_input"].flatten(start_dim=1).sum(dim=1)
        + result["mu_A_cem"].flatten(start_dim=1).sum(dim=1)
    )
    mu_B_total = (
        result["mu_B_input"].flatten(start_dim=1).sum(dim=1)
        + result["mu_B_cem"].flatten(start_dim=1).sum(dim=1)
    )

    assert torch.allclose(mu_A_total, torch.ones_like(mu_A_total), atol=1e-3), (
        f"[{name}/{game}] mu_A total must be 1, got {mu_A_total}"
    )
    assert torch.allclose(mu_B_total, torch.ones_like(mu_B_total), atol=1e-3), (
        f"[{name}/{game}] mu_B total must be 1, got {mu_B_total}"
    )


# ---------------------------------------------------------------------------
# Hellinger identity: H^2 + BC == 1 (Bhattacharyya complement)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,model_factory,input_factory", MODEL_FACTORIES)
@pytest.mark.parametrize("game", ["adf", "rg"])
def test_hellinger_bc_identity(name, model_factory, input_factory, game):
    model_A = model_factory()
    model_B = make_randomized_copy(model_A, seed=42)
    model_A.eval(); model_B.eval()

    x_A = input_factory(seed=0, batch=2)
    x_B = input_factory(seed=7, batch=2)

    result = compute_hellinger_distance(
        model_A, x_A, model_B=model_B, input_B=x_B, game=game
    )
    total = result["h_squared"] + result["bhattacharyya"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-3), (
        f"[{name}/{game}] H^2 + BC must be 1, got {total}"
    )
    assert_invariants(result, name=f"{name}/{game}/identity-check")


# ---------------------------------------------------------------------------
# Symmetry: H(A, B) == H(B, A)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,model_factory,input_factory", MODEL_FACTORIES)
@pytest.mark.parametrize("game", ["adf", "rg"])
def test_hellinger_symmetry(name, model_factory, input_factory, game):
    model_A = model_factory()
    model_B = make_randomized_copy(model_A, seed=42)
    model_A.eval(); model_B.eval()

    x = input_factory(seed=0, batch=2)
    target = torch.zeros(2, dtype=torch.long)

    rAB = compute_hellinger_distance(
        model_A, x, model_B=model_B, game=game, target=target
    )
    rBA = compute_hellinger_distance(
        model_B, x, model_B=model_A, game=game, target=target
    )

    diff = (rAB["hellinger_distance"] - rBA["hellinger_distance"]).abs().max().item()
    assert diff < 1e-3, (
        f"[{name}/{game}] H(A,B) != H(B,A): max diff = {diff:.3e}"
    )


# ---------------------------------------------------------------------------
# Boundedness and non-negativity (covered by assert_invariants)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,model_factory,input_factory", MODEL_FACTORIES)
@pytest.mark.parametrize("game", ["adf", "rg"])
def test_invariants_two_models_two_inputs(name, model_factory, input_factory, game):
    model_A = model_factory()
    model_B = make_randomized_copy(model_A, seed=11)
    model_A.eval(); model_B.eval()

    x_A = input_factory(seed=0, batch=3)
    x_B = input_factory(seed=99, batch=3)

    result = compute_hellinger_distance(
        model_A, x_A, model_B=model_B, input_B=x_B, game=game
    )
    assert_invariants(result, name=f"{name}/{game}/2x2")


# ---------------------------------------------------------------------------
# Per-pixel decomposition: ordinary pp.sum + cemetery == H^2
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,model_factory,input_factory", MODEL_FACTORIES)
@pytest.mark.parametrize("game", ["adf", "rg"])
def test_per_pixel_decomposition(name, model_factory, input_factory, game):
    model_A = model_factory()
    model_B = make_randomized_copy(model_A, seed=3)
    model_A.eval(); model_B.eval()

    x = input_factory(batch=2)
    result = compute_hellinger_distance(model_A, x, model_B=model_B, game=game)

    # The ordinary input-state map excludes the cemetery, which is the
    # separate absorbing state term h^2(†).
    total = (
        result["per_pixel_h2"].flatten(start_dim=1).sum(dim=1)
        + result["cemetery_h2"]
    )
    assert torch.allclose(total, result["h_squared"], atol=1e-3), (
        f"[{name}/{game}] per_pixel.sum + cemetery != H^2: total={total}, "
        f"h_squared={result['h_squared']}"
    )


# ---------------------------------------------------------------------------
# H = sqrt(H^2)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["adf", "rg"])
def test_h_equals_sqrt_h_squared(game):
    model_A = make_simple_conv()
    model_B = make_randomized_copy(model_A, seed=4)
    model_A.eval(); model_B.eval()
    x = make_image_input(batch=2)
    r = compute_hellinger_distance(model_A, x, model_B=model_B, game=game)
    expected = r["h_squared"].clamp(min=0).sqrt()
    diff = (r["hellinger_distance"] - expected).abs().max().item()
    assert diff < 1e-5, f"[{game}] H != sqrt(H^2): max diff = {diff:.3e}"
