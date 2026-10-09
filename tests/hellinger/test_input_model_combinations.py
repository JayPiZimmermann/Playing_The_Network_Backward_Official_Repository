"""All four input/model combinations of the Hellinger diagnostic.

The diagnostic supports four configurations:
    (a) one model + one input
    (b) one model + two inputs
    (c) two models + one input
    (d) two models + two inputs

Each combination should:
- preserve mass (mu_A and mu_B totals == 1)
- satisfy the Hellinger identity (H^2 + BC == 1)
- produce specific behaviors documented below
"""

from __future__ import annotations

import pytest

from bwgame.xai.methods.hellinger import (
    HellingerDiagnostic,
    compute_hellinger_distance,
)

from .conftest import (
    assert_invariants,
    make_image_input,
    make_mlp_input,
    make_randomized_copy,
    make_simple_conv,
    make_simple_mlp,
)


# ---------------------------------------------------------------------------
# (a) Same model + same input  ->  H^2 == 0 always
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["adf", "rg"])
def test_one_model_one_input_zero_distance(game):
    model = make_simple_mlp()
    model.eval()
    x = make_mlp_input(batch=2)

    diag = HellingerDiagnostic(model, game=game)
    result = diag.compute(x)
    assert (result["h_squared"] < 1e-3).all()
    assert_invariants(result, name=f"one-model-one-input/{game}")


@pytest.mark.parametrize("game", ["adf", "rg"])
def test_one_model_one_input_conv_zero_distance(game):
    model = make_simple_conv()
    model.eval()
    x = make_image_input(batch=2)

    result = compute_hellinger_distance(model, x, game=game)
    assert (result["h_squared"] < 1e-3).all()
    assert_invariants(result, name=f"one-model-one-input-conv/{game}")


# ---------------------------------------------------------------------------
# (b) Same model + two different inputs
#
# ADF kernel is independent of activations, so the only source of disagreement
# is the cemetery (different dead-neuron patterns).  RG kernel is fully
# activation-dependent, so disagreement should be larger in general.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["adf", "rg"])
def test_one_model_two_inputs(game):
    model = make_simple_mlp()
    model.eval()
    x_A = make_mlp_input(seed=0, batch=2)
    x_B = make_mlp_input(seed=11, batch=2)

    result = compute_hellinger_distance(model, x_A, input_B=x_B, game=game)
    assert_invariants(result, name=f"one-model-two-inputs/{game}")


def test_one_model_two_inputs_rg_strictly_positive():
    """RG depends on activations, so two distinct inputs must give H^2 > 0."""
    model = make_simple_mlp()
    model.eval()
    x_A = make_mlp_input(seed=0, batch=1)
    x_B = make_mlp_input(seed=11, batch=1)

    result = compute_hellinger_distance(model, x_A, input_B=x_B, game="rg")
    assert (result["h_squared"] > 1e-3).all(), (
        f"RG with two inputs should produce H^2 > 0, got {result['h_squared']}"
    )


# ---------------------------------------------------------------------------
# (c) Two models + same input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["adf", "rg"])
def test_two_models_one_input(game):
    model_A = make_simple_mlp()
    model_B = make_randomized_copy(model_A, seed=99)
    model_A.eval(); model_B.eval()
    x = make_mlp_input(batch=2)

    diag = HellingerDiagnostic(model_A, model_B=model_B, game=game)
    result = diag.compute(x)
    assert_invariants(result, name=f"two-models-one-input/{game}")
    # Different weights should produce strictly positive disagreement
    assert (result["h_squared"] > 1e-3).all()


@pytest.mark.parametrize("game", ["adf", "rg"])
def test_two_models_one_input_conv(game):
    model_A = make_simple_conv()
    model_B = make_randomized_copy(model_A, seed=99)
    model_A.eval(); model_B.eval()
    x = make_image_input(batch=2)

    result = compute_hellinger_distance(model_A, x, model_B=model_B, game=game)
    assert_invariants(result, name=f"two-models-one-input-conv/{game}")
    assert (result["h_squared"] > 1e-3).all()


# ---------------------------------------------------------------------------
# (d) Two models + two inputs (most general case)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("game", ["adf", "rg"])
def test_two_models_two_inputs(game):
    model_A = make_simple_mlp()
    model_B = make_randomized_copy(model_A, seed=42)
    model_A.eval(); model_B.eval()
    x_A = make_mlp_input(seed=0, batch=3)
    x_B = make_mlp_input(seed=99, batch=3)

    result = compute_hellinger_distance(
        model_A, x_A, model_B=model_B, input_B=x_B, game=game
    )
    assert_invariants(result, name=f"two-models-two-inputs/{game}")


# ---------------------------------------------------------------------------
# Cache independence: model_B defaults to model_A but caches must be distinct
# ---------------------------------------------------------------------------

def test_cache_independence_when_model_b_defaults():
    model = make_simple_mlp()
    model.eval()
    x_A = make_mlp_input(seed=0, batch=1)
    x_B = make_mlp_input(seed=5, batch=1)

    diag = HellingerDiagnostic(model, game="rg")
    # If caches were aliased, second call would be polluted by first
    r1 = diag.compute(x_A, input_B=x_B)
    r2 = diag.compute(x_A, input_B=x_B)
    diff = (r1["h_squared"] - r2["h_squared"]).abs().max().item()
    assert diff < 1e-5, f"Repeated call gave different result: diff = {diff:.3e}"
