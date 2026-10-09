"""RG (Active Routing Game) -specific tests.

The RG kernel is fully activation-dependent:
    T_M((j,w_j) -> (i,w_i)) = c_{w_i,w_j} * [a_M(i) * W_M[j,i]]^{w_j xor w_i}^{1/tau} / Z_M(j)

Distinguishing behaviors:

1. Same model + two different inputs: H^2 > 0 because activations differ
   (in contrast to ADF which only depends on dead patterns).
2. Temperature tau controls peakedness of the routing distribution.
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
    make_simple_conv,
    make_simple_mlp,
)


def test_rg_two_inputs_yield_positive_distance():
    """RG kernel sees activations directly so two distinct inputs through
    the same model must produce H^2 > 0."""
    model = make_simple_mlp()
    model.eval()
    x_A = make_mlp_input(seed=0)
    x_B = make_mlp_input(seed=7)
    r = compute_hellinger_distance(model, x_A, input_B=x_B, game="rg")
    assert (r["h_squared"] > 1e-3).all()
    assert_invariants(r, name="rg/two-inputs")


def test_rg_temperature_changes_distance():
    """Different tau values must (in general) produce different H^2."""
    model = make_simple_mlp()
    model.eval()
    x_A = make_mlp_input(seed=0)
    x_B = make_mlp_input(seed=7)

    r1 = compute_hellinger_distance(model, x_A, input_B=x_B, game="rg", tau=1.0)
    r2 = compute_hellinger_distance(model, x_A, input_B=x_B, game="rg", tau=0.5)
    diff = (r1["h_squared"] - r2["h_squared"]).abs().max().item()
    assert diff > 1e-4, f"tau should affect H^2, but max diff = {diff:.3e}"
    assert_invariants(r1, name="rg/tau=1.0")
    assert_invariants(r2, name="rg/tau=0.5")


@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0])
def test_rg_invariants_under_various_tau(tau):
    model_A = make_simple_conv()
    model_B = make_randomized_copy(model_A, seed=11)
    model_A.eval(); model_B.eval()
    x = make_image_input(batch=2)
    r = compute_hellinger_distance(model_A, x, model_B=model_B, game="rg", tau=tau)
    assert_invariants(r, name=f"rg/tau={tau}")


def test_rg_conditioned_identity_zero_and_cemetery_removed():
    model = make_simple_conv()
    model.eval()
    x = make_image_input(batch=2)

    r = compute_hellinger_distance(
        model,
        x,
        game="rg",
        condition_on_survival=True,
    )

    assert (r["h_squared"] < 1e-6).all()
    assert torch.allclose(r["mu_A_input"].flatten(1).sum(1), torch.ones(2), atol=1e-5)
    assert torch.allclose(r["mu_B_input"].flatten(1).sum(1), torch.ones(2), atol=1e-5)
    assert torch.allclose(r["mu_A_cem"], torch.zeros_like(r["mu_A_cem"]), atol=1e-6)
    assert torch.allclose(r["mu_B_cem"], torch.zeros_like(r["mu_B_cem"]), atol=1e-6)
    assert torch.allclose(r["beta_cem"], torch.zeros_like(r["beta_cem"]), atol=1e-6)
    assert torch.allclose(r["cemetery_h2"], torch.zeros_like(r["cemetery_h2"]), atol=1e-6)
    assert torch.allclose(r["h_squared"] + r["bhattacharyya"], torch.ones(2), atol=1e-5)
    assert_invariants(r, name="rg/conditioned/identity")


def test_rg_conditioned_matches_posthoc_formula():
    model_A = make_simple_mlp()
    model_B = make_randomized_copy(model_A, seed=42)
    model_A.eval(); model_B.eval()
    x = make_mlp_input(batch=2)

    uncond = compute_hellinger_distance(
        model_A,
        x,
        model_B=model_B,
        game="rg",
    )
    live = compute_hellinger_distance(
        model_A,
        x,
        model_B=model_B,
        game="rg",
        condition_on_survival=True,
    )

    Z_A = uncond["mu_A_input"].flatten(1).sum(1)
    Z_B = uncond["mu_B_input"].flatten(1).sum(1)
    denom = torch.sqrt((Z_A * Z_B).clamp(min=1e-8))
    expected_bc = uncond["beta_input_ordinary"].flatten(1).sum(1) / denom
    expected_h2 = 1.0 - expected_bc

    assert torch.allclose(live["survival_mass_A"], Z_A, atol=1e-5)
    assert torch.allclose(live["survival_mass_B"], Z_B, atol=1e-5)
    assert torch.allclose(live["bhattacharyya"], expected_bc, atol=1e-5)
    assert torch.allclose(live["h_squared"], expected_h2, atol=1e-5)
    assert torch.allclose(live["mu_A_input"].flatten(1).sum(1), torch.ones(2), atol=1e-5)
    assert torch.allclose(live["mu_B_input"].flatten(1).sum(1), torch.ones(2), atol=1e-5)
    assert torch.allclose(live["cemetery_h2"], torch.zeros_like(live["cemetery_h2"]), atol=1e-6)
    assert_invariants(live, name="rg/conditioned/posthoc")


def test_rg_conditioned_live_maps_match_alpha_beta_formulae():
    model_A = make_simple_mlp()
    model_B = make_randomized_copy(model_A, seed=42)
    model_A.eval(); model_B.eval()
    x = make_mlp_input(batch=2)

    uncond = compute_hellinger_distance(
        model_A,
        x,
        model_B=model_B,
        game="rg",
    )
    live = compute_hellinger_distance(
        model_A,
        x,
        model_B=model_B,
        game="rg",
        condition_on_survival=True,
    )

    Z_A = uncond["mu_A_input"].flatten(1).sum(1)
    Z_B = uncond["mu_B_input"].flatten(1).sum(1)
    scale_A = Z_A.view(-1, *([1] * (uncond["alpha_A_input"].dim() - 1)))
    scale_B = Z_B.view(-1, *([1] * (uncond["alpha_B_input"].dim() - 1)))
    scale_beta = torch.sqrt((Z_A * Z_B).clamp(min=1e-8)).view(
        -1, *([1] * (uncond["beta_input_ordinary"].dim() - 1))
    )
    expected_alpha_A = uncond["alpha_A_input"] / scale_A
    expected_alpha_B = uncond["alpha_B_input"] / scale_B
    expected_beta = uncond["beta_input_ordinary"] / scale_beta
    local_mass = 0.5 * (expected_alpha_A + expected_alpha_B)
    expected_h2 = (local_mass - expected_beta).clamp(min=0)
    expected_marg = (
        local_mass - torch.sqrt((expected_alpha_A * expected_alpha_B).clamp(min=0))
    ).clamp(min=0)
    expected_frac = torch.where(
        local_mass > 1e-8,
        expected_h2 / local_mass.clamp(min=1e-8),
        torch.zeros_like(expected_h2),
    ).clamp(min=0, max=1)

    assert torch.allclose(live["alpha_A_input"], expected_alpha_A, atol=1e-6)
    assert torch.allclose(live["alpha_B_input"], expected_alpha_B, atol=1e-6)
    assert torch.allclose(live["beta_input_ordinary"], expected_beta, atol=1e-6)
    assert torch.allclose(live["per_pixel_h2"], expected_h2, atol=1e-6)
    assert torch.allclose(live["per_pixel_h2_live"], expected_h2, atol=1e-6)
    assert torch.allclose(live["per_pixel_h2_marginal"], expected_marg, atol=1e-6)
    assert torch.allclose(live["per_pixel_h2_marginal_live"], expected_marg, atol=1e-6)
    assert torch.allclose(live["per_pixel_h2_fraction"], expected_frac, atol=1e-6)
    assert torch.allclose(live["per_pixel_h2_fraction_live"], expected_frac, atol=1e-6)
    assert (live["per_pixel_h2"] + 1e-6 >= live["per_pixel_h2_marginal"]).all()
    assert torch.allclose(
        live["per_pixel_h2"].flatten(start_dim=1).sum(dim=1),
        live["h_squared"],
        atol=1e-5,
    )
