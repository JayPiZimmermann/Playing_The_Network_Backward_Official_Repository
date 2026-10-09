"""ADF game-specific tests.

The ADF game uses a sign-split backward kernel:
    T_M(s_{j,+} -> s_{i,+}) = W^+_M[j, i] / gamma_M(j)
    T_M(s_{j,+} -> s_{i,-}) = W^-_M[j, i] / gamma_M(j)
with the symmetric rule from ``s_{j,-}``.
This implies several distinguishing behaviors:

1. Same model + same input: H^2 == 0 (kernel + cemetery agree).
2. Same model + two different inputs: kernels are identical, but the
   *cemetery* (dead-neuron pattern) is activation-dependent.  H^2 > 0 only
   if the two inputs activate disjoint sets of neurons.
3. Two models + same input: H^2 > 0 because kernels differ.
4. For a bias-free homogeneous network, scaling a positive input by a
   positive scalar preserves the dead pattern at every layer, hence H^2 == 0
   under ADF (the kernel is also unchanged because it depends only on |W|).
"""

from __future__ import annotations

import torch
import torch.nn as nn

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


def _make_no_bias_mlp(seed: int = 0) -> nn.Module:
    """A bias-free MLP: scaling a positive input scales every activation
    proportionally, so the dead pattern is fully preserved."""
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(20, 32, bias=False), nn.ReLU(),
        nn.Linear(32, 16, bias=False), nn.ReLU(),
        nn.Linear(16, 5, bias=False),
    )


class _TinyCatAddNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, 4))
        self.pos_embed = nn.Parameter(torch.randn(1, 3, 4))
        self.proj = nn.Linear(12, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.cls_token.expand(x.size(0), -1, -1)
        y = torch.cat([cls, x], dim=1)
        y = y + self.pos_embed.expand(x.size(0), -1, -1)
        return self.proj(torch.flatten(y, start_dim=1))


def test_adf_no_bias_positive_scaling_zero():
    """For a bias-free MLP, positive rescaling of a positive input
    preserves all dead patterns and the ADF kernel itself, so H^2 == 0."""
    model = _make_no_bias_mlp()
    model.eval()
    x = make_mlp_input(batch=2)
    x_scaled = 2.5 * x

    # Pin target so the rescaling cannot change argmax.
    target = torch.zeros(2, dtype=torch.long)
    r = compute_hellinger_distance(
        model, x, input_B=x_scaled, game="adf", target=target
    )
    assert (r["h_squared"] < 1e-3).all(), (
        f"ADF on bias-free network with positive rescaling should give "
        f"H^2 == 0, got {r['h_squared']}"
    )


def test_adf_two_distinct_inputs_only_differ_via_cemetery():
    """ADF with the same model on two distinct inputs: any disagreement
    must come from the cemetery (different dead-neuron patterns) — the
    kernel itself is identical.  The cemetery is a single absorbing state,
    so only the scalar cemetery term can differ between such inputs."""
    model = make_simple_mlp()
    model.eval()
    x_A = make_mlp_input(seed=0)
    x_B = make_mlp_input(seed=11)
    r = compute_hellinger_distance(model, x_A, input_B=x_B, game="adf")
    mu_A_cem_mass = r["mu_A_cem"].flatten(start_dim=1).sum(dim=1)
    mu_B_cem_mass = r["mu_B_cem"].flatten(start_dim=1).sum(dim=1)
    # If neither input causes any cemetery absorption, then h^2 must be 0.
    if (mu_A_cem_mass.max() < 1e-5) and (mu_B_cem_mass.max() < 1e-5):
        assert (r["h_squared"] < 1e-3).all()
    assert_invariants(r, name="adf/two-inputs")


def test_adf_two_models_same_input_positive_distance():
    model_A = make_simple_mlp()
    model_B = make_randomized_copy(model_A, seed=42)
    model_A.eval(); model_B.eval()
    x = make_mlp_input(batch=1)
    r = compute_hellinger_distance(model_A, x, model_B=model_B, game="adf")
    assert (r["h_squared"] > 1e-3).all()
    assert_invariants(r, name="adf/two-models-one-input")


def test_adf_sign_flip_between_models_is_not_quotiented_away():
    """Opposite weight signs with identical magnitudes must disagree.

    A spatial-only quotient would collapse these two kernels to the same
    ``|W| / gamma`` row and incorrectly report zero distance.  The
    sign-split ADF path law keeps the ``(+,-)`` coordinate, so the two
    models disagree even though their absolute weights match.
    """
    model_A = nn.Sequential(nn.Linear(1, 1, bias=False))
    model_B = nn.Sequential(nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model_A[0].weight.fill_(1.0)
        model_B[0].weight.fill_(-1.0)
    model_A.eval(); model_B.eval()

    x = torch.ones(1, 1)
    target = torch.zeros(1, dtype=torch.long)
    diag = HellingerDiagnostic(model_A, model_B=model_B, game="adf")
    r = diag.compute(x, x, target=target)

    assert float(r["hellinger_distance"][0]) > 0.99
    assert float(r["bhattacharyya"][0]) < 1e-6


def test_adf_per_module_h2_populated():
    """ADF backward sweep should populate per_module_h2 for every leaf."""
    model = make_simple_conv()
    model.eval()
    x_A = make_image_input(seed=0)
    x_B = make_image_input(seed=7)
    r = compute_hellinger_distance(model, x_A, input_B=x_B, game="adf")
    assert isinstance(r["per_module_h2"], dict)
    assert len(r["per_module_h2"]) > 0


def test_adf_cemetery_grows_with_model_disagreement():
    """When model_A and model_B differ a lot, the cemetery should pick up
    significant mass under ADF (different dead patterns).  The cemetery is
    a separate absorbing coordinate, so total mass is the ordinary input
    mass plus that scalar cemetery mass."""
    model_A = make_simple_mlp()
    model_B = make_randomized_copy(model_A, seed=12345)
    model_A.eval(); model_B.eval()
    x = make_mlp_input(batch=1)
    r = compute_hellinger_distance(model_A, x, model_B=model_B, game="adf")
    assert (r["mu_A_cem"] >= -1e-5).all()
    assert (r["mu_B_cem"] >= -1e-5).all()
    total_A = (
        r["mu_A_input"].flatten(1).sum(1)
        + r["mu_A_cem"].flatten(1).sum(1)
    )
    total_B = (
        r["mu_B_input"].flatten(1).sum(1)
        + r["mu_B_cem"].flatten(1).sum(1)
    )
    assert torch.allclose(total_A, torch.ones_like(total_A), atol=1e-3)
    assert torch.allclose(total_B, torch.ones_like(total_B), atol=1e-3)


def test_adf_conditioned_identity_zero_and_cemetery_removed():
    model = make_simple_conv()
    model.eval()
    x = make_image_input(batch=2)

    r = compute_hellinger_distance(
        model,
        x,
        game="adf",
        condition_on_survival=True,
    )

    assert (r["h_squared"] < 1e-3).all()
    assert torch.allclose(r["mu_A_input"].flatten(1).sum(1), torch.ones(2), atol=1e-3)
    assert torch.allclose(r["mu_B_input"].flatten(1).sum(1), torch.ones(2), atol=1e-3)
    assert torch.allclose(r["mu_A_cem"], torch.zeros_like(r["mu_A_cem"]), atol=1e-6)
    assert torch.allclose(r["mu_B_cem"], torch.zeros_like(r["mu_B_cem"]), atol=1e-6)
    assert torch.allclose(r["beta_cem"], torch.zeros_like(r["beta_cem"]), atol=1e-6)
    assert torch.allclose(r["cemetery_h2"], torch.zeros_like(r["cemetery_h2"]), atol=1e-6)
    assert (r["survival_mass_A"] < 1.0).any()
    assert (r["survival_mass_B"] < 1.0).any()
    assert_invariants(r, name="adf/conditioned/identity")


def test_adf_conditioned_two_models_normalize_live_path_law():
    model_A = make_simple_mlp()
    model_B = make_randomized_copy(model_A, seed=42)
    model_A.eval(); model_B.eval()
    x = make_mlp_input(batch=2)

    r = compute_hellinger_distance(
        model_A,
        x,
        model_B=model_B,
        game="adf",
        condition_on_survival=True,
    )

    total = r["h_squared"] + r["bhattacharyya"]
    assert torch.allclose(total, torch.ones_like(total), atol=1e-3)
    assert torch.allclose(r["mu_A_input"].flatten(1).sum(1), torch.ones(2), atol=1e-3)
    assert torch.allclose(r["mu_B_input"].flatten(1).sum(1), torch.ones(2), atol=1e-3)
    assert torch.allclose(r["mu_A_cem"], torch.zeros_like(r["mu_A_cem"]), atol=1e-6)
    assert torch.allclose(r["mu_B_cem"], torch.zeros_like(r["mu_B_cem"]), atol=1e-6)
    assert torch.allclose(r["beta_cem"], torch.zeros_like(r["beta_cem"]), atol=1e-6)
    assert (r["survival_mass_A"] < 1.0 + 1e-6).all()
    assert (r["survival_mass_B"] < 1.0 + 1e-6).all()
    assert_invariants(r, name="adf/conditioned/two-models")


def test_adf_conditioned_live_maps_match_alpha_beta_formulae():
    model_A = make_simple_mlp()
    model_B = make_randomized_copy(model_A, seed=42)
    model_A.eval(); model_B.eval()
    x = make_mlp_input(batch=2)

    r = compute_hellinger_distance(
        model_A,
        x,
        model_B=model_B,
        game="adf",
        condition_on_survival=True,
    )

    alpha_A = r["alpha_A_input"]
    alpha_B = r["alpha_B_input"]
    beta = r["beta_input_ordinary"]
    local_mass = 0.5 * (alpha_A + alpha_B)
    expected_h2 = (local_mass - beta).clamp(min=0)
    expected_marg = (
        local_mass - torch.sqrt((alpha_A * alpha_B).clamp(min=0))
    ).clamp(min=0)
    expected_frac = torch.where(
        local_mass > 1e-8,
        expected_h2 / local_mass.clamp(min=1e-8),
        torch.zeros_like(expected_h2),
    ).clamp(min=0, max=1)

    assert torch.allclose(r["per_pixel_h2"], expected_h2, atol=1e-6)
    assert torch.allclose(r["per_pixel_h2_live"], expected_h2, atol=1e-6)
    assert torch.allclose(r["per_pixel_h2_marginal"], expected_marg, atol=1e-6)
    assert torch.allclose(r["per_pixel_h2_marginal_live"], expected_marg, atol=1e-6)
    assert torch.allclose(r["per_pixel_h2_fraction"], expected_frac, atol=1e-6)
    assert torch.allclose(r["per_pixel_h2_fraction_live"], expected_frac, atol=1e-6)
    assert (r["per_pixel_h2"] + 1e-6 >= r["per_pixel_h2_marginal"]).all()
    assert (r["per_pixel_h2_fraction"] >= -1e-6).all()
    assert (r["per_pixel_h2_fraction"] <= 1.0 + 1e-6).all()
    assert torch.allclose(
        r["per_pixel_h2"].flatten(start_dim=1).sum(dim=1),
        r["h_squared"],
        atol=1e-5,
    )


def test_adf_conditioned_handles_constant_cat_and_add_branches():
    torch.manual_seed(0)
    model = _TinyCatAddNet().eval()
    x = torch.randn(2, 2, 4)
    target = torch.zeros(2, dtype=torch.long)

    r = compute_hellinger_distance(
        model,
        x,
        game="adf",
        target=target,
        condition_on_survival=True,
    )

    assert (r["h_squared"] < 1e-6).all()
    assert torch.allclose(r["mu_A_input"].flatten(1).sum(1), torch.ones(2), atol=1e-5)
    assert torch.allclose(r["mu_B_input"].flatten(1).sum(1), torch.ones(2), atol=1e-5)
    assert (r["survival_mass_A"] < 1.0 - 1e-6).all()
    assert (r["survival_mass_B"] < 1.0 - 1e-6).all()


def test_adf_conditioned_prefers_cls_slice_over_same_numel_zero_tensor():
    diag = HellingerDiagnostic(nn.Identity(), game="adf")
    target = torch.randn(2, 4)
    seq_h = torch.randn(2, 2, 3, 4).abs()
    zero_h_3d = torch.zeros(2, 2, 1, 4)
    zero_h_2d = torch.zeros(2, 2, 4)

    survival = {
        1: zero_h_3d,
        2: zero_h_2d,
        3: seq_h,
    }

    resolved = diag._resolve_adf_survival(target, survival)

    assert resolved is not None
    assert torch.allclose(resolved, seq_h[:, :, 0, :])
