"""DC activation cache tests.

Tests that:
1. The capture pass stores pre-activation values for activation layers
   (ReLU/GELU/SiLU: stores the input, not the output) and the output for
   other layers.
2. The cache is only built when requested (``store_activations=True``).
3. The DC backward with the cache (standard and stopping modes, the latter
   as used by the Stopping Game) matches the gradient at every layer.
4. The cache is consumed by the activation layers during the backward pass.
"""

import copy
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pytest
import torch
import torch.nn as nn

from bwgame.dc_context import dc_decomposition
from bwgame.functional_replacer import replace_functional_with_modules


# =============================================================================
# Models
# =============================================================================

def MLP_2layer():
    return nn.Sequential(
        nn.Linear(12, 8), nn.ReLU(),
        nn.Linear(8, 4),
    )


def MLP_3layer():
    return nn.Sequential(
        nn.Linear(12, 8), nn.ReLU(),
        nn.Linear(8, 8), nn.ReLU(),
        nn.Linear(8, 4),
    )


def MLP_5layer():
    return nn.Sequential(
        nn.Linear(12, 16), nn.ReLU(),
        nn.Linear(16, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, 8), nn.ReLU(),
        nn.Linear(8, 4),
    )


def CNN_2conv():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
        nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 2),
    )


def CNN_BN():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 4),
    )
    model.eval()
    return model


def init_random_biases(model: nn.Module) -> None:
    """Initialize weights/biases for stable gradients."""
    scale_factor = 3.0
    for module in model.modules():
        with torch.no_grad():
            if hasattr(module, "weight") and module.weight is not None:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    fan_in = module.weight.shape[1] if len(module.weight.shape) > 1 else module.weight.shape[0]
                    fan_out = module.weight.shape[0]
                    std = scale_factor * (2.0 / (fan_in + fan_out)) ** 0.5
                    module.weight.normal_(0.0, std)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.weight.uniform_(0.9, 1.1)
            if hasattr(module, "bias") and module.bias is not None:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    module.bias.uniform_(0.0, 0.1)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.bias.uniform_(-0.05, 0.05)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    return 42


# =============================================================================
# Helpers
# =============================================================================

TRACKABLE = (nn.Linear, nn.Conv2d, nn.ReLU, nn.BatchNorm2d,
             nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.Flatten)


def _l2(a, b):
    """Compute L2 difference and relative L2 between two flat arrays."""
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    diff = float(np.linalg.norm(a64 - b64))
    norm = max(float(np.linalg.norm(a64)), float(np.linalg.norm(b64)), 1e-10)
    return diff, diff / norm


def _prep_model(model):
    """Deep-copy, eval, disable inplace, replace functionals."""
    m = copy.deepcopy(model)
    m.eval()
    for mod in m.modules():
        if hasattr(mod, "inplace"):
            mod.inplace = False
    return replace_functional_with_modules(m, inplace=True)


def _capture_standard_gradient(model, x, target_grad):
    """Capture the gradient at the input and at each layer output.

    Returns: (layer_maps, input_grad) both as flat ndarrays.
    """
    m = _prep_model(model)

    captured = {}
    hooks = []
    for name, module in m.named_modules():
        if not name or not isinstance(module, TRACKABLE):
            continue

        def make_hook(n):
            def hook(mod, inp, out):
                if isinstance(out, torch.Tensor):
                    out.retain_grad()
                    captured[n] = out
            return hook
        hooks.append(module.register_forward_hook(make_hook(name)))

    x_in = x.detach().clone().requires_grad_(True)
    out = m(x_in)
    out.backward(gradient=target_grad)

    for h in hooks:
        h.remove()

    layer_maps = {}
    for n, act in captured.items():
        if act.grad is not None:
            layer_maps[n] = act.grad.detach().cpu().numpy().flatten()

    input_grad = x_in.grad.detach().cpu().numpy().flatten()
    return layer_maps, input_grad


def _capture_dc(model, x, target_grad, **dc_kw):
    """Capture the DC grad_out (pp-np-pn+nn) at each layer and at the input.

    Returns: (layer_maps, input_recon) with
        layer_maps: dict of layer_name -> flat ndarray (pp - np - pn + nn)
        input_recon: flat ndarray of the input reconstruction
    """
    m = _prep_model(model)

    layer_grad_out = {}
    hooks = []
    for name, module in m.named_modules():
        if not name or not isinstance(module, TRACKABLE):
            continue

        def make_hook(n):
            def hook(mod, grad_in, grad_out):
                g = grad_out[0]
                if g is None:
                    return
                g = g.detach()
                if g.shape[0] % 4 == 0:
                    q = g.shape[0] // 4
                    pp, np_, pn, nn_ = g[:q], g[q:2 * q], g[2 * q:3 * q], g[3 * q:]
                    layer_grad_out[n] = (pp - np_ - pn + nn_).clone()
                else:
                    layer_grad_out[n] = g.clone()
            return hook
        hooks.append(module.register_full_backward_hook(make_hook(name)))

    with dc_decomposition(m, x, **dc_kw) as dc:
        dc.output.backward(gradient=target_grad)

    for h in hooks:
        h.remove()

    layer_maps = {n: v.cpu().numpy().flatten() for n, v in layer_grad_out.items()}
    sens = dc.sensitivities
    input_recon = (sens.delta_pp - sens.delta_np - sens.delta_pn + sens.delta_nn
                   ).detach().cpu().numpy().flatten()
    return layer_maps, input_recon


MODELS = [
    ("MLP_2layer", MLP_2layer, (1, 12)),
    ("MLP_3layer", MLP_3layer, (1, 12)),
    ("MLP_5layer", MLP_5layer, (1, 12)),
    ("CNN_2conv", CNN_2conv, (1, 1, 16, 16)),
    ("CNN_BN", CNN_BN, (1, 1, 16, 16)),
]

GRAD_LAYER_REL_L2_TOL = 1e-3
GRAD_INPUT_REL_L2_TOL = 1e-4


@dataclass
class LayerResult:
    name: str
    dc_norm: float
    ref_norm: float
    l2_diff: float
    rel_l2: float


@dataclass
class ComparisonResult:
    model_name: str
    mode: str
    input_rel_l2: float
    layer_results: List[LayerResult] = field(default_factory=list)
    max_layer_rel_l2: float = 0.0


def _print_result(r: ComparisonResult):
    """Print formatted comparison table."""
    print(f"\n{'=' * 100}")
    print(f"Model: {r.model_name}  |  Mode: {r.mode}")
    print(f"  Input rel L2: {r.input_rel_l2:.2e}  |  Max layer rel L2: {r.max_layer_rel_l2:.2e}")
    print(f"{'=' * 100}")
    print(f"  {'Layer':<30s} {'||DC||':>10s} {'||Ref||':>10s} "
          f"{'L2 diff':>10s} {'Rel L2':>10s}")
    print(f"  {'-' * 70}")
    for lr in r.layer_results:
        print(f"  {lr.name:<30s} {lr.dc_norm:>10.2e} {lr.ref_norm:>10.2e} "
              f"{lr.l2_diff:>10.2e} {lr.rel_l2:>10.2e}")


def _setup(model_fn, input_shape):
    torch.manual_seed(42)
    model = model_fn()
    init_random_biases(model)
    model.eval()
    x = torch.randn(*input_shape)
    with torch.no_grad():
        out = model(x)
    tg = torch.zeros_like(out)
    tg[0, out.argmax(dim=1).item()] = 1.0
    return model, x, tg


def _compare(model_name, mode, model, x, tg, **dc_kw):
    ref_maps, ref_input = _capture_standard_gradient(model, x, tg)
    dc_maps, dc_input = _capture_dc(model, x, tg, **dc_kw)
    _, input_rel = _l2(dc_input, ref_input)
    result = ComparisonResult(model_name=model_name, mode=mode, input_rel_l2=input_rel)
    for name in sorted(set(dc_maps) & set(ref_maps)):
        if dc_maps[name].shape != ref_maps[name].shape:
            continue
        l2, rel = _l2(dc_maps[name], ref_maps[name])
        result.layer_results.append(LayerResult(
            name=name,
            dc_norm=float(np.linalg.norm(dc_maps[name].astype(np.float64))),
            ref_norm=float(np.linalg.norm(ref_maps[name].astype(np.float64))),
            l2_diff=l2, rel_l2=rel,
        ))
    if result.layer_results:
        result.max_layer_rel_l2 = max(lr.rel_l2 for lr in result.layer_results)
    _print_result(result)
    return result


# =============================================================================
# Test 1: DC backward with the cache matches the gradient per layer
# =============================================================================

class TestDCWithCache:
    """DC backward (with the activation cache) matches the gradient at every layer."""

    @pytest.mark.parametrize("model_name,model_fn,input_shape", MODELS,
                             ids=[m[0] for m in MODELS])
    def test_standard_with_cache_matches_gradient(self, model_name, model_fn,
                                                   input_shape):
        model, x, tg = _setup(model_fn, input_shape)
        result = _compare(model_name, "standard+cache", model, x, tg,
                          backprop_mode="standard", store_activations=True)
        assert result.input_rel_l2 < GRAD_INPUT_REL_L2_TOL, (
            f"{model_name}: input rel L2 = {result.input_rel_l2:.2e}")
        assert result.max_layer_rel_l2 < GRAD_LAYER_REL_L2_TOL, (
            f"{model_name}: max layer rel L2 = {result.max_layer_rel_l2:.2e}")

    @pytest.mark.parametrize("model_name,model_fn,input_shape", MODELS,
                             ids=[m[0] for m in MODELS])
    def test_stopping_with_cache_matches_gradient(self, model_name, model_fn,
                                                  input_shape):
        """Stopping mode with the cache: the ReLU mask from the cache equals
        the mask from z = pos - neg."""
        model, x, tg = _setup(model_fn, input_shape)
        result = _compare(model_name, "stopping+cache", model, x, tg,
                          backprop_mode="stopping", store_activations=True)
        assert result.input_rel_l2 < GRAD_INPUT_REL_L2_TOL, (
            f"{model_name}: input rel L2 = {result.input_rel_l2:.2e}")
        assert result.max_layer_rel_l2 < GRAD_LAYER_REL_L2_TOL, (
            f"{model_name}: max layer rel L2 = {result.max_layer_rel_l2:.2e}")


# =============================================================================
# Test 2: Activation cache correctness
# =============================================================================

class TestActivationCacheCorrectness:
    """The activation cache stores the correct pre-activation values."""

    def test_relu_stores_preactivation(self):
        """For ReLU, the cached activation is the input (pre-ReLU), not the output."""
        from bwgame.game_theory.game_context import run_capture_pass, cleanup_layer_data, DC_LAYER_DATA
        from bwgame.patcher import patch_model, unpatch_model

        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        model.eval()
        replace_functional_with_modules(model, inplace=True)
        x = torch.randn(1, 10)

        with torch.no_grad():
            z_ref = model[0](x)  # Linear output = ReLU input

        patch_model(model)
        try:
            run_capture_pass(model, x, capture_activations=True)
            ld = getattr(model[1], DC_LAYER_DATA, None)
            assert ld is not None, "ReLU should have _dc_layer_data"
            assert ld.activation is not None, "ReLU activation should be cached"
            assert torch.allclose(ld.activation, z_ref, atol=1e-6), (
                f"Cached activation should be pre-ReLU, got diff = "
                f"{(ld.activation - z_ref).abs().max():.2e}")
        finally:
            cleanup_layer_data(model)
            unpatch_model(model)

    def test_linear_stores_output(self):
        """For Linear, the cached activation is the output."""
        from bwgame.game_theory.game_context import run_capture_pass, cleanup_layer_data, DC_LAYER_DATA
        from bwgame.patcher import patch_model, unpatch_model

        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        model.eval()
        replace_functional_with_modules(model, inplace=True)
        x = torch.randn(1, 10)

        with torch.no_grad():
            out_ref = model[0](x)

        patch_model(model)
        try:
            run_capture_pass(model, x, capture_activations=True)
            ld = getattr(model[0], DC_LAYER_DATA, None)
            assert ld is not None
            assert ld.activation is not None
            assert torch.allclose(ld.activation, out_ref, atol=1e-6)
        finally:
            cleanup_layer_data(model)
            unpatch_model(model)

    def test_no_cache_by_default(self):
        """Without store_activations (and without ADF) no capture pass runs."""
        from bwgame.game_theory.game_context import DC_LAYER_DATA

        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        model.eval()
        x = torch.randn(1, 10)

        with dc_decomposition(model, x) as dc:
            present = [n for n, m in model.named_modules() if hasattr(m, DC_LAYER_DATA)]
            dc.output.sum().backward()

        assert not present, f"Modules {present} should not have _dc_layer_data"

    def test_cache_with_store_activations(self):
        """store_activations=True builds the cache; it is removed on exit."""
        from bwgame.game_theory.game_context import DC_LAYER_DATA

        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        model.eval()
        x = torch.randn(1, 10)

        with dc_decomposition(model, x, store_activations=True, backprop_mode='stopping') as dc:
            present = [n for n, m in model.named_modules() if hasattr(m, DC_LAYER_DATA)]
            dc.output.sum().backward()

        assert present, "Some modules should have _dc_layer_data with store_activations=True"
        for name, module in model.named_modules():
            assert not hasattr(module, DC_LAYER_DATA), (
                f"Module {name} should not have _dc_layer_data after context exit")

    def test_cache_relu_used_in_backward(self):
        """The cached pre-activation is read by the ReLU backward and equals
        the Linear output."""
        from bwgame.game_theory.game_context import DC_LAYER_DATA

        model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
        model.eval()
        init_random_biases(model)
        x = torch.randn(1, 10)

        with torch.no_grad():
            out = model(x)
        tg = torch.zeros_like(out)
        tg[0, 0] = 1.0

        with dc_decomposition(model, x, store_activations=True,
                              backprop_mode='stopping') as dc:
            ld = getattr(model[1], DC_LAYER_DATA, None)
            assert ld is not None, "ReLU should have layer_data during dc_decomposition"
            assert ld.activation is not None, "ReLU should have cached activation"
            cached_z = ld.activation.clone()
            dc.output.backward(gradient=tg)
            assert ld.activation is None, "The ReLU backward should consume the cached activation"

        with torch.no_grad():
            expected_z = model[0](x)
        assert torch.allclose(cached_z, expected_z, atol=1e-6), (
            "Cached z should match Linear output (pre-ReLU)")
