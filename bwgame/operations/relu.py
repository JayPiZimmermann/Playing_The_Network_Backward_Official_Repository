"""DC Decomposition for ReLU. Forward/Backward: [4*batch] -> [4*batch]

The ReLU split is controlled by a continuous parameter relu_lambda in [0, 1]:
  - lambda=1.0: max mode (forward_pos = max{a, b}, forward_neg = b)
  - lambda=0.0: min mode (forward_pos = a, forward_neg = min{a, b})
  - 0 < lambda < 1: weighted combination of max and min modes
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

from .base import (
    DC_ENABLED, DC_ORIGINAL_FORWARD, DC_RELU_LAMBDA, DC_IS_OUTPUT_LAYER,
    DC_ADF_FORWARD, DC_LAYER_DATA, get_adf_config,
)
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_unpatch_function,
)

DC_BACKPROP_MODE = '_dc_backprop_mode'


# Forward positive subfunctions
def _forward_pos_max(pos: Tensor, neg: Tensor) -> Tensor:
    """Max mode: forward_pos = max{a, b}"""
    return torch.max(pos, neg)


def _forward_pos_min(pos: Tensor, _neg: Tensor) -> Tensor:
    """Min mode: forward_pos = a"""
    return pos


# Forward negative subfunctions
def _forward_neg_max(_pos: Tensor, neg: Tensor) -> Tensor:
    """Max mode: forward_neg = b"""
    return neg


def _forward_neg_min(pos: Tensor, neg: Tensor) -> Tensor:
    """Min mode: forward_neg = min{a, b}"""
    return torch.min(pos, neg)


def forward_relu(pos: Tensor, neg: Tensor, relu_lambda: float) -> Tuple[Tensor, Tensor]:
    """DC forward for ReLU with continuous lambda parameter.

    Args:
        pos: Positive stream tensor.
        neg: Negative stream tensor.
        relu_lambda: Continuous split parameter in [0, 1].
            1.0 = max mode, 0.0 = min mode, 0.5 = half (average).
    """
    if relu_lambda == 1.0:
        return _forward_pos_max(pos, neg), _forward_neg_max(pos, neg)
    elif relu_lambda == 0.0:
        return _forward_pos_min(pos, neg), _forward_neg_min(pos, neg)
    else:
        out_pos = relu_lambda * _forward_pos_max(pos, neg) + (1 - relu_lambda) * _forward_pos_min(pos, neg)
        out_neg = relu_lambda * _forward_neg_max(pos, neg) + (1 - relu_lambda) * _forward_neg_min(pos, neg)
        return out_pos, out_neg


def backward_relu(
    delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor,
    mp: Tensor, mn: Tensor, relu_lambda: float, backprop_mode: str
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """DC backward for ReLU with continuous lambda parameter."""
    if relu_lambda == 1.0:
        return _backprop_max(delta_pp, delta_np, delta_pn, delta_nn, mp, mn, backprop_mode)
    elif relu_lambda == 0.0:
        return _backprop_min(delta_pp, delta_np, delta_pn, delta_nn, mp, mn, backprop_mode)
    else:
        pp_max, np_max, pn_max, nn_max = _backprop_max(
            delta_pp, delta_np, delta_pn, delta_nn, mp, mn, backprop_mode)
        pp_min, np_min, pn_min, nn_min = _backprop_min(
            delta_pp, delta_np, delta_pn, delta_nn, mp, mn, backprop_mode)
        return (
            relu_lambda * pp_max + (1 - relu_lambda) * pp_min,
            relu_lambda * np_max + (1 - relu_lambda) * np_min,
            relu_lambda * pn_max + (1 - relu_lambda) * pn_min,
            relu_lambda * nn_max + (1 - relu_lambda) * nn_min,
        )


class DCReLUFunction(torch.autograd.Function):
    """ReLU DC function with relu_lambda and backprop_mode read at runtime."""

    @staticmethod
    def forward(ctx, input_4: Tensor, relu_lambda: float, backprop_mode: str,
                is_output_layer: bool, cache, layer_name, alpha: float,
                layer_data=None) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name, alpha)
        pos, neg = fb.split_input(input_4)
        z = pos - neg

        out_pos, out_neg = forward_relu(pos, neg, relu_lambda)

        # Check for cached mask in backward-only mode
        use_cached_mask = False
        if fb.should_use_cached_mask():
            cached_mask = fb.get_cached_relu_mask()
            if cached_mask is not None:
                use_cached_mask = True
                ctx.save_for_backward(cached_mask)

        if not use_cached_mask:
            ctx.save_for_backward(z)

        ctx.use_cached_mask = use_cached_mask
        ctx.relu_lambda = relu_lambda
        ctx.backprop_mode = backprop_mode
        ctx._dc_layer_data = layer_data
        return fb.build_output_dc(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None, None, None, None]:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            # ADF override: use variance-based masks instead of point estimates
            # Uses pop_adf_stats() to correctly handle shared modules
            # (e.g. ResNet Bottleneck reuses the same nn.ReLU 3 times).
            adf_mode, adf_lam, probabilistic = get_adf_config()
            ld = getattr(ctx, '_dc_layer_data', None)
            if adf_mode != 'none' and ld is not None:
                adf_stats = ld.pop_adf_stats()
                if adf_stats is not None:
                    adf_mean, adf_var = adf_stats
                    std = torch.sqrt(adf_var.clamp(min=1e-8))
                    # Gate mean: the original pre-activation (capture pass)
                    z_orig = ld.pop_activation()
                    mean = z_orig if z_orig is not None else adf_mean
                    # Confidence bound
                    z_eff = mean + adf_lam * std
                    if probabilistic:
                        mp = torch.special.ndtr(z_eff / std)
                    else:
                        mp = (z_eff >= 0).float()
                    mn = 1.0 - mp
                    return backward_relu(
                        delta_pp, delta_np, delta_pn, delta_nn, mp, mn,
                        ctx.relu_lambda, ctx.backprop_mode)

            # Use cached original pre-activation when available
            # (set by the capture pass when store_activations is active)
            # Uses pop_activation() to correctly handle shared modules
            # (e.g. ResNet Bottleneck reuses the same nn.ReLU 3 times).
            if ld is not None:
                z = ld.pop_activation()
                if z is not None:
                    mp, mn = (z >= 0).float(), (z < 0).float()
                    return backward_relu(
                        delta_pp, delta_np, delta_pn, delta_nn, mp, mn,
                        ctx.relu_lambda, ctx.backprop_mode)

            # Standard path: point-estimate masks from DC z = pos - neg
            saved_tensor, = ctx.saved_tensors

            if ctx.use_cached_mask:
                mp = saved_tensor.float()
                mn = (~saved_tensor).float()
            else:
                z = saved_tensor
                mp, mn = (z >= 0).float(), (z < 0).float()

            return backward_relu(
                delta_pp, delta_np, delta_pn, delta_nn, mp, mn,
                ctx.relu_lambda, ctx.backprop_mode)

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=7)


def _backprop_max(delta_pp, delta_np, delta_pn, delta_nn, mp, mn, backprop_mode):
    """Max mode backward: forward_pos = max{a, b}, forward_neg = b"""
    if backprop_mode == 'dc':
        new_pp = delta_pp * mp
        new_np = delta_np + delta_pp * mn
        new_pn = delta_pn * mp
        new_nn = delta_nn + delta_pn * mn
    elif backprop_mode == 'mask_diff':
        new_pp = delta_pp * (mp - mn)
        new_np = delta_np
        new_pn = delta_pn * (mp - mn)
        new_nn = delta_nn
    elif backprop_mode == 'sum':
        new_pp = delta_pp * mp + delta_np * mn
        new_np = delta_np
        new_pn = delta_pn * mp + delta_nn * mn
        new_nn = delta_nn
    elif backprop_mode in ('stopping', 'standard'):
        # Zero ALL sensitivities at inactive neurons (z < 0)
        new_pp = delta_pp * mp
        new_np = delta_np * mp
        new_pn = delta_pn * mp
        new_nn = delta_nn * mp
    else:
        raise ValueError(f"Unknown backprop_mode: {backprop_mode}")
    return new_pp, new_np, new_pn, new_nn


def _backprop_min(delta_pp, delta_np, delta_pn, delta_nn, mp, mn, backprop_mode):
    """Min mode backward: forward_pos = a, forward_neg = min{a, b}"""
    if backprop_mode == 'dc':
        new_pp = delta_pp + delta_np * mn
        new_np = delta_np * mp
        new_pn = delta_pn + delta_nn * mn
        new_nn = delta_nn * mp
    elif backprop_mode == 'mask_diff':
        new_pp = delta_pp
        new_np = delta_np * (mp - mn)
        new_pn = delta_pn
        new_nn = delta_nn * (mp - mn)
    elif backprop_mode == 'sum':
        new_pp = delta_pp
        new_np = delta_np * mp + delta_pp * mn
        new_pn = delta_pn
        new_nn = delta_nn * mp + delta_pn * mn
    elif backprop_mode in ('stopping', 'standard'):
        # Zero ALL sensitivities at inactive neurons (z < 0)
        new_pp = delta_pp * mp
        new_np = delta_np * mp
        new_pn = delta_pn * mp
        new_nn = delta_nn * mp
    else:
        raise ValueError(f"Unknown backprop_mode: {backprop_mode}")
    return new_pp, new_np, new_pn, new_nn


def dc_forward_relu(m: nn.ReLU, x: Tensor) -> Tensor:
    relu_lambda = getattr(m, DC_RELU_LAMBDA, 1.0)
    backprop_mode = getattr(m, DC_BACKPROP_MODE, 'dc')
    cache, layer_name, alpha = get_cache_info(m)
    layer_data = getattr(m, DC_LAYER_DATA, None)
    return DCReLUFunction.apply(
        x, relu_lambda, backprop_mode,
        getattr(m, DC_IS_OUTPUT_LAYER, False), cache, layer_name, alpha,
        layer_data)


def adf_forward_relu(module: nn.ReLU, input_2: Tensor) -> Tensor:
    """ADF [2*batch] forward for ReLU: moment matching through ReLU nonlinearity."""
    q = input_2.shape[0] // 2
    mean, var = input_2[:q], input_2[q:]

    # Store pre-activation stats on layer_data (detached, for DC backward masks)
    ld = getattr(module, DC_LAYER_DATA, None)
    if ld is not None:
        ld.push_adf_stats(mean.detach(), var.detach())

    std = torch.sqrt(var.clamp(min=1e-8))
    alpha = mean / std

    phi = torch.special.ndtr(alpha)
    pdf = (1.0 / (2.0 * 3.141592653589793) ** 0.5) * torch.exp(-0.5 * alpha * alpha)

    mean_out = mean * phi + std * pdf
    var_out = var * phi
    return torch.cat([mean_out, var_out], dim=0)


def patch_relu(m: nn.ReLU, relu_lambda: float = 1.0, backprop_mode: str = 'dc') -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD):
        return
    setattr(m, DC_ORIGINAL_FORWARD, m.forward)
    setattr(m, DC_ENABLED, True)
    setattr(m, DC_RELU_LAMBDA, relu_lambda)
    setattr(m, DC_BACKPROP_MODE, backprop_mode)
    setattr(m, DC_IS_OUTPUT_LAYER, False)

    def patched(x):
        if getattr(m, DC_ADF_FORWARD, False):
            return adf_forward_relu(m, x)
        elif getattr(m, DC_ENABLED, False):
            return dc_forward_relu(m, x)
        else:
            return getattr(m, DC_ORIGINAL_FORWARD)(x)

    m.forward = patched


unpatch_relu = create_unpatch_function(extra_attrs=(DC_RELU_LAMBDA, DC_BACKPROP_MODE))
