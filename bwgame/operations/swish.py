"""DC Decomposition for SiLU/Swish. Forward/Backward: [4*batch] -> [4*batch]

Via the noisy gate interpretation with logistic noise (see the paper appendix):
SiLU(z) = z·σ(z) arises as the expected noisy gate with logistic noise.

The split is controlled by relu_lambda in [0, 1]:
  - lambda=1.0: max mode (a⁺ = a·σ(z) + b·(1−σ(z)), a⁻ = b)
  - lambda=0.0: min mode (a⁺ = a, a⁻ = a·(1−σ(z)) + b·σ(z))
  - 0 < lambda < 1: weighted combination of max and min modes

Only 'dc' and 'stopping' backprop modes are supported.
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


def forward_swish(pos: Tensor, neg: Tensor, relu_lambda: float) -> Tuple[Tensor, Tensor]:
    """DC forward for SiLU/Swish with continuous lambda parameter.

    Args:
        pos: Positive stream tensor (a).
        neg: Negative stream tensor (b).
        relu_lambda: Continuous split parameter in [0, 1].
            1.0 = max mode, 0.0 = min mode.
    """
    z = pos - neg
    sig_z = torch.sigmoid(z)  # σ(z)

    # Max mode: a⁺ = a·σ(z) + b·(1−σ(z)),  a⁻ = b
    max_pos = pos * sig_z + neg * (1.0 - sig_z)
    max_neg = neg

    # Min mode: a⁺ = a,  a⁻ = a·(1−σ(z)) + b·σ(z)
    min_pos = pos
    min_neg = pos * (1.0 - sig_z) + neg * sig_z

    if relu_lambda == 1.0:
        return max_pos, max_neg
    elif relu_lambda == 0.0:
        return min_pos, min_neg
    else:
        out_pos = relu_lambda * max_pos + (1.0 - relu_lambda) * min_pos
        out_neg = relu_lambda * max_neg + (1.0 - relu_lambda) * min_neg
        return out_pos, out_neg


def forward_swish_stopping(pos: Tensor, neg: Tensor) -> Tuple[Tensor, Tensor]:
    """DC forward for SiLU/Swish in stopping mode: multiply σ(z) to both streams."""
    z = pos - neg
    sig_z = torch.sigmoid(z)
    return sig_z * pos, sig_z * neg


def backward_swish_standard(
    delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor,
    z: Tensor, relu_lambda: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """DC backward for SiLU/Swish (standard mode).

    Jacobian of the forward mapping. Let g = SiLU'(z) = σ(z)·(1 + z·(1−σ(z))):
        A = λ·g + (1−λ)       ∂out_pos/∂pos
        B = λ·(1−g)           ∂out_pos/∂neg
        C = (1−λ)·(1−g)       ∂out_neg/∂pos
        D = λ + (1−λ)·g       ∂out_neg/∂neg
    """
    sig_z = torch.sigmoid(z)
    g = sig_z * (1.0 + z * (1.0 - sig_z))  # SiLU'(z)

    lam = relu_lambda
    one_m_g = 1.0 - g
    A = lam * g + (1.0 - lam)       # ∂out_pos/∂pos
    B = lam * one_m_g                # ∂out_pos/∂neg
    C = (1.0 - lam) * one_m_g       # ∂out_neg/∂pos
    D = lam + (1.0 - lam) * g       # ∂out_neg/∂neg

    new_pp = delta_pp * A + delta_np * C
    new_np = delta_pp * B + delta_np * D
    new_pn = delta_pn * A + delta_nn * C
    new_nn = delta_pn * B + delta_nn * D
    return new_pp, new_np, new_pn, new_nn


def backward_swish_stopping(
    delta_pp: Tensor, delta_np: Tensor, delta_pn: Tensor, delta_nn: Tensor,
    z: Tensor, pos: Tensor, neg: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """DC backward for SiLU/Swish (stopping mode).

    Forward: out_pos = σ(z)·pos,  out_neg = σ(z)·neg  where z = pos − neg.
    Product rule derivatives (logistic density: σ(z)·(1−σ(z))):
        ∂out_pos/∂pos = σ(z) + pos·σ(z)·(1−σ(z))
        ∂out_pos/∂neg = −pos·σ(z)·(1−σ(z))
        ∂out_neg/∂pos = neg·σ(z)·(1−σ(z))
        ∂out_neg/∂neg = σ(z) − neg·σ(z)·(1−σ(z))
    """
    sig_z = torch.sigmoid(z)
    sig_density = sig_z * (1.0 - sig_z)  # logistic density

    pos_density = pos * sig_density
    neg_density = neg * sig_density

    new_pp = delta_pp * (sig_z + pos_density) + delta_np * neg_density
    new_np = delta_pp * (-pos_density) + delta_np * (sig_z - neg_density)
    new_pn = delta_pn * (sig_z + pos_density) + delta_nn * neg_density
    new_nn = delta_pn * (-pos_density) + delta_nn * (sig_z - neg_density)
    return new_pp, new_np, new_pn, new_nn


class DCSwishFunction(torch.autograd.Function):
    """SiLU/Swish DC function with relu_lambda and backprop_mode."""

    @staticmethod
    def forward(ctx, input_4: Tensor, relu_lambda: float, backprop_mode: str,
                is_output_layer: bool, cache, layer_name, alpha: float,
                layer_data=None) -> Tensor:
        fb = ForwardBuilder(ctx, is_output_layer, cache, layer_name, alpha)
        pos, neg = fb.split_input(input_4)

        if backprop_mode in ('stopping', 'standard'):
            out_pos, out_neg = forward_swish_stopping(pos, neg)
            ctx.save_for_backward(pos - neg, pos, neg)
        else:
            out_pos, out_neg = forward_swish(pos, neg, relu_lambda)
            ctx.save_for_backward(pos - neg,)

        ctx.relu_lambda = relu_lambda
        ctx.backprop_mode = backprop_mode
        ctx._dc_layer_data = layer_data
        return fb.build_output_dc(out_pos, out_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tuple[Tensor, None, None, None, None, None, None, None]:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            # ADF override: use variance-based gate probability
            adf_mode, adf_lam, _ = get_adf_config()
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
                    return backward_swish_standard(
                        delta_pp, delta_np, delta_pn, delta_nn,
                        z_eff, ctx.relu_lambda)

            # Use cached original pre-activation when available
            if ld is not None:
                z = ld.pop_activation()
                if z is not None:
                    return backward_swish_standard(
                        delta_pp, delta_np, delta_pn, delta_nn, z, ctx.relu_lambda)

            # Standard path
            if ctx.backprop_mode in ('stopping', 'standard'):
                z, pos, neg = ctx.saved_tensors
                return backward_swish_stopping(
                    delta_pp, delta_np, delta_pn, delta_nn, z, pos, neg)
            else:
                z, = ctx.saved_tensors
                return backward_swish_standard(
                    delta_pp, delta_np, delta_pn, delta_nn, z, ctx.relu_lambda)

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=7)


def dc_forward_swish(m: nn.SiLU, x: Tensor) -> Tensor:
    relu_lambda = getattr(m, DC_RELU_LAMBDA, 1.0)
    backprop_mode = getattr(m, DC_BACKPROP_MODE, 'dc')
    cache, layer_name, alpha = get_cache_info(m)
    layer_data = getattr(m, DC_LAYER_DATA, None)
    return DCSwishFunction.apply(
        x, relu_lambda, backprop_mode,
        getattr(m, DC_IS_OUTPUT_LAYER, False), cache, layer_name, alpha,
        layer_data)


def adf_forward_swish(module: nn.SiLU, input_2: Tensor) -> Tensor:
    """ADF [2*batch] forward for SiLU/Swish: moment matching through SiLU nonlinearity."""
    q = input_2.shape[0] // 2
    mean, var = input_2[:q], input_2[q:]

    # Store pre-activation stats on layer_data (detached, for DC backward masks)
    ld = getattr(module, DC_LAYER_DATA, None)
    if ld is not None:
        ld.push_adf_stats(mean.detach(), var.detach())

    std = torch.sqrt(var.clamp(min=1e-8))
    alpha = mean / std

    sig = torch.sigmoid(alpha)

    mean_out = mean * sig + std * sig * (1.0 - sig)
    var_out = var * sig * sig
    return torch.cat([mean_out, var_out], dim=0)


def patch_swish(m: nn.SiLU, relu_lambda: float = 1.0, backprop_mode: str = 'dc') -> None:
    if hasattr(m, DC_ORIGINAL_FORWARD):
        return
    setattr(m, DC_ORIGINAL_FORWARD, m.forward)
    setattr(m, DC_ENABLED, True)
    setattr(m, DC_RELU_LAMBDA, relu_lambda)
    setattr(m, DC_BACKPROP_MODE, backprop_mode)
    setattr(m, DC_IS_OUTPUT_LAYER, False)

    def patched(x):
        if getattr(m, DC_ADF_FORWARD, False):
            return adf_forward_swish(m, x)
        elif getattr(m, DC_ENABLED, False):
            return dc_forward_swish(m, x)
        else:
            return getattr(m, DC_ORIGINAL_FORWARD)(x)

    m.forward = patched


unpatch_swish = create_unpatch_function(extra_attrs=(DC_RELU_LAMBDA, DC_BACKPROP_MODE))
