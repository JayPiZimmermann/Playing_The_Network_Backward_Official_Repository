"""DC Decomposition for LayerNorm. Forward/Backward: [4*batch] -> [4*batch]

LayerNorm computes: y = (x - mean) / sqrt(var + eps) * weight + bias
For DC: we compute mean and var on z = pos - neg, cache them for backward.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List

from .base import DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


class DCLayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, normalized_shape: List[int],
                weight: Tensor, bias: Tensor, eps: float,
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:
        
        def compute(ctx, pos, neg, normalized_shape, weight, bias, eps):
            z = pos - neg

            dims = list(range(-len(normalized_shape), 0))
            mean = z.mean(dim=dims, keepdim=True)
            var = z.var(dim=dims, unbiased=False, keepdim=True)

            z_norm = (z - mean) / torch.sqrt(var + eps)

            if weight is not None:
                z_norm = z_norm * weight
            if bias is not None:
                z_norm = z_norm + bias

            out_pos = F.relu(z_norm)
            out_neg = F.relu(-z_norm)

            ctx.save_for_backward(z, mean, var, weight)
            ctx.normalized_shape = normalized_shape
            ctx.eps = eps

            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha,
            recenter=False,
            extra_args=(normalized_shape, weight, bias, eps)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            z, mean, var, weight = ctx.saved_tensors
            normalized_shape = ctx.normalized_shape
            eps = ctx.eps

            dims = list(range(-len(normalized_shape), 0))
            std_inv = 1.0 / torch.sqrt(var + eps)
            z_centered = z - mean

            def layernorm_backward(delta):
                if weight is not None:
                    delta = delta * weight
                delta_mean = delta.mean(dim=dims, keepdim=True)
                delta_z_mean = (delta * z_centered).mean(dim=dims, keepdim=True)
                dz = std_inv * (delta - delta_mean - z_centered * delta_z_mean / (var + eps))
                return dz

            z_norm = (z - mean) * std_inv
            if weight is not None:
                z_norm_scaled = z_norm * weight
            else:
                z_norm_scaled = z_norm

            relu_grad_pos = (z_norm_scaled >= 0).float()
            relu_grad_neg = (-z_norm_scaled > 0).float()

            new_pp = layernorm_backward(delta_pp * relu_grad_pos - delta_np * relu_grad_neg)
            new_np = layernorm_backward(delta_np * relu_grad_pos - delta_pp * relu_grad_neg)
            new_pn = layernorm_backward(delta_pn * relu_grad_pos - delta_nn * relu_grad_neg)
            new_nn = layernorm_backward(delta_nn * relu_grad_pos - delta_pn * relu_grad_neg)

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=8)


def dc_forward_layernorm(module: nn.LayerNorm, x: Tensor) -> Tensor:
    cache, layer_name, alpha = get_cache_info(module)
    return DCLayerNormFunction.apply(
        x, list(module.normalized_shape),
        module.weight, module.bias, module.eps,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


def adf_forward_layernorm(module: nn.LayerNorm, input_2: Tensor) -> Tensor:
    """ADF [2*batch] forward for LayerNorm.

    Two modes controlled by get_adf_propagation():

    'analytic' (default):
        var_out = γ².  LayerNorm normalises input to unit variance, so
        the output variance is determined solely by the learned scale γ.
        This is mathematically correct but discards all input variance
        information — problematic for ViT where LayerNorm precedes every
        attention and MLP block.

    'delta_method':
        var_out = (γ² / (σ² + ε)) · var_in.  Preserves the input variance
        signal by scaling it through the normalisation.  σ² is the per-sample
        variance of the mean activations (used for normalisation).  This is a
        first-order approximation that retains relative variance across
        spatial/feature positions.
    """
    from .base import get_adf_propagation

    q = input_2.shape[0] // 2
    mean, var = input_2[:q], input_2[q:]

    # Mean propagation: standard LayerNorm
    mean_out = F.layer_norm(mean, module.normalized_shape, module.weight, module.bias, module.eps)

    gamma = module.weight
    prop_mode = get_adf_propagation()

    if prop_mode == 'delta_method':
        # Preserve input variance: scale by (γ / σ)²
        dims = list(range(-len(module.normalized_shape), 0))
        input_var = mean.var(dim=dims, unbiased=False, keepdim=True)
        if gamma is not None:
            scale = (gamma ** 2) / (input_var + module.eps)
            scale = scale.clamp(max=1e4)  # prevent extreme amplification
            var_out = (scale * var).clamp(min=0)
        else:
            scale = 1.0 / (input_var + module.eps)
            scale = scale.clamp(max=1e4)
            var_out = (scale * var).clamp(min=0)
    else:
        # Standard: variance reset to γ²
        if gamma is not None:
            var_out = (gamma ** 2).expand_as(mean_out)
        else:
            var_out = torch.ones_like(mean_out)

    # Store the ADF stats on layer_data
    from .base import DC_LAYER_DATA
    ld = getattr(module, DC_LAYER_DATA, None)
    if ld is not None:
        ld.adf_mean = mean_out.detach()
        ld.adf_variance = var_out.detach()

    return torch.cat([mean_out, var_out], dim=0)


patch_layernorm = create_patch_function(dc_forward_layernorm, adf_forward_fn=adf_forward_layernorm)
unpatch_layernorm = create_unpatch_function()
