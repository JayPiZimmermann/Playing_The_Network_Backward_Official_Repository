"""
DC Decomposition for Addition (residual connections).

When adding two DC-format tensors, magnitude can grow exponentially.
This module provides re-centering to prevent numerical precision loss.
"""

import torch
import torch.nn as nn
from torch import Tensor

from .base import DC_IS_OUTPUT_LAYER
from .patch_builder import create_patch_function, create_unpatch_function, ForwardBuilder, BackwardBuilder, get_cache_info


class Add(nn.Module):
    """
    Simple addition module that can be patched for DC decomposition.

    This module is inserted by functional_replacer.py to replace `+` and
    `torch.add` operations. When patched, it performs DC-aware addition
    with re-centering to prevent magnitude explosion in residual networks.

    Usage (automatic via functional_replacer):
        model = make_dc_compatible(model)  # Replaces + with Add modules
        patch_model(model)  # Patches Add modules for DC mode
    """

    def __init__(self, recenter: bool = True):
        super().__init__()
        self.recenter = recenter

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x + y


class DCAddFunction(torch.autograd.Function):
    """DC-aware addition using the ForwardBuilder pattern.

    Forward adds the pos and neg streams of the two inputs; the backward
    pass sends the full sensitivities to both inputs.
    """

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor, recenter: bool,
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:

        def compute(ctx, x_pos, x_neg, y, recenter):
            from .base import split_input_4
            # Check if y is in DC format (batch dim = 4 * x_pos batch dim)
            if y.shape[0] == x_pos.shape[0] * 4:
                y_pos, y_neg = split_input_4(y)
                ctx._y_is_dc = True
            else:
                # y is a non-DC tensor (parameter/constant)
                # Decompose: relu(y) - relu(-y) = y
                y_pos = torch.relu(y)
                y_neg = torch.relu(-y)
                ctx._y_is_dc = False

            # Add corresponding components
            result_pos = x_pos + y_pos
            result_neg = x_neg + y_neg

            return result_pos, result_neg

        return ForwardBuilder.run(
            ctx, x, compute, is_output_layer, cache, layer_name, alpha,
            use_recenter_dc=recenter, recenter=True,
            extra_args=(y, recenter)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> tuple:
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            grads = (delta_pp, delta_np, delta_pn, delta_nn)
            return grads, grads

        # Only align x gradient (index 0) if y was not in DC format
        align_indices = None if getattr(ctx, '_y_is_dc', True) else (0,)
        return BackwardBuilder.run_multi(ctx, grad_4, compute, num_outputs=2,
                                         num_extra_returns=5, align_indices=align_indices)


def dc_forward_add(module: Add, x: Tensor, y: Tensor) -> Tensor:
    """DC-aware forward for Add module with alignment support."""
    cache, layer_name, alpha = get_cache_info(module)
    return DCAddFunction.apply(
        x, y, module.recenter,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


def adf_forward_add(module: Add, x: Tensor, y: Tensor) -> Tensor:
    """ADF [2*batch] forward for Add: mean and variance add independently."""
    # Handle non-ADF or empty inputs: fall back to standard add
    if x.shape[0] == 0 or y.shape[0] == 0:
        return x + y
    if x.shape[0] != y.shape[0]:
        # One input is ADF [2*B], the other is plain [B] or different size
        bigger, smaller = (x, y) if x.shape[0] > y.shape[0] else (y, x)
        q = bigger.shape[0] // 2
        mb, vb = bigger[:q], bigger[q:]
        if smaller.shape[0] == q:
            # smaller is plain [B] — treat as constant (zero variance)
            mean_out = mb + smaller
            var_out = vb
        else:
            # Mismatched sizes — fall back to standard add
            return x + y
        return torch.cat([mean_out, var_out], dim=0)
    q = x.shape[0] // 2
    mx, vx = x[:q], x[q:]
    my, vy = y[:q], y[q:]
    mean_out = mx + my
    var_out = vx + vy
    return torch.cat([mean_out, var_out], dim=0)


patch_add = create_patch_function(dc_forward_add, adf_forward_fn=adf_forward_add)
unpatch_add = create_unpatch_function()
