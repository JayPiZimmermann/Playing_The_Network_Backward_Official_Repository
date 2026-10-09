"""DC Decomposition for Conv1d and Conv2d layers. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .base import DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


# =============================================================================
# Output padding helpers
# =============================================================================

def _compute_output_padding_2d(input_shape, output_shape, weight_shape, stride, padding, dilation=1):
    ih, iw = input_shape[2], input_shape[3]
    oh, ow = output_shape[2], output_shape[3]
    kh, kw = weight_shape[2], weight_shape[3]
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    ch = (oh - 1) * stride[0] - 2 * padding[0] + dilation[0] * (kh - 1) + 1
    cw = (ow - 1) * stride[1] - 2 * padding[1] + dilation[1] * (kw - 1) + 1
    return (max(0, ih - ch), max(0, iw - cw))


def _compute_output_padding_1d(input_shape, output_shape, weight_shape, stride, padding, dilation=1):
    l_in = input_shape[2]
    l_out = output_shape[2]
    k = weight_shape[2]
    s = stride if isinstance(stride, int) else stride[0]
    p = padding if isinstance(padding, int) else padding[0]
    d = dilation if isinstance(dilation, int) else dilation[0]
    c_l = (l_out - 1) * s - 2 * p + d * (k - 1) + 1
    return max(0, l_in - c_l)


# =============================================================================
# Conv1d
# =============================================================================

class DCConv1dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                stride, padding, dilation, groups,
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:

        def compute(ctx, pos, neg, weight, bias, stride, padding, dilation, groups):
            weight_pos = F.relu(weight)
            weight_neg = F.relu(-weight)

            out_pos = F.conv1d(pos, weight_pos, None, stride, padding, dilation, groups)
            out_pos = out_pos + F.conv1d(neg, weight_neg, None, stride, padding, dilation, groups)
            out_neg = F.conv1d(pos, weight_neg, None, stride, padding, dilation, groups)
            out_neg = out_neg + F.conv1d(neg, weight_pos, None, stride, padding, dilation, groups)

            if bias is not None:
                out_pos = out_pos + F.relu(bias).view(1, -1, 1)
                out_neg = out_neg + F.relu(-bias).view(1, -1, 1)

            ctx.save_for_backward(weight)
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
            ctx.input_shape = pos.shape

            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha,
            extra_args=(weight, bias, stride, padding, dilation, groups)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            weight, = ctx.saved_tensors
            weight_pos = F.relu(weight)
            weight_neg = F.relu(-weight)

            out_pad = _compute_output_padding_1d(
                ctx.input_shape, delta_pp.shape, weight_pos.shape,
                ctx.stride, ctx.padding, ctx.dilation)

            def conv_t(d, w):
                return F.conv_transpose1d(d, w, None, ctx.stride, ctx.padding, out_pad, ctx.groups, ctx.dilation)

            new_pp = conv_t(delta_pp, weight_pos) + conv_t(delta_np, weight_neg)
            new_np = conv_t(delta_pp, weight_neg) + conv_t(delta_np, weight_pos)
            new_pn = conv_t(delta_pn, weight_pos) + conv_t(delta_nn, weight_neg)
            new_nn = conv_t(delta_pn, weight_neg) + conv_t(delta_nn, weight_pos)

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=10)


def dc_forward_conv1d(m: nn.Conv1d, x: Tensor) -> Tensor:
    cache, layer_name, alpha = get_cache_info(m)
    return DCConv1dFunction.apply(
        x, m.weight, m.bias,
        m.stride[0], m.padding[0], m.dilation[0], m.groups,
        getattr(m, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


def adf_forward_conv1d(module: nn.Conv1d, input_2: Tensor) -> Tensor:
    """ADF [2*batch] forward for Conv1d."""
    q = input_2.shape[0] // 2
    mean, var = input_2[:q], input_2[q:]
    W = module.weight
    s, p, d, g = module.stride[0], module.padding[0], module.dilation[0], module.groups
    mean_out = F.conv1d(mean, W, module.bias, s, p, d, g)
    var_out = F.conv1d(var, W ** 2, None, s, p, d, g)

    # Store the ADF stats on layer_data
    from .base import DC_LAYER_DATA
    ld = getattr(module, DC_LAYER_DATA, None)
    if ld is not None:
        ld.adf_mean = mean_out.detach()
        ld.adf_variance = var_out.detach()

    return torch.cat([mean_out, var_out], dim=0)


patch_conv1d = create_patch_function(dc_forward_conv1d, adf_forward_fn=adf_forward_conv1d)
unpatch_conv1d = create_unpatch_function()


# =============================================================================
# Conv2d
# =============================================================================

class DCConv2dFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, weight: Tensor, bias: Optional[Tensor],
                stride, padding, dilation, groups,
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:

        def compute(ctx, pos, neg, weight, bias, stride, padding, dilation, groups):
            weight_pos = F.relu(weight)
            weight_neg = F.relu(-weight)

            out_pos = F.conv2d(pos, weight_pos, None, stride, padding, dilation, groups)
            out_pos = out_pos + F.conv2d(neg, weight_neg, None, stride, padding, dilation, groups)
            out_neg = F.conv2d(pos, weight_neg, None, stride, padding, dilation, groups)
            out_neg = out_neg + F.conv2d(neg, weight_pos, None, stride, padding, dilation, groups)

            if bias is not None:
                out_pos = out_pos + F.relu(bias).view(1, -1, 1, 1)
                out_neg = out_neg + F.relu(-bias).view(1, -1, 1, 1)

            ctx.save_for_backward(weight)
            ctx.stride, ctx.padding, ctx.dilation, ctx.groups = stride, padding, dilation, groups
            ctx.input_shape = pos.shape

            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha,
            extra_args=(weight, bias, stride, padding, dilation, groups)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            weight, = ctx.saved_tensors
            weight_pos = F.relu(weight)
            weight_neg = F.relu(-weight)

            out_pad = _compute_output_padding_2d(
                ctx.input_shape, delta_pp.shape, weight_pos.shape,
                ctx.stride, ctx.padding, ctx.dilation)

            def conv_t(d, w):
                return F.conv_transpose2d(d, w, None, ctx.stride, ctx.padding, out_pad, ctx.groups, ctx.dilation)

            new_pp = conv_t(delta_pp, weight_pos) + conv_t(delta_np, weight_neg)
            new_np = conv_t(delta_pp, weight_neg) + conv_t(delta_np, weight_pos)
            new_pn = conv_t(delta_pn, weight_pos) + conv_t(delta_nn, weight_neg)
            new_nn = conv_t(delta_pn, weight_neg) + conv_t(delta_nn, weight_pos)

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=10)


def dc_forward_conv2d(m: nn.Conv2d, x: Tensor) -> Tensor:
    cache, layer_name, alpha = get_cache_info(m)
    return DCConv2dFunction.apply(
        x, m.weight, m.bias,
        m.stride, m.padding, m.dilation, m.groups,
        getattr(m, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha,
    )


def adf_forward_conv2d(module: nn.Conv2d, input_2: Tensor) -> Tensor:
    """ADF [2*batch] forward for Conv2d."""
    q = input_2.shape[0] // 2
    mean, var = input_2[:q], input_2[q:]
    W = module.weight
    s, p, d, g = module.stride, module.padding, module.dilation, module.groups
    mean_out = F.conv2d(mean, W, module.bias, s, p, d, g)
    var_out = F.conv2d(var, W ** 2, None, s, p, d, g)

    # Store the ADF stats on layer_data
    from .base import DC_LAYER_DATA
    ld = getattr(module, DC_LAYER_DATA, None)
    if ld is not None:
        ld.adf_mean = mean_out.detach()
        ld.adf_variance = var_out.detach()

    return torch.cat([mean_out, var_out], dim=0)


patch_conv2d = create_patch_function(dc_forward_conv2d, adf_forward_fn=adf_forward_conv2d)
unpatch_conv2d = create_unpatch_function()
