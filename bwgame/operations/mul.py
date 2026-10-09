"""DC Decomposition for Element-wise Multiplication. Forward/Backward: [4*batch] -> [4*batch]"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional

from .base import make_output_4, DC_IS_OUTPUT_LAYER
from .patch_builder import (
    ForwardBuilder, BackwardBuilder, get_cache_info,
    create_patch_function, create_unpatch_function,
)


class DCMul(nn.Module):
    """
    Element-wise multiplication module that can be patched for DC decomposition.
    """

    def __init__(self, scalar: Optional[float] = None):
        super().__init__()
        self.scalar = scalar

    def forward(self, A: Tensor, B: Optional[Tensor] = None) -> Tensor:
        if self.scalar is not None:
            return A * self.scalar
        elif B is not None:
            return A * B
        else:
            raise ValueError("Either provide B tensor or set scalar in __init__")


class DCMulFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, operand_4: Tensor,
                is_output_layer: bool, cache, layer_name, alpha: float,
                operand_is_dc: bool = False) -> Tensor:

        def compute(ctx, A_pos, A_neg, operand_4):
            from .base import split_input_4
            B_pos, B_neg = split_input_4(operand_4)

            out_pos = A_pos * B_pos + A_neg * B_neg
            out_neg = A_pos * B_neg + A_neg * B_pos

            ctx.save_for_backward(A_pos, A_neg, B_pos, B_neg)
            ctx.operand_is_dc = operand_is_dc

            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha,
            extra_args=(operand_4,)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            A_pos, A_neg, B_pos, B_neg = ctx.saved_tensors

            new_pp_A = delta_pp * B_pos + delta_pn * B_neg
            new_np_A = delta_pp * B_neg + delta_pn * B_pos
            new_pn_A = delta_np * B_pos + delta_nn * B_neg
            new_nn_A = delta_np * B_neg + delta_nn * B_pos

            new_pp_B = delta_pp * A_pos + delta_pn * A_neg
            new_np_B = delta_pp * A_neg + delta_pn * A_pos
            new_pn_B = delta_np * A_pos + delta_nn * A_neg
            new_nn_B = delta_np * A_neg + delta_nn * A_pos

            return (new_pp_A, new_np_A, new_pn_A, new_nn_A), (new_pp_B, new_np_B, new_pn_B, new_nn_B)

        # Only align both outputs if operand is DC format (two DC-format inputs)
        # For wrapped weights/tensors, only align the data gradient (index 0)
        align_indices = None if ctx.operand_is_dc else (0,)
        return BackwardBuilder.run_multi(ctx, grad_4, compute, num_outputs=2,
                                         num_extra_returns=5, align_indices=align_indices)


class DCScalarMulFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, scalar: float,
                is_output_layer: bool, cache, layer_name, alpha: float) -> Tensor:
        
        def compute(ctx, A_pos, A_neg, scalar):
            if scalar >= 0:
                out_pos = A_pos * scalar
                out_neg = A_neg * scalar
            else:
                out_pos = A_neg * abs(scalar)
                out_neg = A_pos * abs(scalar)

            ctx.scalar = scalar

            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha,
            extra_args=(scalar,)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            scalar = ctx.scalar
            if scalar >= 0:
                new_pp = delta_pp * scalar
                new_np = delta_np * scalar
                new_pn = delta_pn * scalar
                new_nn = delta_nn * scalar
            else:
                abs_scalar = abs(scalar)
                new_pp = delta_pn * abs_scalar
                new_np = delta_nn * abs_scalar
                new_pn = delta_pp * abs_scalar
                new_nn = delta_np * abs_scalar

            return new_pp, new_np, new_pn, new_nn

        return BackwardBuilder.run(ctx, grad_4, compute, num_extra_returns=5)


def dc_forward_mul(module: DCMul, A: Tensor, B: Optional[Tensor] = None) -> Tensor:
    """DC forward for element-wise multiplication."""
    cache, layer_name, alpha = get_cache_info(module)
    if module.scalar is not None:
        return DCScalarMulFunction.apply(
            A, module.scalar,
            getattr(module, DC_IS_OUTPUT_LAYER, False),
            cache, layer_name, alpha,
        )
    elif B is not None:
        # Check if B is already in DC format (same batch size as A)
        # If so, use it directly. Otherwise, wrap it.
        operand_is_dc = False
        if B.shape[0] == A.shape[0]:
            # B is already in DC format (4*batch)
            operand_4 = B
            operand_is_dc = True
        else:
            # B is a regular tensor, wrap it in DC format
            operand_4 = make_output_4(F.relu(B), F.relu(-B))
        return DCMulFunction.apply(
            A, operand_4,
            getattr(module, DC_IS_OUTPUT_LAYER, False),
            cache, layer_name, alpha, operand_is_dc,
        )
    else:
        raise ValueError("Either provide B tensor or set scalar in __init__")


def adf_forward_mul(module: DCMul, A: Tensor, B: Optional[Tensor] = None) -> Tensor:
    """ADF [2*batch] forward for element-wise Mul.

    Scalar case: Var[c*X] = c^2 * Var[X].
    Two-tensor case (independent): Var[X*Y] = mX^2*vY + vX*mY^2 + vX*vY.
    """
    q = A.shape[0] // 2
    mA, vA = A[:q], A[q:]

    if module.scalar is not None:
        mean_out = mA * module.scalar
        var_out = vA * (module.scalar ** 2)
        return torch.cat([mean_out, var_out], dim=0)

    if B is None:
        raise ValueError("Either provide B tensor or set scalar in __init__")

    if B.shape[0] == A.shape[0]:
        # Operand is also ADF format
        mB, vB = B[:q], B[q:]
        mean_out = mA * mB
        var_out = mA ** 2 * vB + vA * mB ** 2 + vA * vB
    else:
        # Operand is deterministic (vB = 0)
        mB = B
        mean_out = mA * mB
        var_out = vA * mB ** 2

    return torch.cat([mean_out, var_out], dim=0)


patch_dcmul = create_patch_function(dc_forward_mul, adf_forward_fn=adf_forward_mul)
unpatch_dcmul = create_unpatch_function()
