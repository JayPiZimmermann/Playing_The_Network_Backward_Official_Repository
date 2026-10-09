"""DC Decomposition for Matrix Multiplication. Forward/Backward: [4*batch] -> [4*batch]"""

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


class DCMatMul(nn.Module):
    """
    Matrix multiplication module that can be patched for DC decomposition.
    """

    def __init__(
        self,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        transpose_b: bool = False
    ):
        super().__init__()
        self.transpose_b = transpose_b

        if in_features is not None and out_features is not None:
            self.weight = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight)
            self._has_weight = True
        else:
            self.register_parameter('weight', None)
            self._has_weight = False

        self._operand: Optional[Tensor] = None

    def set_operand(self, B: Tensor):
        self._operand = B

    def forward(self, A: Tensor, B: Optional[Tensor] = None) -> Tensor:
        if self._has_weight:
            B_weight = self.weight.t() if not self.transpose_b else self.weight
            return torch.matmul(A, B_weight)
        else:
            if B is not None:
                operand = B
            elif self._operand is not None:
                operand = self._operand
            else:
                raise RuntimeError("Operand not provided.")

            if self.transpose_b:
                operand = operand.transpose(-2, -1)
            return torch.matmul(A, operand)


class DCMatMulFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_4: Tensor, operand_4: Tensor, transpose_b: bool,
                is_output_layer: bool, cache, layer_name, alpha: float,
                operand_is_dc: bool = False) -> Tensor:

        def compute(ctx, A_pos, A_neg, operand_4, transpose_b):
            from .base import split_input_4
            B_pos, B_neg = split_input_4(operand_4)

            if transpose_b:
                B_pos = B_pos.transpose(-2, -1)
                B_neg = B_neg.transpose(-2, -1)

            out_pos = torch.matmul(A_pos, B_pos) + torch.matmul(A_neg, B_neg)
            out_neg = torch.matmul(A_pos, B_neg) + torch.matmul(A_neg, B_pos)

            ctx.save_for_backward(A_pos, A_neg, B_pos, B_neg)
            ctx.transpose_b = transpose_b
            ctx.operand_is_dc = operand_is_dc

            return out_pos, out_neg

        return ForwardBuilder.run(
            ctx, input_4, compute, is_output_layer, cache, layer_name, alpha,
            extra_args=(operand_4, transpose_b)
        )

    @staticmethod
    def backward(ctx, grad_4: Tensor):
        def compute(ctx, delta_pp, delta_np, delta_pn, delta_nn):
            A_pos, A_neg, B_pos, B_neg = ctx.saved_tensors
            transpose_b = ctx.transpose_b

            if transpose_b:
                B_pos_T = B_pos
                B_neg_T = B_neg
            else:
                B_pos_T = B_pos.transpose(-2, -1)
                B_neg_T = B_neg.transpose(-2, -1)

            new_pp_A = torch.matmul(delta_pp, B_pos_T) + torch.matmul(delta_pn, B_neg_T)
            new_np_A = torch.matmul(delta_pp, B_neg_T) + torch.matmul(delta_pn, B_pos_T)
            new_pn_A = torch.matmul(delta_np, B_pos_T) + torch.matmul(delta_nn, B_neg_T)
            new_nn_A = torch.matmul(delta_np, B_neg_T) + torch.matmul(delta_nn, B_pos_T)

            A_pos_T = A_pos.transpose(-2, -1)
            A_neg_T = A_neg.transpose(-2, -1)

            new_pp_B = torch.matmul(A_pos_T, delta_pp) + torch.matmul(A_neg_T, delta_pn)
            new_np_B = torch.matmul(A_neg_T, delta_pp) + torch.matmul(A_pos_T, delta_pn)
            new_pn_B = torch.matmul(A_pos_T, delta_np) + torch.matmul(A_neg_T, delta_nn)
            new_nn_B = torch.matmul(A_neg_T, delta_np) + torch.matmul(A_pos_T, delta_nn)

            return (new_pp_A, new_np_A, new_pn_A, new_nn_A), (new_pp_B, new_np_B, new_pn_B, new_nn_B)

        # Only align both outputs if operand is DC format (two DC-format inputs)
        # For wrapped weights, only align the data gradient (index 0)
        align_indices = None if ctx.operand_is_dc else (0,)
        return BackwardBuilder.run_multi(ctx, grad_4, compute, num_outputs=2,
                                         num_extra_returns=6, align_indices=align_indices)


def dc_forward_matmul(module: DCMatMul, A: Tensor, B: Optional[Tensor] = None) -> Tensor:
    cache, layer_name, alpha = get_cache_info(module)
    operand_is_dc = False
    if module._has_weight:
        B_weight = module.weight.t() if not module.transpose_b else module.weight
        operand_4 = make_output_4(F.relu(B_weight), F.relu(-B_weight))
    else:
        if B is not None:
            operand = B
        elif module._operand is not None:
            operand = module._operand
        else:
            raise RuntimeError("Operand not provided.")
        # Check if operand is already in DC format (same batch dim as A)
        if operand.shape[0] == A.shape[0]:
            operand_4 = operand
            operand_is_dc = True
        else:
            operand_4 = make_output_4(F.relu(operand), F.relu(-operand))

    return DCMatMulFunction.apply(
        A, operand_4, module.transpose_b,
        getattr(module, DC_IS_OUTPUT_LAYER, False),
        cache, layer_name, alpha, operand_is_dc,
    )


def adf_forward_matmul(module: DCMatMul, A: Tensor, B: Optional[Tensor] = None) -> Tensor:
    """ADF [2*batch] forward for MatMul: product rule for independent Gaussians.

    For independent A ~ N(mA, vA) and B ~ N(mB, vB):
        E[AB]   = mA @ mB
        Var[AB] = mA² @ vB + vA @ mB² + vA @ vB
    """
    q = A.shape[0] // 2
    mA, vA = A[:q], A[q:]

    # Resolve operand B
    if module._has_weight:
        B_w = module.weight.t() if not module.transpose_b else module.weight
        # Weight is deterministic: mB = B_w, vB = 0
        mean_out = torch.matmul(mA, B_w)
        var_out = torch.matmul(vA, B_w ** 2)
        return torch.cat([mean_out, var_out], dim=0)

    if B is not None:
        operand = B
    elif module._operand is not None:
        operand = module._operand
    else:
        raise RuntimeError("Operand not provided.")

    if module.transpose_b:
        operand = operand.transpose(-2, -1)

    if operand.shape[0] == A.shape[0]:
        # Operand is also ADF format
        mB, vB = operand[:q], operand[q:]
        mean_out = torch.matmul(mA, mB)
        var_out = (torch.matmul(mA ** 2, vB)
                   + torch.matmul(vA, mB ** 2)
                   + torch.matmul(vA, vB))
    else:
        # Operand is deterministic (vB = 0)
        mB = operand
        mean_out = torch.matmul(mA, mB)
        var_out = torch.matmul(vA, mB ** 2)

    return torch.cat([mean_out, var_out], dim=0)


patch_dcmatmul = create_patch_function(dc_forward_matmul, adf_forward_fn=adf_forward_matmul)
unpatch_dcmatmul = create_unpatch_function()
