"""Hellinger Path Distance for the ADF and Active Routing Games.

This module compares the layered path measures induced by two models on
the common MDP described in the paper.

ADF stores the split sign state throughout the sweep.  The full state
space distinguishes the two stream labels because cross-model Hellinger
comparison is sensitive to whether a spatial edge is same-sign or
cross-sign in each model: collapsing both signs to a single ``|W| /
gamma`` row changes the path law and can create spurious overlap.
Absorption into the global cemetery still uses a single shared absorbing
state.

RG keeps the activation state as the stored sweep state and realizes the
paper's activation-to-linear split explicitly inside each linear module:
surviving activation mass splits uniformly into a positive and a negative
linear branch, each branch routes with its own Gibbs policy based on the
branchwise routed mass, and branch mass with zero normalizer is absorbed
into the cemetery.  The path law does not depend on the LRP weights
(alpha, beta); they enter only the discounted occupation measure.

The Bhattacharyya sub-mass propagates through the entrywise geometric-mean
kernel, and the final disagreement map is
``h^2 = 0.5 * (alpha_A + alpha_B) - beta`` on ordinary input states plus
the separate contribution of the single absorbing cemetery state.

================ Supported PyTorch modules =============================

Linear-style layers (have a backward kernel):
    nn.Linear, nn.Conv2d
Gated activations (cemetery absorption with the gate of the activation):
    nn.ReLU, nn.GELU, nn.SiLU
Identity layers (mass passes through unchanged):
    nn.LeakyReLU, nn.ELU, nn.Tanh, nn.Sigmoid,
    nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm,
    nn.Dropout, nn.Identity, nn.Softmax (detach-logits convention)
Pooling (uniform / argmax redistribution):
    nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.MaxPool2d
Reshape:
    nn.Flatten and the layout modules of ``bwgame.operations.shape_ops``
Residual addition (``bwgame.operations.add.Add``):
    half of the mass to each operand
Attention matmul (DCMatMul):
    treated as a Linear layer with the attention matrix as the weights
    (the "detach logits" interpretation).  For RG the attention product and
    its Value projection form one composite step that keeps the stream
    label of the Value entries (see :func:`_av_backward_merged`).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

try:
    from bwgame.operations.matmul import DCMatMul
    _HAS_DCMATMUL = True
except Exception:
    DCMatMul = None  # type: ignore
    _HAS_DCMATMUL = False

try:
    from bwgame.operations.mul import DCMul
    _HAS_DCMUL = True
except Exception:
    DCMul = None  # type: ignore
    _HAS_DCMUL = False

try:
    from bwgame.operations.add import Add as _DCAdd
    _HAS_DCADD = True
except Exception:
    _DCAdd = None  # type: ignore
    _HAS_DCADD = False

try:
    from bwgame.operations.shape_ops import (
        Reshape as _DCReshape,
        View as _DCView,
        Transpose as _DCTranspose,
        Permute as _DCPermute,
        Squeeze as _DCSqueeze,
        Unsqueeze as _DCUnsqueeze,
    )
    _HAS_DCSHAPE = True
except Exception:
    _DCReshape = _DCView = _DCTranspose = _DCPermute = None  # type: ignore
    _DCSqueeze = _DCUnsqueeze = None  # type: ignore
    _HAS_DCSHAPE = False

try:
    from bwgame.operations.tensor_ops import DCCat as _DCCat
    _HAS_DCCAT = True
except Exception:
    _DCCat = None  # type: ignore
    _HAS_DCCAT = False


_EPS_DENOM = 1e-12
_INV_SQRT2 = 0.7071067811865476  # 1/√2, for Φ(z) = 0.5*(1+erf(z/√2))


# ---------------------------------------------------------------------------
# Module type registry
# ---------------------------------------------------------------------------

_LINEAR_TYPES: Tuple[type, ...] = (nn.Linear,)
_CONV2D_TYPES: Tuple[type, ...] = (nn.Conv2d,)
_IDENTITY_TYPES: Tuple[type, ...] = (
    nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.ELU, nn.Tanh, nn.Sigmoid,
    nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm,
    nn.Dropout, nn.Identity, nn.Softmax,
)

_AVGPOOL2D_TYPES: Tuple[type, ...] = (nn.AvgPool2d,)
_ADAPTIVE_AVGPOOL2D_TYPES: Tuple[type, ...] = (nn.AdaptiveAvgPool2d,)
_MAXPOOL2D_TYPES: Tuple[type, ...] = (nn.MaxPool2d,)
_FLATTEN_TYPES: Tuple[type, ...] = (nn.Flatten,)


def _supported_types() -> Tuple[type, ...]:
    extras: Tuple[type, ...] = ()
    if _HAS_DCMATMUL:
        extras = extras + (DCMatMul,)
    if _HAS_DCMUL:
        extras = extras + (DCMul,)
    if _HAS_DCADD:
        extras = extras + (_DCAdd,)
    if _HAS_DCSHAPE:
        extras = extras + (
            _DCReshape, _DCView, _DCTranspose, _DCPermute,
            _DCSqueeze, _DCUnsqueeze,
        )
    if _HAS_DCCAT:
        extras = extras + (_DCCat,)
    return (
        _LINEAR_TYPES + _CONV2D_TYPES
        + _IDENTITY_TYPES
        + _AVGPOOL2D_TYPES + _ADAPTIVE_AVGPOOL2D_TYPES + _MAXPOOL2D_TYPES
        + _FLATTEN_TYPES
        + extras
    )


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------

class _GameState:
    """Ordinary-state mass for the Hellinger sweep.

    For RG the stored state is the activation-state mass and has shape
    ``(B, *spatial)``.  For ADF the stored state keeps the split stream
    label and has shape ``(B, 2, *spatial)``, where dim 1 is ``(+,-)``.

    The single absorbing cemetery state is tracked separately because it
    is global, does not interact with ordinary kernels, and must not be
    duplicated across residual branches.
    """

    def __init__(
        self,
        mu_A: Tensor,
        mu_B: Tensor,
        beta: Tensor,
        sign_augmented: bool = False,
    ):
        self.mu_A = mu_A
        self.mu_B = mu_B
        self.beta = beta
        self.sign_augmented = bool(sign_augmented)

    @classmethod
    def initial(
        cls,
        output_A: Tensor,
        target: Tensor,
        sign_augmented: bool = False,
    ) -> "_GameState":
        """Build the initial state at the output layer.

        Mass starts concentrated on the target output neuron under both
        models, and beta starts at the same point with full mass.
        """
        B = output_A.size(0)
        out_shape = output_A.shape
        device = output_A.device
        if sign_augmented:
            reg_shape = (B, 2) + out_shape[1:]
            mu_A = torch.zeros(reg_shape, device=device)
            mu_B = torch.zeros(reg_shape, device=device)
            beta = torch.zeros(reg_shape, device=device)
            for b in range(B):
                mu_A[b, 0, target[b]] = 1.0
                mu_B[b, 0, target[b]] = 1.0
                beta[b, 0, target[b]] = 1.0
        else:
            mu_A = torch.zeros(out_shape, device=device)
            mu_B = torch.zeros(out_shape, device=device)
            beta = torch.zeros(out_shape, device=device)
            for b in range(B):
                mu_A[b, target[b]] = 1.0
                mu_B[b, target[b]] = 1.0
                beta[b, target[b]] = 1.0
        return cls(mu_A, mu_B, beta, sign_augmented=sign_augmented)

    def ordinary_shape(self) -> Tuple[int, ...]:
        if self.sign_augmented:
            return (self.mu_A.shape[0],) + tuple(self.mu_A.shape[2:])
        return tuple(self.mu_A.shape)

    def ordinary_spatial_shape(self) -> Tuple[int, ...]:
        return self.ordinary_shape()[1:]

    def ordinary_numel_per_batch(self) -> int:
        if self.sign_augmented:
            return int(self.mu_A[0, 0].numel())
        return int(self.mu_A[0].numel())

    def _sum_sign(self, x: Tensor) -> Tensor:
        return x.sum(dim=1) if self.sign_augmented else x

    def alpha_A(self) -> Tensor:
        return self._sum_sign(self.mu_A)

    def alpha_B(self) -> Tensor:
        return self._sum_sign(self.mu_B)

    def beta_ordinary(self) -> Tensor:
        return self._sum_sign(self.beta)

    def per_pixel_h2(self) -> Tensor:
        """Per-pixel disagreement at the current layer.

        Returns the ordinary-state contribution only, with shape
        ``(B, *spatial_dims)``.
        """
        return (
            0.5 * (self.alpha_A() + self.alpha_B())
            - self.beta_ordinary()
        ).clamp(min=0)

    def per_pixel_h2_marginal(self) -> Tensor:
        """Marginal-only per-pixel surrogate based on ordinary occupations."""
        alpha_A = self.alpha_A()
        alpha_B = self.alpha_B()
        return (
            0.5 * (alpha_A + alpha_B)
            - torch.sqrt((alpha_A * alpha_B).clamp(min=0))
        ).clamp(min=0)

    def per_pixel_h2_fraction(self) -> Tensor:
        """Local disagreement fraction ``h²(s) / (0.5 * (α_A(s) + α_B(s)))``."""
        local_mass = 0.5 * (self.alpha_A() + self.alpha_B())
        return _conditional_div(self.per_pixel_h2(), local_mass).clamp(min=0, max=1)

    def scale(self, factor: float) -> "_GameState":
        """Return a new state with all sub-masses multiplied by ``factor``."""
        return _GameState(
            self.mu_A * factor,
            self.mu_B * factor,
            self.beta * factor,
            sign_augmented=self.sign_augmented,
        )

    def add(self, other: "_GameState") -> "_GameState":
        """Element-wise sum of two states (assumes matching shapes)."""
        if self.sign_augmented != other.sign_augmented:
            raise ValueError("Cannot add Hellinger states with different sign layouts")
        return _GameState(
            self.mu_A + other.mu_A,
            self.mu_B + other.mu_B,
            self.beta + other.beta,
            sign_augmented=self.sign_augmented,
        )

    def reshape_to(self, target_shape: Tuple[int, ...]) -> "_GameState":
        """Reshape to ``target_shape``."""
        if self.sign_augmented:
            reg_shape = (target_shape[0], 2) + tuple(target_shape[1:])
        else:
            reg_shape = target_shape
        return _GameState(
            self.mu_A.reshape(reg_shape),
            self.mu_B.reshape(reg_shape),
            self.beta.reshape(reg_shape),
            sign_augmented=self.sign_augmented,
        )


class _CemeteryState:
    """Single absorbing cemetery state shared across the whole sweep."""

    def __init__(self, mu_A: Tensor, mu_B: Tensor, beta: Tensor):
        self.mu_A = mu_A
        self.mu_B = mu_B
        self.beta = beta

    @classmethod
    def zeros(cls, batch_size: int, device: torch.device) -> "_CemeteryState":
        shape = (batch_size, 1)
        zeros = torch.zeros(shape, device=device)
        return cls(zeros.clone(), zeros.clone(), zeros.clone())

    def h2(self) -> Tensor:
        return (0.5 * (self.mu_A + self.mu_B) - self.beta).flatten(start_dim=1).sum(dim=1).clamp(min=0)


# ---------------------------------------------------------------------------
# Forward hook manager: caches I/O at every supported leaf module
# ---------------------------------------------------------------------------

class _CallRecord:
    """One forward call of a hooked module.

    Records ALL inputs (so binary ops like Add capture both operands) plus
    the produced output, the module reference, and its type.  Each call
    is a distinct record so that shared modules invoked multiple times
    (e.g. ``layer.relu`` called three times in a Bottleneck) are tracked
    independently.
    """

    __slots__ = ("name", "module", "mod_type", "inputs", "output")

    def __init__(self, name, module, mod_type, inputs, output):
        self.name = name
        self.module = module
        self.mod_type = mod_type
        self.inputs = inputs   # tuple of tensors (in call order)
        self.output = output   # tensor


class _HellingerActivationCache:
    """Cache module I/O during a forward pass.

    Records every CALL (not every module) so that:
        - shared modules called multiple times produce distinct entries
        - Add modules expose both operands as ``inputs``
        - tensor identities are stable across calls (output of producer
          is the same Python object as input of consumer), enabling
          identity-keyed backward dispatch.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.calls: List[_CallRecord] = []
        self._hooks: List[Any] = []
        self._inplace_relus: List[nn.ReLU] = []

    def _is_supported_leaf(self, module: nn.Module) -> bool:
        if list(module.children()):
            return False
        return isinstance(module, _supported_types())

    def install(self) -> "_HellingerActivationCache":
        self._inplace_relus = []
        for m in self.model.modules():
            if isinstance(m, nn.ReLU) and getattr(m, "inplace", False):
                m.inplace = False
                self._inplace_relus.append(m)

        for name, module in self.model.named_modules():
            if not self._is_supported_leaf(module):
                continue
            mod_type = type(module)

            def _make_hook(mod_name: str = name, mod: nn.Module = module, mt: type = mod_type):
                def _hook(_m: nn.Module, inp: Tuple[Any, ...], out: Any):
                    if isinstance(inp, tuple):
                        inputs = tuple(t for t in inp)
                    else:
                        inputs = (inp,)
                    # For DCMatMul, also include the deferred operand if any.
                    if _HAS_DCMATMUL and isinstance(mod, DCMatMul) and len(inputs) < 2:
                        if getattr(mod, "_operand", None) is not None:
                            inputs = inputs + (mod._operand,)
                    self.calls.append(_CallRecord(
                        name=mod_name, module=mod, mod_type=mt,
                        inputs=inputs, output=out,
                    ))
                return _hook

            h = module.register_forward_hook(_make_hook())
            self._hooks.append(h)
        return self

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        for m in self._inplace_relus:
            m.inplace = True
        self._inplace_relus.clear()

    def clear(self):
        self.calls.clear()


# ---------------------------------------------------------------------------
# Helper: stabilized division
# ---------------------------------------------------------------------------

def _safe_div(num: Tensor, den: Tensor) -> Tensor:
    return num / den.clamp(min=_EPS_DENOM)


def _conv2d_output_padding(
    in_shape: Tuple[int, int],
    out_shape: Tuple[int, int],
    mod: nn.Conv2d,
) -> Tuple[int, int]:
    """Compute output_padding for conv_transpose2d so its output matches
    the input shape of the forward Conv2d.

    PyTorch's conv_transpose2d defaults to output_padding=0, which yields
    output size  (H_out - 1) * stride - 2 * pad + dilation * (kH - 1) + 1.
    When the original forward conv used stride > 1, this can be smaller
    than H_in by up to (stride - 1).  We add the difference back via
    output_padding.
    """
    H_in, W_in = in_shape
    H_out, W_out = out_shape
    sh, sw = (mod.stride if isinstance(mod.stride, tuple) else (mod.stride, mod.stride))
    ph, pw = (mod.padding if isinstance(mod.padding, tuple) else (mod.padding, mod.padding))
    dh, dw = (mod.dilation if isinstance(mod.dilation, tuple) else (mod.dilation, mod.dilation))
    kh, kw = (mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size, mod.kernel_size))
    base_h = (H_out - 1) * sh - 2 * ph + dh * (kh - 1) + 1
    base_w = (W_out - 1) * sw - 2 * pw + dw * (kw - 1) + 1
    op_h = max(0, H_in - base_h)
    op_w = max(0, W_in - base_w)
    return (op_h, op_w)


# ---------------------------------------------------------------------------
# ADF GAME -- backward kernels (weights-only)
# ---------------------------------------------------------------------------

def _adf_linear_propagate(
    mu: Tensor,  # (B, 2, n_out)
    W: Tensor,   # (n_out, n_in)
) -> Tensor:
    """Propagate one-model sign-split mass backward through Linear."""
    abs_W = W.abs()
    gamma = abs_W.sum(dim=-1)
    W_pos = F.relu(W)
    W_neg = F.relu(-W)
    mu_pos = mu[:, 0, :]
    mu_neg = mu[:, 1, :]
    inv_gamma = _safe_div(torch.ones_like(gamma), gamma)
    mu_pos_n = mu_pos * inv_gamma
    mu_neg_n = mu_neg * inv_gamma
    mu_in_pos = mu_pos_n @ W_pos + mu_neg_n @ W_neg
    mu_in_neg = mu_pos_n @ W_neg + mu_neg_n @ W_pos
    return torch.stack([mu_in_pos, mu_in_neg], dim=1)


def _adf_linear_propagate_beta(
    beta: Tensor,    # (B, 2, n_out)
    W_A: Tensor,
    W_B: Tensor,
) -> Tensor:
    """Propagate the Bhattacharyya sub-mass beta backward through Linear under ADF."""
    abs_WA = W_A.abs()
    abs_WB = W_B.abs()
    gamma_A = abs_WA.sum(dim=-1)
    gamma_B = abs_WB.sum(dim=-1)
    sqrt_gamma = torch.sqrt((gamma_A * gamma_B).clamp(min=0))
    WA_pos = F.relu(W_A)
    WA_neg = F.relu(-W_A)
    WB_pos = F.relu(W_B)
    WB_neg = F.relu(-W_B)
    sqrt_W_pos = torch.sqrt((WA_pos * WB_pos).clamp(min=0))
    sqrt_W_neg = torch.sqrt((WA_neg * WB_neg).clamp(min=0))
    inv_sqrt_gamma = _safe_div(torch.ones_like(sqrt_gamma), sqrt_gamma)
    beta_pos = beta[:, 0, :] * inv_sqrt_gamma
    beta_neg = beta[:, 1, :] * inv_sqrt_gamma
    beta_in_pos = beta_pos @ sqrt_W_pos + beta_neg @ sqrt_W_neg
    beta_in_neg = beta_pos @ sqrt_W_neg + beta_neg @ sqrt_W_pos
    return torch.stack([beta_in_pos, beta_in_neg], dim=1)


# ---------------------------------------------------------------------------
# Cemetery.  In the layered MDP the cemetery is a single absorbing state
# shared across all layers: its transition kernel is the identity
# ``† -> †``, so absorbed mass is only accumulated, never propagated.
# ---------------------------------------------------------------------------

def _sum_to_cemetery(x: Tensor) -> Tensor:
    return x.flatten(start_dim=1).sum(dim=1, keepdim=True)


def _absorb_branch_dead(
    mu_A: Tensor,
    mu_B: Tensor,
    beta: Tensor,
    dead_A: Tensor,
    dead_B: Tensor,
    cemetery: _CemeteryState,
) -> Tuple[Tensor, Tensor, Tensor]:
    cemetery.mu_A = cemetery.mu_A + _sum_to_cemetery(torch.where(dead_A, mu_A, torch.zeros_like(mu_A)))
    cemetery.mu_B = cemetery.mu_B + _sum_to_cemetery(torch.where(dead_B, mu_B, torch.zeros_like(mu_B)))
    dead_both = dead_A & dead_B
    dead_either = dead_A | dead_B
    cemetery.beta = cemetery.beta + _sum_to_cemetery(torch.where(dead_both, beta, torch.zeros_like(beta)))
    return (
        torch.where(dead_A, torch.zeros_like(mu_A), mu_A),
        torch.where(dead_B, torch.zeros_like(mu_B), mu_B),
        torch.where(dead_either, torch.zeros_like(beta), beta),
    )


def _activation_gate(z: Tensor, mod: nn.Module) -> Tensor:
    """Return the per-neuron gating probability for the game's cemetery.

    ReLU:  hard gate, ``P(continue) = 1{z > 0}``.
    GELU:  ``P(continue) = Φ(z)`` (standard normal CDF), since
           ``GELU(z) = z · Φ(z)`` is the expected value of a stochastic
           ReLU with Gaussian mask — the probability the mask lets the
           neuron through IS ``Φ(z)``.
    SiLU:  ``P(continue) = σ(z)`` (logistic sigmoid), by the same
           argument for ``SiLU(z) = z · σ(z)``.
    """
    if isinstance(mod, nn.ReLU):
        return (z > 0).to(z.dtype)
    if isinstance(mod, nn.GELU):
        return 0.5 * (1.0 + torch.erf(z * _INV_SQRT2))
    if isinstance(mod, nn.SiLU):
        return torch.sigmoid(z)
    raise TypeError(f"No gating model for {type(mod).__name__}")


_GATED_ACTIVATION_TYPES: Tuple[type, ...] = (nn.ReLU, nn.GELU, nn.SiLU)


def _backward_activation_cemetery(
    mod_A: nn.Module,
    mod_B: nn.Module,
    z_A: Tensor,
    z_B: Tensor,
    state: _GameState,
    cemetery: _CemeteryState,
) -> _GameState:
    """Cemetery absorption at a gated activation (ReLU/GELU/SiLU).

    Each model independently gates with probability ``g_M(z)`` (see
    :func:`_activation_gate`).  The Bhattacharyya chain rule gives:

    * ``mu_M_continue   = mu_M · g_M``
    * ``mu_M_cemetery   += mu_M · (1 − g_M)``
    * ``beta_continue   = beta · √(g_A · g_B)``
    * ``beta_cemetery   += beta · √((1−g_A) · (1−g_B))``

    For ReLU the gate is hard (``g ∈ {0, 1}``); for GELU/SiLU the soft
    gate keeps part of the mass on the ordinary state.
    """
    gate_A = _activation_gate(z_A, mod_A)
    gate_B = _activation_gate(z_B, mod_B)

    if state.sign_augmented:
        gate_A = gate_A.unsqueeze(1).expand_as(state.mu_A)
        gate_B = gate_B.unsqueeze(1).expand_as(state.mu_B)

    # --- mu: each model gates independently ---
    cemetery.mu_A = cemetery.mu_A + _sum_to_cemetery(state.mu_A * (1.0 - gate_A))
    cemetery.mu_B = cemetery.mu_B + _sum_to_cemetery(state.mu_B * (1.0 - gate_B))
    new_mu_A = state.mu_A * gate_A
    new_mu_B = state.mu_B * gate_B

    # --- beta: geometric-mean kernel ---
    sqrt_gate_cont = torch.sqrt((gate_A * gate_B).clamp(min=0))
    sqrt_gate_cem = torch.sqrt(((1.0 - gate_A) * (1.0 - gate_B)).clamp(min=0))

    cemetery.beta = cemetery.beta + _sum_to_cemetery(state.beta * sqrt_gate_cem)
    new_beta = state.beta * sqrt_gate_cont

    return _GameState(new_mu_A, new_mu_B, new_beta,
                      sign_augmented=state.sign_augmented)


def _adf_backward_linear(
    module_A: nn.Linear,
    module_B: nn.Linear,
    state: _GameState,
) -> _GameState:
    W_A = module_A.weight.detach()
    W_B = module_B.weight.detach()
    new_mu_A = _adf_linear_propagate(state.mu_A, W_A)
    new_mu_B = _adf_linear_propagate(state.mu_B, W_B)
    new_beta = _adf_linear_propagate_beta(state.beta, W_A, W_B)
    return _GameState(
        new_mu_A,
        new_mu_B,
        new_beta,
        sign_augmented=state.sign_augmented,
    )


def _adf_backward_conv2d(
    module_A: nn.Conv2d,
    module_B: nn.Conv2d,
    in_A: Tensor,
    in_B: Tensor,
    state: _GameState,
) -> _GameState:
    """ADF backward through Conv2d using sign-split kernels.

    Per-position gamma:  gamma_M(c_out, h', w') = sum_{c_in, kh, kw}
        |W_M[c_out, c_in, kh, kw]| * 1{(h'-pad+kh, w'-pad+kw) is in input}
    Equivalently: gamma_M = conv(ones_input, |W_M|).

    With this normalization, the kernel T_M((j, h', w') -> (i, h, w)) =
    |W_M| / gamma_M(j, h', w') sums to exactly 1 over valid input positions
    for every output position, so mass is conserved through the backward
    sweep with no padding leakage.
    """
    W_A = module_A.weight.detach()
    W_B = module_B.weight.detach()
    abs_WA = W_A.abs()
    abs_WB = W_B.abs()

    in_shape_A = (in_A.size(-2), in_A.size(-1))
    in_shape_B = (in_B.size(-2), in_B.size(-1))
    out_shape = (state.mu_A.size(-2), state.mu_A.size(-1))
    op_A = _conv2d_output_padding(in_shape_A, out_shape, module_A)
    op_B = _conv2d_output_padding(in_shape_B, out_shape, module_B)

    def _conv(x: Tensor, w: Tensor, mod: nn.Conv2d) -> Tensor:
        return F.conv2d(x, w, bias=None, stride=mod.stride,
                        padding=mod.padding, dilation=mod.dilation,
                        groups=mod.groups)

    def _conv_T(x: Tensor, w: Tensor, mod: nn.Conv2d, op: Tuple[int, int]) -> Tensor:
        return F.conv_transpose2d(x, w, bias=None, stride=mod.stride,
                                  padding=mod.padding, output_padding=op,
                                  dilation=mod.dilation, groups=mod.groups)

    # Per-output-position gamma via conv(ones_input, |W|).  Use the actual
    # cached input shape to get the correct receptive-field accounting.
    ones_A = torch.ones_like(in_A)
    ones_B = torch.ones_like(in_B)
    gamma_A_map = _conv(ones_A, abs_WA, module_A)  # (B, C_out, H_out, W_out)
    gamma_B_map = _conv(ones_B, abs_WB, module_B)
    # Crop in case conv produced a slightly larger spatial extent
    if gamma_A_map.shape[-2:] != state.mu_A.shape[-2:]:
        gamma_A_map = gamma_A_map[..., :state.mu_A.size(-2), :state.mu_A.size(-1)]
    if gamma_B_map.shape[-2:] != state.mu_B.shape[-2:]:
        gamma_B_map = gamma_B_map[..., :state.mu_B.size(-2), :state.mu_B.size(-1)]

    WA_pos = F.relu(W_A)
    WA_neg = F.relu(-W_A)
    WB_pos = F.relu(W_B)
    WB_neg = F.relu(-W_B)

    def _propagate_mu(
        mu: Tensor,
        W_pos: Tensor,
        W_neg: Tensor,
        gamma_map: Tensor,
        a_ref: Tensor,
        mod: nn.Conv2d,
        op: Tuple[int, int],
    ) -> Tensor:
        inv_gamma = 1.0 / gamma_map.clamp(min=_EPS_DENOM)
        mu_pos_n = mu[:, 0] * inv_gamma
        mu_neg_n = mu[:, 1] * inv_gamma
        target_h, target_w = a_ref.size(-2), a_ref.size(-1)

        def _crop(x: Tensor) -> Tensor:
            return x[..., :target_h, :target_w]

        out_pos = _crop(_conv_T(mu_pos_n, W_pos, mod, op)) + _crop(_conv_T(mu_neg_n, W_neg, mod, op))
        out_neg = _crop(_conv_T(mu_pos_n, W_neg, mod, op)) + _crop(_conv_T(mu_neg_n, W_pos, mod, op))
        return torch.stack([out_pos, out_neg], dim=1)

    new_mu_A = _propagate_mu(state.mu_A, WA_pos, WA_neg, gamma_A_map, in_A, module_A, op_A)
    new_mu_B = _propagate_mu(state.mu_B, WB_pos, WB_neg, gamma_B_map, in_B, module_B, op_B)

    sqrt_gamma_map = torch.sqrt((gamma_A_map * gamma_B_map).clamp(min=0))
    inv_sqrt_gamma = 1.0 / sqrt_gamma_map.clamp(min=_EPS_DENOM)
    sqrt_W_pos = torch.sqrt((WA_pos * WB_pos).clamp(min=0))
    sqrt_W_neg = torch.sqrt((WA_neg * WB_neg).clamp(min=0))
    beta_pos_n = state.beta[:, 0] * inv_sqrt_gamma
    beta_neg_n = state.beta[:, 1] * inv_sqrt_gamma
    target_h, target_w = in_A.size(-2), in_A.size(-1)

    def _crop_b(x: Tensor) -> Tensor:
        return x[..., :target_h, :target_w]
    new_beta_pos = _crop_b(_conv_T(beta_pos_n, sqrt_W_pos, module_A, op_A)) + _crop_b(
        _conv_T(beta_neg_n, sqrt_W_neg, module_A, op_A),
    )
    new_beta_neg = _crop_b(_conv_T(beta_pos_n, sqrt_W_neg, module_A, op_A)) + _crop_b(
        _conv_T(beta_neg_n, sqrt_W_pos, module_A, op_A),
    )
    new_beta = torch.stack([new_beta_pos, new_beta_neg], dim=1)
    return _GameState(
        new_mu_A,
        new_mu_B,
        new_beta,
        sign_augmented=state.sign_augmented,
    )


# ---------------------------------------------------------------------------
# RG GAME -- backward kernels (Gibbs routing with temperature)
# ---------------------------------------------------------------------------

def _rg_linear_branch_scores(
    a_in: Tensor,
    W: Tensor,
    tau: float,
) -> Tuple[Tensor, Tensor]:
    aW = a_in.unsqueeze(1) * W.unsqueeze(0)
    r_pos = F.relu(aW)
    r_neg = F.relu(-aW)
    inv_tau = 1.0 / max(tau, 1e-3)
    if abs(inv_tau - 1.0) > 1e-9:
        return r_pos.pow(inv_tau), r_neg.pow(inv_tau)
    return r_pos, r_neg


# ---------------------------------------------------------------------------
# AV matmul kernels (attention Value-routing path).
# The QK^T matmul is detached (no Q/K backprop); the AV matmul is treated
# as a Linear layer with the attention matrix A as the (per-head, per-d)
# detached weight matrix and V as the input activation (appendix
# "Attention as fixed-reference KL Value routing").
# ---------------------------------------------------------------------------

def _av_adf_propagate(mu: Tensor, A: Tensor) -> Tensor:
    """ADF backward through Y = A @ V using only the attention matrix.

    The kernel is ``T(q -> k) = |A[q, k]| / sum_k' |A[q, k']|``, applied
    independently for every ``(head, d)`` channel.  Under the sign-split
    ADF state space, the attention weights are non-negative, so the
    stream label is preserved.
    """
    abs_A = A.abs()
    gamma = abs_A.sum(dim=-1).clamp(min=_EPS_DENOM)   # (B, H, T_q)
    inv_gamma = (1.0 / gamma).unsqueeze(-1)
    if mu.dim() == A.dim() + 1:
        mu_pos_n = mu[:, 0] * inv_gamma
        mu_neg_n = mu[:, 1] * inv_gamma
        out_pos = torch.einsum('bhqd,bhqk->bhkd', mu_pos_n, abs_A)
        out_neg = torch.einsum('bhqd,bhqk->bhkd', mu_neg_n, abs_A)
        return torch.stack([out_pos, out_neg], dim=1)
    mu_n = mu * inv_gamma
    return torch.einsum('bhqd,bhqk->bhkd', mu_n, abs_A)


def _av_adf_propagate_beta(beta: Tensor, A_A: Tensor, A_B: Tensor) -> Tensor:
    """Geometric-mean ADF kernel for the AV matmul (Bhattacharyya sub-mass)."""
    gamma_A = A_A.abs().sum(dim=-1).clamp(min=_EPS_DENOM)
    gamma_B = A_B.abs().sum(dim=-1).clamp(min=_EPS_DENOM)
    sqrt_gamma = torch.sqrt((gamma_A * gamma_B).clamp(min=0))
    inv_sqrt_gamma = (1.0 / sqrt_gamma).unsqueeze(-1)
    sqrt_abs_A = torch.sqrt((A_A.abs() * A_B.abs()).clamp(min=0))
    if beta.dim() == A_A.dim() + 1:
        beta_pos_n = beta[:, 0] * inv_sqrt_gamma
        beta_neg_n = beta[:, 1] * inv_sqrt_gamma
        out_pos = torch.einsum('bhqd,bhqk->bhkd', beta_pos_n, sqrt_abs_A)
        out_neg = torch.einsum('bhqd,bhqk->bhkd', beta_neg_n, sqrt_abs_A)
        return torch.stack([out_pos, out_neg], dim=1)
    beta_n = beta * inv_sqrt_gamma
    return torch.einsum('bhqd,bhqk->bhkd', beta_n, sqrt_abs_A)


def _is_av_matmul(rec) -> bool:
    """Heuristic: a DCMatMul whose output last two dims differ is AV
    (T_q × D_h with D_h ≠ T_q), whereas QK^T is square (T_q × T_k = T × T)."""
    out = rec.output
    if out is None or out.dim() < 2:
        return False
    return out.shape[-1] != out.shape[-2]


def _adf_backward_av(rec_A, rec_B, state: _GameState) -> Dict[int, _GameState]:
    """ADF backward through the AV matmul.  Returns a state at input slot 1
    (the V tensor); slot 0 (the detached A tensor) gets nothing."""
    A_A = rec_A.inputs[0]
    A_B = rec_B.inputs[0]
    new_mu_A = _av_adf_propagate(state.mu_A, A_A)
    new_mu_B = _av_adf_propagate(state.mu_B, A_B)
    new_beta = _av_adf_propagate_beta(state.beta, A_A, A_B)
    return {
        1: _GameState(
            new_mu_A,
            new_mu_B,
            new_beta,
            sign_augmented=state.sign_augmented,
        )
    }


def _rg_backward_linear(
    module_A: nn.Linear,
    module_B: nn.Linear,
    a_in_A: Tensor,
    a_in_B: Tensor,
    state: _GameState,
    tau: float,
    cemetery: _CemeteryState,
) -> _GameState:
    W_A = module_A.weight.detach()
    W_B = module_B.weight.detach()
    uA_pos, uA_neg = _rg_linear_branch_scores(a_in_A, W_A, tau)
    uB_pos, uB_neg = _rg_linear_branch_scores(a_in_B, W_B, tau)

    pos_mu_A, pos_mu_B, pos_beta = _absorb_branch_dead(
        0.5 * state.mu_A,
        0.5 * state.mu_B,
        0.5 * state.beta,
        uA_pos.sum(dim=-1) <= _EPS_DENOM,
        uB_pos.sum(dim=-1) <= _EPS_DENOM,
        cemetery,
    )
    neg_mu_A, neg_mu_B, neg_beta = _absorb_branch_dead(
        0.5 * state.mu_A,
        0.5 * state.mu_B,
        0.5 * state.beta,
        uA_neg.sum(dim=-1) <= _EPS_DENOM,
        uB_neg.sum(dim=-1) <= _EPS_DENOM,
        cemetery,
    )

    SA_pos = uA_pos.sum(dim=-1).clamp(min=_EPS_DENOM)
    SB_pos = uB_pos.sum(dim=-1).clamp(min=_EPS_DENOM)
    SA_neg = uA_neg.sum(dim=-1).clamp(min=_EPS_DENOM)
    SB_neg = uB_neg.sum(dim=-1).clamp(min=_EPS_DENOM)

    new_mu_A = torch.einsum('bj,bji->bi', pos_mu_A / SA_pos, uA_pos) + torch.einsum('bj,bji->bi', neg_mu_A / SA_neg, uA_neg)
    new_mu_B = torch.einsum('bj,bji->bi', pos_mu_B / SB_pos, uB_pos) + torch.einsum('bj,bji->bi', neg_mu_B / SB_neg, uB_neg)
    new_beta = torch.einsum(
        'bj,bji->bi',
        pos_beta / torch.sqrt((SA_pos * SB_pos).clamp(min=0)).clamp(min=_EPS_DENOM),
        torch.sqrt((uA_pos * uB_pos).clamp(min=0)),
    ) + torch.einsum(
        'bj,bji->bi',
        neg_beta / torch.sqrt((SA_neg * SB_neg).clamp(min=0)).clamp(min=_EPS_DENOM),
        torch.sqrt((uA_neg * uB_neg).clamp(min=0)),
    )
    return _GameState(new_mu_A, new_mu_B, new_beta)


# ---------------------------------------------------------------------------
# RG backward for a "token-Linear": a Linear applied per token on a
# ``[..., T, n_in]`` activation (the ViT MLP linears, the attention output
# projection and the Value projection).  :func:`_rg_backward_linear` assumes
# a 2-D input (``einsum('bj,bji->bi', ...)``); the token-Linear helper
# computes the same kernel for any number of leading "batch + token"
# dimensions.  The dispatch picks it whenever ``a_in.ndim > 2``.
# ---------------------------------------------------------------------------

def _rg_backward_token_linear(
    module_A: nn.Linear,
    module_B: nn.Linear,
    a_in_A: Tensor,
    a_in_B: Tensor,
    state: _GameState,
    tau: float,
    cemetery: _CemeteryState,
) -> _GameState:
    """RG backward through a token-wise Linear (``a_in`` has ≥1 leading
    dims before ``n_in``).

    Mirrors :func:`_rg_backward_linear` — same sign-stream splitting, same
    dead-branch cemetery absorption, same geometric-mean beta — but uses a
    **factored LRP-αβ form** that never materialises the per-layer
    ``[..., n_out, n_in]`` routed-mass tensor.  For a ViT MLP layer with
    ``n_out=768, n_in=3072, T=197`` the naive
    ``a_in.unsqueeze(-2) * W`` tensor is ~1.7 GB per layer and OOMs on a
    12 GB card; the factored form keeps every temporary at most
    ``[B, T, n_in] ≈ 2.4 MB``, a ~700× reduction.

    Factorisation.  The routed mass ``u_pos[i, j] = max(0, a[j] W[i, j])``
    splits into the two disjoint sign cases (same-sign product):

        u_pos[i, j]  =  ap[j] · Wp[i, j]   (when a_A>0 and W_A[i,j]>0)
                      + an[j] · Wn[i, j]   (when a_A<0 and W_A[i,j]<0)

    where ``ap/an`` and ``Wp/Wn`` are the non-negative sign branches of the
    input activation and weight raised to ``1/τ`` (for ``a ≥ 0`` post-ReLU
    one of the two is zero, but for post-LayerNorm / post-GELU inputs both
    branches can fire, which is why we track them).  Row sums become

        SA_pos  =  F.linear(ap, Wp) + F.linear(an, Wn)   # per output neuron
        SA_neg  =  F.linear(ap, Wn) + F.linear(an, Wp)

    and the mu backward propagation becomes

        mu_in[..., n_in] =
              ap[..., n_in] · (mu_n @ Wp)[..., n_in]
            + an[..., n_in] · (mu_n @ Wn)[..., n_in]          (pos branch)
            + ap[..., n_in] · (mu_n @ Wn)[..., n_in]
            + an[..., n_in] · (mu_n @ Wp)[..., n_in]          (cross-stream)

    where ``mu_n @ W*`` is an ordinary ``torch.matmul`` of shape
    ``[..., n_out] @ [n_out, n_in] → [..., n_in]``.  Beta uses the same
    trick with the 4-case decomposition of ``√(u_A u_B)`` (see comment
    below).  Only ``[..., n_in]`` temporaries are allocated — no
    ``[..., n_out, n_in]`` tensors.  The flat 2-D path
    (:func:`_rg_backward_linear`) forms the routed-mass tensor directly.
    """
    W_A = module_A.weight.detach()
    W_B = module_B.weight.detach()
    inv_tau = 1.0 / max(tau, 1e-3)

    # Sign-split activations and weights, raised to 1/τ.  ``a_in`` may be
    # post-LayerNorm (can be negative), so both sign branches are kept.
    ap_A = F.relu(a_in_A)
    an_A = F.relu(-a_in_A)
    ap_B = F.relu(a_in_B)
    an_B = F.relu(-a_in_B)
    Wp_A = F.relu(W_A)
    Wn_A = F.relu(-W_A)
    Wp_B = F.relu(W_B)
    Wn_B = F.relu(-W_B)
    if abs(inv_tau - 1.0) > 1e-9:
        ap_A = ap_A.pow(inv_tau); an_A = an_A.pow(inv_tau)
        ap_B = ap_B.pow(inv_tau); an_B = an_B.pow(inv_tau)
        Wp_A = Wp_A.pow(inv_tau); Wn_A = Wn_A.pow(inv_tau)
        Wp_B = Wp_B.pow(inv_tau); Wn_B = Wn_B.pow(inv_tau)

    # Row sums of the routed-mass kernels (see factorisation above).
    # All four are shape ``[..., n_out]``.
    SA_pos = F.linear(ap_A, Wp_A) + F.linear(an_A, Wn_A)
    SA_neg = F.linear(ap_A, Wn_A) + F.linear(an_A, Wp_A)
    SB_pos = F.linear(ap_B, Wp_B) + F.linear(an_B, Wn_B)
    SB_neg = F.linear(ap_B, Wn_B) + F.linear(an_B, Wp_B)

    # Dead branches: row whose normaliser is zero gets absorbed.  Matches
    # the 2-D helper's behaviour.  Shape ``[..., n_out]`` broadcasts over
    # the sign axis of ``state.mu_*`` / ``state.beta`` via
    # ``_absorb_branch_dead``.
    pos_mu_A, pos_mu_B, pos_beta = _absorb_branch_dead(
        0.5 * state.mu_A,
        0.5 * state.mu_B,
        0.5 * state.beta,
        SA_pos <= _EPS_DENOM,
        SB_pos <= _EPS_DENOM,
        cemetery,
    )
    neg_mu_A, neg_mu_B, neg_beta = _absorb_branch_dead(
        0.5 * state.mu_A,
        0.5 * state.mu_B,
        0.5 * state.beta,
        SA_neg <= _EPS_DENOM,
        SB_neg <= _EPS_DENOM,
        cemetery,
    )

    # Clamped normalisers for the row-wise divisions.  RG state is NOT
    # sign-augmented (``state.mu_* / state.beta`` shape ``[B, *spatial,
    # n_out]``), so no sign-axis unsqueeze is needed — ``SA_pos`` already
    # shares the token/spatial layout with the state.
    SA_pos_c = SA_pos.clamp(min=_EPS_DENOM)
    SA_neg_c = SA_neg.clamp(min=_EPS_DENOM)
    SB_pos_c = SB_pos.clamp(min=_EPS_DENOM)
    SB_neg_c = SB_neg.clamp(min=_EPS_DENOM)

    # ---- mu_A, mu_B backward propagation ---------------------------------
    # mu_in[..., j] = Σ_i mu_n[..., i] * u[..., i, j]
    #   = ap[..., j] * (mu_n @ Wp)[..., j] + an[..., j] * (mu_n @ Wn)[..., j]
    # for u_pos = ap·Wp + an·Wn (same-sign case).  u_neg swaps Wp↔Wn on
    # the right factor.
    pos_mu_A_n = pos_mu_A / SA_pos_c
    neg_mu_A_n = neg_mu_A / SA_neg_c
    pos_mu_B_n = pos_mu_B / SB_pos_c
    neg_mu_B_n = neg_mu_B / SB_neg_c

    new_mu_A = (
        ap_A * (pos_mu_A_n @ Wp_A)
        + an_A * (pos_mu_A_n @ Wn_A)
        + ap_A * (neg_mu_A_n @ Wn_A)
        + an_A * (neg_mu_A_n @ Wp_A)
    )
    new_mu_B = (
        ap_B * (pos_mu_B_n @ Wp_B)
        + an_B * (pos_mu_B_n @ Wn_B)
        + ap_B * (neg_mu_B_n @ Wn_B)
        + an_B * (neg_mu_B_n @ Wp_B)
    )

    # ---- beta backward propagation (geometric-mean kernel) --------------
    # sqrt(u_A * u_B) decomposes into 4 mutually exclusive cases based on
    # (sign a_A[j], sign a_B[j]); each case picks a distinct (W_A, W_B)
    # sign pair.  Let AA_** = sqrt of the corresponding (a_A, a_B) branch
    # product and WW_** = sqrt of the corresponding (W_A, W_B) branch
    # product.  For the pos-routed mass kernel:
    #
    #     sqrt(u_pos_A * u_pos_B)  =
    #          AA_pp · WW_pp    (a_A+, a_B+ → W_A+, W_B+)
    #        + AA_pn · WW_pn    (a_A+, a_B− → W_A+, W_B−)
    #        + AA_np · WW_np    (a_A−, a_B+ → W_A−, W_B+)
    #        + AA_nn · WW_nn    (a_A−, a_B− → W_A−, W_B−)
    #
    # and for the neg-routed mass kernel the W sign pair is flipped:
    #
    #     sqrt(u_neg_A * u_neg_B)  =
    #          AA_pp · WW_nn    (a_A+, a_B+ → W_A−, W_B−)
    #        + AA_pn · WW_np    (a_A+, a_B− → W_A−, W_B+)
    #        + AA_np · WW_pn    (a_A−, a_B+ → W_A+, W_B−)
    #        + AA_nn · WW_pp    (a_A−, a_B− → W_A+, W_B+)
    AA_pp = torch.sqrt((ap_A * ap_B).clamp(min=0))   # [..., n_in]
    AA_pn = torch.sqrt((ap_A * an_B).clamp(min=0))
    AA_np = torch.sqrt((an_A * ap_B).clamp(min=0))
    AA_nn = torch.sqrt((an_A * an_B).clamp(min=0))
    WW_pp = torch.sqrt((Wp_A * Wp_B).clamp(min=0))   # [n_out, n_in]
    WW_pn = torch.sqrt((Wp_A * Wn_B).clamp(min=0))
    WW_np = torch.sqrt((Wn_A * Wp_B).clamp(min=0))
    WW_nn = torch.sqrt((Wn_A * Wn_B).clamp(min=0))

    sqrt_Z_pos = torch.sqrt((SA_pos_c * SB_pos_c).clamp(min=0)).clamp(min=_EPS_DENOM)
    sqrt_Z_neg = torch.sqrt((SA_neg_c * SB_neg_c).clamp(min=0)).clamp(min=_EPS_DENOM)
    pos_beta_n = pos_beta / sqrt_Z_pos
    neg_beta_n = neg_beta / sqrt_Z_neg

    new_beta_from_pos = (
        AA_pp * (pos_beta_n @ WW_pp)
        + AA_pn * (pos_beta_n @ WW_pn)
        + AA_np * (pos_beta_n @ WW_np)
        + AA_nn * (pos_beta_n @ WW_nn)
    )
    new_beta_from_neg = (
        AA_pp * (neg_beta_n @ WW_nn)
        + AA_pn * (neg_beta_n @ WW_np)
        + AA_np * (neg_beta_n @ WW_pn)
        + AA_nn * (neg_beta_n @ WW_pp)
    )
    new_beta = new_beta_from_pos + new_beta_from_neg

    return _GameState(
        new_mu_A,
        new_mu_B,
        new_beta,
        sign_augmented=state.sign_augmented,
    )


# ---------------------------------------------------------------------------
# RG attention-Value rule: the merged rule of the Routing Game
# (``av_rule='merged'`` in routing_game.py).
#
# For the conditioned block O = A (x W_v^T + b_v) with A >= 0 fixed, every
# Value entry (k, d) keeps its two stream totals
#     v~^kappa_kd = sum_e ([x_ke W_de]^kappa)^(1/tau),   kappa in {+, -},
# (bias omitted, as in every RG linear kernel of this module).  A trajectory
# at the output entry O_td splits 1/2-1/2 into the streams kappa (the -
# stream switches the player), moves to the Value entry (k, d) with
# probability A_tk^(1/tau) v~^kappa_kd / Z^kappa_td,
#     Z^kappa_td = sum_k' A_tk'^(1/tau) v~^kappa_k'd,
# and from there stays in stream kappa: it moves to the Value-projection
# input (k, e) with probability ([x_ke W_de]^kappa)^(1/tau) / v~^kappa_kd, with
# no second split and no second player switch.  A stream with zero total
# sends its share to the cemetery.
#
# The composite step is realised in two stages that share the sweep's
# tensor-id routing: the attention product returns a stream-labelled state
# on the Value entries (sign-augmented layout ``(B, 2, *V.shape[1:])``,
# dim 1 = kappa), the layout ops between the Value projection and the
# attention operand carry that label through, and the Value projection
# applies the stream-preserving second stage.  Only ``[..., T, T]`` and
# ``[..., T, n]`` tensors are formed (no ``[T, T, d, e]`` products).
# ---------------------------------------------------------------------------

def _rg_token_linear_terms(
    a_in: Tensor,
    W: Tensor,
    tau: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Sign-split input and weights raised to ``1/tau`` and the two stream totals.

    Returns ``(ap, an, Wp, Wn, S_pos, S_neg)`` with
    ``S_pos[..., i] = sum_j ([a_j W_ij]^+)^(1/tau)`` and ``S_neg`` the same
    for the negative products, in the factored form used by
    :func:`_rg_backward_token_linear` (bias omitted).
    """
    inv_tau = 1.0 / max(tau, 1e-3)
    ap = F.relu(a_in)
    an = F.relu(-a_in)
    Wp = F.relu(W)
    Wn = F.relu(-W)
    if abs(inv_tau - 1.0) > 1e-9:
        ap = ap.pow(inv_tau)
        an = an.pow(inv_tau)
        Wp = Wp.pow(inv_tau)
        Wn = Wn.pow(inv_tau)
    S_pos = F.linear(ap, Wp) + F.linear(an, Wn)
    S_neg = F.linear(ap, Wn) + F.linear(an, Wp)
    return ap, an, Wp, Wn, S_pos, S_neg


def _is_value_view(V: Tensor, vp_out: Tensor) -> bool:
    """True if ``V`` is a view of the contiguous tensor ``vp_out``."""
    if not vp_out.is_contiguous():
        return False
    if V.untyped_storage().data_ptr() != vp_out.untyped_storage().data_ptr():
        return False
    if any(s < 0 for s in V.stride()):
        return False
    lo = V.storage_offset() - vp_out.storage_offset()
    hi = lo + sum((n - 1) * s for n, s in zip(V.shape, V.stride()) if n > 0)
    return lo >= 0 and hi < vp_out.numel()


def _value_layout(t: Tensor, vp_out: Tensor, V: Tensor) -> Tensor:
    """Apply to ``t`` (shaped like ``vp_out``) the view that maps the Value
    projection output ``vp_out`` onto the attention operand ``V``."""
    base = t.contiguous()
    offset = base.storage_offset() + V.storage_offset() - vp_out.storage_offset()
    return base.as_strided(V.size(), V.stride(), offset)


def _find_value_projection(calls: List[_CallRecord], av_index: int) -> Optional[int]:
    """Index of the latest ``nn.Linear`` call before ``av_index`` whose output
    the Value operand of the attention product at ``av_index`` is a view of,
    or ``None``.  Any non-layout operation between the two would allocate
    new storage, so the view relation means only layout steps separate them."""
    rec = calls[av_index]
    V = rec.inputs[1] if len(rec.inputs) > 1 else None
    if not isinstance(V, Tensor):
        return None
    for i in range(av_index - 1, -1, -1):
        c = calls[i]
        if isinstance(c.module, nn.Linear) and isinstance(c.output, Tensor) and _is_value_view(V, c.output):
            return i
    return None


def _av_backward_merged(
    rec_A,
    rec_B,
    vp_A,
    vp_B,
    state: _GameState,
    cemetery: _CemeteryState,
    tau: float,
) -> Dict[int, _GameState]:
    """First stage of the merged rule: attention output -> stream-labelled Value entries.

    ``vp_A`` / ``vp_B`` are the call records of the Value projections that
    produce the attention operand ``V`` (a view of their output).  Returns
    a sign-augmented state at input slot 1 (``V``) whose dim 1 is the stream
    ``kappa``; slot 0 (the detached attention matrix) gets nothing.

    Per stream, the geometric-mean kernel of the composite step factors as
    ``sqrt(A_A A_B)_tk sqrt(v~_A v~_B)_kd / sqrt(Z_A Z_B)_td`` times the
    second-stage factor of :func:`_rg_backward_value_projection`.
    """
    inv_tau = 1.0 / max(tau, 1e-3)

    def _terms(rec_av, rec_vp) -> Tuple[Tensor, Tensor, Tensor]:
        A = rec_av.inputs[0]
        V = rec_av.inputs[1]
        W = rec_vp.module.weight.detach()
        *_, S_pos, S_neg = _rg_token_linear_terms(rec_vp.inputs[0], W, tau)
        if abs(inv_tau - 1.0) > 1e-9:
            A = A.pow(inv_tau)
        return (
            A,
            _value_layout(S_pos, rec_vp.output, V),
            _value_layout(S_neg, rec_vp.output, V),
        )

    A_A, vA_pos, vA_neg = _terms(rec_A, vp_A)
    A_B, vB_pos, vB_neg = _terms(rec_B, vp_B)
    At_A = A_A.transpose(-2, -1)
    At_B = A_B.transpose(-2, -1)
    sqrt_At = torch.sqrt((A_A * A_B).clamp(min=0)).transpose(-2, -1)

    mu_A_streams: List[Tensor] = []
    mu_B_streams: List[Tensor] = []
    beta_streams: List[Tensor] = []
    for vA, vB in ((vA_pos, vB_pos), (vA_neg, vB_neg)):
        Z_A = torch.matmul(A_A, vA)
        Z_B = torch.matmul(A_B, vB)
        mu_A, mu_B, beta = _absorb_branch_dead(
            0.5 * state.mu_A, 0.5 * state.mu_B, 0.5 * state.beta,
            Z_A <= _EPS_DENOM, Z_B <= _EPS_DENOM, cemetery,
        )
        Z_A = Z_A.clamp(min=_EPS_DENOM)
        Z_B = Z_B.clamp(min=_EPS_DENOM)
        mu_A_streams.append(vA * torch.matmul(At_A, mu_A / Z_A))
        mu_B_streams.append(vB * torch.matmul(At_B, mu_B / Z_B))
        beta_n = beta / torch.sqrt(Z_A * Z_B).clamp(min=_EPS_DENOM)
        beta_streams.append(torch.sqrt((vA * vB).clamp(min=0)) * torch.matmul(sqrt_At, beta_n))

    return {
        1: _GameState(
            torch.stack(mu_A_streams, dim=1),
            torch.stack(mu_B_streams, dim=1),
            torch.stack(beta_streams, dim=1),
            sign_augmented=True,
        )
    }


def _rg_backward_value_projection(
    module_A: nn.Linear,
    module_B: nn.Linear,
    a_in_A: Tensor,
    a_in_B: Tensor,
    state: _GameState,
    tau: float,
    cemetery: _CemeteryState,
) -> _GameState:
    """Second stage of the merged rule: stream-labelled Value entries -> projection input.

    ``state`` is the sign-augmented state of :func:`_av_backward_merged`
    (dim 1 = stream ``kappa``) in the layout of the projection output.  The
    stream-``kappa`` mass at the Value entry ``(k, d)`` moves to the input
    unit ``(k, e)`` with probability ``([x_ke W_de]^kappa)^(1/tau) / v~^kappa_kd``:
    the routing of :func:`_rg_backward_token_linear` restricted to the
    stream the entry carries, without the 1/2 split.  An entry with zero
    stream total sends its mass to the cemetery.  Returns the unlabelled
    state at the projection input.
    """
    apA, anA, WpA, WnA, SA_pos, SA_neg = _rg_token_linear_terms(a_in_A, module_A.weight.detach(), tau)
    apB, anB, WpB, WnB, SB_pos, SB_neg = _rg_token_linear_terms(a_in_B, module_B.weight.detach(), tau)

    # sqrt(u_A u_B) in the four-case form of _rg_backward_token_linear.
    AA_pp = torch.sqrt((apA * apB).clamp(min=0))
    AA_pn = torch.sqrt((apA * anB).clamp(min=0))
    AA_np = torch.sqrt((anA * apB).clamp(min=0))
    AA_nn = torch.sqrt((anA * anB).clamp(min=0))
    WW_pp = torch.sqrt((WpA * WpB).clamp(min=0))
    WW_pn = torch.sqrt((WpA * WnB).clamp(min=0))
    WW_np = torch.sqrt((WnA * WpB).clamp(min=0))
    WW_nn = torch.sqrt((WnA * WnB).clamp(min=0))

    # Stream + routes along u+ = ap Wp + an Wn, stream - along u- = ap Wn + an Wp.
    streams = (
        (SA_pos, SB_pos, (WpA, WnA), (WpB, WnB), (WW_pp, WW_pn, WW_np, WW_nn)),
        (SA_neg, SB_neg, (WnA, WpA), (WnB, WpB), (WW_nn, WW_np, WW_pn, WW_pp)),
    )
    new_mu_A: Optional[Tensor] = None
    new_mu_B: Optional[Tensor] = None
    new_beta: Optional[Tensor] = None
    for s, (SA, SB, (WlA, WrA), (WlB, WrB), (Wpp, Wpn, Wnp, Wnn)) in enumerate(streams):
        mu_A, mu_B, beta = _absorb_branch_dead(
            state.mu_A[:, s], state.mu_B[:, s], state.beta[:, s],
            SA <= _EPS_DENOM, SB <= _EPS_DENOM, cemetery,
        )
        SA_c = SA.clamp(min=_EPS_DENOM)
        SB_c = SB.clamp(min=_EPS_DENOM)
        mu_A_n = mu_A / SA_c
        mu_B_n = mu_B / SB_c
        beta_n = beta / torch.sqrt(SA_c * SB_c).clamp(min=_EPS_DENOM)
        part_mu_A = apA * (mu_A_n @ WlA) + anA * (mu_A_n @ WrA)
        part_mu_B = apB * (mu_B_n @ WlB) + anB * (mu_B_n @ WrB)
        part_beta = (
            AA_pp * (beta_n @ Wpp)
            + AA_pn * (beta_n @ Wpn)
            + AA_np * (beta_n @ Wnp)
            + AA_nn * (beta_n @ Wnn)
        )
        new_mu_A = part_mu_A if new_mu_A is None else new_mu_A + part_mu_A
        new_mu_B = part_mu_B if new_mu_B is None else new_mu_B + part_mu_B
        new_beta = part_beta if new_beta is None else new_beta + part_beta

    return _GameState(new_mu_A, new_mu_B, new_beta)


def _rg_backward_conv2d(
    module_A: nn.Conv2d,
    module_B: nn.Conv2d,
    a_in_A: Tensor,
    a_in_B: Tensor,
    state: _GameState,
    tau: float,
    cemetery: _CemeteryState,
) -> _GameState:
    """RG backward through Conv2d.

    The routed mass ``([a W]^kappa)^(1/tau)`` of each stream factorizes
    into the sign branches of the input and of the weights, each raised to
    ``1/tau``, so the normalizers ``Z(j)`` are computed with ``F.conv2d``
    and the propagation with ``F.conv_transpose2d``.  The geometric-mean
    kernel keeps the input terms whose sign agrees between A and B; the
    cross-sign terms vanish whenever A and B share the input or the
    weights.
    """
    W_A = module_A.weight.detach()
    W_B = module_B.weight.detach()
    inv_tau = 1.0 / max(tau, 1e-3)

    in_shape_A = (a_in_A.size(-2), a_in_A.size(-1))
    in_shape_B = (a_in_B.size(-2), a_in_B.size(-1))
    out_shape = (state.mu_A.size(-2), state.mu_A.size(-1))
    op_A = _conv2d_output_padding(in_shape_A, out_shape, module_A)
    op_B = _conv2d_output_padding(in_shape_B, out_shape, module_B)

    def _conv(x, w, mod):
        return F.conv2d(x, w, bias=None, stride=mod.stride, padding=mod.padding,
                        dilation=mod.dilation, groups=mod.groups)

    def _conv_T(x, w, mod, op):
        return F.conv_transpose2d(x, w, bias=None, stride=mod.stride,
                                  padding=mod.padding, output_padding=op,
                                  dilation=mod.dilation, groups=mod.groups)

    def _prepare_branch_terms(a_in: Tensor, W: Tensor, mod: nn.Conv2d) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        a_pos = F.relu(a_in)
        a_neg = F.relu(-a_in)
        W_pos = F.relu(W)
        W_neg = F.relu(-W)
        if abs(inv_tau - 1.0) > 1e-9:
            a_pos = a_pos.pow(inv_tau)
            a_neg = a_neg.pow(inv_tau)
            W_pos = W_pos.pow(inv_tau)
            W_neg = W_neg.pow(inv_tau)
        u_pos_map = _conv(a_pos, W_pos, mod) + _conv(a_neg, W_neg, mod)
        u_neg_map = _conv(a_pos, W_neg, mod) + _conv(a_neg, W_pos, mod)
        if u_pos_map.shape[-2:] != state.beta.shape[-2:]:
            u_pos_map = u_pos_map[..., :state.beta.size(-2), :state.beta.size(-1)]
            u_neg_map = u_neg_map[..., :state.beta.size(-2), :state.beta.size(-1)]
        return a_pos, a_neg, W_pos, W_neg, u_pos_map, u_neg_map

    def _propagate_branch(
        branch_mu: Tensor,
        branch_u: Tensor,
        a_pos: Tensor,
        a_neg: Tensor,
        W_left: Tensor,
        W_right: Tensor,
        mod: nn.Conv2d,
        op: Tuple[int, int],
        target_hw: Tuple[int, int],
    ) -> Tensor:
        mu_n = branch_mu / branch_u.clamp(min=_EPS_DENOM)

        def _crop_back(x: Tensor) -> Tensor:
            return x[..., :target_hw[0], :target_hw[1]]

        return a_pos * _crop_back(_conv_T(mu_n, W_left, mod, op)) + a_neg * _crop_back(_conv_T(mu_n, W_right, mod, op))

    aA_pos, aA_neg, WA_pos, WA_neg, uA_pos, uA_neg = _prepare_branch_terms(a_in_A, W_A, module_A)
    aB_pos, aB_neg, WB_pos, WB_neg, uB_pos, uB_neg = _prepare_branch_terms(a_in_B, W_B, module_B)

    pos_mu_A, pos_mu_B, pos_beta = _absorb_branch_dead(
        0.5 * state.mu_A, 0.5 * state.mu_B, 0.5 * state.beta,
        uA_pos <= _EPS_DENOM, uB_pos <= _EPS_DENOM, cemetery,
    )
    neg_mu_A, neg_mu_B, neg_beta = _absorb_branch_dead(
        0.5 * state.mu_A, 0.5 * state.mu_B, 0.5 * state.beta,
        uA_neg <= _EPS_DENOM, uB_neg <= _EPS_DENOM, cemetery,
    )

    target_hw = (a_in_A.size(-2), a_in_A.size(-1))
    new_mu_A = _propagate_branch(pos_mu_A, uA_pos, aA_pos, aA_neg, WA_pos, WA_neg, module_A, op_A, target_hw) + _propagate_branch(
        neg_mu_A, uA_neg, aA_pos, aA_neg, WA_neg, WA_pos, module_A, op_A, target_hw,
    )
    new_mu_B = _propagate_branch(pos_mu_B, uB_pos, aB_pos, aB_neg, WB_pos, WB_neg, module_B, op_B, target_hw) + _propagate_branch(
        neg_mu_B, uB_neg, aB_pos, aB_neg, WB_neg, WB_pos, module_B, op_B, target_hw,
    )

    sqrt_a_pos = torch.sqrt((aA_pos * aB_pos).clamp(min=0))
    sqrt_a_neg = torch.sqrt((aA_neg * aB_neg).clamp(min=0))
    sqrt_W_pos = torch.sqrt((WA_pos * WB_pos).clamp(min=0))
    sqrt_W_neg = torch.sqrt((WA_neg * WB_neg).clamp(min=0))

    def _propagate_branch_beta(
        branch_beta: Tensor,
        ZA: Tensor,
        ZB: Tensor,
        a_left: Tensor,
        a_right: Tensor,
        W_left: Tensor,
        W_right: Tensor,
    ) -> Tensor:
        beta_n = branch_beta / torch.sqrt((ZA.clamp(min=_EPS_DENOM) * ZB.clamp(min=_EPS_DENOM)).clamp(min=0)).clamp(min=_EPS_DENOM)

        def _crop_back(x: Tensor) -> Tensor:
            return x[..., :target_hw[0], :target_hw[1]]

        return a_left * _crop_back(_conv_T(beta_n, W_left, module_A, op_A)) + a_right * _crop_back(_conv_T(beta_n, W_right, module_A, op_A))

    new_beta = _propagate_branch_beta(pos_beta, uA_pos, uB_pos, sqrt_a_pos, sqrt_a_neg, sqrt_W_pos, sqrt_W_neg) + _propagate_branch_beta(
        neg_beta, uA_neg, uB_neg, sqrt_a_pos, sqrt_a_neg, sqrt_W_neg, sqrt_W_pos,
    )
    return _GameState(new_mu_A, new_mu_B, new_beta)


# ---------------------------------------------------------------------------
# Identity / shape / pooling backward steps (shared by both games)
# ---------------------------------------------------------------------------

def _backward_identity(state: _GameState) -> _GameState:
    return state


def _backward_flatten(module: nn.Flatten, in_A: Tensor, state: _GameState) -> _GameState:
    del module
    return state.reshape_to(tuple(in_A.shape))


def _backward_reshape(in_A: Tensor, state: _GameState) -> _GameState:
    """Backward through Reshape/View: reshape spatial dims to the input shape."""
    return state.reshape_to(tuple(in_A.shape))


def _backward_transpose(module: "_DCTranspose", state: _GameState) -> _GameState:
    """Backward through Transpose: transpose is its own inverse.

    The state has shape ``(B, *spatial)`` where the spatial axes line up
    with the forward output, so we apply the same transpose after
    offsetting by the batch axis.
    """
    offset = 2 if state.sign_augmented else 1
    d0 = module.dim0 + offset if module.dim0 >= 0 else module.dim0
    d1 = module.dim1 + offset if module.dim1 >= 0 else module.dim1
    new_mu_A = state.mu_A.transpose(d0, d1)
    new_mu_B = state.mu_B.transpose(d0, d1)
    new_beta = state.beta.transpose(d0, d1)
    return _GameState(
        new_mu_A,
        new_mu_B,
        new_beta,
        sign_augmented=state.sign_augmented,
    )


def _backward_permute(module: "_DCPermute", state: _GameState) -> _GameState:
    """Backward through Permute: apply the inverse permutation.

    ``module.dims`` stores the full-rank tensor permutation (including the
    batch axis), as created by
    :func:`bwgame.functional_replacer._get_method_replacement`.
    Example: ``Permute(dims=(0, 2, 1))`` for a 3-D token tensor.
    """
    dims = list(module.dims)
    inv_dims = [0] * len(dims)
    for i, d in enumerate(dims):
        inv_dims[d] = i
    if state.sign_augmented:
        # State layout: (B, 2, *spatial).  Tensor dim 0 → state dim 0,
        # tensor dim k ≥ 1 → state dim k + 1; the sign dim at state
        # position 1 stays fixed.
        state_inv: List[int] = []
        for d in inv_dims:
            state_inv.append(d if d == 0 else d + 1)
        # Insert "sign" dim right after the batch dim in the output layout.
        # Find where state dim 0 (batch) landed, then put sign right after.
        batch_pos = state_inv.index(0)
        state_inv.insert(batch_pos + 1, 1)
    else:
        state_inv = inv_dims
    new_mu_A = state.mu_A.permute(state_inv).contiguous()
    new_mu_B = state.mu_B.permute(state_inv).contiguous()
    new_beta = state.beta.permute(state_inv).contiguous()
    return _GameState(
        new_mu_A,
        new_mu_B,
        new_beta,
        sign_augmented=state.sign_augmented,
    )


def _backward_dccat(
    module: "_DCCat",
    tensor_list: List[Tensor],
    state: _GameState,
) -> List[_GameState]:
    """Backward through DCCat: split state along the cat dim.

    ``torch.cat`` is a linear op that stitches several tensors into a
    single one along ``dim``.  The path-measure backward is its exact
    inverse: split the incoming state into chunks whose lengths match the
    original input tensors, and route each chunk to the corresponding
    upstream tensor.  Total mass is preserved because the chunks are disjoint
    and cover the input.
    """
    # Map the cat dim (indexing the *tensor*) to the state-tensor dim,
    # accounting for the optional leading sign layer at state dim 1.
    offset = 2 if state.sign_augmented else 1
    tensor_ndim = tensor_list[0].dim()
    cat_dim = module.dim
    if cat_dim < 0:
        cat_dim = cat_dim + tensor_ndim
    state_dim = cat_dim + (offset - 1)  # tensor batch=0 -> state 0 or state 0,1
    # Split sizes straight from the list entries' shapes along the cat dim.
    sizes = [t.shape[cat_dim] for t in tensor_list]

    mu_A_splits = torch.split(state.mu_A, sizes, dim=state_dim)
    mu_B_splits = torch.split(state.mu_B, sizes, dim=state_dim)
    beta_splits = torch.split(state.beta, sizes, dim=state_dim)

    return [
        _GameState(
            mu_a.contiguous(),
            mu_b.contiguous(),
            be.contiguous(),
            sign_augmented=state.sign_augmented,
        )
        for mu_a, mu_b, be in zip(mu_A_splits, mu_B_splits, beta_splits)
    ]


def _backward_avgpool2d(
    module: nn.AvgPool2d,
    in_A: Tensor, in_B: Tensor,
    state: _GameState,
) -> _GameState:
    """Average pooling: each output position contributes uniformly to its
    receptive field.  Same kernel for both games.
    """
    def _grad_input(g: Tensor, a_ref: Tensor) -> Tensor:
        a = a_ref.detach().clone().requires_grad_(True)
        out = F.avg_pool2d(
            a,
            kernel_size=module.kernel_size,
            stride=module.stride if module.stride is not None else module.kernel_size,
            padding=module.padding,
        )
        if g.shape != out.shape:
            g = g[..., :out.size(-2), :out.size(-1)]
        out.backward(g)
        return a.grad if a.grad is not None else torch.zeros_like(a_ref)

    if state.sign_augmented:
        new_mu_A = torch.stack([_grad_input(state.mu_A[:, s], in_A) for s in range(2)], dim=1)
        new_mu_B = torch.stack([_grad_input(state.mu_B[:, s], in_B) for s in range(2)], dim=1)
        new_beta = torch.stack([_grad_input(state.beta[:, s], in_A) for s in range(2)], dim=1)
    else:
        new_mu_A = _grad_input(state.mu_A, in_A)
        new_mu_B = _grad_input(state.mu_B, in_B)
        new_beta = _grad_input(state.beta, in_A)
    return _GameState(
        new_mu_A,
        new_mu_B,
        new_beta,
        sign_augmented=state.sign_augmented,
    )


def _backward_adaptive_avgpool2d(
    module: nn.AdaptiveAvgPool2d,
    in_A: Tensor, in_B: Tensor,
    state: _GameState,
) -> _GameState:
    def _grad_input(g: Tensor, a_ref: Tensor) -> Tensor:
        a = a_ref.detach().clone().requires_grad_(True)
        out = F.adaptive_avg_pool2d(a, module.output_size)
        out.backward(g)
        return a.grad if a.grad is not None else torch.zeros_like(a_ref)

    if state.sign_augmented:
        new_mu_A = torch.stack([_grad_input(state.mu_A[:, s], in_A) for s in range(2)], dim=1)
        new_mu_B = torch.stack([_grad_input(state.mu_B[:, s], in_B) for s in range(2)], dim=1)
        new_beta = torch.stack([_grad_input(state.beta[:, s], in_A) for s in range(2)], dim=1)
    else:
        new_mu_A = _grad_input(state.mu_A, in_A)
        new_mu_B = _grad_input(state.mu_B, in_B)
        new_beta = _grad_input(state.beta, in_A)
    return _GameState(
        new_mu_A,
        new_mu_B,
        new_beta,
        sign_augmented=state.sign_augmented,
    )


def _backward_maxpool2d(
    module: nn.MaxPool2d,
    in_A: Tensor, in_B: Tensor,
    state: _GameState,
) -> _GameState:
    """MaxPool backward.

    The pooling kernel is deterministic: T_M(out -> in) = 1 at the argmax
    position, 0 elsewhere.  For mu we route to each model's argmax.  For
    beta the geometric-mean kernel sqrt(T_A * T_B) is non-zero only where
    BOTH models have their argmax — so beta only flows when the two
    argmaxes agree, and is destroyed otherwise.
    """
    kernel_size = module.kernel_size
    stride = module.stride if module.stride is not None else module.kernel_size
    padding = module.padding
    dilation = module.dilation

    _, idx_A = F.max_pool2d(
        in_A.abs(), kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, return_indices=True,
    )
    _, idx_B = F.max_pool2d(
        in_B.abs(), kernel_size=kernel_size, stride=stride,
        padding=padding, dilation=dilation, return_indices=True,
    )

    def _unpool(g: Tensor, idx: Tensor, full_out_shape: Tuple[int, ...]) -> Tensor:
        # F.max_unpool2d expects output_size as the full 4-D target shape
        # (N, C, H, W); passing only the spatial dims raises a ValueError.
        return F.max_unpool2d(
            g, idx, kernel_size=kernel_size, stride=stride,
            padding=padding, output_size=full_out_shape,
        )

    # Build the per-sign target shape (N, C, H, W) for the spatial unpool.
    # state.mu_A is (N, 2, C, H, W), so the per-sign slice is (N, C, H, W);
    # the spatial dims of the destination are in_A.shape[-2:].
    def _full_shape(state_one: Tensor, in_t: Tensor) -> Tuple[int, ...]:
        return state_one.shape[:-2] + in_t.shape[-2:]

    def _process_mu(state_one: Tensor, idx: Tensor, in_t: Tensor) -> Tensor:
        out_shape = _full_shape(state_one, in_t)
        return _unpool(state_one, idx, out_shape)

    if state.sign_augmented:
        new_mu_A = torch.stack([_process_mu(state.mu_A[:, s], idx_A, in_A) for s in range(2)], dim=1)
        new_mu_B = torch.stack([_process_mu(state.mu_B[:, s], idx_B, in_B) for s in range(2)], dim=1)
        agree = (idx_A == idx_B).float()
        new_beta = torch.stack(
            [_unpool(state.beta[:, s] * agree, idx_A, _full_shape(state.beta[:, s], in_A)) for s in range(2)],
            dim=1,
        )
    else:
        new_mu_A = _process_mu(state.mu_A, idx_A, in_A)
        new_mu_B = _process_mu(state.mu_B, idx_B, in_B)
        agree = (idx_A == idx_B)
        beta_gated = state.beta * agree.float()
        beta_full_shape = _full_shape(state.beta, in_A)
        new_beta = _unpool(beta_gated, idx_A, beta_full_shape)
    return _GameState(
        new_mu_A,
        new_mu_B,
        new_beta,
        sign_augmented=state.sign_augmented,
    )


# ---------------------------------------------------------------------------
# ADF conditioned-survival helpers
# ---------------------------------------------------------------------------

def _adf_survival_from_input(x: Tensor) -> Tensor:
    """Unit survival on ordinary input states for the sign-split ADF space."""
    return torch.ones((x.size(0), 2) + tuple(x.shape[1:]), device=x.device, dtype=x.dtype)


def _adf_survival_numel_per_batch(h: Tensor) -> int:
    return int(h[0, 0].numel())


def _reshape_adf_survival(h: Tensor, target_shape: Tuple[int, ...]) -> Tensor:
    return h.reshape((target_shape[0], 2) + tuple(target_shape[1:]))


def _try_slice_extract_adf_survival(
    h: Tensor,
    target_shape: Tuple[int, ...],
) -> Optional[Tensor]:
    """Best-effort inverse of a forward functional slice like ``x[:, 0]``."""
    state_spatial = tuple(h.shape[2:])
    target_spatial = tuple(target_shape[1:])
    if state_spatial == target_spatial:
        return h
    if len(state_spatial) <= len(target_spatial):
        return None
    if target_spatial != state_spatial[-len(target_spatial):]:
        return None
    leading = len(state_spatial) - len(target_spatial)
    idx = (slice(None), slice(None)) + (0,) * leading + (slice(None),) * len(target_spatial)
    return h[idx].contiguous()


def _mask_adf_conditioned_state(
    state: _GameState,
    h_out_A: Tensor,
    h_out_B: Tensor,
) -> _GameState:
    """Zero out numerically impossible source states before a conditioned step."""
    alive_A = (h_out_A > _EPS_DENOM)
    alive_B = (h_out_B > _EPS_DENOM)
    alive_beta = alive_A & alive_B
    return _GameState(
        torch.where(alive_A, state.mu_A, torch.zeros_like(state.mu_A)),
        torch.where(alive_B, state.mu_B, torch.zeros_like(state.mu_B)),
        torch.where(alive_beta, state.beta, torch.zeros_like(state.beta)),
        sign_augmented=state.sign_augmented,
    )


def _conditional_div(num: Tensor, den: Tensor) -> Tensor:
    return torch.where(den > _EPS_DENOM, num / den.clamp(min=_EPS_DENOM), torch.zeros_like(num))


def _adf_activation_survival(mod: nn.Module, z: Tensor, h_in: Tensor) -> Tensor:
    """Forward survival propagation through a gated activation:
    ``h_out = h_in * g(z)`` with the gate of :func:`_activation_gate`."""
    return h_in * _activation_gate(z, mod).unsqueeze(1)


def _adf_linear_survival(h_in: Tensor, W: Tensor) -> Tensor:
    abs_W = W.abs()
    gamma = abs_W.sum(dim=-1).clamp(min=_EPS_DENOM)
    W_pos = F.relu(W)
    W_neg = F.relu(-W)
    h_pos = h_in[:, 0]
    h_neg = h_in[:, 1]
    out_pos = (h_pos @ W_pos.transpose(-2, -1) + h_neg @ W_neg.transpose(-2, -1)) / gamma
    out_neg = (h_pos @ W_neg.transpose(-2, -1) + h_neg @ W_pos.transpose(-2, -1)) / gamma
    return torch.stack([out_pos, out_neg], dim=1)


def _adf_conv2d_survival(
    module: nn.Conv2d,
    h_in: Tensor,
    in_ref: Tensor,
    out_ref: Tensor,
) -> Tensor:
    W = module.weight.detach()
    abs_W = W.abs()
    W_pos = F.relu(W)
    W_neg = F.relu(-W)

    def _conv(x: Tensor, w: Tensor) -> Tensor:
        return F.conv2d(
            x,
            w,
            bias=None,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

    gamma = _conv(torch.ones_like(in_ref), abs_W).clamp(min=_EPS_DENOM)
    if gamma.shape[-2:] != out_ref.shape[-2:]:
        gamma = gamma[..., :out_ref.size(-2), :out_ref.size(-1)]
    h_pos = h_in[:, 0]
    h_neg = h_in[:, 1]
    out_pos = _conv(h_pos, W_pos) + _conv(h_neg, W_neg)
    out_neg = _conv(h_pos, W_neg) + _conv(h_neg, W_pos)
    if out_pos.shape[-2:] != out_ref.shape[-2:]:
        out_pos = out_pos[..., :out_ref.size(-2), :out_ref.size(-1)]
        out_neg = out_neg[..., :out_ref.size(-2), :out_ref.size(-1)]
    return torch.stack([out_pos / gamma, out_neg / gamma], dim=1)


def _adf_avgpool2d_survival(module: nn.AvgPool2d, h_in: Tensor) -> Tensor:
    kwargs = dict(
        kernel_size=module.kernel_size,
        stride=module.stride if module.stride is not None else module.kernel_size,
        padding=module.padding,
        ceil_mode=module.ceil_mode,
        count_include_pad=module.count_include_pad,
        divisor_override=module.divisor_override,
    )
    out_pos = F.avg_pool2d(h_in[:, 0], **kwargs)
    out_neg = F.avg_pool2d(h_in[:, 1], **kwargs)
    return torch.stack([out_pos, out_neg], dim=1)


def _adf_adaptive_avgpool2d_survival(module: nn.AdaptiveAvgPool2d, h_in: Tensor) -> Tensor:
    out_pos = F.adaptive_avg_pool2d(h_in[:, 0], module.output_size)
    out_neg = F.adaptive_avg_pool2d(h_in[:, 1], module.output_size)
    return torch.stack([out_pos, out_neg], dim=1)


def _adf_maxpool2d_survival(
    module: nn.MaxPool2d,
    h_in: Tensor,
    in_ref: Tensor,
) -> Tensor:
    kernel_size = module.kernel_size
    stride = module.stride if module.stride is not None else module.kernel_size
    padding = module.padding
    dilation = module.dilation
    _, idx = F.max_pool2d(
        in_ref.abs(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=module.ceil_mode,
        return_indices=True,
    )

    def _gather(h_sign: Tensor) -> Tensor:
        flat = h_sign.flatten(start_dim=2)
        gathered = flat.gather(2, idx.flatten(start_dim=2))
        return gathered.view_as(idx)

    return torch.stack([_gather(h_in[:, 0]), _gather(h_in[:, 1])], dim=1)


def _adf_dccat_survival(
    module: "_DCCat",
    tensor_list: List[Tensor],
    h_inputs: List[Tensor],
) -> Tensor:
    cat_dim = module.dim
    tensor_ndim = tensor_list[0].dim()
    if cat_dim < 0:
        cat_dim = cat_dim + tensor_ndim
    state_dim = cat_dim if cat_dim == 0 else cat_dim + 1
    return torch.cat(h_inputs, dim=state_dim).contiguous()


def _zero_adf_survival_like(tensor: Tensor) -> Tensor:
    batch = tensor.shape[0] if tensor.dim() > 0 else 1
    return tensor.new_zeros((batch, 2) + tuple(tensor.shape[1:])).contiguous()


def _adf_av_survival(h_in: Tensor, A: Tensor) -> Tensor:
    abs_A = A.abs()
    gamma = abs_A.sum(dim=-1).clamp(min=_EPS_DENOM)
    A_norm = abs_A / gamma.unsqueeze(-1)
    out_pos = torch.einsum('bhkd,bhqk->bhqd', h_in[:, 0], A_norm)
    out_neg = torch.einsum('bhkd,bhqk->bhqd', h_in[:, 1], A_norm)
    return torch.stack([out_pos, out_neg], dim=1)


def _adf_backward_linear_conditioned(
    module_A: nn.Linear,
    module_B: nn.Linear,
    state: _GameState,
    h_in_A: Tensor,
    h_in_B: Tensor,
    h_out_A: Tensor,
    h_out_B: Tensor,
) -> _GameState:
    W_A = module_A.weight.detach()
    W_B = module_B.weight.detach()
    WA_pos = F.relu(W_A)
    WA_neg = F.relu(-W_A)
    WB_pos = F.relu(W_B)
    WB_neg = F.relu(-W_B)
    gamma_A = W_A.abs().sum(dim=-1).clamp(min=_EPS_DENOM)
    gamma_B = W_B.abs().sum(dim=-1).clamp(min=_EPS_DENOM)

    hA_pos = h_in_A[:, 0]
    hA_neg = h_in_A[:, 1]
    hB_pos = h_in_B[:, 0]
    hB_neg = h_in_B[:, 1]

    muA_pos_scale = _conditional_div(state.mu_A[:, 0], gamma_A * h_out_A[:, 0])
    muA_neg_scale = _conditional_div(state.mu_A[:, 1], gamma_A * h_out_A[:, 1])
    muB_pos_scale = _conditional_div(state.mu_B[:, 0], gamma_B * h_out_B[:, 0])
    muB_neg_scale = _conditional_div(state.mu_B[:, 1], gamma_B * h_out_B[:, 1])

    new_mu_A_pos = hA_pos * (
        muA_pos_scale @ WA_pos + muA_neg_scale @ WA_neg
    )
    new_mu_A_neg = hA_neg * (
        muA_pos_scale @ WA_neg + muA_neg_scale @ WA_pos
    )
    new_mu_B_pos = hB_pos * (
        muB_pos_scale @ WB_pos + muB_neg_scale @ WB_neg
    )
    new_mu_B_neg = hB_neg * (
        muB_pos_scale @ WB_neg + muB_neg_scale @ WB_pos
    )

    sqrt_W_pos = torch.sqrt((WA_pos * WB_pos).clamp(min=0))
    sqrt_W_neg = torch.sqrt((WA_neg * WB_neg).clamp(min=0))
    sqrt_gamma = torch.sqrt((gamma_A * gamma_B).clamp(min=0))
    sqrt_h_out_pos = torch.sqrt((h_out_A[:, 0] * h_out_B[:, 0]).clamp(min=0))
    sqrt_h_out_neg = torch.sqrt((h_out_A[:, 1] * h_out_B[:, 1]).clamp(min=0))
    beta_pos_scale = _conditional_div(state.beta[:, 0], sqrt_gamma * sqrt_h_out_pos)
    beta_neg_scale = _conditional_div(state.beta[:, 1], sqrt_gamma * sqrt_h_out_neg)
    sqrt_h_in_pos = torch.sqrt((hA_pos * hB_pos).clamp(min=0))
    sqrt_h_in_neg = torch.sqrt((hA_neg * hB_neg).clamp(min=0))
    new_beta_pos = sqrt_h_in_pos * (
        beta_pos_scale @ sqrt_W_pos + beta_neg_scale @ sqrt_W_neg
    )
    new_beta_neg = sqrt_h_in_neg * (
        beta_pos_scale @ sqrt_W_neg + beta_neg_scale @ sqrt_W_pos
    )

    return _GameState(
        torch.stack([new_mu_A_pos, new_mu_A_neg], dim=1),
        torch.stack([new_mu_B_pos, new_mu_B_neg], dim=1),
        torch.stack([new_beta_pos, new_beta_neg], dim=1),
        sign_augmented=state.sign_augmented,
    )


def _adf_backward_conv2d_conditioned(
    module_A: nn.Conv2d,
    module_B: nn.Conv2d,
    in_A: Tensor,
    in_B: Tensor,
    state: _GameState,
    h_in_A: Tensor,
    h_in_B: Tensor,
    h_out_A: Tensor,
    h_out_B: Tensor,
) -> _GameState:
    W_A = module_A.weight.detach()
    W_B = module_B.weight.detach()
    WA_pos = F.relu(W_A)
    WA_neg = F.relu(-W_A)
    WB_pos = F.relu(W_B)
    WB_neg = F.relu(-W_B)
    abs_WA = W_A.abs()
    abs_WB = W_B.abs()

    in_shape_A = (in_A.size(-2), in_A.size(-1))
    in_shape_B = (in_B.size(-2), in_B.size(-1))
    out_shape = (state.mu_A.size(-2), state.mu_A.size(-1))
    op_A = _conv2d_output_padding(in_shape_A, out_shape, module_A)
    op_B = _conv2d_output_padding(in_shape_B, out_shape, module_B)

    def _conv(x: Tensor, w: Tensor, mod: nn.Conv2d) -> Tensor:
        return F.conv2d(
            x,
            w,
            bias=None,
            stride=mod.stride,
            padding=mod.padding,
            dilation=mod.dilation,
            groups=mod.groups,
        )

    def _conv_T(x: Tensor, w: Tensor, mod: nn.Conv2d, op: Tuple[int, int]) -> Tensor:
        return F.conv_transpose2d(
            x,
            w,
            bias=None,
            stride=mod.stride,
            padding=mod.padding,
            output_padding=op,
            dilation=mod.dilation,
            groups=mod.groups,
        )

    gamma_A = _conv(torch.ones_like(in_A), abs_WA, module_A)
    gamma_B = _conv(torch.ones_like(in_B), abs_WB, module_B)
    if gamma_A.shape[-2:] != state.mu_A.shape[-2:]:
        gamma_A = gamma_A[..., :state.mu_A.size(-2), :state.mu_A.size(-1)]
    if gamma_B.shape[-2:] != state.mu_B.shape[-2:]:
        gamma_B = gamma_B[..., :state.mu_B.size(-2), :state.mu_B.size(-1)]

    target_h, target_w = in_A.size(-2), in_A.size(-1)

    def _crop(x: Tensor) -> Tensor:
        return x[..., :target_h, :target_w]

    muA_pos_scale = _conditional_div(state.mu_A[:, 0], gamma_A * h_out_A[:, 0])
    muA_neg_scale = _conditional_div(state.mu_A[:, 1], gamma_A * h_out_A[:, 1])
    muB_pos_scale = _conditional_div(state.mu_B[:, 0], gamma_B * h_out_B[:, 0])
    muB_neg_scale = _conditional_div(state.mu_B[:, 1], gamma_B * h_out_B[:, 1])

    new_mu_A_pos = h_in_A[:, 0] * (
        _crop(_conv_T(muA_pos_scale, WA_pos, module_A, op_A))
        + _crop(_conv_T(muA_neg_scale, WA_neg, module_A, op_A))
    )
    new_mu_A_neg = h_in_A[:, 1] * (
        _crop(_conv_T(muA_pos_scale, WA_neg, module_A, op_A))
        + _crop(_conv_T(muA_neg_scale, WA_pos, module_A, op_A))
    )
    new_mu_B_pos = h_in_B[:, 0] * (
        _crop(_conv_T(muB_pos_scale, WB_pos, module_B, op_B))
        + _crop(_conv_T(muB_neg_scale, WB_neg, module_B, op_B))
    )
    new_mu_B_neg = h_in_B[:, 1] * (
        _crop(_conv_T(muB_pos_scale, WB_neg, module_B, op_B))
        + _crop(_conv_T(muB_neg_scale, WB_pos, module_B, op_B))
    )

    sqrt_gamma = torch.sqrt((gamma_A * gamma_B).clamp(min=0))
    sqrt_W_pos = torch.sqrt((WA_pos * WB_pos).clamp(min=0))
    sqrt_W_neg = torch.sqrt((WA_neg * WB_neg).clamp(min=0))
    sqrt_h_out_pos = torch.sqrt((h_out_A[:, 0] * h_out_B[:, 0]).clamp(min=0))
    sqrt_h_out_neg = torch.sqrt((h_out_A[:, 1] * h_out_B[:, 1]).clamp(min=0))
    beta_pos_scale = _conditional_div(state.beta[:, 0], sqrt_gamma * sqrt_h_out_pos)
    beta_neg_scale = _conditional_div(state.beta[:, 1], sqrt_gamma * sqrt_h_out_neg)
    sqrt_h_in_pos = torch.sqrt((h_in_A[:, 0] * h_in_B[:, 0]).clamp(min=0))
    sqrt_h_in_neg = torch.sqrt((h_in_A[:, 1] * h_in_B[:, 1]).clamp(min=0))

    new_beta_pos = sqrt_h_in_pos * (
        _crop(_conv_T(beta_pos_scale, sqrt_W_pos, module_A, op_A))
        + _crop(_conv_T(beta_neg_scale, sqrt_W_neg, module_A, op_A))
    )
    new_beta_neg = sqrt_h_in_neg * (
        _crop(_conv_T(beta_pos_scale, sqrt_W_neg, module_A, op_A))
        + _crop(_conv_T(beta_neg_scale, sqrt_W_pos, module_A, op_A))
    )

    return _GameState(
        torch.stack([new_mu_A_pos, new_mu_A_neg], dim=1),
        torch.stack([new_mu_B_pos, new_mu_B_neg], dim=1),
        torch.stack([new_beta_pos, new_beta_neg], dim=1),
        sign_augmented=state.sign_augmented,
    )


def _adf_backward_avgpool2d_conditioned(
    module: nn.AvgPool2d,
    state: _GameState,
    h_in_A: Tensor,
    h_in_B: Tensor,
    h_out_A: Tensor,
    h_out_B: Tensor,
) -> _GameState:
    kwargs = dict(
        kernel_size=module.kernel_size,
        stride=module.stride if module.stride is not None else module.kernel_size,
        padding=module.padding,
        ceil_mode=module.ceil_mode,
        count_include_pad=module.count_include_pad,
        divisor_override=module.divisor_override,
    )

    def _grad_input(g: Tensor, ref: Tensor) -> Tensor:
        a = ref.detach().clone().requires_grad_(True)
        out = F.avg_pool2d(a, **kwargs)
        out.backward(g)
        return a.grad if a.grad is not None else torch.zeros_like(ref)

    new_mu_A = torch.stack([
        h_in_A[:, s] * _grad_input(_conditional_div(state.mu_A[:, s], h_out_A[:, s]), h_in_A[:, s])
        for s in range(2)
    ], dim=1)
    new_mu_B = torch.stack([
        h_in_B[:, s] * _grad_input(_conditional_div(state.mu_B[:, s], h_out_B[:, s]), h_in_B[:, s])
        for s in range(2)
    ], dim=1)
    sqrt_h_in = torch.sqrt((h_in_A * h_in_B).clamp(min=0))
    sqrt_h_out = torch.sqrt((h_out_A * h_out_B).clamp(min=0))
    new_beta = torch.stack([
        sqrt_h_in[:, s] * _grad_input(_conditional_div(state.beta[:, s], sqrt_h_out[:, s]), sqrt_h_in[:, s])
        for s in range(2)
    ], dim=1)
    return _GameState(new_mu_A, new_mu_B, new_beta, sign_augmented=state.sign_augmented)


def _adf_backward_adaptive_avgpool2d_conditioned(
    module: nn.AdaptiveAvgPool2d,
    state: _GameState,
    h_in_A: Tensor,
    h_in_B: Tensor,
    h_out_A: Tensor,
    h_out_B: Tensor,
) -> _GameState:
    def _grad_input(g: Tensor, ref: Tensor) -> Tensor:
        a = ref.detach().clone().requires_grad_(True)
        out = F.adaptive_avg_pool2d(a, module.output_size)
        out.backward(g)
        return a.grad if a.grad is not None else torch.zeros_like(ref)

    new_mu_A = torch.stack([
        h_in_A[:, s] * _grad_input(_conditional_div(state.mu_A[:, s], h_out_A[:, s]), h_in_A[:, s])
        for s in range(2)
    ], dim=1)
    new_mu_B = torch.stack([
        h_in_B[:, s] * _grad_input(_conditional_div(state.mu_B[:, s], h_out_B[:, s]), h_in_B[:, s])
        for s in range(2)
    ], dim=1)
    sqrt_h_in = torch.sqrt((h_in_A * h_in_B).clamp(min=0))
    sqrt_h_out = torch.sqrt((h_out_A * h_out_B).clamp(min=0))
    new_beta = torch.stack([
        sqrt_h_in[:, s] * _grad_input(_conditional_div(state.beta[:, s], sqrt_h_out[:, s]), sqrt_h_in[:, s])
        for s in range(2)
    ], dim=1)
    return _GameState(new_mu_A, new_mu_B, new_beta, sign_augmented=state.sign_augmented)


def _adf_backward_add_conditioned(
    state: _GameState,
    h_left_A: Tensor,
    h_right_A: Tensor,
    h_left_B: Tensor,
    h_right_B: Tensor,
    h_out_A: Tensor,
    h_out_B: Tensor,
) -> Dict[int, _GameState]:
    half = 0.5
    left_mu_A = state.mu_A * _conditional_div(half * h_left_A, h_out_A)
    right_mu_A = state.mu_A * _conditional_div(half * h_right_A, h_out_A)
    left_mu_B = state.mu_B * _conditional_div(half * h_left_B, h_out_B)
    right_mu_B = state.mu_B * _conditional_div(half * h_right_B, h_out_B)
    sqrt_h_out = torch.sqrt((h_out_A * h_out_B).clamp(min=0))
    left_beta = state.beta * _conditional_div(
        half * torch.sqrt((h_left_A * h_left_B).clamp(min=0)),
        sqrt_h_out,
    )
    right_beta = state.beta * _conditional_div(
        half * torch.sqrt((h_right_A * h_right_B).clamp(min=0)),
        sqrt_h_out,
    )
    return {
        0: _GameState(left_mu_A, left_mu_B, left_beta, sign_augmented=state.sign_augmented),
        1: _GameState(right_mu_A, right_mu_B, right_beta, sign_augmented=state.sign_augmented),
    }


def _av_adf_propagate_conditioned(
    mu: Tensor,
    A: Tensor,
    h_in: Tensor,
    h_out: Tensor,
) -> Tensor:
    abs_A = A.abs()
    gamma = abs_A.sum(dim=-1).clamp(min=_EPS_DENOM)
    source_pos = _conditional_div(mu[:, 0], gamma.unsqueeze(-1) * h_out[:, 0])
    source_neg = _conditional_div(mu[:, 1], gamma.unsqueeze(-1) * h_out[:, 1])
    out_pos = h_in[:, 0] * torch.einsum('bhqd,bhqk->bhkd', source_pos, abs_A)
    out_neg = h_in[:, 1] * torch.einsum('bhqd,bhqk->bhkd', source_neg, abs_A)
    return torch.stack([out_pos, out_neg], dim=1)


def _av_adf_propagate_beta_conditioned(
    beta: Tensor,
    A_A: Tensor,
    A_B: Tensor,
    h_in_A: Tensor,
    h_in_B: Tensor,
    h_out_A: Tensor,
    h_out_B: Tensor,
) -> Tensor:
    gamma_A = A_A.abs().sum(dim=-1).clamp(min=_EPS_DENOM)
    gamma_B = A_B.abs().sum(dim=-1).clamp(min=_EPS_DENOM)
    sqrt_gamma = torch.sqrt((gamma_A * gamma_B).clamp(min=0))
    sqrt_abs_A = torch.sqrt((A_A.abs() * A_B.abs()).clamp(min=0))
    sqrt_h_out_pos = torch.sqrt((h_out_A[:, 0] * h_out_B[:, 0]).clamp(min=0))
    sqrt_h_out_neg = torch.sqrt((h_out_A[:, 1] * h_out_B[:, 1]).clamp(min=0))
    beta_pos = _conditional_div(beta[:, 0], sqrt_gamma.unsqueeze(-1) * sqrt_h_out_pos)
    beta_neg = _conditional_div(beta[:, 1], sqrt_gamma.unsqueeze(-1) * sqrt_h_out_neg)
    sqrt_h_in_pos = torch.sqrt((h_in_A[:, 0] * h_in_B[:, 0]).clamp(min=0))
    sqrt_h_in_neg = torch.sqrt((h_in_A[:, 1] * h_in_B[:, 1]).clamp(min=0))
    out_pos = sqrt_h_in_pos * torch.einsum('bhqd,bhqk->bhkd', beta_pos, sqrt_abs_A)
    out_neg = sqrt_h_in_neg * torch.einsum('bhqd,bhqk->bhkd', beta_neg, sqrt_abs_A)
    return torch.stack([out_pos, out_neg], dim=1)


# ---------------------------------------------------------------------------
# HellingerDiagnostic class
# ---------------------------------------------------------------------------

class HellingerDiagnostic:
    """Compute the per-module / per-pixel Hellinger disagreement map under
    one of the two games.

    Parameters:
        model_A: first PyTorch model.
        model_B: second PyTorch model (defaults to model_A).
        game:    'adf' for the ADF game (sign-split weights-only kernel), or 'rg' for
                 the Active Routing Game (activation-state split plus
                 branchwise Gibbs kernels).
        tau:     temperature for the RG game (ignored for ADF).
        condition_on_survival: replace the ordinary cemetery-augmented
                 path law by the per-model path law conditioned on
                 eventual arrival at an ordinary input state.  For
                 ``game='adf'`` this uses the conditioned kernel sweep;
                 for ``game='rg'`` it applies the exact post-hoc input
                 normalization ``BC_cond = BC_ord / sqrt(Z_A Z_B)``.
                 Defaults to ``False`` (the unconditioned distance).

    For the RG game an attention product and its Value projection form one
    composite step (the merged rule, see :func:`_av_backward_merged`): the
    stream chosen at the attention output is kept through the Value entry
    down to the projection input (one 1/2 split, one player switch).  The
    Value operand must be a layout view of an ``nn.Linear`` output.

    Usage:
        diag = HellingerDiagnostic(model_A, model_B, game='adf')
        result = diag.compute(input_A, input_B, target=None)

    Result keys:
        hellinger_distance, h_squared, bhattacharyya,
        per_pixel_h2, per_module_h2, cemetery_h2,
        mu_A_input, mu_B_input, beta_input, target.
    """

    def __init__(
        self,
        model_A: nn.Module,
        model_B: Optional[nn.Module] = None,
        game: str = "adf",
        tau: float = 1.0,
        condition_on_survival: bool = False,
    ):
        if game not in ("adf", "rg"):
            raise ValueError(f"game must be 'adf' or 'rg', got {game!r}")
        self.model_A = model_A
        self.model_B = model_B if model_B is not None else model_A
        self.game = game
        self.tau = float(tau)
        self.condition_on_survival = bool(condition_on_survival)
        # Populated by _link_value_projections before an RG backward sweep:
        # attention-product call -> Value projection calls (A, B), and the
        # set of Value projection calls.
        self._value_links: Dict[int, Tuple[_CallRecord, _CallRecord]] = {}
        self._value_projections: set = set()
        # Replace functional ops (torch.flatten, torch.reshape, F.softmax,
        # …) with nn.Module equivalents so the cache hooks below see them.
        # Without this, VGG's torch.flatten between the conv backbone and
        # the classifier head produces an unhooked tensor that breaks the
        # backward sweep at the flatten boundary.  routing_game.py uses the
        # same idiom for the same reason.
        #
        # ``replace_functional_with_modules`` runs ``fx.symbolic_trace``,
        # which executes each submodule's forward with Proxy tensors.  If
        # the model has been DC-patched, that routes into the patched
        # forwards (which expect ``[4*batch]`` tensors) and crashes on
        # the symbolic proxies.  Wrap the call in ``dc_disabled`` so
        # tracing sees the ORIGINAL forwards.  No-op for unpatched models.
        try:
            from bwgame.patcher import dc_disabled as _dc_disabled
        except Exception:
            from contextlib import nullcontext

            def _dc_disabled(_m):  # type: ignore[no-redef]
                return nullcontext()
        try:
            from bwgame.functional_replacer import replace_functional_with_modules
            with _dc_disabled(self.model_A):
                replace_functional_with_modules(self.model_A, inplace=True)
            if self.model_B is not self.model_A:
                with _dc_disabled(self.model_B):
                    replace_functional_with_modules(self.model_B, inplace=True)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "HellingerDiagnostic: functional-replacer failed (%s); "
                "falling back to shape-based reshape recovery.", exc,
            )
        # Always use independent caches.
        self._cache_A = _HellingerActivationCache(self.model_A)
        self._cache_B = _HellingerActivationCache(self.model_B)

    def _resolve_adf_survival(
        self,
        tensor: Tensor,
        survival: Dict[int, Tensor],
    ) -> Optional[Tensor]:
        tid = id(tensor)
        if tid in survival:
            return survival[tid]

        target_shape = tuple(tensor.shape)
        target_numel = int(tensor[0].numel()) if tensor.dim() > 0 else int(tensor.numel())

        slice_candidates: List[Tuple[int, int, Tensor]] = []
        for order, h in enumerate(list(survival.values())):
            resolved = _try_slice_extract_adf_survival(h, target_shape)
            if resolved is not None:
                slice_candidates.append((_adf_survival_numel_per_batch(h), order, resolved))
        if slice_candidates:
            _, _, resolved = max(slice_candidates, key=lambda item: (item[0], item[1]))
            survival[tid] = resolved
            return resolved

        for h in list(survival.values()):
            if _adf_survival_numel_per_batch(h) == target_numel:
                resolved = h if tuple(h.shape[2:]) == target_shape[1:] else _reshape_adf_survival(h, target_shape)
                survival[tid] = resolved
                return resolved
        return None

    def _resolve_adf_survival_or_zero(
        self,
        tensor: Tensor,
        survival: Dict[int, Tensor],
    ) -> Tensor:
        h = self._resolve_adf_survival(tensor, survival)
        if h is None:
            h = _zero_adf_survival_like(tensor)
            survival[id(tensor)] = h
        return h

    def _compute_adf_survival_maps(
        self,
        calls: List[_CallRecord],
        model_input: Tensor,
    ) -> Dict[int, Tensor]:
        survival: Dict[int, Tensor] = {
            id(model_input): _adf_survival_from_input(model_input),
        }

        for rec in calls:
            mod = rec.module
            h_out: Optional[Tensor] = None

            if _HAS_DCADD and isinstance(mod, _DCAdd):
                if len(rec.inputs) < 2:
                    raise RuntimeError(f"ADF survival pass needs two Add operands for {rec.name}")
                left = rec.inputs[0]
                right = rec.inputs[1]
                if not isinstance(left, Tensor) or not isinstance(right, Tensor):
                    raise RuntimeError(f"ADF survival pass expected tensor Add operands for {rec.name}")
                h_left = self._resolve_adf_survival_or_zero(left, survival)
                h_right = self._resolve_adf_survival_or_zero(right, survival)
                h_out = 0.5 * (h_left + h_right)

            elif isinstance(mod, nn.Linear):
                if not rec.inputs or not isinstance(rec.inputs[0], Tensor):
                    raise RuntimeError(f"ADF survival pass expected Linear tensor input for {rec.name}")
                h_in = self._resolve_adf_survival(rec.inputs[0], survival)
                if h_in is None:
                    raise RuntimeError(f"ADF survival pass could not resolve Linear input for {rec.name}")
                h_out = _adf_linear_survival(h_in, mod.weight.detach())

            elif isinstance(mod, nn.Conv2d):
                if not rec.inputs or not isinstance(rec.inputs[0], Tensor) or not isinstance(rec.output, Tensor):
                    raise RuntimeError(f"ADF survival pass expected Conv2d tensors for {rec.name}")
                h_in = self._resolve_adf_survival(rec.inputs[0], survival)
                if h_in is None:
                    raise RuntimeError(f"ADF survival pass could not resolve Conv2d input for {rec.name}")
                h_out = _adf_conv2d_survival(mod, h_in, rec.inputs[0], rec.output)

            elif isinstance(mod, _GATED_ACTIVATION_TYPES):
                if not rec.inputs or not isinstance(rec.inputs[0], Tensor):
                    raise RuntimeError(f"ADF survival pass expected activation tensor input for {rec.name}")
                h_in = self._resolve_adf_survival(rec.inputs[0], survival)
                if h_in is None:
                    raise RuntimeError(f"ADF survival pass could not resolve activation input for {rec.name}")
                h_out = _adf_activation_survival(mod, rec.inputs[0], h_in)

            elif isinstance(mod, _IDENTITY_TYPES):
                if not rec.inputs or not isinstance(rec.inputs[0], Tensor):
                    raise RuntimeError(f"ADF survival pass expected identity tensor input for {rec.name}")
                h_out = self._resolve_adf_survival(rec.inputs[0], survival)

            elif isinstance(mod, nn.Flatten):
                if not rec.inputs or not isinstance(rec.inputs[0], Tensor) or not isinstance(rec.output, Tensor):
                    raise RuntimeError(f"ADF survival pass expected Flatten tensors for {rec.name}")
                h_in = self._resolve_adf_survival(rec.inputs[0], survival)
                if h_in is None:
                    raise RuntimeError(f"ADF survival pass could not resolve Flatten input for {rec.name}")
                h_out = _reshape_adf_survival(h_in, tuple(rec.output.shape))

            elif _HAS_DCSHAPE and isinstance(mod, (_DCReshape, _DCView, _DCSqueeze, _DCUnsqueeze)):
                if not rec.inputs or not isinstance(rec.inputs[0], Tensor) or not isinstance(rec.output, Tensor):
                    raise RuntimeError(f"ADF survival pass expected reshape tensors for {rec.name}")
                h_in = self._resolve_adf_survival(rec.inputs[0], survival)
                if h_in is None:
                    raise RuntimeError(f"ADF survival pass could not resolve reshape input for {rec.name}")
                h_out = _reshape_adf_survival(h_in, tuple(rec.output.shape))

            elif _HAS_DCSHAPE and isinstance(mod, _DCTranspose):
                if not rec.inputs or not isinstance(rec.inputs[0], Tensor):
                    raise RuntimeError(f"ADF survival pass expected transpose tensor input for {rec.name}")
                h_in = self._resolve_adf_survival(rec.inputs[0], survival)
                if h_in is None:
                    raise RuntimeError(f"ADF survival pass could not resolve transpose input for {rec.name}")
                h_out = _backward_transpose(mod, _GameState(h_in, h_in, h_in, sign_augmented=True)).mu_A

            elif _HAS_DCSHAPE and isinstance(mod, _DCPermute):
                if not rec.inputs or not isinstance(rec.inputs[0], Tensor):
                    raise RuntimeError(f"ADF survival pass expected permute tensor input for {rec.name}")
                h_in = self._resolve_adf_survival(rec.inputs[0], survival)
                if h_in is None:
                    raise RuntimeError(f"ADF survival pass could not resolve permute input for {rec.name}")
                dims = list(mod.dims)
                state_perm: List[int] = []
                for d in dims:
                    state_perm.append(d if d == 0 else d + 1)
                batch_pos = state_perm.index(0)
                state_perm.insert(batch_pos + 1, 1)
                h_out = h_in.permute(state_perm).contiguous()

            elif _HAS_DCCAT and isinstance(mod, _DCCat):
                tensor_list = rec.inputs[0] if rec.inputs else []
                if not isinstance(tensor_list, (list, tuple)) or not tensor_list:
                    raise RuntimeError(f"ADF survival pass expected DCCat input list for {rec.name}")
                h_inputs: List[Tensor] = []
                for t in tensor_list:
                    if not isinstance(t, Tensor):
                        raise RuntimeError(f"ADF survival pass expected DCCat tensor input for {rec.name}")
                    h_t = self._resolve_adf_survival_or_zero(t, survival)
                    h_inputs.append(h_t)
                h_out = _adf_dccat_survival(mod, list(tensor_list), h_inputs)

            elif isinstance(mod, nn.AvgPool2d):
                if not rec.inputs or not isinstance(rec.inputs[0], Tensor):
                    raise RuntimeError(f"ADF survival pass expected AvgPool2d tensor input for {rec.name}")
                h_in = self._resolve_adf_survival(rec.inputs[0], survival)
                if h_in is None:
                    raise RuntimeError(f"ADF survival pass could not resolve AvgPool2d input for {rec.name}")
                h_out = _adf_avgpool2d_survival(mod, h_in)

            elif isinstance(mod, nn.AdaptiveAvgPool2d):
                if not rec.inputs or not isinstance(rec.inputs[0], Tensor):
                    raise RuntimeError(f"ADF survival pass expected AdaptiveAvgPool2d tensor input for {rec.name}")
                h_in = self._resolve_adf_survival(rec.inputs[0], survival)
                if h_in is None:
                    raise RuntimeError(f"ADF survival pass could not resolve AdaptiveAvgPool2d input for {rec.name}")
                h_out = _adf_adaptive_avgpool2d_survival(mod, h_in)

            elif isinstance(mod, nn.MaxPool2d):
                if not rec.inputs or not isinstance(rec.inputs[0], Tensor):
                    raise RuntimeError(f"ADF survival pass expected MaxPool2d tensor input for {rec.name}")
                h_in = self._resolve_adf_survival(rec.inputs[0], survival)
                if h_in is None:
                    raise RuntimeError(f"ADF survival pass could not resolve MaxPool2d input for {rec.name}")
                h_out = _adf_maxpool2d_survival(mod, h_in, rec.inputs[0])

            elif _HAS_DCMATMUL and isinstance(mod, DCMatMul):
                if _is_av_matmul(rec):
                    if len(rec.inputs) < 2 or not isinstance(rec.inputs[1], Tensor):
                        raise RuntimeError(f"ADF survival pass expected AV matmul V input for {rec.name}")
                    h_in = self._resolve_adf_survival(rec.inputs[1], survival)
                    if h_in is None:
                        raise RuntimeError(f"ADF survival pass could not resolve AV matmul input for {rec.name}")
                    h_out = _adf_av_survival(h_in, rec.inputs[0])

            elif _HAS_DCMUL and isinstance(mod, DCMul):
                if not rec.inputs or not isinstance(rec.inputs[0], Tensor):
                    raise RuntimeError(f"ADF survival pass expected DCMul tensor input for {rec.name}")
                h_out = self._resolve_adf_survival(rec.inputs[0], survival)

            if h_out is not None and isinstance(rec.output, Tensor):
                survival[id(rec.output)] = h_out.contiguous()

        return survival

    def _dispatch_call_backward_conditioned_adf(
        self,
        rec_A: _CallRecord,
        rec_B: _CallRecord,
        state: _GameState,
        survival_A: Dict[int, Tensor],
        survival_B: Dict[int, Tensor],
    ) -> Dict[int, _GameState]:
        mod_A = rec_A.module
        mod_B = rec_B.module
        in_A = rec_A.inputs[0] if rec_A.inputs else None
        in_B = rec_B.inputs[0] if rec_B.inputs else None

        h_out_A = survival_A.get(id(rec_A.output))
        h_out_B = survival_B.get(id(rec_B.output))
        if h_out_A is None or h_out_B is None:
            raise RuntimeError(f"Missing conditioned-survival state at output of {rec_A.name}")
        state = _mask_adf_conditioned_state(state, h_out_A, h_out_B)

        if _HAS_DCADD and isinstance(mod_A, _DCAdd):
            if len(rec_A.inputs) < 2 or len(rec_B.inputs) < 2:
                raise RuntimeError(f"Conditioned Add dispatch needs both operands for {rec_A.name}")
            left_A = rec_A.inputs[0]; right_A = rec_A.inputs[1]
            left_B = rec_B.inputs[0]; right_B = rec_B.inputs[1]
            if not all(isinstance(t, Tensor) for t in (left_A, right_A, left_B, right_B)):
                raise RuntimeError(f"Conditioned Add dispatch expected tensor operands for {rec_A.name}")
            h_left_A = survival_A.get(id(left_A))
            h_right_A = survival_A.get(id(right_A))
            h_left_B = survival_B.get(id(left_B))
            h_right_B = survival_B.get(id(right_B))
            if any(h is None for h in (h_left_A, h_right_A, h_left_B, h_right_B)):
                raise RuntimeError(f"Missing conditioned-survival branch state for {rec_A.name}")
            return _adf_backward_add_conditioned(
                state, h_left_A, h_right_A, h_left_B, h_right_B, h_out_A, h_out_B,
            )

        if isinstance(mod_A, nn.Linear):
            if not isinstance(in_A, Tensor) or not isinstance(in_B, Tensor):
                raise RuntimeError(f"Conditioned Linear dispatch expected tensor inputs for {rec_A.name}")
            h_in_A = survival_A.get(id(in_A))
            h_in_B = survival_B.get(id(in_B))
            if h_in_A is None or h_in_B is None:
                raise RuntimeError(f"Missing conditioned-survival input state for {rec_A.name}")
            return {
                0: _adf_backward_linear_conditioned(
                    mod_A, mod_B, state, h_in_A, h_in_B, h_out_A, h_out_B,
                )
            }

        if isinstance(mod_A, nn.Conv2d):
            if not isinstance(in_A, Tensor) or not isinstance(in_B, Tensor):
                raise RuntimeError(f"Conditioned Conv2d dispatch expected tensor inputs for {rec_A.name}")
            h_in_A = survival_A.get(id(in_A))
            h_in_B = survival_B.get(id(in_B))
            if h_in_A is None or h_in_B is None:
                raise RuntimeError(f"Missing conditioned-survival input state for {rec_A.name}")
            return {
                0: _adf_backward_conv2d_conditioned(
                    mod_A, mod_B, in_A, in_B, state, h_in_A, h_in_B, h_out_A, h_out_B,
                )
            }

        if isinstance(mod_A, nn.AvgPool2d):
            if not isinstance(in_A, Tensor) or not isinstance(in_B, Tensor):
                raise RuntimeError(f"Conditioned AvgPool2d dispatch expected tensor inputs for {rec_A.name}")
            h_in_A = survival_A.get(id(in_A))
            h_in_B = survival_B.get(id(in_B))
            if h_in_A is None or h_in_B is None:
                raise RuntimeError(f"Missing conditioned-survival input state for {rec_A.name}")
            return {
                0: _adf_backward_avgpool2d_conditioned(
                    mod_A, state, h_in_A, h_in_B, h_out_A, h_out_B,
                )
            }

        if isinstance(mod_A, nn.AdaptiveAvgPool2d):
            if not isinstance(in_A, Tensor) or not isinstance(in_B, Tensor):
                raise RuntimeError(f"Conditioned AdaptiveAvgPool2d dispatch expected tensor inputs for {rec_A.name}")
            h_in_A = survival_A.get(id(in_A))
            h_in_B = survival_B.get(id(in_B))
            if h_in_A is None or h_in_B is None:
                raise RuntimeError(f"Missing conditioned-survival input state for {rec_A.name}")
            return {
                0: _adf_backward_adaptive_avgpool2d_conditioned(
                    mod_A, state, h_in_A, h_in_B, h_out_A, h_out_B,
                )
            }

        if isinstance(mod_A, _GATED_ACTIVATION_TYPES):
            return {0: _backward_identity(state)}

        if _HAS_DCMATMUL and isinstance(mod_A, DCMatMul):
            if _is_av_matmul(rec_A):
                if len(rec_A.inputs) < 2 or len(rec_B.inputs) < 2:
                    raise RuntimeError(f"Conditioned AV matmul dispatch needs both operands for {rec_A.name}")
                V_A = rec_A.inputs[1]
                V_B = rec_B.inputs[1]
                if not isinstance(V_A, Tensor) or not isinstance(V_B, Tensor):
                    raise RuntimeError(f"Conditioned AV matmul dispatch expected tensor V inputs for {rec_A.name}")
                h_in_A = survival_A.get(id(V_A))
                h_in_B = survival_B.get(id(V_B))
                if h_in_A is None or h_in_B is None:
                    raise RuntimeError(f"Missing conditioned-survival AV input state for {rec_A.name}")
                return {
                    1: _GameState(
                        _av_adf_propagate_conditioned(state.mu_A, rec_A.inputs[0], h_in_A, h_out_A),
                        _av_adf_propagate_conditioned(state.mu_B, rec_B.inputs[0], h_in_B, h_out_B),
                        _av_adf_propagate_beta_conditioned(
                            state.beta,
                            rec_A.inputs[0],
                            rec_B.inputs[0],
                            h_in_A,
                            h_in_B,
                            h_out_A,
                            h_out_B,
                        ),
                        sign_augmented=state.sign_augmented,
                    )
                }
            return {}

        return self._dispatch_call_backward(rec_A, rec_B, state, _CemeteryState.zeros(state.mu_A.size(0), state.mu_A.device))

    def compute(
        self,
        input_A: Tensor,
        input_B: Optional[Tensor] = None,
        target: Optional[Union[int, Tensor]] = None,
    ) -> Dict[str, Any]:
        if input_B is None:
            input_B = input_A

        # If the models have been DC-patched (operations/patch_builder),
        # the patched forwards expect ``[4*batch]`` DC-decomposed tensors
        # and will crash on a raw single-batch image (see
        # ``operations/maxpool.py::DCMaxPool2dFunction.forward``).  The
        # Hellinger diagnostic only needs the ORIGINAL forwards — it
        # captures module I/O via hooks and runs its own backward sweep.
        # Temporarily disable DC so the patched model dispatches to the
        # original forwards.  This is a no-op for unpatched models.
        try:
            from bwgame.patcher import dc_disabled
        except Exception:
            from contextlib import nullcontext

            def dc_disabled(_m):  # type: ignore[no-redef]
                return nullcontext()

        # === Forward pass A ===
        self._cache_A.clear()
        self._cache_A.install()
        try:
            with dc_disabled(self.model_A), torch.no_grad():
                out_A = self.model_A(input_A)
        finally:
            self._cache_A.remove()

        # === Forward pass B ===
        self._cache_B.clear()
        self._cache_B.install()
        try:
            with dc_disabled(self.model_B), torch.no_grad():
                out_B = self.model_B(input_B)
        finally:
            self._cache_B.remove()

        calls_A = self._cache_A.calls
        calls_B = self._cache_B.calls
        if len(calls_A) != len(calls_B):
            raise RuntimeError(
                f"Forward call counts differ between A and B: "
                f"{len(calls_A)} vs {len(calls_B)}. "
                f"Both models must share the same forward graph."
            )

        # === Determine target neuron ===
        if target is None:
            target = out_A.argmax(dim=-1)
        if isinstance(target, int):
            target = torch.full((out_A.size(0),), target, dtype=torch.long, device=out_A.device)
        if not isinstance(target, Tensor):
            target = torch.as_tensor(target, dtype=torch.long, device=out_A.device)

        survival_A_target: Optional[Tensor] = None
        survival_B_target: Optional[Tensor] = None
        survival_maps_A: Optional[Dict[int, Tensor]] = None
        survival_maps_B: Optional[Dict[int, Tensor]] = None
        if self.condition_on_survival and self.game == "adf":
            survival_maps_A = self._compute_adf_survival_maps(calls_A, input_A)
            survival_maps_B = self._compute_adf_survival_maps(calls_B, input_B)
            h_out_A = survival_maps_A.get(id(out_A))
            h_out_B = survival_maps_B.get(id(out_B))
            if h_out_A is None or h_out_B is None:
                raise RuntimeError("Conditioned ADF survival pass did not reach the model output")
            batch_idx = torch.arange(out_A.size(0), device=out_A.device)
            survival_A_target = h_out_A[batch_idx, 0, target]
            survival_B_target = h_out_B[batch_idx, 0, target]
            bad = (survival_A_target <= _EPS_DENOM) | (survival_B_target <= _EPS_DENOM)
            if bad.any():
                bad_idx = torch.nonzero(bad, as_tuple=False).flatten().tolist()
                raise ValueError(
                    "condition_on_survival=True is undefined when the selected "
                    f"output target has zero survival mass; batch indices {bad_idx}"
                )

        try:
            if self.game == "rg":
                self._link_value_projections(calls_A, calls_B)
            return self._run_backward_sweep(
                input_A, input_B, out_A, out_B, calls_A, calls_B, target,
                survival_A_target, survival_B_target,
                survival_maps_A, survival_maps_B,
            )
        finally:
            self._value_links = {}
            self._value_projections = set()

    def _link_value_projections(
        self,
        calls_A: List[_CallRecord],
        calls_B: List[_CallRecord],
    ) -> None:
        """Pair every attention product with the Value projection it reads.

        Used by the RG game.  The Value operand of each attention product
        must be a view of an ``nn.Linear`` output reached through
        layout-only steps, in both models; otherwise the composite step is
        undefined and a ``ValueError`` is raised before the sweep starts.
        """
        self._value_links = {}
        self._value_projections = set()
        for idx, rec_A in enumerate(calls_A):
            if not (_HAS_DCMATMUL and isinstance(rec_A.module, DCMatMul) and _is_av_matmul(rec_A)):
                continue
            rec_B = calls_B[idx]
            j = _find_value_projection(calls_A, idx)
            ok = j is not None and isinstance(calls_B[j].module, nn.Linear)
            if ok:
                for av, vp in ((rec_A, calls_A[j]), (rec_B, calls_B[j])):
                    V = av.inputs[1] if len(av.inputs) > 1 else None
                    ok = (
                        ok
                        and isinstance(V, Tensor)
                        and bool(vp.inputs)
                        and isinstance(vp.inputs[0], Tensor)
                        and isinstance(vp.output, Tensor)
                        and _is_value_view(V, vp.output)
                    )
            if not ok:
                raise ValueError(
                    f"RG attention: the Value operand of {rec_A.name} is not a "
                    "layout view of an nn.Linear output"
                )
            self._value_links[id(rec_A)] = (calls_A[j], calls_B[j])
            self._value_projections.add(id(calls_A[j]))

    def _run_backward_sweep(
        self,
        input_A, input_B, out_A, out_B, calls_A, calls_B, target,
        survival_A_target, survival_B_target,
        survival_maps_A, survival_maps_B,
    ) -> Dict[str, Any]:
        # === Initialize state at output layer ===
        initial_state = _GameState.initial(
            out_A,
            target,
            sign_augmented=(self.game == "adf"),
        )

        # === Tensor-id-keyed backward sweep ===
        # The state dict is keyed by id() of A-side tensors.  When a call
        # routes mass to multiple inputs (Add: half to each operand), each
        # input id receives its own entry; multiple producers writing to
        # the same id accumulate via _GameState.add.
        #
        # Functional reshape/view/flatten ops between two hooked Modules
        # (e.g.\ ``torch.flatten(x, 1)`` between VGG's avgpool and the
        # classifier head) create tensors that are not present in the
        # cache, breaking strict id-based lookup.  We therefore allow a
        # *shape-fallback*: if the current call's output id is not in the
        # state dict, a stranded state with the same per-batch element
        # count is taken and reshaped to the call's output shape before
        # dispatch.  This recovers the chain across any pure
        # reshape/view/flatten boundary.
        #
        # Only stranded states qualify.  A state is pending, and left for
        # a later visit, while a call still to be visited produces its
        # tensor or a tensor sharing its storage (a functional view such
        # as the ``unflatten`` between a Value projection and its head
        # transpose): that call takes it by the exact id or as a view of
        # its output.  States at the model input are the result of the
        # sweep and never qualify.  Without this rule a call with an
        # equal-sized output (e.g. the key transpose of an attention
        # block, whose output has as many entries per sample as the
        # residual stream and the Value tensor) would take the state of
        # another branch.  Among the qualifying states, one sharing the
        # storage of the call's output is preferred.
        states: Dict[int, _GameState] = {id(out_A): initial_state}
        cemetery = _CemeteryState.zeros(out_A.size(0), out_A.device)

        per_module_h2: "OrderedDict[str, Tensor]" = OrderedDict()

        def _per_batch_numel(t: Tensor) -> int:
            return int(t[0].numel()) if t.dim() > 0 else int(t.numel())

        def _state_per_batch_numel(s: _GameState) -> int:
            return s.ordinary_numel_per_batch()

        def _storage_key(t: Tensor) -> int:
            return t.untyped_storage().data_ptr()

        first_producer: Dict[int, int] = {}
        first_producer_storage: Dict[int, int] = {}
        for idx, rec in enumerate(calls_A):
            if isinstance(rec.output, Tensor):
                first_producer.setdefault(id(rec.output), idx)
                first_producer_storage.setdefault(_storage_key(rec.output), idx)
        # Tensor behind every state id (kept alive by the call records).
        state_tensors: Dict[int, Tensor] = {id(out_A): out_A}
        input_id = id(input_A)

        def _is_pending(sid: int, k: int) -> bool:
            if sid == input_id or first_producer.get(sid, k) < k:
                return True
            t = state_tensors.get(sid)
            return t is not None and first_producer_storage.get(_storage_key(t), k) < k

        def _store(t: Tensor, ns: _GameState) -> None:
            tid = id(t)
            existing = states.get(tid)
            states[tid] = ns if existing is None else existing.add(ns)
            state_tensors[tid] = t

        def _try_slice_embed(st: _GameState, target_shape: Tuple[int, ...]) -> Optional[_GameState]:
            """Bridge a functional slice boundary by zero-padding.

            If the stranded state's spatial dims are a strict suffix of
            the target spatial dims (e.g. state ``(B, 768)`` and
            target ``(B, 197, 768)``), embed the state at index 0 of the
            missing leading spatial dim(s).  This is exactly the inverse
            of the ViT CLS-token slice ``x[:, 0]`` between ``encoder.ln``
            and ``heads.head``.
            """
            state_spatial = st.ordinary_spatial_shape()
            target_spatial = tuple(target_shape[1:])
            if state_spatial == target_spatial:
                return st
            if len(state_spatial) >= len(target_spatial):
                return None
            if state_spatial != target_spatial[-len(state_spatial):]:
                return None
            leading = target_spatial[:len(target_spatial) - len(state_spatial)]
            if st.sign_augmented:
                new_full_reg = (st.mu_A.shape[0], 2) + tuple(leading) + state_spatial
                idx_reg = (
                    (slice(None), slice(None))
                    + (0,) * len(leading)
                    + (slice(None),) * len(state_spatial)
                )
            else:
                new_full_reg = (st.mu_A.shape[0],) + tuple(leading) + state_spatial
                idx_reg = (slice(None),) + (0,) * len(leading) + (slice(None),) * len(state_spatial)
            zeros_reg = torch.zeros(new_full_reg, device=st.mu_A.device, dtype=st.mu_A.dtype)
            new_mu_A = zeros_reg.clone(); new_mu_A[idx_reg] = st.mu_A
            new_mu_B = zeros_reg.clone(); new_mu_B[idx_reg] = st.mu_B
            new_beta = zeros_reg.clone(); new_beta[idx_reg] = st.beta
            return _GameState(
                new_mu_A,
                new_mu_B,
                new_beta,
                sign_augmented=st.sign_augmented,
            )

        for k in range(len(calls_A) - 1, -1, -1):
            rec_A = calls_A[k]
            rec_B = calls_B[k]

            out_id = id(rec_A.output)
            state: Optional[_GameState] = None
            if out_id in states:
                state = states.pop(out_id)
            else:
                # Shape-fallback: search for a stranded state whose flat
                # spatial size matches the current call's output.  This
                # recovers across functional flatten/view/reshape between
                # hooked modules.
                target_numel = _per_batch_numel(rec_A.output)
                stranded = [sid for sid in states if not _is_pending(sid, k)]
                sized = [sid for sid in stranded if _state_per_batch_numel(states[sid]) == target_numel]
                out_storage = _storage_key(rec_A.output)
                views = [sid for sid in sized if _storage_key(state_tensors[sid]) == out_storage]
                match_id = (views or sized or [None])[0]
                if match_id is None:
                    # Slice-embed fallback: a stranded state whose
                    # spatial shape is a strict suffix of the call's
                    # output spatial shape comes from a single-position
                    # slice (CLS token in ViT).  Embed by zero-padding
                    # at index 0 of the missing leading dim(s).
                    target_full_shape = tuple(rec_A.output.shape)
                    embedded_id: Optional[int] = None
                    for sid in stranded:
                        st = states[sid]
                        attempt = _try_slice_embed(st, target_full_shape)
                        if attempt is not None:
                            embedded_id = sid
                            states[sid] = attempt
                            break
                    if embedded_id is None:
                        continue
                    match_id = embedded_id
                state = states.pop(match_id)

            # If the popped state's spatial shape doesn't already match
            # the call's output, reshape it (handles flatten/view/reshape).
            target_full_shape = tuple(rec_A.output.shape)
            if state.ordinary_spatial_shape() != target_full_shape[1:]:
                state = state.reshape_to(target_full_shape)

            try:
                if self.condition_on_survival and self.game == "adf":
                    if survival_maps_A is None or survival_maps_B is None:
                        raise RuntimeError("Missing conditioned survival maps")
                    new_states = self._dispatch_call_backward_conditioned_adf(
                        rec_A, rec_B, state, survival_maps_A, survival_maps_B,
                    )
                else:
                    new_states = self._dispatch_call_backward(rec_A, rec_B, state, cemetery)
            except (RuntimeError, ValueError) as exc:
                logger.warning(
                    "Hellinger sweep skipped %s (%s): %s",
                    rec_A.name, rec_A.mod_type.__name__, exc,
                )
                # Put the popped state back under the module's output id so
                # it isn't silently lost.  The end-of-sweep "max remaining
                # state" fallback will pick the most massive surviving entry
                # and report its per-pixel map rather than failing the
                # whole batch with "no state at any tensor id".  This makes
                # one skipped layer (e.g. an OOM on a single big Linear)
                # a warning, not a fatal error — important for the
                # randomization experiment where a bad step would
                # otherwise discard the entire run.
                _store(rec_A.output, state)
                # Free CUDA cache to help the next iteration recover from
                # a transient OOM on a single huge layer.
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            # DCCat is the one hooked module whose inputs slot 0 holds a
            # *list* of tensors rather than a single tensor.  Its backward
            # dispatcher returns one state per list element keyed on 0..N-1
            # (NOT positional forward-arg slots), so route each to the
            # matching upstream tensor inside the list.
            is_dccat = _HAS_DCCAT and isinstance(rec_A.module, _DCCat)
            if is_dccat:
                tensor_list = rec_A.inputs[0] if rec_A.inputs else []
                if not isinstance(tensor_list, (list, tuple)):
                    tensor_list = []
                for inp_idx, ns in new_states.items():
                    if inp_idx >= len(tensor_list):
                        continue
                    inp_tensor = tensor_list[inp_idx]
                    if not isinstance(inp_tensor, Tensor):
                        continue
                    _store(inp_tensor, ns)
            else:
                for inp_idx, ns in new_states.items():
                    if inp_idx >= len(rec_A.inputs):
                        continue
                    inp_tensor = rec_A.inputs[inp_idx]
                    if not isinstance(inp_tensor, Tensor):
                        continue
                    _store(inp_tensor, ns)

            if 0 in new_states:
                per_module_h2[rec_A.name] = new_states[0].per_pixel_h2().detach()

        # Final state at the user's input tensor.  If the strict id is
        # not in the dict (e.g.\ the very first hooked module took a
        # transformed view of input_A), fall back to the unique state
        # whose per-batch element count matches input_A.
        final_id = id(input_A)
        final_state: Optional[_GameState] = None
        if final_id in states:
            final_state = states[final_id]
        else:
            target_numel = _per_batch_numel(input_A)
            for st in states.values():
                if _state_per_batch_numel(st) == target_numel:
                    final_state = st
                    break
        if final_state is None:
            if states:
                final_state = max(states.values(), key=lambda s: s.ordinary_numel_per_batch())
            else:
                raise RuntimeError("Backward sweep produced no state at any tensor id")
        # Best-effort reshape to the actual input image shape.  Architectures
        # whose first hooked module is Conv2d (VGG, ResNet) end up at exactly
        # the input shape and we just enforce it.  Architectures with token-
        # level early exits (ViT, where the CLS-token attention head reads a
        # single token) leave the state at a coarser token-level shape; we
        # leave that as-is rather than crashing the user.
        if final_state.ordinary_spatial_shape() != tuple(input_A.shape[1:]):
            try:
                final_state = final_state.reshape_to(tuple(input_A.shape))
            except RuntimeError:
                logger.warning(
                    "HellingerDiagnostic: stranded state shape %s does not "
                    "reshape to input shape %s; returning the coarser state.",
                    tuple(final_state.mu_A.shape), tuple(input_A.shape),
                )

        raw_survival_mass_A: Optional[Tensor] = None
        raw_survival_mass_B: Optional[Tensor] = None
        if self.condition_on_survival:
            live_A = final_state.mu_A.flatten(start_dim=1).sum(dim=1)
            live_B = final_state.mu_B.flatten(start_dim=1).sum(dim=1)
            bad = (live_A <= _EPS_DENOM) | (live_B <= _EPS_DENOM)
            if bad.any():
                if self.game == "adf":
                    raise RuntimeError("Conditioned ADF sweep produced zero ordinary mass at the input")
                bad_idx = torch.nonzero(bad, as_tuple=False).flatten().tolist()
                raise ValueError(
                    "condition_on_survival=True is undefined when no ordinary "
                    f"path survives to the input; batch indices {bad_idx}"
                )
            raw_survival_mass_A = live_A
            raw_survival_mass_B = live_B
            scale_A = live_A.view(-1, *([1] * (final_state.mu_A.dim() - 1)))
            scale_B = live_B.view(-1, *([1] * (final_state.mu_B.dim() - 1)))
            scale_beta = torch.sqrt((live_A * live_B).clamp(min=_EPS_DENOM)).view(
                -1, *([1] * (final_state.beta.dim() - 1))
            )
            final_state = _GameState(
                final_state.mu_A / scale_A,
                final_state.mu_B / scale_B,
                final_state.beta / scale_beta,
                sign_augmented=final_state.sign_augmented,
            )

        # The per-pixel map covers ordinary input states only.  The
        # cemetery contribution is the separate scalar coordinate h^2(†).
        alpha_A_input = final_state.alpha_A()
        alpha_B_input = final_state.alpha_B()
        beta_input_ordinary = final_state.beta_ordinary()
        per_pixel_h2 = final_state.per_pixel_h2()
        per_pixel_h2_marginal = final_state.per_pixel_h2_marginal()
        per_pixel_h2_fraction = final_state.per_pixel_h2_fraction()
        bc_reg = beta_input_ordinary.flatten(start_dim=1).sum(dim=1)
        per_pixel_h2_sum = per_pixel_h2.flatten(start_dim=1).sum(dim=1)
        if self.condition_on_survival:
            cem_h2_scalar = torch.zeros_like(per_pixel_h2_sum)
            h_squared = per_pixel_h2_sum
            bc = bc_reg
            mu_A_cem = torch.zeros_like(cemetery.mu_A)
            mu_B_cem = torch.zeros_like(cemetery.mu_B)
            beta_cem = torch.zeros_like(cemetery.beta)
        else:
            cem_h2_scalar = cemetery.h2()
            h_squared = per_pixel_h2_sum + cem_h2_scalar
            bc = bc_reg + cemetery.beta.flatten(start_dim=1).sum(dim=1)
            mu_A_cem = cemetery.mu_A
            mu_B_cem = cemetery.mu_B
            beta_cem = cemetery.beta
        h_dist = h_squared.clamp(min=0).sqrt()
        if self.condition_on_survival:
            if self.game == "adf":
                survival_mass_A = survival_A_target if survival_A_target is not None else torch.ones_like(h_squared)
                survival_mass_B = survival_B_target if survival_B_target is not None else torch.ones_like(h_squared)
            else:
                if raw_survival_mass_A is None or raw_survival_mass_B is None:
                    raise RuntimeError("Missing RG survival masses for conditioned result")
                survival_mass_A = raw_survival_mass_A
                survival_mass_B = raw_survival_mass_B
        else:
            survival_mass_A = final_state.mu_A.flatten(start_dim=1).sum(dim=1)
            survival_mass_B = final_state.mu_B.flatten(start_dim=1).sum(dim=1)

        result = {
            "hellinger_distance": h_dist,
            "h_squared": h_squared,
            "bhattacharyya": bc,
            "per_pixel_h2": per_pixel_h2.detach(),
            "per_pixel_h2_marginal": per_pixel_h2_marginal.detach(),
            "per_pixel_h2_fraction": per_pixel_h2_fraction.detach(),
            "cemetery_h2": cem_h2_scalar.detach(),
            "per_module_h2": per_module_h2,
            "alpha_A_input": alpha_A_input.detach(),
            "alpha_B_input": alpha_B_input.detach(),
            "beta_input_ordinary": beta_input_ordinary.detach(),
            "mu_A_input": final_state.mu_A.detach(),
            "mu_B_input": final_state.mu_B.detach(),
            "beta_input": final_state.beta.detach(),
            "mu_A_cem": mu_A_cem.detach(),
            "mu_B_cem": mu_B_cem.detach(),
            "beta_cem": beta_cem.detach(),
            "survival_mass_A": survival_mass_A.detach(),
            "survival_mass_B": survival_mass_B.detach(),
            "condition_on_survival": self.condition_on_survival,
            "target": target.detach(),
            "game": self.game,
        }
        if self.condition_on_survival:
            result["per_pixel_h2_live"] = per_pixel_h2.detach()
            result["per_pixel_h2_marginal_live"] = per_pixel_h2_marginal.detach()
            result["per_pixel_h2_fraction_live"] = per_pixel_h2_fraction.detach()
        return result

    def _dispatch_call_backward(
        self,
        rec_A: _CallRecord,
        rec_B: _CallRecord,
        state: _GameState,
        cemetery: _CemeteryState,
    ) -> Dict[int, _GameState]:
        """Run one backward step for a single forward call.

        Returns ``{input_index: new_state}`` mapping each input tensor of
        the call to the state that should be routed there.  Most kernels
        return a single entry at index 0; the Add subgame returns two
        entries (one half-state per operand).
        """
        mod_A = rec_A.module
        mod_B = rec_B.module
        in_A = rec_A.inputs[0] if rec_A.inputs else None
        in_B = rec_B.inputs[0] if rec_B.inputs else None

        # ----- Addition Subgame (residual connections) -----
        # Definition (Addition Subgame): the unobserved Oracle uniformly
        # transitions to either operand with probability 1/2.  For the
        # path measure, this halves the state and routes the halves to
        # both operands; total mass is preserved.  This is the same for
        # both ADF and RG (the per-game discount factor only affects
        # OCCUPATION measures, not the path measure that the Hellinger DP
        # propagates).
        if _HAS_DCADD and isinstance(mod_A, _DCAdd):
            return {0: state.scale(0.5), 1: state.scale(0.5)}

        # ----- Linear -----
        if isinstance(mod_A, nn.Linear):
            if self.game == "adf":
                # ADF linear propagation uses ``mu @ W`` which broadcasts
                # over any number of leading batch/token dims natively, so
                # the same helper covers both flat MLPs and ViT tokens.
                ns = _adf_backward_linear(mod_A, mod_B, state)
            else:
                # The flat helper's ``'bj,bji->bi'`` einsum needs a 2-D
                # input; ``a_in.ndim > 2`` (ViT MLPs, attention out_proj,
                # v_proj) takes the token-Linear helper.
                in_rank = in_A.dim() if isinstance(in_A, Tensor) else 2
                if state.sign_augmented:
                    # Stream-labelled Value entries from the attention step.
                    if id(rec_A) not in self._value_projections:
                        raise RuntimeError(
                            f"stream-labelled state reached {rec_A.name}, "
                            "which is not the Value projection of an attention product"
                        )
                    ns = _rg_backward_value_projection(
                        mod_A, mod_B, in_A, in_B, state, self.tau, cemetery,
                    )
                elif in_rank > 2:
                    ns = _rg_backward_token_linear(
                        mod_A, mod_B, in_A, in_B, state, self.tau, cemetery,
                    )
                else:
                    ns = _rg_backward_linear(
                        mod_A, mod_B, in_A, in_B, state, self.tau, cemetery,
                    )
            return {0: ns}

        # ----- Conv2d -----
        if isinstance(mod_A, nn.Conv2d):
            if self.game == "adf":
                ns = _adf_backward_conv2d(mod_A, mod_B, in_A, in_B, state)
            else:
                ns = _rg_backward_conv2d(
                    mod_A, mod_B, in_A, in_B, state, self.tau, cemetery,
                )
            return {0: ns}

        # ----- Gated activations (ReLU / GELU / SiLU) -----
        # Cemetery absorption happens at the gated activation: "dead" is a
        # property of the gate, not of the weight kernel (the output
        # classifier has no gate).
        if isinstance(mod_A, _GATED_ACTIVATION_TYPES):
            z_A = rec_A.inputs[0] if rec_A.inputs else None
            z_B = rec_B.inputs[0] if rec_B.inputs else None
            if z_A is not None and z_B is not None:
                return {0: _backward_activation_cemetery(
                    mod_A, mod_B, z_A, z_B, state, cemetery,
                )}

        # ----- Identity-like (non-gated activations, norms, dropout, …) -----
        if isinstance(mod_A, _IDENTITY_TYPES):
            return {0: _backward_identity(state)}

        # ----- Flatten -----
        if isinstance(mod_A, nn.Flatten):
            return {0: _backward_flatten(mod_A, in_A, state)}

        # ----- Reshape / View / Squeeze / Unsqueeze (shape ops) -----
        if _HAS_DCSHAPE and isinstance(
            mod_A, (_DCReshape, _DCView, _DCSqueeze, _DCUnsqueeze)
        ):
            return {0: _backward_reshape(in_A, state)}
        if _HAS_DCSHAPE and isinstance(mod_A, _DCTranspose):
            return {0: _backward_transpose(mod_A, state)}
        if _HAS_DCSHAPE and isinstance(mod_A, _DCPermute):
            return {0: _backward_permute(mod_A, state)}

        # ----- DCCat (torch.cat wrapper) -----
        # Split state along the cat dim and route each chunk to the matching
        # upstream tensor.  Dict keys index the *list* of inputs[0], not the
        # positional forward args; the main loop detects this via
        # isinstance(mod, DCCat).
        if _HAS_DCCAT and isinstance(mod_A, _DCCat):
            tensor_list = rec_A.inputs[0] if rec_A.inputs else []
            if not isinstance(tensor_list, (list, tuple)) or not tensor_list:
                return {0: _backward_identity(state)}
            splits = _backward_dccat(mod_A, list(tensor_list), state)
            return {i: s for i, s in enumerate(splits)}

        # ----- Pooling -----
        if isinstance(mod_A, nn.AvgPool2d):
            return {0: _backward_avgpool2d(mod_A, in_A, in_B, state)}
        if isinstance(mod_A, nn.AdaptiveAvgPool2d):
            return {0: _backward_adaptive_avgpool2d(mod_A, in_A, in_B, state)}
        if isinstance(mod_A, nn.MaxPool2d):
            return {0: _backward_maxpool2d(mod_A, in_A, in_B, state)}

        # ----- DCMatMul -----
        # Distinguish the QK^T matmul (output is square in the last two
        # dims, T_q × T_k = T × T) from the AV matmul (output is T_q × D_h
        # with D_h ≠ T_q).  The QK^T path is treated as detached: any
        # state arriving there is dropped (the chain stops).  The AV
        # path is dispatched as a Linear-like backward with the
        # attention matrix A as the (per-head, per-d) detached weight,
        # and only the V branch (input slot 1) receives state.  For RG the
        # V branch receives the stream-labelled Value-entry state of the
        # composite step.
        if _HAS_DCMATMUL and isinstance(mod_A, DCMatMul):
            if _is_av_matmul(rec_A):
                if self.game == "rg":
                    link = self._value_links.get(id(rec_A))
                    if link is None:
                        raise RuntimeError(f"RG attention: no Value projection linked to {rec_A.name}")
                    return _av_backward_merged(
                        rec_A, rec_B, link[0], link[1], state, cemetery, self.tau,
                    )
                return _adf_backward_av(rec_A, rec_B, state)
            return {}  # QK^T detached: drop the state
        if _HAS_DCMUL and isinstance(mod_A, DCMul):
            return {0: _backward_identity(state)}

        # ----- Default -----
        return {0: _backward_identity(state)}


def compute_hellinger_distance(
    model_A: nn.Module,
    input_A: Tensor,
    model_B: Optional[nn.Module] = None,
    input_B: Optional[Tensor] = None,
    target: Optional[Union[int, Tensor]] = None,
    game: str = "adf",
    tau: float = 1.0,
    condition_on_survival: bool = False,
) -> Dict[str, Any]:
    """Functional shortcut for :class:`HellingerDiagnostic.compute`."""
    diag = HellingerDiagnostic(
        model_A, model_B, game=game, tau=tau,
        condition_on_survival=condition_on_survival,
    )
    return diag.compute(input_A, input_B, target)
