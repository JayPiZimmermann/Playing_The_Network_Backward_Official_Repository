"""
Base utilities for DC decomposition.

Both forward and backward use [4*batch, ...] format for autograd compatibility:
- Forward input: [pos; neg; 0; 0] (last two are zero)
- Forward output: [out_pos; out_neg; out_pos; out_neg]
- Backward: [delta_pp; delta_np; delta_pn; delta_nn]

This ensures autograd shape compatibility (backward returns same shape as input).
"""

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import torch
from torch import Tensor


class InputMode(Enum):
    CENTER = "center"      # pos = ReLU(x), neg = ReLU(-x)
    POSITIVE = "positive"  # pos = x, neg = 0
    NEGATIVE = "negative"  # pos = 0, neg = -x


# Module attribute names
DC_ENABLED = '_dc_enabled'
DC_ORIGINAL_FORWARD = '_dc_original_forward'
DC_RELU_LAMBDA = '_dc_relu_lambda'
DC_IS_OUTPUT_LAYER = '_dc_is_output_layer'
DC_ADF_FORWARD = '_dc_adf_forward'
DC_ALIGNMENT_CACHE = '_dc_alignment_cache'
DC_CACHE_LAYER_NAME = '_dc_cache_name'


# =============================================================================
# Core: [4*batch] format for both forward and backward
# =============================================================================

def cat4(t0: Tensor, t1: Tensor, t2: Tensor, t3: Tensor) -> Tensor:
    """Concatenate 4 tensors along batch dim."""
    return torch.cat([t0, t1, t2, t3], dim=0)


def split4(catted: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split [4*batch] into 4 parts."""
    q = catted.shape[0] // 4
    return catted[:q], catted[q:2*q], catted[2*q:3*q], catted[3*q:]


# =============================================================================
# Forward: create [4*batch] = [pos; neg; 0; 0]
# =============================================================================

def make_input_4(pos: Tensor, neg: Tensor) -> Tensor:
    """Create [4*batch] forward input: [pos; neg; 0; 0] (with zeros)."""
    return torch.cat([pos, neg, pos, neg], dim=0)


def split_input_4(catted: Tensor) -> Tuple[Tensor, Tensor]:
    """Extract pos and neg from [4*batch] forward tensor (use first 2 quarters)."""
    q = catted.shape[0] // 4
    return catted[:q], catted[q:2*q]


def make_output_4(out_pos: Tensor, out_neg: Tensor) -> Tensor:
    """Create [4*batch] forward output: [out_pos; out_neg; out_pos; out_neg]."""
    return torch.cat([out_pos, out_neg, out_pos, out_neg], dim=0)


# =============================================================================
# Backward: [4*batch] = [delta_pp; delta_np; delta_pn; delta_nn]
# =============================================================================

def split_grad_4(catted: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Split [4*batch] backward gradient into 4 sensitivities."""
    return split4(catted)


def make_grad_4(delta_pp: Tensor, delta_np: Tensor,
                delta_pn: Tensor, delta_nn: Tensor) -> Tensor:
    """Concatenate 4 sensitivities into [4*batch]."""
    return cat4(delta_pp, delta_np, delta_pn, delta_nn)


# Module attribute for sensitivity shift alpha
DC_SENSITIVITY_ALPHA = '_dc_sensitivity_alpha'


def shift_sensitivities(
    delta_pp: Tensor,
    delta_np: Tensor,
    delta_pn: Tensor,
    delta_nn: Tensor,
    alpha: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Apply adaptive shift to sensitivities to reduce magnitudes.

    The shift preserves the gradient reconstruction:
        grad = delta_pp - delta_np - delta_pn + delta_nn

    The formula shifts pos-path and neg-path sensitivities independently:
        shift_p = alpha * (delta_pp + delta_np)
        shift_n = alpha * (delta_pn + delta_nn)
        new_pp = delta_pp - shift_p
        new_np = delta_np - shift_p
        new_pn = delta_pn - shift_n
        new_nn = delta_nn - shift_n

    This preserves the reconstruction because:
        new_pp - new_np = delta_pp - delta_np (shift_p cancels)
        new_pn - new_nn = delta_pn - delta_nn (shift_n cancels)

    Args:
        delta_pp, delta_np, delta_pn, delta_nn: 4 sensitivity tensors
        alpha: Shift factor (0 = no shift, 0.5 = typical for add layers)

    Returns:
        (new_pp, new_np, new_pn, new_nn) shifted sensitivities
    """
    if alpha == 0.0:
        return delta_pp, delta_np, delta_pn, delta_nn

    shift_p = alpha * (delta_pp + delta_np)
    shift_n = alpha * (delta_pn + delta_nn)

    new_pp = delta_pp - shift_p
    new_np = delta_np - shift_p
    new_pn = delta_pn - shift_n
    new_nn = delta_nn - shift_n

    return new_pp, new_np, new_pn, new_nn


def compute_frobenius_alpha(weight: Tensor) -> float:
    """
    Compute alpha based on Frobenius norm of weight matrix.

    Formula: alpha = (1 - 1/rho) / 2 where rho is the Frobenius norm.

    For rho > 1, this gives alpha in (0, 0.5).
    For rho <= 1, this returns 0 (no shift needed).

    Args:
        weight: Weight tensor to compute Frobenius norm from

    Returns:
        Alpha value for shift_sensitivities
    """
    rho = torch.norm(weight, p='fro').item()
    if rho > 1.0:
        return (1.0 - 1.0 / rho) / 2.0
    return 0.0


# =============================================================================
# ADF (Assumed Density Filtering): risk-sensitive backward pass
# =============================================================================

DC_LAYER_DATA = '_dc_layer_data'

_adf_mode: str = 'none'
_adf_lambda: float = 0.0
_adf_probabilistic: bool = True


@contextmanager
def adf_backprop_context(
    mode: str = 'none',
    lam: float = 0.0,
    probabilistic: bool = True,
):
    """Context manager to set the ADF backward mode for activation patches.

    The activation gate uses the confidence bound ``z_eff = μ + λσ`` of the
    pre-activation (λ < 0: risk-averse).  With ``probabilistic=True`` the
    gate is the continuation probability Φ(z_eff/σ); otherwise the hard
    mask I(z_eff ≥ 0).

    Args:
        mode: 'none' (standard DC) or 'adf' (risk-sensitive backward).
        lam: Risk parameter λ.
        probabilistic: Use smooth Φ masks (True) or hard threshold (False).
    """
    global _adf_mode, _adf_lambda, _adf_probabilistic
    old = (_adf_mode, _adf_lambda, _adf_probabilistic)
    _adf_mode = mode
    _adf_lambda = lam
    _adf_probabilistic = probabilistic
    try:
        yield
    finally:
        _adf_mode, _adf_lambda, _adf_probabilistic = old


def get_adf_config():
    """Return current ADF config: (mode, lambda, probabilistic)."""
    return _adf_mode, _adf_lambda, _adf_probabilistic


# ADF propagation mode: how variance is propagated through the network
_adf_propagation: str = 'analytic'


@contextmanager
def adf_propagation_context(mode: str = 'analytic'):
    """Context manager to set ADF variance propagation mode.

    Controls how Gaussian variance is propagated through the network during
    the ADF forward pass.

    Modes:
        'analytic': Standard Jacobian-based moment matching (default).
            Each layer propagates variance analytically using first-order
            Taylor approximation.  Fast but can lose variance through
            LayerNorm (resets to γ²) and Softmax (crushes by s²).

        'delta_method': Modified analytic that preserves variance
            through LayerNorm.  Instead of resetting to γ², propagates as
            (γ² / (σ² + ε)) · v_in.  Better for ViT architectures where
            LayerNorm otherwise kills the variance signal.
    """
    global _adf_propagation
    old = _adf_propagation
    _adf_propagation = mode
    try:
        yield
    finally:
        _adf_propagation = old


def get_adf_propagation() -> str:
    """Return current ADF propagation mode."""
    return _adf_propagation


def init_backward(grad_4: Tensor, is_output_layer: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Initialize 4-sensitivities for backward pass.

    For output layers: converts original gradient to proper 4-sensitivities
    (split 1/2 - 1/2 between delta_pp and delta_nn).
    For other layers: just splits the incoming gradient.

    Args:
        grad_4: Incoming gradient [4*batch] (for output layer: only first quarter used)
        is_output_layer: Whether this is the output layer

    Returns:
        (delta_pp, delta_np, delta_pn, delta_nn) tuple
    """
    if is_output_layer:
        q = grad_4.shape[0] // 4
        grad_orig = grad_4[:q]  # Use only first quarter as original gradient
        delta_pp = 0.5 * grad_orig
        delta_np = torch.zeros_like(grad_orig)
        delta_pn = torch.zeros_like(grad_orig)
        delta_nn = 0.5 * grad_orig
        result = delta_pp, delta_np, delta_pn, delta_nn

        # Log output layer initialization
        try:
            from ..logging_config import get_logger, TENSOR_LEVEL
            import logging
            log = get_logger('dc.backward')
            if log.isEnabledFor(logging.INFO):
                log.info(f"init_backward OUTPUT: shape={list(grad_4.shape)}")
            tensor_log = get_logger('dc.tensors')
            if tensor_log.isEnabledFor(TENSOR_LEVEL):
                tensor_log.log(TENSOR_LEVEL, f"init_bwd: grad_orig={grad_orig.mean().item():.4f}")
        except ImportError:
            pass

        return result
    else:
        return split_grad_4(grad_4)


def recenter_forward(output_4: Tensor) -> Tensor:
    """
    Re-center forward output for numerical stability.

    Converts [pos; neg; 0; 0] to canonical form where one of pos/neg
    is zero for each element, minimizing magnitudes.

    Uses in-place data modification to not affect autograd graph.
    """
    q = output_4.shape[0] // 4
    pos = output_4[:q]
    neg = output_4[q:2*q]

    # Log before recenter (lazy import to avoid circular deps)
    old_pos_mean = old_neg_mean = None
    try:
        from ..logging_config import get_logger, TENSOR_LEVEL
        import logging
        log = get_logger('dc.recenter')
        tensor_log = get_logger('dc.tensors')
        if log.isEnabledFor(logging.INFO):
            old_pos_mean = pos.mean().item()
            old_neg_mean = neg.mean().item()
    except ImportError:
        log = None
        tensor_log = None

    with torch.no_grad():
        z = pos - neg
        new_pos = torch.relu(z)
        new_neg = torch.relu(-z)

    # Create output tensor that maintains gradient connection to input
    # by using clone and in-place data assignment
    result = output_4.clone()
    result[:q].data.copy_(new_pos)
    result[q:2*q].data.copy_(new_neg)
    result[2*q:3*q].data.copy_(new_pos)
    result[3*q:].data.copy_(new_neg)

    # Log after recenter
    if log is not None and log.isEnabledFor(logging.INFO):
        new_pos_mean = new_pos.mean().item()
        new_neg_mean = new_neg.mean().item()
        log.info(f"recenter: pos {old_pos_mean:.2f}->{new_pos_mean:.2f}, neg {old_neg_mean:.2f}->{new_neg_mean:.2f}")

    if tensor_log is not None and tensor_log.isEnabledFor(TENSOR_LEVEL):
        tensor_log.log(TENSOR_LEVEL, f"recenter: shape={list(output_4.shape)}, z_mean={z.mean().item():.4f}")

    return result


# =============================================================================
# Input initialization
# =============================================================================

def init_pos_neg(x: Tensor, mode: InputMode = InputMode.CENTER) -> Tuple[Tensor, Tensor]:
    """Split input into pos/neg streams."""
    if mode == InputMode.CENTER:
        return torch.relu(x), torch.relu(-x)
    elif mode == InputMode.POSITIVE:
        return x, torch.zeros_like(x)
    elif mode == InputMode.NEGATIVE:
        return torch.zeros_like(x), -x
    raise ValueError(f"Unknown mode: {mode}")


def init_catted(x: Tensor, mode: InputMode = InputMode.CENTER) -> Tensor:
    """
    Initialize [4*batch] input for DC forward: [pos; neg; 0; 0].

    This is a PREPROCESSING step that is DETACHED from autograd.
    The returned tensor has requires_grad=True so that backward pass
    accumulates the 4 sensitivities in its .grad attribute.

    Args:
        x: Input tensor of shape [batch, ...]
        mode: How to split x into pos/neg streams
            - CENTER: pos = relu(x), neg = relu(-x)  [default]
            - POSITIVE: pos = x, neg = 0
            - NEGATIVE: pos = 0, neg = -x

    Returns:
        Tensor of shape [4*batch, ...] with requires_grad=True.
        After backward, .grad contains [delta_pp; delta_np; delta_pn; delta_nn].
    """
    with torch.no_grad():
        if mode == InputMode.CENTER:
            pos = torch.relu(x)
            neg = torch.relu(-x)
        elif mode == InputMode.POSITIVE:
            pos = x.clone()
            neg = torch.zeros_like(x)
        else:  # NEGATIVE
            pos = torch.zeros_like(x)
            neg = -x

        result = torch.cat([pos, neg, torch.zeros_like(pos), torch.zeros_like(neg)], dim=0)

    # Enable gradient tracking on the result (detached from x)
    result.requires_grad_(True)

    # Log initialization
    try:
        from ..logging_config import get_logger, TENSOR_LEVEL
        import logging
        log = get_logger('dc.forward')
        if log.isEnabledFor(logging.INFO):
            log.info(f"init_catted: {list(x.shape)} -> {list(result.shape)} (mode={mode.value})")
        tensor_log = get_logger('dc.tensors')
        if tensor_log.isEnabledFor(TENSOR_LEVEL):
            q = result.shape[0] // 4
            pos, neg = result[:q], result[q:2*q]
            tensor_log.log(TENSOR_LEVEL, f"init: x_mean={x.mean().item():.4f}, pos_mean={pos.mean().item():.4f}, neg_mean={neg.mean().item():.4f}")
    except ImportError:
        pass

    return result


# =============================================================================
# Output reconstruction
# =============================================================================

class ReconstructOutputFunction(torch.autograd.Function):
    """
    Reconstruct output from DC format and initialize sensitivities.

    Forward: [pos; neg; 0; 0] -> pos - neg
    Backward: Initializes 4-sensitivities with beta weighting:
        - delta_pp = beta * g
        - delta_np = 0
        - delta_pn = (1-beta) * (-g)  [since d(pos-neg)/d(neg) = -1]
        - delta_nn = 0

    With beta=1 (default): [g, 0, -0, 0] = [g, 0, 0, 0]
    Reconstruction check: delta_pp - delta_np - delta_pn + delta_nn
                        = beta*g - 0 - (-(1-beta)*g) + 0
                        = beta*g + (1-beta)*g = g ✓
    """

    @staticmethod
    def forward(ctx, output_4: Tensor) -> Tensor:
        q = output_4.shape[0] // 4
        out_pos = output_4[:q]
        out_neg = output_4[q:2*q]
        
        return out_pos - out_neg

    @staticmethod
    def backward(ctx, grad: Tensor) -> Tuple[Tensor]:
        zeros = torch.zeros_like(grad)
        # Initialize sensitivities: pp=grad, pn=0, np=nn=0 (equivalent to beta=1.0)
        delta_pp = grad
        delta_np = zeros
        delta_pn = zeros
        delta_nn = zeros
        return torch.cat([delta_pp, delta_np, delta_pn, delta_nn], dim=0),


def reconstruct_output(output_4: Tensor) -> Tensor:
    """Reconstruct original output from [4*batch]: out_pos - out_neg.

    Args:
        output_4: DC format tensor [pos; neg; 0; 0]
        beta: Sensitivity initialization weight (default 1.0).
              Backward pass initializes: delta_pp = beta*g, delta_pn = -(1-beta)*g
    """
    result = ReconstructOutputFunction.apply(output_4)

    # Log reconstruction
    try:
        from ..logging_config import get_logger, TENSOR_LEVEL
        import logging
        log = get_logger('dc.forward')
        if log.isEnabledFor(logging.INFO):
            log.info(f"reconstruct: {list(output_4.shape)} -> {list(result.shape)}")
        tensor_log = get_logger('dc.tensors')
        if tensor_log.isEnabledFor(TENSOR_LEVEL):
            q = output_4.shape[0] // 4
            out_pos, out_neg = output_4[:q], output_4[q:2*q]
            tensor_log.log(TENSOR_LEVEL, f"reconstruct: pos_mean={out_pos.mean().item():.4f}, neg_mean={out_neg.mean().item():.4f}, z_mean={result.mean().item():.4f}")
    except ImportError:
        pass

    return result


# =============================================================================
# Re-centering (prevents magnitude explosion in residual networks)
# =============================================================================

class DCRecenterFunction(torch.autograd.Function):
    """
    Re-center DC representation with proper gradient flow.

    Forward: new_pos = relu(z), new_neg = relu(-z) where z = pos - neg
    Backward: Pass gradients through unchanged.

    The recenter operation preserves z = pos - neg (since new_pos - new_neg = z),
    so for gradient purposes it acts as identity on the underlying value.
    """

    @staticmethod
    def forward(ctx, tensor_4: Tensor) -> Tensor:
        q = tensor_4.shape[0] // 4
        pos = tensor_4[:q]
        neg = tensor_4[q:2*q]

        z = pos - neg
        new_pos = torch.relu(z)
        new_neg = torch.relu(-z)

        return make_input_4(new_pos, new_neg)

    @staticmethod
    def backward(ctx, grad_4: Tensor) -> Tensor:
        # Pass gradient through unchanged - recenter preserves z = pos - neg
        return grad_4


def recenter_dc(tensor_4: Tensor) -> Tensor:
    """
    Re-center DC representation to minimize pos and neg magnitudes.

    Given [pos; neg; 0; 0] where z = pos - neg:
    - Computes new_pos = ReLU(z), new_neg = ReLU(-z)
    - Returns [new_pos; new_neg; new_pos; new_neg]

    This preserves z = new_pos - new_neg = ReLU(z) - ReLU(-z) = z,
    but ensures pos and neg have minimal magnitudes (one of them is 0 for each element).

    Use this after residual additions to prevent exponential magnitude growth.
    """
    return DCRecenterFunction.apply(tensor_4)


# =============================================================================
# Sensitivity extraction utilities
# =============================================================================

@dataclass
class Sensitivities:
    """Container for the 4 DC sensitivities."""
    delta_pp: Tensor
    delta_np: Tensor
    delta_pn: Tensor
    delta_nn: Tensor

    def reconstruct_gradient(self) -> Tensor:
        """Reconstruct the standard gradient: pp - np - pn + nn."""
        return self.delta_pp - self.delta_np - self.delta_pn + self.delta_nn

def extract_sensitivities(grad_4: Tensor) -> Sensitivities:
    """Extract 4 sensitivities from [4*batch] gradient tensor."""
    delta_pp, delta_np, delta_pn, delta_nn = split4(grad_4)
    return Sensitivities(delta_pp, delta_np, delta_pn, delta_nn)


# =============================================================================
# Context Manager API (convenient usage)
# =============================================================================

class DCForward:
    """
    Context manager for convenient DC decomposition forward/backward.

    Usage:
        with DCForward(model, x, beta=1.0) as dc:
            out = dc.output
            loss = criterion(out, target)
            loss.backward()

        # After backward:
        sens = dc.sensitivities  # Sensitivities object
        grad = dc.reconstruct_gradient()  # Standard gradient
    """

    def __init__(
        self,
        model: 'torch.nn.Module',
        x: Tensor,
        mode: InputMode = InputMode.CENTER,
        beta: float = 1.0,
    ):
        """
        Args:
            model: DC-patched model (use prepare_model_for_dc first)
            x: Input tensor
            mode: Input splitting mode (default: CENTER)
            beta: Sensitivity initialization weight (default: 1.0)
        """
        self.model = model
        self.x = x
        self.mode = mode
        self.beta = beta

        self._x_cat: Optional[Tensor] = None
        self._out_cat: Optional[Tensor] = None
        self._output: Optional[Tensor] = None

    def __enter__(self) -> 'DCForward':
        # Create DC input (detached from x, with requires_grad=True)
        self._x_cat = init_catted(self.x, self.mode)

        # Forward pass
        self._out_cat = self.model(self._x_cat)
        self._output = reconstruct_output(self._out_cat)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Nothing to clean up
        return False

    @property
    def output(self) -> Tensor:
        """The reconstructed output (pos - neg)."""
        if self._output is None:
            raise RuntimeError("Must be used within context manager")
        return self._output

    @property
    def output_4(self) -> Tensor:
        """The raw [4*batch] output tensor."""
        if self._out_cat is None:
            raise RuntimeError("Must be used within context manager")
        return self._out_cat

    @property
    def input_4(self) -> Tensor:
        """The [4*batch] input tensor (for accessing .grad after backward)."""
        if self._x_cat is None:
            raise RuntimeError("Must be used within context manager")
        return self._x_cat

    @property
    def sensitivities(self) -> Sensitivities:
        """
        Get the 4 sensitivities after backward.

        Returns:
            Sensitivities object with delta_pp, delta_np, delta_pn, delta_nn
        """
        if self._x_cat is None or self._x_cat.grad is None:
            raise RuntimeError("Call backward() first")
        return extract_sensitivities(self._x_cat.grad)

    def reconstruct_gradient(self) -> Tensor:
        """Reconstruct standard gradient from sensitivities: pp - np - pn + nn."""
        return self.sensitivities.reconstruct_gradient()


def dc_forward(
    model: 'torch.nn.Module',
    x: Tensor,
    mode: InputMode = InputMode.CENTER,
    beta: float = 1.0,
) -> DCForward:
    """
    Create a DCForward context manager.

    Usage:
        with dc_forward(model, x) as dc:
            out = dc.output
            loss.backward()

        grad = dc.reconstruct_gradient()
    """
    return DCForward(model, x, mode, beta)
