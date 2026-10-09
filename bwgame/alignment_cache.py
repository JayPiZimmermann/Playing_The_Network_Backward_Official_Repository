"""
Alignment Cache for DC Decomposition.

This module provides a caching mechanism to store original activations and gradients
AFTER functional replacement but BEFORE DC patching. The cache enables:

1. Forward alignment: shift pos/neg so that pos - neg = original_activation
2. Backward alignment: shift sensitivities so that pp - np - pn + nn = original_gradient
3. Cached mask mode: use pre-computed ReLU/MaxPool masks for backward-only mode

Usage:
    from bwgame.alignment_cache import AlignmentCache, AlignmentMode

    # Create cache with desired mode
    cache = AlignmentCache(mode=AlignmentMode.BOTH)

    # Run original model to populate cache (after functional replacement, before DC patching)
    model = make_dc_compatible(model)
    cache.capture_original(model, x, target_grad)

    # Attach to model for DC pass
    model = prepare_model_for_dc(model, alignment_cache=cache)

    # Run DC forward/backward - alignment happens automatically
    x_cat = init_catted(x)
    out = reconstruct_output(model(x_cat))
    out.backward(target_grad)

    # Get correction statistics
    for stat in cache.get_correction_stats():
        print(f"{stat.layer_name}: fwd={stat.forward_correction_norm:.2e}")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple, List, Any, TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .operations.add import Add


# Module attribute name for cache reference
DC_ALIGNMENT_CACHE = '_dc_alignment_cache'
DC_CACHE_LAYER_NAME = '_dc_cache_name'


class AlignmentMode(Enum):
    """Configuration for alignment behavior."""
    NONE = "none"                    # Disabled
    FORWARD_ONLY = "forward_only"    # Only align forward pass
    BACKWARD_ONLY = "backward_only"  # Only align backward pass (uses cached masks)
    BOTH = "both"                    # Align both forward and backward


@dataclass
class LayerAlignmentData:
    """Cached data for a single layer."""
    # Original activations (computed BEFORE DC patching)
    original_activation: Optional[Tensor] = None
    original_gradient: Optional[Tensor] = None

    # All input gradients for multi-input operations (matmul, mul)
    # original_gradients[0] == original_gradient
    original_gradients: List = field(default_factory=list)

    # Masks for nonlinear operations (ReLU, MaxPool)
    relu_mask: Optional[Tensor] = None        # (input >= 0) mask
    maxpool_indices: Optional[Tensor] = None  # Winner indices

    # Correction norms (for diagnostics)
    forward_correction_norm: float = 0.0
    backward_correction_norm: float = 0.0
    # Per-input backward correction norms for multi-input operations
    backward_correction_norms: List = field(default_factory=list)

    # Input shape for backward reconstruction
    input_shape: Optional[Tuple[int, ...]] = None

    # MaxPool parameters for backward
    maxpool_kernel_size: Optional[Any] = None
    maxpool_stride: Optional[Any] = None
    maxpool_padding: Optional[Any] = None


@dataclass
class CorrectionStats:
    """Statistics for forward/backward corrections per layer."""
    layer_name: str
    forward_correction_norm: float = 0.0
    backward_correction_norm: float = 0.0
    forward_relative_correction: float = 0.0
    backward_relative_correction: float = 0.0


class AlignmentCache:
    """
    Caches original activations/gradients and provides alignment functions.

    This cache is designed to be passed to all patched modules and can:
    1. Store original model outputs (BEFORE DC patching)
    2. Align DC pos/neg to match original activations
    3. Align DC sensitivities to match original gradients
    4. Provide cached masks for backward-only mode

    Usage:
        # Create cache with desired mode
        cache = AlignmentCache(mode=AlignmentMode.BOTH)

        # Run original model to populate cache (before DC patching)
        cache.capture_original(model, x, target_grad)

        # Attach to model for DC pass
        cache.attach_to_model(model)

        # Run DC forward/backward - alignment happens automatically
        x_cat = init_catted(x)
        out_cat = model(x_cat)
        out = reconstruct_output(out_cat)
        out.backward(target_grad)

        # Get correction statistics
        stats = cache.get_correction_stats()
    """

    def __init__(self, mode: AlignmentMode = AlignmentMode.NONE):
        self.mode = mode
        self._layer_data: Dict[str, LayerAlignmentData] = {}
        self._layer_order: List[str] = []
        self._handles: List[Any] = []
        self._model: Optional[nn.Module] = None
        # Store captured final output and input gradient for comparison
        self.original_output: Optional[Tensor] = None
        self.original_input_grad: Optional[Tensor] = None
        # Track modules called multiple times (reused modules)
        self._capture_fwd_counts: Dict[str, int] = {}
        self._multi_call_modules: set = set()

    def clear(self) -> None:
        """Clear all cached data."""
        self._layer_data.clear()
        self._layer_order.clear()
        self._remove_hooks()
        self._model = None
        self.original_output = None
        self.original_input_grad = None
        self._capture_fwd_counts.clear()
        self._multi_call_modules.clear()

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for h in self._handles:
            h.remove()
        self._handles.clear()

    # =========================================================================
    # Phase 1: Capture original model behavior (BEFORE DC patching)
    # =========================================================================

    def capture_original(
        self,
        model: nn.Module,
        x: Tensor,
        target_grad: Optional[Tensor] = None,
        capture_forward: bool = True,
        capture_backward: bool = True,
    ) -> Tensor:
        """
        Run original model and cache activations/gradients.

        This must be called AFTER functional replacement but BEFORE DC patching.

        Args:
            model: Model with functional calls replaced but NOT patched
            x: Input tensor
            target_grad: Gradient to backpropagate (required if capture_backward=True)
            capture_forward: Whether to capture forward activations
            capture_backward: Whether to capture backward gradients

        Returns:
            Original model output
        """
        self.clear()
        self._model = model

        # Register hooks for capturing
        for name, module in model.named_modules():
            if self._should_track(module) and name:
                self._layer_order.append(name)
                self._layer_data[name] = LayerAlignmentData()

                if capture_forward:
                    self._handles.append(
                        module.register_forward_hook(
                            self._make_forward_capture_hook(name, module)
                        )
                    )

                if capture_backward:
                    self._handles.append(
                        module.register_full_backward_hook(
                            self._make_backward_capture_hook(name)
                        )
                    )

        # Forward pass
        x_clone = x.clone().requires_grad_(True)
        output = model(x_clone)

        # Backward pass (if requested)
        if capture_backward and target_grad is not None:
            output.backward(target_grad)
            self.original_input_grad = x_clone.grad.clone() if x_clone.grad is not None else None

        # Store the captured output for test comparison
        self.original_output = output.detach()

        self._remove_hooks()
        return output.detach()

    def _make_forward_capture_hook(self, name: str, module: nn.Module):
        """Create hook to capture forward activation."""
        def hook(mod, inp, out):
            # Track call count to detect reused modules
            count = self._capture_fwd_counts.get(name, 0) + 1
            self._capture_fwd_counts[name] = count

            # Only store data on first call; mark as multi-call if reused
            if count > 1:
                self._multi_call_modules.add(name)
                return

            # Handle modules that return tuples (e.g., DecomposedMultiheadAttention → (output, None))
            act = out[0] if isinstance(out, tuple) else out
            if not isinstance(act, torch.Tensor):
                return  # Skip non-tensor outputs (e.g., shape arithmetic)
            self._layer_data[name].original_activation = act.detach().clone()
            self._layer_data[name].input_shape = inp[0].shape if inp else None

            # Capture masks for nonlinear operations
            if isinstance(mod, nn.ReLU):
                # ReLU mask: where input >= 0
                if inp and inp[0] is not None:
                    self._layer_data[name].relu_mask = (inp[0] >= 0).detach()
            elif isinstance(mod, nn.MaxPool2d):
                # MaxPool indices: run maxpool again with return_indices=True
                if inp and inp[0] is not None:
                    _, indices = F.max_pool2d(
                        inp[0], mod.kernel_size, mod.stride,
                        mod.padding, return_indices=True
                    )
                    self._layer_data[name].maxpool_indices = indices.detach()
                    self._layer_data[name].maxpool_kernel_size = mod.kernel_size
                    self._layer_data[name].maxpool_stride = mod.stride
                    self._layer_data[name].maxpool_padding = mod.padding
            elif isinstance(mod, nn.MaxPool1d):
                if inp and inp[0] is not None:
                    _, indices = F.max_pool1d(
                        inp[0], mod.kernel_size, mod.stride,
                        mod.padding, return_indices=True
                    )
                    self._layer_data[name].maxpool_indices = indices.detach()
                    self._layer_data[name].maxpool_kernel_size = mod.kernel_size
                    self._layer_data[name].maxpool_stride = mod.stride
                    self._layer_data[name].maxpool_padding = mod.padding
        return hook

    def _make_backward_capture_hook(self, name: str):
        """Create hook to capture backward gradients for all inputs."""
        def hook(module, grad_in, grad_out):
            # Capture all input gradients (for multi-input ops like matmul, mul)
            gradients = []
            for g in grad_in:
                if g is not None:
                    gradients.append(g.detach().clone())
                else:
                    gradients.append(None)
            self._layer_data[name].original_gradients = gradients
            # original_gradient: the gradient of the first input
            if grad_in[0] is not None:
                self._layer_data[name].original_gradient = grad_in[0].detach().clone()
        return hook

    def _should_track(self, module: nn.Module) -> bool:
        """Check if module should be tracked."""
        # Import Add here to avoid circular imports
        try:
            from .operations.add import Add
            add_type = (Add,)
        except ImportError:
            add_type = ()

        # Import custom modules
        try:
            from .operations.mul import DCMul
            from .operations.mean import Mean
            from .operations.matmul import DCMatMul
            mul_mean_matmul_types = (DCMul, Mean, DCMatMul)
        except ImportError:
            mul_mean_matmul_types = ()

        return isinstance(module, (
            nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d,
            nn.ReLU, nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm,
            nn.MaxPool1d, nn.MaxPool2d, nn.AvgPool1d, nn.AvgPool2d,
            nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
            nn.Flatten, nn.Dropout, nn.Softmax,
        ) + add_type + mul_mean_matmul_types)

    # =========================================================================
    # Phase 2: Attach cache to model for DC pass
    # =========================================================================

    def attach_to_model(self, model: nn.Module) -> None:
        """
        Attach this cache to all modules in the model.

        This sets the DC_ALIGNMENT_CACHE attribute on each module so that
        patched forward methods can access the cache.
        """
        for name, module in model.named_modules():
            setattr(module, DC_ALIGNMENT_CACHE, self)
            # Store the module's name for lookup
            setattr(module, DC_CACHE_LAYER_NAME, name)

    def detach_from_model(self, model: nn.Module) -> None:
        """Remove cache references from model."""
        for name, module in model.named_modules():
            if hasattr(module, DC_ALIGNMENT_CACHE):
                delattr(module, DC_ALIGNMENT_CACHE)
            if hasattr(module, DC_CACHE_LAYER_NAME):
                delattr(module, DC_CACHE_LAYER_NAME)

    # =========================================================================
    # Alignment Functions (called by patched modules)
    # =========================================================================

    def align_forward(
        self,
        layer_name: str,
        pos: Tensor,
        neg: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Align pos/neg so that pos - neg = original_activation.

        The correction maintains pos and neg as non-negative while shifting
        them to match the original activation.

        Args:
            layer_name: Name of the layer
            pos: Positive component [batch, ...]
            neg: Negative component [batch, ...]

        Returns:
            (aligned_pos, aligned_neg) tuple
        """
        if self.mode not in (AlignmentMode.FORWARD_ONLY, AlignmentMode.BOTH):
            return pos, neg

        # Skip alignment for modules called multiple times (shape may differ)
        if layer_name in self._multi_call_modules:
            return pos, neg

        data = self._layer_data.get(layer_name)
        if data is None or data.original_activation is None:
            return pos, neg

        original = data.original_activation
        current = pos - neg

        # Compute correction needed
        correction = original - current

        # Track correction norm
        data.forward_correction_norm = correction.norm().item()

        # Apply correction: add positive part to pos, negative part to neg
        # This maintains pos, neg >= 0 while achieving pos - neg = original
        aligned_pos = pos + F.relu(correction)
        aligned_neg = neg + F.relu(-correction)

        return aligned_pos, aligned_neg

    def align_backward(
        self,
        layer_name: str,
        delta_pp: Tensor,
        delta_np: Tensor,
        delta_pn: Tensor,
        delta_nn: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Align sensitivities so that pp - np - pn + nn = original_gradient.

        Args:
            layer_name: Name of the layer
            delta_pp, delta_np, delta_pn, delta_nn: 4 sensitivities

        Returns:
            (aligned_pp, aligned_np, aligned_pn, aligned_nn) tuple
        """
        if self.mode not in (AlignmentMode.BACKWARD_ONLY, AlignmentMode.BOTH):
            return delta_pp, delta_np, delta_pn, delta_nn

        # Skip alignment for modules called multiple times
        if layer_name in self._multi_call_modules:
            return delta_pp, delta_np, delta_pn, delta_nn

        data = self._layer_data.get(layer_name)
        if data is None or data.original_gradient is None:
            return delta_pp, delta_np, delta_pn, delta_nn

        original_grad = data.original_gradient
        current_grad = delta_pp - delta_np - delta_pn + delta_nn

        # Compute correction needed
        correction = original_grad - current_grad

        # Track correction norm
        data.backward_correction_norm = correction.norm().item()

        # grad = pp - np - pn + nn
        aligned_pp = delta_pp + 0.25 * correction
        aligned_np = delta_np - 0.25 * correction
        aligned_pn = delta_pn - 0.25 * correction
        aligned_nn = delta_nn + 0.25 * correction

        return aligned_pp, aligned_np, aligned_pn, aligned_nn

    def align_backward_indexed(
        self,
        layer_name: str,
        input_index: int,
        delta_pp: Tensor,
        delta_np: Tensor,
        delta_pn: Tensor,
        delta_nn: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Align sensitivities for a specific input of a multi-input operation.

        For two-input operations (matmul, mul), each input has its own
        original gradient. This method aligns against the correct one.

        Args:
            layer_name: Name of the layer
            input_index: Which input's gradient to align against (0 or 1)
            delta_pp, delta_np, delta_pn, delta_nn: 4 sensitivities

        Returns:
            (aligned_pp, aligned_np, aligned_pn, aligned_nn) tuple
        """
        if self.mode not in (AlignmentMode.BACKWARD_ONLY, AlignmentMode.BOTH):
            return delta_pp, delta_np, delta_pn, delta_nn

        # Skip alignment for modules called multiple times
        if layer_name in self._multi_call_modules:
            return delta_pp, delta_np, delta_pn, delta_nn

        data = self._layer_data.get(layer_name)
        if data is None:
            return delta_pp, delta_np, delta_pn, delta_nn

        # Get the correct original gradient for this input index
        if input_index < len(data.original_gradients):
            original_grad = data.original_gradients[input_index]
        elif input_index == 0 and data.original_gradient is not None:
            # Fallback to original_gradient for single-input compat
            original_grad = data.original_gradient
        else:
            return delta_pp, delta_np, delta_pn, delta_nn

        if original_grad is None:
            return delta_pp, delta_np, delta_pn, delta_nn

        current_grad = delta_pp - delta_np - delta_pn + delta_nn

        # Compute correction needed
        correction = original_grad - current_grad

        # Track per-input correction norm
        correction_norm = correction.norm().item()
        while len(data.backward_correction_norms) <= input_index:
            data.backward_correction_norms.append(0.0)
        data.backward_correction_norms[input_index] = correction_norm

        # Update overall backward_correction_norm as max across inputs
        data.backward_correction_norm = max(data.backward_correction_norms)

        # grad = pp - np - pn + nn
        aligned_pp = delta_pp + 0.25 * correction
        aligned_np = delta_np - 0.25 * correction
        aligned_pn = delta_pn - 0.25 * correction
        aligned_nn = delta_nn + 0.25 * correction

        return aligned_pp, aligned_np, aligned_pn, aligned_nn

    # =========================================================================
    # Cached Mask Access (for backward-only mode)
    # =========================================================================

    def get_relu_mask(self, layer_name: str) -> Optional[Tensor]:
        """Get cached ReLU mask for backward-only mode."""
        data = self._layer_data.get(layer_name)
        if data is not None:
            return data.relu_mask
        return None

    def get_maxpool_indices(self, layer_name: str) -> Optional[Tensor]:
        """Get cached MaxPool indices for backward-only mode."""
        data = self._layer_data.get(layer_name)
        if data is not None:
            return data.maxpool_indices
        return None

    def has_cached_mask(self, layer_name: str) -> bool:
        """Check if this layer has cached masks available."""
        data = self._layer_data.get(layer_name)
        if data is None:
            return False
        return data.relu_mask is not None or data.maxpool_indices is not None

    def should_use_cached_mask(self, layer_name: str) -> bool:
        """Check if backward-only mode with cached masks is active for this layer."""
        return (
            self.mode == AlignmentMode.BACKWARD_ONLY
            and self.has_cached_mask(layer_name)
        )

    # =========================================================================
    # Statistics and Diagnostics
    # =========================================================================

    def get_correction_stats(self) -> List[CorrectionStats]:
        """Get correction statistics for all layers."""
        stats = []
        for name in self._layer_order:
            data = self._layer_data.get(name)
            if data is None:
                continue

            # Compute relative corrections
            fwd_rel = 0.0
            bwd_rel = 0.0

            if data.original_activation is not None:
                orig_norm = data.original_activation.norm().item()
                if orig_norm > 1e-10:
                    fwd_rel = data.forward_correction_norm / orig_norm

            if data.backward_correction_norms:
                # Multi-input: compute max relative correction across inputs
                for i, corr_norm in enumerate(data.backward_correction_norms):
                    if i < len(data.original_gradients) and data.original_gradients[i] is not None:
                        orig_norm = data.original_gradients[i].norm().item()
                        if orig_norm > 1e-10:
                            bwd_rel = max(bwd_rel, corr_norm / orig_norm)
            elif data.original_gradient is not None:
                orig_norm = data.original_gradient.norm().item()
                if orig_norm > 1e-10:
                    bwd_rel = data.backward_correction_norm / orig_norm

            stats.append(CorrectionStats(
                layer_name=name,
                forward_correction_norm=data.forward_correction_norm,
                backward_correction_norm=data.backward_correction_norm,
                forward_relative_correction=fwd_rel,
                backward_relative_correction=bwd_rel,
            ))

        return stats

    def get_layer_data(self, layer_name: str) -> Optional[LayerAlignmentData]:
        """Get cached data for a specific layer."""
        return self._layer_data.get(layer_name)

    def get_layer_names(self) -> List[str]:
        """Get list of tracked layer names in order."""
        return list(self._layer_order)

    def __repr__(self) -> str:
        return f"AlignmentCache(mode={self.mode.value}, layers={len(self._layer_order)})"
