"""Per-layer data cache stored directly on modules.

Provides a capture pass mechanism that runs on the patched model with DC
disabled, populating ``module._dc_layer_data`` on every relevant module.
The DC backward patches read this data via ``ctx._dc_layer_data``: the
original (non-DC) activation at each layer (``activation``) and the ADF
pre-activation mean / variance written by the ADF forward pass.

The capture pass is a single forward pass on the already-patched model
with ``DC_ENABLED=False`` (which routes to the original forward).  No
model copies are made.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import torch
import torch.nn as nn
from torch import Tensor

DC_LAYER_DATA = '_dc_layer_data'


@dataclass
class DCLayerData:
    """Per-layer cached data stored directly on modules.

    For modules that are invoked multiple times during a single forward
    pass (e.g. shared ``nn.ReLU`` in ResNet Bottleneck blocks),
    activations are stored on a stack.  The backward pass pops from the
    stack, which naturally yields the correct activation because autograd
    processes nodes in reverse execution order.
    """

    activation: Optional[Tensor] = None
    """Original (non-DC) activation at this layer's output.
    For modules called once, this is set directly.
    For modules called multiple times, use pop_activation() instead."""

    _activation_stack: Optional[List[Tensor]] = None
    """Stack of activations for modules invoked multiple times."""

    adf_mean: Optional[Tensor] = None
    """Pre-activation mean for activation layers (= input to the activation).
    For modules called multiple times, use pop_adf_stats() instead."""

    adf_variance: Optional[Tensor] = None
    """Pre-activation variance for activation layers (propagated by the ADF forward).
    For modules called multiple times, use pop_adf_stats() instead."""

    _adf_stack: Optional[List[Tuple[Tensor, Tensor]]] = None
    """Stack of (mean, variance) pairs for modules invoked multiple times."""

    def push_activation(self, t: Tensor) -> None:
        """Append an activation to the stack (called during capture pass)."""
        if self._activation_stack is None:
            if self.activation is None:
                # First invocation: store directly
                self.activation = t
                return
            # Second invocation: promote to stack
            self._activation_stack = [self.activation, t]
            self.activation = None
        else:
            self._activation_stack.append(t)

    def pop_activation(self) -> Optional[Tensor]:
        """Pop the most recent activation (called during backward).

        Returns None if no activations remain.
        """
        if self._activation_stack:
            return self._activation_stack.pop()
        # Single-invocation module: consume the activation
        act = self.activation
        self.activation = None
        return act

    def push_adf_stats(self, mean: Tensor, var: Tensor) -> None:
        """Append ADF mean/variance to the stack (called during ADF forward)."""
        if self._adf_stack is None:
            if self.adf_mean is None:
                # First invocation: store directly
                self.adf_mean = mean
                self.adf_variance = var
                return
            # Second invocation: promote to stack
            self._adf_stack = [(self.adf_mean, self.adf_variance), (mean, var)]
            self.adf_mean = None
            self.adf_variance = None
        else:
            self._adf_stack.append((mean, var))

    def pop_adf_stats(self) -> Optional[Tuple[Tensor, Tensor]]:
        """Pop the most recent ADF mean/variance (called during backward).

        Returns None if no stats remain.
        """
        if self._adf_stack:
            return self._adf_stack.pop()
        # Single-invocation module: consume the stats
        if self.adf_mean is not None:
            result = (self.adf_mean, self.adf_variance)
            self.adf_mean = None
            self.adf_variance = None
            return result
        return None


# Layer types that are relevant for activation caching
_CACHEABLE_TYPES: Tuple[type, ...] = (
    nn.Linear, nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d,
    nn.ReLU, nn.GELU, nn.SiLU,
    nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm,
    nn.MaxPool1d, nn.MaxPool2d,
    nn.AvgPool1d, nn.AvgPool2d, nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
    nn.Softmax, nn.Dropout,
)


def _is_cacheable(module: nn.Module) -> bool:
    """Check whether a module should receive a DCLayerData."""
    # Standard PyTorch layers
    if isinstance(module, _CACHEABLE_TYPES):
        return True
    # Custom DC modules (lazy import to avoid circular deps)
    cls_name = type(module).__name__
    if cls_name in ('Add', 'Mul', 'Mean', 'DCMatMul', 'DCMul', 'DCExpand',
                     'Reshape', 'View', 'Squeeze', 'Unsqueeze',
                     'Transpose', 'Permute'):
        return True
    return False


_ACTIVATION_TYPES: Tuple[type, ...] = (nn.ReLU, nn.GELU, nn.SiLU)

def _make_activation_capture_hook(name: str, is_activation: bool = False):
    """Hook that stores the layer's activation.

    For activation functions (ReLU, GELU, SiLU), stores the **input**
    (pre-activation z) so that backward patches can use the original z
    for mask computation instead of the DC-decomposed ``pos - neg``.
    For other layers, stores the output.
    """

    def hook(module: nn.Module, inp: Any, out: Any) -> None:
        ld = getattr(module, DC_LAYER_DATA, None)
        if ld is None:
            return
        if is_activation:
            # Pre-activation: input to ReLU/GELU/SiLU
            x = inp[0] if isinstance(inp, tuple) else inp
            if isinstance(x, Tensor):
                ld.push_activation(x.detach())
        else:
            act = out[0] if isinstance(out, tuple) else out
            if isinstance(act, Tensor):
                ld.push_activation(act.detach())

    return hook


def run_capture_pass(
    model: nn.Module,
    x: Tensor,
    capture_activations: bool = True,
) -> None:
    """Run a forward pass with DC disabled, populating ``_dc_layer_data``.

    After this function returns every relevant module has a
    ``._dc_layer_data`` attribute of type :class:`DCLayerData`.

    Args:
        model: Already-patched model.  ``DC_ENABLED`` will be temporarily
            set to ``False`` and restored afterwards.
        x: Input tensor (normal ``[batch, ...]`` shape).
        capture_activations: If ``True``, store original activations.
    """
    from ..patcher import set_dc_enabled

    # Step 1: Initialise DCLayerData on all relevant modules.
    for _name, module in model.named_modules():
        if _is_cacheable(module):
            setattr(module, DC_LAYER_DATA, DCLayerData())

    # Step 2: Disable DC so the patched model uses original forwards
    set_dc_enabled(model, False)

    hooks: List[torch.utils.hooks.RemovableHandle] = []
    try:
        # Step 3a: Activation capture hooks
        if capture_activations:
            for name, module in model.named_modules():
                if hasattr(module, DC_LAYER_DATA) and name:
                    is_act = isinstance(module, _ACTIVATION_TYPES)
                    hooks.append(
                        module.register_forward_hook(
                            _make_activation_capture_hook(name, is_activation=is_act)
                        )
                    )

        # Step 4: Forward pass (original forwards, hooks fire).
        with torch.no_grad():
            model(x)

    finally:
        # Step 5: Remove hooks and re-enable DC
        for h in hooks:
            h.remove()
        set_dc_enabled(model, True)


def cleanup_layer_data(model: nn.Module) -> None:
    """Remove ``_dc_layer_data`` from all modules."""
    for _name, module in model.named_modules():
        if hasattr(module, DC_LAYER_DATA):
            delattr(module, DC_LAYER_DATA)
