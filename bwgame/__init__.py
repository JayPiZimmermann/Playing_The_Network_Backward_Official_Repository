"""
DC Decomposition for Neural Networks

This module implements Difference of Convex (DC) decomposition for neural networks.
The key idea: f(x) = g(x) - h(x) where g and h are monotone subnetworks.

Recommended API (context manager):
    from bwgame import dc_decomposition

    with dc_decomposition(model, x) as dc:
        out = dc.output
        loss = criterion(out, target)
        loss.backward()

    grad = dc.reconstruct_gradient()
    sens = dc.sensitivities

Forward format: [4*batch] = [pos; neg; 0; 0]
Backward format: [4*batch] = [delta_pp; delta_np; delta_pn; delta_nn]
"""

# Context manager API (recommended)
from .dc_context import dc_decomposition, DCResult, run_adf_forward

# Patcher-based API (lower-level)
from .patcher import (
    patch_model, unpatch_model,
    mark_output_layer, unmark_output_layer,
    mark_output_layer_by_name,
    auto_mark_output_layer, find_output_layer,
    set_dc_enabled, dc_disabled,
    is_patched, get_patched_layers,
    prepare_model_for_dc,
    enable_dc_logging, disable_dc_logging,
)
from .functional_replacer import make_dc_compatible, replace_functional_with_modules, Mul, Mean

# Core utilities from operations
from .operations.base import (
    init_catted, init_pos_neg, reconstruct_output,
    recenter_dc, InputMode as OpInputMode,
)

# Matrix operations
from .operations.matmul import DCMatMul, DCMatMulFunction

# Operations
from .operations.tensor_ops import DCSplit, DCChunk, DCCat, DCSlice
from .operations.shape_ops import Reshape as DCReshape, Transpose as DCTranspose, Permute as DCPermute
from .operations.add import Add as DCAdd
from .operations.gather import Gather as DCGather
from .operations.mean import Mean as DCMean
from .operations.sum import Sum as DCSum
from .operations.contiguous import Contiguous as DCContiguous

__all__ = [
    # Context manager API (recommended)
    "dc_decomposition",
    "DCResult",
    "run_adf_forward",
    # Patcher API (lower-level)
    "prepare_model_for_dc",
    "enable_dc_logging",
    "disable_dc_logging",
    "patch_model",
    "unpatch_model",
    "mark_output_layer",
    "unmark_output_layer",
    "mark_output_layer_by_name",
    "auto_mark_output_layer",
    "find_output_layer",
    "set_dc_enabled",
    "dc_disabled",
    "is_patched",
    "get_patched_layers",
    "make_dc_compatible",
    "replace_functional_with_modules",
    # Core utilities
    "init_catted",
    "init_pos_neg",
    "reconstruct_output",
    "recenter_dc",
    "InputMode",
    # Functional replacer modules
    "Mul",
    "Mean",
    # MatMul
    "DCMatMul",
    "DCMatMulFunction",
    # Operations
    "DCReshape",
    "DCPermute", 
    "DCTranspose",
    "DCContiguous",
    "DCAdd",
    "DCSplit",
    "DCChunk",
    "DCCat", 
    "DCSlice",
    "DCGather",
    "DCMean",
    "DCSum",
]
