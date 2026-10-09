"""DC Decomposition for AvgPool layers. Forward/Backward: [4*batch] -> [4*batch]

AvgPool is linear, so we apply the same operation to both pos and neg streams.
Since it's linear, we let autograd handle the backward pass naturally.
"""

import torch.nn as nn
from torch import Tensor

from .patch_builder import create_patch_function, create_unpatch_function


# For linear operations like AvgPool, we don't need custom autograd.Function
# We just apply the operation to each component and let autograd handle backward

def dc_forward_avgpool2d(m: nn.AvgPool2d, x: Tensor) -> Tensor:
    """DC forward for AvgPool2d: pass-through using the *original* forward.

    AvgPool is a per-batch-element linear operation, so autograd correctly
    handles the [4*batch] → [4*batch] → [4*batch] backward without any
    explicit pos/neg split.  We rely on the original ``nn.AvgPool2d.forward``
    here so all four DC components remain independent in the backward graph.
    """
    from .base import DC_ORIGINAL_FORWARD
    return getattr(m, DC_ORIGINAL_FORWARD)(x)


def dc_forward_avgpool1d(m: nn.AvgPool1d, x: Tensor) -> Tensor:
    from .base import DC_ORIGINAL_FORWARD
    return getattr(m, DC_ORIGINAL_FORWARD)(x)


def dc_forward_adaptive_avgpool2d(m: nn.AdaptiveAvgPool2d, x: Tensor) -> Tensor:
    from .base import DC_ORIGINAL_FORWARD
    return getattr(m, DC_ORIGINAL_FORWARD)(x)


def dc_forward_adaptive_avgpool1d(m: nn.AdaptiveAvgPool1d, x: Tensor) -> Tensor:
    from .base import DC_ORIGINAL_FORWARD
    return getattr(m, DC_ORIGINAL_FORWARD)(x)


# Patch/unpatch functions
patch_avgpool2d = create_patch_function(dc_forward_avgpool2d)
patch_avgpool1d = create_patch_function(dc_forward_avgpool1d)
patch_adaptive_avgpool2d = create_patch_function(dc_forward_adaptive_avgpool2d)
patch_adaptive_avgpool1d = create_patch_function(dc_forward_adaptive_avgpool1d)

unpatch_avgpool2d = create_unpatch_function()
unpatch_avgpool1d = create_unpatch_function()
unpatch_adaptive_avgpool2d = create_unpatch_function()
unpatch_adaptive_avgpool1d = create_unpatch_function()
