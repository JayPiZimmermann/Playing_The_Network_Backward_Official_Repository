"""Plain-PyTorch network copies for ``captum.attr.DeepLift``.

captum's DeepLift (Shrikumar et al. 2017, Rescale rule) attaches its
multipliers to the non-linear modules (``nn.ReLU``, ``nn.MaxPool2d``, ...)
through hooks.  It therefore needs

* the standard ``forward`` of every module: the evaluator replaces module
  forwards by torch.fx-traced versions that route functional operations
  through added ``_dc_*`` modules (``bwgame.functional_replacer``);
* one module instance per application of a non-linearity: torchvision's
  ResNet blocks apply their single ``self.relu`` module two or three times.

:func:`deeplift_network` returns a copy of a model that satisfies both.  The
copy shares every parameter and buffer with the original (no weights are
copied), so it computes the same function as the model it was made from.

Max-pooling: captum reads the multipliers of a module's output with a hook
on the output tensor and replaces the gradient of a module's input with a hook
on the input tensor.  When a ReLU feeds a max-pool directly (all max-pools of
VGG-16 and ResNet-50), both hooks sit on the same tensor and the reading hook
runs first, so the ReLU's Rescale multipliers are combined with the gradient
of the max-pool.  This is the max-pool treatment of the DeepLIFT reference
implementation (kundajelab/deeplift, default ``MaxPoolDeepLiftMode.gradient``);
summation-to-delta is then exact where the reference activations are
spatially constant within the pooling windows.
"""

from __future__ import annotations

import copy
import itertools

import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock, Bottleneck

from bwgame.operations.matmul import DCMatMul
from bwgame.operations.mul import DCMul

# Modules whose computation is not covered by DeepLIFT's Rescale rule in
# captum (products of two activations, token softmax, layer normalisation).
_UNSUPPORTED = (nn.MultiheadAttention, nn.LayerNorm, DCMatMul, DCMul)


class _BasicBlock(nn.Module):
    """torchvision ``BasicBlock`` with a separate ReLU per application."""

    def __init__(self, block: BasicBlock):
        super().__init__()
        self.conv1, self.bn1 = block.conv1, block.bn1
        self.conv2, self.bn2 = block.conv2, block.bn2
        self.downsample = block.downsample
        self.relu1, self.relu2 = nn.ReLU(), nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu2(out + identity)


class _Bottleneck(nn.Module):
    """torchvision ``Bottleneck`` with a separate ReLU per application."""

    def __init__(self, block: Bottleneck):
        super().__init__()
        self.conv1, self.bn1 = block.conv1, block.bn1
        self.conv2, self.bn2 = block.conv2, block.bn2
        self.conv3, self.bn3 = block.conv3, block.bn3
        self.downsample = block.downsample
        self.relu1, self.relu2, self.relu3 = nn.ReLU(), nn.ReLU(), nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu3(out + identity)


_BLOCKS = {BasicBlock: _BasicBlock, Bottleneck: _Bottleneck}


def check_deeplift_support(model: nn.Module) -> None:
    """Raise ``NotImplementedError`` for architectures outside DeepLIFT's scope."""
    bad = sorted({type(m).__name__ for m in model.modules() if isinstance(m, _UNSUPPORTED)})
    if bad:
        raise NotImplementedError(
            "DeepLIFT (captum.attr.DeepLift, Rescale rule) is only supported for "
            "convolutional ReLU networks here; the model contains "
            f"{', '.join(bad)}.  The Rescale rule covers element-wise "
            "non-linearities, not products of activations (attention) or "
            "normalisation by activation statistics (LayerNorm)."
        )


_HOOK_DICTS = ("_forward_hooks", "_forward_hooks_with_kwargs", "_forward_hooks_always_called",
               "_forward_pre_hooks", "_forward_pre_hooks_with_kwargs",
               "_backward_hooks", "_backward_pre_hooks")


def plain_copy(model: nn.Module) -> nn.Module:
    """Copy of ``model`` with the class-defined module forwards and no hooks.

    Instance-level ``forward`` overrides and the ``_dc_*`` helper modules they
    use are dropped; parameters and buffers are shared with ``model``.
    """
    memo = {id(t): t for t in itertools.chain(model.parameters(), model.buffers())}
    for m in model.modules():
        fwd = m.__dict__.get("forward")
        if fwd is not None:
            memo[id(fwd)] = None
    net = copy.deepcopy(model, memo)
    for m in net.modules():
        for key in [k for k in m.__dict__ if k == "forward" or k.startswith("_dc_")]:
            del m.__dict__[key]
        for name in [n for n in m._modules if n.startswith("_dc_")]:
            del m._modules[name]
        for hooks in _HOOK_DICTS:
            if hasattr(m, hooks):
                getattr(m, hooks).clear()
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return net.eval()


def split_block_relus(module: nn.Module) -> None:
    """Replace torchvision ResNet blocks in ``module`` by one-ReLU-per-application blocks."""
    for name, child in list(module.named_children()):
        wrapper = _BLOCKS.get(type(child))
        if wrapper is not None:
            setattr(module, name, wrapper(child))
        else:
            split_block_relus(child)


def deeplift_network(model: nn.Module) -> nn.Module:
    """Network for ``captum.attr.DeepLift`` that computes the same function as ``model``."""
    check_deeplift_support(model)
    net = plain_copy(model)
    split_block_relus(net)
    return net
