"""ViT model utilities for attribution methods.

Provides helpers to:
- Capture attention weights from ``DecomposedMultiheadAttention`` modules
- Identify encoder blocks (in forward order) and target layers for CAM methods
- Reshape ViT token sequences to spatial grids
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..composite_replacer import DecomposedMultiheadAttention


class AttentionWeightCapture:
    """Context manager that hooks ``DecomposedMultiheadAttention`` modules to
    capture attention weights (the softmax output, before dropout).

    The ``attn_dropout`` child of each ``DecomposedMultiheadAttention`` receives
    the attention weights as its input.  We register a forward hook on
    ``attn_dropout`` that stores ``input[0]`` -- the post-softmax, pre-dropout
    attention weight tensor of shape ``(B, num_heads, seq_len, seq_len)``.

    Usage::

        with AttentionWeightCapture(model) as cap:
            output = model(x)
            weights = cap.get_weights()
            # weights: {"encoder.layers.encoder_layer_0.self_attention": tensor}

    The dictionary keys are the *parent* ``DecomposedMultiheadAttention``
    module names (which identify the encoder block).
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._weights: Dict[str, Tensor] = {}

    def __enter__(self) -> "AttentionWeightCapture":
        self._weights.clear()
        self._hooks.clear()

        for parent_name, parent_module in self.model.named_modules():
            if not isinstance(parent_module, DecomposedMultiheadAttention):
                continue

            dropout_module = parent_module.attn_dropout

            # Use a default-arg closure to capture *parent_name* by value.
            def _make_hook(name: str):
                def _hook(module, input, output):
                    # input[0] is the attention weight tensor after softmax,
                    # shape (B, num_heads, seq_len, seq_len).
                    self._weights[name] = input[0].detach()
                return _hook

            handle = dropout_module.register_forward_hook(_make_hook(parent_name))
            self._hooks.append(handle)

        return self

    def get_weights(self) -> Dict[str, Tensor]:
        """Return captured attention weights.

        Returns:
            Dictionary mapping parent module name to attention weight tensor
            of shape ``(B, num_heads, seq_len, seq_len)``.
        """
        return dict(self._weights)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._weights.clear()
        return False


def get_vit_encoder_blocks(
    model: nn.Module,
) -> List[Tuple[str, nn.Module]]:
    """Return a list of ``(name, module)`` for each encoder block in a ViT.

    Works with ``torchvision.models.VisionTransformer`` which has structure::

        encoder.layers.encoder_layer_0
        encoder.layers.encoder_layer_1
        ...

    Returns:
        Ordered list of ``(fully_qualified_name, module)`` pairs.
    """
    blocks = []
    for name, module in model.named_modules():
        # torchvision ViT: "encoder.layers.encoder_layer_N"
        if "encoder_layer_" in name and name.count(".") == 2:
            blocks.append((name, module))
        # timm ViT (DINO, DINOv2): "blocks.N"
        elif name.startswith("blocks.") and name.count(".") == 1:
            try:
                int(name.split(".")[1])  # verify it's blocks.0, blocks.1, etc.
                blocks.append((name, module))
            except ValueError:
                pass
    return blocks


def order_by_encoder_block(names, model: nn.Module) -> List[str]:
    """Sort module names by the forward position of their encoder block.

    Each name must equal, or lie inside, one of the blocks returned by
    :func:`get_vit_encoder_blocks` (matched on whole dotted path components,
    so ``encoder_layer_1`` does not match ``encoder_layer_10``).

    Returns:
        The names ordered from the first (input-side) to the last block.

    Raises:
        ValueError: if a name does not belong to any encoder block.
    """
    blocks = [name for name, _ in get_vit_encoder_blocks(model)]

    def block_index(name: str) -> int:
        for i, block in enumerate(blocks):
            if name == block or name.startswith(block + "."):
                return i
        raise ValueError(f"{name!r} is not inside a ViT encoder block")

    return sorted(names, key=block_index)


def vit_reshape_transform(
    tensor: Tensor,
    height: int = 14,
    width: int = 14,
) -> Tensor:
    """Reshape a ViT encoder output from token-sequence to spatial grid.

    Drops the CLS token (position 0) and reshapes the remaining patch tokens.
    If the tensor shape is incompatible (e.g. bias gradients from FullGrad),
    returns the tensor unchanged or with a best-effort reshape.

    Args:
        tensor: Shape ``(B, seq_len, D)`` where ``seq_len = 1 + H*W``.
        height: Spatial height of the patch grid.
        width: Spatial width of the patch grid.

    Returns:
        Tensor of shape ``(B, D, height, width)``.
    """
    if tensor.ndim == 4:
        # Already spatial (B, C, H, W) — return as-is
        return tensor

    if tensor.ndim != 3:
        # Unexpected shape — try to make it work
        return tensor.unsqueeze(-1).unsqueeze(-1) if tensor.ndim == 2 else tensor

    B, seq_len, D = tensor.shape
    n_patches = height * width  # expected: 196

    if seq_len == n_patches + 1:
        # Standard case: has CLS token, drop it
        result = tensor[:, 1:, :]
    elif seq_len == n_patches:
        # No CLS token
        result = tensor
    else:
        # Non-standard sequence length — infer grid from actual size
        # Try dropping N prefix tokens (CLS, registers) for N = 1..8
        result = None
        for n_prefix in range(0, 9):
            tokens = seq_len - n_prefix
            grid = int(tokens ** 0.5)
            if grid * grid == tokens:
                result = tensor[:, n_prefix:, :]
                height, width = grid, grid
                break
        if result is None:
            # Cannot reshape to square grid — return (B, D, 1, seq_len)
            return tensor.permute(0, 2, 1).unsqueeze(2)

    result = result.reshape(B, height, width, -1)  # (B, H, W, D)
    result = result.permute(0, 3, 1, 2)  # (B, D, H, W)
    return result


def get_vit_target_layers_for_cam(
    model: nn.Module,
) -> List[Tuple[str, nn.Module]]:
    """Return target layers for GradCAM-family methods on a ViT.

    For Vision Transformers, the recommended target is ``ln_1``
    (pre-attention LayerNorm) within each encoder block. This layer
    has non-trivial gradients that survive the ReLU in LayerCAM/GradCAM,
    whereas ``ln_2`` (pre-MLP) often produces all-zero maps.

    Returns:
        List of ``(layer_name, module)`` pairs for all encoder blocks.
    """
    targets = []
    for block_name, block_module in get_vit_encoder_blocks(model):
        # torchvision: ln_1, timm: norm1
        if hasattr(block_module, "ln_1"):
            targets.append((f"{block_name}.ln_1", block_module.ln_1))
        elif hasattr(block_module, "norm1"):
            targets.append((f"{block_name}.norm1", block_module.norm1))
    return targets


def get_cnn_target_layers_for_cam(
    model: nn.Module,
) -> List[Tuple[str, nn.Module]]:
    """Return target layers for GradCAM-family methods on CNNs (e.g. ResNet).

    Looks for named children that are nn.Sequential blocks (like layer1..layer4
    in ResNet).  If none found, falls back to the last Conv2d in the model.

    Returns:
        List of ``(layer_name, module)`` pairs.
    """
    # Strategy 1: Find Sequential blocks named "layerN" (ResNet pattern)
    sequential_blocks = []
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential) and len(list(module.children())) > 0:
            # Check if this sequential contains conv layers (i.e., it's a residual stage)
            has_conv = any(
                isinstance(m, nn.Conv2d) for m in module.modules()
            )
            if has_conv:
                sequential_blocks.append((name, module))

    if sequential_blocks:
        # Return the last block (deepest features, best for CAM)
        last_name, last_block = sequential_blocks[-1]
        return [(last_name, last_block)]

    # Strategy 2: Fall back to last Conv2d layer
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = (name, module)
    if last_conv is not None:
        return [last_conv]

    return []


def get_target_layers_for_cam(
    model: nn.Module,
) -> Tuple[List[Tuple[str, nn.Module]], bool]:
    """Return target layers and whether the model is a ViT.

    Tries ViT target layers first; if empty, falls back to CNN target layers.

    Returns:
        (target_layers, is_vit): List of (name, module) pairs and a boolean
        indicating whether the model appears to be a ViT.
    """
    vit_layers = get_vit_target_layers_for_cam(model)
    if vit_layers:
        return vit_layers, True
    cnn_layers = get_cnn_target_layers_for_cam(model)
    return cnn_layers, False
