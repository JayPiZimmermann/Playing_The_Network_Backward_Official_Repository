"""Shared utilities for layer-level attribution hooks.

Provides ``LayerGradientCapture``: hook into specific layers, capture
activations and gradients for per-layer relevance maps.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from bwgame.operations.matmul import DCMatMul
from bwgame.operations.mul import DCMul
from .postprocess import tensor_to_numpy

logger = logging.getLogger(__name__)


def _detect_num_prefix_tokens(seq_len: int) -> int:
    """Detect number of prefix tokens (CLS + registers) from sequence length.

    Tries common prefix counts: 1 (CLS only), 5 (CLS + 4 registers for
    DINOv2 reg4), 2 (CLS + 1 register).
    Returns the prefix count if seq_len = prefix + grid² for grid >= 4,
    otherwise returns -1.
    """
    for prefix in (1, 5, 2):
        n = seq_len - prefix
        if n < 16:
            continue
        grid = round(n ** 0.5)
        if grid * grid == n and grid >= 4:
            return prefix
    return -1


def _is_attention_shaped(shape) -> bool:
    """Check if a 4D shape is (B, H, S, S) attention matrix (not conv)."""
    if len(shape) != 4:
        return False
    _, _, h, w = shape
    if h != w:
        return False
    return _detect_num_prefix_tokens(h) >= 0


def _is_token_seq_len(n: int) -> bool:
    """Check if n = prefix + grid² (prefix tokens + patches) for grid >= 4."""
    return _detect_num_prefix_tokens(n) >= 0


def _find_module(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Resolve a dotted module name to the actual nn.Module."""
    parts = name.split(".")
    mod = model
    for p in parts:
        if hasattr(mod, p):
            mod = getattr(mod, p)
        else:
            return None
    return mod


def _activation_to_spatial(act: Tensor) -> Tensor:
    """Convert an activation/gradient tensor to a signed (B, H, W) spatial map.

    Preserves sign so the rendering pipeline can produce both signed
    (diverging blue/red) and magnitude (abs) views.

    Handles:
    - (B, H, S, S) attention matrix -> sum heads, CLS row → patch grid
    - (B, H, S, D) multi-head tokens -> sum heads, then token reduction
    - (B, H, D, S) transposed multi-head -> transpose, sum heads, then token reduction
    - (B, C, H, W) conv features -> sum(dim=1)
    - (B, seq_len, D) token sequences -> drop CLS, sum(dim=-1), reshape.
      If patch tokens are near-zero (e.g. post-attention in last ViT block
      where gradient flows only through CLS), adds CLS as uniform fallback.
    - (B, D) flat features -> unsqueeze to (B, 1, 1)
    """
    # 4D: attention matrix, multi-head features, or conv features
    if act.ndim == 4:
        if _is_attention_shaped(act.shape):
            # Attention matrix (B, H, S, S): sum heads, extract CLS row → patch grid
            S = act.shape[2]
            prefix = _detect_num_prefix_tokens(S)
            pooled = act.sum(dim=1)       # (B, S, S)
            cls_row = pooled[:, 0, prefix:]  # (B, n_patches)
            grid = round(cls_row.shape[1] ** 0.5)
            return cls_row.reshape(-1, grid, grid)

        _, _, d2, d3 = act.shape
        if _is_token_seq_len(d2):
            # Multi-head token features (B, H, S, D): sum over heads → (B, S, D)
            act = act.sum(dim=1)
            # Fall through to 3D processing below
        elif _is_token_seq_len(d3):
            # Transposed multi-head (B, H, D, S): transpose and sum heads → (B, S, D)
            act = act.transpose(2, 3).sum(dim=1)
            # Fall through to 3D processing below
        else:
            return act.sum(dim=1)  # Conv features (B, C, H, W) → (B, H, W)

    # 3D: token sequence (B, S, D) or spatial (B, H, W)
    if act.ndim == 3:
        B, S, D = act.shape
        prefix = _detect_num_prefix_tokens(S)
        if prefix < 0:
            prefix = 1  # fallback: assume single CLS token
        tokens = act[:, prefix:]  # drop prefix tokens (CLS + registers)
        n = S - prefix
        if n == 0:
            # Only CLS token (e.g. after class token selection) — return scalar per batch
            return act.sum(dim=-1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        grid = int(n ** 0.5)
        if grid * grid == n:
            patch_map = tokens.sum(dim=-1).reshape(B, grid, grid)
            # If patches are near-zero, add CLS as uniform fallback
            if patch_map.numel() == 0 or patch_map.abs().max() < 1e-7:
                cls_val = act[:, :1].sum(dim=-1).unsqueeze(-1)  # (B, 1, 1)
                return patch_map + cls_val
            return patch_map
        else:
            # Non-square: try without dropping CLS
            grid = int(S ** 0.5)
            if grid * grid == S:
                return act.sum(dim=-1).reshape(B, grid, grid)
            return act.sum(dim=-1).unsqueeze(1)  # (B, 1, S)

    if act.ndim == 2:
        return act.sum(dim=-1).unsqueeze(-1).unsqueeze(-1)

    return act.flatten(1).sum(dim=1).unsqueeze(-1).unsqueeze(-1)


class LayerGradientCapture:
    """Hook into selected layers to capture **input** activation gradients.

    Each layer's map shows the gradient flowing *into* that module, so
    every node (including Flatten, Dropout, etc.) displays the spatial
    structure of the signal arriving at it.

    Usage::

        capture = LayerGradientCapture(model, layer_names)
        capture.install()
        output = model(x)
        output.backward(gradient=one_hot)
        layer_maps = capture.get_maps()
        capture.remove()
    """

    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._captured: Dict[str, Tensor] = {}
        self._matmul_grads: Dict[str, Tuple[Optional[Tensor], Optional[Tensor]]] = {}

    def install(self) -> "LayerGradientCapture":
        self._hooks.clear()
        self._captured.clear()
        self._matmul_grads.clear()

        for name in self.layer_names:
            module = _find_module(self.model, name)
            if module is None:
                continue

            is_dual_input = isinstance(module, (DCMatMul, DCMul))

            # Forward hook: capture input tensors (and retain_grad for non-dual-input)
            def _make_fwd_hook(layer_name, _is_dual_input):
                def _hook(mod, inp, out):
                    if _is_dual_input and isinstance(inp, tuple) and len(inp) >= 2:
                        for i, t in enumerate(inp[:2]):
                            if isinstance(t, Tensor) and t.requires_grad:
                                t.retain_grad()
                            suffix = 'A' if i == 0 else 'B'
                            self._captured[f"{layer_name}:{suffix}"] = t
                    else:
                        t = inp[0] if isinstance(inp, tuple) else inp
                        if isinstance(t, Tensor) and t.requires_grad:
                            t.retain_grad()
                        self._captured[layer_name] = t
                return _hook

            h = module.register_forward_hook(_make_fwd_hook(name, is_dual_input))
            self._hooks.append(h)

            # Backward hook for dual-input modules: reliably captures both input gradients
            if is_dual_input:
                def _make_bwd_hook(layer_name):
                    def _bwd_hook(mod, grad_input, grad_output):
                        ga = grad_input[0] if len(grad_input) > 0 else None
                        gb = grad_input[1] if len(grad_input) > 1 else None
                        if ga is not None:
                            ga = ga.detach()
                        if gb is not None:
                            gb = gb.detach()
                        self._matmul_grads[layer_name] = (ga, gb)
                    return _bwd_hook

                hb = module.register_full_backward_hook(_make_bwd_hook(name))
                self._hooks.append(hb)

        return self

    def get_maps(self, mode: str = 'gradient') -> Dict[str, np.ndarray]:
        """Extract spatial relevance maps from captured input activations.

        Args:
            mode: How to compute the map from captured activation ``act``:
                ``'gradient'`` — ``act.grad`` (default, plain gradient)
                ``'gradient_abs'`` — ``|act.grad|``
                ``'gradient_x_act'`` — ``act.grad * act``
        """
        results = {}

        # For dual-input modules (matmul, mul), prefer backward-hook gradients (more reliable)
        for base_name, (ga, gb) in self._matmul_grads.items():
            for suffix, grad in [('A', ga), ('B', gb)]:
                key = f"{base_name}:{suffix}"
                act = self._captured.get(key)
                if grad is None:
                    continue
                if mode == 'gradient_x_act' and act is not None:
                    data = grad * act.detach()
                elif mode == 'gradient_abs':
                    data = grad.abs()
                else:
                    data = grad
                results[key] = tensor_to_numpy(
                    _activation_to_spatial(data)
                )

        # Non-dual-input modules: use retain_grad as before
        for name, act in self._captured.items():
            if name in results:
                continue  # Already handled by backward hook
            if isinstance(act, Tensor):
                if act.grad is not None:
                    if mode == 'gradient_x_act':
                        data = act.grad * act.detach()
                    elif mode == 'gradient_abs':
                        data = act.grad.abs()
                    else:
                        data = act.grad
                    results[name] = tensor_to_numpy(
                        _activation_to_spatial(data)
                    )
        return results

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._captured.clear()
        self._matmul_grads.clear()
