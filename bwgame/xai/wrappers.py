"""Equivariant averaging (RG+Eq) around any registered attribution method.

``equivariant_average`` averages the attributions of random spatial
transforms (rotation, translation, horizontal flip) of the input, mapping
each heatmap back to the input coordinates before averaging.  It returns a
maps dictionary in the same format as ``attribute()``
(``Dict[str, np.ndarray]``).
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .registry import MethodRegistry


# =========================================================================
# Spatial transform helpers
# =========================================================================

def _build_affine(angle_deg: float, tx: float, ty: float, flip: bool):
    """Build a 2x3 affine matrix for rotation + translation + optional hflip.

    tx, ty are in normalised grid coordinates [-1, 1].
    Returns (theta, theta_inv).
    """
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    fx = -1.0 if flip else 1.0

    theta = torch.tensor([
        [fx * cos_a, -sin_a, tx],
        [fx * sin_a,  cos_a, ty],
    ], dtype=torch.float32)

    cos_neg, sin_neg = math.cos(-rad), math.sin(-rad)
    tx_inv = -(cos_neg * tx + (-sin_neg) * ty)
    ty_inv = -(sin_neg * tx + cos_neg * ty)
    theta_inv = torch.tensor([
        [fx * cos_neg, fx * (-sin_neg), fx * tx_inv],
        [sin_neg,      cos_neg,         ty_inv],
    ], dtype=torch.float32)

    return theta, theta_inv


def _apply_affine(x: Tensor, theta: Tensor) -> Tensor:
    """Apply a 2x3 affine to a (B, C, H, W) tensor."""
    theta_b = theta.unsqueeze(0).expand(x.size(0), -1, -1).to(x.device)
    grid = F.affine_grid(theta_b, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False,
                         padding_mode='reflection')


def _apply_affine_to_heatmap(hmap: np.ndarray, theta: Tensor) -> np.ndarray:
    """Apply affine transform to a (B, H, W) numpy heatmap."""
    t = torch.from_numpy(hmap).float().unsqueeze(1)  # (B, 1, H, W)
    theta_b = theta.unsqueeze(0).expand(t.size(0), -1, -1)
    grid = F.affine_grid(theta_b, t.size(), align_corners=False)
    result = F.grid_sample(t, grid, align_corners=False, padding_mode='zeros')
    return result.squeeze(1).numpy()


# Keys that belong to the wrapper, not to the inner method
_WRAPPER_KEYS = frozenset({
    'equivariant_averaging', 'equivariant_n_samples',
    'equivariant_max_angle', 'equivariant_max_shift_frac',
    'equivariant_flip_prob',
})


def _inner_params(method_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Filter out wrapper-specific keys, keep only inner method params."""
    if not method_params:
        return {}
    return {k: v for k, v in method_params.items() if k not in _WRAPPER_KEYS}


def _make_method(method_name, model, device, params):
    """Instantiate a method and apply params."""
    method = MethodRegistry.instantiate(method_name, model, device)
    for k, v in params.items():
        if hasattr(method, k):
            setattr(method, k, v)
    return method


# =========================================================================
# equivariant_average
# =========================================================================

def equivariant_average(
    method_name: str,
    model: nn.Module,
    device: torch.device,
    x: Tensor,
    target: Tensor,
    layers: Optional[List[str]] = None,
    n_samples: int = 50,
    max_angle: float = 20.0,
    max_shift_frac: float = 0.1,
    flip_prob: float = 0.5,
    method_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, np.ndarray]:
    """Average an attribution method over random spatial transforms.

    For each sample: apply random rotation + translation + optional flip,
    compute attribution, inverse-transform heatmaps, accumulate.

    Args:
        method_name: Registered attribution method name.
        model: Neural network model.
        device: Torch device.
        x: Input image (1, C, H, W).
        target: Target class (1,).
        layers: Optional layer names for per-layer attribution.
        n_samples: Number of random transform samples.
        max_angle: Max rotation in degrees.
        max_shift_frac: Max translation as fraction of image extent.
        flip_prob: Probability of horizontal flip per sample.
        method_params: Parameters to set on each method instance.
    """
    params = _inner_params(method_params)
    accumulated: Dict[str, np.ndarray] = {}

    for _ in range(n_samples):
        angle = random.uniform(-max_angle, max_angle) if max_angle > 0 else 0.0
        tx = random.uniform(-1, 1) * max_shift_frac * 2
        ty = random.uniform(-1, 1) * max_shift_frac * 2
        flip = random.random() < flip_prob
        theta, theta_inv = _build_affine(angle, tx, ty, flip)

        x_t = _apply_affine(x, theta)
        method = _make_method(method_name, model, device, params)
        try:
            maps = method.attribute(x_t, target.expand(1), layers)
        finally:
            method.cleanup()

        # Inverse-transform the heatmap
        for key, hmap in maps.items():
            h_inv = _apply_affine_to_heatmap(hmap[0:1], theta_inv)
            if key in accumulated:
                accumulated[key] = accumulated[key] + h_inv
            else:
                accumulated[key] = h_inv.copy()

    for key in accumulated:
        accumulated[key] = accumulated[key] / n_samples

    return accumulated
