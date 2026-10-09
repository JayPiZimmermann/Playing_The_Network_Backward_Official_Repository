"""Post-processing utilities for attribution maps."""

from typing import Optional

import numpy as np
import torch
from torch import Tensor


def tokens_to_spatial(
    tokens,
    grid_size: Optional[int] = None,
    has_cls: bool = True,
) -> np.ndarray:
    """Convert token-level attributions to spatial map.

    Args:
        tokens: (B, S) tensor or array.  S = grid_size^2 (+ 1 if has_cls).
        grid_size: Patch grid size.  If ``None``, inferred as ``int(sqrt(n_patches))``.
        has_cls: Whether the first token is CLS (dropped).

    Returns:
        (B, grid_size, grid_size) spatial maps as numpy array.
    """
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.detach().cpu().numpy()
    if has_cls:
        tokens = tokens[:, 1:]  # Drop CLS
    B, n_patches = tokens.shape
    if grid_size is None:
        grid_size = int(n_patches ** 0.5)
        if grid_size * grid_size != n_patches:
            # Non-square — return (B, 1, n_patches) as a last resort
            return tokens.reshape(B, 1, n_patches)
    return tokens.reshape(B, grid_size, grid_size)


def tensor_to_numpy(t: Tensor) -> np.ndarray:
    """Safely convert a torch Tensor to numpy."""
    return t.detach().cpu().float().numpy()
