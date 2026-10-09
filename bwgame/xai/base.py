"""Abstract base class for attribution methods."""

import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .types import AttributionResult, MethodConfig


class AttributionMethod(ABC):
    """Base class that every attribution method must subclass.

    Subclasses must implement:
        - ``config()`` (classmethod): return a ``MethodConfig`` describing the method.
        - ``attribute()``: compute attribution maps for a batch of inputs.

    Optionally override ``cleanup()`` to release hooks/buffers after use.
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    @classmethod
    @abstractmethod
    def config(cls) -> MethodConfig:
        """Return the static configuration for this method."""
        ...

    @abstractmethod
    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Compute attribution maps.

        Args:
            x: Input images, shape ``(B, C, H, W)``.
            target: Target class indices, shape ``(B,)``.
            layers: Optional list of layer names to compute attributions for.
                If ``None``, use method-specific defaults.

        Returns:
            Dictionary mapping layer names (or ``"input"``) to numpy arrays
            of shape ``(B, H, W)``.
        """
        ...

    def cleanup(self) -> None:
        """Release any hooks, buffers, or temporary state."""
        pass

    def run(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> AttributionResult:
        """Convenience wrapper: call ``attribute()`` and wrap with timing.

        Returns:
            ``AttributionResult`` with method name, maps, and elapsed time.
        """
        start = time.perf_counter()
        maps = self.attribute(x, target, layers)
        elapsed = time.perf_counter() - start
        return AttributionResult(
            method_name=self.config().name,
            maps=maps,
            timing=elapsed,
        )
