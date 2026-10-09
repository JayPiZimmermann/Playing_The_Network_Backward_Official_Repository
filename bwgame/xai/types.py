"""Core data types for the XAI attribution framework."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np


class MethodCategory(Enum):
    """Category of attribution method."""
    GRADIENT = "gradient"
    CAM = "cam"
    ATTENTION = "attention"
    LRP = "lrp"
    PERTURBATION = "perturbation"
    DC = "dc"
    GAME_THEORY = "game_theory"


@dataclass
class MethodConfig:
    """Configuration metadata for an attribution method.

    Attributes:
        name: Unique registry name of the method (e.g. "routing_game", "gradcam").
        category: Which family the method belongs to.
        class_specific: Whether the method produces class-specific attributions.
        supports_batch: If True, attribute() can handle full (N,C,H,W) batches
            at once.  If False, callers loop over single images.
        requires_grad: Whether the method needs gradient computation.
        per_layer: Whether the method produces per-layer attribution maps.
        description: One-line human-readable description.
        reference: Citation or URL for the original paper.
    """
    name: str
    category: MethodCategory
    class_specific: bool
    supports_batch: bool
    requires_grad: bool = True
    per_layer: bool = False
    requires_attention: bool = False
    description: str = ""
    reference: str = ""


@dataclass
class AttributionResult:
    """Container for the output of a single attribution method.

    Attributes:
        method_name: Name matching ``MethodConfig.name``.
        maps: Dictionary mapping layer names (or ``"input"``) to numpy arrays
            of shape ``(B, H, W)`` where B is the batch size.
        timing: Wall-clock seconds for the ``attribute()`` call.
    """
    method_name: str
    maps: Dict[str, np.ndarray]
    timing: float
