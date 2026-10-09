"""Attribution methods with a decorator-based registry.

Usage:
    from bwgame.xai import MethodRegistry

    method = MethodRegistry.instantiate("routing_game", model, device)
    maps = method.attribute(images, labels)
"""

from .types import MethodConfig, MethodCategory, AttributionResult
from .base import AttributionMethod
from .registry import MethodRegistry

# Import methods subpackage to trigger @MethodRegistry.register decorators
from . import methods as _methods  # noqa: F401

__all__ = [
    "MethodConfig",
    "MethodCategory",
    "AttributionResult",
    "AttributionMethod",
    "MethodRegistry",
]
