"""Decorator-based method registry.

Usage:
    @MethodRegistry.register
    class MyMethod(AttributionMethod):
        @classmethod
        def config(cls):
            return MethodConfig(name="MyMethod", ...)
        ...

    # Later:
    method = MethodRegistry.instantiate("MyMethod", model, device)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict


if TYPE_CHECKING:
    from .base import AttributionMethod


class MethodRegistry:
    """Central registry mapping method names to their implementing classes.

    Methods are registered via the ``@MethodRegistry.register`` class decorator.
    The ``methods/__init__.py`` module imports all method files, which triggers
    the decorators and populates the registry at import time.
    """

    _methods: Dict[str, type] = {}

    @classmethod
    def register(cls, method_cls: type) -> type:
        """Class decorator that registers an ``AttributionMethod`` subclass.

        The method's ``config().name`` is used as the registry key.

        Args:
            method_cls: The class to register.

        Returns:
            The class, unmodified (allows stacking decorators).
        """
        cfg = method_cls.config()
        cls._methods[cfg.name] = method_cls
        return method_cls

    @classmethod
    def get(cls, name: str) -> type:
        """Look up a method class by name.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in cls._methods:
            available = list(cls._methods.keys())
            raise KeyError(
                f"Unknown method: {name!r}. Available: {available}"
            )
        return cls._methods[name]

    @classmethod
    def instantiate(
        cls,
        name: str,
        model,
        device,
    ) -> "AttributionMethod":
        """Create an instance of the named method.

        Args:
            name: Registered method name.
            model: The neural network model.
            device: Torch device.

        Returns:
            An initialized ``AttributionMethod`` instance.
        """
        return cls.get(name)(model, device)
