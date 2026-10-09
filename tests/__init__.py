"""Test-suite package marker.

Makes ``tests/conftest.py`` import as ``tests.conftest`` (not as a top-level
``conftest`` module), so it cannot shadow ``tests/lrp/conftest.py``, which
the lrp test modules import via ``from conftest import ...``.
"""
