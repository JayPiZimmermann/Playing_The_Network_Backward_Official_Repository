"""Root conftest: make the test helpers importable regardless of test order.

Several test modules import shared helpers as top-level modules
(``from utils import ...`` in ``tests/models``) or through the package
(``from tests.utils.dc_tester import ...``).  Inserting both the project
root and ``tests/`` into ``sys.path`` here, before any test module is
collected, makes these imports independent of the test order.
"""

import os
import sys

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_TESTS_DIR)

for _p in (_PROJECT_ROOT, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
