"""Auto-import all method modules to trigger @MethodRegistry.register decorators.

Each module below defines one or more ``AttributionMethod`` subclasses
decorated with ``@MethodRegistry.register``.  Importing them here ensures
they are registered as soon as ``bwgame.xai`` is imported.

Imports are wrapped in individual try/except blocks so that a missing
optional dependency (captum, grad-cam, zennit, lxt) does not prevent the
rest of the framework from loading.
"""

import importlib
import logging

logger = logging.getLogger(__name__)

_METHOD_MODULES = [
    "captum_methods",
    "gradcam_methods",
    "lrp_library",
    "attention_rollout",
    "grad_attention_rollout",
    "tibav",
    "adf_game",
    "routing_game",
    "dave",
]

for _mod_name in _METHOD_MODULES:
    try:
        importlib.import_module(f".{_mod_name}", package=__name__)
    except ImportError as e:
        logger.debug(
            "Skipping method module %s (missing dependency: %s)", _mod_name, e
        )
    except Exception as e:
        logger.warning(
            "Failed to import method module %s: %s", _mod_name, e
        )
