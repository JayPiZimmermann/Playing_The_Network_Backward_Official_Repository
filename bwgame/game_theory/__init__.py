"""Game-theoretic modifications of the DC decomposition.

- Fan-in entropy of the weights and the entropy bias (``entropy``)
- Per-layer activation / ADF cache stored on modules (``game_context``)
"""

from .entropy import compute_entropy, entropy_bias_adapt_context
from .game_context import DCLayerData, DC_LAYER_DATA, run_capture_pass, cleanup_layer_data
