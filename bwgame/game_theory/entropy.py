"""Fan-in entropy bias (lambda_ent) of the game rewards.

In the Stopping Game, transition probabilities from neuron j to neuron k are
p_jk = |W_jk| / gamma_j where gamma_j = sum_k |W_jk|.

The normalized entropy of the outgoing transition distribution is:
    H̃_j = H_j / log(fan_in)  ∈ [0, 1]

where H_j = -sum_k p_jk * log(p_jk) is the raw Shannon entropy.

This module subtracts lambda * H̃_j from each neuron's bias (the game reward),
penalizing high-entropy (uncertain) transition nodes. Low-entropy nodes
(strongly connected to few targets) keep their full reward.

Normalizing by log(fan_in) ensures the penalty is in [0, lambda] regardless
of layer width, making lambda interpretable and consistent across layers.
"""

import logging
import math
from contextlib import contextmanager
from typing import List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


def compute_entropy(weight: Tensor, normalize: bool = False) -> Tensor:
    """Compute transition entropy H_j for each output neuron j.

    For a weight matrix W of shape (out_features, in_features), computes:
        p_jk = |W_jk| / sum_k |W_jk|
        H_j = -sum_k p_jk * log(p_jk)

    For Conv2d weights of shape (out_channels, in_channels, kH, kW),
    the kernel dimensions are flattened so that each output channel j
    has transitions to all (in_channels * kH * kW) positions.

    Args:
        weight: Weight tensor, shape (out, in) or (out, in, kH, kW).
        normalize: If True, normalize by log(fan_in) so H_j ∈ [0, 1].

    Returns:
        Tensor of shape (out,) with entropy H_j per output neuron.
    """
    # Flatten to 2D: (out_features, fan_in)
    w = weight.detach().abs().reshape(weight.shape[0], -1)
    fan_in = w.shape[1]

    # Transition probabilities: p_jk = |W_jk| / gamma_j
    gamma = w.sum(dim=1, keepdim=True).clamp(min=1e-12)
    p = w / gamma

    # Entropy: H_j = -sum_k p_jk * log(p_jk), with 0*log(0) = 0
    log_p = torch.log(p.clamp(min=1e-12))
    H = -(p * log_p).sum(dim=1)

    if normalize and fan_in > 1:
        H = H / math.log(fan_in)

    return H


def _get_weighted_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Get all Linear and Conv2d modules with their names."""
    modules = []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            modules.append((name, m))
    return modules


@contextmanager
def entropy_bias_adapt_context(model: nn.Module, lam: float):
    """Context manager that applies entropy penalty by directly shifting biases.

    For each Linear/Conv2d module with a bias, subtracts λ · H̃_j from b_j:

        b_j -= λ · H̃_j

    This is an additive modification of the game rewards: high-entropy
    neurons get their bias reduced, making them less influential.

    Biases are restored on exit.

    Args:
        model: PyTorch model.
        lam: Entropy adaptation coefficient (lambda > 0).
    """
    modules = _get_weighted_modules(model)
    saved_biases = []

    try:
        for name, m in modules:
            if m.bias is None:
                continue
            H = compute_entropy(m.weight, normalize=True)
            saved_biases.append((m, m.bias.data.clone()))
            # Subtract entropy penalty from bias
            with torch.no_grad():
                m.bias.data -= lam * H.to(m.bias.device)

            logger.debug(
                "entropy_bias_adapt [%s]: H_norm mean=%.3f, "
                "bias_shift mean=%.4f",
                name, H.mean().item(), (lam * H).mean().item(),
            )

        yield

    finally:
        for m, original_bias in saved_biases:
            m.bias.data.copy_(original_bias)

