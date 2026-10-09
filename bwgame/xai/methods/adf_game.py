"""Stopping Game (SG) attribution.

Runs the DC decomposition of the network in the stopping backward mode
with an ADF (Assumed Density Filtering) variance pass: the activation gate
of every ReLU / GELU uses the propagated pre-activation mean ``mu`` and
standard deviation ``sigma`` (input variance ``adf_input_variance``).

Modes (paper, Appendix "Analytical Modes"):

- risk aversion ``adf_lambda`` (lambda <= 0): the gate compares the
  confidence bound ``mu + lambda * sigma`` with zero;
- noisy-observation gate ``adf_probabilistic``: the hard gate is replaced
  by the continuation probability ``Phi(mu / sigma + lambda)``;
- fan-in entropy bias ``entropy_lambda``: ``b_j -> b_j - lambda_ent * H_j``.

The attribution is the ``pp`` component of the four DC sensitivities,
summed over input channels.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..base import AttributionMethod
from ..postprocess import tensor_to_numpy
from ..registry import MethodRegistry
from ..types import MethodCategory, MethodConfig

logger = logging.getLogger(__name__)


@MethodRegistry.register
class AdfGame(AttributionMethod):
    """Stopping Game attribution via the variance-aware DC decomposition."""

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="adf_game",
            category=MethodCategory.GAME_THEORY,
            class_specific=True,
            supports_batch=False,
            requires_grad=True,
            per_layer=False,
            description=(
                "Stopping Game attribution: risk-sensitive backward pass "
                "using propagated Gaussian variance"
            ),
            reference="bwgame.dc_context",
        )

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.adf_lambda = 0.0
        self.adf_input_variance = 1.0
        self.adf_probabilistic = True
        self.entropy_lambda = 0.0

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        B, C, H, W = x.shape
        self.model.eval()

        from bwgame.dc_context import dc_decomposition

        logger.info(
            "SG: lambda=%.3f input_var=%.3f gate=%s entropy_lambda=%.3f",
            self.adf_lambda, self.adf_input_variance,
            self.adf_probabilistic, self.entropy_lambda,
        )

        maps: List[np.ndarray] = []
        for b in range(B):
            x_single = x[b:b + 1].detach().clone()
            target_cls = target[b].item()
            try:
                with dc_decomposition(
                    self.model, x_single,
                    backprop_mode='stopping',
                    adf_mode='adf',
                    adf_lambda=self.adf_lambda,
                    adf_input_variance=self.adf_input_variance,
                    adf_probabilistic=self.adf_probabilistic,
                    entropy_bias_adapt=self.entropy_lambda,
                ) as dc:
                    output = dc.output
                    one_hot = torch.zeros_like(output)
                    one_hot[0, target_cls] = 1.0
                    output.backward(gradient=one_hot)
                maps.append(tensor_to_numpy(dc.sensitivities.delta_pp.sum(dim=1)))
            except Exception as e:
                # NaN map: excluded from every metric and counted as a map
                # failure by the evaluator.
                logger.warning("SG failed for sample %d: %s. Returning NaN attribution.", b, e)
                maps.append(np.full((1, H, W), np.nan, dtype=np.float32))

        return {"pp": np.concatenate(maps, axis=0)}
