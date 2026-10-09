"""Attention Rollout (Abnar & Zuidema, ACL 2020).

For encoder blocks l = 1..L (forward order) with post-softmax attention
A_l of shape (heads, S, S):

    A_bar_l = w * I + (1 - w) * mean_heads(A_l),   rows normalised to sum 1
    J_1     = A_bar_1
    J_l     = A_bar_l @ J_{l-1}

The attribution map is the row of J_L of the CLS token over the patch tokens,
reshaped to the patch grid.  ``w`` is the residual weight; w = 0.5 is the
reference setting ``(A + I) / 2`` of ``compute_joint_attention`` in
github.com/samiraabnar/attention_flow (attention_graph_util.py).
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor

from ..base import AttributionMethod
from ..types import MethodConfig, MethodCategory
from ..registry import MethodRegistry
from ..model_utils import AttentionWeightCapture, order_by_encoder_block
from ..postprocess import tokens_to_spatial


@MethodRegistry.register
class AttentionRollout(AttributionMethod):
    """Recursively multiply head-averaged attention across layers with residual identity."""

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="attention_rollout",
            category=MethodCategory.ATTENTION,
            class_specific=False,
            supports_batch=True,
            requires_grad=False,
            per_layer=True,
            requires_attention=True,
            description="Recursive attention multiplication across layers (Abnar & Zuidema 2020)",
            reference="Abnar & Zuidema 2020",
        )

    def __init__(self, model, device):
        super().__init__(model, device)
        self.residual_weight = 0.5

    def attribute(self, x: Tensor, target: Tensor,
                  layers: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        with AttentionWeightCapture(self.model) as capture:
            with torch.no_grad():
                self.model(x)
            weights = capture.get_weights()
        if not weights:
            raise RuntimeError(
                "No attention weights captured. Ensure the model has been "
                "decomposed with decompose_composites()."
            )

        w = float(self.residual_weight)
        token_idx = getattr(self, "token_idx", 0)
        result: Dict[str, np.ndarray] = {}
        joint = None
        for name in order_by_encoder_block(weights, self.model):
            attn = weights[name].mean(dim=1)  # (B, S, S), mean over heads
            eye = torch.eye(attn.size(-1), device=attn.device, dtype=attn.dtype)
            a_bar = w * eye + (1.0 - w) * attn
            a_bar = a_bar / a_bar.sum(dim=-1, keepdim=True)
            joint = a_bar if joint is None else torch.bmm(a_bar, joint)
            result[name] = tokens_to_spatial(joint[:, token_idx, :], has_cls=True)

        result["input"] = result[name]

        if layers:
            result = {k: v for k, v in result.items() if k in layers}
        return result
