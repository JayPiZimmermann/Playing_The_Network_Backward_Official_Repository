"""Gradient-weighted attention rollout: Generic Attention-model Explainability (GAE).

Chefer, Gur & Wolf, "Generic Attention-model Explainability for Interpreting
Bi-Modal and Encoder-Decoder Transformers", ICCV 2021 (self-attention
rules 5 and 6).  For encoder blocks l = 1..L in forward order, with post-softmax
attention A_l (heads, S, S) and its gradient dA_l = d y_target / d A_l:

    A_bar_l = mean_heads( (dA_l * A_l)^+ )     positive part per head, then mean
    R_0     = I
    R_l     = R_{l-1} + A_bar_l @ R_{l-1}

The attribution map is R_L[CLS, 1:] reshaped to the patch grid.  This follows
``avg_heads`` / ``apply_self_attention_rules`` / ``generate_relevance`` in
github.com/hila-chefer/Transformer-MM-Explainability
(Transformer_MM_explainability_ViT.ipynb).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..base import AttributionMethod
from ..model_utils import get_vit_encoder_blocks, order_by_encoder_block
from ..postprocess import tokens_to_spatial
from ..registry import MethodRegistry
from ..types import MethodCategory, MethodConfig

from ...composite_replacer import DecomposedMultiheadAttention


class _GradAttentionCapture:
    """Capture attention weights AND their gradients during forward+backward.

    Hooks the ``attn_dropout`` module of each ``DecomposedMultiheadAttention``
    to store the attention weight tensor (with grad enabled) during forward,
    then retrieves ``.grad`` after backward.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._attn_tensors: Dict[str, Tensor] = {}

    def install(self) -> "_GradAttentionCapture":
        self._attn_tensors.clear()
        self._hooks.clear()

        for parent_name, parent_module in self.model.named_modules():
            if not isinstance(parent_module, DecomposedMultiheadAttention):
                continue

            dropout_module = parent_module.attn_dropout

            def _make_hook(name: str):
                def _hook(module, input, output):
                    attn_w = input[0]
                    if not attn_w.requires_grad:
                        attn_w.requires_grad_(True)
                    if not attn_w.is_leaf:
                        attn_w.retain_grad()
                    self._attn_tensors[name] = attn_w
                return _hook

            handle = dropout_module.register_forward_hook(_make_hook(parent_name))
            self._hooks.append(handle)

        return self

    def get_attention_and_grads(self) -> Dict[str, tuple]:
        """Return ``{name: (attn_weights, attn_grads)}`` after backward.

        Both tensors have shape ``(B, num_heads, seq_len, seq_len)``.
        """
        result = {}
        for name, attn_w in self._attn_tensors.items():
            grad = attn_w.grad if attn_w.grad is not None else torch.zeros_like(attn_w)
            result[name] = (attn_w.detach(), grad.detach())
        return result

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._attn_tensors.clear()


@MethodRegistry.register
class GradAttentionRollout(AttributionMethod):
    """Gradient-weighted Attention Rollout (Chefer et al. 2021b, GAE).

    Produces class-specific attribution maps by weighting attention matrices
    with their gradients, then performing rollout across layers.
    """

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="grad_attention_rollout",
            category=MethodCategory.ATTENTION,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=True,
            requires_attention=True,
            description=(
                "Gradient-weighted Attention Rollout "
                "(Chefer et al. 2021, Generic Attention Explainability)."
            ),
            reference="https://arxiv.org/abs/2103.15679",
        )

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        self.model.eval()
        B = x.size(0)

        capture = _GradAttentionCapture(self.model)
        capture.install()
        try:
            with torch.enable_grad():
                output = self.model(x.detach())  # (B, num_classes)
                grad_seed = torch.zeros_like(output)
                grad_seed[torch.arange(B, device=output.device), target.to(output.device)] = 1.0
                output.backward(gradient=grad_seed)
            attn_and_grads = capture.get_attention_and_grads()
        finally:
            capture.remove()

        if not attn_and_grads:
            raise RuntimeError(
                "No attention weights captured. Ensure the model has been "
                "decomposed with decompose_composites()."
            )

        block_names = [name for name, _ in get_vit_encoder_blocks(self.model)]
        token_idx = getattr(self, 'token_idx', 0)
        results: Dict[str, np.ndarray] = {}
        R = None
        for name in order_by_encoder_block(attn_and_grads, self.model):
            attn_w, attn_g = attn_and_grads[name]  # (B, heads, S, S)
            cam = (attn_g * attn_w).clamp(min=0).mean(dim=1)  # (B, S, S)
            if R is None:
                R = torch.eye(cam.size(-1), device=cam.device, dtype=cam.dtype)
                R = R.unsqueeze(0).repeat(B, 1, 1)
            R = R + torch.bmm(cam, R)
            block = next(bn for bn in block_names if name == bn or name.startswith(bn + "."))
            results[block] = tokens_to_spatial(R[:, token_idx, :], has_cls=True)

        results["input"] = results[block]

        if layers:
            results = {k: v for k, v in results.items() if k in layers}
        return results
