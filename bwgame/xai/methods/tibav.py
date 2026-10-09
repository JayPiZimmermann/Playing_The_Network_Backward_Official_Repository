"""TIBAV: Transformer Interpretability Beyond Attention Visualization.

Chefer, Gur & Wolf, CVPR 2021.  The map is produced by the authors' reference
implementation (github.com/hila-chefer/Transformer-Explainability), vendored in
:mod:`bwgame.xai.methods._chefer_vit_lrp`:

    LRP(model).generate_LRP(x, index=target,
                            method="transformer_attribution", start_layer=0)

i.e. the relevance R(A_l) at each attention map from the paper's LRP rules
(alpha = 1, beta = 0 on linear layers, the z-rule halved between the operands of
each attention matmul, the normalised rule for residual additions; LayerNorm,
softmax and GELU pass relevance unchanged), the gradient dA_l of the target
logit, and the rollout

    A_bar_l = I + mean_heads( (dA_l * R(A_l))^+ ),   C = A_bar_L ... A_bar_1,

with the map C[CLS, 1:] reshaped to the patch grid.

The reference ViT is built once per method instance from the weights of the
torchvision ``VisionTransformer`` handed to the method (plain or with
``DecomposedMultiheadAttention``), so it computes the same function as the
evaluated model.  The reference implementation explains one image at a time
(its rollout reads batch element 0 and its Add rule normalises over the whole
tensor), so batches are processed image by image.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ..base import AttributionMethod
from ..postprocess import tokens_to_spatial
from ..registry import MethodRegistry
from ..types import MethodCategory, MethodConfig
from . import _chefer_vit_lrp as chefer


def _qkv(sd: Dict[str, Tensor], prefix: str, name: str) -> Tensor:
    """Packed [q; k; v] projection ``weight`` or ``bias`` of a torchvision attention module."""
    packed = f"{prefix}in_proj_{name}"
    if packed in sd:
        return sd[packed]
    return torch.cat([sd[f"{prefix}{p}_proj.{name}"] for p in ("q", "k", "v")], dim=0)


def chefer_vit_from_torchvision(model: nn.Module) -> chefer.VisionTransformer:
    """Chefer's LRP ViT with the weights and LayerNorm epsilons of a torchvision ViT.

    Parameter mapping (torchvision -> timm-style names of the reference model):
    ``conv_proj`` -> ``patch_embed.proj``, ``class_token`` -> ``cls_token``,
    ``encoder.pos_embedding`` -> ``pos_embed``, per block ``ln_1/ln_2`` ->
    ``norm1/norm2``, packed ``self_attention`` q/k/v -> ``attn.qkv``,
    ``out_proj`` -> ``attn.proj``, ``mlp.0/mlp.3`` -> ``mlp.fc1/mlp.fc2``,
    ``encoder.ln`` -> ``norm``, ``heads.head`` -> ``head``.
    """
    from torchvision.models.vision_transformer import VisionTransformer as TVViT

    if not isinstance(model, TVViT):
        raise TypeError(f"TIBAV needs a torchvision VisionTransformer, got {type(model).__name__}")
    if model.representation_size is not None or not isinstance(model.conv_proj, nn.Conv2d):
        raise TypeError("TIBAV supports torchvision ViTs with a patch conv and a linear head")

    layers = list(model.encoder.layers)
    sd = model.state_dict()
    # Built on the meta device and then materialised, which skips the random
    # initialisation; every parameter is overwritten by load_state_dict below.
    with torch.device("meta"):
        ref = chefer.VisionTransformer(
            img_size=model.image_size, patch_size=model.patch_size, num_classes=model.num_classes,
            embed_dim=model.hidden_dim, depth=len(layers), num_heads=layers[0].self_attention.num_heads,
            mlp_ratio=model.mlp_dim / model.hidden_dim, qkv_bias=True,
        )
    ref = ref.to_empty(device=sd["class_token"].device)
    out = {
        "patch_embed.proj.weight": sd["conv_proj.weight"],
        "patch_embed.proj.bias": sd["conv_proj.bias"],
        "cls_token": sd["class_token"],
        "pos_embed": sd["encoder.pos_embedding"],
        "norm.weight": sd["encoder.ln.weight"],
        "norm.bias": sd["encoder.ln.bias"],
        "head.weight": sd["heads.head.weight"],
        "head.bias": sd["heads.head.bias"],
    }
    for i, layer in enumerate(layers):
        src = f"encoder.layers.encoder_layer_{i}."
        dst = f"blocks.{i}."
        attn = f"{src}self_attention."
        out.update({
            dst + "norm1.weight": sd[src + "ln_1.weight"],
            dst + "norm1.bias": sd[src + "ln_1.bias"],
            dst + "attn.qkv.weight": _qkv(sd, attn, "weight"),
            dst + "attn.qkv.bias": _qkv(sd, attn, "bias"),
            dst + "attn.proj.weight": sd[attn + "out_proj.weight"],
            dst + "attn.proj.bias": sd[attn + "out_proj.bias"],
            dst + "norm2.weight": sd[src + "ln_2.weight"],
            dst + "norm2.bias": sd[src + "ln_2.bias"],
            dst + "mlp.fc1.weight": sd[src + "mlp.0.weight"],
            dst + "mlp.fc1.bias": sd[src + "mlp.0.bias"],
            dst + "mlp.fc2.weight": sd[src + "mlp.3.weight"],
            dst + "mlp.fc2.bias": sd[src + "mlp.3.bias"],
        })
        ref.blocks[i].norm1.eps = layer.ln_1.eps
        ref.blocks[i].norm2.eps = layer.ln_2.eps
    ref.load_state_dict(out, strict=True)
    ref.norm.eps = model.encoder.ln.eps
    return ref.eval()


@MethodRegistry.register
class TIBAV(AttributionMethod):
    """Transformer attribution of Chefer et al. 2021 (CVPR), reference implementation."""

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="tibav",
            category=MethodCategory.ATTENTION,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=False,
            requires_attention=True,
            description=(
                "Transformer Interpretability Beyond Attention Visualization "
                "(Chefer et al. 2021, CVPR): gradient-weighted LRP relevance "
                "of the attention maps, rolled out over the blocks."
            ),
            reference="https://arxiv.org/abs/2012.09838",
        )

    def __init__(self, model, device):
        super().__init__(model, device)
        self._lrp = chefer.LRP(chefer_vit_from_torchvision(model).to(device))

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        rows = []
        with torch.enable_grad():
            for b in range(x.size(0)):
                rel = self._lrp.generate_LRP(
                    x[b:b + 1].detach().to(self.device), index=int(target[b]),
                    method="transformer_attribution", start_layer=0,
                )
                rows.append(rel.detach())
        result = {"input": tokens_to_spatial(torch.cat(rows), has_cls=False)}
        if layers:
            result = {k: v for k, v in result.items() if k in layers}
        return result

    def cleanup(self) -> None:
        # The reference model stores its forward activations and attention
        # maps as module attributes; with the attention-gradient hooks these
        # form a reference cycle through the autograd graph that the garbage
        # collector cannot break.  Drop them so a discarded instance frees
        # its (GPU) memory.
        for m in self._lrp.model.modules():
            for k, v in list(vars(m).items()):
                if isinstance(v, Tensor) or (
                        isinstance(v, (list, tuple)) and any(isinstance(e, Tensor) for e in v)):
                    setattr(m, k, None)
