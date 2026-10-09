"""LRP for torchvision Vision Transformers with the efficient rules of lxt.

Implements the LRP recipe of lxt (2.x) for torchvision ViTs
(``lxt.efficient.models.vit_torch.cp_LRP`` together with
``lxt.efficient.monkey_patch_zennit``; Achtibat et al., "AttnLRP:
Attention-Aware Layer-Wise Relevance Propagation for Transformers",
ICML 2024).  Relevance is computed in lxt's Gradient x Input formulation:
the backward pass is modified such that ``t * grad(t)`` is the LRP relevance
of every tensor ``t``, and the input heatmap is ``(x * grad(x)).sum(1)``.

Rules (all taken from ``lxt.efficient``):

* LayerNorm: identity rule, the standard deviation is a constant
  (``patches.layer_norm_forward``).
* GELU: identity rule (``rules.identity_rule_implicit``).
* Self-attention: CP-LRP inside the attention (Ali et al., ICML 2022), i.e.
  no relevance flows through the softmax; queries and keys are constants and
  relevance flows through the values only
  (``patches.cp_multi_head_attention_forward``).
* Attention-weighted sum of the values, residual additions, patch
  flattening, class token and position embedding: Gradient x Input, i.e. the
  z-rule.  Relevance assigned to the class token and position embedding
  (parameters) is not propagated to the input.
* ``nn.Linear`` / ``nn.Conv2d``: zennit rules (Epsilon, Gamma, ZBox) in the
  Gradient x Input formulation of ``lxt.efficient.zennit_patches``.

In lxt's ViT recipe (``examples/vit_torch.py`` of lxt) the Q/K/V and output
projections of the attention are evaluated inside ``nn.MultiheadAttention``
and hence propagated with Gradient x Input (the z-rule).  Here they are
``nn.Linear`` modules of the module tree seen by the composite and can
receive their own rule (:func:`rule_map`), as in the ViT composite of
Achtibat et al. (2024, Table B.5); without a rule the result is identical to
lxt's recipe.

The forward pass is that of ``torchvision.models.VisionTransformer`` in
evaluation mode, computed with the modules and parameters of the given model
(either an original torchvision ViT or the decomposed model built by
``bwgame.composite_replacer``).  No class, function or library state is
modified; the zennit hooks exist only inside :func:`lrp_heatmap`.
"""
from __future__ import annotations

import functools
import importlib
import os
import sys
import types
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def is_torchvision_vit(model: nn.Module) -> bool:
    """True if ``model`` is a ``torchvision.models.VisionTransformer``."""
    try:
        from torchvision.models import VisionTransformer
    except ImportError:
        return False
    return isinstance(model, VisionTransformer)


@functools.lru_cache(maxsize=None)
def lxt_efficient() -> types.SimpleNamespace:
    """Return the modules ``rules``, ``patches`` and ``zennit_patches`` of ``lxt.efficient``.

    ``lxt/efficient/__init__.py`` also imports lxt's patches for HuggingFace
    language models, which cannot be imported with every ``transformers``
    release (lxt 2.1 with transformers 5).  The three modules used here do
    not depend on them, so they are loaded without executing the package's
    ``__init__.py``.
    """
    names = ("rules", "patches", "zennit_patches")
    if "lxt.efficient" in sys.modules:
        mods = [importlib.import_module(f"lxt.efficient.{n}") for n in names]
    else:
        import lxt
        package = types.ModuleType("lxt.efficient")
        package.__path__ = [os.path.join(os.path.dirname(lxt.__file__), "efficient")]
        sys.modules["lxt.efficient"] = package
        try:
            mods = [importlib.import_module(f"lxt.efficient.{n}") for n in names]
        finally:
            del sys.modules["lxt.efficient"]
    return types.SimpleNamespace(**dict(zip(names, mods)))


@functools.lru_cache(maxsize=None)
def gxi_rule(rule_cls: type) -> type:
    """Subclass of a zennit rule evaluated in lxt's Gradient x Input formulation.

    Equivalent to ``lxt.efficient.monkey_patch_zennit()``, which replaces
    ``zennit.core.BasicHook.forward`` / ``backward`` globally, but limited to
    the returned class.
    """
    zp = lxt_efficient().zennit_patches
    return type(rule_cls.__name__, (rule_cls,),
                {"forward": zp.forward, "backward": zp.backward, "__module__": __name__})


def Epsilon(epsilon: float):
    """zennit ``Epsilon`` rule for :func:`lrp_heatmap`."""
    from zennit.rules import Epsilon as _Epsilon
    return gxi_rule(_Epsilon)(epsilon=epsilon)


def Gamma(gamma: float, stabilizer: float = 1e-6):
    """zennit ``Gamma`` rule for :func:`lrp_heatmap`."""
    from zennit.rules import Gamma as _Gamma
    return gxi_rule(_Gamma)(gamma=gamma, stabilizer=stabilizer)


def ZBox(low: float, high: float, stabilizer: float = 1e-6):
    """zennit ``ZBox`` rule for :func:`lrp_heatmap`."""
    from zennit.rules import ZBox as _ZBox
    return gxi_rule(_ZBox)(low=low, high=high, stabilizer=stabilizer)


def _projection_view(weight: Tensor, bias: Optional[Tensor], i: int, dim: int) -> nn.Linear:
    """``nn.Linear`` sharing storage with rows ``i*dim:(i+1)*dim`` of a packed in-projection."""
    lin = nn.Linear(dim, dim, bias=bias is not None, device="meta")
    rows = slice(i * dim, (i + 1) * dim)
    lin.weight = nn.Parameter(weight.detach()[rows], requires_grad=False)
    if bias is not None:
        lin.bias = nn.Parameter(bias.detach()[rows], requires_grad=False)
    return lin


class _EncoderBlock(nn.Module):
    """``torchvision.models.vision_transformer.EncoderBlock`` with lxt's CP-LRP rules."""

    def __init__(self, block: nn.Module):
        super().__init__()
        attn = block.self_attention
        if not getattr(attn, "batch_first", False):
            raise ValueError("expected batch-first self-attention (torchvision ViT)")
        self.ln_1 = block.ln_1
        self.ln_2 = block.ln_2
        self.mlp = block.mlp
        self.num_heads = attn.num_heads
        if isinstance(attn, nn.MultiheadAttention):
            if not attn._qkv_same_embed_dim or attn.bias_k is not None or attn.add_zero_attn:
                raise ValueError("unsupported nn.MultiheadAttention configuration")
            dim = attn.embed_dim
            self.q_proj, self.k_proj, self.v_proj = (
                _projection_view(attn.in_proj_weight, attn.in_proj_bias, i, dim) for i in range(3))
        else:
            if attn.bias_k is not None or attn.add_zero_attn:
                raise ValueError("unsupported attention configuration")
            self.q_proj, self.k_proj, self.v_proj = attn.q_proj, attn.k_proj, attn.v_proj
        self.out_proj = attn.out_proj

    def attention(self, x: Tensor) -> Tensor:
        # CP-LRP: queries and keys are computed from a detached input
        # (lxt ``cp_multi_head_attention_forward``), so the attention weights
        # are constants and relevance flows through the values only.
        stop_gradient = lxt_efficient().rules.stop_gradient
        n, length, dim = x.shape
        heads = self.num_heads

        def split(t: Tensor) -> Tensor:
            return t.view(n, length, heads, dim // heads).transpose(1, 2)

        q = split(self.q_proj(stop_gradient(x)))
        k = split(self.k_proj(stop_gradient(x)))
        v = split(self.v_proj(x))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.out_proj(out.transpose(1, 2).reshape(n, length, dim))

    def forward(self, x: Tensor) -> Tensor:
        lxt = lxt_efficient()
        h = lxt.patches.layer_norm_forward(self.ln_1, x)
        x = x + self.attention(h)
        h = lxt.patches.layer_norm_forward(self.ln_2, x)
        for layer in self.mlp:
            if isinstance(layer, nn.GELU):
                h = lxt.rules.identity_rule_implicit(layer, h)
            else:
                h = layer(h)
        return x + h


class LxtVisionTransformer(nn.Module):
    """``torchvision.models.VisionTransformer.forward`` with lxt's efficient LRP rules.

    Holds references to (not copies of) the modules and parameters of
    ``vit``; build it right before use so that it reflects the current
    parameters.
    """

    def __init__(self, vit: nn.Module):
        super().__init__()
        if not is_torchvision_vit(vit):
            raise TypeError(f"expected a torchvision VisionTransformer, got {type(vit).__name__}")
        self.hidden_dim = vit.hidden_dim
        self.conv_proj = vit.conv_proj
        self.class_token = vit.class_token
        self.pos_embedding = vit.encoder.pos_embedding
        self.blocks = nn.ModuleList(_EncoderBlock(b) for b in vit.encoder.layers)
        self.ln = vit.encoder.ln
        self.heads = vit.heads
        self._projections = {id(m) for b in self.blocks
                             for m in (b.q_proj, b.k_proj, b.v_proj, b.out_proj)}

    def is_attention_projection(self, module: nn.Module) -> bool:
        """True for the Q, K, V and output projections of the attention layers."""
        return id(module) in self._projections

    def forward(self, x: Tensor) -> Tensor:
        n = x.shape[0]
        x = self.conv_proj(x)
        x = x.reshape(n, self.hidden_dim, -1).permute(0, 2, 1)
        x = torch.cat([self.class_token.expand(n, -1, -1), x], dim=1)
        x = x + self.pos_embedding
        for block in self.blocks:
            x = block(x)
        x = lxt_efficient().patches.layer_norm_forward(self.ln, x)
        return self.heads(x[:, 0])


def rule_map(conv=None, linear=None, attention=None, first=None):
    """``rule_for`` argument of :func:`lrp_heatmap`.

    Args:
        conv: Rule of the ``nn.Conv2d`` layers (the patch embedding).
        linear: Rule of the ``nn.Linear`` layers outside the attention
            (MLPs and classification head).
        attention: Rule of the Q/K/V and output projections of the
            attention.  ``None`` propagates them with Gradient x Input (the
            z-rule), which is how lxt's ViT recipe treats them: they are
            evaluated inside ``nn.MultiheadAttention``, where no zennit rule
            reaches them.
        first: Rule of the patch embedding, overriding ``conv``.
    """
    def rule_for(net: LxtVisionTransformer, name: str, module: nn.Module):
        if net.is_attention_projection(module):
            return attention
        if first is not None and module is net.conv_proj:
            return first
        if isinstance(module, nn.Conv2d):
            return conv
        if isinstance(module, nn.Linear):
            return linear
        return None
    return rule_for


def lrp_heatmap(
    vit: nn.Module,
    x: Tensor,
    output_grad: Callable[[Tensor], Tensor],
    rule_for: Callable[[LxtVisionTransformer, str, nn.Module], Optional[object]],
) -> Tensor:
    """Input relevance ``x * grad(x)`` of a torchvision ViT under lxt's rules.

    Args:
        vit: torchvision ``VisionTransformer`` (original or decomposed).
        x: Input batch ``(B, C, H, W)``.
        output_grad: Maps the logits ``(B, K)`` to the initial relevance
            weights at the output (e.g. a one-hot target mask); the output
            relevance is ``logits * output_grad(logits)``.
        rule_for: ``(net, name, module) -> rule or None`` assigning a zennit
            rule (from :func:`Epsilon`, :func:`Gamma`, :func:`ZBox`) to the
            ``nn.Linear`` / ``nn.Conv2d`` modules of ``net``; see
            :func:`rule_map`.

    Returns:
        Relevance of every input element, shape ``(B, C, H, W)``.
    """
    from zennit.core import Composite

    net = LxtVisionTransformer(vit)
    composite = Composite(module_map=lambda ctx, name, module: rule_for(net, name, module))
    x = x.detach().clone().requires_grad_(True)
    with torch.enable_grad(), composite.context(net):
        logits = net(x)
        (grad,) = torch.autograd.grad(logits, x, grad_outputs=output_grad(logits.detach()))
    return (x * grad).detach()
