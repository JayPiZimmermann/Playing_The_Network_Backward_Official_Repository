"""LRP-based attribution methods using external libraries.

CNNs: LRP rules from ``zennit`` with its torchvision canonizers.

torchvision Vision Transformers (``lrp_standard``, ``gamma_lrp``,
``attn_lrp``): the efficient LRP rules of ``lxt`` (Achtibat et al., ICML
2024) for LayerNorm, GELU, attention and residual connections, with zennit
rules on the Linear / Conv2d layers; see :mod:`._lxt_vit`.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import AttributionMethod
from ..postprocess import tensor_to_numpy
from ..registry import MethodRegistry
from ..types import MethodCategory, MethodConfig
from . import _lxt_vit

logger = logging.getLogger(__name__)


def _patch_zennit_batchnorm_eps():
    """Fix zennit canonizer for PyTorch ≥2.6 which requires BN eps > 0.

    Zennit's ``MergeBatchNorm.merge_batch_norm`` sets ``eps`` to 0.0 after
    merging BN into the preceding Conv/Linear.  PyTorch now raises
    ``ValueError: batch_norm eps must be positive`` on forward.  We
    monkey-patch the static method to use a tiny positive value instead.
    """
    try:
        from zennit.canonizers import MergeBatchNorm
    except ImportError:
        return

    _original = MergeBatchNorm.merge_batch_norm.__func__ if hasattr(
        MergeBatchNorm.merge_batch_norm, '__func__') else MergeBatchNorm.merge_batch_norm

    @staticmethod
    def _patched(modules, batch_norm):
        _original(modules, batch_norm)
        # Ensure eps stays positive for newer PyTorch
        if hasattr(batch_norm, 'eps') and batch_norm.eps <= 0:
            object.__setattr__(batch_norm, 'eps', 1e-30)

    MergeBatchNorm.merge_batch_norm = _patched


_patch_zennit_batchnorm_eps()


def _get_canonizers(model: nn.Module):
    """Return the appropriate zennit canonizer(s) for the given model.

    Uses ``ResNetCanonizer`` for models with residual skip connections,
    ``VGGCanonizer`` for everything else.
    """
    try:
        from zennit.torchvision import VGGCanonizer, ResNetCanonizer
    except ImportError:
        return []

    # Detect ResNet-family by checking for typical residual block attributes
    has_residual = any(
        hasattr(m, 'downsample') or type(m).__name__ in ('BasicBlock', 'Bottleneck')
        for m in model.modules()
    )
    if has_residual:
        return [ResNetCanonizer()]
    return [VGGCanonizer()]


@contextlib.contextmanager
def zennit_context(composite, model: nn.Module):
    """``composite.context(model)`` that leaves every module of the model as it was.

    zennit's attribute canonizers (``ResNetCanonizer``) overload ``forward``
    of the residual blocks and remove the overload with ``delattr``, which
    also deletes a ``forward`` set on the instance before (the traced
    forward installed by ``bwgame.functional_replacer``).  Its batch-norm
    merge writes ``weight`` / ``bias`` of the merged layers with
    ``object.__setattr__`` and its parameter modifiers re-register them with
    ``load_state_dict(assign=True)``; after the context the forward reads
    instance attributes while ``parameters()`` holds the merged tensors.
    The registered parameters and buffers and the instance attributes
    ``forward``, ``weight``, ``bias``, ... of every module are restored on
    exit, so later in-place parameter changes (cascading randomization)
    reach the forward pass.
    """
    saved = []
    for m in model.modules():
        names = set(m._parameters) | set(m._buffers) | {"forward"}
        saved.append((m, dict(m._parameters), dict(m._buffers),
                      {k: m.__dict__[k] for k in names if k in m.__dict__}))
    try:
        with composite.context(model) as modified:
            yield modified
    finally:
        for m, params, buffers, attrs in saved:
            m._parameters.clear()
            m._parameters.update(params)
            m._buffers.clear()
            m._buffers.update(buffers)
            for k in (set(params) | set(buffers) | {"forward"}) - set(attrs):
                m.__dict__.pop(k, None)
            m.__dict__.update(attrs)


def _standard_cnn_composite(rule, model: nn.Module):
    """zennit's standard composite construction with ``rule`` on all Conv / Linear layers.

    As zennit's built-in composites: ``layer_map_base()`` (Pass on
    activations and BatchNorm, Norm on the residual sums inserted by the
    canonizer and on average pooling) followed by ``rule`` on
    ``zennit.types.Convolution`` and ``nn.Linear``, with the model's
    torchvision canonizers (BatchNorm merged into the preceding
    convolution).
    """
    from zennit.composites import LayerMapComposite, layer_map_base
    from zennit.types import Convolution
    return LayerMapComposite(
        layer_map=layer_map_base() + [(Convolution, rule), (nn.Linear, rule)],
        canonizers=_get_canonizers(model),
    )


def _no_layers(method: AttributionMethod, layers: Optional[List[str]]) -> None:
    if layers and any(layer != "input" for layer in layers):
        raise NotImplementedError(f"{method.config().name}: per-layer maps are not available")


def _cnn_lrp(model: nn.Module, composite, x: Tensor, target: Tensor) -> Dict[str, np.ndarray]:
    """Input relevance map (channel sum) of a zennit composite on a CNN."""
    x_input = x.detach().clone().requires_grad_(True)
    with zennit_context(composite, model) as modified_model:
        output = modified_model(x_input)
        one_hot = torch.zeros_like(output)
        for i in range(target.shape[0]):
            one_hot[i, target[i]] = 1.0
        output.backward(gradient=one_hot)
        relevance = x_input.grad
        if relevance is None:
            raise RuntimeError("No gradient on input after backward pass")
    return {"input": tensor_to_numpy(relevance.sum(dim=1))}


def _vit_lrp(method: AttributionMethod, x: Tensor, target: Tensor,
             layers: Optional[List[str]], rule_for) -> Dict[str, np.ndarray]:
    """Input relevance map of a torchvision ViT under lxt's rules (:mod:`._lxt_vit`).

    The output relevance is the target logit.
    """
    _no_layers(method, layers)

    def output_grad(logits: Tensor) -> Tensor:
        return F.one_hot(target.to(logits.device), logits.shape[1]).to(logits.dtype)

    relevance = _lxt_vit.lrp_heatmap(method.model, x, output_grad, rule_for)
    return {"input": tensor_to_numpy(relevance.sum(dim=1))}


# ---------------------------------------------------------------------------
# LRP-epsilon
# ---------------------------------------------------------------------------

@MethodRegistry.register
class LRPEpsilon(AttributionMethod):
    """LRP-ε (Bach et al. 2015).

    CNNs (``zennit``): the ε-rule on every Conv / Linear layer,
    R_j = Σ_k a_j w_jk / (z_k + ε·sign(z_k)) · R_k, in zennit's standard
    composite construction (``layer_map_base()``: Norm rule on residual sums
    and average pooling, Pass on activations; torchvision canonizers merge
    BatchNorm into the preceding convolution).

    torchvision ViTs: lxt's efficient rules for LayerNorm, GELU, attention
    (CP-LRP) and residual connections (:mod:`._lxt_vit`) with ``Epsilon(ε)``
    on the Conv2d and Linear layers; the attention projections are
    propagated with Gradient x Input as in lxt's ViT recipe.
    """

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="lrp_standard",
            category=MethodCategory.LRP,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=False,
            description="Pure epsilon-LRP: uniform epsilon rule on all layers.",
            reference="Bach et al. 2015",
        )

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.epsilon = 0.25

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        if _lxt_vit.is_torchvision_vit(self.model):
            rule = _lxt_vit.Epsilon(self.epsilon)
            return _vit_lrp(self, x, target, layers,
                            _lxt_vit.rule_map(conv=rule, linear=rule))
        _no_layers(self, layers)
        try:
            from zennit.rules import Epsilon

            composite = _standard_cnn_composite(Epsilon(epsilon=self.epsilon), self.model)
            return _cnn_lrp(self.model, composite, x, target)
        except Exception as e:
            logger.warning("LRPEpsilon (zennit) failed: %s. Falling back to the gradient.", e)
            x_input = x.detach().clone().requires_grad_(True)
            output = self.model(x_input)
            one_hot = torch.zeros_like(output)
            for i in range(target.shape[0]):
                one_hot[i, target[i]] = 1.0
            output.backward(gradient=one_hot)
            return {"input": tensor_to_numpy(x_input.grad.sum(dim=1))}


# ---------------------------------------------------------------------------
# AttnLRP (lxt efficient rules, torchvision ViTs)
# ---------------------------------------------------------------------------

@MethodRegistry.register
class AttnLRP(AttributionMethod):
    """AttnLRP (Achtibat et al., ICML 2024) for torchvision Vision Transformers.

    Computed with lxt's efficient rules for torchvision ViTs
    (``lxt.efficient.models.vit_torch.cp_LRP``; see :mod:`._lxt_vit`):
    identity rule on LayerNorm and GELU, CP-LRP inside the attention
    (queries and keys constant), Gradient x Input on residual connections and
    the attention-weighted sum, and zennit rules on the Linear / Conv2d
    layers.  Two composites, selected via ``gamma_composite``:

    * ``False``: ``Epsilon(epsilon)`` on the Conv2d and Linear layers, as in
      lxt's ViT recipe with an ε-composite.  On ViTs this is the same
      computation as ``lrp_standard``.
    * ``True``: the ViT composite of Achtibat et al. (2024, Table B.5):
      ``Gamma(lrp_gamma_conv)`` on the patch embedding,
      ``Gamma(lrp_gamma_linear)`` on the Linear layers outside the attention
      and ``Epsilon(lrp_epsilon_attn)`` on the Q/K/V and output projections
      of the attention.  The default ``lrp_epsilon_attn`` is the default ε
      of lxt's Epsilon rules (1e-6); in this limit the composite coincides
      with lxt's ViT example (``examples/vit_torch.py``), which applies
      ``Gamma`` to ``nn.Conv2d`` / ``nn.Linear`` modules and propagates the
      projections inside ``nn.MultiheadAttention`` with Gradient x Input.

    Gamma rules use zennit's default stabiliser (1e-6), as in lxt's example.
    """

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="attn_lrp",
            category=MethodCategory.LRP,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=False,
            description="Attention-aware LRP (lxt efficient rules for ViTs).",
            reference="Achtibat et al. 2024",
        )

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.gamma_composite = False
        self.epsilon = 0.25
        self.lrp_gamma_conv = 0.25
        self.lrp_gamma_linear = 0.05
        self.lrp_epsilon_attn = 1e-6

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        if not _lxt_vit.is_torchvision_vit(self.model):
            raise TypeError("attn_lrp is implemented for torchvision Vision Transformers, "
                            f"got {type(self.model).__name__}")
        if self.gamma_composite:
            rule_for = _lxt_vit.rule_map(
                conv=_lxt_vit.Gamma(self.lrp_gamma_conv),
                linear=_lxt_vit.Gamma(self.lrp_gamma_linear),
                attention=_lxt_vit.Epsilon(self.lrp_epsilon_attn),
            )
        else:
            rule = _lxt_vit.Epsilon(self.epsilon)
            rule_for = _lxt_vit.rule_map(conv=rule, linear=rule)
        return _vit_lrp(self, x, target, layers, rule_for)


# ---------------------------------------------------------------------------
# GammaLRP (zennit)
# ---------------------------------------------------------------------------

@MethodRegistry.register
class GammaLRP(AttributionMethod):
    """Gamma-LRP via ``zennit`` with tunable gamma parameter.

    The gamma rule amplifies positive contributions by a factor of (1+gamma)
    relative to negative ones, producing sharper, more focused attributions
    with larger gamma values.

    CNNs (``zennit``): ``'all_gamma'`` applies ``Gamma(γ, stabilizer=ε)`` to
    every Conv / Linear layer in zennit's standard composite construction
    (see :class:`LRPEpsilon`); ``'epsilon_gamma_box'`` is zennit's
    ``EpsilonGammaBox``.

    torchvision ViTs: lxt's efficient rules for LayerNorm, GELU, attention
    (CP-LRP) and residual connections (:mod:`._lxt_vit`) with
    ``Gamma(γ, stabilizer=ε)`` on the Conv2d and Linear layers
    (``'all_gamma'``) or the rules of zennit's ``EpsilonGammaBox``
    (``'epsilon_gamma_box'``); the attention projections are propagated with
    Gradient x Input as in lxt's ViT recipe.
    """

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="gamma_lrp",
            category=MethodCategory.LRP,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=False,
            description="Gamma-LRP: amplify positive contributions (Montavon et al. 2019).",
            reference="Montavon et al. 2019",
        )

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.gamma = 0.25
        self.epsilon = 0.25
        self.low = -3.0
        self.high = 3.0
        # 'all_gamma': Gamma on every Linear/Conv layer (on ViTs zennit's
        # EpsilonGammaBox reduces to epsilon-LRP, as they have no convolution
        # after the input layer).  'epsilon_gamma_box': zennit's standard
        # EpsilonGammaBox (ZBox input layer, Gamma on convolutions, Epsilon on
        # dense layers).  CNNs: both use zennit's layer_map_base() and the
        # torchvision canonizers.
        self.composite = 'all_gamma'

    def _vit_rule_map(self):
        if self.composite == 'all_gamma':
            rule = _lxt_vit.Gamma(self.gamma, stabilizer=self.epsilon)
            return _lxt_vit.rule_map(conv=rule, linear=rule)
        if self.composite == 'epsilon_gamma_box':
            return _lxt_vit.rule_map(
                conv=_lxt_vit.Gamma(self.gamma),
                linear=_lxt_vit.Epsilon(self.epsilon),
                first=_lxt_vit.ZBox(self.low, self.high),
            )
        raise ValueError(f"composite must be 'all_gamma' or 'epsilon_gamma_box', "
                         f"got {self.composite!r}")

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        if _lxt_vit.is_torchvision_vit(self.model):
            return _vit_lrp(self, x, target, layers, self._vit_rule_map())
        _no_layers(self, layers)
        try:
            from zennit.rules import Gamma

            if self.composite == 'epsilon_gamma_box':
                from zennit.composites import EpsilonGammaBox
                composite = EpsilonGammaBox(
                    low=self.low, high=self.high, epsilon=self.epsilon, gamma=self.gamma,
                    canonizers=_get_canonizers(self.model),
                )
            elif self.composite == 'all_gamma':
                composite = _standard_cnn_composite(
                    Gamma(gamma=self.gamma, stabilizer=self.epsilon), self.model)
            else:
                raise ValueError(f"composite must be 'all_gamma' or 'epsilon_gamma_box', "
                                 f"got {self.composite!r}")
            return _cnn_lrp(self.model, composite, x, target)
        except Exception as e:
            logger.warning("GammaLRP (zennit) failed: %s. Falling back to gradient x input.", e)
            x_input = x.detach().clone().requires_grad_(True)
            output = self.model(x_input)
            one_hot = torch.zeros_like(output)
            for i in range(target.shape[0]):
                one_hot[i, target[i]] = 1.0
            output.backward(gradient=one_hot)
            relevance = x_input.grad * x_input
            spatial = relevance.sum(dim=1)
            return {"input": tensor_to_numpy(spatial)}
