"""Gradient- and perturbation-based attribution methods.

Implements RawGradient, SmoothGrad, IntegratedGradients, DeepLIFT (captum;
zennit LRP rules as alternatives), LIME, and KernelSHAP.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from captum.attr import KernelShap, Lime

from ..base import AttributionMethod
from ..layer_hooks import LayerGradientCapture, _activation_to_spatial, _find_module
from ..postprocess import tensor_to_numpy
from ..registry import MethodRegistry
from ..types import MethodCategory, MethodConfig

logger = logging.getLogger(__name__)

# ImageNet normalisation of the model inputs (torchvision convention).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CHANNEL_REDUCTIONS = ("sum", "sum_abs", "max_abs")


def reduce_channels(attr: Tensor, mode: str = "sum") -> Tensor:
    """Reduce a ``(B, C, H, W)`` attribution to a ``(B, H, W)`` map.

    * ``'sum'``: signed sum over channels (captum ``visualize_image_attr``).
    * ``'sum_abs'``: sum of absolute values over channels (PAIR ``saliency``
      library, ``VisualizeImageGrayscale``; Smilkov et al. 2017).
    * ``'max_abs'``: maximum absolute value over channels, the saliency map of
      Simonyan et al. 2014, ``M_ij = max_c |w_h(i,j,c)|``.
    """
    if mode == "sum":
        return attr.sum(dim=1)
    if mode == "sum_abs":
        return attr.abs().sum(dim=1)
    if mode == "max_abs":
        return attr.abs().amax(dim=1)
    raise ValueError(f"channel_reduction must be one of {CHANNEL_REDUCTIONS}, got {mode!r}")


def input_baseline(x: Tensor, baseline: str = "zero") -> Tensor:
    """Reference input for path / reference-based attributions.

    * ``'zero'``: all zeros in the normalised input space, i.e. the ImageNet
      mean colour (captum's default baseline).
    * ``'black'``: a black image, ``(0 - mean) / std`` per channel
      (Sundararajan et al. 2017).
    """
    if baseline == "zero":
        return torch.zeros_like(x)
    if baseline == "black":
        mean = torch.tensor(IMAGENET_MEAN, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
        return ((0.0 - mean) / std).expand_as(x).clone()
    raise ValueError(f"baseline must be 'zero' or 'black', got {baseline!r}")


def _build_gradient_seed(output: Tensor, target: Tensor) -> Tensor:
    """One-hot backward seed of the target logits."""
    one_hot = torch.zeros_like(output)
    for i in range(target.shape[0]):
        one_hot[i, target[i]] = 1.0
    return one_hot


# ---------------------------------------------------------------------------
# Helper for perturbation-based methods (LIME / KernelSHAP)
# ---------------------------------------------------------------------------


def _get_feature_mask(x: Tensor, patch_size: int = 16) -> Tensor:
    """Create a grid-based feature mask for perturbation methods.

    Divides the spatial dimensions into a grid of ``patch_size x patch_size``
    pixel patches, each assigned a unique integer label.  For a 224x224 input
    with patch_size=16 this yields a 14x14 = 196 superpixel grid.

    Args:
        x: Input tensor of shape ``(B, C, H, W)``.
        patch_size: Side length of each square superpixel patch.

    Returns:
        Integer tensor of shape ``(B, C, H, W)`` where each spatial region
        shares the same label across all channels.
    """
    B, C, H, W = x.shape
    grid_h = (H + patch_size - 1) // patch_size
    grid_w = (W + patch_size - 1) // patch_size

    # Build a (H, W) label map
    label_map = torch.zeros(H, W, dtype=torch.long, device=x.device)
    for i in range(grid_h):
        for j in range(grid_w):
            y0 = i * patch_size
            y1 = min((i + 1) * patch_size, H)
            x0 = j * patch_size
            x1 = min((j + 1) * patch_size, W)
            label_map[y0:y1, x0:x1] = i * grid_w + j

    # Expand to (B, C, H, W) — same mask for all channels and batch elements
    mask = label_map.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    return mask


def _get_superpixel_mask(
    x: Tensor, method: str = "slic", n_segments: int = 80,
) -> Tensor:
    """Create a superpixel feature mask for perturbation methods.

    Args:
        x: Input tensor of shape ``(B, C, H, W)``.
        method: Segmentation algorithm — ``slic``, ``quickshift``, or ``felzenszwalb``.
        n_segments: Approximate number of superpixel segments.

    Returns:
        Integer tensor of shape ``(B, C, H, W)`` where each superpixel
        shares the same label across all channels.
    """
    B, C, H, W = x.shape
    masks = []

    for b in range(B):
        img = x[b].detach().cpu().float()
        # Denormalize from ImageNet normalization to [0, 1]
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        img = img * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()  # (H, W, C)

        if method == "quickshift":
            from skimage.segmentation import quickshift
            kernel_size = max(3, int(H / (n_segments ** 0.5)))
            segments = quickshift(img, kernel_size=kernel_size, max_dist=10, ratio=0.5)
        elif method == "felzenszwalb":
            from skimage.segmentation import felzenszwalb
            scale = max(10, 50000 / max(n_segments, 1))
            segments = felzenszwalb(img, scale=scale, sigma=0.5, min_size=50)
        else:  # slic (default)
            from skimage.segmentation import slic
            segments = slic(
                img, n_segments=n_segments, compactness=10, sigma=1.0,
                start_label=0, channel_axis=2,
            )

        label_tensor = torch.from_numpy(segments.astype(int)).long().to(x.device)
        masks.append(label_tensor.unsqueeze(0).unsqueeze(0).expand(1, C, -1, -1))

    return torch.cat(masks, dim=0)


# ---------------------------------------------------------------------------
# Gradient-based methods
# ---------------------------------------------------------------------------


def _no_layers(method: AttributionMethod, layers: Optional[List[str]]) -> None:
    if layers and any(layer != "input" for layer in layers):
        raise NotImplementedError(f"{method.config().name}: per-layer maps are not available")


@MethodRegistry.register
class RawGradient(AttributionMethod):
    """Gradient of the class score w.r.t. the input, ``d(score)/d(x)``.

    ``channel_reduction`` reduces the ``(B, C, H, W)`` gradient to a map (see
    :func:`reduce_channels`): ``'sum'`` keeps the signed channel sum;
    ``'max_abs'`` is the saliency map of Simonyan et al. 2014.
    """

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="raw_gradient",
            category=MethodCategory.GRADIENT,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=False,
            description="Gradient of the class score w.r.t. the input image.",
            reference="Simonyan et al. 2014",
        )

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.channel_reduction = "sum"

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        _no_layers(self, layers)
        x_input = x.detach().clone().requires_grad_(True)
        output = self.model(x_input)
        seed = _build_gradient_seed(output, target)
        output.backward(gradient=seed)

        spatial = reduce_channels(x_input.grad, self.channel_reduction)
        return {"input": tensor_to_numpy(spatial)}


@MethodRegistry.register
class SmoothGrad(AttributionMethod):
    """SmoothGrad: mean or std of gradients over noisy copies of the input.

    SmoothGrad (Smilkov et al. 2017, Eq. 2) averages the gradients of
    ``n = nt_samples`` copies ``x + N(0, sigma^2)``.  The noise level is

    * ``sigma = stdev_spread * (max(x) - min(x))`` per image when
      ``stdev_spread`` is set: the paper's parametrisation (10-20 % noise
      works best) and that of the authors' reference implementation (PAIR
      ``saliency``, ``GetSmoothedMask(stdev_spread=0.15)``);
    * ``sigma = stdevs`` in normalised input units otherwise.  The range of
      an ImageNet-normalised channel is ``1 / std = 4.4``, so ``stdevs = 0.66``
      corresponds to about 15 %.

    ``use_abs=False`` averages the signed gradients (the paper's definition);
    ``use_abs=True`` averages their absolute values.  ``use_std_aggregation``
    returns the standard deviation instead of the mean.
    ``channel_reduction`` maps the result to ``(B, H, W)`` (see
    :func:`reduce_channels`).
    """

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="smooth_grad",
            category=MethodCategory.GRADIENT,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=False,
            description="Mean or std of gradients over noisy copies of the input.",
            reference="Smilkov et al. 2017",
        )

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.nt_samples = 20
        self.stdevs = 0.1
        self.stdev_spread: Optional[float] = None
        self.use_std_aggregation = False
        self.use_abs = True
        self.channel_reduction = "sum"

    def _noise_std(self, x: Tensor) -> Union[float, Tensor]:
        """Noise standard deviation, per image when ``stdev_spread`` is set."""
        if self.stdev_spread is None:
            return float(self.stdevs)
        flat = x.detach().flatten(1)
        value_range = flat.amax(dim=1) - flat.amin(dim=1)
        return (float(self.stdev_spread) * value_range).view(-1, *([1] * (x.ndim - 1)))

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        _no_layers(self, layers)
        N = self.nt_samples
        sigma = self._noise_std(x)

        # Running sum and sum of squares of the (absolute) gradients.
        g_sum = None
        g_sq = None
        for _ in range(N):
            noise = torch.randn_like(x) * sigma
            x_input = (x + noise).detach().clone().requires_grad_(True)
            output = self.model(x_input)
            seed = _build_gradient_seed(output, target)
            output.backward(gradient=seed)

            g = x_input.grad.detach()
            if self.use_abs:
                g = g.abs()
            if g_sum is None:
                g_sum = g.clone()
                g_sq = (g * g).clone()
            else:
                g_sum += g
                g_sq += g * g

        mean = g_sum / N
        if self.use_std_aggregation:
            var = g_sq / N - mean * mean
            agg = torch.sqrt(torch.clamp(var, min=0))
        else:
            agg = mean
        return {"input": tensor_to_numpy(reduce_channels(agg, self.channel_reduction))}


@MethodRegistry.register
class IntegratedGradientsMethod(AttributionMethod):
    """Integrated Gradients (Sundararajan et al. 2017).

    ``(x - x') * mean_k grad f(x' + a_k (x - x'))`` with the midpoint Riemann
    rule ``a_k = (k + 1/2) / n_steps`` (captum ``method='riemann_middle'``).
    ``baseline`` selects the reference ``x'`` (see :func:`input_baseline`):
    ``'zero'`` (captum's default, the ImageNet mean colour) or ``'black'``
    (the black image of Sundararajan et al.).  The attribution is reduced over
    channels with ``channel_reduction`` (default ``'sum'``, as in captum's
    ``visualize_image_attr``).
    """

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="integrated_gradients",
            category=MethodCategory.GRADIENT,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=False,
            description="Path integral of gradients from a baseline image.",
            reference="Sundararajan et al. 2017",
        )

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.n_steps = 30
        self.baseline = "zero"
        self.channel_reduction = "sum"

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        _no_layers(self, layers)
        baseline = input_baseline(x, self.baseline)
        diff = x - baseline
        K = self.n_steps
        accum_input = torch.zeros_like(x)

        for step in range(K):
            alpha = (step + 0.5) / K
            x_alpha = (baseline + alpha * diff).detach().clone().requires_grad_(True)
            output = self.model(x_alpha)
            seed = _build_gradient_seed(output, target)
            output.backward(gradient=seed)
            accum_input += x_alpha.grad

        # IG formula: (1/K) * sum(grads) * (x - baseline)
        ig_attr = (accum_input / K) * diff
        spatial = reduce_channels(ig_attr, self.channel_reduction)
        return {"input": tensor_to_numpy(spatial)}


@MethodRegistry.register
class DeepLIFT(AttributionMethod):
    """DeepLIFT (``rule='rescale'``) or a zennit LRP rule on all layers.

    ``rule='rescale'``: DeepLIFT with the Rescale rule (Shrikumar et al. 2017)
    via ``captum.attr.DeepLift`` (``multiply_by_inputs=True``), i.e.
    ``(x - x') * m`` with the DeepLIFT multipliers ``m`` of the target logit
    and the reference ``x'`` given by ``baseline`` (see
    :func:`input_baseline`; ``'zero'`` is captum's default).  It runs on a
    copy of the model with the standard module forwards and one ReLU module
    per application (:mod:`._deeplift_network`, which also describes the
    max-pool treatment); attention models raise ``NotImplementedError``.  The
    RevealCancel rule of the paper is not implemented in captum.

    Other rules: the zennit LRP rule of that name (epsilon, gamma, z+,
    alpha-beta, flat, w-square; Montavon et al. 2019) on every Linear and
    Conv2d module, with per-layer overrides in ``layer_rules``.  These are LRP
    variants, not DeepLIFT.
    """

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="deeplift",
            category=MethodCategory.LRP,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=False,
            description="DeepLIFT Rescale (captum) or zennit LRP rules.",
            reference="Shrikumar et al. 2017; Montavon et al. 2019",
        )

    RULE_NAMES = ['rescale', 'epsilon', 'gamma', 'z_plus', 'alpha_beta', 'flat', 'w_square']

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.rule = 'epsilon'
        self.epsilon = 0.25
        self.gamma = 0.25
        self.layer_rules: Dict[str, str] = {}  # {layer_name: rule_name}
        self.baseline = 'zero'
        self.channel_reduction = 'sum'

    def _rescale(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]],
    ) -> Dict[str, np.ndarray]:
        """DeepLIFT (Rescale rule) via ``captum.attr.DeepLift``."""
        from captum.attr import DeepLift, LayerDeepLift
        from ._deeplift_network import deeplift_network

        net = deeplift_network(self.model)
        explained = net
        inputs = x.detach().clone().requires_grad_(True)
        baselines = input_baseline(inputs, self.baseline)

        attr = DeepLift(explained).attribute(inputs, baselines=baselines, target=target)
        results = {"input": tensor_to_numpy(reduce_channels(attr, self.channel_reduction))}
        for name in layers or []:
            module = _find_module(net, name)
            if module is None:
                logger.warning("DeepLIFT: layer %s not found in the network copy", name)
                continue
            layer_attr = LayerDeepLift(explained, module).attribute(
                inputs, baselines=baselines, target=target)
            results[name] = tensor_to_numpy(_activation_to_spatial(layer_attr))
        return results

    def _make_rule(self, name: str):
        """Instantiate a zennit rule by name."""
        from zennit.rules import Epsilon, Gamma, ZPlus, AlphaBeta, Flat, WSquare
        if name == 'gamma':
            return Gamma(gamma=self.gamma)
        elif name == 'z_plus':
            return ZPlus()
        elif name == 'alpha_beta':
            return AlphaBeta(alpha=2, beta=1)
        elif name == 'flat':
            return Flat()
        elif name == 'w_square':
            return WSquare()
        else:  # 'epsilon' or default
            return Epsilon(epsilon=self.epsilon)

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        if self.rule == 'rescale':
            return self._rescale(x, target, layers)

        from zennit.composites import NameMapComposite

        # Group modules by their assigned rule
        rule_groups: Dict[str, list] = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                rule_name = self.layer_rules.get(name, self.rule)
                if rule_name not in rule_groups:
                    rule_groups[rule_name] = []
                rule_groups[rule_name].append(name)

        name_map = [
            (tuple(names), self._make_rule(rn))
            for rn, names in rule_groups.items()
        ]
        # Add Pass() for all LayerNorm modules by name
        ln_names = [
            n for n, m in self.model.named_modules()
            if isinstance(m, nn.LayerNorm)
        ]
        if ln_names:
            from zennit.rules import Pass
            name_map.append((tuple(ln_names), Pass()))
        from bwgame.xai.methods.lrp_library import _get_canonizers, zennit_context
        composite = NameMapComposite(
            name_map=name_map,
            canonizers=_get_canonizers(self.model),
        )

        x_input = x.detach().clone().requires_grad_(True)

        layer_capture = None
        if layers:
            layer_capture = LayerGradientCapture(self.model, layers)
            layer_capture.install()

        try:
            with zennit_context(composite, self.model) as modified_model:
                output = modified_model(x_input)
                one_hot = torch.zeros_like(output)
                for i in range(target.shape[0]):
                    one_hot[i, target[i]] = 1.0
                output.backward(gradient=one_hot)

                relevance = x_input.grad
                if relevance is None:
                    raise RuntimeError("No gradient on input after backward pass")
                spatial = reduce_channels(relevance, self.channel_reduction)

            results = {"input": tensor_to_numpy(spatial)}
            if layer_capture:
                # Use gradient × activation for LRP-type per-layer relevance;
                # plain gradient sums cancel over ViT's high-D token vectors.
                results.update(layer_capture.get_maps(mode='gradient_x_act'))
            return results

        finally:
            if layer_capture:
                layer_capture.remove()


# ---------------------------------------------------------------------------
# Perturbation-based methods
# ---------------------------------------------------------------------------


def _softmax_forward(method):
    """Forward function of the perturbation methods: class probabilities."""
    return lambda inp: F.softmax(method.model(inp), dim=-1)


FILLS = ("zero", "segment_mean", "image_mean")


def fill_values(x: Tensor, feature_mask: Tensor, fill: str = "zero") -> Union[float, Tensor]:
    """Values that replace removed superpixels (captum ``baselines``).

    * ``'zero'``: 0 in the normalised input space, i.e. the ImageNet mean
      colour (captum's default).
    * ``'segment_mean'``: the mean colour of each superpixel, as in
      ``lime_image.LimeImageExplainer.explain_instance(hide_color=None)``
      (Ribeiro et al. 2016).
    * ``'image_mean'``: the mean colour of the image, the default background
      of ``mask_image`` in the shap KernelExplainer image example.

    Normalisation is affine per channel, so means in the normalised space are
    the normalised means of the RGB colours.
    """
    if fill == "zero":
        return 0.0
    if fill == "image_mean":
        return x.mean(dim=(2, 3), keepdim=True).expand_as(x).clone()
    if fill == "segment_mean":
        B, C, H, W = x.shape
        labels = feature_mask[:, :1].reshape(B, 1, H * W).long()
        n = int(labels.max()) + 1
        flat = x.reshape(B, C, H * W)
        sums = torch.zeros(B, C, n, dtype=x.dtype, device=x.device)
        sums.scatter_add_(2, labels.expand(B, C, H * W), flat)
        counts = torch.zeros(B, 1, n, dtype=x.dtype, device=x.device)
        counts.scatter_add_(2, labels, torch.ones_like(flat[:, :1]))
        means = sums / counts.clamp(min=1)
        return means.gather(2, labels.expand(B, C, H * W)).reshape(B, C, H, W)
    raise ValueError(f"fill must be one of {FILLS}, got {fill!r}")


def _lime_image_sampler(original_inp: Tensor, **kwargs):
    """Binary samples as in ``lime_image``: the unperturbed input (all ones)
    first, then i.i.d. Bernoulli(0.5) masks."""
    m = int(kwargs["num_interp_features"])
    device = original_inp.device
    yield torch.ones(1, m, dtype=torch.long, device=device)
    while True:
        yield torch.bernoulli(torch.full((1, m), 0.5)).long().to(device)


def _binary_cosine_kernel(kernel_width: float):
    """``lime_image`` sample weights: ``exp(-d^2 / (2 w^2))``
    (= ``sqrt(exp(-d^2 / w^2))``) with ``d`` the cosine distance between the
    binary representation ``z`` and the all-ones vector, ``1 - sqrt(|z| / M)``."""
    def similarity(original_inp, perturbed_inp, interpretable_inp, **kwargs):
        z = interpretable_inp.float().flatten()
        d = 1.0 - math.sqrt(float(z.sum()) / z.numel())
        return math.exp(-(d ** 2) / (2 * kernel_width ** 2))
    return similarity


def _feature_mask_for(method, x: Tensor) -> Tensor:
    if method.superpixel_method != 'grid':
        return _get_superpixel_mask(
            x, method=method.superpixel_method,
            n_segments=method.superpixel_n_segments,
        )
    return _get_feature_mask(x)


@MethodRegistry.register
class LIMEMethod(AttributionMethod):
    """LIME (Ribeiro et al. 2016) via ``captum.attr.Lime``.

    Features are superpixels: a 14x14 grid of 16-px patches
    (``superpixel_method='grid'``) or ``slic`` / ``quickshift`` /
    ``felzenszwalb`` segments.  Removed superpixels are replaced according to
    ``fill`` (see :func:`fill_values`).

    Sample weights: ``kernel_distance='input'`` uses captum's exponential
    kernel on the cosine distance between the original and the perturbed
    model input; ``kernel_distance='binary'`` uses the ``lime_image``
    definition, the cosine distance between the binary representation and the
    all-ones vector, and (as ``lime_image``) includes the unperturbed input as
    the first sample.  ``kernel_width=None`` is captum's default width 1.0.
    With ``surrogate='ridge'``, ``kernel_width=0.25``,
    ``kernel_distance='binary'`` and ``fill='segment_mean'`` the method is
    ``lime_image.LimeImageExplainer`` (``hide_color=None``, Ridge(alpha=1))
    on the same segmentation.
    """

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="lime",
            category=MethodCategory.PERTURBATION,
            class_specific=True,
            supports_batch=False,
            requires_grad=False,
            per_layer=False,
            description="Local linear surrogate on superpixels.",
            reference="Ribeiro et al. 2016",
        )

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.n_samples = 500
        self.superpixel_method = 'grid'
        self.superpixel_n_segments = 80
        # Surrogate model: 'lasso' = captum default (SkLearnLasso, alpha 0.01),
        # which can zero every coefficient when the output changes are small;
        # 'ridge' = the reference LIME implementation (Ridge, alpha 1).
        self.surrogate = 'lasso'
        self.ridge_alpha = 1.0
        # Exponential kernel width (None = captum default 1.0; lime_image: 0.25).
        self.kernel_width = None
        self.kernel_distance = 'input'
        self.fill = 'zero'

    def _lime_kwargs(self) -> dict:
        kwargs = {}
        if self.surrogate == 'ridge':
            from captum._utils.models.linear_model import SkLearnRidge
            kwargs['interpretable_model'] = SkLearnRidge(alpha=self.ridge_alpha)
        elif self.surrogate != 'lasso':
            raise ValueError(f"surrogate must be 'lasso' or 'ridge', got {self.surrogate!r}")
        if self.kernel_distance == 'binary':
            width = 1.0 if self.kernel_width is None else float(self.kernel_width)
            kwargs['similarity_func'] = _binary_cosine_kernel(width)
            kwargs['perturb_func'] = _lime_image_sampler
        elif self.kernel_distance != 'input':
            raise ValueError(
                f"kernel_distance must be 'input' or 'binary', got {self.kernel_distance!r}")
        elif self.kernel_width is not None:
            from captum.attr._core.lime import get_exp_kernel_similarity_function
            kwargs['similarity_func'] = get_exp_kernel_similarity_function(
                'cosine', kernel_width=float(self.kernel_width))
        return kwargs

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        _no_layers(self, layers)
        lime = Lime(_softmax_forward(self), **self._lime_kwargs())
        feature_mask = _feature_mask_for(self, x)
        attr = lime.attribute(
            x,
            target=target,
            baselines=fill_values(x, feature_mask, self.fill),
            feature_mask=feature_mask,
            n_samples=self.n_samples,
            perturbations_per_eval=10,
        )
        spatial = attr.sum(dim=1)
        return {"input": tensor_to_numpy(spatial)}


@MethodRegistry.register
class KernelSHAPMethod(AttributionMethod):
    """KernelSHAP (Lundberg & Lee 2017) via ``captum.attr.KernelShap``.

    Features are superpixels as in :class:`LIMEMethod`; removed superpixels
    are replaced according to ``fill`` (see :func:`fill_values`).
    ``n_samples='auto'`` uses the shap ``KernelExplainer`` default
    ``2 M + 2048`` for ``M`` superpixels.
    """

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="kernel_shap",
            category=MethodCategory.PERTURBATION,
            class_specific=True,
            supports_batch=False,
            requires_grad=False,
            per_layer=False,
            description="Shapley value estimation via kernel-weighted linear model.",
            reference="Lundberg & Lee 2017",
        )

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.n_samples: Union[int, str] = 500
        self.superpixel_method = 'grid'
        self.superpixel_n_segments = 80
        self.fill = 'zero'

    def _num_samples(self, feature_mask: Tensor) -> int:
        if self.n_samples == 'auto':
            n_features = int(feature_mask.max()) + 1
            return 2 * n_features + 2048
        return int(self.n_samples)

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        _no_layers(self, layers)
        ks = KernelShap(_softmax_forward(self))
        feature_mask = _feature_mask_for(self, x)
        n_samples = self._num_samples(feature_mask)
        attr = ks.attribute(
            x,
            target=target,
            baselines=fill_values(x, feature_mask, self.fill),
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=10,
        )
        spatial = attr.sum(dim=1)
        return {"input": tensor_to_numpy(spatial)}
