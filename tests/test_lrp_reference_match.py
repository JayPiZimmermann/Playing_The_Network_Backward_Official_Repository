"""LRP baselines against their reference implementations.

* ViT (``lrp_standard``, ``gamma_lrp``, ``attn_lrp``): lxt's ViT recipe
  (lxt ``examples/vit_torch.py``: ``monkey_patch(vision_transformer,
  cp_LRP)``, ``monkey_patch_zennit()`` and a zennit ``LayerMapComposite`` on
  an original torchvision ViT), evaluated in the same process with the
  global patches undone afterwards.  The methods run on the decomposed model
  built by the evaluator.
* CNNs (``lrp_standard``, ``gamma_lrp``): zennit's standard composite
  construction (``layer_map_base()`` + the rule on Conv / Linear layers, or
  ``EpsilonGammaBox``) with the torchvision canonizers, on the original
  model.

The default tests use small randomly initialised models; the tests marked
``slow`` use the pretrained ViT-B/16 and ResNet-50 of the paper.
"""
from __future__ import annotations

import contextlib
import copy
import importlib
import os
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

pytest.importorskip("zennit")
pytest.importorskip("lxt")

from bwgame.xai.methods import _lxt_vit  # noqa: E402
from bwgame.xai.registry import MethodRegistry  # noqa: E402
import bwgame.xai.methods  # noqa: E402,F401  (method registration)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluator_model(model: nn.Module) -> nn.Module:
    """The model as ``Evaluator.__init__`` prepares it."""
    from bwgame.composite_replacer import decompose_composites
    from bwgame.functional_replacer import replace_inline_modules, _replace_functional_in_module
    decompose_composites(model)
    model.eval()
    replace_inline_modules(model, inplace=True)
    _replace_functional_in_module(model)
    return model


def _tiny_vit(seed: int = 0) -> nn.Module:
    from torchvision.models import VisionTransformer
    torch.manual_seed(seed)
    vit = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2,
                            hidden_dim=16, mlp_dim=32, num_classes=10)
    with torch.no_grad():  # torchvision initialises the head and class token with zeros
        nn.init.normal_(vit.heads.head.weight, std=0.5)
        nn.init.normal_(vit.heads.head.bias, std=0.1)
        nn.init.normal_(vit.class_token, std=0.5)
        for m in vit.modules():
            if isinstance(m, nn.LayerNorm):
                nn.init.normal_(m.weight, mean=1.0, std=0.2)
                nn.init.normal_(m.bias, std=0.2)
    return vit.eval()


def _tiny_resnet(seed: int = 0) -> nn.Module:
    from torchvision.models import resnet18
    torch.manual_seed(seed)
    net = resnet18(num_classes=10)
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.normal_(0.0, 0.2)
                m.running_var.uniform_(0.5, 2.0)
                nn.init.normal_(m.weight, mean=1.0, std=0.2)
                nn.init.normal_(m.bias, std=0.2)
    return net.eval()


def _import_lxt_vit_torch():
    """``lxt.efficient.models.vit_torch`` without running the ``lxt.efficient``
    package ``__init__`` files (they import lxt's HuggingFace patches)."""
    name = "lxt.efficient.models.vit_torch"
    if name in sys.modules:
        return sys.modules[name]
    _lxt_vit.lxt_efficient()
    import lxt
    root = os.path.join(os.path.dirname(lxt.__file__), "efficient")
    stubs = []
    for pkg, path in (("lxt.efficient", root), ("lxt.efficient.models", os.path.join(root, "models"))):
        if pkg not in sys.modules:
            module = types.ModuleType(pkg)
            module.__path__ = [path]
            sys.modules[pkg] = module
            stubs.append(pkg)
    try:
        return importlib.import_module(name)
    finally:
        for pkg in stubs:
            del sys.modules[pkg]


def _patched_state():
    from zennit.core import BasicHook
    classes = (nn.GELU, nn.LayerNorm, nn.MultiheadAttention, BasicHook)
    return {(cls, k): cls.__dict__.get(k) for cls in classes
            for k in ("forward", "backward", "original_forward")}


@contextlib.contextmanager
def _lxt_vit_recipe():
    """lxt's global ViT patches (``monkey_patch(vision_transformer,
    patch_map=cp_LRP)`` and ``monkey_patch_zennit()``), undone on exit."""
    vit_torch = _import_lxt_vit_torch()
    zennit_patches = _lxt_vit.lxt_efficient().zennit_patches
    saved = _patched_state()
    try:
        for target, patch in vit_torch.cp_LRP.items():
            assert patch(target), target
        zennit_patches.monkey_patch_zennit()
        yield
    finally:
        for (cls, key), value in saved.items():
            if value is None:
                if key in cls.__dict__:
                    delattr(cls, key)
            else:
                setattr(cls, key, value)


def _lxt_reference(vit: nn.Module, x: torch.Tensor, target: torch.Tensor, make_composite) -> np.ndarray:
    """Heatmap of lxt's ViT recipe on a copy of the original (undecomposed) ViT."""
    ref = copy.deepcopy(vit).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    x = x.detach().clone().requires_grad_(True)
    with _lxt_vit_recipe():
        composite = make_composite()
        composite.register(ref)
        try:
            y = ref(x)
            y[torch.arange(len(target)), target].sum().backward()
        finally:
            composite.remove()
    return (x * x.grad).sum(1).detach().numpy()


def _run(name: str, model: nn.Module, x: torch.Tensor, target: torch.Tensor, **params) -> np.ndarray:
    method = MethodRegistry.instantiate(name, model, torch.device("cpu"))
    for k, v in params.items():
        assert hasattr(method, k), k
        setattr(method, k, v)
    return method.attribute(x, target)["input"]


def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _vit_cases():
    """(method, parameters, zennit composite of lxt's recipe)."""
    from zennit.composites import LayerMapComposite, SpecialFirstLayerMapComposite
    from zennit.rules import Epsilon, Gamma, ZBox
    return [
        ("lrp_standard", {"epsilon": 0.25},
         lambda: LayerMapComposite([(nn.Conv2d, Epsilon(0.25)), (nn.Linear, Epsilon(0.25))])),
        ("attn_lrp", {"gamma_composite": False, "epsilon": 0.1},
         lambda: LayerMapComposite([(nn.Conv2d, Epsilon(0.1)), (nn.Linear, Epsilon(0.1))])),
        ("gamma_lrp", {"gamma": 0.5, "epsilon": 0.25},
         lambda: LayerMapComposite([(nn.Conv2d, Gamma(0.5, stabilizer=0.25)),
                                    (nn.Linear, Gamma(0.5, stabilizer=0.25))])),
        ("gamma_lrp", {"composite": "epsilon_gamma_box", "gamma": 0.5, "epsilon": 0.25},
         lambda: SpecialFirstLayerMapComposite(
             layer_map=[(nn.Conv2d, Gamma(0.5)), (nn.Linear, Epsilon(0.25))],
             first_map=[(nn.Conv2d, ZBox(-3.0, 3.0))])),
        # lxt's recipe propagates the attention projections with Gradient x
        # Input, i.e. the limit lrp_epsilon_attn -> 0 of the Table B.5 composite.
        ("attn_lrp", {"gamma_composite": True, "lrp_gamma_conv": 0.25, "lrp_gamma_linear": 0.05,
                      "lrp_epsilon_attn": 1e-12},
         lambda: LayerMapComposite([(nn.Conv2d, Gamma(0.25)), (nn.Linear, Gamma(0.05))])),
    ]


# ---------------------------------------------------------------------------
# ViT: lxt reference (small random model)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", range(5))
def test_vit_methods_match_lxt_recipe(case):
    name, params, make_composite = _vit_cases()[case]
    vit = _tiny_vit()
    model = _evaluator_model(copy.deepcopy(vit))
    torch.manual_seed(1)
    x = torch.randn(3, 3, 32, 32)
    target = torch.tensor([0, 3, 7])

    ours = _run(name, model, x, target, **params)
    ref = _lxt_reference(vit, x, target, make_composite)

    assert ours.shape == (3, 32, 32)
    for a, b in zip(ours, ref):
        assert _rel_l2(a, b) < 1e-5


def test_vit_undecomposed_model_matches_decomposed():
    vit = _tiny_vit()
    decomposed = _evaluator_model(copy.deepcopy(vit))
    x = torch.randn(2, 3, 32, 32)
    target = torch.tensor([1, 2])
    for name, params, _ in _vit_cases():
        np.testing.assert_allclose(_run(name, vit, x, target, **params),
                                   _run(name, decomposed, x, target, **params), rtol=1e-4, atol=1e-7)


def test_vit_batch_equals_single_images():
    model = _evaluator_model(_tiny_vit())
    x = torch.randn(3, 3, 32, 32)
    target = torch.tensor([4, 5, 6])
    batch = _run("attn_lrp", model, x, target, gamma_composite=True)
    single = np.concatenate([_run("attn_lrp", model, x[i:i + 1], target[i:i + 1], gamma_composite=True)
                             for i in range(3)])
    np.testing.assert_allclose(batch, single, rtol=1e-4, atol=1e-7)


def test_attn_lrp_epsilon_attn_acts_on_attention_projections():
    model = _evaluator_model(_tiny_vit())
    x = torch.randn(1, 3, 32, 32)
    target = torch.tensor([2])
    small = _run("attn_lrp", model, x, target, gamma_composite=True, lrp_epsilon_attn=1e-6)
    large = _run("attn_lrp", model, x, target, gamma_composite=True, lrp_epsilon_attn=1.0)
    assert _rel_l2(large, small) > 1e-2


def test_vit_relevance_is_conserved_without_biases():
    """The rules are conservative: with the z-rule on all layers and without
    biases, class token and position embedding (which absorb relevance), the
    heatmap sums to the target logit."""
    vit = _tiny_vit()
    with torch.no_grad():
        vit.class_token.zero_()
        vit.encoder.pos_embedding.zero_()
        for m in vit.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.LayerNorm)) and m.bias is not None:
                m.bias.zero_()
            if isinstance(m, nn.MultiheadAttention):
                m.in_proj_bias.zero_()
    model = _evaluator_model(copy.deepcopy(vit))
    x = torch.randn(1, 3, 32, 32)
    target = torch.tensor([3])
    heat = _run("lrp_standard", model, x, target, epsilon=1e-9)
    with torch.no_grad():
        logit = vit(x)[0, 3].item()
    assert abs(logit) > 1e-2
    assert heat.sum() == pytest.approx(logit, rel=1e-3)


def test_attn_lrp_requires_vision_transformer():
    model = _evaluator_model(_tiny_resnet())
    with pytest.raises(TypeError):
        _run("attn_lrp", model, torch.randn(1, 3, 64, 64), torch.tensor([0]))


# ---------------------------------------------------------------------------
# No global state is modified
# ---------------------------------------------------------------------------

def _hook_counts(model: nn.Module):
    return [(len(m._forward_hooks), len(m._forward_pre_hooks), len(m._backward_hooks),
             len(m._backward_pre_hooks)) for m in model.modules()]


def test_vit_methods_leave_no_global_or_model_state():
    vit_model = _evaluator_model(_tiny_vit())
    cnn_model = _evaluator_model(_tiny_resnet())
    x_vit, t_vit = torch.randn(1, 3, 32, 32), torch.tensor([1])
    x_cnn, t_cnn = torch.randn(1, 3, 64, 64), torch.tensor([2])

    state = _patched_state()
    hooks = _hook_counts(vit_model)
    rg_before = _run("routing_game", vit_model, x_vit, t_vit)
    eps_cnn_before = _run("lrp_standard", cnn_model, x_cnn, t_cnn)

    for name, params, _ in _vit_cases():
        _run(name, vit_model, x_vit, t_vit, **params)

    assert _patched_state() == state
    assert _hook_counts(vit_model) == hooks
    assert np.array_equal(_run("routing_game", vit_model, x_vit, t_vit), rg_before)
    assert np.array_equal(_run("lrp_standard", cnn_model, x_cnn, t_cnn), eps_cnn_before)


# ---------------------------------------------------------------------------
# CNNs: zennit's standard composites
# ---------------------------------------------------------------------------

def _tiny_cnn(seed: int = 0) -> nn.Module:
    """VGG-style CNN (Conv-BN-ReLU, max / average pooling, dense head)."""
    torch.manual_seed(seed)
    net = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(16 * 16, 32), nn.ReLU(inplace=True),
        nn.Dropout(0.5), nn.Linear(32, 10),
    )
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.normal_(0.0, 0.2)
                m.running_var.uniform_(0.5, 2.0)
    return net.eval()


def _zennit_standard(net: nn.Module, x: torch.Tensor, target: torch.Tensor, make_composite) -> np.ndarray:
    """Relevance of zennit's composite on a copy of the original model."""
    ref = copy.deepcopy(net).eval()
    x = x.detach().clone().requires_grad_(True)
    with make_composite().context(ref) as m:
        y = m(x)
        y[torch.arange(len(target)), target].sum().backward()
    return x.grad.sum(1).numpy()


def _cnn_cases(canonizer):
    """(method, parameters, zennit composite): zennit's standard construction."""
    from zennit.composites import EpsilonGammaBox, LayerMapComposite, layer_map_base
    from zennit.rules import Epsilon, Gamma
    from zennit.types import Convolution

    def standard(rule):
        return lambda: LayerMapComposite(layer_map_base() + [(Convolution, rule), (nn.Linear, rule)],
                                         canonizers=[canonizer()])
    return [
        ("lrp_standard", {"epsilon": 0.25}, standard(Epsilon(0.25))),
        ("gamma_lrp", {"gamma": 0.5, "epsilon": 0.25}, standard(Gamma(0.5, stabilizer=0.25))),
        ("gamma_lrp", {"composite": "epsilon_gamma_box", "gamma": 0.5, "epsilon": 0.25},
         lambda: EpsilonGammaBox(low=-3.0, high=3.0, epsilon=0.25, gamma=0.5, canonizers=[canonizer()])),
    ]


@pytest.mark.parametrize("arch", ["resnet", "vgg"])
@pytest.mark.parametrize("case", range(3))
def test_cnn_methods_match_zennit_standard_composites(arch, case):
    from zennit.torchvision import ResNetCanonizer, VGGCanonizer
    net, canonizer = (_tiny_resnet(), ResNetCanonizer) if arch == "resnet" else (_tiny_cnn(), VGGCanonizer)
    name, params, make_composite = _cnn_cases(canonizer)[case]
    model = _evaluator_model(copy.deepcopy(net))
    x = torch.randn(2, 3, 64, 64)
    target = torch.tensor([1, 8])
    ours = _run(name, model, x, target, **params)
    ref = _zennit_standard(net, x, target, make_composite)
    np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-6 * np.abs(ref).max())


def test_resnet_residual_sums_receive_norm_rule():
    """The canonizer's residual sums get zennit's Norm rule: with it, LRP-0
    on a bias-free ResNet conserves the output relevance (zennit: the
    one-hot target mask, i.e. a total of 1) up to the stabilisers; without
    it both branches of every residual sum receive the full relevance."""
    net = _tiny_resnet()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.zero_()
                m.bias.zero_()
            if isinstance(m, nn.Linear):
                m.bias.zero_()
    model = _evaluator_model(copy.deepcopy(net))
    x = torch.randn(1, 3, 64, 64)
    target = torch.tensor([4])
    heat = _run("lrp_standard", model, x, target, epsilon=1e-9)
    assert heat.sum() == pytest.approx(1.0, rel=1e-3)


def test_cnn_methods_restore_model_instance_attributes():
    """zennit's ResNet canonizer removes its ``forward`` overload with
    ``delattr``; the traced forwards of the decomposed model must survive."""
    model = _evaluator_model(_tiny_resnet())
    forwards = {id(m): m.__dict__["forward"] for m in model.modules() if "forward" in m.__dict__}
    assert forwards
    x, target = torch.randn(1, 3, 64, 64), torch.tensor([3])
    rg_before = _run("routing_game", model, x, target)
    with torch.no_grad():
        logits = model(x)
    for name, params, _ in _cnn_cases(lambda: None):
        _run(name, model, x, target, **params)
    _run("deeplift", model, x, target)
    assert {id(m): m.__dict__["forward"] for m in model.modules() if "forward" in m.__dict__} == forwards
    with torch.no_grad():
        assert torch.equal(model(x), logits)
    assert np.array_equal(_run("routing_game", model, x, target), rg_before)


# ---------------------------------------------------------------------------
# Pretrained models of the paper
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_vit_b16_methods_match_lxt_recipe():
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).eval()
    model = _evaluator_model(copy.deepcopy(vit))
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        logits = vit(x)
        assert torch.allclose(model(x), logits, atol=1e-4)
    target = logits.argmax(1)
    for name, params, make_composite in _vit_cases():
        ours = _run(name, model, x, target, **params)
        ref = _lxt_reference(vit, x, target, make_composite)
        assert ours.shape == (1, 224, 224)
        assert np.corrcoef(ours.ravel(), ref.ravel())[0, 1] > 0.9999, (name, params)


@pytest.mark.slow
@pytest.mark.parametrize("arch", ["resnet50", "vgg16"])
def test_pretrained_cnn_methods_match_zennit_standard_composites(arch):
    from torchvision import models
    from zennit.torchvision import ResNetCanonizer, VGGCanonizer
    if arch == "resnet50":
        net, canonizer = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2), ResNetCanonizer
    else:
        net, canonizer = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1), VGGCanonizer
    net.eval()
    model = _evaluator_model(copy.deepcopy(net))
    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)
    target = torch.tensor([207])
    for name, params, make_composite in _cnn_cases(canonizer):
        ours = _run(name, model, x, target, **params)
        ref = _zennit_standard(net, x, target, make_composite)
        np.testing.assert_allclose(ours, ref, rtol=1e-4, atol=1e-6 * np.abs(ref).max())
