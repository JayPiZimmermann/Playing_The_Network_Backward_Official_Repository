"""Baseline attribution methods agree with their reference implementations.

Gradient (Simonyan et al. 2014) vs captum Saliency; SmoothGrad (Smilkov et
al. 2017) vs a direct implementation of the PAIR ``saliency`` algorithm;
Integrated Gradients vs captum; DeepLIFT vs captum.attr.DeepLift; LIME vs the
``lime_image`` algorithm; KernelSHAP efficiency; CAMs vs pytorch-grad-cam.
All models are small and randomly initialised (CPU, no downloads) and are
decomposed exactly as by the evaluator.  Tests with pretrained ImageNet
weights are marked ``slow``.
"""
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck

import bwgame.xai  # noqa: F401  (registers the methods)
import bwgame.xai.methods.captum_methods as cm
from bwgame.xai.registry import MethodRegistry

CPU = torch.device("cpu")


def decompose(model):
    """The in-place decomposition applied by ``Evaluator.__init__``."""
    from bwgame.composite_replacer import decompose_composites
    from bwgame.functional_replacer import replace_inline_modules, _replace_functional_in_module
    decompose_composites(model)
    model.eval()
    replace_inline_modules(model, inplace=True)
    _replace_functional_in_module(model)
    return model


def make(name, model, **params):
    m = MethodRegistry.instantiate(name, model, CPU)
    for k, v in params.items():
        assert hasattr(m, k), k
        setattr(m, k, v)
    return m


class TinyVGG(nn.Module):
    """VGG-style CNN: ReLU -> MaxPool, flatten via torch.flatten, ReLU classifier."""

    def __init__(self, n_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.avgpool = nn.AdaptiveAvgPool2d(4)
        self.classifier = nn.Sequential(nn.Linear(16 * 4 * 4, 32), nn.ReLU(inplace=True),
                                        nn.Linear(32, n_classes))

    def forward(self, x):
        return self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))


def tiny_vgg(seed=0):
    torch.manual_seed(seed)
    return TinyVGG().eval()


@pytest.fixture
def batch():
    g = torch.Generator().manual_seed(1)
    return torch.randn(2, 3, 16, 16, generator=g), torch.tensor([1, 3])


# ---------------------------------------------------------------------------
# Channel reduction, Gradient, SmoothGrad, Integrated Gradients
# ---------------------------------------------------------------------------

def test_reduce_channels():
    a = torch.tensor([[[[1.0]], [[-3.0]], [[2.0]]]])
    assert cm.reduce_channels(a, "sum").item() == 0.0
    assert cm.reduce_channels(a, "sum_abs").item() == 6.0
    assert cm.reduce_channels(a, "max_abs").item() == 3.0
    with pytest.raises(ValueError):
        cm.reduce_channels(a, "mean")


def test_gradient_max_abs_is_simonyan_saliency(batch):
    from captum.attr import Saliency
    x, y = batch
    model = tiny_vgg()
    ref = Saliency(model).attribute(x.clone(), target=y, abs=True).amax(dim=1)
    got = make("raw_gradient", decompose(tiny_vgg()), channel_reduction="max_abs").attribute(x, y)
    np.testing.assert_allclose(got["input"], ref.numpy(), rtol=1e-5, atol=1e-7)


def test_gradient_default_is_signed_channel_sum(batch):
    x, y = batch
    model = tiny_vgg()
    xg = x.clone().requires_grad_(True)
    model(xg).gather(1, y[:, None]).sum().backward()
    got = make("raw_gradient", model).attribute(x, y)["input"]
    np.testing.assert_allclose(got, xg.grad.sum(1).numpy(), rtol=1e-5, atol=1e-7)


def test_smoothgrad_stdev_spread_matches_reference_algorithm(batch):
    """SmoothGrad with sigma = spread * (max - min) per image, mean of raw gradients
    (PAIR saliency ``GetSmoothedMask(magnitude=False)``)."""
    x, y = batch
    model = tiny_vgg()
    n, spread = 6, 0.15
    torch.manual_seed(7)
    got = make("smooth_grad", model, nt_samples=n, stdev_spread=spread, use_abs=False,
               channel_reduction="sum_abs").attribute(x, y)["input"]

    torch.manual_seed(7)
    sigma = spread * (x.flatten(1).amax(1) - x.flatten(1).amin(1))
    total = torch.zeros_like(x)
    for _ in range(n):
        xn = (x + torch.randn_like(x) * sigma.view(-1, 1, 1, 1)).requires_grad_(True)
        model(xn).gather(1, y[:, None]).sum().backward()
        total += xn.grad
    ref = (total / n).abs().sum(dim=1)
    np.testing.assert_allclose(got, ref.numpy(), rtol=1e-5, atol=1e-7)


def test_smoothgrad_default_noise_is_absolute(batch):
    x, _ = batch
    m = make("smooth_grad", tiny_vgg())
    assert m.stdev_spread is None and m._noise_std(x) == m.stdevs
    m.stdev_spread = 0.2
    np.testing.assert_allclose(m._noise_std(x).flatten().numpy(),
                               (0.2 * (x.flatten(1).amax(1) - x.flatten(1).amin(1))).numpy())


@pytest.mark.parametrize("baseline", ["zero", "black"])
def test_integrated_gradients_matches_captum(batch, baseline):
    from captum.attr import IntegratedGradients
    x, y = batch
    base = cm.input_baseline(x, baseline)
    ref = IntegratedGradients(tiny_vgg()).attribute(
        x.clone(), baselines=base, target=y, n_steps=16, method="riemann_middle").sum(dim=1)
    got = make("integrated_gradients", decompose(tiny_vgg()), n_steps=16,
               baseline=baseline).attribute(x, y)["input"]
    np.testing.assert_allclose(got, ref.detach().numpy(), rtol=1e-4, atol=1e-6)


def test_black_baseline_is_black_image():
    x = torch.zeros(1, 3, 2, 2)
    black = cm.input_baseline(x, "black")
    rgb = black * torch.tensor(cm.IMAGENET_STD).view(1, 3, 1, 1) + torch.tensor(
        cm.IMAGENET_MEAN).view(1, 3, 1, 1)
    assert torch.allclose(rgb, torch.zeros_like(rgb), atol=1e-6)


# ---------------------------------------------------------------------------
# DeepLIFT
# ---------------------------------------------------------------------------

def test_deeplift_rescale_matches_captum_on_vgg_style_cnn(batch):
    from captum.attr import DeepLift
    x, y = batch
    ref = DeepLift(tiny_vgg()).attribute(x.clone(), target=y).sum(dim=1)
    got = make("deeplift", decompose(tiny_vgg()), rule="rescale").attribute(x, y)["input"]
    np.testing.assert_allclose(got, ref.detach().numpy(), rtol=1e-4, atol=1e-6)


def _block_forward(self, x):
    identity = x if self.downsample is None else self.downsample(x)
    out = self.r1(self.bn1(self.conv1(x)))
    if isinstance(self, Bottleneck):
        out = self.r2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
    else:
        out = self.bn2(self.conv2(out))
    return self.r3(out + identity)


def _one_relu_per_application(model):
    for m in model.modules():
        if isinstance(m, (BasicBlock, Bottleneck)):
            m.r1, m.r2, m.r3 = nn.ReLU(), nn.ReLU(), nn.ReLU()
            m.forward = types.MethodType(_block_forward, m)
        if isinstance(m, nn.ReLU):
            m.inplace = False
    return model


def _random_resnet(arch, seed=0):
    torch.manual_seed(seed)
    model = getattr(torchvision.models, arch)(weights=None, num_classes=5).eval()
    with torch.no_grad():  # non-trivial BatchNorm statistics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-0.2, 0.2)
                m.running_var.uniform_(0.5, 1.5)
    return model


@pytest.mark.parametrize("arch", ["resnet18", "resnet50"])
def test_deeplift_rescale_on_resnet(arch):
    from captum.attr import DeepLift
    from bwgame.xai.methods._deeplift_network import deeplift_network
    g = torch.Generator().manual_seed(2)
    x, y = torch.randn(1, 3, 64, 64, generator=g), torch.tensor([2])

    dec = decompose(_random_resnet(arch))
    net = deeplift_network(dec)
    with torch.no_grad():
        assert torch.allclose(net(x), dec(x), atol=1e-5)
        assert torch.allclose(net(x), _random_resnet(arch)(x), atol=1e-5)

    ref_net = _one_relu_per_application(_random_resnet(arch))
    ref = DeepLift(ref_net).attribute(x.clone(), target=y).sum(dim=1)
    got = make("deeplift", dec, rule="rescale").attribute(x, y)["input"]
    np.testing.assert_allclose(got, ref.detach().numpy(), rtol=1e-4, atol=1e-6)
    # summation-to-delta (the reference activations are spatially constant in
    # the stem's max-pool windows for the zero reference)
    with torch.no_grad():
        delta = (dec(x) - dec(torch.zeros_like(x)))[0, y].item()
    assert got.sum() == pytest.approx(delta, rel=1e-3, abs=1e-5)


def test_deeplift_rescale_layer_maps(batch):
    from captum.attr import LayerDeepLift
    x, y = batch
    got = make("deeplift", decompose(tiny_vgg()), rule="rescale").attribute(x, y, ["features.5"])
    ref_model = tiny_vgg()
    ref = LayerDeepLift(ref_model, ref_model.features[5]).attribute(x.clone(), target=y).sum(dim=1)
    assert got["features.5"].shape == (2, 8, 8)
    np.testing.assert_allclose(got["features.5"], ref.detach().numpy(), rtol=1e-4, atol=1e-6)


def test_deeplift_default_rule_is_zennit_epsilon(batch):
    x, y = batch
    m = make("deeplift", tiny_vgg())
    assert m.rule == "epsilon" and m.baseline == "zero"
    assert m.attribute(x, y)["input"].shape == (2, 16, 16)


def test_deeplift_rescale_rejects_attention_models():
    torch.manual_seed(0)
    vit = torchvision.models.VisionTransformer(
        image_size=32, patch_size=16, num_layers=1, num_heads=2, hidden_dim=16, mlp_dim=32,
        num_classes=5).eval()
    vit = decompose(vit)
    with pytest.raises(NotImplementedError, match="DeepLIFT"):
        make("deeplift", vit, rule="rescale").attribute(torch.randn(1, 3, 32, 32), torch.tensor([0]))


# ---------------------------------------------------------------------------
# LIME / KernelSHAP
# ---------------------------------------------------------------------------

def test_fill_values():
    x = torch.arange(2 * 4 * 4, dtype=torch.float32).view(1, 2, 4, 4)
    seg = torch.zeros(4, 4, dtype=torch.long)
    seg[:, 2:] = 1
    mask = seg.expand(1, 2, 4, 4)
    assert cm.fill_values(x, mask, "zero") == 0.0
    sm = cm.fill_values(x, mask, "segment_mean")
    for c in range(2):
        for s in range(2):
            assert torch.allclose(sm[0, c][seg == s], x[0, c][seg == s].mean())
    im = cm.fill_values(x, mask, "image_mean")
    assert torch.allclose(im[0, :, 0, 0], x[0].mean(dim=(1, 2)))


def test_binary_cosine_kernel_is_lime_image_kernel():
    from sklearn.metrics import pairwise_distances
    kernel = cm._binary_cosine_kernel(0.25)
    z = np.array([[1, 1, 1, 1, 1], [1, 0, 1, 0, 0], [0, 0, 0, 0, 0]])
    d = pairwise_distances(z, z[:1], metric="cosine").ravel()
    lime_w = np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
    ours = [kernel(None, None, torch.tensor(row)[None]) for row in z]
    np.testing.assert_allclose(ours, lime_w, rtol=1e-6)


def test_lime_sampler_starts_with_unperturbed_input():
    gen = cm._lime_image_sampler(torch.zeros(1), num_interp_features=7)
    first = next(gen)
    assert first.shape == (1, 7) and bool((first == 1).all())
    assert set(torch.cat([next(gen) for _ in range(50)]).unique().tolist()) <= {0, 1}


def test_lime_matches_lime_image_algorithm(monkeypatch):
    """With fixed binary samples, LIME (ridge, kernel 0.25 on the binary
    representation, segment-mean fill) equals the lime_image computation:
    Ridge(alpha=1) on the samples with weights sqrt(exp(-d^2 / w^2)),
    d = cosine distance to the all-ones vector, targets = class probabilities."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import pairwise_distances
    torch.manual_seed(0)
    model = tiny_vgg()
    x, y = torch.randn(1, 3, 16, 16), torch.tensor([2])
    n = 60
    feature_mask = cm._get_feature_mask(x, patch_size=4)       # 16 superpixels
    m_feat = int(feature_mask.max()) + 1
    rs = np.random.RandomState(0)
    z = rs.randint(0, 2, n * m_feat).reshape(n, m_feat)
    z[0] = 1

    def samples(original_inp, **kwargs):
        for row in z:
            yield torch.from_numpy(row).long()[None]

    monkeypatch.setattr(cm, "_lime_image_sampler", samples)
    monkeypatch.setattr(cm, "_get_feature_mask", lambda inp, patch_size=16: feature_mask)
    got = make("lime", model, n_samples=n, surrogate="ridge", kernel_width=0.25,
               kernel_distance="binary", fill="segment_mean").attribute(x, y)["input"][0]

    fill = cm.fill_values(x, feature_mask, "segment_mean")
    seg = feature_mask[0, 0]
    keep = torch.from_numpy(z).float()[:, seg][:, None]
    with torch.no_grad():
        probs = F.softmax(model(x * keep + fill * (1 - keep)), dim=-1)[:, 2].numpy()
    d = pairwise_distances(z, z[:1], metric="cosine").ravel()
    w = np.sqrt(np.exp(-(d ** 2) / 0.25 ** 2))
    coef = Ridge(alpha=1.0).fit(z, probs, sample_weight=w).coef_
    ref = 3 * coef[seg.numpy()]      # the map sums the per-channel attribution
    np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-6)


def test_lime_defaults_unchanged():
    m = make("lime", tiny_vgg())
    assert (m.fill, m.kernel_distance, m.kernel_width, m.surrogate) == ("zero", "input", None, "lasso")


@pytest.mark.parametrize("fill", ["zero", "image_mean", "segment_mean"])
def test_kernel_shap_efficiency_with_fill(fill):
    """KernelSHAP values sum to f(x) - f(all superpixels removed)."""
    torch.manual_seed(0)
    model = tiny_vgg()
    x, y = torch.randn(1, 3, 32, 32), torch.tensor([1])      # 2 x 2 grid of 16-px patches
    m = make("kernel_shap", model, n_samples="auto", fill=fill)
    mask = cm._get_feature_mask(x)
    assert m._num_samples(mask) == 2 * (int(mask.max()) + 1) + 2048
    m.n_samples = 300
    torch.manual_seed(0)
    got = m.attribute(x, y)["input"][0]
    removed = torch.zeros_like(x) + cm.fill_values(x, mask, fill)
    with torch.no_grad():
        p = F.softmax(model(torch.cat([x, removed])), dim=-1)[:, 1]
    # each 16x16 patch is one superpixel: its value appears on 256 pixels x 3 channels
    total = got.sum() / (3 * 256)
    assert total == pytest.approx((p[0] - p[1]).item(), abs=1e-3)


# ---------------------------------------------------------------------------
# CAMs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,cls_name", [("gradcam", "GradCAM"), ("gradcam_pp", "GradCAMPlusPlus"),
                                           ("layercam", "LayerCAM")])
def test_cam_target_layer_matches_pytorch_grad_cam(batch, name, cls_name):
    import pytorch_grad_cam
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    x, y = batch
    got = make(name, decompose(tiny_vgg()), target_layer_name="features.6").attribute(x, y)
    assert set(got) == {"input"} and got["input"].shape == (2, 16, 16)
    fresh = tiny_vgg()
    cam = getattr(pytorch_grad_cam, cls_name)(model=fresh, target_layers=[fresh.features[6]])
    try:
        ref = cam(input_tensor=x.clone(), targets=[ClassifierOutputTarget(int(t)) for t in y])
    finally:
        cam.activations_and_grads.release()
    np.testing.assert_allclose(got["input"], ref, rtol=1e-4, atol=1e-5)


def test_cam_default_is_per_layer(batch):
    x, y = batch
    m = make("layercam", tiny_vgg())
    assert m.target_layer_name is None
    out = m.attribute(x, y, ["features.5"])
    assert set(out) == {"features.5"} and out["features.5"].shape == (2, 8, 8)


# ---------------------------------------------------------------------------
# Pretrained ImageNet models
# ---------------------------------------------------------------------------

def _imagenet_input():
    g = torch.Generator().manual_seed(0)
    return torch.randn(1, 3, 224, 224, generator=g) * 0.5, torch.tensor([207])


@pytest.mark.slow
def test_deeplift_vgg16_matches_captum_on_torchvision_model():
    from captum.attr import DeepLift
    from bwgame.evaluation.run_eval import load_model
    x, y = _imagenet_input()
    ref = DeepLift(load_model("vgg16", CPU)).attribute(x.clone(), target=y).sum(dim=1)
    got = make("deeplift", decompose(load_model("vgg16", CPU)), rule="rescale").attribute(x, y)
    np.testing.assert_allclose(got["input"], ref.detach().numpy(), rtol=1e-4, atol=1e-6)


@pytest.mark.slow
def test_deeplift_resnet50_network_copy_and_completeness():
    from bwgame.evaluation.run_eval import load_model
    from bwgame.xai.methods._deeplift_network import deeplift_network
    x, y = _imagenet_input()
    dec = decompose(load_model("resnet50", CPU))
    with torch.no_grad():
        assert torch.allclose(deeplift_network(dec)(x), load_model("resnet50", CPU)(x), atol=1e-4)
        delta = (dec(x) - dec(torch.zeros_like(x)))[0, y].item()
    got = make("deeplift", dec, rule="rescale").attribute(x, y)["input"]
    assert got.sum() == pytest.approx(delta, rel=1e-3)


@pytest.mark.slow
@pytest.mark.parametrize("arch,layer", [("vgg16", "features.29"), ("resnet50", "layer4")])
def test_gradcam_matches_pytorch_grad_cam_on_torchvision_model(arch, layer):
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from bwgame.evaluation.run_eval import load_model
    from bwgame.xai.layer_hooks import _find_module
    x, y = _imagenet_input()
    got = make("gradcam", decompose(load_model(arch, CPU)), target_layer_name=layer).attribute(x, y)
    fresh = load_model(arch, CPU)
    cam = GradCAM(model=fresh, target_layers=[_find_module(fresh, layer)])
    try:
        ref = cam(input_tensor=x.clone(), targets=[ClassifierOutputTarget(int(y))])
    finally:
        cam.activations_and_grads.release()
    np.testing.assert_allclose(got["input"], ref, rtol=1e-4, atol=1e-5)
