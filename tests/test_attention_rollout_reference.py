"""AR, GAE and TIBAV against direct reference computations on a tiny random ViT.

The model is a 12-block torchvision ``VisionTransformer`` (so that block names
``encoder_layer_1`` and ``encoder_layer_10`` coexist), decomposed exactly as the
Evaluator does before handing it to the methods.  The reference attention
maps and their gradients come from an explicit re-implementation of the
torchvision forward pass on the plain (not decomposed) model.

* AR (Abnar & Zuidema 2020): A_bar_l = w I + (1 - w) mean_h A_l, row-normalised,
  J_l = A_bar_l @ J_{l-1}; map J_L[CLS, 1:].
* GAE (Chefer et al., ICCV 2021): A_bar_l = mean_h (dA_l * A_l)^+,
  R_l = R_{l-1} + A_bar_l @ R_{l-1}, R_0 = I; map R_L[CLS, 1:].
* TIBAV (Chefer et al., CVPR 2021): the reference LRP ViT built from the
  torchvision weights reproduces the logits and attention gradients, and the
  map equals the rollout A_bar_l = I + mean_h (dA_l * R(A_l))^+ of the
  relevances R(A_l) it stores, in forward order and without normalisation.

CPU only, fast.
"""

import copy
import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torchvision.models.vision_transformer import VisionTransformer

from bwgame.composite_replacer import decompose_composites
from bwgame.functional_replacer import _replace_functional_in_module, replace_inline_modules
from bwgame.xai import methods  # noqa: F401  (registers the methods)
from bwgame.xai.methods.tibav import chefer_vit_from_torchvision
from bwgame.xai.registry import MethodRegistry

DEVICE = torch.device("cpu")
N_LAYERS, N_HEADS, DIM = 12, 4, 32


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    plain = VisionTransformer(image_size=32, patch_size=8, num_layers=N_LAYERS, num_heads=N_HEADS,
                              hidden_dim=DIM, mlp_dim=64, num_classes=10)
    with torch.no_grad():
        for p in plain.parameters():
            p.add_(0.3 * torch.randn_like(p))
    plain.eval()
    decomposed = copy.deepcopy(plain)
    decompose_composites(decomposed)
    decomposed.eval()
    replace_inline_modules(decomposed, inplace=True)
    _replace_functional_in_module(decomposed)
    return plain, decomposed


@pytest.fixture(scope="module")
def batch():
    g = torch.Generator().manual_seed(1)
    return torch.randn(2, 3, 32, 32, generator=g), torch.tensor([3, 7])


def reference_forward(vit, x):
    """Explicit torchvision ViT forward returning logits and per-block attention (forward order)."""
    n = x.shape[0]
    h = vit._process_input(x)
    h = torch.cat([vit.class_token.expand(n, -1, -1), h], dim=1) + vit.encoder.pos_embedding
    S = h.shape[1]
    attns = []
    for layer in vit.encoder.layers:
        mha = layer.self_attention
        y = layer.ln_1(h)
        q, k, v = F.linear(y, mha.in_proj_weight, mha.in_proj_bias).reshape(
            n, S, 3, N_HEADS, DIM // N_HEADS).permute(2, 0, 3, 1, 4)
        A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(DIM // N_HEADS), dim=-1)
        if A.requires_grad:
            A.retain_grad()
        attns.append(A)
        h = h + mha.out_proj((A @ v).transpose(1, 2).reshape(n, S, DIM))
        h = h + layer.mlp(layer.ln_2(h))
    return vit.heads(vit.encoder.ln(h)[:, 0]), attns


def reference_attention_and_grads(vit, x, target):
    logits, attns = reference_forward(vit, x)
    logits[torch.arange(len(target)), target].sum().backward()
    return logits.detach(), [A.detach() for A in attns], [A.grad.detach() for A in attns]


def rollout_ar(attns, w):
    joint = None
    for A in attns:
        a = A.mean(dim=1)
        a_bar = w * torch.eye(a.shape[-1]) + (1 - w) * a
        a_bar = a_bar / a_bar.sum(-1, keepdim=True)
        joint = a_bar if joint is None else a_bar @ joint
    return joint[:, 0, 1:].reshape(-1, 4, 4).numpy()


def assert_maps_close(got, want, rtol=1e-4):
    assert got.shape == want.shape
    scale = np.abs(want).max(axis=(1, 2), keepdims=True)
    np.testing.assert_array_less(np.abs(got - want) / scale, rtol)


def test_reference_forward_matches_models(models, batch):
    plain, decomposed = models
    x, _ = batch
    with torch.no_grad():
        want = plain(x)
        got_ref, _ = reference_forward(plain, x)
        got_dec = decomposed(x)
    torch.testing.assert_close(got_ref, want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(got_dec, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("w", [0.0, 0.25, 0.5, 0.75])
def test_attention_rollout_matches_reference(models, batch, w):
    plain, decomposed = models
    x, target = batch
    with torch.no_grad():
        _, attns = reference_forward(plain, x)
    want = rollout_ar(attns, w)

    method = MethodRegistry.instantiate("attention_rollout", decomposed, DEVICE)
    method.residual_weight = w
    got = method.attribute(x, target, ["input"])["input"]
    assert_maps_close(got, want)

    # The check is sensitive to the block order: lexicographic name order
    # (0, 1, 10, 11, 2, ...) gives a different map.
    lexicographic = [attns[i] for i in sorted(range(N_LAYERS), key=str)]
    assert np.abs(rollout_ar(lexicographic, w) - want).max() > 1e-3 * np.abs(want).max()


def test_attention_rollout_half_weight_is_abnar_zuidema(models, batch):
    """w = 0.5 is (A + I) / 2, the setting of Abnar & Zuidema's compute_joint_attention."""
    plain, decomposed = models
    x, target = batch
    with torch.no_grad():
        _, attns = reference_forward(plain, x)
    joint = None
    for A in attns:
        a = A.mean(dim=1) + torch.eye(A.shape[-1])
        a = a / a.sum(-1, keepdim=True)
        joint = a if joint is None else a @ joint
    want = joint[:, 0, 1:].reshape(-1, 4, 4).numpy()
    got = MethodRegistry.instantiate("attention_rollout", decomposed, DEVICE).attribute(
        x, target, ["input"])["input"]
    assert_maps_close(got, want)


def test_gae_matches_reference(models, batch):
    plain, decomposed = models
    x, target = batch
    _, attns, grads = reference_attention_and_grads(plain, x, target)
    R = torch.eye(attns[0].shape[-1]).expand(len(x), -1, -1)
    for A, G in zip(attns, grads):
        cam = (G * A).clamp(min=0).mean(dim=1)
        R = R + cam @ R
    want = R[:, 0, 1:].reshape(-1, 4, 4).numpy()

    method = MethodRegistry.instantiate("grad_attention_rollout", decomposed, DEVICE)
    got = method.attribute(x, target, ["input"])["input"]
    assert_maps_close(got, want)


@pytest.mark.parametrize("which", ["plain", "decomposed"])
def test_tibav_reference_model_reproduces_torchvision(models, batch, which):
    plain, decomposed = models
    source = plain if which == "plain" else decomposed
    x, target = batch
    ref = chefer_vit_from_torchvision(source)
    want_logits, _, want_grads = reference_attention_and_grads(plain, x[:1], target[:1])
    logits = ref(x[:1])
    torch.testing.assert_close(logits.detach(), want_logits, rtol=1e-5, atol=1e-5)
    logits[0, int(target[0])].backward()
    for blk, G in zip(ref.blocks, want_grads):
        got = blk.attn.get_attn_gradients()
        assert (got - G).abs().max() <= 1e-4 * G.abs().max()


def test_tibav_is_forward_order_rollout_of_stored_relevances(models, batch):
    _, decomposed = models
    x, target = batch
    method = MethodRegistry.instantiate("tibav", decomposed, DEVICE)
    got = method.attribute(x, target, ["input"])["input"]
    assert got.shape == (2, 4, 4) and np.isfinite(got).all()

    for b in range(len(x)):
        method.attribute(x[b:b + 1], target[b:b + 1])
        C = None
        for blk in method._lrp.model.blocks:
            cam = (blk.attn.get_attn_gradients()[0] * blk.attn.get_attn_cam()[0]).clamp(min=0).mean(0)
            a_bar = torch.eye(cam.shape[-1]) + cam
            C = a_bar if C is None else a_bar @ C
        want = C[0, 1:].reshape(1, 4, 4).detach().numpy()
        assert_maps_close(got[b:b + 1], want)


def test_tibav_cleanup_releases_reference_model(models, batch):
    """A discarded TIBAV instance frees its reference model (no autograd-graph cycle)."""
    import gc
    import weakref
    _, decomposed = models
    x, target = batch
    method = MethodRegistry.instantiate("tibav", decomposed, DEVICE)
    before = method.attribute(x, target)["input"]
    method.cleanup()
    np.testing.assert_array_equal(method.attribute(x, target)["input"], before)
    method.cleanup()
    ref = weakref.ref(method._lrp.model)
    del method
    gc.collect()
    assert ref() is None
