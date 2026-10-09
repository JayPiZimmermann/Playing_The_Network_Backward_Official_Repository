"""Smoke + sanity tests for the Chefer 2021a Transformer-LRP (TIBAV) method.

Covers:
- the reference LRP ViT built from the torchvision weights reproduces the
  torchvision ViT-B/16 logits (plain and decomposed model)
- forward+backward runs without error on ViT-B/16
- output has the expected (1, 14, 14) spatial shape for a 224x224 input
- output is non-zero and finite
- output differs materially from GAE (grad_attention_rollout), i.e.
  is not a trivial duplicate of Chefer 2021b
- TIBAV is class-specific (different target yields different output)

These tests assume the torchvision ViT_B_16 pretrained weights are locally
available via ``torchvision.models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)``.
If no CUDA is available the tests fall back to CPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bwgame.composite_replacer import decompose_composites
from bwgame.xai import methods  # noqa: F401  -- triggers registration
from bwgame.xai.registry import MethodRegistry


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def vit_model():
    try:
        from torchvision.models import ViT_B_16_Weights, vit_b_16
    except ImportError:  # pragma: no cover
        pytest.skip("torchvision unavailable")
    try:
        m = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"could not load ViT-B/16 weights: {exc}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m.eval().to(device)
    decompose_composites(m)
    return m, device


def _random_batch(device: str, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(1, 3, 224, 224, generator=g).to(device)


def test_reference_vit_reproduces_torchvision_logits(vit_model):
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    from bwgame.xai.methods.tibav import chefer_vit_from_torchvision

    m, device = vit_model
    plain = vit_b_16(weights=ViT_B_16_Weights.DEFAULT).eval().to(device)
    x = _random_batch(device, seed=3)
    with torch.no_grad():
        want = plain(x)
        got_dec = m(x)
    got_ref = chefer_vit_from_torchvision(m).to(device)(x).detach()
    got_ref_plain = chefer_vit_from_torchvision(plain).to(device)(x).detach()
    for got in (got_dec, got_ref, got_ref_plain):
        assert (got - want).abs().max().item() < 1e-4


def test_tibav_runs_and_shape(vit_model):
    m, device = vit_model
    x = _random_batch(device, seed=42)
    target = torch.tensor([207], device=device)  # arbitrary ImageNet class
    method = MethodRegistry.instantiate("tibav", m, device)
    out = method.attribute(x, target, layers=["input"])
    assert "input" in out, "TIBAV must emit an 'input' layer map"
    m_input = out["input"]
    assert m_input.shape == (1, 14, 14), (
        f"ViT-B/16 patches give 14x14; got {m_input.shape}"
    )
    assert np.isfinite(m_input).all(), "output must be finite"
    assert float(np.abs(m_input).max()) > 0.0, "output must be non-zero"


def test_tibav_differs_from_gae(vit_model):
    """Chefer 2021a (TIBAV) weights the LRP relevance of the attention maps;
    Chefer 2021b (GAE) weights the raw attention. The two methods must not
    collapse to the same numerical result."""
    m, device = vit_model
    x = _random_batch(device, seed=1)
    target = torch.tensor([207], device=device)

    gae_method = MethodRegistry.instantiate("grad_attention_rollout", m, device)
    tibav_method = MethodRegistry.instantiate("tibav", m, device)

    gae = gae_method.attribute(x, target, layers=["input"])["input"].flatten()
    tibav = tibav_method.attribute(x, target, layers=["input"])["input"].flatten()

    max_abs_diff = float(np.abs(gae - tibav).max())
    # Correlation sanity: not equal and not a mere rescaling (both directions
    # of correlation are acceptable; what we want to rule out is |corr| ~= 1).
    corr = float(np.corrcoef(gae, tibav)[0, 1])

    assert max_abs_diff > 1e-6, (
        "TIBAV and GAE produced effectively identical maps "
        f"(max |Δ|={max_abs_diff:.2e}); TIBAV is not distinguished from GAE."
    )
    assert abs(corr) < 0.99, (
        f"TIBAV correlates {corr:+.3f} with GAE; within rescaling error, "
        "so the LRP relevance does not enter the map."
    )


def test_tibav_class_specific(vit_model):
    """Changing the target class should change the attribution map: TIBAV
    is advertised as class-specific."""
    m, device = vit_model
    x = _random_batch(device, seed=2)
    method = MethodRegistry.instantiate("tibav", m, device)

    t_a = torch.tensor([207], device=device)
    t_b = torch.tensor([400], device=device)

    a = method.attribute(x, t_a, layers=["input"])["input"].flatten()
    b = method.attribute(x, t_b, layers=["input"])["input"].flatten()

    assert float(np.abs(a - b).max()) > 1e-6, (
        "TIBAV output did not change when the target class changed; the "
        "backward relevance is not class-dependent."
    )
