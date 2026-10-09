"""DAVE: Distribution-Aware Attribution via ViT Gradient DEcomposition.

Implements the DAVE method (Wróbel et al., 2026) which decomposes the input
gradient into the *effective transformation* (the direct, locally equivariant
action of the model on the input) and the *operator variation* (sensitivity
of input-dependent operators to perturbations).  DAVE discards the operator
variation and averages the effective transformation over spatial transforms
and Gaussian noise to produce stable, high-resolution attribution maps.

The key is the **conditioned forward pass** (Algorithm 1 in the paper):
for each layer, input-dependent operators (attention weights, LayerNorm
statistics, GELU/SiLU gating) are computed from a *detached* copy of the
activation, so gradients only flow through the linear part of each layer.

Three components run in sequence:
  1. Effective transformation — conditioned forward + backward
  2. Equivariant aggregation — average over spatial transforms τ⁻¹ ∘ W_L ∘ τ
  3. Low-pass filtering — average over Gaussian noise ε

The final attribution is:  DAVE(X) = W̃_L^eq(X) ⊙ X

Reference:
    Wróbel, Gairola, Tabor, Schiele, Zielinski & Rymarczyk,
    "DAVE: Distribution-aware Attribution via ViT Gradient Decomposition",
    arXiv:2602.06613, 2026.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..base import AttributionMethod
from ..postprocess import tensor_to_numpy
from ..registry import MethodRegistry
from ..types import MethodCategory, MethodConfig

logger = logging.getLogger(__name__)


# =========================================================================
# Spatial transform utilities (rotation + translation)
# =========================================================================

def _sample_transform(
    max_angle: float = 20.0,
    max_shift_frac: float = 0.1,
    flip_prob: float = 0.5,
) -> Tuple[Tensor, Tensor]:
    """Sample a random spatial transform and its inverse.

    Following DAVE Appendix D.2, the transform group includes:
      - Random horizontal flip with probability ``flip_prob``
      - In-plane rotation uniformly sampled from (-max_angle, max_angle)
      - Translation with offsets up to ``max_shift_frac`` of image extent

    Returns:
        (theta, theta_inv) — both shape (2, 3) affine matrices.
    """
    # Horizontal flip (self-inverse)
    flip = -1.0 if torch.rand(1).item() < flip_prob else 1.0

    # Rotation
    angle = (torch.rand(1).item() * 2 - 1) * max_angle
    rad = math.radians(angle)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    # Translation in normalised [-1, 1] coordinates
    # max_shift_frac=0.1 → offsets up to 0.1 of image extent = 0.2 in grid coords
    tx = (torch.rand(1).item() * 2 - 1) * max_shift_frac * 2
    ty = (torch.rand(1).item() * 2 - 1) * max_shift_frac * 2

    # Forward: flip → rotate → translate
    # M = T @ R @ F
    theta = torch.tensor([
        [flip * cos_a, -sin_a, tx],
        [flip * sin_a,  cos_a, ty],
    ], dtype=torch.float32)

    # Inverse: translate⁻¹ → rotate⁻¹ → flip⁻¹
    cos_neg, sin_neg = math.cos(-rad), math.sin(-rad)
    # R⁻¹ @ T⁻¹: first undo translation, then undo rotation
    tx_inv = -(cos_neg * tx + (-sin_neg) * ty)
    ty_inv = -(sin_neg * tx + cos_neg * ty)
    theta_inv = torch.tensor([
        [flip * cos_neg, flip * (-sin_neg), flip * tx_inv],
        [sin_neg,        cos_neg,           ty_inv],
    ], dtype=torch.float32)

    return theta, theta_inv


def _apply_affine(x: Tensor, theta: Tensor) -> Tensor:
    """Apply a 2×3 affine transform to a batch of images."""
    theta_batch = theta.unsqueeze(0).expand(x.size(0), -1, -1).to(x.device)
    grid = F.affine_grid(theta_batch, x.size(), align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode='reflection')


# =========================================================================
# Conditioned forward pass hooks (Algorithm 1 in the paper)
# =========================================================================

class _ConditionedForwardHookManager:
    """Install forward hooks that implement the conditioned forward pass.

    For each layer type, the hook replaces the output with a version where
    input-dependent operators are computed from a detached copy ``c = x.detach()``
    while the linear part uses ``x`` (with gradients).

    This means:
      - LayerNorm:  output = (x − μ(c)) / σ(c) · γ + β   (μ, σ from detached c)
      - GELU:       output = x · Φ(c)                     (gate from detached c)
      - SiLU:       output = x · sigmoid(c)                (gate from detached c)
      - Softmax:    output = softmax(c)                    (fully detached)
      - Attention:  Q, K from detached c; V from x         (only V has gradient)

    The result is that ``∇_x output`` captures only the effective transformation
    L(X) (Eq. 3 in the paper), not the operator variation.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._hooks: list = []

    def install(self) -> '_ConditionedForwardHookManager':
        """Register conditioned forward hooks on all relevant modules."""
        for name, module in self.model.named_modules():
            if isinstance(module, nn.LayerNorm):
                self._hooks.append(module.register_forward_hook(
                    self._make_layernorm_hook(name)))
            elif isinstance(module, nn.GELU):
                self._hooks.append(module.register_forward_hook(
                    self._make_gelu_hook(name)))
            elif isinstance(module, nn.SiLU):
                self._hooks.append(module.register_forward_hook(
                    self._make_silu_hook(name)))
            elif isinstance(module, nn.Softmax):
                self._hooks.append(module.register_forward_hook(
                    self._make_softmax_hook(name)))
        return self

    def remove(self):
        """Remove all hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # --- LayerNorm: use detached statistics, gradient-bearing input ---
    def _make_layernorm_hook(self, name: str):
        def hook(mod, inp, out):
            x = inp[0]
            c = x.detach()
            mean = c.mean(dim=-1, keepdim=True)
            var = c.var(dim=-1, keepdim=True, unbiased=False)
            x_norm = (x - mean) / (var + mod.eps).sqrt()
            if mod.weight is not None:
                return x_norm * mod.weight + mod.bias
            return x_norm
        return hook

    # --- GELU: gate from detached input, linear part from real input ---
    # GELU(x) = x · Φ(x) where Φ is the standard normal CDF.
    # Conditioned: output = x · Φ(c)
    def _make_gelu_hook(self, name: str):
        def hook(mod, inp, out):
            x = inp[0]
            c = x.detach()
            gate = torch.special.ndtr(c)  # Φ(c)
            return x * gate
        return hook

    # --- SiLU: gate from detached input ---
    # SiLU(x) = x · σ(x). Conditioned: output = x · σ(c)
    def _make_silu_hook(self, name: str):
        def hook(mod, inp, out):
            x = inp[0]
            c = x.detach()
            return x * torch.sigmoid(c)
        return hook

    # --- Softmax: fully detached (attention weights are constants) ---
    def _make_softmax_hook(self, name: str):
        def hook(mod, inp, out):
            return out.detach()
        return hook


# =========================================================================
# DAVE method
# =========================================================================

@MethodRegistry.register
class DAVE(AttributionMethod):
    """DAVE: Distribution-Aware Attribution via ViT Gradient DEcomposition.

    Computes the effective transformation of the model (discarding operator
    variation) and averages it over spatial transforms and Gaussian noise.

    Parameters (instance attributes; set them after construction):
        n_samples: Number of (transform, noise) samples to average.
        max_angle: Maximum rotation angle in degrees.
        max_shift_frac: Maximum translation as a fraction of the image extent.
        flip_prob: Probability of a horizontal flip per sample.
        noise_t_max: Maximum interpolation weight of the variance-preserving
            Gaussian noise (0 disables the noise).
        multiply_input: Multiply the averaged effective transformation by X.
        batch_size: Samples per forward pass (speed only).
    """

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="dave",
            category=MethodCategory.GRADIENT,
            class_specific=True,
            supports_batch=False,
            requires_grad=True,
            per_layer=False,
            description=(
                "DAVE: Distribution-aware Attribution via ViT Gradient "
                "DEcomposition. Equivariant effective transformation averaging."
            ),
            reference="Wróbel et al. 2026",
        )

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.n_samples = 50
        self.max_angle = 20.0
        self.max_shift_frac = 0.1   # fraction of image extent
        self.flip_prob = 0.5
        self.noise_t_max = 0.5      # max interpolation parameter for noise
        self.batch_size = 1         # samples per forward pass (>1 = batched)
        self.multiply_input = True  # DAVE(X) = W̃ ⊙ X (paper default)

    def _build_batch(
        self, x: Tensor, n: int,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Build a batch of n transformed+noised copies and their inverse transforms."""
        _, C, H, W = x.shape
        x_batch = []
        theta_invs = []

        for _ in range(n):
            theta, theta_inv = _sample_transform(
                self.max_angle, self.max_shift_frac, self.flip_prob)
            theta_invs.append(theta_inv)

            # Variance-preserving noise (Appendix D.2)
            t_noise = torch.rand(1).item() * self.noise_t_max
            scale = 1.0 - t_noise
            noise_scale = math.sqrt(1.0 - scale * scale)
            eps = torch.randn_like(x)
            x_noisy = scale * x + noise_scale * eps

            x_t = _apply_affine(x_noisy, theta)
            x_batch.append(x_t)

        return torch.cat(x_batch, dim=0), theta_invs  # (n, C, H, W)

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        self.model.eval()
        B, C, H, W = x.shape
        assert B == 1, "DAVE processes one image at a time"

        # Replace functional ops (F.softmax, torch.matmul, etc.) with
        # nn.Module equivalents so the conditioned forward hooks can
        # intercept them.  Without this, attention softmax is a functional
        # call inside nn.MultiheadAttention and never gets detached.
        from bwgame.functional_replacer import replace_functional_with_modules
        replace_functional_with_modules(self.model, inplace=True)

        bs = max(1, self.batch_size)

        # Accumulate inverse-transformed effective transformations
        W_eq = torch.zeros(1, C, H, W, device=x.device)
        n_done = 0

        while n_done < self.n_samples:
            n_batch = min(bs, self.n_samples - n_done)

            # Build batch of transformed inputs
            x_batch, theta_invs = self._build_batch(x, n_batch)

            # Conditioned forward pass
            hook_mgr = _ConditionedForwardHookManager(self.model)
            hook_mgr.install()

            try:
                x_input = x_batch.detach().clone().requires_grad_(True)
                output = self.model(x_input)  # (n_batch, num_classes)

                # Backward for target class
                one_hot = torch.zeros_like(output)
                for i in range(n_batch):
                    one_hot[i, target[0]] = 1.0

                output.backward(gradient=one_hot)
                W_batch = x_input.grad.detach()  # (n_batch, C, H, W)

            finally:
                hook_mgr.remove()

            # Inverse-transform each sample and accumulate
            for i in range(n_batch):
                W_i = W_batch[i:i+1]  # (1, C, H, W)
                W_i_inv = _apply_affine(W_i, theta_invs[i])
                W_eq = W_eq + W_i_inv

            n_done += n_batch

        # Average
        W_eq = W_eq / self.n_samples

        # Final attribution: optionally multiply by input (paper Eq. 9)
        if self.multiply_input:
            dave_map = W_eq * x  # DAVE(X) = W̃_L^eq(X) ⊙ X
        else:
            dave_map = W_eq  # raw effective transformation
        spatial = dave_map.sum(dim=1)  # (1, H, W)

        return {"input": tensor_to_numpy(spatial)}
