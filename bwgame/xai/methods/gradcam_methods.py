"""GradCAM-family attribution methods.

Three CAM methods (GradCAM, GradCAM++, LayerCAM).  With
``target_layer_name`` set, a method runs the corresponding ``pytorch-grad-cam``
class (Gildenblat et al., github.com/jacobgil/pytorch-grad-cam) on that module
and returns the library's output: the CAM min-max scaled and bilinearly
upsampled to the input resolution, under the key ``"input"``.  Otherwise the
CAM is computed per layer with direct hooks at the layer's resolution.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import cv2

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, LayerCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils import image as _pgc_image

from ..base import AttributionMethod
from ..layer_hooks import _find_module
from ..model_utils import get_target_layers_for_cam, vit_reshape_transform
from ..postprocess import tensor_to_numpy
from ..registry import MethodRegistry
from ..types import MethodCategory, MethodConfig


def _scale_cam_nearest(cam, target_size=None):
    """Like pytorch_grad_cam's scale_cam_image but uses nearest-neighbor."""
    result = []
    for img in cam:
        img = img - np.min(img)
        img = img / (1e-7 + np.max(img))
        if target_size is not None:
            img = cv2.resize(np.float32(img), target_size,
                             interpolation=cv2.INTER_NEAREST)
        result.append(img)
    return np.float32(result)


from bwgame.operations.matmul import DCMatMul
from bwgame.operations.mul import DCMul

logger = logging.getLogger(__name__)


def _build_cam_targets(target: Tensor):
    """pytorch_grad_cam targets: the score of each image's target class."""
    return [ClassifierOutputTarget(t.item()) for t in target]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _run_cam_per_layer(
    cam_class,
    model: nn.Module,
    x: Tensor,
    target: Tensor,
    layers: Optional[List[str]] = None,
) -> Dict[str, np.ndarray]:
    """Run a CAM method on each target layer individually.

    When *layers* is provided, runs CAM on those specific layers using a
    direct implementation (no library upscaling — raw-resolution output).
    Otherwise uses the predefined target layers via pytorch_grad_cam with
    nearest-neighbour upsampling.

    Returns:
        Dictionary mapping layer names to numpy arrays of shape ``(B, H, W)``.
    """
    if layers:
        return _direct_cam(cam_class, model, x, target, layers)

    target_layers, is_vit = get_target_layers_for_cam(model)
    if not target_layers:
        raise RuntimeError(
            f"{cam_class.__name__}: no target layers found. "
            f"Model may not be a supported architecture."
        )
    targets_list = _build_cam_targets(target)
    reshape_fn = vit_reshape_transform if is_vit else None

    results: Dict[str, np.ndarray] = {}

    for layer_name, layer_module in target_layers:
        cam = None
        try:
            cam = cam_class(
                model=model,
                target_layers=[layer_module],
                reshape_transform=reshape_fn,
            )
            # Temporarily swap upsampling function for nearest-neighbor
            _orig_scale = _pgc_image.scale_cam_image
            _pgc_image.scale_cam_image = _scale_cam_nearest
            try:
                grayscale_cam = cam(input_tensor=x, targets=targets_list)
            finally:
                _pgc_image.scale_cam_image = _orig_scale
            results[layer_name] = grayscale_cam
        except Exception as e:
            logger.warning(
                "%s failed on layer %s: %s", cam_class.__name__, layer_name, e,
            )
        finally:
            if cam is not None:
                try:
                    cam.activations_and_grads.release()
                except Exception:
                    pass
                del cam

    if not results:
        raise RuntimeError(
            f"{cam_class.__name__}: all target layers failed"
        )
    return results


def _to_spatial(t: Tensor) -> Tensor:
    """Reshape a tensor to (B, C, H, W) for CAM computation.

    Handles:
    - (B, C, H, W): return as-is
    - (B, seq_len, D): drop CLS, reshape to (B, D, grid, grid)
    - (B, D): unsqueeze to (B, D, 1, 1)
    """
    if t.ndim == 4:
        return t
    if t.ndim == 3:
        B, S, D = t.shape
        n = S - 1
        grid = int(n ** 0.5)
        if grid * grid == n:
            return t[:, 1:].reshape(B, grid, grid, D).permute(0, 3, 1, 2)
        grid = int(S ** 0.5)
        if grid * grid == S:
            return t.reshape(B, grid, grid, D).permute(0, 3, 1, 2)
        return t.permute(0, 2, 1).unsqueeze(2)
    if t.ndim == 2:
        return t.unsqueeze(-1).unsqueeze(-1)
    return t


def _gradcam_formula(act: Tensor, grad: Tensor) -> np.ndarray:
    """Standard GradCAM: ReLU(Σ_c w_c * A_c), w_c = GAP(∂y/∂A_c)."""
    spatial_act = _to_spatial(act)    # (B, C, H, W)
    spatial_grad = _to_spatial(grad)  # (B, C, H, W)
    weights = spatial_grad.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
    cam = (weights * spatial_act).sum(dim=1)  # (B, H, W)
    cam = torch.relu(cam)
    return tensor_to_numpy(cam)


def _gradcampp_formula(act: Tensor, grad: Tensor) -> np.ndarray:
    """GradCAM++: pixel-level weighting of positive gradients."""
    spatial_act = _to_spatial(act)
    spatial_grad = _to_spatial(grad)
    grad2 = spatial_grad ** 2
    grad3 = spatial_grad ** 3
    sum_act = spatial_act.sum(dim=(2, 3), keepdim=True)
    alpha = grad2 / (2 * grad2 + sum_act * grad3 + 1e-8)
    alpha = torch.where(spatial_grad > 0, alpha, torch.zeros_like(alpha))
    weights = (alpha * torch.relu(spatial_grad)).sum(dim=(2, 3), keepdim=True)
    cam = (weights * spatial_act).sum(dim=1)
    cam = torch.relu(cam)
    return tensor_to_numpy(cam)


def _layercam_formula(act: Tensor, grad: Tensor) -> np.ndarray:
    """LayerCAM: element-wise positive gradient weighting."""
    spatial_act = _to_spatial(act)
    spatial_grad = _to_spatial(grad)
    cam = (torch.relu(spatial_grad) * spatial_act).sum(dim=1)
    cam = torch.relu(cam)
    return tensor_to_numpy(cam)


_CAM_FORMULAS = {
    GradCAM: _gradcam_formula,
    GradCAMPlusPlus: _gradcampp_formula,
    LayerCAM: _layercam_formula,
}


def _direct_cam(
    cam_class,
    model: nn.Module,
    x: Tensor,
    target: Tensor,
    layers: List[str],
) -> Dict[str, np.ndarray]:
    """Compute CAM on specific layers using direct hooks (no library upscaling).

    Returns the raw-resolution maps of the layers.

    For dual-input modules (DCMatMul, DCMul), captures both inputs and
    applies the CAM formula to each, producing ``layer:A`` and ``layer:B``
    keyed results.
    """
    formula = _CAM_FORMULAS.get(cam_class, _gradcam_formula)

    # Resolve layer modules
    layer_modules = []
    for name in layers:
        mod = _find_module(model, name)
        if mod is not None:
            layer_modules.append((name, mod))

    if not layer_modules:
        raise RuntimeError(f"No valid layers found for CAM: {layers}")

    # Capture output activations + gradients for each layer
    hooks = []
    captured = {}  # key -> Tensor (key may include :A/:B suffix)
    dual_layers = set()

    for name, mod in layer_modules:
        is_dual = isinstance(mod, (DCMatMul, DCMul))
        if is_dual:
            dual_layers.add(name)

        def _make_hook(layer_name, _is_dual):
            def _hook(module, inp, out):
                if _is_dual and isinstance(inp, tuple) and len(inp) >= 2:
                    # Dual-input: capture both inputs
                    for i, t in enumerate(inp[:2]):
                        if isinstance(t, Tensor) and t.requires_grad:
                            t.retain_grad()
                        suffix = 'A' if i == 0 else 'B'
                        captured[f"{layer_name}:{suffix}"] = t
                else:
                    # Single-input: capture output (standard CAM)
                    t = out if isinstance(out, Tensor) else out[0]
                    if t.requires_grad:
                        t.retain_grad()
                    captured[layer_name] = t
            return _hook
        hooks.append(mod.register_forward_hook(_make_hook(name, is_dual)))

    try:
        x_input = x.detach().clone().requires_grad_(True)
        output = model(x_input)
        grad = torch.zeros_like(output)
        for i in range(target.shape[0]):
            grad[i, target[i]] = 1.0
        output.backward(gradient=grad)
    finally:
        for h in hooks:
            h.remove()

    results: Dict[str, np.ndarray] = {}
    for name, act in captured.items():
        if isinstance(act, Tensor) and act.grad is not None:
            try:
                results[name] = formula(act.detach(), act.grad.detach())
            except Exception as e:
                logger.warning("CAM failed on %s: %s", name, e)

    if not results:
        raise RuntimeError(f"{cam_class.__name__}: all layers failed")
    return results


def _library_cam(
    cam_class,
    model: nn.Module,
    x: Tensor,
    target: Tensor,
    layer_name: str,
) -> np.ndarray:
    """Run a ``pytorch-grad-cam`` class on the module named ``layer_name``.

    Returns the library output of shape ``(B, H, W)``: per image the CAM of
    the target class, min-max scaled to [0, 1] and resized to the input size
    with bilinear interpolation (``pytorch_grad_cam.utils.image.scale_cam_image``).
    """
    module = _find_module(model, layer_name)
    if module is None:
        raise ValueError(f"{cam_class.__name__}: no module named {layer_name!r}")
    _, is_vit = get_target_layers_for_cam(model)
    targets = _build_cam_targets(target)
    cam = cam_class(model=model, target_layers=[module],
                    reshape_transform=vit_reshape_transform if is_vit else None)
    try:
        return cam(input_tensor=x, targets=targets)
    finally:
        cam.activations_and_grads.release()


# ---------------------------------------------------------------------------
# CAM methods
# ---------------------------------------------------------------------------

class _CAMMethod(AttributionMethod):
    """Shared implementation of the CAM methods (``_cam_class`` per subclass).

    ``target_layer_name``: module for the ``pytorch-grad-cam`` output under
    ``"input"`` (e.g. ``features.29``, the rectified last convolution of
    VGG-16, or ``layer4`` of ResNet-50).  ``layers``: per-layer CAMs at the
    layers' own resolution.  Without either, the predefined target layers of
    :func:`get_target_layers_for_cam` are used.
    """

    _cam_class = GradCAM

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        self.target_layer_name: Optional[str] = None

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        if self.target_layer_name is None:
            return _run_cam_per_layer(self._cam_class, self.model, x, target, layers)
        results = {"input": _library_cam(
            self._cam_class, self.model, x, target, self.target_layer_name)}
        if layers:
            results.update(_run_cam_per_layer(self._cam_class, self.model, x, target, layers))
        return results


@MethodRegistry.register
class GradCAMMethod(_CAMMethod):
    """Grad-CAM via ``pytorch_grad_cam.GradCAM``."""

    _cam_class = GradCAM

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="gradcam",
            category=MethodCategory.CAM,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=True,
            description="Gradient-weighted class activation mapping.",
            reference="Selvaraju et al. 2017",
        )


@MethodRegistry.register
class GradCAMPPMethod(_CAMMethod):
    """Grad-CAM++ via ``pytorch_grad_cam.GradCAMPlusPlus``."""

    _cam_class = GradCAMPlusPlus

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="gradcam_pp",
            category=MethodCategory.CAM,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=True,
            description="Improved GradCAM with pixel-level weighting.",
            reference="Chattopadhyay et al. 2018",
        )


@MethodRegistry.register
class LayerCAMMethod(_CAMMethod):
    """LayerCAM via ``pytorch_grad_cam.LayerCAM``."""

    _cam_class = LayerCAM

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="layercam",
            category=MethodCategory.CAM,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=True,
            description="Layer-wise gradient activation mapping.",
            reference="Jiang et al. 2021",
        )
