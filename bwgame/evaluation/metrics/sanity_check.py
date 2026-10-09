"""Model Parameter Randomization Sanity Check (Adebayo et al., 2018).

Implements the cascading randomization test: progressively randomize
model layers from top (output) to bottom (input) and measure how much
the attribution changes compared to the original model.

A method that passes the sanity check should show decreasing similarity
between its original and randomized-model attributions.  A method that
*fails* (e.g. edge detectors) will stay highly similar even with fully
randomized weights.

``_get_randomizable_layers`` and ``_randomize_layer`` define the layer
order and the re-initialisation (also used by
``experiments/hellinger_randomization``); ``iter_cascading_randomization``
runs the cascade and yields the per-step, per-image similarities.
"""

from __future__ import annotations

import copy
import gc
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

logger = logging.getLogger(__name__)


# =========================================================================
# Primitives: layer discovery and weight randomization
# =========================================================================


def _get_randomizable_layers(model: nn.Module) -> List[str]:
    """Return leaf learnable-layer names in **output-to-input** order.

    Walks ``named_modules`` and collects every ``nn.Linear`` and
    ``nn.Conv2d``, which are the two layer types on which Adebayo et al.
    (2018) perform their cascading randomization.  The returned list is
    reversed so that iterating over it randomizes the topmost (closest
    to the output / logits) layer first.
    """
    layers: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            layers.append(name)
    return list(reversed(layers))


def _resolve_submodule(model: nn.Module, qualified_name: str) -> nn.Module:
    mod: nn.Module = model
    for part in qualified_name.split("."):
        mod = getattr(mod, part)
    return mod


def _randomize_layer(model: nn.Module, layer_name: str) -> None:
    """Reinitialize all learnable parameters **inside** ``layer_name``.

    Works both at leaf-layer granularity (``layer4.2.conv3``) and at
    block granularity (``layer4.2`` or ``encoder.layers.encoder_layer_11``).
    ``module.parameters()`` is recursive, so passing a block reinitialises
    every sublayer the block contains -- matching the block-at-a-time
    cascading randomization of the paper.
    """
    module = _resolve_submodule(model, layer_name)
    for param in module.parameters():
        if param.dim() >= 2:
            nn.init.kaiming_normal_(param)
        else:
            nn.init.normal_(param, std=0.01)


# =========================================================================
# Similarity metrics between saliency maps
# =========================================================================

# The four metrics reported in Adebayo et al. (2018), Section 2.2:
#   - Spearman rank correlation with absolute values (ABS)
#   - Spearman rank correlation without absolute values (no-ABS / diverging)
#   - SSIM (absolute value)
#   - Pearson correlation of HOGs (absolute value)
#
# All callables take two numpy arrays with matching shapes and return a
# single scalar float.  ``nan`` is clamped to 0 so downstream averaging
# stays stable when one of the maps is degenerate (constant).
SIMILARITY_METRICS: Dict[str, str] = {
    "spearman_abs":
        "Spearman rank correlation of |maps| (Adebayo 'rank corr ABS')",
    "spearman":
        "Spearman rank correlation of raw diverging maps",
    "ssim":
        "Structural Similarity Index of |maps|",
    "hog_pearson":
        "Pearson correlation of HOG features of |maps|",
}

#: Default similarity metrics evaluated when none specified.
DEFAULT_SIMILARITY_METRICS: Tuple[str, ...] = tuple(SIMILARITY_METRICS.keys())


def _safe_flatten(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _spearman_rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation of two flattened arrays."""
    a_flat = _safe_flatten(a)
    b_flat = _safe_flatten(b)
    if a_flat.std() < 1e-12 or b_flat.std() < 1e-12:
        return 0.0
    corr, _ = stats.spearmanr(a_flat, b_flat)
    return float(corr) if not np.isnan(corr) else 0.0


def _spearman_rank_correlation_abs(a: np.ndarray, b: np.ndarray) -> float:
    return _spearman_rank_correlation(np.abs(a), np.abs(b))


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """SSIM of absolute-value 2D maps (skimage).

    Adebayo uses absolute-value visualization when reporting SSIM, which
    matches our default ``abs`` heatmap convention.  Maps are jointly
    normalised to [0, 1] (per-pair) because SSIM is sensitive to the
    declared data range.
    """
    try:
        from skimage.metrics import structural_similarity as _ssim_fn
    except Exception as e:  # pragma: no cover - import guard
        logger.warning("skimage unavailable; SSIM returns nan (%s)", e)
        return float("nan")

    aa = np.abs(np.asarray(a, dtype=np.float64))
    bb = np.abs(np.asarray(b, dtype=np.float64))
    if aa.ndim == 1:
        # Flattened -- reshape to a square if possible so SSIM has a 2D window
        side = int(np.sqrt(aa.size))
        if side * side == aa.size:
            aa = aa.reshape(side, side)
            bb = bb.reshape(side, side)
        else:
            return float("nan")
    if aa.shape != bb.shape or aa.ndim != 2:
        return float("nan")

    lo = float(min(aa.min(), bb.min()))
    hi = float(max(aa.max(), bb.max()))
    data_range = hi - lo
    if data_range < 1e-12:
        # Both maps constant --> perfectly similar by convention
        return 1.0
    try:
        return float(_ssim_fn(aa, bb, data_range=data_range))
    except Exception as e:
        logger.debug("SSIM failed: %s", e)
        return float("nan")


def _hog_pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between HOG feature vectors of |maps|."""
    try:
        from skimage.feature import hog as _hog
    except Exception as e:  # pragma: no cover - import guard
        logger.warning("skimage.feature.hog unavailable (%s)", e)
        return float("nan")

    aa = np.abs(np.asarray(a, dtype=np.float64))
    bb = np.abs(np.asarray(b, dtype=np.float64))
    if aa.ndim == 1:
        side = int(np.sqrt(aa.size))
        if side * side == aa.size:
            aa = aa.reshape(side, side)
            bb = bb.reshape(side, side)
        else:
            return float("nan")
    if aa.shape != bb.shape or aa.ndim != 2:
        return float("nan")

    # HOG is translation-/contrast-robust; normalise each map to [0, 1]
    # so sparse attributions don't collapse the histogram bins.
    def _unit(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-12:
            return np.zeros_like(x)
        return (x - lo) / (hi - lo)

    # Adapt cell size so there are ~14 cells per side on typical inputs
    # (224x224 -> 16px cells; 28x28 -> 2px cells; 16x16 -> 2px cells).
    side = int(min(aa.shape[0], aa.shape[1]))
    ppc = max(2, side // 14)
    cpb = 2 if side >= 4 * ppc else 1
    try:
        fa = _hog(_unit(aa), pixels_per_cell=(ppc, ppc),
                  cells_per_block=(cpb, cpb), feature_vector=True)
        fb = _hog(_unit(bb), pixels_per_cell=(ppc, ppc),
                  cells_per_block=(cpb, cpb), feature_vector=True)
    except Exception as e:
        logger.debug("HOG failed: %s", e)
        return float("nan")

    if fa.size == 0 or fb.size == 0:
        return float("nan")
    if fa.std() < 1e-12 or fb.std() < 1e-12:
        return 0.0
    r = np.corrcoef(fa, fb)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


_SIMILARITY_FNS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "spearman_abs": _spearman_rank_correlation_abs,
    "spearman":     _spearman_rank_correlation,
    "ssim":         _ssim,
    "hog_pearson":  _hog_pearson,
}


def compute_similarity(name: str, a: np.ndarray, b: np.ndarray) -> float:
    """Dispatch to the named similarity function."""
    try:
        fn = _SIMILARITY_FNS[name]
    except KeyError as e:
        raise KeyError(
            f"Unknown similarity metric {name!r}. "
            f"Available: {list(_SIMILARITY_FNS)}"
        ) from e
    return fn(a, b)


def compute_similarities(
    metrics: Sequence[str],
    original_maps: np.ndarray,
    randomized_maps: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute per-image similarity for every requested metric.

    ``original_maps`` and ``randomized_maps`` are expected to be shape
    ``(N, H, W)`` or ``(N, F)``.  Returns ``{metric: (N,) float64}``.
    """
    n = original_maps.shape[0]
    out: Dict[str, np.ndarray] = {}
    for m in metrics:
        vals = np.empty(n, dtype=np.float64)
        for i in range(n):
            vals[i] = compute_similarity(m, original_maps[i], randomized_maps[i])
        out[m] = vals
    return out


# =========================================================================
# Cascading randomization driver
# =========================================================================


@dataclass
class StepResult:
    """One cascading-randomization step."""

    step_idx: int
    """Zero-based index within the cascade."""

    layer_name: str
    """Fully-qualified name of the layer that was just randomized."""

    num_randomized_layers: int
    """Total number of layers randomized so far (always ``step_idx + 1``)."""

    similarities: Dict[str, np.ndarray]
    """Per-metric array of per-image similarity values, shape ``(N,)``."""


def _coerce_images_cpu(
    images: Union[np.ndarray, torch.Tensor],
) -> torch.Tensor:
    """Ensure images are a float32 **CPU** tensor.

    GPU transfer is deferred to per-batch slicing inside
    ``_run_attribution`` so peak device memory stays bounded.
    """
    if isinstance(images, torch.Tensor):
        return images.detach().cpu().to(dtype=torch.float32)
    return torch.as_tensor(np.asarray(images), dtype=torch.float32)


def _coerce_targets_cpu(
    labels: Union[np.ndarray, torch.Tensor, Sequence[int]],
) -> torch.Tensor:
    """Ensure targets are a long **CPU** tensor."""
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().to(dtype=torch.long)
    return torch.as_tensor(np.asarray(labels), dtype=torch.long)


def _prepare_randomized_model(model: nn.Module, device: torch.device) -> nn.Module:
    """Deep-copy the model and run the DC composite / functional patches.

    The evaluator applies these patches on the trained model before
    computing attributions; the randomized model needs identical
    structural preparation for the same attribution method to run.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rand = copy.deepcopy(model).to(device).eval()

    # Mirror the preparation pipeline used by the eval Evaluator
    try:
        from bwgame.composite_replacer import decompose_composites
        decompose_composites(rand)
    except Exception as e:
        logger.debug("decompose_composites skipped: %s", e)
    try:
        from bwgame.functional_replacer import (
            replace_inline_modules,
            _replace_functional_in_module,
        )
        replace_inline_modules(rand, inplace=True)
        _replace_functional_in_module(rand)
    except Exception as e:
        logger.debug("functional_replacer skipped: %s", e)

    return rand


def _run_attribution(
    model: nn.Module,
    method_name: str,
    device: torch.device,
    images_cpu: torch.Tensor,
    targets_cpu: torch.Tensor,
    method_params: Optional[Dict[str, Any]],
    component: str,
    batch_size: int = 4,
) -> Optional[np.ndarray]:
    """Compute attributions in GPU-safe batches.

    ``images_cpu`` and ``targets_cpu`` are expected to reside on **CPU**;
    each mini-batch is moved to *device*, forward+backward'd, then
    released so peak GPU memory stays proportional to ``batch_size``
    rather than the full dataset.
    """
    from bwgame.xai.registry import MethodRegistry

    n = images_cpu.shape[0]
    all_maps: List[np.ndarray] = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x = images_cpu[start:end].to(device)
        t = targets_cpu[start:end].to(device)

        method = MethodRegistry.instantiate(method_name, model, device)
        if method_params:
            for k, v in method_params.items():
                if hasattr(method, k):
                    setattr(method, k, v)
        try:
            if method.config().supports_batch:
                maps = method.run(x, t, None).maps.get(component)
            else:
                # Methods such as DAVE explain one image per call.
                per = [method.run(x[i:i + 1], t[i:i + 1], None).maps.get(component)
                       for i in range(x.shape[0])]
                maps = None if any(m is None for m in per) else np.concatenate(
                    [np.asarray(m) for m in per], axis=0)
        finally:
            try:
                method.cleanup()
            except Exception:
                pass
            del x, t
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if maps is not None:
            all_maps.append(np.asarray(maps))

    if not all_maps:
        return None
    return np.concatenate(all_maps, axis=0)


def _select_layer_subset(
    layers: Sequence[str],
    *,
    earliest_layer: Optional[str],
    max_steps: Optional[int],
) -> List[str]:
    """Slice / subsample the layer list according to the user's limits.

    ``earliest_layer`` (inclusive) sets the deepest layer to randomize.
    ``max_steps`` caps the total number of steps by an even stride.  The
    final layer in the truncated list is always included so the curve
    bottoms out.
    """
    selected: List[str] = list(layers)

    if earliest_layer is not None:
        if earliest_layer not in selected:
            raise ValueError(
                f"earliest_layer {earliest_layer!r} is not in the layer list. "
                f"First five candidates: {selected[:5]}"
            )
        stop_idx = selected.index(earliest_layer)
        selected = selected[: stop_idx + 1]

    if max_steps is not None and max_steps > 0 and len(selected) > max_steps:
        # Prefer an even stride over the pre-truncation list so we get
        # a representative cascade rather than only the topmost block.
        step = max(1, len(selected) // max_steps)
        sampled = selected[::step][:max_steps]
        if selected[-1] not in sampled:
            sampled = sampled[:-1] + [selected[-1]]
        selected = sampled

    return selected


class _IsolatedRNG:
    """Run an attribution without advancing the torch / numpy random streams.

    The cascade re-initialises the layers from the seeded global stream.
    Methods that draw random samples (SmoothGrad, LIME, KernelSHAP, DAVE)
    would otherwise shift that stream, so the randomised weights at a step
    would depend on which attributions were computed before it.  With the
    streams isolated, the model at every step is a function of the seed and
    the layer order only, and single steps can be computed on their own.
    """

    def __init__(self, device: torch.device):
        self._fork = torch.random.fork_rng(
            devices=[device] if device.type == "cuda" else [])

    def __enter__(self):
        self._np = np.random.get_state()
        self._fork.__enter__()
        return self

    def __exit__(self, *exc):
        self._fork.__exit__(*exc)
        np.random.set_state(self._np)
        return False


def iter_cascading_randomization(
    model: nn.Module,
    method_name: str,
    device: torch.device,
    images: Union[np.ndarray, torch.Tensor],
    labels: Union[np.ndarray, torch.Tensor, Sequence[int]],
    *,
    method_params: Optional[Dict[str, Any]] = None,
    component: str = "input",
    similarity_metrics: Sequence[str] = DEFAULT_SIMILARITY_METRICS,
    earliest_layer: Optional[str] = None,
    max_steps: Optional[int] = None,
    batch_size: int = 4,
    seed: int = 42,
    compute_step: Optional[Callable[[int, str], bool]] = None,
) -> Iterator[StepResult]:
    """Yield one :class:`StepResult` per cascading randomization step.

    ``compute_step(step_idx, layer_name)`` returning False skips the
    attribution of that step (the layers are still randomised, so later
    steps are unaffected); nothing is yielded for skipped steps.

    Parameters
    ----------
    model
        Trained model; will **not** be mutated -- cascading operates on a
        deep copy prepared with the standard composite/functional patches.
    method_name, method_params, component
        Attribution method to compute on both the trained and randomized
        models.  ``component`` is the key into ``method.run(...).maps``.
    images, labels
        Evaluation batch (already preprocessed).  Kept on **CPU**;
        per-batch slices are moved to *device* during attribution to
        keep GPU memory bounded.
    similarity_metrics
        Subset of :data:`SIMILARITY_METRICS`.  Each yielded step contains
        a ``(N,)`` array per metric.
    earliest_layer
        Name of the deepest layer to randomize (cascade stops here,
        inclusive).  The cascade runs over every ``nn.Linear`` /
        ``nn.Conv2d`` layer (:func:`_get_randomizable_layers`).
    max_steps
        Subsample the cascade to at most this many steps (even stride);
        the final layer is always included.
    batch_size
        Number of images per GPU mini-batch during attribution.  Larger
        batches are faster but need more device memory.  Set to 1 for
        large models / many images; the default of 4 fits a 12 GiB GPU
        for most methods.
    seed
        Seeds the rng used by the randomization init (reproducible).

    Yields
    ------
    StepResult
    """
    from bwgame.xai.registry import MethodRegistry  # noqa: F401 - registration side-effect

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_cpu = _coerce_images_cpu(images)
    target_cpu = _coerce_targets_cpu(labels)
    n = int(x_cpu.shape[0])


    # -- Build the layer cascade ----------------------------------------
    layer_order = _get_randomizable_layers(model)
    if not layer_order:
        raise RuntimeError("No randomizable layers found for this model.")

    selected_layers = _select_layer_subset(
        layer_order,
        earliest_layer=earliest_layer,
        max_steps=max_steps,
    )
    logger.info(
        "Sanity check: cascading %d/%d layers (method=%s, metrics=%s)",
        len(selected_layers), len(layer_order), method_name,
        list(similarity_metrics),
    )
    if compute_step is not None and not any(
            compute_step(k, sel) for k, sel in enumerate(selected_layers)):
        return

    # -- Baseline attribution on the trained model (from the seeded state)
    with _IsolatedRNG(device):
        base_maps = _run_attribution(
            model, method_name, device, x_cpu, target_cpu,
            method_params, component, batch_size=batch_size,
        )
    if base_maps is None:
        raise RuntimeError(
            f"Method {method_name!r} returned no {component!r} component on "
            f"the trained model; cannot run sanity check."
        )
    base_maps = np.asarray(base_maps)

    # -- Progressively randomize a deep copy ----------------------------
    rand_model = _prepare_randomized_model(model, device)
    try:
        already_randomized = set()
        # The selected steps subsample the full layer order; each step
        # randomizes every layer between the previous selected layer and
        # the current one, so the cascade covers all layers.
        full_order = list(layer_order)
        step_runs: List[Tuple[str, List[str]]] = []
        last_full_idx = -1
        for sel in selected_layers:
            idx = full_order.index(sel)
            run = full_order[last_full_idx + 1: idx + 1]
            step_runs.append((sel, run))
            last_full_idx = idx

        for step_idx, (sel_layer, run) in enumerate(step_runs):
            for layer_name in run:
                if layer_name not in already_randomized:
                    _randomize_layer(rand_model, layer_name)
                    already_randomized.add(layer_name)

            if compute_step is not None and not compute_step(step_idx, sel_layer):
                continue

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            try:
                with _IsolatedRNG(device):
                    rand_maps = _run_attribution(
                        rand_model, method_name, device, x_cpu, target_cpu,
                        method_params, component, batch_size=batch_size,
                    )
            except Exception as e:
                logger.warning(
                    "Randomized attribution failed at step %d (%s): %s",
                    step_idx, sel_layer, e,
                )
                rand_maps = None

            if rand_maps is None:
                nan_sims = {m: np.full(n, np.nan) for m in similarity_metrics}
                yield StepResult(
                    step_idx=step_idx,
                    layer_name=sel_layer,
                    num_randomized_layers=len(already_randomized),
                    similarities=nan_sims,
                )
                continue

            rand_maps = np.asarray(rand_maps)
            sims = compute_similarities(
                similarity_metrics, base_maps, rand_maps,
            )

            yield StepResult(
                step_idx=step_idx,
                layer_name=sel_layer,
                num_randomized_layers=len(already_randomized),
                similarities=sims,
            )
    finally:
        del rand_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
