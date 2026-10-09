"""Bridge between bwgame attribution methods and Quantus metrics.

Runs the Quantus metrics of the paper on precomputed saliency maps; the
robustness metrics regenerate maps through the MethodRegistry.
"""

import gc
import logging
import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Metric class name -> quantus class mapping
_QUANTUS_METRIC_MAP = None


def _get_quantus_metric_map():
    global _QUANTUS_METRIC_MAP
    if _QUANTUS_METRIC_MAP is not None:
        return _QUANTUS_METRIC_MAP

    import quantus

    _QUANTUS_METRIC_MAP = {
        # Faithfulness
        "pixel_flipping": quantus.PixelFlipping,
        "selectivity": quantus.Selectivity,
        # Localisation
        "pointing_game": quantus.PointingGame,
        "attribution_localisation": quantus.AttributionLocalisation,
        "top_k_intersection": quantus.TopKIntersection,
        # Robustness
        "avg_sensitivity": quantus.AvgSensitivity,
        "max_sensitivity": quantus.MaxSensitivity,
    }
    return _QUANTUS_METRIC_MAP


def build_quantus_metric(config: Dict[str, Any]):
    """Instantiate a Quantus metric from a config dict.

    Args:
        config: Dict with 'metric_class' and 'params' keys.

    Returns:
        Quantus metric instance.
    """
    metric_map = _get_quantus_metric_map()
    metric_class_name = config["metric_class"]
    if metric_class_name not in metric_map:
        raise ValueError(
            f"Unknown metric class: {metric_class_name}. "
            f"Available: {list(metric_map.keys())}"
        )
    metric_cls = metric_map[metric_class_name]
    params = dict(config.get("params", {}))

    # Strip custom keys that are not Quantus constructor params
    params.pop("auc_percentage", None)

    # Selectivity takes patch_size, not features_in_step: translate
    # features_in_step to patch_size (its square root) if present.
    if metric_class_name == "selectivity" and "features_in_step" in params:
        fis = int(params.pop("features_in_step"))
        ps = int(math.isqrt(fis))
        if ps * ps != fis:
            ps = max(1, ps)
        if "patch_size" not in params:
            params["patch_size"] = ps

    # PixelFlipping: always get raw curves, we compute AUC ourselves
    if metric_class_name == "pixel_flipping":
        params["return_auc_per_sample"] = False

    return metric_cls(**params)


def _compute_truncated_auc(curves: np.ndarray, percentage: float) -> List[float]:
    """Compute AUC up to a given percentage of the perturbation curve.

    Args:
        curves: (N, n_steps) array of prediction scores at each step.
        percentage: Fraction of steps to use (e.g. 0.05 for 5%, 0.20 for 20%).

    Returns:
        List of AUC values, one per sample.
    """
    n_steps = curves.shape[1]
    cutoff = max(2, int(math.ceil(n_steps * percentage)))
    truncated = curves[:, :cutoff]
    # Trapezoidal AUC, normalized to [0, 1]
    x = np.linspace(0, 1, cutoff)
    auc_values = np.trapz(truncated, x, axis=1)
    return auc_values.tolist()


def make_explain_func(
    method_name: str,
    model: nn.Module,
    device: torch.device,
    method_params: Optional[Dict[str, Any]] = None,
    component: str = "input",
    layers: Optional[List[str]] = None,
) -> Callable:
    """Create an explain_func compatible with Quantus sensitivity metrics.

    Returns a callable: explain_func(model, inputs, targets, **kwargs) -> np.ndarray
    """
    from bwgame.xai.registry import MethodRegistry

    # For non-input components, pass layers so only the needed map is generated
    _layers = layers
    if _layers is None and component != "input":
        _layers = [component]

    # Equivariant-averaging (RG+Eq) rows must explain the perturbed inputs
    # with the same wrapper as the reference map (Evaluator._generate_maps);
    # otherwise MaxS / AvgS would compare the averaged reference map with
    # maps of the plain inner method.
    from bwgame.xai.wrappers import _WRAPPER_KEYS
    mp = dict(method_params or {})
    use_equivariant = bool(mp.get("equivariant_averaging", False))
    inner_params = {k: v for k, v in mp.items() if k not in _WRAPPER_KEYS}

    def _make_method():
        method = MethodRegistry.instantiate(method_name, model, device)
        for k, v in inner_params.items():
            if hasattr(method, k):
                setattr(method, k, v)
        return method

    def _wrapped_maps(x_i, t_i):
        from bwgame.xai.wrappers import equivariant_average
        return equivariant_average(
            method_name, model, device, x_i, t_i, _layers,
            n_samples=int(mp.get("equivariant_n_samples", 50)),
            max_angle=float(mp.get("equivariant_max_angle", 20)),
            max_shift_frac=float(mp.get("equivariant_max_shift_frac", 0.1)),
            flip_prob=float(mp.get("equivariant_flip_prob", 0.5)),
            method_params=inner_params,
        )

    # Always loop per-image inside the explain_func. Quantus sensitivity
    # metrics call us with a batch of nr_samples=10 perturbations per image;
    # running those as one forward/backward pass multiplies peak GPU memory
    # ~10x and OOMs on ViT / ResNet-50 even at outer BS=1. Per-image looping
    # with torch.cuda.empty_cache() between iterations keeps memory bounded
    # at a modest speed cost.
    def explain_func(model, inputs, targets, **kwargs):
        x = torch.tensor(inputs, dtype=torch.float32, device=device)
        t = torch.tensor(targets, dtype=torch.long, device=device)
        per_image_maps = []
        for i in range(x.shape[0]):
            if use_equivariant:
                maps_i = _wrapped_maps(x[i:i+1], t[i:i+1])
                m = maps_i.get(component)
                if m is None:
                    m = next(iter(maps_i.values()))
                m = np.asarray(m)
            else:
                method = _make_method()
                res = method.run(x[i:i+1], t[i:i+1], _layers)
                m = res.maps.get(component)
                if m is None:
                    m = next(iter(res.maps.values()))
                method.cleanup()
                del method, res
            per_image_maps.append(m[0] if m.ndim >= 3 else m)
            # Methods that build a model copy with hooks (e.g. the reference
            # TIBAV/GAE implementations) leave reference cycles; collect them
            # so that the per-image loop runs in bounded memory.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        maps = np.stack(per_image_maps)

        # Upsample to input spatial resolution if needed (nearest-neighbor)
        H_in, W_in = inputs.shape[-2], inputs.shape[-1]
        if maps.ndim >= 2 and (maps.shape[-2] != H_in or maps.shape[-1] != W_in):
            import cv2
            upsampled = []
            for m in maps:
                upsampled.append(cv2.resize(
                    m.astype(np.float32), (W_in, H_in),
                    interpolation=cv2.INTER_NEAREST))
            maps = np.stack(upsampled)

        return maps

    return explain_func


_SENSITIVITY_METRICS = ("max_sensitivity", "avg_sensitivity")

# MaxSensitivity and AvgSensitivity (Yeh et al. 2019) are the maximum and the
# mean of the same per-sample sensitivities.  When both are requested with the
# same parameters they are computed in one pass (each perturbation is
# explained once); the scores equal those of the two separate Quantus metrics.
JOINT_SENSITIVITY = True


_JOINT_SENSITIVITY_CLS = None


def _joint_sensitivity_class():
    """MaxSensitivity subclass that also records the per-instance mean (AvgSensitivity)."""
    global _JOINT_SENSITIVITY_CLS
    if _JOINT_SENSITIVITY_CLS is not None:
        return _JOINT_SENSITIVITY_CLS
    import quantus

    class _JointSensitivity(quantus.MaxSensitivity):
        def evaluate_batch(self, model, x_batch, y_batch, a_batch, **kwargs):
            batch_size = x_batch.shape[0]
            a_flat = a_batch.reshape(batch_size, -1)
            similarities = np.zeros((batch_size, self.nr_samples)) * np.nan
            for step_id in range(self.nr_samples):
                x_perturbed = self.perturb_func(
                    arr=x_batch.reshape(batch_size, -1),
                    indices=np.tile(np.arange(0, x_batch[0].size), (batch_size, 1)),
                ).reshape(*x_batch.shape)
                changed = self.changed_prediction_indices_func(model, x_batch, x_perturbed)
                a_perturbed = self.explain_batch(model, x_perturbed, y_batch).reshape(batch_size, -1)
                numerator = self.norm_numerator(a=self.similarity_func(a=a_flat, b=a_perturbed))
                similarities[:, step_id] = numerator / self.norm_denominator(a=a_flat)
                similarities[changed, step_id] = np.nan
            # Same NaN handling as the aggregation of MaxSensitivity (nanmax <-> nanmean).
            mean_func = np.nanmean if self.max_func is np.nanmax else np.mean
            self._avg_scores.extend(mean_func(similarities, axis=1).tolist())
            return self.max_func(similarities, axis=1)

    _JOINT_SENSITIVITY_CLS = _JointSensitivity
    return _JOINT_SENSITIVITY_CLS


def _joint_sensitivity_pair(metric_configs):
    """(max_cfg, avg_cfg) if both sensitivities are requested with equal params."""
    by_class = {c.get("metric_class"): c for c in metric_configs}
    mx, av = by_class.get("max_sensitivity"), by_class.get("avg_sensitivity")
    if JOINT_SENSITIVITY and mx is not None and av is not None \
            and mx.get("params", {}) == av.get("params", {}):
        return mx, av
    return None


def _seed_from_image(x: np.ndarray) -> None:
    """Seed numpy / torch / random deterministically from an image's content."""
    import hashlib
    import random
    data = np.ascontiguousarray(np.asarray(x, dtype=np.float32).round(4)).tobytes()
    seed = int.from_bytes(hashlib.sha256(data).digest()[:4], "little")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_metric_minibatched(metric_builder, kwargs, cfg, mini_batch_size=8, aux_out=None):
    """Run a metric in mini-batches to avoid CUDA OOM.

    Splits x_batch/y_batch/a_batch/s_batch into chunks, runs each,
    and concatenates per-sample scores.

    For MaxSensitivity / AvgSensitivity: per-image all-zero attributions are
    skipped (inserted as NaN) — Quantus otherwise raises on them. This happens
    for methods like LIME whose surrogate collapses to zero coefficients on
    some inputs.
    """
    import gc

    N = kwargs["x_batch"].shape[0]
    all_scores = [np.nan] * N

    is_sensitivity = cfg.get("metric_class") in _SENSITIVITY_METRICS
    if is_sensitivity:
        # One image per call, with the noise seeded from the image content:
        # every method sees the same perturbations of a given image (paired
        # comparisons), independent of the image order, of how the images
        # are split over jobs, and of random numbers the method itself draws.
        mini_batch_size = 1

    for start in range(0, N, mini_batch_size):
        end = min(start + mini_batch_size, N)
        chunk_kwargs = dict(kwargs)
        chunk_kwargs["x_batch"] = kwargs["x_batch"][start:end]
        chunk_kwargs["y_batch"] = kwargs["y_batch"][start:end]
        chunk_kwargs["a_batch"] = kwargs["a_batch"][start:end]
        if "s_batch" in chunk_kwargs and chunk_kwargs["s_batch"] is not None:
            chunk_kwargs["s_batch"] = kwargs["s_batch"][start:end]

        # Filter out per-image all-zero attributions for sensitivity metrics.
        # Quantus asserts a_batch has at least one non-zero element; a chunk
        # consisting of only zero-attr images would kill the whole call.
        local_idx_keep = list(range(end - start))
        if is_sensitivity:
            a_flat = chunk_kwargs["a_batch"].reshape(end - start, -1)
            nonzero_mask = (np.abs(a_flat).sum(axis=1) > 0)
            if not nonzero_mask.all():
                local_idx_keep = [i for i, ok in enumerate(nonzero_mask) if ok]
                if not local_idx_keep:
                    continue  # entire chunk is zero-attr — leave scores as NaN
                chunk_kwargs["x_batch"] = chunk_kwargs["x_batch"][local_idx_keep]
                chunk_kwargs["y_batch"] = chunk_kwargs["y_batch"][local_idx_keep]
                chunk_kwargs["a_batch"] = chunk_kwargs["a_batch"][local_idx_keep]
                if "s_batch" in chunk_kwargs and chunk_kwargs["s_batch"] is not None:
                    chunk_kwargs["s_batch"] = chunk_kwargs["s_batch"][local_idx_keep]

        # Fresh metric instance per chunk (some metrics have internal state)
        metric = metric_builder()
        if is_sensitivity:
            _seed_from_image(chunk_kwargs["x_batch"][0])

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        chunk_scores = metric(**chunk_kwargs)

        if isinstance(chunk_scores, list):
            vals = chunk_scores
        elif isinstance(chunk_scores, np.ndarray):
            vals = chunk_scores.tolist()
        else:
            vals = [chunk_scores]

        # Scatter back into the full-length score list (NaN where skipped)
        for v, li in zip(vals, local_idx_keep):
            all_scores[start + li] = v
        if aux_out is not None:
            for v, li in zip(getattr(metric, "_avg_scores", []), local_idx_keep):
                aux_out[start + li] = v

    return all_scores


def _aggregate_scores(scores, metric_name, cfg, results):
    """Aggregate raw metric scores into results dict.

    Besides the mean (``<name>``), std (``<name>_std``) and coverage
    (``<name>_coverage`` = fraction of images with a finite score), this
    stores the per-image values that the mean is computed from as
    ``<name>_per_image`` (float array, NaN where an image has no score).
    For curve metrics (pixel flipping, selectivity) the per-image value is
    the AUC that the tables report, and ``<name>_curve_len`` records the
    number of points of the underlying perturbation curve.
    """
    metric_class = cfg.get("metric_class", "")
    auc_pct = cfg.get("params", {}).get("auc_percentage")

    per_image = None
    if metric_class == "pixel_flipping" and auc_pct is not None:
        curves = np.array(scores, dtype=float)
        auc_scores = np.asarray(_compute_truncated_auc(curves, auc_pct), dtype=float)
        per_image = auc_scores
        results[f"{metric_name}_curve_len"] = int(curves.shape[1])
        results[f"{metric_name}_auc_points"] = int(max(2, int(math.ceil(curves.shape[1] * auc_pct))))
        results[f"{metric_name}_all"] = scores
    elif isinstance(scores, list):
        if scores and isinstance(scores[0], (list, np.ndarray)):
            curves = np.array(scores, dtype=float)
            x = np.linspace(0, 1, curves.shape[1])
            per_image = np.asarray(np.trapz(curves, x, axis=1), dtype=float)
            results[f"{metric_name}_curve_len"] = int(curves.shape[1])
        else:
            per_image = np.asarray(scores, dtype=float)
        results[f"{metric_name}_all"] = scores
    elif isinstance(scores, dict):
        results[metric_name] = scores
        return
    else:
        results[metric_name] = float(scores)
        return

    n_total = len(per_image)
    n_used = int(np.isfinite(per_image).sum())
    results[metric_name] = float(np.nanmean(per_image)) if n_used > 0 else float("nan")
    results[f"{metric_name}_std"] = float(np.nanstd(per_image, ddof=1)) if n_used > 1 else float("nan")
    # Coverage: fraction of images actually contributing — < 1.0 when
    # some images had an all-zero / failed attribution and were skipped.
    results[f"{metric_name}_coverage"] = n_used / n_total if n_total else float("nan")
    results[f"{metric_name}_per_image"] = per_image


def is_per_image_key(key: str) -> bool:
    """True for result keys that hold per-image arrays (not JSON scalars)."""
    return key.endswith("_all") or key.endswith("_per_image")


def evaluate_saliency_maps(
    model: nn.Module,
    images: np.ndarray,
    labels: np.ndarray,
    saliency_maps: np.ndarray,
    metric_configs: List[Dict[str, Any]],
    masks: Optional[np.ndarray] = None,
    explain_func: Optional[Callable] = None,
    softmax: bool = True,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate precomputed saliency maps against Quantus metrics.

    Args:
        model: PyTorch model.
        images: (N, C, H, W) numpy array.
        labels: (N,) numpy array of class indices.
        saliency_maps: (N, H, W) numpy array of attribution maps.
        metric_configs: List of metric config dicts.
        masks: Optional (N, H, W) binary segmentation masks (for localisation).
        explain_func: Callable for sensitivity metrics (generates maps on the fly).
        softmax: Whether to apply softmax to model outputs.
        device: Device string.

    Returns:
        Dict mapping metric_name -> score (float or dict of detailed results).
    """
    import gc

    # Images whose attribution map could not be computed are stored as NaN
    # maps by the evaluator.  Evaluate the metrics on the remaining images
    # only and report NaN (not a fake zero-map score) for the failed ones.
    n_total = len(saliency_maps)
    valid = np.isfinite(np.asarray(saliency_maps, dtype=float).reshape(n_total, -1)).all(axis=1)
    if n_total and not valid.all():
        n_bad = int((~valid).sum())
        logger.warning(
            "%d/%d attribution maps are NaN (failed generation); metrics are "
            "computed on the remaining %d images and those images get NaN.",
            n_bad, n_total, n_total - n_bad,
        )
        if valid.any():
            sub = evaluate_saliency_maps(
                model=model, images=images[valid], labels=labels[valid],
                saliency_maps=saliency_maps[valid], metric_configs=metric_configs,
                masks=masks[valid] if masks is not None else None,
                explain_func=explain_func, softmax=softmax, device=device,
            )
        else:
            sub = {}
        results = {}
        for cfg in metric_configs:
            name = cfg["name"]
            full = np.full(n_total, np.nan)
            if f"{name}_per_image" in sub:
                full[valid] = sub[f"{name}_per_image"]
            n_used = int(np.isfinite(full).sum())
            results[name] = float(np.nanmean(full)) if n_used > 0 else float("nan")
            results[f"{name}_std"] = float(np.nanstd(full, ddof=1)) if n_used > 1 else float("nan")
            results[f"{name}_coverage"] = n_used / n_total
            results[f"{name}_per_image"] = full
            for extra in ("_curve_len", "_auc_points"):
                if f"{name}{extra}" in sub:
                    results[f"{name}{extra}"] = sub[f"{name}{extra}"]
            curves = np.asarray(sub.get(f"{name}_all", []), dtype=float) if f"{name}_all" in sub else None
            if curves is not None and curves.ndim == 2:
                full_curves = np.full((n_total, curves.shape[1]), np.nan)
                full_curves[valid] = curves
                results[f"{name}_all"] = full_curves
        return results

    results = {}

    # Upsample attribution maps to input spatial resolution if needed.
    # TIBAV, GAE and attention rollout produce (N, H_patch, W_patch) maps
    # on the ViT patch grid (14x14 for ViT-B/16); Quantus asserts that
    # attribution dims are a (consecutive) sub-shape of inputs and otherwise
    # aborts every metric for the method. Upsample with nearest-neighbour so
    # the per-patch mass is preserved.
    if (images.ndim == 4 and saliency_maps.ndim == 3 and
            saliency_maps.shape[-2:] != images.shape[-2:]):
        import cv2
        H_in, W_in = images.shape[-2], images.shape[-1]
        upsampled = np.stack([
            cv2.resize(m.astype(np.float32), (W_in, H_in),
                       interpolation=cv2.INTER_NEAREST)
            for m in saliency_maps
        ])
        saliency_maps = upsampled

    # Total features for pixel flipping divisibility
    n_features = int(np.prod(images.shape[1:])) if images.ndim == 4 else None

    joint = _joint_sensitivity_pair(metric_configs)

    for cfg in metric_configs:
        metric_name = cfg["name"]
        category = cfg.get("category", "unknown")

        # AvgSensitivity is produced together with MaxSensitivity.
        if joint is not None and cfg is joint[1]:
            continue

        try:
            # features_in_step must divide the number of input features.
            if n_features is not None and "features_in_step" in cfg.get("params", {}):
                cfg = dict(cfg)
                cfg["params"] = dict(cfg["params"])
                step = cfg["params"]["features_in_step"]
                if n_features % step != 0:
                    # Find nearest divisor <= step
                    new_step = step
                    while new_step > 1 and n_features % new_step != 0:
                        new_step -= 1
                    cfg["params"]["features_in_step"] = new_step
                    logger.info("Auto-adjusted features_in_step %d->%d for %s (n_features=%d)",
                                step, new_step, metric_name, n_features)
                else:
                    cfg["params"]["features_in_step"] = step

            # Localisation metrics need segmentation masks
            is_localisation = category == "localisation"
            needs_explain_func = category == "robustness"

            # Faithfulness metrics should use logits (not softmax probabilities)
            # to avoid saturation effects that compress score differences.
            use_softmax = softmax
            if category == "faithfulness":
                use_softmax = False

            kwargs = {
                "model": model,
                "x_batch": images,
                "y_batch": labels,
                "a_batch": saliency_maps,
                "softmax": use_softmax,
                "device": device,
            }

            if is_localisation and masks is not None:
                s = masks
                # Convert to binary (N, 1, H, W) as Quantus expects
                if s.ndim == 4 and s.shape[-1] == 3:
                    # (N, H, W, 3) RGB -> binary (N, H, W)
                    s = s.any(axis=-1).astype(np.float32)
                if s.ndim == 3:
                    # (N, H, W) -> (N, 1, H, W)
                    s = s[:, np.newaxis, :, :]
                kwargs["s_batch"] = s

            if needs_explain_func and explain_func is not None:
                kwargs["explain_func"] = explain_func

            # Build metric and try chunked processing.
            # Metrics are evaluated in fixed-size chunks (localisation metrics of
            # Quantus 0.6 are only reliable for moderate batch sizes).
            frozen_cfg = cfg

            is_joint = joint is not None and cfg is joint[0]

            def _build_metric(c=frozen_cfg):
                metric = build_quantus_metric(c)
                if is_joint:
                    metric.__class__ = _joint_sensitivity_class()
                    metric._avg_scores = []
                return metric

            avg_scores = [np.nan] * len(images) if is_joint else None
            try:
                scores = _run_metric_minibatched(
                    _build_metric, kwargs, cfg, mini_batch_size=50, aux_out=avg_scores)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err:
                if "CUDA out of memory" not in str(oom_err) and not isinstance(oom_err, torch.cuda.OutOfMemoryError):
                    raise
                logger.warning("Metric %s OOM at batch=50, retrying with batch=4...", metric_name)
                gc.collect()
                torch.cuda.empty_cache()
                try:
                    scores = _run_metric_minibatched(_build_metric, kwargs, cfg, mini_batch_size=4,
                                                     aux_out=avg_scores)
                except (torch.cuda.OutOfMemoryError, RuntimeError) as oom_err2:
                    if "CUDA out of memory" not in str(oom_err2) and not isinstance(oom_err2, torch.cuda.OutOfMemoryError):
                        raise
                    logger.warning("Metric %s OOM at batch=4, retrying with batch=1...", metric_name)
                    gc.collect()
                    torch.cuda.empty_cache()
                    scores = _run_metric_minibatched(_build_metric, kwargs, cfg, mini_batch_size=1,
                                                     aux_out=avg_scores)

            _aggregate_scores(scores, metric_name, cfg, results)
            if is_joint:
                _aggregate_scores(avg_scores, joint[1]["name"], joint[1], results)
            logger.debug("Metric %s: %.4f", metric_name, results.get(metric_name, float("nan")))

        except Exception as e:
            logger.warning("Metric %s failed: %s", metric_name, e)
            results[metric_name] = float("nan")
            results[f"{metric_name}_coverage"] = 0.0
            results[f"{metric_name}_per_image"] = np.full(len(saliency_maps), np.nan)

    return results
