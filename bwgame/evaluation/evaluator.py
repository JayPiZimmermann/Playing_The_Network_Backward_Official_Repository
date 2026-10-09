"""Evaluator: attribution maps of one method configuration and their metrics.

Processes one method at a time across all images (method-first loop) to
minimize memory usage.
"""

import json
import logging
import math as _math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None

from .config_loader import load_metric_configs
from .dataset import EvalCollator
from .quantus_bridge import evaluate_saliency_maps, is_per_image_key, make_explain_func

logger = logging.getLogger(__name__)



def _makedirs(path) -> None:
    """mkdir -p that tolerates concurrent creation on network filesystems."""
    import time
    for k in range(5):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return
        except FileExistsError:
            if Path(path).is_dir():
                return
            time.sleep(0.2 * (k + 1))
    Path(path).mkdir(parents=True, exist_ok=True)

class Evaluator:
    """Evaluation engine for XAI attribution methods.

    Runs a method-first loop: for each method configuration, generates
    attribution maps for all images, then evaluates with Quantus metrics.

    Supports:
    - Per-layer maps (pass ``layers``)
    - Caching of saliency maps as .npz files
    - Per-image persistence: ``<output_dir>/per_image/<label>.npz`` holds the
      image ids, the per-image score of every metric (for pixel flipping /
      selectivity the per-image AUC the tables report), per-image wall-clock
      time and a per-image map-failure flag.
    """

    def __init__(
        self,
        model: nn.Module,
        dataset: Dataset,
        device: torch.device,
        output_dir: Path,
        metric_configs: Optional[List[Dict]] = None,
        batch_size: int = 1,
        softmax: bool = True,
        seed: int = 42,
        save_heatmaps: bool = True,
        cache_maps: bool = True,
        warmup: int = 0,
    ):
        """
        Args (in addition to the obvious ones):
            save_heatmaps: Write one PNG per image and map under ``heatmaps/``.
            cache_maps: Cache generated maps (and the input images) in
                ``.saliency_cache/<label>.npz`` for metric recomputation.
            warmup: Number of untimed warm-up calls of the method on the
                first batch before per-image timing starts (0 = none).
        """
        # Decompose MultiheadAttention into explicit Q/K/V + DCMatMul +
        # Softmax and replace functional ops with modules so hook-based
        # methods can intercept attention internals.
        from bwgame.composite_replacer import decompose_composites
        from bwgame.functional_replacer import (
            replace_inline_modules, _replace_functional_in_module,
        )
        decompose_composites(model)
        model.eval()
        replace_inline_modules(model, inplace=True)
        _replace_functional_in_module(model)

        self.model = model.to(device)
        self.dataset = dataset
        self.device = device
        self.output_dir = Path(output_dir)
        _makedirs(self.output_dir)
        self.batch_size = batch_size
        self.softmax = softmax
        self.seed = seed
        self.save_heatmaps = save_heatmaps
        self.cache_maps = cache_maps
        self.warmup = int(warmup)

        if metric_configs is None:
            metric_configs = load_metric_configs()
        self.metric_configs = metric_configs

        # Dataset positions evaluated by run_method / _generate_maps.
        self._indices = list(range(len(self.dataset)))

    def _get_dataloader(self) -> DataLoader:
        from torch.utils.data import Subset
        subset = Subset(self.dataset, self._indices)
        return DataLoader(
            subset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=EvalCollator(),
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    # ------------------------------------------------------------------
    # Per-image persistence
    # ------------------------------------------------------------------
    def _per_image_path(self, result_key: str) -> Path:
        return self.output_dir / "per_image" / f"{result_key}.npz"

    def _collect_image_ids(self) -> np.ndarray:
        """Image ids of this split, in evaluation order (one data pass)."""
        ids: List[str] = []
        for batch in self._get_dataloader():
            ids.extend(str(n) for n in batch["image_names"])
        return np.asarray(ids)

    def _load_per_image(self, result_key: str, image_ids: Optional[np.ndarray]) -> Dict[str, Any]:
        """Load the per-image store; discard it if the image ids do not match."""
        path = self._per_image_path(result_key)
        if not path.exists():
            return {}
        try:
            with np.load(path, allow_pickle=False) as data:
                store = {k: data[k] for k in data.files}
        except Exception as e:
            logger.warning("Ignoring unreadable per-image file %s: %s", path, e)
            return {}
        if image_ids is not None and (
            "image_ids" not in store
            or len(store["image_ids"]) != len(image_ids)
            or not np.array_equal(store["image_ids"].astype(str), np.asarray(image_ids).astype(str))
        ):
            logger.warning("Per-image file %s is for a different image set; ignoring it", path)
            return {}
        return store

    def _save_per_image(self, result_key: str, store: Dict[str, Any]) -> None:
        """Atomically write the per-image store (no pickled objects)."""
        import os
        path = self._per_image_path(result_key)
        _makedirs(path.parent)
        tmp = path.with_name(path.name + ".tmp")
        arrays = {}
        for k, v in store.items():
            if k == "meta":
                arrays[k] = np.asarray(json.dumps(v, sort_keys=True, default=str))
            else:
                arrays[k] = np.asarray(v)
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, path)

    def _quantus_metric_names(self) -> List[str]:
        return [mc["name"] for mc in self.metric_configs]

    def run_method(
        self,
        method_name: str,
        method_params: Optional[Dict[str, Any]] = None,
        component: str = "input",
        layers: Optional[List[str]] = None,
        config_label: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run a single method configuration and evaluate it.

        When ``layers`` is provided, the method is called with those layer
        names and each returned layer map is evaluated independently.
        The input-level map (``component``) is always included when the
        method produces one.

        Args:
            method_name: Registered method name (e.g. 'adf_game').
            method_params: Hyperparameters to set on the method instance.
            component: Which output component to evaluate (e.g. 'input', 'pp-np').
            layers: Optional layer names to hook and evaluate per-layer maps.
            config_label: Label for this config (for results). Defaults to method_name.

        Returns:
            List of result dicts -- one per evaluated map (input + per-layer).
        """
        from bwgame.xai.registry import MethodRegistry

        label = config_label or method_name
        if method_params:
            param_str = "_".join(f"{k}={v}" for k, v in sorted(method_params.items()))
            if not config_label:
                label = f"{method_name}__{param_str}"

        logger.info("Evaluating: %s (layers=%s)", label, layers)

        # Hidden npz cache for metric recomputation
        npz_cache_dir = self.output_dir / ".saliency_cache"
        npz_cache_dir.mkdir(exist_ok=True)
        cache_path = npz_cache_dir / f"{label}.npz"

        # Phase 0: If the metrics_cache already has a finite (non-NaN/Inf)
        # value for every requested metric, return those directly — no need
        # to regenerate saliency. Critical for cache hits where the npz was
        # purged but the JSON-cached metrics remain authoritative.
        import math as _math0, json as _json0
        mc_dir_peek = self.output_dir / "metrics_cache"
        if mc_dir_peek.exists():
            peek_files = list(mc_dir_peek.glob(f"{label}.json")) \
                       + list(mc_dir_peek.glob(f"{label}@*.json"))
            required = set(self._quantus_metric_names())
            if peek_files:
                all_complete = True
                results_short = []
                # The cached scores only count if they were computed on exactly
                # the current image set (same ids, same order).
                current_ids = self._collect_image_ids()
                for p in peek_files:
                    try:
                        with open(p) as f:
                            j = _json0.load(f)
                    except Exception:
                        all_complete = False
                        break
                    if not required.issubset(j.keys()):
                        all_complete = False
                        break
                    if any(isinstance(j[k], float) and (_math0.isnan(j[k]) or _math0.isinf(j[k]))
                           for k in required):
                        all_complete = False
                        break
                    # Per-image scores must be persisted as well.
                    pi = self._load_per_image(p.stem, current_ids)
                    if "image_ids" not in pi or not required.issubset(pi.keys()):
                        all_complete = False
                        break
                    # Build results row from cached metrics
                    layer_tag = p.stem.split("@", 1)[1] if "@" in p.stem else component
                    row = {"method": label, "layer": layer_tag, "component": component,
                           "method_name": method_name, "n_images": int(len(pi["image_ids"]))}
                    for k, v in j.items():
                        if not is_per_image_key(k):
                            row[k] = v
                    if method_params:
                        row["params"] = str(method_params)
                    results_short.append(row)
                if all_complete and results_short:
                    logger.info("All metrics cached for %s; skipping saliency regen", label)
                    return results_short

        # Phase 1: Generate saliency maps (input + per-layer)
        per_image_times: Optional[np.ndarray] = None

        # A metrics cache with NaN / Inf values marks the saliency cache as
        # unreliable (e.g. all-zero maps written after an out-of-memory
        # error); it is invalidated and the maps are regenerated.
        def _metrics_cache_is_broken():
            import math as _m
            mc_dir = self.output_dir / "metrics_cache"
            if not mc_dir.exists():
                return False
            # Metric cache files are named by label (+ optional @map_name).
            for p in mc_dir.glob(f'{label}*.json'):
                try:
                    with open(p) as f:
                        j = __import__('json').load(f)
                except Exception:
                    continue
                for v in j.values():
                    if isinstance(v, float) and (_m.isnan(v) or _m.isinf(v)):
                        return True
            return False

        if cache_path.exists():
            logger.info("Loading cached maps from %s", cache_path)
            data = np.load(cache_path, allow_pickle=True)
            all_images = data["images"]
            all_labels = data["labels"]
            all_masks = data["masks"] if "masks" in data and data["masks"].size > 0 else None
            image_ids = data["image_ids"].astype(str) if "image_ids" in data else None
            named_maps = {}
            for key in data.files:
                if key.startswith("map__"):
                    named_maps[key[5:]] = data[key]

            # Invalidate and regenerate when (1) every map is zero (the
            # placeholder written after an out-of-memory error) or (2) some
            # per-image maps are all zero and a cached metric is NaN / Inf.
            invalidate = False
            if any(
                np.asarray(m).size > 0 and np.abs(np.asarray(m)).sum() == 0.0
                for m in named_maps.values()
            ):
                logger.warning(
                    "Discarding all-zero cached maps at %s (stale OOM placeholder)",
                    cache_path,
                )
                invalidate = True
            elif _metrics_cache_is_broken():
                # Check for any per-image zero maps
                for m in named_maps.values():
                    arr = np.asarray(m)
                    if arr.ndim < 2 or arr.size == 0:
                        continue
                    per_img_abs = np.abs(arr).reshape(arr.shape[0], -1).sum(axis=1)
                    if (per_img_abs == 0).any():
                        logger.warning(
                            "Saliency cache at %s has per-image zero slices and "
                            "the associated metrics_cache has NaN/Inf entries; "
                            "invalidating to recompute",
                            cache_path,
                        )
                        invalidate = True
                        break

            if invalidate:
                cache_path.unlink(missing_ok=True)
                named_maps = {}
            # Load cached per-image timing if available
            if named_maps and "per_image_times" in data and data["per_image_times"].size > 0:
                per_image_times = data["per_image_times"]
            if named_maps and image_ids is None:
                image_ids = self._collect_image_ids()

        n_map_failures = 0
        if not cache_path.exists() or not named_maps:
            (named_maps, all_images, all_labels, all_masks, total_time,
             per_image_times, image_ids, n_map_failures) = (
                self._generate_maps(method_name, method_params, component, layers)
            )
            # All-zero maps (the placeholder written after an out-of-memory
            # error) and maps with failed (NaN) images are not cached, so that
            # a rerun recomputes them.
            any_nonzero = any(
                np.asarray(m).size > 0 and np.nansum(np.abs(np.asarray(m))) > 0.0
                for m in named_maps.values()
            )
            if not any_nonzero:
                logger.warning(
                    "All generated maps for %s are zero or failed (likely OOM); not caching",
                    label,
                )
            elif n_map_failures > 0:
                logger.warning(
                    "%s: map generation failed for %d image(s) (stored as NaN); "
                    "not caching the maps so that a rerun retries them",
                    label, n_map_failures,
                )
            elif not self.cache_maps:
                pass
            else:
                # Save npz for metrics (including per-image timing)
                save_kwargs = {
                    "images": all_images,
                    "labels": all_labels,
                    "masks": all_masks if all_masks is not None else np.array([]),
                    "per_image_times": per_image_times if per_image_times is not None else np.array([]),
                    "image_ids": np.asarray(image_ids).astype(str),
                }
                for map_name, map_arr in named_maps.items():
                    save_kwargs[f"map__{map_name}"] = map_arr
                np.savez_compressed(cache_path, **save_kwargs)
            # Save viewable PNG heatmaps
            if self.save_heatmaps:
                self._save_heatmap_images(label, named_maps, all_images)
            logger.info(
                "Generated %d map types in %.1fs (heatmap PNGs: %s)",
                len(named_maps), total_time,
                self.output_dir / "heatmaps" / label if self.save_heatmaps else "disabled",
            )

        if not named_maps:
            logger.warning("No maps generated for %s", label)
            return [{"method": label, "layer": "none", "n_images": 0}]

        # Phase 2: Evaluate each map independently (with per-metric caching)
        import json as _json
        metrics_cache_dir = self.output_dir / "metrics_cache"
        metrics_cache_dir.mkdir(exist_ok=True)

        results = []
        for map_name, saliency_maps in named_maps.items():
            if len(saliency_maps) == 0:
                continue

            result_key = f"{label}@{map_name}" if map_name != component else label
            metrics_cache_path = metrics_cache_dir / f"{result_key}.json"

            # Check for cached metric results
            cached_metrics = {}
            if metrics_cache_path.exists():
                try:
                    with open(metrics_cache_path) as f:
                        cached_metrics = _json.load(f)
                    logger.info("  Loaded %d cached metrics for %s",
                                len(cached_metrics), map_name)
                except Exception:
                    cached_metrics = {}

            # Per-image store for this map (image ids must match).
            per_image = self._load_per_image(result_key, image_ids)
            per_image["image_ids"] = np.asarray(image_ids).astype(str)
            map_arr = np.asarray(saliency_maps, dtype=float).reshape(len(saliency_maps), -1)
            per_image["map_failed"] = ~np.isfinite(map_arr).all(axis=1)
            per_image["map_is_zero"] = np.nan_to_num(np.abs(map_arr)).sum(axis=1) == 0
            if per_image_times is not None and len(per_image_times) == len(saliency_maps):
                per_image["time_s"] = np.asarray(per_image_times, dtype=float)
            meta = {}
            if "meta" in per_image:
                try:
                    meta = json.loads(str(per_image["meta"]))
                except Exception:
                    meta = {}
            per_image["meta"] = meta

            # Determine which metrics still need computing.
            # Retry metrics that previously failed (stored as NaN) and
            # metrics whose per-image scores were not persisted.
            needed_configs = []
            for mc in self.metric_configs:
                name = mc["name"]
                if name not in cached_metrics:
                    needed_configs.append(mc)
                elif name not in per_image:
                    needed_configs.append(mc)
                elif isinstance(cached_metrics[name], float) and (
                    _math.isnan(cached_metrics[name])
                    or _math.isinf(cached_metrics[name])
                ):
                    logger.info("  Retrying previously-failed metric: %s", name)
                    needed_configs.append(mc)

            if needed_configs:
                # Free GPU memory before running Quantus metrics
                # (perturbation metrics need headroom for perturbed forward passes)
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                explain_func = make_explain_func(
                    method_name, self.model, self.device, method_params, component, layers,
                )
                masks = all_masks if all_masks is not None and len(all_masks) > 0 else None

                new_results = evaluate_saliency_maps(
                    model=self.model,
                    images=all_images,
                    labels=all_labels,
                    saliency_maps=saliency_maps,
                    metric_configs=needed_configs,
                    masks=masks,
                    explain_func=explain_func,
                    softmax=self.softmax,
                    device=str(self.device),
                )
                # Merge new results into cache; per-image arrays go to the
                # per-image store instead of the JSON cache.
                for k, v in new_results.items():
                    if k.endswith("_per_image"):
                        per_image[k[: -len("_per_image")]] = np.asarray(v, dtype=float)
                    elif k.endswith("_all"):
                        # Per-image perturbation curves (PF, Sel): (n_images, curve_len)
                        try:
                            arr = np.asarray(v, dtype=float)
                        except (TypeError, ValueError):
                            arr = None
                        if arr is not None and arr.ndim == 2 and arr.shape[0] == len(image_ids):
                            per_image[f"{k[: -len('_all')]}__curve"] = arr
                    elif not is_per_image_key(k):
                        cached_metrics[k] = v
                    if k.endswith("_curve_len") or k.endswith("_auc_points"):
                        meta[k] = v

                # Save updated cache
                try:
                    with open(metrics_cache_path, "w") as f:
                        _json.dump(cached_metrics, f, indent=2, default=str)
                except Exception as e:
                    logger.warning("Failed to save metrics cache: %s", e)

                logger.info("  %s: %d new + %d cached metrics",
                            map_name, len(needed_configs),
                            len(self.metric_configs) - len(needed_configs))
            else:
                logger.info("  %s: all %d metrics cached (skip)", map_name, len(cached_metrics))

            try:
                self._save_per_image(result_key, per_image)
            except Exception as e:
                logger.warning("Failed to save per-image scores for %s: %s", result_key, e)
            n_failed_here = int(np.asarray(per_image["map_failed"]).sum())
            if n_failed_here:
                logger.warning("  %s: %d/%d maps failed; their metric values are NaN "
                               "(see *_coverage)", result_key, n_failed_here, len(saliency_maps))

            result = {
                "method": result_key,
                "method_name": method_name,
                "component": component,
                "layer": map_name,
                "n_images": len(saliency_maps),
                "map_failures": n_failed_here,
            }
            # Inject inference time metrics (failed images have NaN time)
            if per_image_times is not None and np.isfinite(per_image_times).any():
                t = np.asarray(per_image_times, dtype=float)
                n_t = int(np.isfinite(t).sum())
                result["inference_time_mean"] = float(np.nanmean(t))
                result["inference_time_std"] = float(np.nanstd(t, ddof=1)) if n_t > 1 else float("nan")
                result["inference_time_total"] = float(np.nansum(t))
                result["inference_time_per_image"] = float(np.nanmean(t))
            result.update(cached_metrics)
            # params last so it doesn't shift metric columns in CSV
            if method_params:
                result["params"] = str(method_params)
            results.append(result)

        return results

    def _save_heatmap_images(
        self,
        label: str,
        named_maps: Dict[str, np.ndarray],
        all_images: np.ndarray,
    ):
        """Save attribution maps as viewable PNG images.

        Creates one PNG per (method, map_name, image_index) with the heatmap
        overlaid on the original image using a diverging colormap.
        """
        if PILImage is None:
            logger.warning("Pillow not installed, skipping heatmap image export")
            return

        import matplotlib
        matplotlib.use("Agg")

        heatmap_dir = self.output_dir / "heatmaps" / label
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        for map_name, maps in named_maps.items():
            map_dir = heatmap_dir / map_name if map_name != label else heatmap_dir
            map_dir.mkdir(parents=True, exist_ok=True)

            for idx in range(len(maps)):
                sal = np.nan_to_num(maps[idx])  # (H, W); failed maps are NaN

                # Red = positive (scaled 0→1), Blue = negative (scaled -1→0)
                pos = np.clip(sal, 0, None)
                neg = np.clip(sal, None, 0)
                pos_max = pos.max() if pos.max() > 0 else 1e-8
                neg_min = neg.min() if neg.min() < 0 else -1e-8
                pos_norm = pos / pos_max          # [0, 1]
                neg_norm = neg / abs(neg_min)      # [-1, 0]

                heatmap_rgb = np.zeros((*sal.shape, 3), dtype=np.uint8)
                heatmap_rgb[:, :, 0] = (pos_norm * 255).astype(np.uint8)       # R
                heatmap_rgb[:, :, 2] = ((-neg_norm) * 255).astype(np.uint8)    # B

                hmap_img = PILImage.fromarray(heatmap_rgb)
                hmap_img.save(map_dir / f"img{idx:04d}.png")


    def _generate_maps(
        self,
        method_name: str,
        method_params: Optional[Dict[str, Any]],
        component: str,
        layers: Optional[List[str]] = None,
    ):
        """Generate attribution maps for all images in the split.

        ``equivariant_averaging: true`` in ``method_params`` wraps the method
        in :func:`bwgame.xai.wrappers.equivariant_average` (RG+Eq); the
        ``equivariant_*`` keys are its parameters.

        Images for which map generation raises are stored as NaN maps (not
        zeros) so that they are excluded from every metric and show up in
        the per-metric coverage.  Timing: ``self.warmup`` untimed calls on
        the first batch precede the timed loop, and CUDA is synchronised
        before and after every timed call.

        Returns:
            (named_maps, images, labels, masks, elapsed, per_image_times,
             image_ids, n_failed)
            where named_maps is a dict: map_name -> (N, H, W) array.
            The input-level component is keyed by ``component``.
            Per-layer maps are keyed by their layer name (or "layer:component").
            per_image_times is (N,) array of seconds per image (NaN for
            failed images); image_ids is (N,) array of str; n_failed is the
            number of images whose map generation failed.
        """
        from bwgame.xai.registry import MethodRegistry

        mp = dict(method_params or {})
        use_equivariant = mp.pop('equivariant_averaging', False)

        # Extract wrapper-specific params, pass the rest to the method
        eq_params = {}
        inner_params = {}
        for k, v in mp.items():
            if k.startswith('equivariant_'):
                eq_params[k] = v
            else:
                inner_params[k] = v

        def _sync():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        # Accumulators per map name
        per_name_maps: Dict[str, List[np.ndarray]] = {}
        all_images = []
        all_labels = []
        all_masks = []
        image_ids: List[str] = []
        per_image_times: List[float] = []
        n_failed = 0
        t0 = time.time()

        loader = self._get_dataloader()

        if use_equivariant:
            def _wrapper_maps(x_i, t_i):
                from bwgame.xai.wrappers import equivariant_average
                return equivariant_average(
                    method_name, self.model, self.device,
                    x_i, t_i, layers,
                    n_samples=int(eq_params.get('equivariant_n_samples', 50)),
                    max_angle=float(eq_params.get('equivariant_max_angle', 20)),
                    max_shift_frac=float(eq_params.get('equivariant_max_shift_frac', 0.1)),
                    flip_prob=float(eq_params.get('equivariant_flip_prob', 0.5)),
                    method_params=inner_params,
                )

            # Wrapper path: call per-image (wrappers expect (1,C,H,W))
            warmed_up = self.warmup <= 0
            for batch in loader:
                images = batch["images"].to(self.device)
                labels_t = batch["labels"].to(self.device)
                masks = batch["masks"]
                B, _, H, W = images.shape

                if not warmed_up:
                    for _ in range(self.warmup):
                        try:
                            _wrapper_maps(images[0:1], labels_t[0:1])
                        except Exception as e:
                            logger.warning("Warm-up call failed: %s", e)
                    warmed_up = True

                for i in range(B):
                    x_i = images[i:i+1]
                    t_i = labels_t[i:i+1]
                    _sync()
                    t_img = time.time()
                    try:
                        maps = _wrapper_maps(x_i, t_i)
                        _sync()
                        per_image_times.append(time.time() - t_img)

                        for key, arr in maps.items():
                            if component != "input" and key != component:
                                continue
                            heatmap = arr
                            while heatmap.ndim > 2:
                                heatmap = heatmap[0] if heatmap.shape[0] == 1 else heatmap.sum(axis=0)
                            if key not in per_name_maps:
                                per_name_maps[key] = []
                            per_name_maps[key].append(heatmap[np.newaxis])  # (1,H,W)

                    except Exception as e:
                        per_image_times.append(float("nan"))
                        n_failed += 1
                        logger.warning("Wrapper map generation failed (stored as NaN): %s", e)
                        failed = np.full((1, H, W), np.nan, dtype=np.float32)
                        if component not in per_name_maps:
                            per_name_maps[component] = []
                        per_name_maps[component].append(failed)

                all_images.append(images.cpu().numpy())
                all_labels.append(labels_t.cpu().numpy())
                image_ids.extend(str(n) for n in batch["image_names"])
                if masks is not None:
                    all_masks.append(masks)
        else:
            # Standard path: direct method call
            method = MethodRegistry.instantiate(method_name, self.model, self.device)
            if inner_params:
                for k, v in inner_params.items():
                    if hasattr(method, k):
                        setattr(method, k, v)

            warmed_up = self.warmup <= 0
            for batch in loader:
                images = batch["images"].to(self.device)
                labels_t = batch["labels"].to(self.device)
                masks = batch["masks"]
                B, _, H, W = images.shape

                if not warmed_up:
                    for _ in range(self.warmup):
                        try:
                            method.run(images, labels_t, layers)
                        except Exception as e:
                            logger.warning("Warm-up call failed: %s", e)
                    warmed_up = True

                _sync()
                t_batch = time.time()
                try:
                    result = method.run(images, labels_t, layers)

                    _sync()
                    batch_elapsed = time.time() - t_batch
                    # Distribute batch time equally across images
                    per_img = batch_elapsed / B
                    per_image_times.extend([per_img] * B)

                    for key, arr in result.maps.items():
                        # Filter: only keep the component we want to evaluate
                        if component != "input" and key != component:
                            continue
                        if key not in per_name_maps:
                            per_name_maps[key] = []
                        per_name_maps[key].append(arr)

                except Exception as e:
                    per_image_times.extend([float("nan")] * B)
                    n_failed += B
                    logger.warning("Map generation failed for batch (stored as NaN): %s", e)
                    failed = np.full((B, H, W), np.nan, dtype=np.float32)
                    if component not in per_name_maps:
                        per_name_maps[component] = []
                    per_name_maps[component].append(failed)

                all_images.append(images.cpu().numpy())
                all_labels.append(labels_t.cpu().numpy())
                image_ids.extend(str(n) for n in batch["image_names"])
                if masks is not None:
                    all_masks.append(masks)

            method.cleanup()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        total_time = time.time() - t0
        per_image_arr = np.array(per_image_times, dtype=np.float64)
        if n_failed:
            logger.warning("%s: map generation failed for %d/%d images",
                           method_name, n_failed, len(image_ids))

        # Concatenate
        named_maps = {}
        for key, arrs in per_name_maps.items():
            if arrs:
                named_maps[key] = np.concatenate(arrs)

        if all_images:
            return (
                named_maps,
                np.concatenate(all_images),
                np.concatenate(all_labels),
                np.concatenate(all_masks) if all_masks else None,
                total_time,
                per_image_arr,
                np.asarray(image_ids),
                n_failed,
            )
        return {}, np.array([]), np.array([]), None, total_time, per_image_arr, np.asarray(image_ids), n_failed
