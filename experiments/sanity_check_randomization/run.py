"""Adebayo 2018 cascading randomization sanity check on ImageNet-S.

For each (model, method config, randomization step, similarity metric)
triple this script:

  1. Loads the ImageNet-S split (default: test split, the same one the
     paper's main tables use).
  2. Loads a pretrained model (ViT-B/16, VGG-16, ResNet-50) and computes
     the **original** attribution maps with the requested method + params.
  3. Progressively randomizes the model from the top (output-side) layer
     down to the input, recomputing the attribution maps at each step.
  4. For each step it evaluates the four Adebayo similarity metrics
     (Spearman-ABS, Spearman, SSIM, HOG-Pearson) between the original
     and randomized maps, per image.
  5. Caches per-image similarities as ``.npz`` files under
     ``results/cache/<model>/<method_label>/step_XX_<layer>.npz`` and
     writes aggregate ``avg.csv`` / ``std.csv`` / ``per_image.csv`` +
     ``scores.csv`` (AUC across the randomization curve per metric).

The method configurations are the ones reported in the paper, read from
the validation selection (``--from-selection``, see
``method_configs.configs_from_selection``).
"""

from __future__ import annotations

import os
import argparse
import csv
import gc
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    VGG16_Weights,
    ViT_B_16_Weights,
    resnet50,
    vgg16,
    vit_b_16,
)

from bwgame.evaluation.metrics.sanity_check import (
    SIMILARITY_METRICS,
    StepResult,
    iter_cascading_randomization,
)
import bwgame.xai.methods  # noqa: F401 - populates MethodRegistry

from experiments.sanity_check_randomization.method_configs import (
    MethodConfig,
    config_component,
    configs_from_selection,
)


DEFAULT_MODELS: Tuple[str, ...] = ("vit_b_16", "vgg16", "resnet50")
DEFAULT_SIMS: Tuple[str, ...] = tuple(SIMILARITY_METRICS.keys())

_HERE = Path(__file__).resolve().parent
_DEFAULT_RESULTS_DIR = _HERE / "results"
_DEFAULT_IMAGENET_ROOTS = tuple(p for p in (os.environ.get("IMAGENET_ROOT"),) if p) + (
    str(Path(__file__).resolve().parents[2] / "data" / "imagenet"),
)
_DEFAULT_IMAGENET_S_ROOTS = tuple(p for p in (os.environ.get("IMAGENET_S_ROOT"),) if p) + (
    str(Path(__file__).resolve().parents[2] / "data" / "imagenet-s"),
)

_MODEL_FACTORY = {
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.DEFAULT),
    "vgg16":    (vgg16,    VGG16_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
}


logger = logging.getLogger("sanity_check_randomization")


# =========================================================================
# Config
# =========================================================================


@dataclass
class RunConfig:
    models: List[str]
    methods: Optional[List[str]]           # label filter; None = all
    similarity_metrics: List[str]
    max_images: int
    max_steps: int
    n_first_steps: int
    earliest_layer: Optional[str]
    results_dir: Path
    imagenet_root: Optional[str]
    imagenet_s_root: Optional[str]
    imagenet_s_split: str
    split: str
    n_per_class_val: int
    image_size: int
    batch_size: int
    device: str
    overwrite: bool
    seed: int
    from_selection: str
    only_steps: Optional[List[int]] = None


# =========================================================================
# Path & I/O helpers
# =========================================================================


def _resolve_path_candidates(
    user_path: Optional[str],
    default_candidates: Sequence[str],
    label: str,
) -> str:
    candidates = [user_path] if user_path else list(default_candidates)
    for candidate in candidates:
        if candidate and Path(candidate).is_dir():
            return candidate
    raise FileNotFoundError(
        f"{label}: none of the candidate paths exist: {candidates}"
    )


def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s.replace(".", "-"))


def _build_model(name: str, device: torch.device) -> nn.Module:
    if name not in _MODEL_FACTORY:
        raise ValueError(f"Unknown model {name!r}; choose from {list(_MODEL_FACTORY)}")
    ctor, weights = _MODEL_FACTORY[name]
    model = ctor(weights=weights).to(device).eval()

    # Match the evaluator's preparation pipeline so attribution methods
    # that expect DC composites work unchanged on the pretrained model.
    try:
        from bwgame.composite_replacer import decompose_composites
        decompose_composites(model)
    except Exception:
        pass
    try:
        from bwgame.functional_replacer import (
            replace_inline_modules, _replace_functional_in_module,
        )
        replace_inline_modules(model, inplace=True)
        _replace_functional_in_module(model)
    except Exception:
        pass

    return model


# =========================================================================
# Dataset loading
# =========================================================================


def _load_samples(cfg: RunConfig) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    imagenet_root = _resolve_path_candidates(
        cfg.imagenet_root, _DEFAULT_IMAGENET_ROOTS, "ImageNet root",
    )
    imagenet_s_root = _resolve_path_candidates(
        cfg.imagenet_s_root, _DEFAULT_IMAGENET_S_ROOTS, "ImageNet-S root",
    )
    logger.info(
        "Loading ImageNet-S split=%s subset=%s (imagenet=%s, imagenet-s=%s)",
        cfg.imagenet_s_split, cfg.split, imagenet_root, imagenet_s_root,
    )
    from reproduce.common import imagenet_s_split
    eval_dataset = imagenet_s_split(
        cfg.split, imagenet_root, imagenet_s_root, seed=cfg.seed,
        n_per_class_val=cfg.n_per_class_val, image_size=cfg.image_size,
        imagenet_s_variant=cfg.imagenet_s_split,
    )
    n_total = len(eval_dataset)
    n = min(cfg.max_images, n_total) if cfg.max_images > 0 else n_total
    logger.info("Selected %d/%d samples", n, n_total)

    images: List[torch.Tensor] = []
    labels: List[int] = []
    records: List[dict] = []
    for idx in range(n):
        image_t, _mask, label, class_name, image_name = eval_dataset[idx]
        sample_id = Path(image_name).stem
        images.append(image_t)
        labels.append(int(label))
        records.append({
            "index": idx,
            "id": sample_id,
            "image_name": image_name,
            "class_name": class_name,
            "label": int(label),
        })
    return (
        torch.stack(images, dim=0).contiguous(),
        torch.tensor(labels, dtype=torch.long),
        records,
    )


# =========================================================================
# Per-model cascading driver
# =========================================================================


def _filter_methods(
    all_configs: List[MethodConfig],
    wanted_labels: Optional[List[str]],
) -> List[MethodConfig]:
    if not wanted_labels:
        return all_configs
    wanted = set(wanted_labels)
    return [c for c in all_configs if c[0] in wanted]


def _step_cache_path(
    cfg: RunConfig,
    model_name: str,
    method_label: str,
    step_idx: int,
    layer_name: str,
) -> Path:
    d = cfg.results_dir / "cache" / model_name / method_label
    d.mkdir(parents=True, exist_ok=True)
    return d / f"step_{step_idx:02d}_{_sanitize(layer_name)}.npz"


def _write_step_cache(
    path: Path,
    model_name: str,
    method_label: str,
    method_name: str,
    step_idx: int,
    step: StepResult,
    sample_records: Sequence[dict],
):
    payload = dict(
        model_name=model_name,
        method_label=method_label,
        method_name=method_name,
        step_idx=np.int32(step_idx),
        num_randomized_layers=np.int32(step.num_randomized_layers),
        layer_name=step.layer_name,
        sample_ids=np.asarray([r["id"] for r in sample_records]),
        class_names=np.asarray([r["class_name"] for r in sample_records]),
        image_names=np.asarray([r["image_name"] for r in sample_records]),
    )
    for metric, values in step.similarities.items():
        payload[f"sim__{metric}"] = values.astype(np.float32, copy=False)
    np.savez_compressed(path, **payload)


def _load_step_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _same_samples(data: dict, sample_records: Sequence[dict]) -> bool:
    """True if a cache was computed on exactly the current images (same order)."""
    names = data.get("image_names")
    return names is not None and [str(n) for n in names] == [str(r["image_name"]) for r in sample_records]


def _run_one_method(
    cfg: RunConfig,
    model_name: str,
    method_label: str,
    method_name: str,
    method_params: dict,
    component: str,
    images: torch.Tensor,
    labels: torch.Tensor,
    sample_records: Sequence[dict],
    device: torch.device,
) -> bool:
    """Run the cascade for one method; skip steps that are already cached.

    Returns False if the method failed (the error is logged; the remaining
    methods still run and ``main`` exits non-zero).

    Each method gets a freshly built copy of the model, so that hooks or
    module replacements left by one method cannot affect the next.
    """
    logger.info(
        "== [%s] %s (%s) %s component=%s ==",
        model_name, method_label, method_name, method_params, component,
    )

    def _compute_step(step_idx: int, layer_name: str) -> bool:
        """Skip steps outside --only-steps and steps already cached for these images."""
        if cfg.only_steps is not None and step_idx not in cfg.only_steps:
            return False
        if cfg.n_first_steps > 0 and step_idx >= cfg.n_first_steps:
            return False
        path = _step_cache_path(cfg, model_name, method_label, step_idx, layer_name)
        return cfg.overwrite or not (path.exists()
                                     and _same_samples(_load_step_cache(path), sample_records))

    t0 = time.perf_counter()
    model = _build_model(model_name, device)
    ok = True
    try:
        for step in iter_cascading_randomization(
            model=model,
            method_name=method_name,
            device=device,
            images=images,
            labels=labels,
            method_params=method_params,
            component=component,
            similarity_metrics=cfg.similarity_metrics,
            earliest_layer=cfg.earliest_layer,
            max_steps=cfg.max_steps if cfg.max_steps > 0 else None,
            seed=cfg.seed,
            compute_step=_compute_step,
        ):
            # Early exit once we have the requested number of leading
            # (layer-by-layer) steps. This is what the caller wants when
            # --n-first-steps is set and --max-steps=0 (no downsampling),
            # so the kept steps match the fine-grained layer-by-layer
            # granularity of the Hellinger randomization.
            if cfg.n_first_steps > 0 and step.step_idx >= cfg.n_first_steps:
                break
            cache_path = _step_cache_path(
                cfg, model_name, method_label, step.step_idx, step.layer_name,
            )
            if (cache_path.exists() and not cfg.overwrite
                    and _same_samples(_load_step_cache(cache_path), sample_records)):
                logger.info(
                    "  step %d (%s): cached -> %s",
                    step.step_idx, step.layer_name, cache_path.name,
                )
            else:
                _write_step_cache(
                    cache_path, model_name, method_label, method_name,
                    step.step_idx, step, sample_records,
                )
                summary = " ".join(
                    f"{m}={float(np.nanmean(v)):+.3f}"
                    for m, v in step.similarities.items()
                )
                logger.info(
                    "  step %d (%s, %d/%d): %s",
                    step.step_idx, step.layer_name, step.num_randomized_layers,
                    step.num_randomized_layers, summary,
                )
    except Exception as e:
        ok = False
        logger.exception(
            "[%s] %s (%s) failed: %s",
            model_name, method_label, method_name, e,
        )

    logger.info(
        "== [%s] %s done in %.1fs ==",
        model_name, method_label, time.perf_counter() - t0,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ok


# =========================================================================
# Aggregation / CSV writing
# =========================================================================


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _aggregate_to_csv(cfg: RunConfig, sample_records: Sequence[dict]) -> None:
    avg_rows: List[dict] = []
    std_rows: List[dict] = []
    per_image_rows: List[dict] = []
    score_rows: List[dict] = []

    base = cfg.results_dir / "cache"
    if not base.exists():
        return

    for model_dir in sorted(base.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name
        for method_dir in sorted(model_dir.iterdir()):
            if not method_dir.is_dir():
                continue
            method_label = method_dir.name
            step_files = sorted(method_dir.glob("step_*.npz"))
            if not step_files:
                continue

            curves: dict = {m: [] for m in cfg.similarity_metrics}

            for cache_path in step_files:
                data = _load_step_cache(cache_path)
                if not _same_samples(data, sample_records):
                    continue            # computed on another image set
                step_idx = int(data["step_idx"])
                num_randomized = int(data["num_randomized_layers"])
                layer_name = str(data["layer_name"])
                method_name = str(data.get("method_name", ""))

                base_row = {
                    "model": model_name,
                    "method_label": method_label,
                    "method_name": method_name,
                    "step_idx": step_idx,
                    "num_randomized_layers": num_randomized,
                    "last_randomized_layer": layer_name,
                }

                sample_ids = data.get("sample_ids", np.array([]))

                for metric in cfg.similarity_metrics:
                    key = f"sim__{metric}"
                    if key not in data:
                        continue
                    values = np.asarray(data[key], dtype=np.float64)
                    clean = values[~np.isnan(values)]
                    mean_v = float(clean.mean()) if clean.size else float("nan")
                    std_v = float(clean.std(ddof=0)) if clean.size else float("nan")

                    avg_rows.append({**base_row, "metric": metric,
                                     "n_samples": len(values), "mean": mean_v})
                    std_rows.append({**base_row, "metric": metric,
                                     "n_samples": len(values), "std": std_v})
                    curves[metric].append((step_idx, num_randomized, mean_v))

                    for i, v in enumerate(values):
                        sid = str(sample_ids[i]) if i < len(sample_ids) else ""
                        per_image_rows.append({
                            **base_row,
                            "metric": metric,
                            "sample_id": sid,
                            "similarity": float(v),
                        })

            # AUC scores per metric
            for metric, pts in curves.items():
                pts.sort(key=lambda t: t[0])
                if len(pts) >= 2:
                    n = len(pts)
                    x_axis = np.linspace(0.0, 1.0, n)
                    ys = np.array([p[2] for p in pts], dtype=np.float64)
                    mask = ~np.isnan(ys)
                    if mask.sum() >= 2:
                        auc = float(np.trapz(ys[mask], x_axis[mask]))
                    elif mask.sum() == 1:
                        auc = float(ys[mask][0])
                    else:
                        auc = float("nan")
                else:
                    auc = float("nan")
                score_rows.append({
                    "model": model_name,
                    "method_label": method_label,
                    "metric": metric,
                    "n_steps": len(pts),
                    "auc": auc,
                })

    _sort_key_avg = lambda r: (r["model"], r["method_label"], r["metric"], int(r["step_idx"]))
    _sort_key_per = lambda r: (r["model"], r["method_label"], r["metric"],
                               int(r["step_idx"]), r.get("sample_id", ""))
    avg_rows.sort(key=_sort_key_avg)
    std_rows.sort(key=_sort_key_avg)
    per_image_rows.sort(key=_sort_key_per)
    score_rows.sort(key=lambda r: (r["model"], r["method_label"], r["metric"]))

    avg_fields = ["model", "method_label", "method_name", "metric", "step_idx",
                  "num_randomized_layers", "last_randomized_layer", "n_samples", "mean"]
    std_fields = ["model", "method_label", "method_name", "metric", "step_idx",
                  "num_randomized_layers", "last_randomized_layer", "n_samples", "std"]
    per_fields = ["model", "method_label", "method_name", "metric", "step_idx",
                  "num_randomized_layers", "last_randomized_layer",
                  "sample_id", "similarity"]
    score_fields = ["model", "method_label", "metric", "n_steps", "auc"]

    _write_csv(cfg.results_dir / "avg.csv", avg_fields, avg_rows)
    _write_csv(cfg.results_dir / "std.csv", std_fields, std_rows)
    _write_csv(cfg.results_dir / "per_image.csv", per_fields, per_image_rows)
    _write_csv(cfg.results_dir / "scores.csv", score_fields, score_rows)


# =========================================================================
# Orchestration
# =========================================================================


def run(cfg: RunConfig) -> int:
    """Run the cascades; returns the number of failed (model, method) pairs."""
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    images_cpu, labels_cpu, sample_records = _load_samples(cfg)
    if device.type == "cuda":
        images_cpu = images_cpu.pin_memory()

    meta = {
        "split": cfg.split,
        "imagenet_s_split": cfg.imagenet_s_split,
        "n_samples": len(sample_records),
        "models": cfg.models,
        "methods_filter": cfg.methods,
        "similarity_metrics": cfg.similarity_metrics,
        "max_steps": cfg.max_steps,
        "earliest_layer": cfg.earliest_layer,
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "device": str(device),
        "seed": cfg.seed,
        "samples": sample_records,
    }
    # Several invocations (one per model / method) share the results dir:
    # accumulate the models and methods they ran.
    meta_path = cfg.results_dir / "run_meta.json"
    try:
        old = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        old = {}
    if all(old.get(k) == meta[k] for k in ("split", "n_samples", "seed", "similarity_metrics")):
        meta["models"] = sorted(set(old.get("models", [])) | set(cfg.models))
        meta["methods_filter"] = sorted(set(old.get("methods_filter") or []) | set(cfg.methods or []))
    meta_path.write_text(json.dumps(meta, indent=2))

    failed: List[str] = []
    for model_name in cfg.models:
        selection_model = {"vit_b_16": "vit_b16"}.get(model_name, model_name)
        table = configs_from_selection(cfg.from_selection.format(model=selection_model))
        configs = _filter_methods(table, cfg.methods)
        if not configs:
            logger.warning("No methods matched the filter for %s", model_name)
            continue

        for mc in configs:
            method_label, method_name, method_params = mc[0], mc[1], mc[2]
            component = config_component(mc)
            if not _run_one_method(
                cfg, model_name, method_label, method_name, method_params,
                component,
                images_cpu, labels_cpu, sample_records, device,
            ):
                failed.append(f"{model_name}/{method_label}")
            _aggregate_to_csv(cfg, sample_records)

    _aggregate_to_csv(cfg, sample_records)
    logger.info("All done. Results -> %s", cfg.results_dir)
    if failed:
        logger.error("%d method(s) failed: %s", len(failed), ", ".join(failed))
    return len(failed)


# =========================================================================
# CLI
# =========================================================================


def _parse_args(argv: Optional[List[str]] = None) -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS),
                   choices=list(_MODEL_FACTORY.keys()))
    p.add_argument("--methods", nargs="+", default=None,
                   help="Method-label filter (e.g. rg__loc sg__loc). "
                        "Omit to run every selected configuration.")
    p.add_argument("--similarity-metrics", nargs="+", default=list(DEFAULT_SIMS),
                   choices=list(DEFAULT_SIMS))
    p.add_argument("--max-images", type=int, default=0,
                   help="Limit images; 0 = entire split.")
    p.add_argument("--max-steps", type=int, default=12,
                   help="Max randomization steps per method; "
                        "0 = every randomizable layer.")
    p.add_argument("--earliest-layer", type=str, default=None,
                   help="Deepest (closest-to-input) layer name to randomize "
                        "(inclusive). Overrides --max-steps.")
    p.add_argument("--n-first-steps", type=int, default=0,
                   help="Stop after this many cumulative-randomization steps "
                        "(layer-by-layer). Combine with --max-steps=0 to keep "
                        "the top-to-bottom granularity aligned with the "
                        "Hellinger randomization run. 0 disables the early break.")
    p.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--imagenet-root", type=str, default=None)
    p.add_argument("--imagenet-s-root", type=str, default=None)
    p.add_argument("--imagenet-s-split", type=str, default="ImageNetS50")
    p.add_argument("--split", type=str, default="test",
                   choices=["test", "val"])
    p.add_argument("--n-per-class-val", type=int, default=1)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--from-selection", type=str, required=True,
                   help="Path template of selected.json ('{model}' is replaced): the "
                        "configurations of the paper.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--only-steps", type=int, nargs="+", default=None,
                   help="Compute only these cascade steps (e.g. 11 = full randomisation). "
                        "Attribution sampling does not touch the stream that "
                        "re-initialises the layers, so a step alone gives the same "
                        "result as within the full cascade.")

    args = p.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return RunConfig(
        models=args.models,
        methods=args.methods,
        similarity_metrics=args.similarity_metrics,
        max_images=args.max_images,
        max_steps=args.max_steps,
        n_first_steps=args.n_first_steps,
        only_steps=args.only_steps,
        earliest_layer=args.earliest_layer,
        results_dir=args.results_dir,
        imagenet_root=args.imagenet_root,
        imagenet_s_root=args.imagenet_s_root,
        imagenet_s_split=args.imagenet_s_split,
        split=args.split,
        n_per_class_val=args.n_per_class_val,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=device,
        overwrite=args.overwrite,
        seed=args.seed,
        from_selection=args.from_selection,
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = _parse_args(argv)
    return 1 if run(cfg) else 0


if __name__ == "__main__":
    sys.exit(main())
