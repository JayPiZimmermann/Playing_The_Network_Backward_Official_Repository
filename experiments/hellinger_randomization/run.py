"""Cascading Hellinger randomization diagnostic on ImageNet-S.

For each requested model, this script:

  1. Loads the ImageNet-S split and selects the requested subset
     (defaults to the evaluation-style test split: all images except one
     held-out validation image per class).
  2. Builds a progressively randomized copy of the pretrained network by
     reinitializing learnable layers from the output side back toward the
     input.
  3. For each cumulative randomization step and each requested game,
     computes the Hellinger path distance between the original model and
     the randomized copy on the same inputs.
  4. Caches the per-image Hellinger distances for every
     ``(model, game, step)`` as compressed ``.npz`` files.
  5. Writes final ``avg.csv`` and ``std.csv`` summaries, plus a
     ``per_image.csv`` table.
"""

from __future__ import annotations

import os
import argparse
import copy
import csv
import gc
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

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
    _get_randomizable_layers,
    _randomize_layer,
)
from bwgame.xai.methods.hellinger import HellingerDiagnostic


DEFAULT_MODELS: Tuple[str, ...] = ("vgg16", "resnet50")
DEFAULT_GAMES: Tuple[str, ...] = ("rg", "adf")
_STEP_CACHE_VERSION = 3

_HERE = Path(__file__).resolve().parent
_DEFAULT_RESULTS_DIR = _HERE / "results"
_DEFAULT_IMAGENET_ROOTS = tuple(p for p in (os.environ.get("IMAGENET_ROOT"),) if p) + (
    str(Path(__file__).resolve().parents[2] / "data" / "imagenet"),
)
_DEFAULT_IMAGENET_S_ROOTS = tuple(p for p in (os.environ.get("IMAGENET_S_ROOT"),) if p) + (
    str(Path(__file__).resolve().parents[2] / "data" / "imagenet-s"),
)

_MODEL_FACTORY = {
    "vgg16": (vgg16, VGG16_Weights.DEFAULT),
    "resnet50": (resnet50, ResNet50_Weights.DEFAULT),
    "vit_b_16": (vit_b_16, ViT_B_16_Weights.DEFAULT),
}


@dataclass
class RunConfig:
    max_images: int
    max_steps: int
    final_only: bool
    models: List[str]
    games: List[str]
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


def _build_model(name: str, device: torch.device) -> nn.Module:
    if name not in _MODEL_FACTORY:
        raise ValueError(f"Unknown model {name!r}; choose from {list(_MODEL_FACTORY)}")
    ctor, weights = _MODEL_FACTORY[name]
    model = ctor(weights=weights).to(device).eval()
    return model


def _sanitize_layer_name(layer_name: str) -> str:
    safe = layer_name.replace(".", "-")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", safe)


def _empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_samples(cfg: RunConfig) -> Tuple[torch.Tensor, List[dict]]:
    log = logging.getLogger("hellinger_randomization")
    imagenet_root = _resolve_path_candidates(
        cfg.imagenet_root, _DEFAULT_IMAGENET_ROOTS, "ImageNet root"
    )
    imagenet_s_root = _resolve_path_candidates(
        cfg.imagenet_s_root, _DEFAULT_IMAGENET_S_ROOTS, "ImageNet-S root"
    )
    log.info(
        "Loading ImageNet-S split=%s subset=%s (imagenet=%s, imagenet-s=%s)",
        cfg.imagenet_s_split,
        cfg.split,
        imagenet_root,
        imagenet_s_root,
    )
    from reproduce.common import imagenet_s_split
    eval_dataset = imagenet_s_split(
        cfg.split, imagenet_root, imagenet_s_root, seed=cfg.seed,
        n_per_class_val=cfg.n_per_class_val, image_size=cfg.image_size,
        imagenet_s_variant=cfg.imagenet_s_split,
    )
    n_total = len(eval_dataset)
    n = min(cfg.max_images, n_total) if cfg.max_images > 0 else n_total
    log.info("Selected %d/%d samples", n, n_total)

    images: List[torch.Tensor] = []
    records: List[dict] = []
    for idx in range(n):
        image_t, _mask, label, class_name, image_name = eval_dataset[idx]
        sample_id = Path(image_name).stem
        images.append(image_t)
        records.append(
            {
                "index": idx,
                "id": sample_id,
                "image_name": image_name,
                "class_name": class_name,
                "label": int(label),
            }
        )
    stacked = torch.stack(images, dim=0).contiguous()
    return stacked, records


def _iter_batches(
    images: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
) -> Iterable[Tuple[int, int, torch.Tensor, torch.Tensor]]:
    n = images.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        yield start, end, images[start:end], targets[start:end]


def _predict_targets(
    model: nn.Module,
    images_cpu: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    targets: List[torch.Tensor] = []
    with torch.no_grad():
        for _start, _end, batch_cpu, _dummy in _iter_batches(
            images_cpu,
            torch.zeros(len(images_cpu), dtype=torch.long),
            batch_size,
        ):
            batch = batch_cpu.to(device, non_blocking=True)
            logits = model(batch)
            targets.append(logits.argmax(dim=1).cpu())
    return torch.cat(targets, dim=0)


def _compute_distances(
    model_A: nn.Module,
    model_B: nn.Module,
    game: str,
    images_cpu: torch.Tensor,
    targets_cpu: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the Hellinger diagnostic over all batches.

    Returns five per-sample arrays
    ``(hellinger, h_squared, cemetery_h2, hellinger_live, h_squared_live)``.

    ``*_live`` is the path law conditioned on survival to an ordinary
    input state.
    """
    diag = HellingerDiagnostic(
        model_A,
        model_B=model_B,
        game=game,
        tau=1.0,
    )
    live_diag = HellingerDiagnostic(
        model_A,
        model_B=model_B,
        game=game,
        tau=1.0,
        condition_on_survival=True,
    )
    n = images_cpu.shape[0]
    hellinger = np.empty(n, dtype=np.float32)
    h_squared = np.empty(n, dtype=np.float32)
    cemetery_h2 = np.empty(n, dtype=np.float32)
    hellinger_live = np.empty(n, dtype=np.float32)
    h_squared_live = np.empty(n, dtype=np.float32)
    log = logging.getLogger("hellinger_randomization")
    try:
        for start, end, batch_cpu, target_cpu in _iter_batches(
            images_cpu, targets_cpu, batch_size
        ):
            batch = batch_cpu.to(device, non_blocking=True)
            target = target_cpu.to(device, non_blocking=True)
            result = diag.compute(batch, batch, target=target)
            live_failed = False
            try:
                live_result = live_diag.compute(batch, batch, target=target)
            except ValueError as exc:
                log.warning(
                    "%s live-conditioned batch %d:%d undefined: %s",
                    game,
                    start,
                    end,
                    exc,
                )
                live_result = None
                live_failed = True
                hellinger_live[start:end] = np.nan
                h_squared_live[start:end] = np.nan

            def _to_np(v):
                if isinstance(v, torch.Tensor):
                    return v.detach().cpu().numpy().reshape(-1)
                return np.asarray(v).reshape(-1)

            hellinger[start:end] = _to_np(result["hellinger_distance"]).astype(np.float32, copy=False)
            h_squared[start:end] = _to_np(result["h_squared"]).astype(np.float32, copy=False)
            cemetery_h2[start:end] = _to_np(result["cemetery_h2"]).astype(np.float32, copy=False)
            if live_result is not None:
                hellinger_live[start:end] = _to_np(live_result["hellinger_distance"]).astype(np.float32, copy=False)
                h_squared_live[start:end] = _to_np(live_result["h_squared"]).astype(np.float32, copy=False)
            elif not live_failed:
                raise RuntimeError(f"Missing live-conditioned result for game={game!r}")
    finally:
        del diag
        del live_diag
        gc.collect()
        _empty_cuda_cache()
    return hellinger, h_squared, cemetery_h2, hellinger_live, h_squared_live


def _step_cache_path(
    cfg: RunConfig,
    model_name: str,
    game: str,
    step_idx: int,
    layer_name: str,
) -> Path:
    cache_dir = cfg.results_dir / "cache" / f"{model_name}_{game}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = f"step_{step_idx:02d}_{_sanitize_layer_name(layer_name)}.npz"
    return cache_dir / fname


def _write_step_cache(
    path: Path,
    sample_records: Sequence[dict],
    model_name: str,
    game: str,
    step_idx: int,
    layer_name: str,
    distances: np.ndarray,
    h_squared: np.ndarray,
    cemetery_h2: np.ndarray,
    hellinger_live: np.ndarray,
    h_squared_live: np.ndarray,
):
    np.savez_compressed(
        path,
        cache_version=np.int32(_STEP_CACHE_VERSION),
        model_name=model_name,
        game=game,
        step_idx=np.int32(step_idx),
        num_randomized_layers=np.int32(step_idx + 1),
        layer_name=layer_name,
        sample_ids=np.asarray([r["id"] for r in sample_records]),
        class_names=np.asarray([r["class_name"] for r in sample_records]),
        image_names=np.asarray([r["image_name"] for r in sample_records]),
        hellinger_distance=distances.astype(np.float32, copy=False),
        h_squared=h_squared.astype(np.float32, copy=False),
        cemetery_h2=cemetery_h2.astype(np.float32, copy=False),
        hellinger_live=hellinger_live.astype(np.float32, copy=False),
        h_squared_live=h_squared_live.astype(np.float32, copy=False),
    )


def _load_step_cache(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _cache_is_current(path: Path, model_name: str, game: str,
                      sample_records: Optional[Sequence[dict]] = None) -> bool:
    """Cache format is current and, if given, it holds exactly these images."""
    del model_name, game
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(data["cache_version"]) if "cache_version" in data.files else 0
            names = [str(n) for n in data["image_names"]] if "image_names" in data.files else None
    except Exception:
        return False
    if sample_records is not None and names != [str(r["image_name"]) for r in sample_records]:
        return False
    return version >= _STEP_CACHE_VERSION


def _write_csvs(cfg: RunConfig, sample_records: Sequence[dict]):
    avg_rows: List[dict] = []
    std_rows: List[dict] = []
    per_image_rows: List[dict] = []

    def _scalar_or_nan(arr: np.ndarray, reducer) -> float:
        if arr.size == 0 or np.all(np.isnan(arr)):
            return float("nan")
        return float(reducer(arr[~np.isnan(arr)]))

    # Runs split per model share the results dir: aggregate every model whose
    # caches hold the current images, not only the models of this invocation.
    for model_name in dict.fromkeys(list(cfg.models) + list(_MODEL_FACTORY)):
        for game in cfg.games:
            cache_dir = cfg.results_dir / "cache" / f"{model_name}_{game}"
            if not cache_dir.exists():
                continue
            for cache_path in sorted(cache_dir.glob("step_*.npz")):
                if not _cache_is_current(cache_path, model_name, game, sample_records):
                    continue
                data = _load_step_cache(cache_path)
                distances = data["hellinger_distance"].astype(np.float64, copy=False)
                n = len(distances)
                h_live = data["hellinger_live"].astype(np.float64, copy=False)
                h2_live = data["h_squared_live"].astype(np.float64, copy=False)
                h2 = data["h_squared"].astype(np.float64, copy=False)
                cem = data["cemetery_h2"].astype(np.float64, copy=False)
                step_idx = int(data["step_idx"])
                num_randomized = int(data["num_randomized_layers"])
                layer_name = str(data["layer_name"])
                avg_rows.append(
                    {
                        "model": model_name,
                        "game": game,
                        "step_idx": step_idx,
                        "num_randomized_layers": num_randomized,
                        "last_randomized_layer": layer_name,
                        "n_samples": n,
                        "avg_hellinger": float(distances.mean()),
                        "avg_hellinger_live": _scalar_or_nan(h_live, np.mean),
                        "avg_h2": _scalar_or_nan(h2, np.mean),
                        "avg_h2_live": _scalar_or_nan(h2_live, np.mean),
                        "avg_cemetery_h2": _scalar_or_nan(cem, np.mean),
                    }
                )
                std_rows.append(
                    {
                        "model": model_name,
                        "game": game,
                        "step_idx": step_idx,
                        "num_randomized_layers": num_randomized,
                        "last_randomized_layer": layer_name,
                        "n_samples": n,
                        "std_hellinger": float(distances.std(ddof=0)),
                        "std_hellinger_live": _scalar_or_nan(h_live, lambda a: a.std(ddof=0)),
                        "std_h2_live": _scalar_or_nan(h2_live, lambda a: a.std(ddof=0)),
                    }
                )
                sample_ids = data["sample_ids"]
                class_names = data["class_names"]
                image_names = data["image_names"]
                for idx, dist in enumerate(distances):
                    per_image_rows.append(
                        {
                            "model": model_name,
                            "game": game,
                            "step_idx": step_idx,
                            "num_randomized_layers": num_randomized,
                            "last_randomized_layer": layer_name,
                            "sample_id": str(sample_ids[idx]),
                            "class_name": str(class_names[idx]),
                            "image_name": str(image_names[idx]),
                            "hellinger_distance": float(dist),
                            "hellinger_live": float(h_live[idx]),
                            "H_live": float(h_live[idx]),
                            "h_squared": float(h2[idx]),
                            "h2_live": float(h2_live[idx]),
                            "cemetery_h2": float(cem[idx]),
                        }
                    )

    def _sort_key(row: dict):
        return (row["model"], row["game"], int(row["step_idx"]), row.get("sample_id", ""))

    avg_rows.sort(key=_sort_key)
    std_rows.sort(key=_sort_key)
    per_image_rows.sort(key=_sort_key)

    _write_csv(
        cfg.results_dir / "avg.csv",
        [
            "model",
            "game",
            "step_idx",
            "num_randomized_layers",
            "last_randomized_layer",
            "n_samples",
            "avg_hellinger",
            "avg_hellinger_live",
            "avg_h2",
            "avg_h2_live",
            "avg_cemetery_h2",
        ],
        avg_rows,
    )
    _write_csv(
        cfg.results_dir / "std.csv",
        [
            "model",
            "game",
            "step_idx",
            "num_randomized_layers",
            "last_randomized_layer",
            "n_samples",
            "std_hellinger",
            "std_hellinger_live",
            "std_h2_live",
        ],
        std_rows,
    )
    _write_csv(
        cfg.results_dir / "per_image.csv",
        [
            "model",
            "game",
            "step_idx",
            "num_randomized_layers",
            "last_randomized_layer",
            "sample_id",
            "class_name",
            "image_name",
            "hellinger_distance",
            "hellinger_live",
            "H_live",
            "h_squared",
            "h2_live",
            "cemetery_h2",
        ],
        per_image_rows,
    )

    meta_path = cfg.results_dir / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {}
    meta["n_cached_rows"] = len(per_image_rows)
    meta["csv_files"] = {
        "avg": "avg.csv",
        "std": "std.csv",
        "per_image": "per_image.csv",
    }
    meta_path.write_text(json.dumps(meta, indent=2))


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(cfg: RunConfig) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("hellinger_randomization")
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    images_cpu, sample_records = _load_samples(cfg)
    if device.type == "cuda":
        images_cpu = images_cpu.pin_memory()

    meta = {
        "split": cfg.split,
        "imagenet_s_split": cfg.imagenet_s_split,
        "n_per_class_val": cfg.n_per_class_val,
        "n_samples": len(sample_records),
        "models": cfg.models,
        "games": cfg.games,
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "device": str(device),
        "seed": cfg.seed,
        "samples": sample_records,
        "layers": {},
    }
    (cfg.results_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    _write_csvs(cfg, sample_records)

    for model_name in cfg.models:
        log.info("Building %s on %s", model_name, device)
        model_A = _build_model(model_name, device)
        rand_model = copy.deepcopy(model_A).to(device).eval()

        t0 = time.perf_counter()
        targets_cpu = _predict_targets(model_A, images_cpu, cfg.batch_size, device)
        log.info(
            "%s: computed %d targets in %.1fs",
            model_name,
            len(targets_cpu),
            time.perf_counter() - t0,
        )

        layers = _get_randomizable_layers(model_A)
        if cfg.max_steps > 0:
            layers = layers[:cfg.max_steps]
        meta["layers"][model_name] = layers
        (cfg.results_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
        log.info("%s: %d randomizable layers", model_name, len(layers))

        if cfg.final_only and layers:
            for layer_name in layers:
                _randomize_layer(rand_model, layer_name)
            eval_steps = [(len(layers) - 1, layers[-1])]
            log.info(
                "%s: final-only mode -> randomized all %d layers through %s",
                model_name,
                len(layers),
                layers[-1],
            )
        else:
            for step_idx, layer_name in enumerate(layers):
                _randomize_layer(rand_model, layer_name)
                log.info(
                    "%s: step %d/%d randomized through %s",
                    model_name,
                    step_idx + 1,
                    len(layers),
                    layer_name,
                )
                for game in cfg.games:
                    cache_path = _step_cache_path(cfg, model_name, game, step_idx, layer_name)
                    if cache_path.exists() and not cfg.overwrite:
                        if _cache_is_current(cache_path, model_name, game, sample_records):
                            log.info("  %-3s cached -> %s", game, cache_path.name)
                            continue
                        log.info("  %-3s stale cache -> recomputing %s", game, cache_path.name)
                    t1 = time.perf_counter()
                    distances, h_squared, cemetery_h2, hellinger_live, h_squared_live = _compute_distances(
                        model_A,
                        rand_model,
                        game,
                        images_cpu,
                        targets_cpu,
                        cfg.batch_size,
                        device,
                    )
                    _write_step_cache(
                        cache_path,
                        sample_records,
                        model_name,
                        game,
                        step_idx,
                        layer_name,
                        distances,
                        h_squared,
                        cemetery_h2,
                        hellinger_live,
                        h_squared_live,
                    )
                    log.info(
                        "  %-3s avg=%.4f std=%.4f min=%.4f max=%.4f  live_avg=%.4f  cem_avg=%.4f (%.1fs)",
                        game,
                        float(distances.mean()),
                        float(distances.std(ddof=0)),
                        float(distances.min()),
                        float(distances.max()),
                        float(np.nanmean(hellinger_live)),
                        float(cemetery_h2.mean()),
                        time.perf_counter() - t1,
                    )
                    gc.collect()
                    _empty_cuda_cache()

                _write_csvs(cfg, sample_records)
            eval_steps = []

        for step_idx, layer_name in eval_steps:
            for game in cfg.games:
                cache_path = _step_cache_path(cfg, model_name, game, step_idx, layer_name)
                if cache_path.exists() and not cfg.overwrite:
                    if _cache_is_current(cache_path, model_name, game, sample_records):
                        log.info("  %-3s cached -> %s", game, cache_path.name)
                        continue
                    log.info("  %-3s stale cache -> recomputing %s", game, cache_path.name)
                t1 = time.perf_counter()
                distances, h_squared, cemetery_h2, hellinger_live, h_squared_live = _compute_distances(
                    model_A,
                    rand_model,
                    game,
                    images_cpu,
                    targets_cpu,
                    cfg.batch_size,
                    device,
                )
                _write_step_cache(
                    cache_path,
                    sample_records,
                    model_name,
                    game,
                    step_idx,
                    layer_name,
                    distances,
                    h_squared,
                    cemetery_h2,
                    hellinger_live,
                    h_squared_live,
                )
                log.info(
                    "  %-3s avg=%.4f std=%.4f min=%.4f max=%.4f  live_avg=%.4f  cem_avg=%.4f (%.1fs)",
                    game,
                    float(distances.mean()),
                    float(distances.std(ddof=0)),
                    float(distances.min()),
                    float(distances.max()),
                    float(np.nanmean(hellinger_live)),
                    float(cemetery_h2.mean()),
                    time.perf_counter() - t1,
                )
                gc.collect()
                _empty_cuda_cache()

            _write_csvs(cfg, sample_records)

        del rand_model
        del model_A
        gc.collect()
        _empty_cuda_cache()

    _write_csvs(cfg, sample_records)


def _parse_args(argv: Optional[List[str]] = None) -> RunConfig:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Maximum number of images to use from the chosen split (0 = all).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Maximum number of cumulative randomization steps per model (0 = all).",
    )
    p.add_argument(
        "--final-only",
        action="store_true",
        help="Evaluate only the fully randomized model state for each model.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=list(_MODEL_FACTORY.keys()),
    )
    p.add_argument(
        "--games",
        nargs="+",
        default=list(DEFAULT_GAMES),
        choices=["rg", "adf"],
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
    )
    p.add_argument(
        "--imagenet-root",
        type=str,
        default=None,
        help="Root containing ImageNet val images.",
    )
    p.add_argument(
        "--imagenet-s-root",
        type=str,
        default=None,
        help="Root containing ImageNet-S annotations.",
    )
    p.add_argument(
        "--imagenet-s-split",
        type=str,
        default="ImageNetS50",
        help="ImageNet-S annotation split (default: ImageNetS50).",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test", "val"],
        help="Evaluation subset of the ImageNet-S pool.",
    )
    p.add_argument(
        "--n-per-class-val",
        type=int,
        default=1,
        help="Validation holdout size per class when split=test.",
    )
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (defaults to cuda if available, else cpu).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute cached steps even when cache files already exist.",
    )

    args = p.parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return RunConfig(
        max_images=args.max_images,
        max_steps=args.max_steps,
        final_only=args.final_only,
        models=args.models,
        games=args.games,
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
    )


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    cfg = _parse_args(argv)
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
