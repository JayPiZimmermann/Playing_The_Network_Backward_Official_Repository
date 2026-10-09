r"""Input-noise Hellinger diagnostic on ImageNet-S.

Companion to ``experiments/hellinger_randomization/run.py``. Instead of
perturbing the model's weights, we perturb the \emph{input}: for each
architecture, each noise standard deviation ``sigma`` in
``--sigmas``, and each ImageNet-S test image ``x``, we compute the
Hellinger trajectory distance

    H( Pi_A(x) , Pi_A(x + sigma * eps) )

under both game kernels (ADF Stopping Game, RG at tau=1), with the
argmax label of the \emph{clean} input serving as the target for both
trajectories. ``eps`` is standard-Gaussian noise drawn once per
``(model, sigma, image)`` tuple.

Outputs match the randomization run: ``cache/{model}_{game}/sigma_*.npz``
plus aggregate ``avg.csv`` / ``std.csv`` / ``per_image.csv`` (the ``step_idx``
/ ``num_randomized_layers`` / ``last_randomized_layer`` columns of the
randomization run are replaced by ``sigma`` here). The appendix figure is
generated from these CSVs by ``reproduce/diagnostic_figures.py``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

# Reuse infrastructure from the randomization run.
from experiments.hellinger_randomization.run import (
    _build_model,
    _load_samples,
    _iter_batches,
    _predict_targets,
    _empty_cuda_cache,
    _write_csv,
)

from bwgame.xai.methods.hellinger import HellingerDiagnostic


DEFAULT_MODELS: Tuple[str, ...] = ("vgg16", "resnet50", "vit_b_16")
DEFAULT_GAMES: Tuple[str, ...] = ("rg", "adf")
DEFAULT_SIGMAS: Tuple[float, ...] = (
    0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5,
)

_HERE = Path(__file__).resolve().parent
_DEFAULT_RESULTS_DIR = _HERE / "results"
_CACHE_VERSION = 1


@dataclass
class RunConfig:
    max_images: int
    models: List[str]
    games: List[str]
    sigmas: List[float]
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


def _format_sigma(sigma: float) -> str:
    """Filesystem-safe sigma tag: sigma_0p100, sigma_0p005, etc.

    Uses 3 decimals so small sigmas like 0.005 don't collide with 0.01.
    """
    return f"sigma_{sigma:.3f}".replace(".", "p")


def _compute_distances_noise(
    model: nn.Module,
    game: str,
    images_cpu: torch.Tensor,
    targets_cpu: torch.Tensor,
    sigma: float,
    noise_seed: int,
    batch_size: int,
    device: torch.device,
):
    """Hellinger( Pi(x), Pi(x + sigma * eps) ) for a single model at fixed sigma.

    ``eps`` is drawn once per (model, sigma) as a tensor the same shape as
    the full image batch and then sliced per mini-batch, so the same noise
    realisation is used for both CPU-stacked batch traversal passes (Pi_A
    and Pi_B share the same model; only the input differs).
    Returns five per-sample numpy arrays mirroring the randomization run.
    """
    diag = HellingerDiagnostic(model, model_B=model, game=game, tau=1.0)
    live_diag = HellingerDiagnostic(
        model, model_B=model, game=game, tau=1.0,
        condition_on_survival=True,
    )
    n = images_cpu.shape[0]
    hellinger = np.empty(n, dtype=np.float32)
    h_squared = np.empty(n, dtype=np.float32)
    cemetery_h2 = np.empty(n, dtype=np.float32)
    hellinger_live = np.empty(n, dtype=np.float32)
    h_squared_live = np.empty(n, dtype=np.float32)
    log = logging.getLogger("hellinger_input_noise")
    # Deterministic noise realisation per (model, sigma).
    g = torch.Generator(device="cpu").manual_seed(noise_seed)
    eps_all = torch.randn(images_cpu.shape, generator=g, dtype=images_cpu.dtype)
    try:
        for start, end, batch_cpu, target_cpu in _iter_batches(
            images_cpu, targets_cpu, batch_size
        ):
            batch_A = batch_cpu.to(device, non_blocking=True)
            noisy_cpu = batch_cpu + sigma * eps_all[start:end]
            batch_B = noisy_cpu.to(device, non_blocking=True)
            target = target_cpu.to(device, non_blocking=True)
            result = diag.compute(batch_A, batch_B, target=target)
            try:
                live_result = live_diag.compute(batch_A, batch_B, target=target)
            except ValueError as exc:
                log.warning(
                    "%s live-conditioned batch %d:%d undefined: %s",
                    game, start, end, exc,
                )
                live_result = None
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
    finally:
        del diag
        del live_diag
        gc.collect()
        _empty_cuda_cache()
    return hellinger, h_squared, cemetery_h2, hellinger_live, h_squared_live


def _sigma_cache_path(
    cfg: RunConfig, model_name: str, game: str, sigma: float,
) -> Path:
    cache_dir = cfg.results_dir / "cache" / f"{model_name}_{game}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{_format_sigma(sigma)}.npz"


def _write_sigma_cache(
    path: Path,
    sample_records: Sequence[dict],
    model_name: str,
    game: str,
    sigma: float,
    hellinger: np.ndarray,
    h_squared: np.ndarray,
    cemetery_h2: np.ndarray,
    hellinger_live: np.ndarray,
    h_squared_live: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        cache_version=np.int32(_CACHE_VERSION),
        model_name=model_name,
        game=game,
        sigma=np.float32(sigma),
        sample_ids=np.asarray([r["id"] for r in sample_records]),
        class_names=np.asarray([r["class_name"] for r in sample_records]),
        image_names=np.asarray([r["image_name"] for r in sample_records]),
        hellinger_distance=hellinger.astype(np.float32, copy=False),
        h_squared=h_squared.astype(np.float32, copy=False),
        cemetery_h2=cemetery_h2.astype(np.float32, copy=False),
        hellinger_live=hellinger_live.astype(np.float32, copy=False),
        h_squared_live=h_squared_live.astype(np.float32, copy=False),
    )


def _cache_matches(path: Path, sample_records: Sequence[dict]) -> bool:
    """True if a cache was computed on exactly the current images (same order)."""
    try:
        with np.load(path, allow_pickle=False) as data:
            names = [str(n) for n in data["image_names"]]
    except Exception:
        return False
    return names == [str(r["image_name"]) for r in sample_records]


def _write_csvs(cfg: RunConfig, sample_records: Sequence[dict]) -> None:
    avg_rows: List[dict] = []
    std_rows: List[dict] = []
    per_image_rows: List[dict] = []

    def _scalar_or_nan(arr: np.ndarray, reducer) -> float:
        if arr.size == 0 or np.all(np.isnan(arr)):
            return float("nan")
        return float(reducer(arr[~np.isnan(arr)]))

    # Runs split per model share the results dir: aggregate every model whose
    # caches hold the current images, not only the models of this invocation.
    for model_name in dict.fromkeys(list(cfg.models) + list(DEFAULT_MODELS)):
        for game in cfg.games:
            cache_dir = cfg.results_dir / "cache" / f"{model_name}_{game}"
            if not cache_dir.exists():
                continue
            for cache_path in sorted(cache_dir.glob("sigma_*.npz")):
                if not _cache_matches(cache_path, sample_records):
                    continue            # computed on another image set
                with np.load(cache_path, allow_pickle=False) as data:
                    sigma = float(data["sigma"])
                    distances = data["hellinger_distance"].astype(np.float64, copy=False)
                    h2 = data["h_squared"].astype(np.float64, copy=False)
                    cem = data["cemetery_h2"].astype(np.float64, copy=False)
                    h_live = data["hellinger_live"].astype(np.float64, copy=False)
                    h2_live = data["h_squared_live"].astype(np.float64, copy=False)
                    sids = [str(s) for s in data["sample_ids"]]
                    classes = [str(s) for s in data["class_names"]]
                    imgs = [str(s) for s in data["image_names"]]
                n = len(distances)
                avg_rows.append({
                    "model": model_name, "game": game, "sigma": sigma,
                    "n_samples": n,
                    "avg_hellinger": float(distances.mean()),
                    "avg_hellinger_live": _scalar_or_nan(h_live, np.mean),
                    "avg_h2": _scalar_or_nan(h2, np.mean),
                    "avg_h2_live": _scalar_or_nan(h2_live, np.mean),
                    "avg_cemetery_h2": _scalar_or_nan(cem, np.mean),
                })
                std_rows.append({
                    "model": model_name, "game": game, "sigma": sigma,
                    "n_samples": n,
                    "std_hellinger": _scalar_or_nan(distances, np.std),
                    "std_hellinger_live": _scalar_or_nan(h_live, np.std),
                    "std_h2_live": _scalar_or_nan(h2_live, np.std),
                })
                for i, (sid, cn, im) in enumerate(zip(sids, classes, imgs)):
                    per_image_rows.append({
                        "model": model_name, "game": game, "sigma": sigma,
                        "sample_id": sid, "class_name": cn, "image_name": im,
                        "hellinger_distance": float(distances[i]),
                        "hellinger_live": float(h_live[i]),
                        "h_squared": float(h2[i]),
                        "h2_live": float(h2_live[i]),
                        "cemetery_h2": float(cem[i]),
                    })

    avg_rows.sort(key=lambda r: (r["model"], r["game"], r["sigma"]))
    std_rows.sort(key=lambda r: (r["model"], r["game"], r["sigma"]))
    per_image_rows.sort(key=lambda r: (r["model"], r["game"], r["sigma"], r["sample_id"]))

    _write_csv(cfg.results_dir / "avg.csv",
               ["model","game","sigma","n_samples",
                "avg_hellinger","avg_hellinger_live","avg_h2","avg_h2_live","avg_cemetery_h2"],
               avg_rows)
    _write_csv(cfg.results_dir / "std.csv",
               ["model","game","sigma","n_samples",
                "std_hellinger","std_hellinger_live","std_h2_live"],
               std_rows)
    _write_csv(cfg.results_dir / "per_image.csv",
               ["model","game","sigma","sample_id","class_name","image_name",
                "hellinger_distance","hellinger_live","h_squared","h2_live","cemetery_h2"],
               per_image_rows)

    meta_path = cfg.results_dir / "run_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["n_cached_rows"] = len(per_image_rows)
    meta["csv_files"] = {"avg": "avg.csv", "std": "std.csv", "per_image": "per_image.csv"}
    meta_path.write_text(json.dumps(meta, indent=2))


def run(cfg: RunConfig) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("hellinger_input_noise")
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Same image loader as the randomization run.
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
        "sigmas": cfg.sigmas,
        "image_size": cfg.image_size,
        "batch_size": cfg.batch_size,
        "device": str(device),
        "seed": cfg.seed,
        "samples": sample_records,
    }
    (cfg.results_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    for model_name in cfg.models:
        log.info("Building %s on %s", model_name, device)
        model = _build_model(model_name, device)

        t0 = time.perf_counter()
        targets_cpu = _predict_targets(model, images_cpu, cfg.batch_size, device)
        log.info(
            "%s: computed %d targets in %.1fs",
            model_name, len(targets_cpu), time.perf_counter() - t0,
        )

        for si, sigma in enumerate(cfg.sigmas):
            # Per (model, sigma) reproducible noise seed.
            noise_seed = int(cfg.seed) + 1000 * (
                zlib.crc32(f"{model_name}:{float(sigma):.6f}".encode()) % 10_000)
            for game in cfg.games:
                cache_path = _sigma_cache_path(cfg, model_name, game, sigma)
                if (cache_path.exists() and not cfg.overwrite
                        and _cache_matches(cache_path, sample_records)):
                    log.info("%s %s sigma=%.3f: cache hit %s", model_name, game, sigma, cache_path.name)
                    continue
                t0 = time.perf_counter()
                h, h2, cem, hl, h2l = _compute_distances_noise(
                    model, game, images_cpu, targets_cpu, sigma, noise_seed,
                    cfg.batch_size, device,
                )
                dt = time.perf_counter() - t0
                log.info(
                    "%s %s sigma=%.3f: %d images, mean H=%.4f, mean Hlive=%.4f (%.1fs)",
                    model_name, game, sigma, len(h),
                    float(np.nanmean(h)), float(np.nanmean(hl)), dt,
                )
                _write_sigma_cache(
                    cache_path, sample_records, model_name, game, sigma,
                    h, h2, cem, hl, h2l,
                )

        del model
        gc.collect()
        _empty_cuda_cache()

    _write_csvs(cfg, sample_records)
    log.info("wrote aggregate CSVs to %s", cfg.results_dir)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--games", nargs="+", default=list(DEFAULT_GAMES), choices=["adf", "rg"])
    p.add_argument("--sigmas", nargs="+", type=float, default=list(DEFAULT_SIGMAS))
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--n-per-class-val", type=int, default=1)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--imagenet-root", type=str, default=None)
    p.add_argument("--imagenet-s-root", type=str, default=None)
    p.add_argument("--imagenet-s-split", type=str, default="ImageNetS50")
    p.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _parse_args(argv)
    cfg = RunConfig(
        max_images=args.max_images,
        models=args.models,
        games=args.games,
        sigmas=args.sigmas,
        results_dir=args.results_dir,
        imagenet_root=args.imagenet_root,
        imagenet_s_root=args.imagenet_s_root,
        imagenet_s_split=args.imagenet_s_split,
        split=args.split,
        n_per_class_val=args.n_per_class_val,
        image_size=args.image_size,
        batch_size=args.batch_size,
        device=args.device,
        overwrite=args.overwrite,
        seed=args.seed,
    )
    run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
