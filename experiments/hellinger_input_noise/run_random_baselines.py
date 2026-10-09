"""Random-input reference baselines for the input-noise Hellinger experiment.

Two rows per (model, game):

  * **clean-vs-random**: for each ImageNet-S test image ``x``
    (``reproduce.common.imagenet_s_split("test")``, 565 images) we draw one
    fresh standard-Gaussian tensor ``r`` (same shape) and compute
    ``H( Pi_A(x) , Pi_A(r) )`` with the argmax target of ``x`` on both
    sides. Mean and per-image std over the test images.
  * **random-vs-random**: we draw ``R`` independent standard-Gaussian
    pairs ``(r1, r2)`` and compute ``H( Pi_A(r1), Pi_A(r2) )`` with the
    argmax target of ``r1`` on both sides. Round-to-round mean and std.

Both rows serve as a sigma-to-infinity reference for
``experiments/hellinger_input_noise/run.py``: they bound the saturation
that the noise sweep approaches and give a calibrated ``fully
independent input'' comparison point per architecture.

Output: ``results/random_baselines.csv`` (one row per model x game x
baseline-kind).
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import sys
import time
import zlib
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from experiments.hellinger_randomization.run import (
    _build_model, _load_samples, _iter_batches, _predict_targets,
    _empty_cuda_cache,
)
from experiments.hellinger_input_noise.run import RunConfig

from bwgame.xai.methods.hellinger import HellingerDiagnostic


DEFAULT_MODELS: Tuple[str, ...] = ("vgg16", "resnet50", "vit_b_16")
DEFAULT_GAMES: Tuple[str, ...] = ("rg", "adf")

_HERE = Path(__file__).resolve().parent
_DEFAULT_RESULTS_DIR = _HERE / "results"


def _compute_pair_hellinger(
    model,
    game: str,
    input_A: torch.Tensor,
    input_B: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    batch_size: int,
):
    diag = HellingerDiagnostic(model, model_B=model, game=game, tau=1.0)
    live_diag = HellingerDiagnostic(
        model, model_B=model, game=game, tau=1.0, condition_on_survival=True,
    )
    n = input_A.shape[0]
    h = np.empty(n, dtype=np.float32)
    hl = np.empty(n, dtype=np.float32)
    log = logging.getLogger("hellinger_input_noise")
    try:
        for start, end, batch_a, batch_t in _iter_batches(input_A, target, batch_size):
            a_dev = batch_a.to(device, non_blocking=True)
            b_dev = input_B[start:end].to(device, non_blocking=True)
            t_dev = batch_t.to(device, non_blocking=True)
            r = diag.compute(a_dev, b_dev, target=t_dev)
            try:
                rlive = live_diag.compute(a_dev, b_dev, target=t_dev)
            except ValueError as exc:
                log.warning("live-conditioned skipped %s %d:%d: %s", game, start, end, exc)
                rlive = None
                hl[start:end] = np.nan

            def _np(v):
                return (v.detach().cpu().numpy().reshape(-1)
                        if isinstance(v, torch.Tensor) else np.asarray(v).reshape(-1))

            h[start:end] = _np(r["hellinger_distance"]).astype(np.float32, copy=False)
            if rlive is not None:
                hl[start:end] = _np(rlive["hellinger_distance"]).astype(np.float32, copy=False)
    finally:
        del diag
        del live_diag
        gc.collect()
        _empty_cuda_cache()
    return h, hl


def run(cfg: RunConfig, rounds_rand: int, max_rand_images: int) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("hellinger_input_noise")
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    images_cpu, sample_records = _load_samples(cfg)
    if device.type == "cuda":
        images_cpu = images_cpu.pin_memory()
    n_test = images_cpu.shape[0]
    log.info("Loaded %d test images", n_test)

    # Per-image ImageNet standard deviation proxy for the random draws:
    # clean images in ImageNet-normalized space have per-channel
    # sample std close to 1, so standard-Gaussian noise is the natural
    # ``random input'' on the same scale.
    rows: list[dict] = []

    for model_name in cfg.models:
        log.info("Building %s on %s", model_name, device)
        model = _build_model(model_name, device)
        # Target class distributions
        targets_clean = _predict_targets(model, images_cpu, cfg.batch_size, device)

        for game in cfg.games:
            # --- Row 1: clean vs random ---
            g_cpu = torch.Generator(device="cpu").manual_seed(cfg.seed + zlib.crc32(f"{model_name}:{game}:cvr".encode()) % 10_000)
            rand_imgs = torch.randn(images_cpu.shape, generator=g_cpu, dtype=images_cpu.dtype)
            t0 = time.perf_counter()
            h_cvr, hl_cvr = _compute_pair_hellinger(
                model, game, images_cpu, rand_imgs, targets_clean,
                device, cfg.batch_size,
            )
            rows.append({
                "model": model_name, "game": game, "baseline": "clean_vs_random",
                "n": len(h_cvr),
                "mean_H": float(np.nanmean(h_cvr)),
                "std_H": float(np.nanstd(h_cvr)),
                "mean_Hlive": float(np.nanmean(hl_cvr)),
                "std_Hlive": float(np.nanstd(hl_cvr)),
            })
            log.info(
                "%s %s clean-vs-rand: n=%d H=%.3f±%.3f Hlive=%.3f±%.3f (%.1fs)",
                model_name, game, len(h_cvr),
                rows[-1]["mean_H"], rows[-1]["std_H"],
                rows[-1]["mean_Hlive"], rows[-1]["std_Hlive"],
                time.perf_counter() - t0,
            )

            # --- Row 2: random vs random ---
            R = min(rounds_rand, n_test) if max_rand_images < 0 else min(rounds_rand, max_rand_images)
            g1 = torch.Generator(device="cpu").manual_seed(cfg.seed + zlib.crc32(f"{model_name}:{game}:rvr_a".encode()) % 10_000)
            g2 = torch.Generator(device="cpu").manual_seed(cfg.seed + zlib.crc32(f"{model_name}:{game}:rvr_b".encode()) % 10_000)
            shape_R = (R,) + tuple(images_cpu.shape[1:])
            rand_A = torch.randn(shape_R, generator=g1, dtype=images_cpu.dtype)
            rand_B = torch.randn(shape_R, generator=g2, dtype=images_cpu.dtype)
            # Argmax target per random pair: use argmax of the A side.
            targets_rand = _predict_targets(model, rand_A, cfg.batch_size, device)
            t0 = time.perf_counter()
            h_rvr, hl_rvr = _compute_pair_hellinger(
                model, game, rand_A, rand_B, targets_rand,
                device, cfg.batch_size,
            )
            rows.append({
                "model": model_name, "game": game, "baseline": "random_vs_random",
                "n": len(h_rvr),
                "mean_H": float(np.nanmean(h_rvr)),
                "std_H": float(np.nanstd(h_rvr)),
                "mean_Hlive": float(np.nanmean(hl_rvr)),
                "std_Hlive": float(np.nanstd(hl_rvr)),
            })
            log.info(
                "%s %s rand-vs-rand: n=%d H=%.3f±%.3f Hlive=%.3f±%.3f (%.1fs)",
                model_name, game, len(h_rvr),
                rows[-1]["mean_H"], rows[-1]["std_H"],
                rows[-1]["mean_Hlive"], rows[-1]["std_Hlive"],
                time.perf_counter() - t0,
            )

        del model
        gc.collect()
        _empty_cuda_cache()

    out = cfg.results_dir / "random_baselines.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "model","game","baseline","n","mean_H","std_H","mean_Hlive","std_Hlive",
        ])
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %s (%d rows)", out, len(rows))


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    p.add_argument("--games", nargs="+", default=list(DEFAULT_GAMES), choices=["adf","rg"])
    p.add_argument("--max-images", type=int, default=-1)
    p.add_argument("--rounds-rand", type=int, default=200,
                   help="number of (r1, r2) rounds for random-vs-random")
    p.add_argument("--max-rand-images", type=int, default=-1)
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
        sigmas=[],  # unused by this script
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
    run(cfg, rounds_rand=args.rounds_rand, max_rand_images=args.max_rand_images)
    return 0


if __name__ == "__main__":
    sys.exit(main())
