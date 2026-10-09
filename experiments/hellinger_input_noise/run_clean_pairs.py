"""Clean-vs-clean Hellinger pair baselines.

Two rows per (model, game):

  * **intraclass**: ``R`` pairs $(x_1, x_2)$ sampled uniformly from images
    sharing the same ground-truth ImageNet-S class label. Compute
    $\\mathrm{H}(\\Pi_A(x_1), \\Pi_A(x_2))$ with the argmax target of
    $x_1$ on both sides. Round-to-round mean and std.
  * **interclass**: ``R`` pairs with distinct ground-truth class labels;
    same computation.

Paired with ``run_random_baselines.py``, these supply four reference
distances per (model, game):
clean-intra, clean-inter, clean-vs-random, random-vs-random. Exhaustive
exhaustive pair enumeration over the test split is unfeasible; we sample
``R=200`` pairs per cell. All pairs are drawn with replacement from the
ImageNet-S test split used throughout the paper
(``reproduce.common.imagenet_s_split("test")``, 565 images).
"""

from __future__ import annotations

import argparse
import csv
import gc
import logging
import random
import sys
import time
import zlib
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from experiments.hellinger_randomization.run import (
    _build_model, _load_samples, _predict_targets, _empty_cuda_cache,
)
from experiments.hellinger_input_noise.run import RunConfig
from experiments.hellinger_input_noise.run_random_baselines import _compute_pair_hellinger


DEFAULT_MODELS: Tuple[str, ...] = ("vgg16", "resnet50", "vit_b_16")
DEFAULT_GAMES: Tuple[str, ...] = ("rg", "adf")

_HERE = Path(__file__).resolve().parent
_DEFAULT_RESULTS_DIR = _HERE / "results"


def _sample_clean_pairs(
    sample_records: List[dict],
    num_pairs: int,
    intraclass: bool,
    seed: int,
) -> List[Tuple[int, int]]:
    """Sample ``num_pairs`` index pairs ``(i, j)`` with i != j.

    If ``intraclass`` is True, both indices share the same
    ``class_name``; otherwise they have distinct class names.
    """
    rng = random.Random(seed)
    # Group image indices by class.
    by_class: dict[str, List[int]] = {}
    for rec in sample_records:
        by_class.setdefault(rec["class_name"], []).append(rec["index"])
    classes = list(by_class.keys())

    pairs: List[Tuple[int, int]] = []
    max_attempts = 50 * num_pairs
    attempts = 0
    while len(pairs) < num_pairs and attempts < max_attempts:
        attempts += 1
        if intraclass:
            # Pick a class with at least 2 images.
            candidates = [c for c in classes if len(by_class[c]) >= 2]
            if not candidates:
                break
            c = rng.choice(candidates)
            i, j = rng.sample(by_class[c], 2)
        else:
            c1, c2 = rng.sample(classes, 2)
            i = rng.choice(by_class[c1])
            j = rng.choice(by_class[c2])
        pairs.append((i, j))
    if len(pairs) < num_pairs:
        raise RuntimeError(
            f"Could not draw {num_pairs} {'intraclass' if intraclass else 'interclass'} "
            f"pairs in {max_attempts} attempts"
        )
    return pairs


def run(cfg: RunConfig, num_pairs: int) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("hellinger_input_noise")
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    images_cpu, sample_records = _load_samples(cfg)
    if device.type == "cuda":
        images_cpu = images_cpu.pin_memory()
    log.info("Loaded %d test images", images_cpu.shape[0])

    # Classes
    n_classes = len({r["class_name"] for r in sample_records})
    log.info("Spanning %d distinct classes", n_classes)

    rows: List[dict] = []
    for model_name in cfg.models:
        log.info("Building %s on %s", model_name, device)
        model = _build_model(model_name, device)
        targets_clean = _predict_targets(model, images_cpu, cfg.batch_size, device)

        for game in cfg.games:
            for kind in ("intraclass", "interclass"):
                seed_kind = cfg.seed + zlib.crc32(f"{model_name}:{game}:{kind}".encode()) % 10_000
                pairs = _sample_clean_pairs(
                    sample_records, num_pairs,
                    intraclass=(kind == "intraclass"),
                    seed=seed_kind,
                )
                idx_A = torch.tensor([p[0] for p in pairs], dtype=torch.long)
                idx_B = torch.tensor([p[1] for p in pairs], dtype=torch.long)
                inp_A = images_cpu[idx_A].contiguous()
                inp_B = images_cpu[idx_B].contiguous()
                targets_A = targets_clean[idx_A]

                t0 = time.perf_counter()
                h, hl = _compute_pair_hellinger(
                    model, game, inp_A, inp_B, targets_A, device, cfg.batch_size,
                )
                rows.append({
                    "model": model_name, "game": game, "baseline": f"clean_{kind}",
                    "n": len(h),
                    "mean_H": float(np.nanmean(h)),
                    "std_H": float(np.nanstd(h)),
                    "mean_Hlive": float(np.nanmean(hl)),
                    "std_Hlive": float(np.nanstd(hl)),
                })
                log.info(
                    "%s %s clean-%s: n=%d H=%.3f±%.3f Hlive=%.3f±%.3f (%.1fs)",
                    model_name, game, kind, len(h),
                    rows[-1]["mean_H"], rows[-1]["std_H"],
                    rows[-1]["mean_Hlive"], rows[-1]["std_Hlive"],
                    time.perf_counter() - t0,
                )

        del model
        gc.collect()
        _empty_cuda_cache()

    out = cfg.results_dir / "clean_pair_baselines.csv"
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
    p.add_argument("--num-pairs", type=int, default=200)
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
        sigmas=[],
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
    run(cfg, num_pairs=args.num_pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
