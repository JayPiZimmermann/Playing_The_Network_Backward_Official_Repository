#!/usr/bin/env python3
"""Reproduce the parameter-randomization sanity check and Hellinger
trajectory diagnostics on ImageNet-S (50 classes).

Covers four blocks from appendix/sanity_check.tex:

  1. Adebayo 2018 cascading heatmap-similarity sanity check
     (experiments.sanity_check_randomization.run).
  2. Cascading Hellinger trajectory randomization
     (experiments.hellinger_randomization.run).
  3. Double-random Hellinger baseline (R=500 rounds)
     (experiments.hellinger_randomization.run_double_random).
  4. Hellinger under input perturbations:
       - clean-pair Hellinger calibration
         (experiments.hellinger_input_noise.run_clean_pairs)
       - clean-vs-random and random-vs-random reference rows
         (experiments.hellinger_input_noise.run_random_baselines)
       - additive Gaussian noise sweep
         (experiments.hellinger_input_noise.run)

Outputs land under each experiment's own ``results/`` dir; this script
is just a thin orchestrator that pins seed=42 and the published
ImageNet-S 50-class test split for every block.

Usage::

    python -m reproduce.hellinger_sanity --block all
    python -m reproduce.hellinger_sanity --block hellinger
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
SEED = 42
ARCHS = ("vgg16", "resnet50", "vit_b_16")


def _python(module: str, extra: List[str]) -> None:
    cmd = [sys.executable, "-m", module] + extra
    logger.info("RUN  %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO)


def _common_args(args) -> List[str]:
    out = ["--seed", str(SEED)]
    if args.imagenet_root:
        out += ["--imagenet-root", args.imagenet_root]
    if args.imagenet_s_root:
        out += ["--imagenet-s-root", args.imagenet_s_root]
    return out


def _archs(args) -> tuple:
    return ARCHS[:1] if args.smoke_test else ARCHS


def block_sanity(args) -> None:
    """Adebayo 2018 cascading heatmap-similarity test on the test split."""
    extra = _common_args(args) + [
        "--models", *_archs(args),
        "--split", "test",
        # The configurations reported in the paper (validation selection).
        "--from-selection", str(REPO / "results" / "selection" / "{model}" / "selected.json"),
    ]
    if args.smoke_test:
        extra += ["--max-images", "1", "--max-steps", "2"]
    _python("experiments.sanity_check_randomization.run", extra)


def block_hellinger(args) -> None:
    """Cascading Hellinger trajectory randomization."""
    extra = _common_args(args) + [
        "--models", *_archs(args),
        "--games", "rg", "adf",
    ]
    if args.smoke_test:
        extra += ["--max-images", "1", "--max-steps", "2"]
    _python("experiments.hellinger_randomization.run", extra)


def block_double_random(args) -> None:
    """R=500 round double-random Hellinger baseline."""
    extra = _common_args(args) + [
        "--models", *_archs(args),
        "--num-rounds", "2" if args.smoke_test else "500",
    ]
    _python("experiments.hellinger_randomization.run_double_random", extra)


def block_input_noise(args) -> None:
    """Hellinger under additive Gaussian input noise + clean-pair calibration."""
    common = _common_args(args) + [
        "--models", *_archs(args),
        "--games", "rg", "adf",
        "--split", "test",
    ]
    pair_count = "2" if args.smoke_test else "200"
    # run_clean_pairs needs both intraclass and interclass pairs. The test
    # split is ordered class-by-class, so a small max-images window stays
    # inside a single class. Use 20 (>1 class worth of test images) so both
    # branches have a viable sampling pool.
    pair_common = common + (["--max-images", "20"] if args.smoke_test else [])
    _python("experiments.hellinger_input_noise.run_clean_pairs",
            pair_common + ["--num-pairs", pair_count])
    _python("experiments.hellinger_input_noise.run_random_baselines",
            common + (["--max-images", "2", "--rounds-rand", "2"] if args.smoke_test else []))
    if args.smoke_test:
        common += ["--max-images", "1"]
    _python("experiments.hellinger_input_noise.run", common)


BLOCKS = {
    "sanity": block_sanity,
    "hellinger": block_hellinger,
    "double-random": block_double_random,
    "input-noise": block_input_noise,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", choices=tuple(BLOCKS) + ("all",), default="all")
    ap.add_argument("--imagenet-root", type=str, default=None)
    ap.add_argument("--imagenet-s-root", type=str, default=None)
    ap.add_argument("--smoke-test", action="store_true",
                    help="Plumbing-only mode: 1 architecture, 1 image, 2 cascade "
                         "steps, 2 double-random rounds, 2 input-noise pairs.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    blocks = list(BLOCKS) if args.block == "all" else [args.block]
    for b in blocks:
        logger.info("=== Hellinger reproduction block: %s ===", b)
        BLOCKS[b](args)


if __name__ == "__main__":
    main()
