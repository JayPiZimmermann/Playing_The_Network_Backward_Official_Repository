#!/usr/bin/env python3
"""Single-process driver for the ImageNet-S tables (thin wrapper).

Runs, per model, the same stages as the sharded multi-GPU recipe in the
README, one after the other on one GPU:

    build_grids -> select_configs (+ --aggregate) -> evaluate_selected
    -> [ViT-B/16 only] sweep_rows -> evaluate_selected --extra-rows
    -> bootstrap_ci -> runtime_benchmark

and finally ``make_tables``.  Every stage is resumable (``--resume``), so the
command can simply be re-run after an interruption.  The configurations are
taken from the validation selection (selected.json).

Usage::

    python -m reproduce.main_tables_imagenet_s --models vit_b16,vgg16,resnet50
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from reproduce.common import MODELS, REPO, setup_logging

logger = logging.getLogger("reproduce.main_tables_imagenet_s")


def _run(module: str, *argv: str) -> None:
    cmd = [sys.executable, "-m", module, *argv]
    logger.info("RUN %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=REPO)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="vit_b16,vgg16,resnet50")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--tables-out", type=Path, default=Path("results/tables"))
    ap.add_argument("--skip-robustness", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args(argv)
    setup_logging()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in MODELS:
            raise SystemExit(f"unknown model {m!r}; choose from {sorted(MODELS)}")
    for m in models:
        grid = Path("experiments/grids") / f"imagenet_s_val_{m}.csv"
        sel_dir = args.results / "selection" / m
        test_dir = args.results / "test" / m
        if not (REPO / grid).exists():
            _run("reproduce.build_grids", "--model", m)
        _run("reproduce.select_configs", "--model", m, "--grid", str(grid), "--split", "val",
             "--out", str(sel_dir), "--resume")
        _run("reproduce.select_configs", "--model", m, "--grid", str(grid), "--split", "val",
             "--out", str(sel_dir), "--aggregate")
        common = ["--model", m, "--selection", str(sel_dir / "selected.json"), "--split", "test",
                  "--out", str(test_dir), "--resume", "--cache-maps"]
        # MaxS / AvgS on the fixed 200-image subset of the paper
        common += ["--skip-robustness"] if args.skip_robustness else ["--robustness-subset", "200"]
        _run("reproduce.evaluate_selected", *common)
        if m == "vit_b16":
            sweeps = args.results / "sweeps_vit.json"
            _run("reproduce.sweep_rows", "--model", m, "--selection", str(sel_dir / "selected.json"),
                 "--out", str(sweeps))
            _run("reproduce.evaluate_selected", *common, "--extra-rows", str(sweeps))
        _run("reproduce.bootstrap_ci", "--results", str(test_dir), "--n-boot", str(args.n_boot),
             "--seed", "0")
        _run("reproduce.runtime_benchmark", "--model", m, "--selection", str(sel_dir / "selected.json"),
             "--n-images", "50", "--warmup", "5", "--out", str(args.results / "runtime" / f"{m}.csv"))
    _run("reproduce.pg_granularity", "--results", str(args.results), "--out", str(args.tables_out))
    _run("reproduce.make_tables", "--results", str(args.results), "--out", str(args.tables_out),
         "--models", ",".join(models))


if __name__ == "__main__":
    main()
