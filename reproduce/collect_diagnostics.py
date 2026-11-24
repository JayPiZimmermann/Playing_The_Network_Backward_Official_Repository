#!/usr/bin/env python3
"""Collect the recorded diagnostic measurements into ``results_summary/diagnostics/``.

The randomisation, Hellinger and pointing-game granularity tables are computed from
the CSV files written by the ``experiments/`` runs and by ``reproduce.pg_granularity``.
This script copies the measurements those tables are computed from, gzip-compressed,
into the compact results archive, so that

    python -m reproduce.diagnostic_figures --out tables
    python -m reproduce.pg_granularity --results results_summary --out tables

recompute the tables, the macros and the paired tests without a GPU, the dataset
or the cached attribution maps.

Collected files (source -> ``results_summary/diagnostics/``):

* ``sanity_check_randomization/results/{avg,per_image}.csv``: the per-image file is
  reduced to the two reported similarity metrics at the cascade steps the tables
  use (full randomisation and the step aligned with the Hellinger cascade);
* ``hellinger_randomization/results/{avg,per_image}.csv``;
* ``hellinger_randomization/results_double_random/{aggregate,per_round}.csv``;
* ``hellinger_input_noise/results/{avg,clean_pair_baselines,random_baselines}.csv``;
* ``<results>/analysis/pg_granularity_per_image.csv.gz``.

Missing sources are reported and skipped; ``--strict`` turns them into an error.
No torch needed.

Usage::

    python -m reproduce.collect_diagnostics [--results results] [--out results_summary/diagnostics]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from reproduce.common import REPO, setup_logging
from reproduce.diagnostic_figures import ALIGNED_STEPS, EXP
from reproduce.pg_granularity import PER_IMAGE_FILE

logger = logging.getLogger("reproduce.collect_diagnostics")

SANITY_METRICS = ("spearman_abs", "hog_pearson")
PLAIN = {
    "hellinger_randomization": ("hellinger_randomization/results", ("avg.csv", "per_image.csv")),
    "double_random": ("hellinger_randomization/results_double_random", ("aggregate.csv", "per_round.csv")),
    "hellinger_input_noise": ("hellinger_input_noise/results",
                              ("avg.csv", "clean_pair_baselines.csv", "random_baselines.csv")),
}


def _gzip_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the archive bytes reproducible for unchanged measurements
    with open(src, "rb") as f, open(dst, "wb") as raw, gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0) as g:
        shutil.copyfileobj(f, g)


def _write_frame(df: pd.DataFrame, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as g:
        g.write(df.to_csv(index=False).encode())


def reduce_sanity_per_image(per: pd.DataFrame) -> pd.DataFrame:
    """Rows of the map-similarity cascade that the tables are computed from."""
    per = per[per.metric.isin(SANITY_METRICS)]
    keep = pd.Series(False, index=per.index)
    for model, grp in per.groupby("model"):
        aligned = ALIGNED_STEPS.get(model, (None, None))[1]
        keep.loc[grp.index] = (grp.step_idx == grp.step_idx.max()) | (grp.last_randomized_layer == aligned)
    return per[keep]


def collect(results: Path, out: Path, exp: Path = EXP) -> List[str]:
    """Copy the measurements; returns the list of missing sources."""
    missing: List[str] = []
    written: Dict[str, Dict] = {}

    def record(dst: Path, src: Path) -> None:
        # the source is recorded relative to its root: absolute paths are machine-specific
        root = exp if src.is_relative_to(exp) else results
        written[str(dst.relative_to(out))] = {
            "source": str(src.relative_to(root)), "sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
            "bytes": dst.stat().st_size}
        logger.info("wrote %s", dst)

    def source(path: Path) -> Optional[Path]:
        if path.exists():
            return path
        missing.append(str(path))
        logger.warning("missing %s", path)
        return None

    for name, (sub, files) in PLAIN.items():
        for fn in files:
            src = source(exp / sub / fn)
            if src is not None:
                _gzip_copy(src, out / name / (fn + ".gz"))
                record(out / name / (fn + ".gz"), src)

    sdir = exp / "sanity_check_randomization" / "results"
    src = source(sdir / "avg.csv")
    if src is not None:
        _gzip_copy(src, out / "sanity_check_randomization" / "avg.csv.gz")
        record(out / "sanity_check_randomization" / "avg.csv.gz", src)
    src = source(sdir / "per_image.csv")
    if src is not None:
        dst = out / "sanity_check_randomization" / "per_image.csv.gz"
        _write_frame(reduce_sanity_per_image(pd.read_csv(src)), dst)
        record(dst, src)

    src = source(results / "analysis" / PER_IMAGE_FILE)
    if src is not None:
        dst = out / PER_IMAGE_FILE
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        record(dst, src)

    if written:
        (out / "MANIFEST.json").write_text(json.dumps(written, indent=2, sort_keys=True) + "\n")
    return missing


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=REPO / "results",
                    help="Results directory holding analysis/pg_granularity_per_image.csv.gz.")
    ap.add_argument("--experiments", type=Path, default=EXP,
                    help="Directory of the experiment outputs (default: experiments/).")
    ap.add_argument("--out", type=Path, default=REPO / "results_summary" / "diagnostics")
    ap.add_argument("--strict", action="store_true", help="Fail if a source file is missing.")
    args = ap.parse_args(argv)
    setup_logging()
    missing = collect(args.results, args.out, args.experiments)
    if missing and args.strict:
        raise SystemExit(f"--strict: {len(missing)} source file(s) missing")


if __name__ == "__main__":
    main()
