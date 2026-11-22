#!/usr/bin/env python3
"""Copy a compact summary of the test results into ``results_summary/``.

For every evaluated setting (ImageNet-S test split per model, Pascal VOC
test split) this writes

* the per-row result JSONs (means, runtimes, parameters, selection roles),
* ``per_image_scores.csv.gz``: one line per (row, image) with the scalar
  per-image scores of every metric (the PF / Sel curves are omitted),

which is enough to recompute every table entry, confidence interval and
paired test, and for every validation selection (ImageNet-S per model,
Pascal VOC) ``selected.json``, ``val_scores.csv`` and
``selection_summary.md``.  Usage::

    python tools/export_results_summary.py --results results --out results_summary
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import shutil
from pathlib import Path

import numpy as np

SETTINGS = {"test/vit_b16": "test/vit_b16", "test/vgg16": "test/vgg16",
            "test/resnet50": "test/resnet50", "voc/test": "voc/test"}
SELECTIONS = {"selection/vit_b16": "selection/vit_b16", "selection/vgg16": "selection/vgg16",
              "selection/resnet50": "selection/resnet50", "voc/selection": "voc/selection"}
SELECTION_FILES = ("selected.json", "val_scores.csv", "selection_summary.md")


def export(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    rows = sorted(p for p in src.glob("*.json") if not p.name.endswith(".incomplete.json"))
    for p in rows:
        shutil.copy2(p, dst / p.name)
    records = []
    for p in rows:
        npz = src / "per_image" / f"{p.stem}.npz"
        if not npz.exists():
            continue
        with np.load(npz) as d:
            ids = d["image_ids"].astype(str)
            metrics = [k for k in d.files if d[k].shape == ids.shape and "__" not in k
                       and k not in ("image_ids", "map_failed", "map_is_zero")]
            cols = {k: d[k] for k in metrics}
        for i, img in enumerate(ids):
            records.append({"row_label": p.stem, "image_id": img,
                            **{k: ("" if not np.isfinite(v[i]) else f"{float(v[i]):.6g}")
                               for k, v in cols.items()}})
    fields = ["row_label", "image_id"] + sorted({k for r in records for k in r} - {"row_label", "image_id"})
    # mtime=0: unchanged scores give a byte-identical file
    with open(dst / "per_image_scores.csv.gz", "wb") as raw, \
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as gz, \
            io.TextIOWrapper(gz, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)
    return len(rows)


def export_selection(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    files = [src / name for name in SELECTION_FILES if (src / name).exists()]
    for p in files:
        shutil.copy2(p, dst / p.name)
    return len(files)


def export_runtime(src: Path, dst: Path, test_root: Path) -> int:
    """``results/runtime/<model>.csv`` (runtime_benchmark output, the t/img column).

    Only the rows that have a test result (``<test_root>/<model>/<row>.json``) are
    copied; the benchmark may contain rows of an earlier selection.
    """
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted(src.glob("*.csv"))
    for p in files:
        model = p.stem
        with open(p) as f:
            lines = f.read().splitlines()
        header = [ln for ln in lines if ln.startswith("#")]
        body = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
        kept = [body[0]] + [ln for ln in body[1:]
                            if (test_root / model / f"{ln.split(',')[1]}.json").exists()]
        with open(dst / p.name, "w") as f:
            f.write("\n".join(header + kept) + "\n")
    return len(files)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results_summary"))
    args = ap.parse_args(argv)
    for src, dst in SETTINGS.items():
        if (args.results / src).exists():
            n = export(args.results / src, args.out / dst)
            print(f"{src}: {n} rows -> {args.out / dst}")
    for src, dst in SELECTIONS.items():
        if (args.results / src).exists():
            n = export_selection(args.results / src, args.out / dst)
            print(f"{src}: {n} files -> {args.out / dst}")
    if (args.results / "runtime").exists():
        n = export_runtime(args.results / "runtime", args.out / "runtime", args.results / "test")
        print(f"runtime: {n} files -> {args.out / 'runtime'}")


if __name__ == "__main__":
    main()
