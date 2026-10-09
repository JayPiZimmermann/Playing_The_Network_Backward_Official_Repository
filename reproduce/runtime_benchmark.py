#!/usr/bin/env python3
"""Single-GPU per-image runtime of the selected rows.

Usage::

    python -m reproduce.runtime_benchmark --model vit_b16 \\
        --selection results/selection/vit_b16/selected.json \\
        --n-images 50 --warmup 5 --out results/runtime/vit_b16.csv

For every distinct row of ``selected.json`` (plus optional ``--extra-rows``)
the attribution map of each of the first ``--n-images`` test images is
computed with batch size 1 exactly as in the evaluation
(``Evaluator._generate_maps``: same model preparation, same wrappers);
``--warmup`` untimed calls precede the timed loop and CUDA is synchronised
before and after every timed call.  No metrics are computed.

Output CSV (lines starting with ``#`` hold the GPU name and the torch /
CUDA versions): method, row_label, n_images, median_s, q25_s, q75_s,
iqr_s, mean_s, failures.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from reproduce.common import (
    MODELS, SEED, add_data_args, build_split, check_model, get_spec, gpu_info, load_json,
    seed_everything, setup_logging, validate_params,
)
from reproduce.evaluate_selected import collect_rows

logger = logging.getLogger("reproduce.runtime_benchmark")


def read_runtime_csv(path: Path):
    """Parse a CSV written by this script -> (header dict, {row_label: row dict})."""
    header, rows = {}, {}
    path = Path(path)
    if not path.exists():
        return header, rows
    with open(path) as f:
        lines = f.read().splitlines()
    body = []
    for ln in lines:
        if ln.startswith("#"):
            if ":" in ln:
                k, v = ln[1:].split(":", 1)
                header[k.strip()] = v.strip()
        elif ln.strip():
            body.append(ln)
    for r in csv.DictReader(body):
        rows[r["row_label"]] = r
    return header, rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Per-image runtime benchmark of the selected rows.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--extra-rows", type=Path, default=None)
    ap.add_argument("--rows", type=str, default=None, help="Comma-separated row labels.")
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--n-images", type=int, default=50)
    ap.add_argument("--out", type=Path, required=True)
    add_data_args(ap)
    ap.set_defaults(warmup=5)
    args = ap.parse_args(argv)
    check_model(args.model)
    setup_logging()

    import numpy as np
    import torch
    from bwgame.evaluation.evaluator import Evaluator
    from bwgame.evaluation.run_eval import load_model

    selection = load_json(args.selection)
    extra = load_json(args.extra_rows) if args.extra_rows else None
    rows = collect_rows(selection, extra)
    if args.rows:
        wanted = {s.strip() for s in args.rows.split(",") if s.strip()}
        rows = [r for r in rows if r["row_label"] in wanted]

    args.max_images = args.n_images  # first N images of the split
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(MODELS[args.model], device)
    data = build_split("imagenet-s", args.split, args)
    work = args.out.parent / f".runtime_work_{args.model}"
    ev = Evaluator(model=model, dataset=data, device=device, output_dir=work, metric_configs=[],
                   batch_size=1, softmax=True, seed=SEED, save_heatmaps=False, cache_maps=False,
                   warmup=args.warmup)
    info = gpu_info()

    out_rows = []
    for i, r in enumerate(rows):
        spec = get_spec(r["method"])
        logger.info("[%d/%d] %s", i + 1, len(rows), r["row_label"])
        validate_params(ev.model, device, spec, r["params"])
        seed_everything(SEED)
        res = ev._generate_maps(spec.registry, dict(r["params"]), spec.component,
                                list(spec.layers) if spec.layers else None)
        times = np.asarray(res[5], dtype=float)
        n_failed = int(res[7])
        t = times[np.isfinite(times)]
        q25, med, q75 = (np.percentile(t, [25, 50, 75]) if t.size else (np.nan,) * 3)
        out_rows.append({"method": spec.key, "row_label": r["row_label"], "n_images": int(t.size),
                         "median_s": f"{med:.5f}", "q25_s": f"{q25:.5f}", "q75_s": f"{q75:.5f}",
                         "iqr_s": f"{q75 - q25:.5f}", "mean_s": f"{np.mean(t) if t.size else np.nan:.5f}",
                         "failures": n_failed})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["method", "row_label", "n_images", "median_s", "q25_s", "q75_s", "iqr_s", "mean_s", "failures"]
    with open(args.out, "w", newline="") as f:
        f.write(f"# gpu: {info['gpu']}\n# torch: {info['torch']}\n# cuda: {info['cuda']}\n")
        f.write(f"# warmup: {args.warmup}\n# batch_size: 1\n# split: {args.split}\n")
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    sys.exit(main())
