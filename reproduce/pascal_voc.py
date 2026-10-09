#!/usr/bin/env python3
"""Pascal VOC 2012 transfer experiment (ViT-B/16).

Split: per-image disjoint random split of VOC2012 trainval (val:test = 1:5,
seed 42; ``bwgame.evaluation.dataset.split_voc_trainval``), one sample per
(image, foreground class) with the curated VOC -> ImageNet target mapping of
``data/pascal_voc.py``.

Stages (``--stage``):

  tune    evaluate every *baseline* row of the ViT-B/16 validation grid
          (``experiments/grids/imagenet_s_val_vit_b16.csv`` without RG /
          RG+Eq / SG) on the VOC val pool with AL, PG, TK, PF5, PF20, Sel.
          Shardable (``--shard i --num-shards n``), ``--resume`` skips done rows.
  select  loc / faith selection per baseline method with exactly the rules of
          ``reproduce.select_configs`` (same metric directions and tie-breaks)
          -> results/voc/selection/{val_scores.csv, selected.json, selection_summary.md}
  test    evaluate on the VOC test pool (a) the VOC-selected baseline rows and
          (b) the RG / RG+Eq rows (anchor, loc, faith) of the ImageNet-S
          ``selected.json`` without any re-tuning -> results/voc/test/.
          Shardable; MaxS / AvgS are skipped unless ``--with-robustness``.
  all     tune -> select -> test in one process.

Runtimes recorded in results/voc/test/*.json are measured on the VOC run
itself (evaluator timing), not copied from ImageNet-S.

Usage::

    python -m reproduce.pascal_voc --stage tune --shard 0 --num-shards 4 --resume
    python -m reproduce.pascal_voc --stage select
    python -m reproduce.pascal_voc --stage test \\
        --selection results/selection/vit_b16/selected.json --resume

Pascal VOC is downloaded with ``data/download_pascal_voc.sh`` if the default
``data/voc/VOCdevkit/VOC2012`` tree is missing.
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from reproduce.common import (
    OURS, REPO, add_data_args, display_path, load_json, read_grid, setup_logging, shard, write_grid,
)

logger = logging.getLogger("reproduce.pascal_voc")

MODEL = "vit_b16"
DEFAULT_GRID = REPO / "experiments" / "grids" / "imagenet_s_val_vit_b16.csv"
DEFAULT_VOC_ROOT = REPO / "data" / "voc" / "VOCdevkit" / "VOC2012"


def _ensure_voc(voc_root: Path) -> None:
    if (voc_root / "SegmentationClass").exists():
        return
    if voc_root.resolve() != DEFAULT_VOC_ROOT.resolve():
        raise SystemExit(f"Pascal VOC 2012 not found at {voc_root} (expected SegmentationClass/)")
    logger.info("Pascal VOC 2012 not found at %s -- running data/download_pascal_voc.sh", voc_root)
    subprocess.check_call(["bash", str(REPO / "data" / "download_pascal_voc.sh")])


def baseline_grid(grid_csv: Path):
    rows = [r for r in read_grid(grid_csv) if r.method not in OURS]
    for i, r in enumerate(rows):
        r.order = i
    return rows


def stage_tune(args) -> None:
    from reproduce.select_configs import evaluate_grid
    rows = baseline_grid(args.grid)
    write_grid(args.out / "selection" / "voc_val_grid.csv", rows)
    todo = shard(rows, args.shard, args.num_shards)
    logger.info("VOC tune: %d of %d baseline rows", len(todo), len(rows))
    failed = evaluate_grid(todo, model=MODEL, dataset="voc", split="val",
                           out=args.out / "selection", args=args, resume=args.resume)
    if failed:
        raise SystemExit(f"{len(failed)} VOC val rows failed: {failed}")


def stage_select(args) -> None:
    from reproduce.select_configs import aggregate
    rows = baseline_grid(args.grid)
    sel_dir = args.out / "selection"
    aggregate(rows, sel_dir / "rows", sel_dir,
              meta={"model": MODEL, "dataset": "voc", "split": "val", "grid": display_path(args.grid),
                    "note": "baselines only; RG / RG+Eq are not re-tuned on VOC"},
              allow_missing=args.allow_missing)


def test_rows(args):
    from reproduce.evaluate_selected import collect_rows
    voc_sel = load_json(args.out / "selection" / "selected.json")
    in_sel = load_json(args.selection)
    if in_sel.get("model") not in (None, MODEL):
        raise SystemExit(f"{args.selection} is for {in_sel.get('model')}, expected {MODEL}")
    ours = {"rows": [r for r in in_sel.get("rows", []) if r["method"] in ("rg", "rgeq")]}
    if not ours["rows"]:
        raise SystemExit(f"No RG / RG+Eq rows in {args.selection}")
    base = {"rows": [r for r in voc_sel.get("rows", []) if r["method"] not in OURS]}
    return collect_rows(base, None) + collect_rows(ours, None)


def stage_test(args) -> None:
    from reproduce.evaluate_selected import main as eval_main
    import json
    rows = test_rows(args)
    extra = args.out / "test" / "voc_test_rows.json"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text(json.dumps([{"method": r["method"], "row_label": r["row_label"],
                                  "params": r["params"], "is_anchor": r["is_anchor"],
                                  "tunable": r["tunable"], "selected_by": r["selected_by"]}
                                 for r in rows], indent=2))
    argv = ["--model", MODEL, "--dataset", "voc", "--split", "test", "--out", str(args.out / "test"),
            "--extra-rows", str(extra), "--voc-root", str(args.voc_root), "--warmup", str(args.warmup)]
    if not args.with_robustness:
        argv.append("--skip-robustness")
    if args.shard is not None:
        argv += ["--shard", str(args.shard), "--num-shards", str(args.num_shards)]
    for flag in ("resume", "cache_maps", "save_heatmaps"):
        if getattr(args, flag):
            argv.append("--" + flag.replace("_", "-"))
    if args.max_images:
        argv += ["--max-images", str(args.max_images)]
    if args.device:
        argv += ["--device", args.device]
    eval_main(argv)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Pascal VOC 2012 transfer experiment (ViT-B/16).")
    ap.add_argument("--stage", choices=("tune", "select", "test", "all"), default="all")
    ap.add_argument("--out", type=Path, default=REPO / "results" / "voc")
    ap.add_argument("--grid", type=Path, default=DEFAULT_GRID,
                    help="ViT validation grid; its baseline rows are tuned on VOC val.")
    ap.add_argument("--selection", type=Path,
                    default=REPO / "results" / "selection" / MODEL / "selected.json",
                    help="ImageNet-S selected.json whose RG / RG+Eq rows are evaluated unchanged.")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--allow-missing", action="store_true", help="(select) tolerate missing val rows")
    ap.add_argument("--with-robustness", action="store_true",
                    help="(test) also compute MaxS / AvgS (expensive; the paper skips them on VOC)")
    add_data_args(ap)
    args = ap.parse_args(argv)
    args.voc_root = Path(args.voc_root)
    setup_logging(args.out / "logs" / f"voc_{args.stage}.log")
    _ensure_voc(args.voc_root)

    if args.stage in ("tune", "all"):
        stage_tune(args)
    if args.stage in ("select", "all"):
        stage_select(args)
    if args.stage in ("test", "all"):
        stage_test(args)


if __name__ == "__main__":
    sys.exit(main())
