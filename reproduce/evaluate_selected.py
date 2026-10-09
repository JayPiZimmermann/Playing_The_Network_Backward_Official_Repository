#!/usr/bin/env python3
"""Evaluate the validation-selected rows on the test split.

Usage::

    python -m reproduce.evaluate_selected --model vit_b16 \\
        --selection results/selection/vit_b16/selected.json --split test \\
        --out results/test/vit_b16 [--rows LABEL,...] [--extra-rows sweeps_vit.json] \\
        [--skip-robustness] [--shard i --num-shards n] [--resume] [--list-rows]

Rows: every distinct row of ``selected.json`` (fixed, anchor, loc, faith;
deduplicated by row label) plus the optional ``--extra-rows`` JSON (a list
of ``{"method", "row_label", "params"}`` objects, e.g. the trade-off sweeps
written by ``reproduce.sweep_rows``; an optional ``selected_by`` list is kept).
``--rows`` and ``--shard`` select a
subset (for array jobs); ``--list-rows`` prints the selected labels and exits.

Metrics: AL, PG, TK, PF5, PF20, Sel, MaxS, AvgS with the paper's Quantus
settings (``configs/metrics_main_table``); ``--skip-robustness`` drops MaxS /
AvgS.  ``--robustness-nr-samples`` overrides nr_samples (default: paper
value 50) and ``--robustness-subset N`` computes MaxS / AvgS only on a fixed
random subset of N test images (``--robustness-subset-seed``, default 0),
identical for all rows.

Outputs per row: ``<out>/<row_label>.json`` (means, stds, coverage, n_images,
config, selection criteria, per-image runtime statistics with warm-up
excluded and CUDA synchronised) and ``<out>/per_image/<row_label>.npz``
(image ids, per-image metric values, per-image seconds, map-failure flags).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from reproduce.common import (
    add_data_args, check_model, finite_stats, get_spec, load_json, load_per_image, merge_sweeps,
    row_label, setup_logging, shard, update_row_metadata, write_json_atomic,
)

logger = logging.getLogger("reproduce.evaluate_selected")

ALL_METRICS = ("AL", "PG", "TK", "PF5", "PF20", "Sel", "MaxS", "AvgS")
NON_ROBUST = ("AL", "PG", "TK", "PF5", "PF20", "Sel")
ROBUST = ("MaxS", "AvgS")


def collect_rows(selection: Optional[Dict[str, Any]], extra: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Distinct rows of selected.json + extra rows, merged by row label."""
    rows: Dict[str, Dict[str, Any]] = {}

    def _add(r: Dict[str, Any], roles: List[str], sweeps: List[Dict[str, Any]]) -> None:
        for k in ("method", "row_label", "params"):
            if k not in r:
                raise ValueError(f"Row {r} lacks key {k!r}")
        get_spec(r["method"])
        label = r["row_label"]
        expected = row_label(r["method"], r["params"])
        if expected != label:
            logger.warning("Row label %s differs from the canonical label %s of its params",
                           label, expected)
        if label in rows:
            if json.dumps(rows[label]["params"], sort_keys=True) != json.dumps(r["params"], sort_keys=True):
                raise ValueError(f"Row label {label} is used with two different parameter sets")
            e = rows[label]
        else:
            e = rows[label] = {"method": r["method"], "row_label": label, "params": r["params"],
                               "is_anchor": bool(r.get("is_anchor", False)),
                               "tunable": bool(r.get("tunable", True)),
                               "selected_by": [], "sweeps": []}
        e["is_anchor"] = e["is_anchor"] or bool(r.get("is_anchor", False))
        e["selected_by"] = sorted(set(e["selected_by"]) | set(roles))
        e["sweeps"] = merge_sweeps(e["sweeps"], sweeps)

    for r in (selection or {}).get("rows", []):
        _add(r, list(r.get("selected_by", [])), [])
    for r in extra or []:
        sw = r.get("sweep")
        roles = list(r.get("selected_by", []))   # e.g. roles carried over by pascal_voc
        if sw:
            roles.append(f"sweep:{sw['series']}")
        _add(r, roles or ["extra"], [sw] if sw else [])
    return list(rows.values())


def _complete(out: Path, label: str, metrics) -> bool:
    p = out / f"{label}.json"
    if not (p.exists() and (out / "per_image" / f"{label}.npz").exists()):
        return False
    try:
        return (set(metrics).issubset(load_json(p).get("metrics", {}).keys())
                and _incomplete(p) is None)
    except Exception:
        return False


def _merge_robustness_arrays(out: Path, label: str, sub_ids, sub_arrays: Dict[str, Any],
                             sub_metric_params: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """Write MaxS / AvgS per-image arrays (for ``sub_ids``) into the main row files."""
    import os
    import numpy as np
    from bwgame.evaluation.metric_spec import SHORT_TO_METRIC

    main = load_per_image(out, label)
    if main is None:
        raise RuntimeError(f"Main per-image file missing for {label}; run it with --skip-robustness first")
    pos = {str(i): k for k, i in enumerate(main["image_ids"].astype(str))}
    obj = load_json(out / f"{label}.json")
    for s in ROBUST:
        name = SHORT_TO_METRIC[s]
        full = np.full(len(main["image_ids"]), np.nan)
        for img, v in zip(np.asarray(sub_ids).astype(str), sub_arrays[name]):
            full[pos[img]] = v
        main[name] = full
        stats = finite_stats(full[[pos[str(i)] for i in np.asarray(sub_ids).astype(str)]])
        # coverage is over the images the robustness metrics are defined on
        obj["metrics"][s] = dict(stats, **({"subset": meta} if meta else {}))
    obj.setdefault("metric_params", {}).update(
        {s: v for s, v in sub_metric_params.items() if s in ROBUST})
    path = out / "per_image" / f"{label}.npz"
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **{k: np.asarray(v) for k, v in main.items()})
    os.replace(tmp, path)
    write_json_atomic(out / f"{label}.json", obj)


def _merge_robustness_subset(out: Path, sub_dir: Path, label: str, subset_meta: Dict[str, Any]) -> None:
    """Copy MaxS / AvgS of the subset run into the main per-image file and row JSON."""
    from bwgame.evaluation.metric_spec import SHORT_TO_METRIC
    sub = load_per_image(sub_dir, label)
    if sub is None:
        raise RuntimeError(f"Robustness per-image file missing for {label} in {sub_dir}")
    sub_obj = load_json(sub_dir / f"{label}.json")
    _merge_robustness_arrays(out, label, sub["image_ids"],
                             {SHORT_TO_METRIC[s]: sub[SHORT_TO_METRIC[s]] for s in ROBUST},
                             sub_obj.get("metric_params", {}), subset_meta)


def _chunk_dir(out: Path, k: int, n_chunks: int, args) -> Path:
    images = (f"sub{args.robustness_subset}s{args.robustness_subset_seed}"
              if args.robustness_subset else "all")
    return out / "robustness_chunks" / images / f"of{n_chunks:03d}" / f"c{k:03d}"


def _robustness_indices(n: int, args) -> Tuple[List[int], Optional[Dict[str, Any]]]:
    """Image positions of the robustness metrics (subset and/or chunk) and their metadata."""
    import numpy as np
    meta = None
    idx = list(range(n))
    if args.robustness_subset:
        if not (0 < args.robustness_subset <= n):
            raise SystemExit(f"--robustness-subset must be in [1, {n}]")
        idx = sorted(np.random.RandomState(args.robustness_subset_seed)
                     .choice(n, size=args.robustness_subset, replace=False).tolist())
        meta = {"n": args.robustness_subset, "seed": args.robustness_subset_seed, "of": n}
    if args.robustness_chunks:
        meta = dict(meta or {"n": n, "of": n}, chunks=args.robustness_chunks)
        if args.robustness_chunk is not None:
            idx = idx[args.robustness_chunk::args.robustness_chunks]
    return idx, meta


def _merge_robustness_chunks(out: Path, label: str, n_images: int, args) -> None:
    """Merge the K robustness chunks of a row into its main files (all chunks required)."""
    import numpy as np
    from bwgame.evaluation.metric_spec import SHORT_TO_METRIC
    K = args.robustness_chunks
    expected, meta = _robustness_indices(n_images, argparse.Namespace(**{**vars(args), "robustness_chunk": None}))
    ids, arrays, mparams = [], {SHORT_TO_METRIC[s]: [] for s in ROBUST}, {}
    for k in range(K):
        d = _chunk_dir(out, k, K, args)
        sub = load_per_image(d, label)
        if sub is None:
            raise RuntimeError(f"{label}: robustness chunk {k}/{K} missing in {d}")
        ids.extend(np.asarray(sub["image_ids"]).astype(str).tolist())
        for name in arrays:
            arrays[name].extend(np.asarray(sub[name], dtype=float).tolist())
        mparams = load_json(d / f"{label}.json").get("metric_params", mparams)
    if len(ids) != len(expected) or len(set(ids)) != len(ids):
        raise RuntimeError(f"{label}: chunks cover {len(set(ids))} distinct images, expected {len(expected)}")
    _merge_robustness_arrays(out, label, ids, arrays, mparams, meta)


def _image_chunk_dir(out: Path, k: int, n_chunks: int) -> Path:
    return out / "image_chunks" / f"of{n_chunks:03d}" / f"c{k:03d}"


def _image_chunk_modes(args, rows: List[Dict[str, Any]]) -> None:
    """--image-chunk k (compute the main metrics on images k, k+K, ...) or --merge-image-chunks."""
    import os
    import numpy as np
    from reproduce.common import RowRunner, build_split
    from bwgame.evaluation.metric_spec import SHORT_TO_METRIC
    K = args.image_chunks
    n = len(build_split(args.dataset, args.split, args))
    if not args.merge_image_chunks:
        k = args.image_chunk
        if not 0 <= k < K:
            raise SystemExit(f"--image-chunk must be in [0, {K})")
        d = _image_chunk_dir(args.out, k, K)
        setup_logging(d / "logs" / "evaluate.log")
        todo = [r for r in rows if not (args.resume and _complete(d, r["row_label"], NON_ROBUST))]
        if not todo:
            return
        runner = RowRunner(model_key=args.model, dataset=args.dataset, split=args.split,
                           out_dir=d, json_dir=d, metric_short=NON_ROBUST, args=args,
                           subset_indices=list(range(n))[k::K])
        failed = []
        for r in todo:
            try:
                runner.run(r)
            except Exception as e:
                logger.exception("Row %s, image chunk %d/%d failed: %s", r["row_label"], k, K, e)
                failed.append(r["row_label"])
        if failed:
            sys.exit(f"{len(failed)} image-chunk rows failed: {failed}")
        return

    setup_logging(args.out / "logs" / "merge_image_chunks.log")
    failed = []
    for r in rows:
        label = r["row_label"]
        try:
            parts = []
            for k in range(K):
                sub = load_per_image(_image_chunk_dir(args.out, k, K), label)
                if sub is None:
                    raise RuntimeError(f"image chunk {k}/{K} missing")
                parts.append((list(range(n))[k::K], sub))
            merged: Dict[str, Any] = {}
            for key, val in parts[0][1].items():
                arr = np.asarray(val)
                if arr.ndim == 0:                  # scalar entries (meta) are equal in all chunks
                    merged[key] = arr
                    continue
                full = np.empty((n,) + arr.shape[1:], dtype=arr.dtype)
                for idx, sub in parts:
                    full[idx] = np.asarray(sub[key])
                merged[key] = full
            if len(set(merged["image_ids"].astype(str))) != n:
                raise RuntimeError("image chunks do not cover every image exactly once")
            path = args.out / "per_image" / f"{label}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            with open(tmp, "wb") as f:
                np.savez_compressed(f, **merged)
            os.replace(tmp, path)
            obj = load_json(_image_chunk_dir(args.out, 0, K) / f"{label}.json")
            obj["n_images"] = n
            obj["metrics"] = {s: finite_stats(merged[SHORT_TO_METRIC[s]]) for s in obj["metrics"]}
            obj["map_failures"] = int(np.asarray(merged.get("map_failed", [])).sum())
            obj["zero_maps"] = int(np.asarray(merged.get("map_is_zero", [])).sum())
            t = np.asarray(merged.get("time_s", []), dtype=float)
            t = t[np.isfinite(t)]
            if t.size:
                obj["runtime_s"].update(n=int(t.size), mean=float(t.mean()), median=float(np.median(t)),
                                        q25=float(np.percentile(t, 25)), q75=float(np.percentile(t, 75)))
            obj["image_chunks"] = K
            jp = args.out / f"{label}.json"
            write_json_atomic(jp, obj)
            update_row_metadata(jp, r)
            problem = _incomplete(jp)
            if problem:
                jp.replace(jp.with_suffix(".incomplete.json"))
                raise RuntimeError(f"incomplete row ({problem}); moved to .incomplete.json")
            logger.info("%s: merged %d image chunks (%d images)", label, K, n)
        except Exception as e:
            logger.exception("Merging the image chunks of %s failed: %s", label, e)
            failed.append(label)
    if failed:
        sys.exit(f"{len(failed)} merges failed: {failed}")


def _incomplete(json_path: Path) -> Optional[str]:
    """Reason why a written test row is not complete, or None."""
    import math
    obj = json.loads(Path(json_path).read_text())
    if int(obj.get("map_failures", 0) or 0):
        return f"{obj['map_failures']} failed maps"
    if int(obj.get("zero_maps", 0) or 0):
        return f"{obj['zero_maps']} all-zero maps"
    for k, v in obj.get("metrics", {}).items():
        if v.get("coverage") is None or float(v["coverage"]) < 1.0 or v.get("mean") is None \
                or not math.isfinite(float(v["mean"])):
            return f"{k}: mean={v.get('mean')} coverage={v.get('coverage')}"
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate selected rows on the test split.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--selection", type=Path, default=None,
                    help="selected.json from reproduce.select_configs (omit to run --extra-rows only)")
    ap.add_argument("--split", default="test", choices=("test",))
    ap.add_argument("--dataset", default="imagenet-s", choices=("imagenet-s", "voc"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", type=str, default=None, help="Comma-separated row labels to evaluate.")
    ap.add_argument("--extra-rows", type=Path, default=None,
                    help="JSON list of {method, row_label, params} rows (e.g. sweeps).")
    ap.add_argument("--skip-robustness", action="store_true", help="Do not compute MaxS / AvgS.")
    ap.add_argument("--robustness-nr-samples", type=int, default=None,
                    help="Override nr_samples of MaxS / AvgS (default: paper value from the configs).")
    ap.add_argument("--robustness-subset", type=int, default=None,
                    help="Compute MaxS / AvgS on a fixed random subset of N test images only.")
    ap.add_argument("--robustness-subset-seed", type=int, default=0)
    ap.add_argument("--robustness-chunks", type=int, default=None,
                    help="Split the robustness images of each row into K chunks (with "
                         "--robustness-chunk k: compute chunk k only; with --merge-robustness: merge).")
    ap.add_argument("--robustness-chunk", type=int, default=None)
    ap.add_argument("--merge-robustness", action="store_true",
                    help="Merge the K robustness chunks into the main row files (the main row "
                         "must exist, e.g. from a --skip-robustness run).")
    ap.add_argument("--image-chunks", type=int, default=None,
                    help="Split the images of each row into K chunks for the main (non-robustness) "
                         "metrics (with --image-chunk k: compute chunk k only; with "
                         "--merge-image-chunks: merge them into the row files).")
    ap.add_argument("--image-chunk", type=int, default=None)
    ap.add_argument("--merge-image-chunks", action="store_true")
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--resume", action="store_true",
                    help="Skip rows whose result files already contain all requested metrics "
                         "(their selection metadata is still updated).")
    ap.add_argument("--list-rows", action="store_true", help="Print the row labels and exit.")
    add_data_args(ap)
    args = ap.parse_args(argv)
    check_model(args.model)

    selection = load_json(args.selection) if args.selection else None
    if selection is not None and selection.get("model") not in (None, args.model):
        raise SystemExit(f"{args.selection} is for model {selection.get('model')}, not {args.model}")
    extra = load_json(args.extra_rows) if args.extra_rows else None
    if extra is not None and not isinstance(extra, list):
        raise SystemExit(f"{args.extra_rows} must contain a JSON list of rows")
    if selection is None and extra is None:
        raise SystemExit("Give --selection and/or --extra-rows")

    rows = collect_rows(selection, extra)
    if args.rows:
        wanted = [s.strip() for s in args.rows.split(",") if s.strip()]
        known = {r["row_label"] for r in rows}
        unknown = [w for w in wanted if w not in known]
        if unknown:
            raise SystemExit(f"--rows: unknown labels {unknown}; use --list-rows to see the available ones")
        rows = [r for r in rows if r["row_label"] in wanted]
    rows = shard(rows, args.shard, args.num_shards)
    if args.list_rows:
        for r in rows:
            print(r["row_label"])
        return

    if (args.image_chunk is not None or args.merge_image_chunks) and not args.image_chunks:
        raise SystemExit("--image-chunk / --merge-image-chunks need --image-chunks")
    if args.image_chunk is not None or args.merge_image_chunks:
        if not args.skip_robustness:
            raise SystemExit("image chunks cover the main metrics only; add --skip-robustness")
        _image_chunk_modes(args, rows)
        return
    if args.robustness_chunk is not None and not args.robustness_chunks:
        raise SystemExit("--robustness-chunk needs --robustness-chunks")
    if args.robustness_chunk is not None or args.merge_robustness:
        _chunk_modes(args, rows)
        return

    metrics = NON_ROBUST if args.skip_robustness else ALL_METRICS
    tag = f"shard{args.shard}of{args.num_shards}" if args.shard is not None else "all"
    setup_logging(args.out / "logs" / f"evaluate_{tag}.log")
    logger.info("%s / %s / %s: %d rows, metrics %s", args.model, args.dataset, args.split,
                len(rows), ",".join(metrics))

    todo = []
    for r in rows:
        if args.resume and _complete(args.out, r["row_label"], metrics):
            update_row_metadata(args.out / f"{r['row_label']}.json", r)
            logger.info("%s: complete, metadata updated (--resume)", r["row_label"])
        else:
            todo.append(r)
    if not todo:
        logger.info("Nothing to do.")
        return

    from reproduce.common import RowRunner
    subset = args.robustness_subset if not args.skip_robustness else None
    main_metrics = NON_ROBUST if subset else metrics
    runner = RowRunner(model_key=args.model, dataset=args.dataset, split=args.split,
                       out_dir=args.out, json_dir=args.out, metric_short=main_metrics, args=args,
                       nr_samples=args.robustness_nr_samples)
    rob_runner = None
    subset_meta = None
    if subset:
        import numpy as np
        n = len(runner.data)
        if not (0 < subset <= n):
            raise SystemExit(f"--robustness-subset must be in [1, {n}]")
        idx = sorted(np.random.RandomState(args.robustness_subset_seed).choice(n, size=subset, replace=False).tolist())
        subset_meta = {"n": subset, "seed": args.robustness_subset_seed, "of": n}
        rob_runner = RowRunner(model_key=args.model, dataset=args.dataset, split=args.split,
                               out_dir=args.out / "robustness_subset",
                               json_dir=args.out / "robustness_subset", metric_short=ROBUST,
                               args=args, nr_samples=args.robustness_nr_samples,
                               subset_indices=idx)

    failed = []
    for i, r in enumerate(todo):
        logger.info("[%d/%d] %s (%s)", i + 1, len(todo), r["row_label"], ",".join(r["selected_by"]))
        try:
            runner.run(r)
            if rob_runner is not None:
                rob_runner.run(r)
                _merge_robustness_subset(args.out, args.out / "robustness_subset", r["row_label"], subset_meta)
            problem = _incomplete(args.out / f"{r['row_label']}.json")
            if problem:
                # Never keep a partial row: every reported number must cover all images.
                jp = args.out / f"{r['row_label']}.json"
                jp.replace(jp.with_suffix(".incomplete.json"))
                raise RuntimeError(f"incomplete row ({problem}); moved to .incomplete.json")
        except Exception as e:
            logger.exception("Row %s failed: %s", r["row_label"], e)
            failed.append(r["row_label"])
    if failed:
        logger.error("%d rows failed: %s", len(failed), failed)
        sys.exit(1)


def _chunk_modes(args, rows: List[Dict[str, Any]]) -> None:
    """--robustness-chunk k (compute) or --merge-robustness (merge + completeness check)."""
    K = args.robustness_chunks
    if args.merge_robustness:
        setup_logging(args.out / "logs" / "merge_robustness.log")
        from reproduce.common import build_split
        n = len(build_split(args.dataset, args.split, args))
        failed = []
        for r in rows:
            label = r["row_label"]
            try:
                _merge_robustness_chunks(args.out, label, n, args)
                update_row_metadata(args.out / f"{label}.json", r)
                problem = _incomplete(args.out / f"{label}.json")
                if problem:
                    jp = args.out / f"{label}.json"
                    jp.replace(jp.with_suffix(".incomplete.json"))
                    raise RuntimeError(f"incomplete row ({problem}); moved to .incomplete.json")
                logger.info("%s: merged %d robustness chunks", label, K)
            except Exception as e:
                logger.exception("Merge of %s failed: %s", label, e)
                failed.append(label)
        if failed:
            sys.exit(f"{len(failed)} merges failed: {failed}")
        return

    k = args.robustness_chunk
    d = _chunk_dir(args.out, k, K, args)
    setup_logging(args.out / "logs" / f"robustness_c{k:03d}of{K:03d}.log")
    from reproduce.common import RowRunner, build_split
    n = len(build_split(args.dataset, args.split, args))
    idx, _ = _robustness_indices(n, args)
    todo = [r for r in rows if not (args.resume and _complete(d, r["row_label"], ROBUST))]
    if not todo:
        logger.info("Nothing to do.")
        return
    runner = RowRunner(model_key=args.model, dataset=args.dataset, split=args.split,
                       out_dir=d, json_dir=d, metric_short=ROBUST, args=args,
                       nr_samples=args.robustness_nr_samples, subset_indices=idx)
    failed = []
    for r in todo:
        logger.info("%s: robustness chunk %d/%d (%d images)", r["row_label"], k, K, len(idx))
        try:
            runner.run(r)
            problem = _incomplete(d / f"{r['row_label']}.json")
            if problem:
                jp = d / f"{r['row_label']}.json"
                jp.replace(jp.with_suffix(".incomplete.json"))
                raise RuntimeError(f"incomplete chunk ({problem})")
        except Exception as e:
            logger.exception("Chunk %d of %s failed: %s", k, r["row_label"], e)
            failed.append(r["row_label"])
    if failed:
        sys.exit(f"{len(failed)} chunk rows failed: {failed}")


if __name__ == "__main__":
    main()
