#!/usr/bin/env python3
"""Evaluate the declared validation grid and select loc / faith configurations.

Evaluation (one or several shards, e.g. one per GPU)::

    python -m reproduce.select_configs --model vit_b16 \\
        --grid experiments/grids/imagenet_s_val_vit_b16.csv --split val \\
        --out results/selection/vit_b16 --shard 0 --num-shards 8 --resume

Every grid row is evaluated on the 50-image ImageNet-S validation split with
AL, PG, TK, PF5, PF20 and Sel (no MaxS / AvgS).  Shards write disjoint files:
``<out>/rows/<row_label>.json`` and ``<out>/per_image/<row_label>.npz``;
``--resume`` skips rows whose result files are complete.

Aggregation (no GPU needed)::

    python -m reproduce.select_configs --model vit_b16 \\
        --grid experiments/grids/imagenet_s_val_vit_b16.csv --split val \\
        --out results/selection/vit_b16 --aggregate

writes ``val_scores.csv``, ``selected.json`` and ``selection_summary.md``.
For every tunable method two configurations are selected (independently;
they may coincide):

* ``loc``   minimal rank-sum over {AL, PG, TK}
* ``faith`` minimal rank-sum over {PF5, PF20, Sel}

Ranks are computed within the method across its configurations, respecting
metric directions (``bwgame.evaluation.metric_spec``); a missing / NaN value
gets the worst rank.  Ties are broken in favour of the anchor / default
configuration, then of fewer non-default parameters, then of grid order.
Methods with a single configuration, LIME / KernelSHAP and the anchors are
recorded as ``fixed``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from reproduce.common import (
    GridRow, METHOD_SPECS, add_data_args, check_model, config_plain, display_path, fmt_float, get_spec,
    load_row_results, n_nondefault, read_grid, setup_logging, shard, write_json_atomic,
)

logger = logging.getLogger("reproduce.select_configs")

VAL_METRICS = ("AL", "PG", "TK", "PF5", "PF20", "Sel")
LOC = ("AL", "PG", "TK")
FAITH = ("PF5", "PF20", "Sel")
CRITERIA = ("loc", "faith")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_grid(rows: Sequence[GridRow], *, model: str, dataset: str, split: str,
                  out: Path, args, resume: bool) -> List[str]:
    """Evaluate ``rows``; returns the labels of rows that failed."""
    from reproduce.common import RowRunner

    out = Path(out)
    runner = RowRunner(model_key=model, dataset=dataset, split=split, out_dir=out,
                       json_dir=out / "rows", metric_short=VAL_METRICS, args=args)
    failed: List[str] = []
    for i, r in enumerate(rows):
        if resume and runner.is_complete(r.row_label):
            logger.info("[%d/%d] %s: complete, skipping (--resume)", i + 1, len(rows), r.row_label)
            continue
        logger.info("[%d/%d] %s", i + 1, len(rows), r.row_label)
        try:
            runner.run({"method": r.method, "row_label": r.row_label, "params": r.params,
                        "is_anchor": r.is_anchor, "tunable": r.tunable})
        except Exception as e:  # keep going; the aggregate step reports missing rows
            logger.exception("Row %s failed: %s", r.row_label, e)
            failed.append(r.row_label)
            err = out / "rows" / f"{r.row_label}.error.txt"
            err.parent.mkdir(parents=True, exist_ok=True)
            err.write_text(f"{type(e).__name__}: {e}\n")
    return failed


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def average_ranks(values: Sequence[float], direction: str) -> List[float]:
    """1-based average ranks (best = 1); NaN / missing values rank last."""
    if direction not in ("max", "min"):
        raise ValueError(direction)
    keys = []
    for v in values:
        if v is None or not math.isfinite(v):
            keys.append(math.inf)
        else:
            keys.append(-v if direction == "max" else v)
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    ranks = [0.0] * len(keys)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and keys[order[j + 1]] == keys[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _metric_mean(res: Optional[Dict[str, Any]], short: str) -> float:
    if res is None:
        return float("nan")
    m = res.get("metrics", {}).get(short)
    return float(m["mean"]) if m and m.get("mean") is not None else float("nan")


def inadmissible_reason(res: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why a validation row may not be ranked, or None if it is admissible.

    A configuration is admissible only if its map is finite and non-zero on
    every validation image and every selection metric is defined on every
    image.  Otherwise the metrics are undefined (Quantus rejects all-zero
    maps) and the row is excluded from ranking instead of receiving a NaN.
    """
    if res is None:
        return "not evaluated"
    if int(res.get("map_failures", 0) or 0) > 0:
        return f"map failed on {res['map_failures']} image(s)"
    if int(res.get("zero_maps", 0) or 0) > 0:
        return f"all-zero map on {res['zero_maps']} image(s)"
    for s in VAL_METRICS:
        m = res.get("metrics", {}).get(s)
        if not m or m.get("mean") is None or not math.isfinite(float(m["mean"])):
            return f"{s} undefined"
        if m.get("coverage") is None or float(m["coverage"]) < 1.0:
            return f"{s} coverage {m.get('coverage')}"
    return None


def select(grid: Sequence[GridRow], results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Per-method loc / faith selection.  Returns the ``methods`` / ``rows`` parts."""
    from bwgame.evaluation.metric_spec import metric_direction

    by_method: Dict[str, List[GridRow]] = {}
    for r in grid:
        by_method.setdefault(r.method, []).append(r)

    methods: Dict[str, Any] = {}
    flat: Dict[str, Dict[str, Any]] = {}
    rank_info: Dict[str, Dict[str, float]] = {}

    def _flat_add(r: GridRow, role: str) -> None:
        e = flat.setdefault(r.row_label, {
            "method": r.method, "row_label": r.row_label, "params": r.params,
            "is_anchor": r.is_anchor, "tunable": r.tunable, "selected_by": [],
        })
        if role not in e["selected_by"]:
            e["selected_by"].append(role)

    for method, rows in by_method.items():
        spec = get_spec(method)
        inadmissible = {r.row_label: inadmissible_reason(results[r.row_label])
                        for r in rows if r.row_label in results}
        inadmissible = {k: v for k, v in inadmissible.items() if v is not None}
        for r in rows:
            if r.row_label in inadmissible and (r.is_anchor or not r.tunable):
                raise SystemExit(f"{method}: reported row {r.row_label} is inadmissible "
                                 f"({inadmissible[r.row_label]}); fix before selecting")
        evaluated = [r for r in rows if r.row_label in results and r.row_label not in inadmissible]
        rs_loc = [0.0] * len(evaluated)
        rs_faith = [0.0] * len(evaluated)
        for group, acc in ((LOC, rs_loc), (FAITH, rs_faith)):
            for s in group:
                ranks = average_ranks([_metric_mean(results[r.row_label], s) for r in evaluated],
                                      metric_direction(s))
                for k, rk in enumerate(ranks):
                    acc[k] += rk
        for k, r in enumerate(evaluated):
            rank_info[r.row_label] = {"rank_sum_loc": rs_loc[k], "rank_sum_faith": rs_faith[k]}

        def _entry(r: GridRow) -> Dict[str, Any]:
            res = results.get(r.row_label)
            return {
                "row_label": r.row_label, "params": r.params, "is_anchor": r.is_anchor,
                "n_nondefault": n_nondefault(method, r.params),
                **rank_info.get(r.row_label, {}),
                "val": {s: _metric_mean(res, s) for s in VAL_METRICS},
                "val_coverage": {s: (res or {}).get("metrics", {}).get(s, {}).get("coverage")
                                 for s in VAL_METRICS},
            }

        rec: Dict[str, Any] = {"display": spec.display, "block": spec.block, "ours": spec.ours,
                               "n_configs": len(rows), "n_evaluated": len(evaluated) + len(inadmissible),
                               "n_admissible": len(evaluated),
                               "inadmissible": [{"row_label": k, "reason": v}
                                                for k, v in inadmissible.items()],
                               "tunable": bool(rows[0].tunable), "fixed": []}
        tunable = rows[0].tunable
        if tunable and evaluated:
            for crit, rs in (("loc", rs_loc), ("faith", rs_faith)):
                def _key(k):
                    r = evaluated[k]
                    nd = n_nondefault(method, r.params)
                    preferred = r.is_anchor or nd == 0
                    return (rs[k], 0 if preferred else 1, nd, r.order)
                best = min(range(len(evaluated)), key=_key)
                rec[crit] = _entry(evaluated[best])
                _flat_add(evaluated[best], crit)
        elif tunable:
            logger.warning("%s: no evaluated rows; nothing selected", method)
        for r in rows:
            if r.is_anchor or not tunable:
                rec["fixed"].append(_entry(r))
                _flat_add(r, "anchor" if r.is_anchor else "fixed")
        methods[method] = rec

    order = {r.row_label: r.order for r in grid}
    rows_out = sorted(flat.values(), key=lambda e: order[e["row_label"]])
    for e in rows_out:
        e["selected_by"] = sorted(e["selected_by"])
    return {"methods": methods, "rows": rows_out, "rank_info": rank_info}


def aggregate(grid: Sequence[GridRow], res_dir: Path, out: Path, *, meta: Dict[str, Any],
              allow_missing: bool) -> Dict[str, Any]:
    """Write val_scores.csv, selected.json and selection_summary.md into ``out``."""
    out = Path(out)
    results = load_row_results(res_dir)
    missing = [r.row_label for r in grid if r.row_label not in results]
    if missing:
        msg = (f"{len(missing)}/{len(grid)} grid rows have no result in {res_dir} "
               f"(e.g. {missing[:3]}). Run the missing shards (with --resume) first.")
        if not allow_missing:
            raise SystemExit(msg + " Or pass --allow-missing to select among the evaluated rows.")
        logger.warning(msg)
    unknown = sorted(set(results) - {r.row_label for r in grid})
    if unknown:
        logger.warning("%d result files are not in the grid and are ignored (e.g. %s)",
                       len(unknown), unknown[:3])

    sel = select(grid, results)
    rank_info = sel.pop("rank_info")

    # val_scores.csv
    import csv
    cols = (["order", "method", "row_label", "is_anchor", "tunable", "n_nondefault", "status",
             "n_images", "map_failures"] + list(VAL_METRICS)
            + [f"{s}_coverage" for s in VAL_METRICS]
            + ["rank_sum_loc", "rank_sum_faith", "time_median_s", "params_json"])
    bad = {e["row_label"]: e["reason"] for rec in sel["methods"].values()
           for e in rec.get("inadmissible", [])}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "val_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in grid:
            res = results.get(r.row_label)
            status = ("missing" if res is None else
                      f"inadmissible: {bad[r.row_label]}" if r.row_label in bad else "ok")
            if r.row_label in bad:
                res = None          # no numbers for inadmissible rows
            m = (res or {}).get("metrics", {})
            ri = rank_info.get(r.row_label, {})
            w.writerow([r.order, r.method, r.row_label, int(r.is_anchor), int(r.tunable),
                        n_nondefault(r.method, r.params), status,
                        (res or {}).get("n_images", ""), (res or {}).get("map_failures", "")]
                       + [fmt_float(m.get(s, {}).get("mean"), 6) if res else "" for s in VAL_METRICS]
                       + [fmt_float(m.get(s, {}).get("coverage"), 4) if res else "" for s in VAL_METRICS]
                       + [ri.get("rank_sum_loc", ""), ri.get("rank_sum_faith", ""),
                          fmt_float((res or {}).get("runtime_s", {}).get("median"), 4) if res else "",
                          json.dumps(r.params, sort_keys=True)])

    selection = {
        **meta,
        "criteria": {"loc": list(LOC), "faith": list(FAITH)},
        "tie_break": "anchor/default first, then fewer non-default parameters, then grid order",
        "n_grid_rows": len(grid),
        "n_val_configs": len(grid) - len(missing),
        "n_admissible": len(grid) - len(missing) - len(bad),
        "admissibility": "rows with a failed or all-zero map, or an undefined selection metric, "
                         "on any validation image are excluded from ranking",
        "inadmissible_rows": [{"row_label": k, "reason": v} for k, v in bad.items()],
        "missing_rows": missing,
        "methods": sel["methods"],
        "rows": sel["rows"],
    }
    write_json_atomic(out / "selected.json", selection)
    (out / "selection_summary.md").write_text(summary_markdown(selection))
    logger.info("Wrote %s, %s, %s", out / "val_scores.csv", out / "selected.json",
                out / "selection_summary.md")
    return selection


def summary_markdown(selection: Dict[str, Any]) -> str:
    lines = [
        f"# Validation selection: {selection.get('model', '?')} "
        f"({selection.get('dataset', '?')}, {selection.get('split', '?')})",
        "",
        f"Grid: `{selection.get('grid', '?')}` - {selection['n_val_configs']} of "
        f"{selection['n_grid_rows']} configurations evaluated.",
        "",
        "loc = min rank-sum over {AL, PG, TK}; faith = min rank-sum over {PF5, PF20, Sel} "
        "(ranks within each method).",
        "",
        "| Method | Criterion | Row label | Config | AL | PG | TK | PF5 | PF20 | Sel | "
        "Loc rank-sum | Faith rank-sum |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for method, rec in selection["methods"].items():
        entries = [(c, rec[c]) for c in CRITERIA if c in rec]
        entries += [("anchor" if e["is_anchor"] else "fixed", e) for e in rec["fixed"]]
        for crit, e in entries:
            v = e["val"]
            lines.append(
                f"| {METHOD_SPECS[method].plain} ({rec['n_evaluated']}/{rec['n_configs']}) | {crit} | "
                f"`{e['row_label']}` | {config_plain(method, e['params'])} | "
                + " | ".join(fmt_float(v[s], 2 if s.startswith('PF') else 3) for s in VAL_METRICS)
                + f" | {e.get('rank_sum_loc', '')} | {e.get('rank_sum_faith', '')} |")
    if selection.get("inadmissible_rows"):
        lines += ["", f"**Inadmissible rows ({len(selection['inadmissible_rows'])}, excluded from "
                  "ranking)**: " + ", ".join(f"`{e['row_label']}` ({e['reason']})"
                                             for e in selection["inadmissible_rows"])]
    if selection.get("missing_rows"):
        lines += ["", f"**Missing rows ({len(selection['missing_rows'])})**: "
                  + ", ".join(f"`{m}`" for m in selection["missing_rows"][:20])]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Evaluate the validation grid and select loc/faith configs.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--grid", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=("val",))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="Skip rows whose result files are complete.")
    ap.add_argument("--aggregate", action="store_true",
                    help="Merge shard results and write val_scores.csv / selected.json.")
    ap.add_argument("--allow-missing", action="store_true",
                    help="(--aggregate) select among evaluated rows even if some are missing.")
    add_data_args(ap)
    args = ap.parse_args(argv)
    check_model(args.model)

    grid = read_grid(args.grid)
    meta = {"model": args.model, "dataset": "imagenet-s", "split": args.split, "grid": display_path(args.grid)}
    if args.aggregate:
        setup_logging()
        aggregate(grid, args.out / "rows", args.out, meta=meta, allow_missing=args.allow_missing)
        return

    tag = f"shard{args.shard}of{args.num_shards}" if args.shard is not None else "all"
    setup_logging(args.out / "logs" / f"select_{tag}.log")
    rows = shard(grid, args.shard, args.num_shards)
    logger.info("%s: evaluating %d of %d grid rows (%s)", args.model, len(rows), len(grid), tag)
    failed = evaluate_grid(rows, model=args.model, dataset="imagenet-s", split=args.split,
                           out=args.out, args=args, resume=args.resume)
    if failed:
        logger.error("%d rows failed: %s", len(failed), failed)
        sys.exit(1)


if __name__ == "__main__":
    main()
