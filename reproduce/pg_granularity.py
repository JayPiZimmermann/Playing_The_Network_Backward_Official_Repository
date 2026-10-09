#!/usr/bin/env python3
"""Pointing game at matched granularity (appendix analysis).

Quantus' PointingGame counts a hit if *any* pixel attaining the map maximum
lies in the object mask.  Piecewise-constant maps (superpixel methods, maps on
the ViT patch grid) have a maximum region of hundreds to thousands of pixels,
pixel-level maps usually a single pixel.  This script recomputes, from the
saliency maps cached by ``reproduce.evaluate_selected --cache-maps``:

* ``PG``         the Quantus pointing game (as reported in the tables; checked
                 against the stored per-image values);
* ``max_px``     median size of the set of pixels attaining the maximum;
* ``PG_1px``     expected hit rate of a single maximum pixel drawn uniformly
                 from that set (|mask ∩ argmax| / |argmax|);
* ``PG_16px``    pointing game after average-pooling every map to the same
                 16×16-pixel grid (the ViT-B/16 patch grid, 14×14 cells).

For each variant our rows are compared with the best baseline row by McNemar's
chi-squared test with continuity correction.  Rows: the reported row of every method (localisation selection or
the single fixed configuration) and the faith rows of our games.

Usage::

    python -m reproduce.pg_granularity --results results \\
        --out ../paper/tables/generated
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from reproduce.common import MODEL_METHODS, get_spec, load_json, setup_logging

logger = logging.getLogger("reproduce.pg_granularity")

MODELS = ("vit_b16", "vgg16", "resnet50")
MODEL_NAME = {"vit_b16": "ViT-B/16", "vgg16": "VGG-16", "resnet50": "ResNet-50"}
CELL = 16
VARIANTS = ("PG", "PG_1px", "PG_16px")


def _upsample(a: np.ndarray, size: int = 224) -> np.ndarray:
    """Nearest-neighbour upsampling to the input grid (as in the Quantus bridge)."""
    if a.shape[-1] == size:
        return a
    import cv2
    return cv2.resize(a.astype(np.float32), (size, size), interpolation=cv2.INTER_NEAREST)


def pg_variants(maps: np.ndarray, masks: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-image PG, PG_1px, PG_16px and the size of the maximum set."""
    out = {k: [] for k in VARIANTS + ("max_px",)}
    for a, s in zip(maps, masks):
        a = np.abs(_upsample(a)).astype(np.float64)
        h, w = a.shape
        top = a == a.max()
        out["PG"].append(float((top & s).any()))
        out["max_px"].append(int(top.sum()))
        out["PG_1px"].append(float((top & s).sum() / top.sum()))
        pooled = a.reshape(h // CELL, CELL, w // CELL, CELL).mean(axis=(1, 3))
        cell_mask = s.reshape(h // CELL, CELL, w // CELL, CELL).any(axis=(1, 3))
        out["PG_16px"].append(float(((pooled == pooled.max()) & cell_mask).any()))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def reported_rows(selection: Dict) -> List[Tuple[str, str, str]]:
    """(method, role, row_label): the reported (loc) row of each method + our faith rows."""
    rows = []
    for method, rec in selection["methods"].items():
        spec = get_spec(method)
        if "loc" in rec:
            rows.append((method, "loc", rec["loc"]["row_label"]))
        else:
            rows += [(method, "fixed", e["row_label"]) for e in rec.get("fixed", [])
                     if not e.get("is_anchor")]
        if spec.ours:
            for crit in ("faith",):
                if crit in rec and rec[crit]["row_label"] not in {r[2] for r in rows}:
                    rows.append((method, crit, rec[crit]["row_label"]))
    return rows


def _load_cache(test_dir: Path, label: str) -> Optional[Dict[str, np.ndarray]]:
    p = test_dir / ".saliency_cache" / f"{label}.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        key = "map__input" if "map__input" in d.files else next(k for k in d.files if k.startswith("map__"))
        return {"maps": d[key], "masks": d["masks"], "image_ids": d["image_ids"].astype(str)}


def analyse_model(results: Path, model: str) -> Tuple[List[Dict], List[Dict]]:
    from bwgame.evaluation.stats import mcnemar_test
    test_dir = results / "test" / model
    selection = load_json(results / "selection" / model / "selected.json")
    per_row, per_image = [], {}
    ids = None
    for method, role, label in reported_rows(selection):
        c = _load_cache(test_dir, label)
        if c is None:
            logger.warning("%s %s: no cached maps for %s; skipped", model, method, label)
            continue
        if ids is None:
            ids = c["image_ids"]
        if not np.array_equal(ids, c["image_ids"]):
            raise RuntimeError(f"{model}/{label}: image order differs from the other rows")
        masks = c["masks"].any(axis=-1) if c["masks"].ndim == 4 else c["masks"]
        v = pg_variants(c["maps"], masks)
        stored = np.load(test_dir / "per_image" / f"{label}.npz")["pointing_game"]
        if not np.allclose(stored, v["PG"], equal_nan=True):
            raise RuntimeError(f"{model}/{label}: recomputed PG differs from the stored values")
        per_image[label] = v
        per_row.append({"model": model, "method": method, "role": role, "row_label": label,
                        "ours": get_spec(method).ours,
                        **{k: float(v[k].mean()) for k in VARIANTS},
                        "max_px_median": float(np.median(v["max_px"]))})
    tests = []
    base = [r for r in per_row if not r["ours"]]
    for r in [r for r in per_row if r["ours"]]:
        for var in VARIANTS:
            if not base:
                continue
            best = max(base, key=lambda b: b[var])
            x, y = per_image[r["row_label"]][var], per_image[best["row_label"]][var]
            if var == "PG_1px":           # fractional per-image hits: paired sign of the difference
                from bwgame.evaluation.stats import paired_bootstrap_test
                p = paired_bootstrap_test(x, y, n_bootstrap=10000, seed=0)["p_value"]
            else:
                p = mcnemar_test(x > 0.5, y > 0.5)["p_value"]
            tests.append({"model": model, "row_label": r["row_label"], "method": r["method"],
                          "role": r["role"], "variant": var, "ours": r[var],
                          "best_baseline": best["method"], "baseline": best[var], "p": float(p)})
    return per_row, tests


def write_table(rows: List[Dict], path: Path) -> None:
    lines = [r"\begin{tabular}{@{}l l c c c c@{}}", r"\toprule",
             r"Model & Method & PG & max.\ set (px) & PG$_{1\mathrm{px}}$ & PG$_{16\mathrm{px}}$ \\",
             r"\midrule"]
    for mi, model in enumerate(MODELS):
        mrows = [r for r in rows if r["model"] == model]
        if not mrows:
            continue
        if mi:
            lines.append(r"\midrule")
        order = {m: i for i, m in enumerate(MODEL_METHODS[model])}
        mrows.sort(key=lambda r: (order.get(r["method"], 99), r["role"]))
        best = {v: max(r[v] for r in mrows) for v in VARIANTS}
        for i, r in enumerate(mrows):
            spec = get_spec(r["method"])
            name = spec.display + (f" ({r['role']})" if spec.ours and r["role"] != "loc" else "")
            cells = []
            for v in VARIANTS:
                s = f"{r[v]:.3f}"
                cells.append(r"\textbf{" + s + "}" if r[v] == best[v] else s)
            lines.append(f"{MODEL_NAME[model] if i == 0 else ''} & {name} & {cells[0]} & "
                         f"{r['max_px_median']:.0f} & {cells[1]} & {cells[2]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/tables"))
    ap.add_argument("--models", type=str, default=",".join(MODELS))
    args = ap.parse_args(argv)
    setup_logging()
    rows, tests = [], []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        r, t = analyse_model(args.results, model)
        rows += r
        tests += t
    analysis = args.results / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    for name, data in (("pg_granularity_rows.csv", rows), ("pg_granularity_tests.csv", tests)):
        with open(analysis / name, "w", newline="") as f:
            if data:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
    write_table(rows, args.out / "pg_granularity_tabular.tex")
    logger.info("wrote %s, %s", analysis / "pg_granularity_rows.csv", args.out / "pg_granularity_tabular.tex")


if __name__ == "__main__":
    main()
