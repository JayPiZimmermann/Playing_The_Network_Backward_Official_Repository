#!/usr/bin/env python3
"""Write the LaTeX tables / CSV / macros that the paper ``\\input``s.

Usage::

    python -m reproduce.make_tables --results results --out ../paper/tables/generated \\
        [--models vit_b16,vgg16,resnet50] [--voc results/voc]

Expected inputs (all optional; missing data become red ``??`` placeholders)::

    results/selection/<model>/selected.json     reproduce.select_configs --aggregate
    results/test/<model>/*.json, per_image/     reproduce.evaluate_selected
    results/test/<model>/ci.csv                 reproduce.bootstrap_ci (else computed here)
    results/runtime/<model>.csv                 reproduce.runtime_benchmark (t/img column)
    results/voc/selection/selected.json         reproduce.pascal_voc
    results/voc/test/*.json, per_image/         reproduce.pascal_voc

Outputs (file names are a fixed contract with the paper)::

    main_vit_tabular.tex                 main ViT-B/16 table (complete tabular)
    test_<model>_ci_longtable.tex        all test rows with 95% CIs (longtable)
    test_voc_vit_b16_ci_longtable.tex    same for the Pascal VOC ViT-B/16 test rows
    val_selection_<model>_tabular.tex    selected configs and validation rank-sums (tabular)
    numbers.tex                          \\global\\prelimfalse + macros quoted in the text

Significance markers use ``reproduce.bootstrap_ci`` (paired bootstrap, McNemar
for PG, Holm).  In the main table the Holm family of a column is the set of
distinct rows shown in the table.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from reproduce.common import (
    METHOD_SPECS, MODEL_METHODS, _lnum, config_latex, get_spec, load_row_results, setup_logging,
)

logger = logging.getLogger("reproduce.make_tables")

MISSING = r"\textcolor{red}{??}"
METRICS = ("AL", "PG", "TK", "PF5", "PF20", "Sel", "MaxS", "AvgS")
METRIC_HEAD = {"AL": r"AL$\uparrow$", "PG": r"PG$\uparrow$", "TK": r"TK$\uparrow$",
               "PF5": r"PF$_5$$\downarrow$", "PF20": r"PF$_{20}$$\downarrow$",
               "Sel": r"Sel$\downarrow$", "MaxS": r"MaxS$\downarrow$", "AvgS": r"AvgS$\downarrow$"}
# Main-text table: a compact subset of the baselines (every method with
# confidence intervals is in the appendix longtables).
MAIN_VIT_BLOCKS = (("gradient", "smoothgrad"),
                   ("lrp_gamma", "attnlrp"),
                   ("tibav", "gae"),
                   ("kernelshap",),
                   ("rg", "rgeq"))
MODEL_FILE = {"vit_b16": "vit_b16", "vgg16": "vgg16", "resnet50": "resnet50"}
ROLE_ORDER = {"anchor": 0, "loc": 1, "faith": 2, "fixed": 4, "extra": 6}
CRITERIA = ("loc", "faith")


def shade(cell: str, best: bool, approx: bool) -> str:
    """Light background for the best cell of a column and for cells not
    significantly different from it (\\cellbest / \\cellapprox, defined in main.tex)."""
    if best:
        return r"\cellbest{" + cell + "}"
    if approx:
        return r"\cellapprox{" + cell + "}"
    return cell


def ci_cell(value: str, lo: float, hi: float, d: int) -> str:
    """Value with its 95% CI in small print below; top-aligned so that the value
    sits on the row baseline next to the method name."""
    return (r"\begin{tabular}[t]{@{}c@{}}" + value + r"\\[-1pt]{\tiny[" + f"{lo:.{d}f},{hi:.{d}f}"
            + r"]}\end{tabular}")


def ndec(metric: str) -> int:
    return 2 if metric.startswith("PF") or metric == "time" else 3


def fnum(v: Optional[float], metric: str) -> Optional[str]:
    if v is None or not math.isfinite(v):
        return None
    return f"{v:.{ndec(metric)}f}"


def _finite(v) -> bool:
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v)


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
class ResultSet:
    """Test results of one setting (a results/test/<model> or results/voc/test dir)."""

    def __init__(self, test_dir: Path, selection: Optional[Dict[str, Any]],
                 runtime_csv: Optional[Path], n_boot: int, seed: int):
        self.dir = Path(test_dir)
        self.rows = load_row_results(self.dir)
        self.selection = selection
        self.n_boot, self.seed = n_boot, seed
        self._scores = None
        self._ci = None
        self.runtime: Dict[str, Dict[str, str]] = {}
        if runtime_csv is not None and Path(runtime_csv).exists():
            from reproduce.runtime_benchmark import read_runtime_csv
            _, self.runtime = read_runtime_csv(runtime_csv)

    @property
    def empty(self) -> bool:
        return not self.rows

    def mean(self, label: Optional[str], metric: str) -> Optional[float]:
        if label is None or label not in self.rows:
            return None
        m = self.rows[label].get("metrics", {}).get(metric)
        v = None if m is None else m.get("mean")
        return float(v) if _finite(v) else None

    def time(self, label: Optional[str]) -> Optional[float]:
        if label is None:
            return None
        if label in self.runtime:
            try:
                v = float(self.runtime[label]["median_s"])
                if math.isfinite(v):
                    return v
            except (KeyError, ValueError):
                pass
        if label in self.rows:
            v = self.rows[label].get("runtime_s", {}).get("median")
            return float(v) if _finite(v) else None
        return None

    def scores(self) -> Dict[str, Dict[str, Any]]:
        if self._scores is None:
            if self.empty:
                self._scores = {}
            else:
                from reproduce.bootstrap_ci import load_scores
                _, _, self._scores = load_scores(self.dir)
        return self._scores

    def ci(self, label: str, metric: str) -> Optional[Tuple[float, float, float]]:
        if self._ci is None:
            self._ci = {}
            p = self.dir / "ci.csv"
            if p.exists():
                with open(p) as f:
                    for r in csv.DictReader(f):
                        self._ci[(r["row"], r["metric"])] = (float(r["mean"]), float(r["lo"]), float(r["hi"]))
        if (label, metric) in self._ci:
            return self._ci[(label, metric)]
        sc = self.scores().get(label, {})
        if metric not in sc:
            return None
        from reproduce.bootstrap_ci import boot_indices, ci_from_indices
        idx = boot_indices(len(sc[metric]), self.n_boot, self.seed)
        mean, lo, hi, _ = ci_from_indices(sc[metric], idx)
        self._ci[(label, metric)] = (mean, lo, hi)
        return self._ci[(label, metric)]

    # -- roles ---------------------------------------------------------------
    def role_labels(self, method: str) -> Dict[str, Any]:
        """{'loc'/'faith': label, 'anchor': [labels], 'fixed': [labels]}."""
        out: Dict[str, Any] = {"anchor": [], "fixed": []}
        rec = (self.selection or {}).get("methods", {}).get(method)
        if rec is not None:
            for crit in CRITERIA:
                if crit in rec:
                    out[crit] = rec[crit]["row_label"]
            for e in rec.get("fixed", []):
                out["anchor" if e.get("is_anchor") else "fixed"].append(e["row_label"])
            return out
        for label, obj in sorted(self.rows.items()):
            if obj.get("method") != method:
                continue
            roles = obj.get("selected_by", [])
            for crit in CRITERIA:
                if crit in roles:
                    out[crit] = label
            if "anchor" in roles:
                out["anchor"].append(label)
            if "fixed" in roles:
                out["fixed"].append(label)
        return out

    def params_of(self, method: str, label: str) -> Optional[Dict[str, Any]]:
        if label in self.rows:
            return self.rows[label].get("params")
        rec = (self.selection or {}).get("methods", {}).get(method, {})
        for e in [rec.get(c) for c in CRITERIA] + list(rec.get("fixed", [])):
            if e and e.get("row_label") == label:
                return e.get("params")
        return None

    def recommended(self, method: str) -> Optional[str]:
        """The one row reported per method: the localisation selection, or the single fixed row."""
        roles = self.role_labels(method)
        if roles.get("loc"):
            return roles["loc"]
        return roles["fixed"][0] if roles["fixed"] else None

    def recommended_rows(self, ours_only: Optional[bool] = None) -> List[str]:
        """Recommended row of every method present (optionally only ours / only baselines).

        Methods without an entry in the selection file (on VOC: our games, whose
        ImageNet-S selections are transferred) are found through the roles
        stored in their test rows.
        """
        methods = list((self.selection or {}).get("methods", {}))
        methods += sorted({o["method"] for o in self.rows.values()} - set(methods))
        out = []
        for m in methods:
            if ours_only is not None and get_spec(m).ours != ours_only:
                continue
            lbl = self.recommended(m)
            if lbl is not None and lbl in self.rows:
                out.append(lbl)
        return out


def metric_expected(obj: Optional[Dict[str, Any]], metric: str) -> bool:
    """True if the metric belongs to this row (missing = pending, shown as ??).

    Ablation / sweep rows and the VOC transfer are evaluated without the
    robustness metrics by design (shown as --)."""
    from reproduce.bootstrap_ci import is_sweep_only
    if obj is None:
        return True
    if metric in ("MaxS", "AvgS") and (obj.get("dataset") == "voc" or is_sweep_only(obj)):
        return False
    return True


def load_selection(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    logger.warning("No selection file %s", path)
    return None


# ---------------------------------------------------------------------------
# Main ViT table
# ---------------------------------------------------------------------------
def main_rows(rs: ResultSet) -> List[List[Dict[str, Any]]]:
    """Blocks of display rows: dict(name, method, label).

    One row per method: its localisation selection (chosen by the same rule for
    every method) or its single fixed configuration; for our games the untuned
    anchor is listed as reference.  The faithfulness selections are in the
    appendix longtables.
    """
    blocks = []
    for block in MAIN_VIT_BLOCKS:
        rows = []
        for method in block:
            if method not in MODEL_METHODS["vit_b16"]:
                continue
            spec = get_spec(method)
            roles = rs.role_labels(method)
            if spec.ours:
                anchors = roles["anchor"] or [None]
                rows.append({"name": f"{spec.display} anchor", "method": method, "label": anchors[0]})
            rows.append({"name": spec.display, "method": method, "label": rs.recommended(method)})
        if rows:
            blocks.append(rows)
    return blocks


def _column_marks(rs: ResultSet, labels: Sequence[Optional[str]], metric: str):
    """(best value, set of labels not significantly different from the best)."""
    from bwgame.evaluation.metric_spec import metric_direction
    from reproduce.bootstrap_ci import significance_vs_best
    vals = {l: rs.mean(l, metric) for l in labels if l is not None}
    vals = {l: v for l, v in vals.items() if v is not None}
    if not vals:
        return None, set()
    best = (max if metric_direction(metric) == "max" else min)(vals.values())
    approx = set()
    fam = sorted({l for l in vals if metric in rs.scores().get(l, {})})
    if len(fam) > 1:
        for r in significance_vs_best(rs.scores(), fam, metric, rs.n_boot, rs.seed):
            if not r["significant"]:
                approx.add(r["row"])
    return best, approx


def write_main_vit(rs: ResultSet, out: Path) -> None:
    blocks = main_rows(rs)
    labels = [r["label"] for b in blocks for r in b]
    marks = {m: _column_marks(rs, labels, m) for m in METRICS}
    times = [rs.time(l) for l in labels]
    tmin = min([t for t in times if t is not None], default=None)

    lines = [r"\begin{tabular}{@{}l l c c c c c c c c c@{}}", r"\toprule",
             "Method & Config & " + " & ".join(METRIC_HEAD[m] for m in METRICS) + r" & $t$/img \\",
             r"\midrule"]
    for bi, block in enumerate(blocks):
        if bi:
            lines.append(r"\midrule")
        for r in block:
            label, method = r["label"], r["method"]
            params = rs.params_of(method, label) if label else None
            cfg = config_latex(method, params) if params is not None else MISSING
            cells = []
            for m in METRICS:
                v = rs.mean(label, m)
                s = fnum(v, m)
                if s is None:
                    cells.append(MISSING if metric_expected(rs.rows.get(label), m) else "--")
                    continue
                best, approx = marks[m]
                is_best = best is not None and v == best
                is_approx = not is_best and label in approx
                if is_best:
                    s = r"\textbf{" + s + "}"
                elif is_approx:
                    s += r"\textsuperscript{$\approx$}"
                ci = rs.ci(label, m)
                if ci is not None and math.isfinite(ci[1]) and math.isfinite(ci[2]):
                    d = ndec(m)
                    s = ci_cell(s, ci[1], ci[2], d)
                cells.append(shade(s, is_best, is_approx))
            t = rs.time(label)
            ts = fnum(t, "time")
            if ts is None:
                ts = MISSING
            elif tmin is not None and t == tmin:
                ts = r"\textbf{" + ts + "}"
            lines.append(f"{r['name']} & {cfg} & " + " & ".join(cells) + f" & {ts} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out / "main_vit_tabular.tex", lines)


# ---------------------------------------------------------------------------
# Appendix tables
# ---------------------------------------------------------------------------
def _row_name(obj: Dict[str, Any]) -> str:
    """Method name + selection role, e.g. 'RG (loc/faith)', 'RG anchor', 'RG ($\\tau$ sweep)'."""
    spec = get_spec(obj["method"])
    roles = [r for r in CRITERIA if r in obj.get("selected_by", [])]
    if "anchor" in obj.get("selected_by", []) or (obj.get("is_anchor") and not roles):
        return f"{spec.display} anchor" + (" (" + "/".join(roles) + ")" if roles else "")
    if roles:
        return f"{spec.display} (" + "/".join(roles) + ")"
    sweeps = sorted({s["series"] for s in obj.get("sweeps", [])})
    if sweeps:
        sym = {"tau": r"$\tau$", "lambda_sm": r"$\lambda_{\mathrm{sm}}$"}
        return f"{spec.display} (" + ", ".join(sym.get(s, s) for s in sweeps) + " sweep)"
    return spec.display


def _sort_key(model_methods: Sequence[str]):
    def key(obj):
        m = obj["method"]
        roles = obj.get("selected_by", [])
        role = min([ROLE_ORDER.get(r, 4) for r in roles] or [4])
        sw = sorted((s["series"], s["value"]) for s in obj.get("sweeps", []))
        return (model_methods.index(m) if m in model_methods else 99, role, sw, obj["row_label"])
    return key


def write_ci_longtable(rs: Optional[ResultSet], model: str, path: Path) -> None:
    ncol = 2 + len(METRICS)
    head = "Method & Config & " + " & ".join(METRIC_HEAD[m] for m in METRICS) + r" \\"
    # Tight column separation and a two-line cell (mean over a tiny CI) keep
    # the ten columns within a single-column page width.
    spec = r"@{}l@{\hspace{2.5pt}}p{2.7cm}" + r"@{\hspace{2.5pt}}c" * len(METRICS) + "@{}"
    lines = [r"\begin{longtable}{" + spec + "}",
             r"\toprule", head, r"\midrule", r"\endfirsthead",
             r"\toprule", head, r"\midrule", r"\endhead",
             r"\midrule", rf"\multicolumn{{{ncol}}}{{r}}{{\emph{{continued on next page}}}} \\", r"\endfoot",
             r"\bottomrule", r"\endlastfoot"]
    if rs is None or rs.empty:
        lines.append(rf"\multicolumn{{{ncol}}}{{c}}{{{MISSING}}} \\")
    else:
        from reproduce.bootstrap_ci import is_sweep_only
        rows = [o for o in rs.rows.values() if not is_sweep_only(o)]
        labels = [o["row_label"] for o in rows]
        marks = {m: _column_marks(rs, labels, m) for m in METRICS}
        for obj in sorted(rows, key=_sort_key(MODEL_METHODS.get(model, ()))):
            cells = []
            for m in METRICS:
                if m not in obj.get("metrics", {}):
                    cells.append(MISSING if metric_expected(obj, m) else "--")
                    continue
                ci = rs.ci(obj["row_label"], m)
                if ci is None or not math.isfinite(ci[0]):
                    cells.append(MISSING)
                    continue
                mean, lo, hi = ci
                d = ndec(m)
                best, approx = marks[m]
                v = rs.mean(obj["row_label"], m)
                is_best = best is not None and v == best
                is_approx = not is_best and obj["row_label"] in approx
                s = f"{mean:.{d}f}"
                if is_best:
                    s = r"\textbf{" + s + "}"
                elif is_approx:
                    s += r"\textsuperscript{$\approx$}"
                cells.append(shade(ci_cell(s, lo, hi, d), is_best, is_approx))
            cfg = config_latex(obj["method"], obj["params"]).replace(",", r",\allowbreak ")
            lines.append(f"{_row_name(obj)} & {cfg} & " + " & ".join(cells) + r" \\")
    lines.append(r"\end{longtable}")
    _write(path, lines)


ABLATION_PARAM = {"tau": r"$\tau$", "lambda_sm": r"$\lambda_{\mathrm{sm}}$", "lambda": r"$\lambda$",
                  "sigma2": r"$\sigma^2$", "eps": r"$\varepsilon$", "gate": r"gate $\tilde\Phi$",
                  "equivariance": r"+Eq ($N{=}50$)", "attention": "attention rule"}
ABLATION_METRICS = ("AL", "PG", "TK", "PF5", "PF20", "Sel")
# Series of the attention-rule comparison (own table, not in the one-factor longtables).
ATTENTION_CENTRES = ("anchor", "loc", "faith")


def _fmt_ablation_value(param: str, value: Any) -> str:
    if param == "gate":
        return "on" if value else "off"
    if param == "equivariance":
        return "--"
    return _lnum(float(value))


def write_ablation_longtable(rs: Optional[ResultSet], model: str, path: Path) -> None:
    """One-factor ablations around the selected rows (test split, means)."""
    ncol = 3 + len(ABLATION_METRICS)
    head = ("Center & Factor & Value & " + " & ".join(METRIC_HEAD[m] for m in ABLATION_METRICS) + r" \\")
    spec = r"@{}l l c" + " c" * len(ABLATION_METRICS) + "@{}"
    lines = [r"\begin{longtable}{" + spec + "}", r"\toprule", head, r"\midrule", r"\endfirsthead",
             r"\toprule", head, r"\midrule", r"\endhead", r"\bottomrule", r"\endlastfoot"]
    series: Dict[str, List[Tuple[Any, str, bool]]] = {}
    if rs is not None and not rs.empty:
        for obj in rs.rows.values():
            for sw in obj.get("sweeps", []):
                if ":" not in str(sw.get("series", "")) or sw.get("param") == "attention":
                    continue
                centre = sw.get("center") == obj["row_label"]
                series.setdefault(sw["series"], []).append((sw["value"], obj["row_label"], centre))
    if not series:
        lines.append(rf"\multicolumn{{{ncol}}}{{c}}{{{MISSING}}} \\")
    order_m = {m: i for i, m in enumerate(("rg", "rgeq", "sg"))}
    order_p = {p: i for i, p in enumerate(ABLATION_PARAM)}

    def skey(name):
        meth, crit, param = name.split(":")
        return (order_m.get(meth, 9), crit, order_p.get(param, 9))
    first = True
    for name in sorted(series, key=skey):
        meth, crit, param = name.split(":")
        if not first:
            lines.append(r"\midrule")
        first = False
        pts = sorted(series[name], key=lambda t: (t[0] if not isinstance(t[0], bool) else int(t[0])))
        for i, (value, label, centre) in enumerate(pts):
            cells = []
            for m in ABLATION_METRICS:
                s = fnum(rs.mean(label, m), m)
                cells.append(MISSING if s is None else s)
            v = _fmt_ablation_value(param, value)
            if centre:
                v = r"\textbf{" + v + "}"
            first_cols = (f"{METHOD_SPECS[meth].display} ({crit.replace('+', '/')})" if i == 0 else "",
                          ABLATION_PARAM.get(param, param) if i == 0 else "")
            lines.append(f"{first_cols[0]} & {first_cols[1]} & {v} & " + " & ".join(cells) + r" \\")
    lines.append(r"\end{longtable}")
    _write(path, lines)


def write_attention_rule(rs: Optional[ResultSet], path: Path) -> None:
    """RG on ViT-B/16: gradient-fallback attention vs Value-only routing (test split).

    One row per centre (the alpha-beta anchor and the loc / faith
    selections); each cell holds the gradient fallback on top and Value-only
    routing (gray) below.  $^{\\dagger}$: paired difference significant
    (paired bootstrap / McNemar for PG, Holm over all cells, p < 0.05).
    """
    import numpy as np
    from bwgame.evaluation.stats import holm_correction
    from reproduce.bootstrap_ci import ALPHA, compare
    ncol = 1 + len(ABLATION_METRICS)
    lines = [r"\begin{tabular}{@{}l" + " c" * len(ABLATION_METRICS) + "@{}}", r"\toprule",
             "Center & " + " & ".join(METRIC_HEAD[m] for m in ABLATION_METRICS) + r" \\", r"\midrule"]
    pairs: Dict[str, Dict[float, str]] = {}
    if rs is not None and not rs.empty:
        for obj in rs.rows.values():
            for sw in obj.get("sweeps", []):
                parts = str(sw.get("series", "")).split(":")
                if len(parts) == 3 and parts[0] == "rg" and parts[2] == "attention":
                    pair = pairs.setdefault(parts[1], {})
                    pair[float(sw["value"])] = obj["row_label"]
                    pair.setdefault(0.0, sw.get("center"))
    tags = [t for t in ATTENTION_CENTRES if t in pairs]
    tests: Dict[Tuple[str, str], float] = {}
    for t in tags:
        grad, vonly = pairs[t].get(1.0), pairs[t].get(0.0)
        for m in ABLATION_METRICS:
            a = rs.scores().get(grad, {}).get(m) if grad else None
            b = rs.scores().get(vonly, {}).get(m) if vonly else None
            if a is not None and b is not None:
                tests[(t, m)] = compare(a, b, m, rs.n_boot, rs.seed)[1]
    keys = list(tests)
    holm = dict(zip(keys, holm_correction([tests[k] if np.isfinite(tests[k]) else 1.0 for k in keys])))
    if not tags:
        lines.append(rf"\multicolumn{{{ncol}}}{{c}}{{{MISSING}}} \\")
    for t in tags:
        vonly = pairs[t].get(0.0)
        params = rs.rows[vonly]["params"] if vonly in rs.rows else None
        name = t + (" (" + config_latex("rg", params) + ")" if params else "")
        cells = []
        for m in ABLATION_METRICS:
            g = fnum(rs.mean(pairs[t].get(1.0), m), m)
            v = fnum(rs.mean(vonly, m), m)
            g = MISSING if g is None else g + (r"$^{\dagger}$" if holm.get((t, m), 1.0) < ALPHA else "")
            cells.append(r"\absval{" + g + "}{" + (MISSING if v is None else v) + "}")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(path, lines)


def write_val_selection(selection: Optional[Dict[str, Any]], model: str, path: Path) -> None:
    lines = [r"\begin{tabular}{@{}l l l c c@{}}", r"\toprule",
             r"Method & Criterion & Selected configuration & Loc.\ rank-sum & Faith.\ rank-sum \\",
             r"\midrule"]
    if selection is None:
        lines.append(rf"\multicolumn{{5}}{{c}}{{{MISSING}}} \\")
    else:
        def rsum(e, k):
            v = e.get(k)
            return f"{v:g}" if _finite(v) else "--"
        first_block = True
        for method in MODEL_METHODS.get(model, ()):
            rec = selection.get("methods", {}).get(method)
            spec = METHOD_SPECS[method]
            if rec is None:
                entries = [("--", None)]
            else:
                entries = [(c, rec[c]) for c in CRITERIA if c in rec]
                entries += [("anchor" if e.get("is_anchor") else "fixed", e) for e in rec.get("fixed", [])]
            if not first_block:
                lines.append(r"\addlinespace")
            first_block = False
            n = rec.get("n_configs") if rec else None
            for i, (crit, e) in enumerate(entries):
                name = (spec.display + (f" ({n})" if n else "")) if i == 0 else ""
                if e is None:
                    lines.append(f"{name} & -- & {MISSING} & {MISSING} & {MISSING} \\\\")
                    continue
                single = not rec.get("tunable", True)
                lines.append(f"{name} & {crit} & {config_latex(method, e['params'])} & "
                             f"{'--' if single else rsum(e, 'rank_sum_loc')} & "
                             f"{'--' if single else rsum(e, 'rank_sum_faith')} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(path, lines)


# ---------------------------------------------------------------------------
# numbers.tex
# ---------------------------------------------------------------------------
NUMBER_MACROS = ("vitRGlocAL", "vitRGlocPG", "vitRGlocTK", "vitRGlocPFtwenty", "vitRGlocSel",
                 "vitRGlocMaxS", "vitRGlocTime", "vitRGfaithPFtwenty", "vitRGfaithSel",
                 "vitRGanchorMaxS", "vitRGanchorTime",
                 "vitRGEqlocAL", "vitRGEqlocTK", "vitRGEqlocPG", "vitRGEqanchorMaxS",
                 "vitBestBaseAL", "vitBestBaseALname", "vitBestBasePFtwenty", "vitBestBasePFtwentyname",
                 "vitBestBaseSel", "vitBestBaseSelname", "vitBestBaseTK", "vitBestBaseTKname",
                 "vitBestBasePG", "vitBestBasePGname", "vitBestBaseMaxS", "vitBestBaseMaxSname",
                 "vitRGEqTime", "vitDAVETime", "vitLIMETime", "vitSHAPTime",
                 "vocRGlocAL", "vocRGEqlocAL", "vocBestBaseAL", "vocBestBaseALname", "nValGrid",
                 "vggRGlocAL", "vggSGlocMaxS", "vggBestBaseAL", "vggBestBaseALname",
                 "vggBestBaseMaxS", "vggBestBaseMaxSname",
                 "resRGlocAL", "resBestBaseAL", "resBestBaseALname")


def _best_baseline(rs: Optional[ResultSet], metric: str) -> Tuple[Optional[float], Optional[str]]:
    from bwgame.evaluation.metric_spec import metric_direction
    if rs is None or rs.empty:
        return None, None
    vals = [(rs.mean(l, metric), l) for l in rs.recommended_rows(False)]
    vals = [(v, l) for v, l in vals if v is not None]
    if not vals:
        return None, None
    v, l = (max if metric_direction(metric) == "max" else min)(vals, key=lambda t: t[0])
    return v, get_spec(rs.rows[l]["method"]).display


def numbers(vit: Optional[ResultSet], voc: Optional[ResultSet], n_val: Optional[int],
            vgg: Optional[ResultSet] = None, res: Optional[ResultSet] = None) -> Dict[str, Optional[str]]:
    nums: Dict[str, Optional[str]] = {k: None for k in NUMBER_MACROS}

    def role(rs, method, crit):
        if rs is None:
            return None
        r = rs.role_labels(method)
        if crit == "anchor":
            return r["anchor"][0] if r["anchor"] else None
        if crit == "fixed":
            return r["fixed"][0] if r["fixed"] else (r.get("loc"))
        return r.get(crit)

    if vit is not None and not vit.empty:
        loc, faith, anc = role(vit, "rg", "loc"), role(vit, "rg", "faith"), role(vit, "rg", "anchor")
        nums["vitRGlocAL"] = fnum(vit.mean(loc, "AL"), "AL")
        nums["vitRGlocPFtwenty"] = fnum(vit.mean(loc, "PF20"), "PF20")
        nums["vitRGlocSel"] = fnum(vit.mean(loc, "Sel"), "Sel")
        nums["vitRGlocPG"] = fnum(vit.mean(loc, "PG"), "PG")
        nums["vitRGlocTK"] = fnum(vit.mean(loc, "TK"), "TK")
        nums["vitRGfaithPFtwenty"] = fnum(vit.mean(faith, "PF20"), "PF20")
        nums["vitRGfaithSel"] = fnum(vit.mean(faith, "Sel"), "Sel")
        nums["vitRGanchorMaxS"] = fnum(vit.mean(anc, "MaxS"), "MaxS")
        nums["vitRGlocMaxS"] = fnum(vit.mean(loc, "MaxS"), "MaxS")
        for key, metric in (("AL", "AL"), ("PFtwenty", "PF20"), ("Sel", "Sel")):
            v, name = _best_baseline(vit, metric)
            nums[f"vitBestBase{key}"] = fnum(v, metric)
            nums[f"vitBestBase{key}name"] = name
        eql = role(vit, "rgeq", "loc")
        for key in ("AL", "TK", "PG"):
            nums[f"vitRGEqloc{key}"] = fnum(vit.mean(eql, key), key)
        nums["vitRGEqanchorMaxS"] = fnum(vit.mean(role(vit, "rgeq", "anchor"), "MaxS"), "MaxS")
        for key in ("TK", "PG", "MaxS"):
            v, name = _best_baseline(vit, key)
            nums[f"vitBestBase{key}"] = fnum(v, key)
            nums[f"vitBestBase{key}name"] = name
        nums["vitRGlocTime"] = fnum(vit.time(loc), "time")
        nums["vitRGanchorTime"] = fnum(vit.time(anc), "time")
        nums["vitRGEqTime"] = fnum(vit.time(role(vit, "rgeq", "loc")), "time")
        nums["vitDAVETime"] = fnum(vit.time(role(vit, "dave", "loc")), "time")
        nums["vitLIMETime"] = fnum(vit.time(role(vit, "lime", "fixed")), "time")
        nums["vitSHAPTime"] = fnum(vit.time(role(vit, "kernelshap", "fixed")), "time")
    if voc is not None and not voc.empty:
        nums["vocRGlocAL"] = fnum(voc.mean(role(voc, "rg", "loc"), "AL"), "AL")
        nums["vocRGEqlocAL"] = fnum(voc.mean(role(voc, "rgeq", "loc"), "AL"), "AL")
        v, name = _best_baseline(voc, "AL")
        nums["vocBestBaseAL"] = fnum(v, "AL")
        nums["vocBestBaseALname"] = name
    if vgg is not None and not vgg.empty:
        nums["vggRGlocAL"] = fnum(vgg.mean(role(vgg, "rg", "loc"), "AL"), "AL")
        nums["vggSGlocMaxS"] = fnum(vgg.mean(role(vgg, "sg", "loc"), "MaxS"), "MaxS")
        for key in ("AL", "MaxS"):
            v, name = _best_baseline(vgg, key)
            nums[f"vggBestBase{key}"] = fnum(v, key)
            nums[f"vggBestBase{key}name"] = name
    if res is not None and not res.empty:
        nums["resRGlocAL"] = fnum(res.mean(role(res, "rg", "loc"), "AL"), "AL")
        v, name = _best_baseline(res, "AL")
        nums["resBestBaseAL"] = fnum(v, "AL")
        nums["resBestBaseALname"] = name
    if n_val:
        nums["nValGrid"] = f"{n_val:,}".replace(",", "{,}")
    return nums


def write_numbers(nums: Dict[str, Optional[str]], out: Path) -> None:
    lines = [r"\global\prelimfalse",
             "% Generated by code/reproduce/make_tables.py -- do not edit by hand.",
             "% Missing data are shown as a red ??.  nValGrid counts the evaluated ImageNet-S",
             "% validation configurations summed over the models with a selected.json.",
             "% Best-baseline macros consider the reported baseline rows (localization selection or the",
             "% single fixed configuration of each method), as in the main table."]
    for k in NUMBER_MACROS:
        v = nums.get(k)
        lines.append(rf"\newcommand{{\{k}}}{{{v if v is not None else MISSING}}}")
    _write(out / "numbers.tex", lines)


# ---------------------------------------------------------------------------
def _write(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s", path)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Write the paper's generated tables.")
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/tables"))
    ap.add_argument("--models", type=str, default="vit_b16,vgg16,resnet50")
    ap.add_argument("--voc", type=Path, default=None, help="Default: <results>/voc if it exists.")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict", action="store_true",
                    help="Abort if any table cell is missing/NaN or any reported test row is "
                         "incomplete (coverage < 1, zero or failed maps).  Use for the paper.")
    args = ap.parse_args(argv)
    setup_logging()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in MODEL_FILE]
    if unknown:
        raise SystemExit(f"--models: unknown {unknown}")
    sets: Dict[str, Optional[ResultSet]] = {}
    selections: Dict[str, Optional[Dict[str, Any]]] = {}
    for m in MODEL_FILE:
        sel = load_selection(args.results / "selection" / m / "selected.json") if m in models else None
        selections[m] = sel
        tdir = args.results / "test" / m
        sets[m] = (ResultSet(tdir, sel, args.results / "runtime" / f"{m}.csv", args.n_boot, args.seed)
                   if m in models else None)
        if sets[m] is not None and sets[m].empty:
            logger.warning("No test results in %s", tdir)

    voc_root = args.voc or (args.results / "voc")
    voc = None
    if voc_root.exists():
        voc_sel = load_selection(voc_root / "selection" / "selected.json")
        voc = ResultSet(voc_root / "test", voc_sel, None, args.n_boot, args.seed)
    else:
        logger.warning("No VOC results at %s", voc_root)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    empty_vit = ResultSet(args.results / "test" / "__none__", None, None, args.n_boot, args.seed)
    vit = sets.get("vit_b16")
    write_main_vit(vit if vit is not None else empty_vit, out)
    for m in MODEL_FILE:
        write_ci_longtable(sets.get(m), m, out / f"test_{MODEL_FILE[m]}_ci_longtable.tex")
        write_val_selection(selections.get(m), m, out / f"val_selection_{MODEL_FILE[m]}_tabular.tex")
        write_ablation_longtable(sets.get(m), m, out / f"ablations_{MODEL_FILE[m]}_longtable.tex")
    # Always written (red ?? placeholder row when there are no VOC results).
    write_ci_longtable(voc, "vit_b16", out / "test_voc_vit_b16_ci_longtable.tex")
    write_attention_rule(vit, out / "attention_rule_vit_b16_tabular.tex")
    n_val = sum(int(s.get("n_val_configs", 0)) for s in selections.values() if s) or None
    write_numbers(numbers(vit, voc, n_val, sets.get("vgg16"), sets.get("resnet50")), out)
    if args.strict:
        problems = strict_check(out, [args.results / "test" / m for m in models]
                                + ([voc_root / "test"] if voc_root.exists() else []))
        if problems:
            for p_ in problems:
                logger.error("strict: %s", p_)
            raise SystemExit(f"--strict: {len(problems)} problem(s); the tables are not paper-ready")
        logger.info("strict: no missing/NaN cells, all reported rows complete")


def strict_check(out: Path, test_dirs) -> List[str]:
    """Missing/NaN cells in the written files and incomplete reported test rows."""
    import re
    problems = []
    for f in sorted(out.iterdir()):
        if f.suffix not in (".tex", ".csv"):
            continue
        text = f.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("%"):
                continue
            if "??" in line or re.search(r"(?i)(?<![a-z])nan(?![a-z])", line):
                problems.append(f"{f.name}:{i}: {line.strip()[:120]}")
    for d in test_dirs:
        if not Path(d).is_dir():
            continue
        for jf in sorted(Path(d).glob("*.json")):
            if jf.name.endswith(".incomplete.json"):
                continue
            try:
                obj = json.loads(jf.read_text())
            except Exception:
                continue
            if not isinstance(obj, dict) or obj.get("kind") != "row_result":
                continue            # e.g. the row list voc_test_rows.json
            if int(obj.get("zero_maps", 0) or 0) or int(obj.get("map_failures", 0) or 0):
                problems.append(f"{jf}: zero_maps={obj.get('zero_maps')} "
                                f"map_failures={obj.get('map_failures')}")
            for k, v in obj.get("metrics", {}).items():
                if v.get("coverage") is None or float(v["coverage"]) < 1.0 or v.get("mean") is None \
                        or not math.isfinite(float(v["mean"])):
                    problems.append(f"{jf}: {k} mean={v.get('mean')} coverage={v.get('coverage')}")
            # Reported ImageNet-S rows need all metrics (ablation rows and the VOC
            # transfer are evaluated without MaxS / AvgS by design).
            from reproduce.bootstrap_ci import is_sweep_only
            if obj.get("dataset") != "voc" and not is_sweep_only(obj):
                missing = [k for k in METRICS if k not in obj.get("metrics", {})]
                if missing:
                    problems.append(f"{jf}: metrics missing {missing}")
    return problems


if __name__ == "__main__":
    sys.exit(main())
