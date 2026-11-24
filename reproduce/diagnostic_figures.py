#!/usr/bin/env python3
"""Figures and tables of the randomisation diagnostics (appendix).

Reads the CSVs written by

* ``experiments.hellinger_randomization.run``            (cascading Hellinger),
* ``experiments.hellinger_randomization.run_double_random`` (double-random baseline),
* ``experiments.hellinger_input_noise.run``, ``run_clean_pairs`` and
  ``run_random_baselines``                              (input noise, reference pairs),
* ``experiments.sanity_check_randomization.run``        (map-similarity cascade),

and writes

* the coordinate lists of the three pgfplots figures under
  ``figures/plots/`` of the paper (layout and styling are kept; only the data,
  and the reference lines change):
  ``sanity_check/hellinger_live_plots.inc.tex``,
  ``hellinger_input_noise/hellinger_input_noise_plots.inc.tex``;
* tables under ``tables/generated/``: ``sanity_check_tabular.tex``,
  ``double_random_tabular.tex``, ``per_image_corr_tabular.tex``,
  ``input_baseline_tabular.tex`` and the macros ``diagnostic_numbers.tex``.

Missing inputs are written as a red ``??``.  No torch needed.

Where an experiment output directory does not exist, the recorded measurements
shipped in ``results_summary/diagnostics/`` are read instead (written there by
``reproduce.collect_diagnostics``; gzip-compressed CSVs are read transparently).

Usage::

    python -m reproduce.diagnostic_figures --paper <paper dir>
    python -m reproduce.diagnostic_figures --out tables          # tables and macros only
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from reproduce.common import METHOD_SPECS, MODEL_METHODS, REPO, config_latex, setup_logging

logger = logging.getLogger("reproduce.diagnostic_figures")

MISSING = r"\textcolor{red}{??}"
MODELS = (("vgg16", "VGG-16"), ("resnet50", "ResNet-50"), ("vit_b_16", "ViT-B/16"))
SELECTION_MODEL = {"vit_b_16": "vit_b16"}
EXP = REPO / "experiments"
SUMMARY = REPO / "results_summary"
COORDS = re.compile(r"coordinates \{([^}]*)\}")
PAIR = re.compile(r"\(([^,()]+),\s*([^()]+)\)")

# Steps at which the Hellinger cascade and the map-similarity cascade have
# randomised the same parameters (VGG-16, ResNet-50) or the same block
# (ViT-B/16: the Hellinger cascade re-initialises the MLP and the output
# projection of a block, the map cascade additionally Q, K and V).
ALIGNED_STEPS = {"vgg16": ("features.26", "features.26"),
                 "resnet50": ("layer4.1.conv3", "layer4.1.conv3"),
                 "vit_b_16": ("encoder.layers.encoder_layer_11.self_attention.out_proj",
                              "encoder.layers.encoder_layer_11.self_attention.q_proj")}
H_VARIANTS = (("rg", "hellinger_distance", r"$\mathrm{H}_{\mathrm{RG,plain}}$"),
              ("rg", "hellinger_live", r"$\mathrm{H}_{\mathrm{RG,surv}}$"),
              ("adf", "hellinger_distance", r"$\mathrm{H}_{\mathrm{SG,plain}}$"),
              ("adf", "hellinger_live", r"$\mathrm{H}_{\mathrm{SG,surv}}$"))
MAP_VARIANTS = (("gradient__fixed", "spearman_abs"), ("gradient__fixed", "hog_pearson"),
                ("rg__anchor", "spearman_abs"), ("rg__anchor", "hog_pearson"))


def _read(path: Path) -> Optional[pd.DataFrame]:
    """Read ``path`` or its gzip-compressed copy ``path.gz`` (as shipped in the summaries)."""
    for cand in (path, path.with_name(path.name + ".gz")):
        if cand.exists():
            return pd.read_csv(cand)
    logger.warning("missing %s", path)
    return None


def _default_dir(experiment: str, sub: str, shipped: str) -> Path:
    """The experiment output directory if it exists, else the shipped measurements."""
    live = EXP / experiment / sub
    return live if live.exists() else SUMMARY / "diagnostics" / shipped


def _selection(model: str) -> Dict:
    name = SELECTION_MODEL.get(model, model)
    for root in (REPO / "results", SUMMARY):
        path = root / "selection" / name / "selected.json"
        if path.exists():
            return json.loads(path.read_text())
    raise FileNotFoundError(f"no selected.json for {name} under results/ or results_summary/")


def _write(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    logger.info("wrote %s", path)


def _curve_style(line: str) -> Tuple[str, bool, bool]:
    """(game, survival-conditioned, reference line) of a template \\addplot line."""
    game = "adf" if "blue" in line else "rg"
    return game, "densely dotted" in line, "forget plot" in line


def _set_coords(line: str, pts: Sequence[Tuple[float, float]], xfmt: str = "{:g}") -> str:
    if len(pts) < 2:
        logger.warning("no data for a curve; kept the existing coordinates: %s", line.strip()[:60])
        return line
    m = COORDS.search(line)
    body = " ".join(f"({xfmt.format(x)}, {y:.10f})" for x, y in pts)
    return line[:m.start(1)] + body + line[m.end(1):]


def _rewrite(template: Path, per_panel) -> None:
    """Replace the data of every \\addplot of a 3-panel groupplot (VGG, ResNet, ViT)."""
    if not template.exists():
        logger.warning("no figure template %s; figure data not written", template)
        return
    out, panel = [], -1
    for line in template.read_text().splitlines():
        s = line.strip()
        if s.startswith(r"\nextgroupplot"):
            panel += 1
            line = per_panel(MODELS[panel][0], None, line)
        elif s.startswith(r"\addplot"):
            line = per_panel(MODELS[panel][0], _curve_style(s), line)
        out.append(line)
    _write(template, out)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def cascade_figures(hdir: Path, drdir: Path, paper: Path) -> None:
    avg = _read(hdir / "avg.csv")
    dr = _read(drdir / "aggregate.csv")
    if avg is None:
        return

    def series(df, model, game, col):
        s = df[(df.model == model) & (df.game == game)].sort_values("step_idx")
        return [(0, 0.0)] + [(int(r.step_idx) + 1, float(getattr(r, col))) for r in s.itertuples()]

    def mean_panel(model, style, line):
        if style is None:
            return line
        game, surv, ref = style
        if ref:
            if dr is None:
                return line
            r = dr[(dr.model == model) & (dr.game == game)].iloc[0]
            y = float(r.h_live_mean if surv else r.h_plain_mean)
            return re.sub(r"\((\d+), [0-9.]+\)", lambda m: f"({m.group(1)}, {y:.4f})", line)
        return _set_coords(line, series(avg, model, game, "avg_hellinger_live" if surv else "avg_hellinger"))

    _rewrite(paper / "figures/plots/sanity_check/hellinger_live_plots.inc.tex", mean_panel)


def input_noise_figure(ndir: Path, paper: Path) -> None:
    avg = _read(ndir / "avg.csv")
    if avg is None:
        return
    sig = sorted(avg.sigma.round(6).unique())

    def panel(model, style, line):
        if style is None:
            line = re.sub(r"xmax=[0-9.]+", f"xmax={max(sig):.3f}", line)
            line = re.sub(r"xmin=[0-9.]+", f"xmin={min(sig) * 0.9:g}", line)
            return re.sub(r"xtick=\{[^}]*\}", "xtick={" + ", ".join(f"{s:g}" for s in sig) + "}", line)
        game, surv, _ = style
        d = avg[(avg.model == model) & (avg.game == game)].sort_values("sigma")
        col = "avg_hellinger_live" if surv else "avg_hellinger"
        return _set_coords(line, list(zip(d.sigma, d[col])), xfmt="{:.6f}")

    _rewrite(paper / "figures/plots/hellinger_input_noise/hellinger_input_noise_plots.inc.tex", panel)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def _sanity_rows(model: str) -> List[Tuple[str, str]]:
    """(cache label, display name) of the rows of the sanity table for one model."""
    sel = _selection(model)
    rows = []
    for method in MODEL_METHODS[SELECTION_MODEL.get(model, model)]:
        rec = sel["methods"].get(method)
        if rec is None:
            continue
        spec = METHOD_SPECS[method]
        if method == "rgeq":
            continue                                   # 50 transformed inputs per map
        if spec.ours:
            for e in rec.get("fixed", []):
                if e.get("is_anchor"):
                    rows.append((f"{method}__anchor", f"{spec.display} anchor"))
            if "loc" in rec:
                rows.append((f"{method}__loc",
                             f"{spec.display} ({config_latex(method, rec['loc']['params'])})"))
        elif "loc" in rec:
            rows.append((f"{method}__loc", spec.display))
        elif rec.get("fixed"):
            rows.append((f"{method}__fixed", spec.display))
    return rows


# Full-randomization similarities quoted in the text (\sanity<prefix><key><Rho|HOG>):
# per model the rows that keep the most map structure, and the RG rows.
SANITY_MACROS = {
    "resnet50": ("Res", (("lrp_gamma__loc", "LRPgamma"), ("lrp_eps__loc", "LRPeps"))),
    "vgg16": ("Vgg", (("lrp_eps__loc", "LRPeps"),)),
    "vit_b_16": ("Vit", (("rg__anchor", "RGanchor"), ("rg__loc", "RGloc"))),
}


def sanity_table(sdir: Path, out: Path) -> Dict[str, str]:
    """Similarity at full randomisation; returns macros for the text."""
    avg, per = _read(sdir / "avg.csv"), _read(sdir / "per_image.csv")
    macros: Dict[str, str] = {}
    lines = [r"\begin{tabular}{l c c}", r"\toprule",
             r"Method & Spear.\ $\rho(|a|)$ & HOG $r$ \\", r"\midrule"]
    for mi, (model, name) in enumerate(MODELS):
        lines += [rf"\multicolumn{{3}}{{l}}{{\textbf{{{name}}}}} \\", r"\midrule"]
        rows = _sanity_rows(model)
        vals: Dict[Tuple[str, str], Optional[float]] = {}
        const: Dict[str, bool] = {}
        if avg is not None:
            a = avg[avg.model == model]
            last = int(a.step_idx.max()) if len(a) else -1
            for label, _ in rows:
                for metric in ("spearman_abs", "hog_pearson"):
                    r = a[(a.method_label == label) & (a.metric == metric) & (a.step_idx == last)]
                    vals[(label, metric)] = float(r["mean"].iloc[0]) if len(r) else None
                if per is not None:
                    p = per[(per.model == model) & (per.method_label == label)
                            & (per.step_idx == last) & (per.metric == "spearman_abs")]
                    const[label] = bool(len(p)) and bool((p.similarity == 0.0).all())
        # closest to zero, among methods whose maps are not constant at full randomisation
        best = {m: min((abs(v) for (l, mm), v in vals.items()
                        if mm == m and v is not None and not const.get(l)), default=None)
                for m in ("spearman_abs", "hog_pearson")}
        for label, disp in rows:
            cells = []
            for m in ("spearman_abs", "hog_pearson"):
                v = vals.get((label, m))
                if v is None:
                    cells.append(MISSING)
                    continue
                s = f"{v:.3f}"
                is_best = best[m] is not None and abs(v) == best[m] and not const.get(label)
                cells.append(r"\textbf{" + s + "}" if is_best else s)
            mark = r"$^{\ddagger}$" if const.get(label) else ""
            lines.append(f"{disp}{mark} & {cells[0]} & {cells[1]} \\\\")
        if mi < len(MODELS) - 1:
            lines.append(r"\midrule")
        prefix, entries = SANITY_MACROS.get(model, ("", ()))
        for label, key in entries:
            for metric, suffix in (("spearman_abs", "Rho"), ("hog_pearson", "HOG")):
                if vals.get((label, metric)) is not None:
                    macros[f"sanity{prefix}{key}{suffix}"] = f"{vals[(label, metric)]:.3f}"
    lines += [r"\bottomrule", r"\end{tabular}"]
    _write(out / "sanity_check_tabular.tex", lines)
    return macros


def double_random_rows(drdir: Path, out: Path) -> Dict[str, str]:
    dr = _read(drdir / "aggregate.csv")
    rounds = _read(drdir / "per_round.csv")
    lines, macros = [], {}
    for model, name in MODELS:
        cells = []
        for game in ("adf", "rg"):
            r = None if dr is None else dr[(dr.model == model) & (dr.game == game)]
            if r is None or not len(r):
                cells += [MISSING, MISSING]
                continue
            r = r.iloc[0]
            # H <= 1 by definition; clip float round-off of the saturated cells
            cells += [f"${min(r.h_plain_mean, 1.0):.3f} \\pm {r.h_plain_std:.3f}$",
                      f"${min(r.h_live_mean, 1.0):.3f} \\pm {r.h_live_std:.3f}$"]
        lines.append(f"    {name} & " + " & ".join(cells) + r" \\")
    head = [r"\begin{tabular}{lcccc}", r"  \toprule",
            r"  & \multicolumn{2}{c}{Stopping Game} & \multicolumn{2}{c}{Routing Game} \\",
            r"  \cmidrule(lr){2-3} \cmidrule(lr){4-5}",
            r"  Architecture & $\mathrm{H}$ & $\mathrm{H}_{\mathrm{surv}}$ & $\mathrm{H}$ & $\mathrm{H}_{\mathrm{surv}}$ \\",
            r"  \midrule"]
    _write(out / "double_random_tabular.tex", head + lines + [r"  \bottomrule", r"\end{tabular}"])
    if rounds is not None:
        col = next((c for c in ("image_id", "image_name", "sample_id") if c in rounds.columns), None)
        if col is not None:
            per_model = rounds.groupby("model")[col].nunique()
            macros["drDistinctImages"] = str(int(per_model.max()))
    if dr is not None:
        plain = dr.h_plain_mean.clip(upper=1.0)
        floor = plain[plain < 0.9995]
        if len(floor):
            macros["drFloorMin"] = f"{floor.min():.2f}"
            macros["drFloorMax"] = f"{floor.max():.2f}"
    return macros


def per_image_corr_rows(hdir: Path, sdir: Path, out: Path) -> Dict[str, str]:
    """Spearman correlations across test images at the aligned cascade steps."""
    H = _read(hdir / "per_image.csv")
    S = _read(sdir / "per_image.csv")
    lines, macros = [], {}
    cross_max, n_images = 0.0, set()
    for mi, (model, name) in enumerate(MODELS):
        if mi:
            lines.append(r"    \midrule")
        rho = None
        if H is not None and S is not None:
            hl, sl = ALIGNED_STEPS[model]
            h = H[(H.model == model) & (H.last_randomized_layer == hl)]
            s = S[(S.model == model) & (S.last_randomized_layer == sl)]
            cols = {}
            for game, col, disp in H_VARIANTS:
                cols[disp] = h[h.game == game].set_index("sample_id")[col]
            for label, metric in MAP_VARIANTS:
                cols[f"{label}|{metric}"] = s[(s.method_label == label) & (s.metric == metric)] \
                    .set_index("sample_id")["similarity"]
            df = pd.DataFrame(cols).dropna()
            if len(df) > 2 and all(len(c) for c in cols.values()):
                rho = df.rank().corr()
                n_images.add(len(df))
                hk = [d for _, _, d in H_VARIANTS]
                mk = [f"{l}|{m}" for l, m in MAP_VARIANTS]
                cross_max = max(cross_max, float(rho.loc[hk, mk].abs().to_numpy().max()))
        for j, (_, _, rn) in enumerate(H_VARIANTS):
            cells = []
            for key in [d for _, _, d in H_VARIANTS] + [f"{l}|{m}" for l, m in MAP_VARIANTS]:
                if key == rn:
                    cells.append("$-$    ")
                elif rho is None:
                    cells.append(MISSING)
                else:
                    cells.append(f"${rho.loc[rn, key]:+.2f}$")
            lead = rf"\multirow{{4}}{{*}}{{{name}}}" if j == 0 else " " * 24
            lines.append(f"    {lead} & {rn} & " + " & ".join(cells) + r" \\")
    head = [r"\begin{tabular}{ll cccc cccc}", r"  \toprule",
            r"  & & \multicolumn{4}{c}{Within Hellinger} & \multicolumn{2}{c}{Gradient} & \multicolumn{2}{c}{$\mathrm{RG}$ anchor} \\",
            r"  \cmidrule(lr){3-6} \cmidrule(lr){7-8} \cmidrule(lr){9-10}",
            r"  Architecture & Hellinger variant",
            r"    & $\mathrm{H}_{\mathrm{RG,pl}}$ & $\mathrm{H}_{\mathrm{RG,surv}}$",
            r"    & $\mathrm{H}_{\mathrm{SG,pl}}$ & $\mathrm{H}_{\mathrm{SG,surv}}$",
            r"    & Spear.\,$\rho(|a|)$ & HOG\,$r$ & Spear.\,$\rho(|a|)$ & HOG\,$r$ \\",
            r"  \midrule"]
    _write(out / "per_image_corr_tabular.tex", head + lines + [r"  \bottomrule", r"\end{tabular}"])
    if n_images:
        macros["perImageCrossMax"] = f"{cross_max:.2f}"
        macros["perImageN"] = str(min(n_images))
    return macros


def input_baseline_rows(ndir: Path, out: Path) -> None:
    frames = [f for f in (_read(ndir / "clean_pair_baselines.csv"),
                          _read(ndir / "random_baselines.csv")) if f is not None]
    df = pd.concat(frames) if frames else None
    lines = []
    kinds = ("clean_intraclass", "clean_interclass", "clean_vs_random", "random_vs_random")
    for mi, (model, name) in enumerate(MODELS):
        if mi:
            lines.append(r"    \midrule")
        for j, kind in enumerate(kinds):
            cells = []
            for game in ("adf", "rg"):
                r = None if df is None else df[(df.model == model) & (df.game == game) & (df.baseline == kind)]
                if r is None or not len(r):
                    cells += [MISSING, MISSING]
                    continue
                r = r.iloc[0]
                cells += [f"${min(r.mean_H, 1.0):.3f} \\pm {r.std_H:.3f}$",
                          f"${min(r.mean_Hlive, 1.0):.3f} \\pm {r.std_Hlive:.3f}$"]
            lead = rf"\multirow{{4}}{{*}}{{{name}}}" if j == 0 else " " * 26
            lines.append(f"    {lead} & {kind.replace('_', '-'):18s} & " + " & ".join(cells) + r" \\")
    head = [r"\begin{tabular}{l l cc cc}", r"  \toprule",
            r"  & & \multicolumn{2}{c}{Stopping Game} & \multicolumn{2}{c}{Routing Game} \\",
            r"  \cmidrule(lr){3-4} \cmidrule(lr){5-6}",
            r"  Architecture & Baseline & $\mathrm{H}$ & $\mathrm{H}_{\mathrm{surv}}$ & $\mathrm{H}$ & $\mathrm{H}_{\mathrm{surv}}$ \\",
            r"  \midrule"]
    _write(out / "input_baseline_tabular.tex", head + lines + [r"  \bottomrule", r"\end{tabular}"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper", type=Path, default=None,
                    help="Directory of the paper sources (figures/plots, tables/generated).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Directory of the tables (default: <paper>/tables/generated). "
                         "Without --paper only the tables and macros are written.")
    ap.add_argument("--hellinger", type=Path,
                    default=_default_dir("hellinger_randomization", "results", "hellinger_randomization"))
    ap.add_argument("--double-random", type=Path,
                    default=_default_dir("hellinger_randomization", "results_double_random", "double_random"))
    ap.add_argument("--input-noise", type=Path,
                    default=_default_dir("hellinger_input_noise", "results", "hellinger_input_noise"))
    ap.add_argument("--sanity", type=Path,
                    default=_default_dir("sanity_check_randomization", "results", "sanity_check_randomization"))
    args = ap.parse_args(argv)
    setup_logging()
    if args.paper is None and args.out is None:
        ap.error("give --paper (figures and tables) or --out (tables only)")
    out = args.out if args.out is not None else args.paper / "tables" / "generated"
    if args.paper is not None:
        cascade_figures(args.hellinger, args.double_random, args.paper)
        input_noise_figure(args.input_noise, args.paper)
    macros: Dict[str, str] = {}
    macros.update(sanity_table(args.sanity, out))
    macros.update(double_random_rows(args.double_random, out))
    macros.update(per_image_corr_rows(args.hellinger, args.sanity, out))
    input_baseline_rows(args.input_noise, out)
    names = tuple(f"sanity{prefix}{key}{suffix}" for prefix, entries in SANITY_MACROS.values()
                  for _, key in entries for suffix in ("Rho", "HOG")) + (
        "drDistinctImages", "drFloorMin", "drFloorMax", "perImageCrossMax", "perImageN")
    _write(out / "diagnostic_numbers.tex",
           ["% Generated by code/reproduce/diagnostic_figures.py -- do not edit by hand."]
           + [rf"\newcommand{{\{k}}}{{{macros.get(k, MISSING)}}}" for k in names])


if __name__ == "__main__":
    main()
