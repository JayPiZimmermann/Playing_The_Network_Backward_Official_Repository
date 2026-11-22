"""Render the cached SG max-pooling control as a LaTeX tabular.

The renderer consumes the aggregate written by control_sg_pool_all_metrics.
It also discovers completed per-case summaries while a later case is still
running.  A case enters the table only after all metrics and paired deltas
contain all 50 validation images; partial measurements remain visibly pending.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


CASES = (
    ("anchor", r"Hard-gate $\lambda{=}0$"),
    ("loc", "Localization-selected"),
    ("faith", "Faithfulness-selected"),
)
ROWS = (("legacy_pp", "Recompute"), ("original_pool_pp", "Cached"))
METRICS = ("AL", "PG", "TK", "PF5", "PF20", "Sel", "MaxS", "AvgS", "time_s")


def _number(x: float, metric: str, signed: bool = False) -> str:
    precision = 4 if metric in ("AL", "PG", "TK", "MaxS", "AvgS") else 3
    if metric == "time_s":
        precision = 3
    if signed and x != 0 and abs(x) < 0.5 * 10 ** (-precision):
        return f"{x:+.2e}"
    return f"{x:+.{precision}f}" if signed else f"{x:.{precision}f}"


def _cell(stat: dict, metric: str, delta: bool = False) -> str:
    if not stat or stat.get("mean") is None:
        return r"\emph{pending}"
    mean = _number(float(stat["mean"]), metric, signed=delta)
    ci = stat.get("ci95")
    if delta and ci and len(ci) == 2:
        lo = _number(float(ci[0]), metric, signed=True)
        hi = _number(float(ci[1]), metric, signed=True)
        return rf"\begin{{tabular}}[c]{{@{{}}c@{{}}}}{mean}\\[-1pt]{{\tiny[{lo},{hi}]}}\end{{tabular}}"
    return mean


def render(summary: dict) -> str:
    cases = summary.get("cases", {})
    lines = [
        r"\begin{tabular}{lllrrrrrrrrr}",
        r"\toprule",
        r"Configuration & Routing & Statistic & AL & PG & TK & PF$_5$ & PF$_{20}$ & Sel & MaxS & AvgS & $t$/img \\",
        r"\midrule",
    ]
    for index, (key, label) in enumerate(CASES):
        if index:
            lines.append(r"\midrule")
        case = cases.get(key, {})
        if case.get("n") != 50:
            for row_index, (_, routing) in enumerate(ROWS):
                prefix = label if row_index == 0 else ""
                lines.append(
                    rf"{prefix} & {routing} & mean & \multicolumn{{9}}{{c}}{{\emph{{pending cached all-metric rerun}}}} \\"
                )
            lines.append(
                r" & Cached $-$ recompute & paired $\Delta$ & \multicolumn{9}{c}{\emph{pending cached all-metric rerun}} \\"
            )
        else:
            for row_index, (row_key, routing) in enumerate(ROWS):
                prefix = label if row_index == 0 else ""
                values = [_cell(case["rows"][row_key].get(metric, {}), metric) for metric in METRICS]
                lines.append(" & ".join((prefix, routing, "mean", *values)) + r" \\")
            values = [_cell(case["paired_deltas"].get(metric, {}), metric, delta=True) for metric in METRICS]
            lines.append(" & ".join(("", r"Cached $-$ recompute", r"paired $\Delta$", *values)) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}", ""))
    return "\n".join(lines)


def _case_complete(case: dict) -> bool:
    if case.get("n") != 50:
        return False
    rows = case.get("rows", {})
    if any(row not in rows for row, _ in ROWS):
        return False
    for row, _ in ROWS:
        if any(rows[row].get(metric, {}).get("n") != 50 for metric in METRICS):
            return False
    deltas = case.get("paired_deltas", {})
    return all(deltas.get(metric, {}).get("n") == 50 for metric in METRICS)


def load_completed_cases(summary_path: Path) -> dict:
    """Merge only complete cases from the root and resumable case summaries."""
    root = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    cases = dict(root.get("cases", {}))
    for key, _ in CASES:
        sibling = summary_path.parent / key / "summary.json"
        if sibling.exists():
            candidate = json.loads(sibling.read_text())
            if _case_complete(candidate):
                cases[key] = candidate
        if key in cases and not _case_complete(cases[key]):
            cases.pop(key)
    return {"protocol": root.get("protocol"), "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = load_completed_cases(args.summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(summary))
    print(f"wrote {args.out}; complete={all(summary.get('cases', {}).get(k, {}).get('n') == 50 for k, _ in CASES)}")


if __name__ == "__main__":
    main()
