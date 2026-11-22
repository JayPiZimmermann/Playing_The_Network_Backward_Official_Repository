"""Generate the corrected paired ViT randomization table from raw records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


CONDITIONS = ("identity", "head", "aligned", "endpoint")
DISPLAY = {
    "identity": "Identity",
    "head": "Head only",
    "aligned": "Last block",
    "endpoint": "Cascade endpoint",
}


def records(path: Path):
    rows = []
    for p in sorted(path.glob("[0-9][0-9][0-9][0-9].json")):
        rows.append(json.loads(p.read_text()))
    return rows


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    assert len(a) == 200 and np.isfinite(a).all()
    return float(a.mean()), float(a.std(ddof=1))


def corr_ci(x, y, seed=42):
    x, y = np.asarray(x, float), np.asarray(y, float)
    assert len(x) == len(y) == 200 and np.isfinite(x).all() and np.isfinite(y).all()
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return None, None
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(2000):
        idx = rng.integers(0, len(x), len(x))
        if np.ptp(x[idx]) and np.ptp(y[idx]):
            boot.append(float(spearmanr(x[idx], y[idx]).statistic))
    return rho, np.quantile(boot, [.025, .975]).tolist()


def summarize(root: Path):
    out = {}
    common_indices = None
    for condition in CONDITIONS:
        rr = records(root / condition)
        if len(rr) != 200:
            raise RuntimeError(f"{condition}: expected 200 records, found {len(rr)}")
        indices = sorted(r["index"] for r in rr)
        if common_indices is None:
            common_indices = indices
        elif indices != common_indices:
            raise RuntimeError("Conditions do not use the same image subset")
        for r in rr:
            if r["condition"]["name"] != condition:
                raise RuntimeError("Condition mismatch")
            if r["hashes"] != rr[0]["hashes"]:
                raise RuntimeError(f"{condition}: state hashes vary across image records")
        hkeys = ("H_adf_plain", "H_adf_surv", "H_rg_plain", "H_rg_surv")
        h = {k: mean_sd([r["trajectory"][k] for r in rr]) for k in hkeys}
        maps = {
            "gradient_spearman": mean_sd([r["maps"]["gradient"]["spearman_abs"] for r in rr]),
            "rg_spearman": mean_sd([r["maps"]["rg"]["spearman_abs"] for r in rr]),
            "abs_pf20_change": mean_sd([abs(r["maps"]["rg"]["PF20_change"]) for r in rr]),
        }
        rho, ci = corr_ci(
            [r["trajectory"]["H_rg_plain"] for r in rr],
            [abs(r["maps"]["rg"]["PF20_change"]) for r in rr],
        )
        rho_map, ci_map = corr_ci(
            [r["trajectory"]["H_rg_plain"] for r in rr],
            [r["maps"]["rg"]["spearman_abs"] for r in rr],
        )
        out[condition] = {
            "n": 200,
            "indices": indices,
            "hashes": rr[0]["hashes"],
            "hellinger": h,
            "maps": maps,
            "rho_H_rg_plain_abs_PF20_change": {"rho": rho, "ci95": ci},
            "rho_H_rg_plain_RG_map_spearman": {"rho": rho_map, "ci95": ci_map},
        }
    return out


def add_prediction_performance(summary, prediction_path: Path):
    """Join the independently recomputed predictions after verifying exact pairing."""
    prediction = json.loads(prediction_path.read_text())
    if prediction.get("protocol") != "exact paired diagnostic checkpoints and image indices":
        raise RuntimeError("Unexpected prediction-performance protocol")
    for condition in CONDITIONS:
        p = prediction["conditions"][condition]
        if p["reference"]["n"] != 200 or p["randomized"]["n"] != 200:
            raise RuntimeError(f"{condition}: prediction count is not 200")
        if sorted(p["indices"]) != summary[condition]["indices"]:
            raise RuntimeError(f"{condition}: prediction and diagnostic image indices differ")
        if p["hashes"] != summary[condition]["hashes"]:
            raise RuntimeError(f"{condition}: prediction and diagnostic state hashes differ")
        summary[condition]["prediction_performance"] = {
            "checkpoint_sha256": p["checkpoint_sha256"],
            "manifest_sha256": p["manifest_sha256"],
            "reference": p["reference"],
            "randomized": p["randomized"],
            "paired_top1_change": p["paired_top1_change"],
            "paired_target_probability_change": p["paired_target_probability_change"],
        }
    return summary


def pm(pair, digits=3):
    return f"${pair[0]:.{digits}f} \\pm {pair[1]:.{digits}f}$"


def rho_cell(item):
    if item["rho"] is None:
        return "$--$"
    lo, hi = item["ci95"]
    return f"${item['rho']:+.3f}$ \\tiny{{[{lo:+.3f},{hi:+.3f}]}}"


def render(summary):
    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        r"Condition & $\mathrm{H}_{\mathrm{SG}}$ & $\mathrm{H}_{\mathrm{SG,surv}}$ & $\mathrm{H}_{\mathrm{RG}}$ & $\mathrm{H}_{\mathrm{RG,surv}}$ & Grad. $\rho(|a|)$ & RG $\rho(|a|)$ & $|\Delta\mathrm{PF}_{20}|$ & $\rho(\mathrm{H}_{\mathrm{RG}},\rho_{\mathrm{map}})$ & $\rho(\mathrm{H}_{\mathrm{RG}},|\Delta\mathrm{PF}_{20}|)$ \\",
        r"\midrule",
    ]
    for condition in CONDITIONS:
        s = summary[condition]
        lines.append(" & ".join([
            DISPLAY[condition],
            pm(s["hellinger"]["H_adf_plain"]),
            pm(s["hellinger"]["H_adf_surv"]),
            pm(s["hellinger"]["H_rg_plain"]),
            pm(s["hellinger"]["H_rg_surv"]),
            pm(s["maps"]["gradient_spearman"]),
            pm(s["maps"]["rg_spearman"]),
            pm(s["maps"]["abs_pf20_change"]),
            rho_cell(s["rho_H_rg_plain_RG_map_spearman"]),
            rho_cell(s["rho_H_rg_plain_abs_PF20_change"]),
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def render_prediction(summary):
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Condition & Top-1 acc. & Mean $p(y)$ & RG map $\rho(|a|)$ & $\mathrm{H}_{\mathrm{RG}}$ \\",
        r"\midrule",
    ]
    for condition in CONDITIONS:
        s = summary[condition]
        pred = s["prediction_performance"]["randomized"]
        lines.append(" & ".join([
            DISPLAY[condition],
            f"${100 * pred['top1_accuracy']:.1f}\\%$",
            f"${pred['mean_target_probability']:.6f}$",
            f"${s['maps']['rg_spearman'][0]:.6f}$",
            f"${s['hellinger']['H_rg_plain'][0]:.3f}$",
        ]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("results/controls/diagnostics/vit_b16/paired"))
    ap.add_argument("--prediction-summary", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--tex", type=Path, required=True)
    ap.add_argument("--prediction-tex", type=Path, required=True)
    args = ap.parse_args()
    summary = summarize(args.root)
    summary = add_prediction_performance(summary, args.prediction_summary)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.prediction_tex.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n")
    args.tex.write_text(render(summary))
    args.prediction_tex.write_text(render_prediction(summary))
    print(json.dumps({c: summary[c]["hellinger"] for c in CONDITIONS}, indent=2))


if __name__ == "__main__":
    main()
