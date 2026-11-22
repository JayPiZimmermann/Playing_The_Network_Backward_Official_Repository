"""Aggregate the full deletion/insertion curves and render the figures."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reproduce.control_common import atomic_json


ROWS = (
    ("lrp_gamma__eps-0.1__gamma-2", "LRP-$\\gamma$ (loc)"),
    ("lrp_gamma__eps-0.1__gamma-0.1", "LRP-$\\gamma$ (faith)"),
    ("attnlrp__eattn-0.25__gcomp-on__gconv-0.25__glin-0.05", "AttnLRP (loc)"),
    ("attnlrp__eattn-0.01__gcomp-on__gconv-0.25__glin-0.05", "AttnLRP (faith)"),
    ("rg__av-merged__eps-0.1__lam-0__lamsm-m10__tau-0.4__var-5", "RG (loc)"),
    ("rg__av-merged__eps-0.1__lam-0__lamsm-m5__tau-0.7__var-5", "RG (faith)"),
    ("rg__av-merged__eps-0.5__lam-0__lamsm-0__tau-1", "RG anchor"),
)
REFERENCE = "rg__av-merged__eps-0.1__lam-0__lamsm-m10__tau-0.4__var-5"
SELECTED = {
    "lrp_gamma__eps-0.1__gamma-2",
    "attnlrp__eattn-0.25__gcomp-on__gconv-0.25__glin-0.05",
    REFERENCE,
}
COLORS = {
    "lrp_gamma": "#0072B2",
    "attnlrp": "#D55E00",
    "rg": "#009E73",
}


def bootstrap_mean_ci(a: np.ndarray, *, seed: int, draws: int = 10_000):
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 250):
        stop = min(draws, start + 250)
        idx = rng.integers(0, len(a), size=(stop - start, len(a)))
        means[start:stop] = a[idx].mean(axis=1)
    return [float(v) for v in np.quantile(means, [.025, .975])]


def bootstrap_curve_ci(a: np.ndarray, *, seed: int, draws: int = 1_000):
    rng = np.random.default_rng(seed)
    means = np.empty((draws, a.shape[1]), dtype=np.float32)
    for start in range(0, draws, 50):
        stop = min(draws, start + 50)
        idx = rng.integers(0, len(a), size=(stop - start, len(a)))
        means[start:stop] = a[idx].mean(axis=1)
    return np.quantile(means, [.025, .975], axis=0)


def load_row(root: Path, label: str):
    folder = root / "vit_b16" / label
    files = sorted(folder.glob("[0-9][0-9][0-9][0-9].json"))
    records = [json.loads(p.read_text()) for p in files]
    if len(records) != 565:
        return None
    manifest_path = folder / "manifest_shard000.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Complete row has no manifest: {folder}")
    manifest = json.loads(manifest_path.read_text())
    records.sort(key=lambda r: r["index"])
    if len({r["index"] for r in records}) != 565:
        raise RuntimeError(f"Duplicate indices in {folder}")
    if manifest["indices"] != [r["index"] for r in records]:
        raise RuntimeError(f"Manifest index mismatch in {folder}")
    if manifest["row"]["row_label"] != label:
        raise RuntimeError(f"Manifest row mismatch in {folder}")
    fractions = np.asarray(records[0]["fractions"], dtype=float)
    for r in records:
        if r["protocol"] != "full-spatial-deletion-insertion-v1":
            raise RuntimeError(f"Protocol mismatch in {folder}")
        np.testing.assert_allclose(r["fractions"], fractions, rtol=0, atol=0)
        # The same endpoint can differ at the last few float32 bits because it
        # appears in a different inference batch in the two curve directions.
        for score in ("logits", "probabilities"):
            np.testing.assert_allclose(
                r[f"deletion_{score}"][0], r[f"insertion_{score}"][-1],
                rtol=1e-5, atol=2e-6)
            np.testing.assert_allclose(
                r[f"deletion_{score}"][-1], r[f"insertion_{score}"][0],
                rtol=1e-5, atol=2e-6)
        for mode in ("deletion", "insertion"):
            for score, curve_suffix in (("logit", "logits"),
                                        ("probability", "probabilities")):
                recomputed = np.trapz(
                    r[f"{mode}_{curve_suffix}"], r["fractions"])
                np.testing.assert_allclose(
                    r[f"{mode}_{score}_auc"], recomputed,
                    rtol=1e-12, atol=1e-12)
    arrays = {key: np.asarray([r[key] for r in records], dtype=float)
              for key in ("deletion_logits", "deletion_probabilities",
                          "insertion_logits", "insertion_probabilities")}
    aucs = {key: np.asarray([r[key] for r in records], dtype=float)
            for key in ("deletion_logit_auc", "deletion_probability_auc",
                        "insertion_logit_auc", "insertion_probability_auc")}
    return {"records": records, "indices": [r["index"] for r in records],
            "fractions": fractions, "curves": arrays, "aucs": aucs,
            "row": records[0]["row"], "manifest": manifest}


def draw_plot(rows, out: Path, *, score: str, selected_only: bool):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), sharex=True)
    for row_index, (label, display, data) in enumerate(rows):
        if selected_only and label not in SELECTED:
            continue
        method = data["row"]["method"]
        color = COLORS[method]
        linestyle = "-" if "(loc)" in display else (":" if "anchor" in display else "--")
        for axis, mode in zip(axes, ("deletion", "insertion")):
            values = data["curves"][f"{mode}_{score}"]
            mean = values.mean(axis=0)
            lo, hi = bootstrap_curve_ci(values, seed=42 + row_index * 2 + (mode == "insertion"))
            axis.plot(data["fractions"], mean, label=display, color=color,
                      linestyle=linestyle, linewidth=1.5)
            axis.fill_between(data["fractions"], lo, hi, color=color, alpha=.10, linewidth=0)
    axes[0].set_title("Deletion (lower AUC is better)")
    axes[1].set_title("Insertion (higher AUC is better)")
    for axis in axes:
        axis.set_xlabel("Perturbed spatial-pixel fraction")
        axis.grid(alpha=.2, linewidth=.5)
    axes[0].set_ylabel("Target " + ("probability" if score == "probabilities" else "logit"))
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    suffix = "selected" if selected_only else "all"
    fig.savefig(out / f"full_auc_{score}_{suffix}.pdf", bbox_inches="tight")
    fig.savefig(out / f"full_auc_{score}_{suffix}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_tex_table(rows, out: Path):
    metrics = (
        ("deletion_logit_auc", False),
        ("insertion_logit_auc", True),
        ("deletion_probability_auc", False),
        ("insertion_probability_auc", True),
    )
    best = {}
    for metric, higher_is_better in metrics:
        values = [data["aucs"][metric].mean() for _, _, data in rows]
        best[metric] = (max(values) if higher_is_better else min(values))
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Del. logit $\downarrow$ & Ins. logit $\uparrow$ & "
        r"Del. prob. $\downarrow$ & Ins. prob. $\uparrow$ \\",
        r"\midrule",
    ]
    for label, display, data in rows:
        cells = []
        for metric, _ in metrics:
            values = data["aucs"][metric]
            mean = float(values.mean())
            lo, hi = bootstrap_mean_ci(values, seed=42 + ROWS.index((label, display)))
            cell = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
            if np.isclose(mean, best[metric], rtol=0, atol=1e-12):
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(display + " & " + " & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    (out / "full_auc_tabular.tex").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    loaded = []
    for label, display in ROWS:
        data = load_row(args.root, label)
        if data is not None:
            loaded.append((label, display, data))
    reference = next((data for label, _, data in loaded if label == REFERENCE), None)
    model_hashes = sorted({data["manifest"]["model_state_sha256"]
                           for _, _, data in loaded})
    if len(model_hashes) > 1:
        raise RuntimeError(f"Rows use different model states: {model_hashes}")
    summary = {"protocol": "full-spatial-deletion-insertion-v1", "expected_n": 565,
               "complete_rows": len(loaded), "expected_rows": len(ROWS),
               "model_state_sha256": model_hashes[0] if model_hashes else None,
               "rows": {}}
    for row_index, (label, display, data) in enumerate(loaded):
        entry = {"display": display, "row": data["row"], "n": len(data["records"]), "metrics": {}}
        for metric, values in data["aucs"].items():
            item = {"mean": float(values.mean()),
                    "ci95": bootstrap_mean_ci(values, seed=42 + row_index)}
            if reference is not None:
                if data["indices"] != reference["indices"]:
                    raise RuntimeError(f"Index mismatch for {label}")
                difference = values - reference["aucs"][metric]
                item["difference_vs_rg_loc"] = {
                    "mean": float(difference.mean()),
                    "ci95": bootstrap_mean_ci(difference, seed=142 + row_index),
                }
            entry["metrics"][metric] = item
        summary["rows"][label] = entry
    atomic_json(args.out / "summary.json", summary)
    if loaded:
        write_tex_table(loaded, args.out)
        for score in ("probabilities", "logits"):
            draw_plot(loaded, args.out, score=score, selected_only=True)
            if len(loaded) == len(ROWS):
                draw_plot(loaded, args.out, score=score, selected_only=False)
    print(json.dumps({"complete_rows": len(loaded), "expected_rows": len(ROWS)}, indent=2))


if __name__ == "__main__":
    main()
