"""Cache-only, rank-based ImageNet-S localization controls.

This script deliberately does not select a method or parameter on the test set.
It evaluates every complete cached ViT row and gives the seven predeclared rows
separate display labels.  Pixel AUROC, AP, rank percentiles, and the top-fraction
curves depend only on score ordering and are invariant under strictly increasing
rescalings of a heatmap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata

from reproduce.common import imagenet_s_split
from reproduce.control_common import atomic_json


Q_GRID = np.asarray((.001, .005, .01, .02, .05, .1, .2, .3, .5, .75, 1.), dtype=float)
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 42
RG_REFERENCE = "rg__av-merged__eps-0.1__lam-0__lamsm-m10__tau-0.4__var-5"
MAIN_ROWS = {
    "lrp_gamma__eps-0.1__gamma-2": "LRP-gamma (loc)",
    "lrp_gamma__eps-0.1__gamma-0.1": "LRP-gamma (faith)",
    "attnlrp__eattn-0.25__gcomp-on__gconv-0.25__glin-0.05": "AttnLRP (loc)",
    "attnlrp__eattn-0.01__gcomp-on__gconv-0.25__glin-0.05": "AttnLRP (faith)",
    RG_REFERENCE: "RG (loc)",
    "rg__av-merged__eps-0.1__lam-0__lamsm-m5__tau-0.7__var-5": "RG (faith)",
    "rg__av-merged__eps-0.5__lam-0__lamsm-0__tau-1": "RG anchor",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _score_groups(scores, mask):
    """Counts and positive counts for descending score groups."""
    values, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    positives = np.bincount(inv, weights=mask.astype(float), minlength=len(values))
    return counts[::-1].astype(float), positives[::-1].astype(float)


def rank_localization_metrics(raw_map, mask, q_grid=Q_GRID):
    scores = np.abs(np.asarray(raw_map, dtype=np.float64)).squeeze().ravel()
    mask = np.asarray(mask, dtype=bool).squeeze().ravel()
    if scores.shape != mask.shape:
        raise ValueError((scores.shape, mask.shape))
    if not np.isfinite(scores).all() or scores.max(initial=0.0) <= 0:
        raise ValueError("nonfinite or zero heatmap")
    n, npos = len(scores), int(mask.sum())
    nneg = n - npos
    scores /= scores.max()

    # Pairwise AUROC with average ranks gives half credit for tied pairs.
    asc_ranks = rankdata(scores, method="average")
    auc = ((asc_ranks[mask].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
           if npos and nneg else None)

    counts, positives = _score_groups(scores, mask)
    cum_n = np.cumsum(counts)
    cum_p = np.cumsum(positives)
    # Noninterpolated AP, evaluated after each complete tie group.
    ap = np.sum((positives / npos) * (cum_p / cum_n)) if npos else None

    precision, recall, outside = [], [], []
    for q in q_grid:
        k = min(float(n), max(0.0, float(q) * n))
        group = int(np.searchsorted(cum_n, k, side="left"))
        before_n = 0.0 if group == 0 else cum_n[group - 1]
        before_p = 0.0 if group == 0 else cum_p[group - 1]
        frac = 0.0 if counts[group] == 0 else (k - before_n) / counts[group]
        tp = before_p + frac * positives[group]
        fp = k - tp
        precision.append(tp / k if k else np.nan)
        recall.append(tp / npos if npos else None)
        outside.append(fp / nneg if nneg else None)

    # Midrank percentiles: 0 is the top and 1 the bottom of the heatmap.
    descending_midrank = rankdata(-scores, method="average")
    rank_percentile = (descending_midrank - .5) / n
    mask_fraction = npos / n
    precision = np.asarray(precision)
    median_inside = float(np.median(rank_percentile[mask])) if npos else None
    median_outside = float(np.median(rank_percentile[~mask])) if nneg else None
    return {
        "valid_pairwise": bool(npos and nneg),
        "invalid_pairwise_reason": None if npos and nneg else
            f"degenerate mask: positives={npos}, negatives={nneg}",
        "AUROC": float(auc) if auc is not None else None,
        "AP": float(ap) if ap is not None else None,
        "median_rank_inside": median_inside,
        "median_rank_outside": median_outside,
        "median_rank_gap_outside_minus_inside": (
            median_outside - median_inside if median_inside is not None and median_outside is not None
            else None),
        "mask_fraction": float(mask_fraction),
        "topq_precision": precision.tolist(),
        "topq_precision_lift": ((precision / mask_fraction).tolist()
                                if mask_fraction > 0 else [None] * len(precision)),
        "topq_mask_recall": recall,
        "topq_outside_fraction": outside,
    }


_WEIGHT_CACHE = {}


def _weights(n):
    if n not in _WEIGHT_CACHE:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        _WEIGHT_CACHE[n] = rng.multinomial(
            n, np.full(n, 1.0 / n), size=BOOTSTRAP_RESAMPLES).astype(np.float32)
    return _WEIGHT_CACHE[n]


def _summary(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    weights = _weights(len(a))
    boot = weights @ a / len(a)
    return {"mean": float(a.mean()), "ci95": np.quantile(boot, (.025, .975)).tolist(),
            "n": int(len(a)), "coverage": float(len(a) / len(values))}


def _paired_summary(values, reference):
    values, reference = np.asarray(values, dtype=float), np.asarray(reference, dtype=float)
    good = np.isfinite(values) & np.isfinite(reference)
    result = _summary((values - reference)[good])
    result["coverage"] = float(good.mean())
    result["total_n"] = int(len(good))
    return result


def _load_masks(imagenet_root, imagenet_s_root):
    ds = imagenet_s_split("test", imagenet_root, imagenet_s_root)
    masks, identities = [], []
    for index in range(len(ds)):
        _, mask, target, _, name = ds[index]
        mask = np.asarray(mask, dtype=bool).squeeze()
        if mask.ndim == 3 and mask.shape[-1] == 3:
            mask = mask.any(axis=-1)
        masks.append(mask)
        identities.append({"index": index, "image": str(name), "target": int(target),
                           "mask_pixels": int(mask.sum()), "pixels": int(mask.size)})
    return masks, identities


def _row_records(root, row, masks, identities):
    directory = root / row
    paths = sorted(directory.glob("????.json"))
    if len(paths) != len(masks):
        raise ValueError(f"{row}: found {len(paths)} records, expected {len(masks)}")
    records, sources = [], []
    for json_path in paths:
        meta = json.loads(json_path.read_text())
        index = int(meta["index"])
        identity = identities[index]
        if meta["image"] != identity["image"] or int(meta["target"]) != identity["target"]:
            raise ValueError((row, index, meta["image"], identity))
        npz_path = json_path.with_suffix(".npz")
        raw_map = np.load(npz_path)["map"]
        metrics = rank_localization_metrics(raw_map, masks[index])
        records.append({"index": index, "image": identity["image"],
                        "target": identity["target"], **metrics})
        sources.append({"index": index, "json": str(json_path),
                        "json_sha256": sha256(json_path), "npz": str(npz_path),
                        "npz_sha256": sha256(npz_path)})
    records.sort(key=lambda x: x["index"])
    if [r["index"] for r in records] != list(range(len(masks))):
        raise ValueError(f"{row}: indices are not exactly 0..{len(masks)-1}")
    return records, sources


def _aggregate(rows):
    n = len(next(iter(rows.values())))
    reference = rows[RG_REFERENCE]
    scalar_keys = ("AUROC", "AP", "median_rank_inside", "median_rank_outside",
                   "median_rank_gap_outside_minus_inside", "mask_fraction")
    curve_keys = ("topq_precision", "topq_precision_lift", "topq_mask_recall",
                  "topq_outside_fraction")
    out = {}
    for row, records in rows.items():
        result = {"label": MAIN_ROWS.get(row, row), "main_row": row in MAIN_ROWS,
                  "scalars": {}, "curves": {}}
        for key in scalar_keys:
            values = np.asarray([r.get(key, np.nan) for r in records])
            ref = np.asarray([r.get(key, np.nan) for r in reference])
            result["scalars"][key] = _summary(values)
            result["scalars"][key]["paired_difference_vs_rg_loc"] = _paired_summary(
                values, ref)
        for key in curve_keys:
            values = np.asarray([r.get(key, [np.nan] * len(Q_GRID)) for r in records])
            ref = np.asarray([r.get(key, [np.nan] * len(Q_GRID)) for r in reference])
            result["curves"][key] = []
            for j, q in enumerate(Q_GRID):
                stat = _summary(values[:, j])
                stat["q"] = float(q)
                stat["paired_difference_vs_rg_loc"] = _paired_summary(
                    values[:, j], ref[:, j])
                result["curves"][key].append(stat)
        out[row] = result
    return out


def _plot_selected(aggregate, output):
    styles = ["-", "--", "-.", ":", "-", "--", ":"]
    selected = [(row, MAIN_ROWS[row]) for row in MAIN_ROWS]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), sharex=True)
    for style, (row, label) in zip(styles, selected):
        recall = aggregate[row]["curves"]["topq_mask_recall"]
        outside = aggregate[row]["curves"]["topq_outside_fraction"]
        axes[0].plot(Q_GRID, [x["mean"] for x in recall], style, label=label)
        axes[1].plot(Q_GRID, [x["mean"] for x in outside], style, label=label)
    axes[0].plot((0, 1), (0, 1), color="0.6", lw=1, label="random ranking")
    axes[1].plot((0, 1), (0, 1), color="0.6", lw=1)
    axes[0].set(xlabel="Top fraction of heatmap pixels", ylabel="Object-mask recall",
                xlim=(0, 1), ylim=(0, 1))
    axes[1].set(xlabel="Top fraction of heatmap pixels", ylabel="Outside-mask fraction selected",
                xlim=(0, 1), ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _compact_label(row):
    if row in MAIN_ROWS:
        return MAIN_ROWS[row]
    if row.startswith("rg__"):
        risk = "-10" if "lamsm-m10" in row else "-5" if "lamsm-m5" in row else "0"
        tau = row.split("__tau-")[1].split("__")[0] if "__tau-" in row else "?"
        return f"RG risk {risk}, tau={tau}"
    return row


def _plot_all(aggregate, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), sharex=True)
    for row in sorted(aggregate):
        label = _compact_label(row)
        recall = aggregate[row]["curves"]["topq_mask_recall"]
        outside = aggregate[row]["curves"]["topq_outside_fraction"]
        lw = 1.8 if row in MAIN_ROWS else .9
        alpha = 1.0 if row in MAIN_ROWS else .55
        axes[0].plot(Q_GRID, [x["mean"] for x in recall], lw=lw, alpha=alpha, label=label)
        axes[1].plot(Q_GRID, [x["mean"] for x in outside], lw=lw, alpha=alpha, label=label)
    for ax in axes:
        ax.plot((0, 1), (0, 1), color="0.5", lw=1, ls=":")
        ax.grid(alpha=.2)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Top fraction of heatmap pixels")
    axes[0].set_ylabel("Object-mask recall")
    axes[1].set_ylabel("Outside-mask fraction selected")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=6)
    fig.tight_layout(rect=(0, .20, 1, 1))
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=220)
    plt.close(fig)


def _write_markdown(result, path):
    pairwise_n = result["aggregate"][RG_REFERENCE]["scalars"]["AUROC"]["n"]
    total_n = result["protocol"]["n_images"]
    degenerate_n = total_n - pairwise_n
    lines = ["# Rank-based localization controls", "",
             (f"All rows use {total_n} held-out ViT-B/16 ImageNet-S test images. "
              "No test-set selection is performed. AUROC and AP are invariant under "
              f"strictly increasing heatmap rescalings. {degenerate_n} images have "
              f"full-image masks: AP, recall, and inside rank use n={total_n}; AUROC, "
              f"outside rank, and the rank gap are explicitly undefined there and use n={pairwise_n}."), "",
             "| Row | AUROC | AP | Median inside rank | Median outside rank | Rank gap |",
             "|---|---:|---:|---:|---:|---:|"]
    for row, label in MAIN_ROWS.items():
        scalars = result["aggregate"][row]["scalars"]
        lines.append("| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
            label, scalars["AUROC"]["mean"], scalars["AP"]["mean"],
            scalars["median_rank_inside"]["mean"],
            scalars["median_rank_outside"]["mean"],
            scalars["median_rank_gap_outside_minus_inside"]["mean"]))
    lines += ["", "AUROC is the probability that a random inside-mask pixel outranks a random outside-mask pixel, with half credit for ties. Median normalized rank uses 0 for the top and 1 for the bottom; a positive outside-minus-inside gap indicates better object ordering. AP depends on mask prevalence, which is reported in the JSON artifact.", ""]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imagenet-root", required=True)
    ap.add_argument("--imagenet-s-root", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    masks, identities = _load_masks(args.imagenet_root, args.imagenet_s_root)
    complete = sorted(d.name for d in args.cache_root.iterdir()
                      if d.is_dir() and len(list(d.glob("????.npz"))) == len(masks))
    missing_main = sorted(set(MAIN_ROWS) - set(complete))
    if missing_main:
        raise ValueError(f"missing predeclared rows: {missing_main}")
    per_image, source_manifest = {}, {}
    for row in complete:
        print(f"rank localization: {row}", flush=True)
        per_image[row], source_manifest[row] = _row_records(
            args.cache_root, row, masks, identities)
    aggregate = _aggregate(per_image)
    protocol = {
        "name": "rank-localization-v1", "model": "vit_b16", "split": "test",
        "n_images": len(masks), "selection": "none; all complete cached rows evaluated",
        "degenerate_masks": "undefined pairwise/background metrics recorded as null; other defined metrics retained",
        "main_rows_predeclared": MAIN_ROWS, "rg_reference": RG_REFERENCE,
        "map": "absolute channel-collapsed 224x224 map, divided by positive maximum",
        "metrics": {
            "AUROC": "P(score_inside > score_outside) + 0.5 P(tie)",
            "AP": "noninterpolated pixel average precision, complete tie groups",
            "topq": "fractional inclusion at the boundary score tie",
            "rank": "average descending midrank percentile (rank-0.5)/50176",
        }, "q_grid": Q_GRID.tolist(),
        "uncertainty": {"unit": "image", "resamples": BOOTSTRAP_RESAMPLES,
                        "seed": BOOTSTRAP_SEED, "interval": "percentile 95%",
                        "comparisons": "paired versus fixed submitted RG localization row"},
        "interpretation": "mask ranking/agreement only; no faithfulness or causal claim",
    }
    result = {"protocol": protocol, "aggregate": aggregate, "per_image": per_image}
    atomic_json(args.out / "summary.json", result)
    atomic_json(args.out / "manifest.json", {"protocol": protocol,
                "script": str(Path(__file__)), "script_sha256": sha256(Path(__file__)),
                "identities": identities, "sources": source_manifest})
    _write_markdown(result, args.out / "results.md")
    _plot_selected(aggregate, args.out / "selected_rank_cdf")
    _plot_all(aggregate, args.out / "all_rows_rank_cdf")
    print(json.dumps({"rows": len(complete), "images": len(masks),
                      "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
