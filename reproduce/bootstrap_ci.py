#!/usr/bin/env python3
"""Bootstrap confidence intervals and significance tests for test results.

Usage::

    python -m reproduce.bootstrap_ci --results results/test/vit_b16 --n-boot 10000 --seed 0

Reads row results and aligned per-example scores, and writes percentile
confidence intervals (``ci.csv``), paired comparisons (``significance.csv``),
and the sampling design (``bootstrap_metadata.json``). ImageNet-S resamples
individual images. VOC resamples original images as clusters, retaining all
image--class records and their pairing across methods. Means remain weighted
by records, including in each bootstrap replicate.

VOC uses null-centered paired cluster-bootstrap tests for every metric,
including binary PG. ImageNet-S uses the paired iid bootstrap and McNemar's
test for PG. For each metric Holm correction covers every unordered eligible
row pair; only comparisons against the observed best row are exported.
Non-rejection does not establish equivalence. delta = mean(row) - mean(best).

By default the sweep-only rows (selected_by only ``sweep:*``) are excluded
from the significance family (``--include-sweeps`` adds them); CIs are
computed for every row.
"""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from reproduce.common import load_per_image, load_row_results, setup_logging

logger = logging.getLogger("reproduce.bootstrap_ci")

METRICS = ("AL", "PG", "TK", "PF5", "PF20", "Sel", "MaxS", "AvgS")
ALPHA = 0.05


def is_sweep_only(obj: Dict) -> bool:
    roles = obj.get("selected_by", [])
    return bool(roles) and all(r.startswith("sweep:") for r in roles)


def load_scores(results_dir: Path, metrics: Sequence[str] = METRICS):
    """Return (rows, image_ids, scores) with scores[label][metric] -> (N,) array.

    All rows are aligned to the image ids of the first row; images missing in
    a row become NaN.  Metrics a row does not have are omitted for that row.
    """
    import numpy as np
    from bwgame.evaluation.metric_spec import SHORT_TO_METRIC

    rows = load_row_results(results_dir)
    image_ids: Optional[np.ndarray] = None
    scores: Dict[str, Dict[str, "np.ndarray"]] = {}
    for label in sorted(rows):
        pi = load_per_image(results_dir, label)
        if pi is None:
            logger.warning("%s: per-image file missing; row skipped", label)
            continue
        ids = pi["image_ids"].astype(str)
        if image_ids is None:
            image_ids = ids
        pos = {i: k for k, i in enumerate(ids)}
        if not np.array_equal(ids, image_ids):
            logger.warning("%s: image ids differ from the other rows; aligning by id", label)
        scores[label] = {}
        for s in metrics:
            name = SHORT_TO_METRIC[s]
            if name not in pi:
                continue
            v = np.asarray(pi[name], dtype=float)
            aligned = np.array([v[pos[i]] if i in pos else np.nan for i in image_ids])
            scores[label][s] = aligned
    return rows, (image_ids if image_ids is not None else np.array([])), scores


def boot_indices(n: int, n_boot: int, seed: int):
    import numpy as np
    rng = np.random.RandomState(seed)
    return rng.randint(0, n, size=(n_boot, n)).astype(np.int32)


def ci_from_indices(values, idx) -> Tuple[float, float, float, int]:
    """(mean, lo, hi, n) with a 95% percentile bootstrap over ``idx``."""
    import numpy as np
    v = np.asarray(values, dtype=float)
    fin = np.isfinite(v)
    n = int(fin.sum())
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(v[fin].mean())
    if n < 2:
        return mean, float("nan"), float("nan"), n
    s = np.where(fin, v, 0.0)[idx].sum(axis=1)
    c = fin.astype(float)[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = s / c
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi), n


def uses_image_clusters(rows: Dict[str, Dict]) -> bool:
    """VOC evaluates multiple class records from the same original image."""
    datasets = {row.get("dataset") for row in rows.values()}
    if "voc" in datasets and datasets != {"voc"}:
        raise ValueError("VOC and other datasets cannot share a bootstrap design")
    return datasets == {"voc"}


class BootstrapPlan:
    """Shared image draws, with record-weighted ratios for clustered records."""

    def __init__(self, image_ids, n_boot: int, seed: int, clustered: bool = False):
        import numpy as np
        if n_boot < 1:
            raise ValueError("n_boot must be positive")
        self.image_ids = np.asarray(image_ids, dtype=str)
        self.n_boot, self.seed, self.clustered = n_boot, seed, clustered
        if clustered:
            if any("__" not in image_id or not image_id.rsplit("__", 1)[1]
                   for image_id in self.image_ids):
                raise ValueError("VOC sample IDs must end in '__<class>'")
            cluster_ids = [image_id.rsplit("__", 1)[0] for image_id in self.image_ids]
            self.units, self.group = np.unique(cluster_ids, return_inverse=True)
        else:
            self.units = self.image_ids
            self.group = np.arange(len(self.image_ids))
        self.n_units = len(self.units)
        self.sampling_unit = "image_cluster" if clustered else "image"
        self._indices = None
        self._means = {}
        self._iid_indices = {}

    @property
    def indices(self):
        if self._indices is None:
            self._indices = boot_indices(self.n_units, self.n_boot, self.seed)
        return self._indices

    @indices.setter
    def indices(self, value):
        self._indices = value
        self._means.clear()

    def bootstrap_means(self, values, mask=None):
        """Mean over sampled records; a sampled cluster retains every class."""
        import numpy as np
        v = np.asarray(values, dtype=float)
        if v.shape != self.image_ids.shape:
            raise ValueError("Scores must align with the plan's sample IDs")
        finite = np.isfinite(v)
        if mask is not None:
            finite &= np.asarray(mask, dtype=bool)
        key = (v.tobytes(), finite.tobytes())
        if key not in self._means:
            sums = np.bincount(self.group, weights=np.where(finite, v, 0.0),
                               minlength=self.n_units)
            counts = np.bincount(self.group, weights=finite.astype(float),
                                 minlength=self.n_units)
            boot = np.empty(len(self.indices))
            for start in range(0, len(boot), 256):
                idx = self.indices[start:start + 256]
                with np.errstate(invalid="ignore", divide="ignore"):
                    boot[start:start + len(idx)] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
            self._means[key] = boot
        return self._means[key]

    def ci(self, values) -> Tuple[float, float, float, int]:
        import numpy as np
        v = np.asarray(values, dtype=float)
        finite = np.isfinite(v)
        n = int(finite.sum())
        mean = float(v[finite].mean()) if n else float("nan")
        if len(np.unique(self.group[finite])) < 2:
            return mean, float("nan"), float("nan"), n
        boot = self.bootstrap_means(v)
        boot = boot[np.isfinite(boot)]
        if len(boot) == 0:
            return mean, float("nan"), float("nan"), n
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return mean, float(lo), float(hi), n

    def compare(self, a, b, metric: str) -> Tuple[float, float, int]:
        import numpy as np
        from bwgame.evaluation.stats import mcnemar_test
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        mask = np.isfinite(a) & np.isfinite(b)
        n = int(mask.sum())
        if n < 2 or len(np.unique(self.group[mask])) < 2:
            return float("nan"), float("nan"), n
        delta = float(a[mask].mean() - b[mask].mean())
        if np.array_equal(a[mask], b[mask]):
            return delta, 1.0, n
        if self.clustered:
            boot = self.bootstrap_means(a, mask) - self.bootstrap_means(b, mask)
            boot = boot[np.isfinite(boot)]
            if len(boot) == 0:
                return delta, float("nan"), n
            exceed = np.count_nonzero(np.abs(boot - delta) >= abs(delta))
            p = (1.0 + exceed) / (len(boot) + 1.0)
        elif metric == "PG":
            p = mcnemar_test(a[mask] > 0.5, b[mask] > 0.5)["p_value"]
        else:
            # Same draws and statistic as paired_bootstrap_test, evaluated in batches.
            if n not in self._iid_indices:
                self._iid_indices[n] = boot_indices(n, self.n_boot, self.seed)
            indices = self._iid_indices[n]
            diff = a[mask] - b[mask]
            observed = float(diff.mean())
            exceed = 0
            for start in range(0, self.n_boot, 256):
                boot = diff[indices[start:start + 256]].mean(axis=1)
                exceed += np.count_nonzero(np.abs(boot - observed) >= abs(observed))
            p = exceed / self.n_boot
        return delta, float(p), n

    def metadata(self):
        return {"sampling_unit": self.sampling_unit,
                "estimand": "image_class_record_mean" if self.clustered else "image_mean",
                "n_records": len(self.image_ids), "n_units": self.n_units,
                "n_boot": self.n_boot, "seed": self.seed,
                "ci": "percentile_95",
                "paired_test": "null_centered_cluster_bootstrap_plus_one" if self.clustered
                else "null_centered_iid_bootstrap_or_mcnemar_for_PG",
                "multiplicity": "holm_all_unordered_eligible_row_pairs_per_metric"}


def compare(a, b, metric: str, n_boot: int, seed: int) -> Tuple[float, float, int]:
    """(delta = mean(a) - mean(b), two-sided p, n) on images where both are finite."""
    import numpy as np
    from bwgame.evaluation.stats import mcnemar_test, paired_bootstrap_test
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < 2:
        return float("nan"), float("nan"), n
    delta = float(a[m].mean() - b[m].mean())
    if np.array_equal(a[m], b[m]):
        return delta, 1.0, n
    if metric == "PG":
        p = mcnemar_test(a[m] > 0.5, b[m] > 0.5)["p_value"]
    else:
        p = paired_bootstrap_test(a[m], b[m], n_bootstrap=n_boot, seed=seed)["p_value"]
    return delta, float(p), n


def best_label(means: Dict[str, float], metric: str) -> Optional[str]:
    import math
    from bwgame.evaluation.metric_spec import metric_direction
    cand = [(l, v) for l, v in means.items() if v is not None and math.isfinite(v)]
    if not cand:
        return None
    sign = -1.0 if metric_direction(metric) == "max" else 1.0
    return min(cand, key=lambda t: (sign * t[1], t[0]))[0]


def significance_vs_best(scores: Dict[str, Dict], labels: Sequence[str], metric: str,
                         n_boot: int, seed: int, plan: Optional[BootstrapPlan] = None) -> List[Dict]:
    """Holm over all eligible row pairs, then report edges to the observed best."""
    import numpy as np
    from bwgame.evaluation.stats import holm_correction
    have = sorted({label for label in labels if metric in scores.get(label, {})})
    means = {label: float(np.nanmean(scores[label][metric]))
             if np.isfinite(scores[label][metric]).any() else float("nan") for label in have}
    best = best_label(means, metric)
    if best is None:
        return []
    pairs = []
    for left, right in combinations(have, 2):
        args = (scores[left][metric], scores[right][metric], metric)
        delta, p, n = plan.compare(*args) if plan else compare(*args, n_boot, seed)
        pairs.append({"left": left, "right": right, "delta": delta, "p": p, "n": n})
    adjusted = holm_correction([pair["p"] if np.isfinite(pair["p"]) else 1.0 for pair in pairs])
    out = []
    for pair, ph in zip(pairs, adjusted):
        if best not in (pair["left"], pair["right"]):
            continue
        row = pair["right"] if pair["left"] == best else pair["left"]
        delta = -pair["delta"] if pair["left"] == best else pair["delta"]
        out.append({"metric": metric, "best_row": best, "row": row, "delta": delta,
                    "p": pair["p"], "n": pair["n"], "p_holm": float(ph),
                    "family_size": len(pairs),
                    "significant": bool(np.isfinite(pair["p"]) and ph < ALPHA)})
    return sorted(out, key=lambda row: row["row"])


def run(results_dir: Path, n_boot: int, seed: int, include_sweeps: bool,
        out_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir or results_dir)
    rows, image_ids, scores = load_scores(results_dir)
    if not scores:
        raise SystemExit(f"No row results with per-image files in {results_dir}")
    n = len(image_ids)
    plan = BootstrapPlan(image_ids, n_boot, seed, clustered=uses_image_clusters(rows))
    logger.info("%d rows, %d records, %d %s units, %d bootstrap resamples",
                len(scores), n, plan.n_units, plan.sampling_unit, n_boot)

    ci_path = out_dir / "ci.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(ci_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "metric", "mean", "lo", "hi", "n"])
        for label in sorted(scores):
            for s in METRICS:
                if s not in scores[label]:
                    continue
                mean, lo, hi, k = plan.ci(scores[label][s])
                w.writerow([label, s, f"{mean:.6g}", f"{lo:.6g}", f"{hi:.6g}", k])

    family = [l for l in sorted(scores) if include_sweeps or not is_sweep_only(rows[l])]
    sig_path = out_dir / "significance.csv"
    with open(sig_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "best_row", "row", "delta", "p", "p_holm", "significant"])
        for s in METRICS:
            for r in significance_vs_best(scores, family, s, n_boot, seed, plan=plan):
                w.writerow([r["metric"], r["best_row"], r["row"], f"{r['delta']:.6g}",
                            f"{r['p']:.6g}", f"{r['p_holm']:.6g}", int(r["significant"])])
    metadata = plan.metadata()
    metadata["family_rows"] = family
    metadata["source_scores"] = str(results_dir.resolve())
    (out_dir / "bootstrap_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Wrote %s and %s", ci_path, sig_path)
    return ci_path, sig_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Bootstrap CIs and significance tests.")
    ap.add_argument("--results", type=Path, required=True, help="e.g. results/test/vit_b16")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-sweeps", action="store_true",
                    help="Include sweep-only rows in the significance family.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: the --results directory.")
    args = ap.parse_args(argv)
    setup_logging()
    run(args.results, args.n_boot, args.seed, args.include_sweeps, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
