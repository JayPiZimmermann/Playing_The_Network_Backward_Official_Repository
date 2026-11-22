"""Aggregate the three-way VGG RG gate/risk robustness control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from reproduce.control_common import atomic_json


ORDINARY = "rg__av-merged__eps-0.5__lam-0__lamsm-0__tau-1"
RISK = "rg__av-merged__eps-0.5__lam-m2__lamsm-0__tau-1__var-2"
GATE = "rg__tau-1__eps-0.5__lam-0__var-2__probit-forced"
METRICS = ("MaxS", "AvgS", "CosineMean", "Top10OverlapMean", "AL")


def records(directory: Path):
    result = {}
    for path in sorted(directory.glob("[0-9][0-9][0-9][0-9].json")):
        row = json.loads(path.read_text())
        if row.get("undefined_reason"):
            raise ValueError(f"Undefined record in complete control: {path}")
        result[int(row["index"])] = row
    return result


def paired_summary(left, right, ids, *, seed=42, resamples=10_000):
    rng = np.random.RandomState(seed)
    result = {}
    for metric in METRICS:
        a = np.asarray([left[i]["metrics"][metric] for i in ids], dtype=float)
        b = np.asarray([right[i]["metrics"][metric] for i in ids], dtype=float)
        delta = b - a
        draw = rng.randint(0, len(ids), size=(resamples, len(ids)))
        boot = delta[draw].mean(axis=1)
        result[metric] = {
            "left_mean": float(a.mean()),
            "right_mean": float(b.mean()),
            "delta_right_minus_left": float(delta.mean()),
            "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = args.full_root / "vgg16" / "test"
    ordinary, risk = records(base / ORDINARY), records(base / RISK)
    gate = records(args.gate_root / "vgg16" / "test" / GATE)
    old = records(args.original_root / "vgg16" / "test" / ORDINARY)
    ids = sorted(set(ordinary) & set(risk) & set(gate))
    if len(ids) != 565 or set(ordinary) != set(ids) or set(risk) != set(ids) or set(gate) != set(ids):
        raise ValueError(f"Expected three complete identical 565-image sets, got {len(ordinary)}, {len(gate)}, {len(risk)}")
    original_ids = sorted(old)
    if len(original_ids) != 200 or not set(original_ids) < set(ids):
        raise ValueError("Original deterministic 200-image subset is missing or invalid")
    complement = sorted(set(ids) - set(original_ids))
    if len(complement) != 365:
        raise ValueError(f"Expected complementary n=365, got {len(complement)}")

    # Verify the paired image identity, target, draw count and global seed.
    for i in ids:
        triples = (ordinary[i], gate[i], risk[i])
        for key in ("image", "target", "samples", "seed"):
            if len({json.dumps(row[key], sort_keys=True) for row in triples}) != 1:
                raise ValueError(f"Pair mismatch at index {i}, field {key}")

    comparisons = {
        "gate_effect_forced_probit_minus_ordinary_hard": (ordinary, gate),
        "risk_coefficient_minus2_minus0_with_probit_gate_matched": (gate, risk),
        "combined_risk_mode_minus_ordinary": (ordinary, risk),
    }
    subsets = {"original_200": original_ids, "complement_365": complement, "full_565": ids}
    summary = {
        "protocol": {
            "model": "VGG-16",
            "method": "RG",
            "fixed": {"tau": 1.0, "epsilon": 0.5, "input_variance": 2.0},
            "noise": "50 paired content-seeded Uniform[-0.2,0.2] draws in normalized input coordinates",
            "bootstrap": "10,000 paired image resamples, percentile 95% intervals, seed 42",
        },
        "rows": {"ordinary_hard": ORDINARY, "forced_probit_risk0": GATE, "probit_risk_minus2": RISK},
        "subsets": {name: subset for name, subset in subsets.items()},
        "comparisons": {},
    }
    for comparison, (left, right) in comparisons.items():
        summary["comparisons"][comparison] = {
            name: paired_summary(left, right, subset) for name, subset in subsets.items()
        }
    atomic_json(args.out, summary)


if __name__ == "__main__":
    main()
