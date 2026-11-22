"""Audit and summarize the frozen full-test robustness extension."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


METRICS = ("MaxS", "AvgS", "AL", "CosineMean", "Top10OverlapMean")
COMPARISONS = (
    ("vgg_combined_risk_gate", "vgg16",
     "rg__av-merged__eps-0.5__lam-0__lamsm-0__tau-1",
     "rg__av-merged__eps-0.5__lam-m2__lamsm-0__tau-1__var-2"),
    ("vit_localization_validation_selected_risk", "vit_b16",
     "rg__av-merged__eps-0.1__lam-0__lamsm-0__tau-0.4",
     "rg__av-merged__eps-0.1__lam-0__lamsm-m50__tau-0.4__var-5"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(root: Path, model: str, row: str):
    folder = root / model / "test" / row
    paths = sorted(folder.glob("[0-9][0-9][0-9][0-9].json"))
    out = {int(path.stem): json.loads(path.read_text()) for path in paths}
    if len(out) != len(paths):
        raise RuntimeError(f"Duplicate indices in {folder}")
    return out, paths


def paired_ci(values, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(10_000, len(values)))].mean(1)
    return {"mean": float(values.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
            "n": int(len(values))}


def summarize(off, on, ids):
    summary = {}
    for metric in METRICS:
        off_values = np.asarray([off[i]["metrics"][metric] for i in ids], dtype=float)
        on_values = np.asarray([on[i]["metrics"][metric] for i in ids], dtype=float)
        if not np.isfinite(off_values).all() or not np.isfinite(on_values).all():
            raise RuntimeError(f"Non-finite {metric}")
        summary[metric] = {"off_mean": float(off_values.mean()),
                           "on_mean": float(on_values.mean()),
                           "delta_on_minus_off": paired_ci(on_values - off_values)}
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--snapshot-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", choices=[row[0] for row in COMPARISONS])
    args = ap.parse_args()
    selected_comparisons = tuple(row for row in COMPARISONS
                                 if args.only is None or row[0] == args.only)
    reference_row = selected_comparisons[0]
    snapshot, snapshot_paths = records(args.snapshot_root, reference_row[1], reference_row[2])
    snapshot_ids = sorted(snapshot)
    if len(snapshot_ids) != 200:
        raise RuntimeError(f"Expected 200 snapshot IDs, got {len(snapshot_ids)}")
    output = {"status": "complete", "protocol": {
        "selection": "Frozen validation-selected configurations; no test-data tuning.",
        "noise": "50 image-keyed uniform[-0.2,0.2] perturbations in normalized input coordinates.",
        "seed": 42,
        "bootstrap": "10,000 paired image resamples, seed 42, unadjusted percentile 95% intervals.",
        "subsets": {"snapshot": 200, "complement": 365, "full": 565}},
        "snapshot_id_sha256": hashlib.sha256("\n".join(f"{i:04d}.json" for i in snapshot_ids).encode() + b"\n").hexdigest(),
        "comparisons": {}}
    for name, model, off_row, on_row in selected_comparisons:
        off, off_paths = records(args.root, model, off_row)
        on, on_paths = records(args.root, model, on_row)
        if set(off) != set(on) or len(off) != 565:
            raise RuntimeError(f"Expected the same 565 IDs for {name}: {len(off)} vs {len(on)}")
        for i in sorted(off):
            for row in (off[i], on[i]):
                if row["seed"] != 42 or row["samples"] != 50:
                    raise RuntimeError(f"Protocol mismatch in {name}, index {i}")
                if row.get("protocol") != "paired-uniform-0.2; fixed estimator randomness; v1":
                    raise RuntimeError(f"Noise protocol mismatch in {name}, index {i}")
            if off[i]["image"] != on[i]["image"] or off[i]["target"] != on[i]["target"]:
                raise RuntimeError(f"Pairing mismatch in {name}, index {i}")
        all_ids = sorted(off)
        complement_ids = sorted(set(all_ids) - set(snapshot_ids))
        if len(complement_ids) != 365:
            raise RuntimeError(f"Expected 365 complementary IDs, got {len(complement_ids)}")
        output["comparisons"][name] = {
            "model": model, "off_row": off_row, "on_row": on_row,
            "class_coverage": {label: len({off[i]["target"] for i in ids}) for label, ids in
                               (("snapshot", snapshot_ids), ("complement", complement_ids), ("full", all_ids))},
            "results": {"snapshot": summarize(off, on, snapshot_ids),
                        "complement": summarize(off, on, complement_ids),
                        "full": summarize(off, on, all_ids)},
            "raw_sha256": {"off": {path.name: sha256(path) for path in off_paths},
                           "on": {path.name: sha256(path) for path in on_paths}}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"status": output["status"], "comparisons": list(output["comparisons"]),
                      "out": str(args.out)}))


if __name__ == "__main__":
    main()
