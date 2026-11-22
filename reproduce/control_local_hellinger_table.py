"""Audit and tabulate completed local ViT Hellinger perturbation controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONDITIONS = (
    ("single_heads.head_s0.001", "head_0.001", "Classifier head", 0.001),
    ("single_heads.head_s0.01", "head_0.01", "Classifier head", 0.01),
    ("single_heads.head_s0.1", "head_0.1", "Classifier head", 0.1),
    ("single_heads.head_s1", "head_1", "Classifier head", 1.0),
    ("single_encoder.layers.encoder_layer_11.self_attention.out_proj_s0.001",
     "last_attention_out_proj_0.001", "Last-block attention output", 0.001),
)


def load(path: Path):
    return json.loads(path.read_text())


def audit_condition(root: Path, name: str, verified_name: str, module: str, scale: float):
    directory = root / name
    records = [load(p) for p in sorted(directory.glob("[0-9][0-9][0-9][0-9].json"))]
    manifests = [load(p) for p in sorted(directory.glob("manifest_shard*.json"))]
    if len(records) != 200 or not manifests:
        raise RuntimeError(f"{name}: expected 200 records and manifests")
    indices = sorted(r["index"] for r in records)
    manifest_indices = sorted(i for m in manifests for i in m["indices"])
    if indices != manifest_indices or len(set(indices)) != 200:
        raise RuntimeError(f"{name}: record/manifest indices differ")
    hashes = records[0]["hashes"]
    expected_params = None
    for manifest in manifests:
        if manifest["condition"] != {"name": name, "layers": records[0]["condition"]["layers"], "scale": scale}:
            raise RuntimeError(f"{name}: manifest condition mismatch")
        if manifest["hashes"] != hashes or manifest["seed"] != 42:
            raise RuntimeError(f"{name}: hash or seed mismatch")
        if manifest["targets"] != "dataset labels, fixed for both diagnostics":
            raise RuntimeError(f"{name}: target convention mismatch")
        if manifest["qkv"] != "pretrained in both cascades":
            raise RuntimeError(f"{name}: QKV convention mismatch")
        expected_params = expected_params or manifest["changed_parameters"]
        if manifest["changed_parameters"] != expected_params:
            raise RuntimeError(f"{name}: changed parameters vary across shards")
    for record in records:
        if record["condition"]["name"] != name or record["hashes"] != hashes:
            raise RuntimeError(f"{name}: record condition/hash mismatch")
    values = np.asarray([r["trajectory"]["H_rg_plain"] for r in records], dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError(f"{name}: non-finite Hellinger value")
    return {
        "condition": name,
        "verified_condition": verified_name,
        "module": module,
        "scale": scale,
        "n": 200,
        "mean_H_rg": float(values.mean()),
        "sd_H_rg": float(values.std(ddof=1)),
        "hashes": hashes,
        "changed_parameters": expected_params,
        "seed": 42,
        "indices": indices,
    }


def summarize(root: Path, verified_path: Path):
    rows = [audit_condition(root, *condition) for condition in CONDITIONS]
    if any(row["indices"] != rows[0]["indices"] for row in rows[1:]):
        raise RuntimeError("Local controls do not use the same 200 images")
    verified = load(verified_path)
    if verified["n_per_condition"] != 200 or verified["parameter_seed"] != 42:
        raise RuntimeError("Verified summary protocol differs")
    for row in rows:
        item = verified["conditions"][row["verified_condition"]]
        if item["checkpoint_B_sha256"] != row["hashes"]["B"] or item["scale"] != row["scale"]:
            raise RuntimeError(f"{row['condition']}: verified checkpoint/scale mismatch")
        if abs(item["H_rg_plain"]["mean"] - row["mean_H_rg"]) > 1e-12:
            raise RuntimeError(f"{row['condition']}: verified mean mismatch")
        row["ci95_H_rg"] = item["H_rg_plain"]["ci95"]
    return {
        "protocol": {
            "perturbation": "theta' = theta + s * max(RMS(theta), 1e-6) * xi; xi ~ N(0,I)",
            "noise_seed": 42,
            "tensor_seed": "deterministically derived from noise seed and full parameter name",
            "targets": "dataset labels fixed across checkpoint pair",
            "qkv": "pretrained; only the named module is perturbed",
            "uncertainty": verified["uncertainty"],
            "verified_summary": str(verified_path),
        },
        "rows": rows,
    }


def render(summary):
    lines = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Perturbed module & Noise scale $s$ & $\mathrm{H}_{\mathrm{RG}}$ [95\% CI] \\",
        r"\midrule",
    ]
    for row in summary["rows"]:
        lo, hi = row["ci95_H_rg"]
        lines.append(
            f"{row['module']} & ${row['scale']:g}$ & "
            f"${row['mean_H_rg']:.6f}$ $[{lo:.6f},{hi:.6f}]$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--verified-summary", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--tex", type=Path, required=True)
    args = ap.parse_args()
    summary = summarize(args.root, args.verified_summary)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(summary, indent=2) + "\n")
    args.tex.write_text(render(summary))
    print(json.dumps([{k: row[k] for k in ("condition", "mean_H_rg", "sd_H_rg")}
                      for row in summary["rows"]], indent=2))


if __name__ == "__main__":
    main()
