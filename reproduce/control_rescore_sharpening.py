"""Rescore cached sharpening maps with the paper's fixed TopK k=1000.

This is a migration for results produced before the protocol correction.  It
never regenerates explanations or PF20 curves: it reads each cached NPZ map,
loads only the corresponding dataset mask, and atomically rewrites the JSON
scores after preserving the original JSON in a separate backup tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from reproduce.control_common import add_args, atomic_json, dataset
from reproduce.control_sharpening import sharpen_metrics


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    ds = dataset(args)
    records = []
    for path in sorted(args.input.glob("*/*.json")):
        payload = json.loads(path.read_text())
        npz = path.with_suffix(".npz")
        if not npz.exists():
            raise FileNotFoundError(f"Missing cached map: {npz}")
        rel = path.relative_to(args.input)
        backup = args.backup / rel
        if not backup.exists():
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, backup)
        old_hash = digest(backup)
        cached = np.load(npz)
        if "map" not in cached:
            raise KeyError(f"Cached map key missing: {npz}")
        attribution = np.asarray(cached["map"])
        _, mask, _, _, image = ds[int(payload["index"])]
        mask = np.asarray(mask).astype(bool)
        if mask.ndim == 3 and mask.shape[-1] == 3:
            mask = mask.any(axis=-1)
        if str(image) != payload["image"]:
            raise ValueError(f"Dataset identity mismatch for {path}: {image} != {payload['image']}")
        old_scores = payload["scores"]
        scores = {}
        for power, score in old_scores.items():
            fresh = sharpen_metrics(attribution, mask.squeeze(), float(power))
            fresh["PF20"] = score["PF20"]
            scores[power] = fresh
        payload["scores"] = scores
        payload["scoring_protocol"] = {
            "version": "paper-k1000-v2",
            "TK": "fixed k=1000; descending NumPy argsort, matching Quantus TopKIntersection",
            "AL": "absolute nonnegative map mass inside mask",
            "PG": "any maximizer inside mask",
            "PF20": "preserved from cached v1 record; ranking is unchanged by power sharpening",
            "map_cache": str(npz),
        }
        atomic_json(path, payload)
        records.append({
            "result": str(path),
            "backup": str(backup),
            "map_cache": str(npz),
            "old_sha256": old_hash,
            "new_sha256": digest(path),
            "map_sha256": digest(npz),
            "index": int(payload["index"]),
            "image": payload["image"],
        })
    atomic_json(args.audit, {
        "protocol": "paper-k1000-v2",
        "records": records,
        "count": len(records),
        "maps_regenerated": False,
        "PF20_recomputed": False,
    })
    print(json.dumps({"rescored": len(records), "audit": str(args.audit)}))


if __name__ == "__main__":
    main()
