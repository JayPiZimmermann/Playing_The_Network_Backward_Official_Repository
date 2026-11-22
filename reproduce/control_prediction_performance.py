"""Classification performance for the checkpoint-aligned ViT diagnostic.

Evaluates the exact A/B checkpoint tensors and 200 dataset-label targets used
by the paired trajectory/map experiment.  This separates a model-performance
claim from the already cached target-probability and attribution diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from reproduce.control_common import atomic_json, dataset, digest_state, image_record, load_model


CONDITIONS = ("identity", "head", "aligned", "endpoint")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_ci(values, seed=42):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(1)
    return {"mean": float(values.mean()),
            "ci95": np.quantile(boot, [.025, .975]).tolist(), "n": len(values)}


def manifests(condition_dir: Path):
    found = sorted(condition_dir.glob("manifest_shard*.json"))
    if not found:
        raise FileNotFoundError(f"No manifests in {condition_dir}")
    docs = [json.loads(p.read_text()) for p in found]
    hashes = docs[0]["hashes"]
    if any(d["hashes"] != hashes for d in docs):
        raise RuntimeError(f"Shard hashes disagree in {condition_dir}")
    indices = sorted(i for d in docs for i in d["indices"])
    if len(indices) != 200 or len(set(indices)) != 200:
        raise RuntimeError(f"Expected 200 unique indices, got {len(indices)}")
    return hashes, indices, found


def evaluate(model, ds, indices, device, batch_size):
    records = []
    for start in range(0, len(indices), batch_size):
        rows = [image_record(ds, i, device) for i in indices[start:start + batch_size]]
        x = torch.cat([r[0] for r in rows])
        y = torch.cat([r[2] for r in rows])
        with torch.inference_mode():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        predicted = logits.argmax(1)
        target_prob = probs.gather(1, y[:, None])[:, 0]
        top5 = logits.topk(5, dim=1).indices.eq(y[:, None]).any(1)
        for j, (index, row) in enumerate(zip(indices[start:start + batch_size], rows)):
            records.append({"index": int(index), "image": row[3],
                            "target": int(y[j]), "prediction": int(predicted[j]),
                            "top1_correct": bool(predicted[j] == y[j]),
                            "top5_correct": bool(top5[j]),
                            "target_probability": float(target_prob[j]),
                            "target_logit": float(logits[j, y[j]])})
    return records


def summary(records):
    return {"n": len(records),
            "top1_accuracy": float(np.mean([r["top1_correct"] for r in records])),
            "top5_accuracy": float(np.mean([r["top5_correct"] for r in records])),
            "mean_target_probability": float(np.mean([r["target_probability"] for r in records])),
            "mean_target_logit": float(np.mean([r["target_logit"] for r in records]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    from reproduce.control_common import add_args
    add_args(ap)
    ap.set_defaults(model="vit_b16", split="test")
    ap.add_argument("--diagnostics", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    if args.model != "vit_b16" or args.split != "test":
        raise ValueError("This audit is fixed to the paired ViT test experiment")
    device = torch.device(args.device or "cpu")
    ds = dataset(args)
    model = load_model(args.model, device)
    output = {"protocol": "exact paired diagnostic checkpoints and image indices",
              "device": str(device), "conditions": {}}
    reference = None
    reference_hash = None
    for condition in CONDITIONS:
        condition_dir = args.diagnostics / condition
        hashes, indices, manifest_paths = manifests(condition_dir)
        checkpoint_a = condition_dir / "checkpoint_A.pt"
        checkpoint_b = condition_dir / "checkpoint_B.pt"
        if not checkpoint_a.exists() or not checkpoint_b.exists():
            raise FileNotFoundError(f"Missing checkpoints in {condition_dir}")
        state_a = torch.load(checkpoint_a, map_location=device, weights_only=True)
        state_b = torch.load(checkpoint_b, map_location=device, weights_only=True)
        model.load_state_dict(state_a, strict=True)
        if digest_state(model) != hashes["A"]:
            raise RuntimeError(f"A hash mismatch: {condition}")
        if reference is None:
            reference = evaluate(model, ds, indices, device, args.batch_size)
            reference_hash = hashes["A"]
        elif hashes["A"] != reference_hash:
            raise RuntimeError("Pretrained A checkpoint differs across conditions")
        model.load_state_dict(state_b, strict=True)
        if digest_state(model) != hashes["B"]:
            raise RuntimeError(f"B hash mismatch: {condition}")
        compared = evaluate(model, ds, indices, device, args.batch_size)
        old = {json.loads(p.read_text())["index"]: json.loads(p.read_text())
               for p in condition_dir.glob("[0-9][0-9][0-9][0-9].json")}
        max_probability_error = max(
            abs(r["target_probability"] - old[r["index"]]["probability_B"])
            for r in compared)
        ref_by_index = {r["index"]: r for r in reference}
        delta_top1 = [float(r["top1_correct"]) - float(ref_by_index[r["index"]]["top1_correct"])
                      for r in compared]
        delta_prob = [r["target_probability"] - ref_by_index[r["index"]]["target_probability"]
                      for r in compared]
        output["conditions"][condition] = {
            "hashes": hashes, "checkpoint_sha256": {"A": sha256(checkpoint_a),
                                                       "B": sha256(checkpoint_b)},
            "manifest_sha256": {p.name: sha256(p) for p in manifest_paths},
            "indices": indices, "reference": summary(reference),
            "randomized": summary(compared),
            "paired_top1_change": paired_ci(delta_top1),
            "paired_target_probability_change": paired_ci(delta_prob),
            "max_abs_difference_from_cached_probability_B": max_probability_error,
            "records": compared}
        print(condition, output["conditions"][condition]["randomized"], flush=True)
    atomic_json(args.out / "summary.json", output)


if __name__ == "__main__":
    main()
