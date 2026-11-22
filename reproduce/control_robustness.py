"""Paired risk/gate ablations: Quantus MaxS/AvgS and map-stability controls."""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import quantus

from reproduce.control_common import (
    add_args, atomic_json, dataset, explain, image_record, isolated_rng,
    load_model, map_stability, sample_indices, scalar_stats, stable_seed)


class RobustnessMetrics(quantus.MaxSensitivity):
    """Same samples/formula as Quantus; also return mean and two stability scores."""
    def evaluate_batch(self, model, x_batch, y_batch, a_batch, **kwargs):
        if len(x_batch) != 1:
            raise ValueError("One image at a time for reproducible image-keyed noise")
        scores = []
        for _ in range(self.nr_samples):
            xp = self.perturb_func(arr=x_batch.reshape(1,-1),
                                  indices=np.arange(x_batch[0].size)[None]).reshape(x_batch.shape)
            ap = self.explain_batch(model, xp, y_batch)
            scores.append(map_stability(a_batch, ap))
        self.extra = {
            "AvgS": float(np.mean([s["sensitivity"] for s in scores])),
            "CosineMean": scalar_stats([s["cosine"] for s in scores])["mean"],
            "Top10OverlapMean": scalar_stats([s["top10_overlap"] for s in scores])["mean"],
            "zero_perturbed_maps":sum(s["cosine"] is None for s in scores),
        }
        return np.array([max(s["sensitivity"] for s in scores)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_args(ap)
    ap.add_argument("--rows", required=True, type=str)
    ap.add_argument("--row", required=True)
    ap.add_argument("--samples", type=int, default=50)
    args = ap.parse_args()
    rows = json.loads(open(args.rows).read())
    row = next(r for r in rows if r["row_label"] == args.row)
    ds = dataset(args)
    device = torch.device(args.device or "cuda")
    model = load_model(args.model, device)
    dest = args.out / args.model / args.split / row["row_label"]
    ids = sample_indices(ds, args.split, args.subset, args.shard, args.shards)
    for index in ids:
        path = dest / f"{index:04d}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old["samples"] == args.samples and old["seed"] == args.seed:
                continue
            raise ValueError(f"Incompatible cached result: {path}")
        x, mask, target, name = image_record(ds, index, device)
        t = time.perf_counter()
        expl_seed = stable_seed("explanation", args.seed, name)
        try:
            a = explain(model, row, x, target, expl_seed)
        except ValueError as exc:
            if "Nonfinite or zero explanation" not in str(exc): raise
            atomic_json(path,{"image":name,"index":index,"target":int(target.item()),
                "row":row,"samples":args.samples,"seed":args.seed,
                "metrics":{k:None for k in ("MaxS","AvgS","CosineMean","Top10OverlapMean","AL")},
                "undefined_reason":str(exc),"seconds":time.perf_counter()-t})
            continue
        # Same transform draws for clean and noisy RG+Eq maps: input stability,
        # not Monte Carlo variation of the averaging estimator.
        def callback(model, inputs, targets, **kwargs):
            return explain(model, row, torch.as_tensor(inputs, device=device),
                           torch.as_tensor(targets, device=device), expl_seed,allow_zero=True)
        metric = RobustnessMetrics(nr_samples=args.samples, abs=True, normalise=True,
                                   lower_bound=.2, disable_warnings=True)
        with isolated_rng(stable_seed("noise",args.seed,name)):
            score = metric(model=model, x_batch=x.detach().cpu().numpy(),
                           y_batch=target.cpu().numpy(), a_batch=a,
                           explain_func=callback, device=device, channel_first=True,
                           softmax=True, batch_size=1)
        mass = np.abs(a[0])
        metrics = {"MaxS":float(score[0]), **metric.extra,
                   "AL":float(mass[mask].sum()/mass.sum())}
        atomic_json(path, {"image": name, "index":index, "target":int(target.item()),
                          "row":row, "samples":args.samples,"seed":args.seed,
                          "metrics":metrics,"seconds":time.perf_counter()-t,
                          "protocol":"paired-uniform-0.2; fixed estimator randomness; v1"})
        print(json.dumps({"row":row["row_label"],"image":index,**metrics}),flush=True)
    # Each shard writes its own summary; aggregation never treats it as a full row.
    records=[json.loads((dest/f"{i:04d}.json").read_text()) for i in ids]
    atomic_json(dest/f"summary_shard{args.shard:03d}.json",
                {"row":row,"indices":ids,"metrics":{
                    k:scalar_stats([r["metrics"][k] for r in records])
                    for k in ("MaxS","AvgS","CosineMean","Top10OverlapMean","AL")}})


if __name__ == "__main__":
    main()
