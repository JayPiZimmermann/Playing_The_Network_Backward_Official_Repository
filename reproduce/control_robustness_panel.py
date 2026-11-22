"""Protocol-sensitivity panel for attribution robustness measurements.

The panel varies the input-noise distribution, scale, and Monte Carlo count.
Each noisy attribution is computed once and reused for every metric and every
nested sample count.  Noise is keyed by image/protocol/draw, hence paired
across configurations and independent of explanation RNG state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from reproduce.control_common import (
    add_args, atomic_json, dataset, digest_state, explain, image_record,
    load_model, stable_seed,
)


DISTRIBUTIONS = ("uniform", "gaussian")
SCALES = (.05, .1, .2)
NESTED_COUNTS = (10, 25, 50, 100, 200)
BLOCK_SIZE = 50
METRICS = (
    "sensitivity", "cosine", "spearman", "top10_overlap", "mass_tv",
    "prediction_agreement", "target_probability_change",
)


def panel_rows():
    anchor = dict(lrp_alpha=2., lrp_beta=1., lrp_eps=.5,
                  routing_temperature=1., adf_lambda=0.,
                  adf_softmax_lambda=0., av_rule="merged")
    loc = dict(anchor, lrp_eps=.1, routing_temperature=.4,
               adf_input_variance=5.)
    faith = dict(anchor, lrp_eps=.1, routing_temperature=.7,
                 adf_input_variance=5.)
    def rg(name, background, risk, family):
        p = dict(background, adf_softmax_lambda=float(risk))
        if risk == 0: p.pop("adf_input_variance", None)
        return {"id": name, "method": "rg", "params": p,
                "row_label": name, "family": family, "risk": float(risk)}
    return [
        rg("anchor_risk0", anchor, 0, "anchor"),
        rg("anchor_riskm50", dict(anchor, adf_input_variance=2.), -50, "anchor"),
        rg("loc_risk0", loc, 0, "localization"),
        rg("loc_riskm10", loc, -10, "localization"),
        rg("loc_riskm50", loc, -50, "localization"),
        rg("faith_risk0", faith, 0, "faithfulness"),
        rg("faith_riskm5", faith, -5, "faithfulness"),
        {"id":"attnlrp_loc", "method":"attnlrp", "family":"baseline",
         "risk":None, "row_label":"attnlrp_loc", "params":{
             "gamma_composite":True, "lrp_epsilon_attn":.25,
             "lrp_gamma_conv":.25, "lrp_gamma_linear":.05}},
    ]


def draw_noise(shape, distribution, scale, seed):
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        return rng.uniform(-scale, scale, size=shape).astype(np.float32)
    if distribution == "gaussian":
        return rng.normal(0., scale / np.sqrt(3.), size=shape).astype(np.float32)
    raise ValueError(distribution)


def noise_seed(global_seed, image_name, distribution, scale, draw):
    return stable_seed("robustness-panel-noise-v1", global_seed, image_name,
                       distribution, float(scale), int(draw))


def _normalizations(a):
    x = np.abs(np.asarray(a, dtype=np.float64)).ravel()
    mx, sm = x.max(), x.sum()
    if not np.isfinite(x).all() or mx <= 0 or sm <= 0:
        return None
    return x / mx, x / sm


def stability_vector(clean, noisy):
    ca = _normalizations(clean); cb = _normalizations(noisy)
    if ca is None or cb is None:
        return {k:np.nan for k in METRICS[:5]}
    a, ap = ca; b, bp = cb
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    k = max(1, int(np.ceil(.1 * a.size)))
    ai = np.argsort(-a, kind="stable")[:k]
    bi = np.argsort(-b, kind="stable")[:k]
    ra, rb = rankdata(a, method="average"), rankdata(b, method="average")
    rac, rbc = ra-ra.mean(), rb-rb.mean()
    denom = np.linalg.norm(rac)*np.linalg.norm(rbc)
    return {
        "sensitivity":float(np.linalg.norm(a-b)/an),
        "cosine":float(np.dot(a,b)/(an*bn)),
        "spearman":float(np.dot(rac,rbc)/denom) if denom else np.nan,
        "top10_overlap":float(np.intersect1d(ai,bi).size/k),
        "mass_tv":float(.5*np.abs(ap-bp).sum()),
    }


def summarize_prefix(vectors, count):
    out = {}
    for key in METRICS:
        a = np.asarray(vectors[key][:count], dtype=float)
        good = a[np.isfinite(a)]
        out[key] = {
            "mean":float(good.mean()) if good.size else None,
            "max":float(good.max()) if good.size else None,
            "q95":float(np.quantile(good,.95)) if good.size else None,
            "n":int(good.size),
        }
    return out


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), root/"reproduce/control_common.py"]
    paths.extend(sorted((root/"bwgame").rglob("*.py")))
    return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def bootstrap_mean(values, seed=42):
    a=np.asarray(values,dtype=float); a=a[np.isfinite(a)]
    if not a.size:return {"mean":None,"ci95":None,"n":0}
    rng=np.random.default_rng(seed)
    means=a[rng.integers(0,a.size,size=(5000,a.size))].mean(1)
    return {"mean":float(a.mean()),"ci95":np.quantile(means,[.025,.975]).tolist(),"n":int(a.size)}


def aggregate(out):
    rows=panel_rows(); result={"rows":[],"paired_risk_effects":[],"complete":True}
    cached={}
    for row in rows:
      for distribution in DISTRIBUTIONS:
       for scale in SCALES:
        base=out/"records"/row["id"]/f"{distribution}_r{scale:g}"
        records=[]
        for i in range(50):
            p=base/f"{i:04d}.json"
            if not p.exists(): result["complete"]=False; continue
            records.append(json.loads(p.read_text()))
        cached[(row["id"],distribution,scale)]=records
        if not records: continue
        last_entry=None
        for count in NESTED_COUNTS:
            if count>records[0]["draws"]:continue
            per_metric={
              "MaxS":[r["nested"][str(count)]["sensitivity"]["max"] for r in records],
              "AvgS":[r["nested"][str(count)]["sensitivity"]["mean"] for r in records],
              "P95S":[r["nested"][str(count)]["sensitivity"]["q95"] for r in records],
            }
            for m in METRICS[1:]:
                per_metric[m]=[r["nested"][str(count)][m]["mean"] for r in records]
            last_entry={"row":row,"distribution":distribution,
                "scale":scale,"count":count,"n_images":len(records),
                "metrics":{k:bootstrap_mean(v) for k,v in per_metric.items()}}
            result["rows"].append(last_entry)
        if records[0]["draws"]>=200 and last_entry is not None:
            last_entry["independent_blocks50"]=[{"block":bi,"n_images":len(records),
              "MaxS_mean":float(np.mean([r["blocks50"][bi]["metrics"]["sensitivity"]["max"] for r in records])),
              "AvgS_mean":float(np.mean([r["blocks50"][bi]["metrics"]["sensitivity"]["mean"] for r in records]))}
              for bi in range(4)]
    off={"anchor":"anchor_risk0","localization":"loc_risk0","faithfulness":"faith_risk0"}
    for row in rows:
        if row["risk"] in (None,0):continue
        for distribution in DISTRIBUTIONS:
         for scale in SCALES:
          on=cached.get((row["id"],distribution,scale),[])
          base=cached.get((off[row["family"]],distribution,scale),[])
          bm={r["index"]:r for r in base}
          pairs=[(r,bm[r["index"]]) for r in on if r["index"] in bm]
          for count in NESTED_COUNTS:
            if not pairs or count>pairs[0][0]["draws"]:continue
            for label,metric,stat in (("MaxS","sensitivity","max"),
                                      ("AvgS","sensitivity","mean"),
                                      ("P95S","sensitivity","q95"),
                                      ("cosine","cosine","mean"),
                                      ("spearman","spearman","mean"),
                                      ("top10_overlap","top10_overlap","mean"),
                                      ("mass_tv","mass_tv","mean")):
                delta=[a["nested"][str(count)][metric][stat]-b["nested"][str(count)][metric][stat]
                       for a,b in pairs]
                result["paired_risk_effects"].append({"risk_row":row["id"],
                  "off_row":off[row["family"]],"distribution":distribution,"scale":scale,
                  "count":count,"metric":label,"effect_on_minus_off":bootstrap_mean(delta)})
    atomic_json(out/"aggregate.json",result)
    return result


def image_fingerprint(ds):
    h = hashlib.sha256(); records=[]
    for i in range(len(ds)):
        x, mask, label, cls, name = ds[i]
        for value in (str(i),str(name),str(label),str(cls)):
            h.update(value.encode())
        h.update(np.ascontiguousarray(x.numpy()).tobytes())
        h.update(np.ascontiguousarray(mask).tobytes())
        records.append({"index":i,"name":str(name),"label":int(label)})
    return h.hexdigest(), records


def protocol_tasks():
    return [(r,d,s) for r in range(len(panel_rows()))
            for d in DISTRIBUTIONS for s in SCALES]


def run_task(args, row, distribution, scale, ds, model, device):
    dest = args.out/"records"/row["id"]/f"{distribution}_r{scale:g}"
    ids = list(range(len(ds)))
    if args.max_images: ids=ids[:args.max_images]
    draws = args.draws
    for index in ids:
        path = dest/f"{index:04d}.npz"
        meta_path = dest/f"{index:04d}.json"
        if path.exists() and meta_path.exists():
            old=json.loads(meta_path.read_text())
            if old["draws"]==draws and old["seed"]==args.seed: continue
            raise ValueError(f"Incompatible cache: {path}")
        x, mask, target, name = image_record(ds,index,device)
        started=time.perf_counter()
        expl_seed=stable_seed("robustness-panel-explanation",args.seed,name,row["id"])
        clean=explain(model,row,x,target,expl_seed)
        with torch.no_grad():
            logits=model(x); clean_pred=int(logits.argmax(1).item())
            clean_prob=float(torch.softmax(logits,1)[0,target.item()].item())
        vectors={k:[] for k in METRICS}; noise_rms=[]; noise_hashes=[]
        for draw in range(draws):
            seed=noise_seed(args.seed,name,distribution,scale,draw)
            noise=draw_noise(tuple(x.shape),distribution,scale,seed)
            noise_rms.append(float(np.sqrt(np.mean(noise.astype(np.float64)**2))))
            noise_hashes.append(hashlib.sha256(noise.tobytes()).hexdigest())
            xn=x+torch.from_numpy(noise).to(device)
            noisy=explain(model,row,xn,target,expl_seed,allow_zero=True)
            sv=stability_vector(clean,noisy)
            for k in METRICS[:5]: vectors[k].append(sv[k])
            with torch.no_grad():
                out=model(xn); pred=int(out.argmax(1).item())
                prob=float(torch.softmax(out,1)[0,target.item()].item())
            vectors["prediction_agreement"].append(float(pred==clean_pred))
            vectors["target_probability_change"].append(abs(prob-clean_prob))
        path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, clean_map=clean.astype(np.float32),
                                noise_rms=np.asarray(noise_rms,dtype=np.float32),
                                noise_sha256=np.asarray(noise_hashes),
                                **{k:np.asarray(v,dtype=np.float32) for k,v in vectors.items()})
        tmp.replace(path)
        counts=[n for n in NESTED_COUNTS if n<=draws]
        blocks=[]
        for start in range(0,draws,BLOCK_SIZE):
            if start+BLOCK_SIZE<=draws:
                blocks.append({"start":start,"end":start+BLOCK_SIZE,
                    "metrics":summarize_prefix({k:v[start:start+BLOCK_SIZE] for k,v in vectors.items()},BLOCK_SIZE)})
        atomic_json(meta_path,{"row":row,"index":index,"image":name,
            "target":int(target.item()),"distribution":distribution,"scale":scale,
            "draws":draws,"seed":args.seed,"clean_prediction":clean_pred,
            "clean_target_probability":clean_prob,
            "nested":{str(n):summarize_prefix(vectors,n) for n in counts},
            "blocks50":blocks,"seconds":time.perf_counter()-started})
        print(json.dumps({"row":row["id"],"protocol":f"{distribution}:{scale}",
                          "image":index,"seconds":time.perf_counter()-started}),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__); add_args(ap)
    ap.set_defaults(model="vit_b16",split="val")
    ap.add_argument("--task-id",type=int)
    ap.add_argument("--draws",type=int,default=200)
    ap.add_argument("--no-aggregate",action="store_true",
                    help="defer the aggregate write (used by concurrent workers)")
    ap.add_argument("--aggregate-only",action="store_true",
                    help="aggregate completed records without loading model/data")
    args=ap.parse_args()
    if args.aggregate_only:
        result=aggregate(args.out)
        print(json.dumps({"aggregate":str(args.out/"aggregate.json"),
                          "complete":result["complete"]}),flush=True)
        return
    if args.model!="vit_b16" or args.split!="val":
        raise ValueError("Phase 1 is fixed to ViT-B/16 validation")
    if args.task_id is None:
        ap.error("--task-id is required unless --aggregate-only is used")
    tasks=protocol_tasks()
    if not 0<=args.task_id<len(tasks):raise ValueError(args.task_id)
    ri,distribution,scale=tasks[args.task_id]; row=panel_rows()[ri]
    device=torch.device(args.device or "cuda")
    ds=dataset(args); model=load_model(args.model,device)
    data_hash,images=image_fingerprint(ds)
    manifest={"protocol":"robustness-panel-v1","model":"vit_b16","split":"val",
      "n_images":len(ds),"rows":panel_rows(),"distributions":list(DISTRIBUTIONS),
      "scales":list(SCALES),"draws":args.draws,"nested_counts":list(NESTED_COUNTS),
      "block_size":BLOCK_SIZE,"metrics":list(METRICS),"seed":args.seed,
      "model_sha256":digest_state(model),"data_sha256":data_hash,
      "images":images,"source_sha256":source_hashes(),
      "noise_units":"normalized model-input coordinates; gaussian std=r/sqrt(3)",
      "selection":"validation protocol-sensitivity analysis; no test selection"}
    mp=args.out/"manifest.json"
    if mp.exists() and json.loads(mp.read_text())!=manifest:
        raise ValueError(f"Manifest mismatch: {mp}")
    atomic_json(mp,manifest)
    run_task(args,row,distribution,scale,ds,model,device)
    if not args.no_aggregate:
        aggregate(args.out)


if __name__=="__main__":main()
