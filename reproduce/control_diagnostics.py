"""Paired checkpoints, local perturbations, and independent PF20.

Both trajectory and heatmap diagnostics read the exact same prepared models,
with dataset targets fixed across every perturbation. Checkpoints and hashes
are saved before evaluation; hash checks detect attribution-side mutation.
"""
from __future__ import annotations
import argparse
import copy
import gc
import json
import time
import numpy as np
import torch
from scipy.stats import spearmanr

from reproduce.common import load_metric_configs
from reproduce.control_common import (
    add_args, atomic_json, dataset, digest_state, explain, image_record,
    isolated_rng, load_model, map_stability, perturb_parameters, sample_indices, stable_seed)


ANCHOR={"method":"rg","row_label":"rg_anchor", "params":{
    "lrp_alpha":2.,"lrp_beta":1.,"lrp_eps":.5,"routing_temperature":1.,
    "adf_lambda":0.,"adf_softmax_lambda":0.,"av_rule":"merged"}}
GRADIENT={"method":"gradient","row_label":"gradient","params":{"channel_reduction":"max_abs"}}


def conditions(model,model_key,mode):
    from bwgame.evaluation.metrics.sanity_check import _get_randomizable_layers
    # Match the original Hellinger convention in both measurements explicitly.
    names=[n for n in _get_randomizable_layers(model)
           if not n.endswith((".q_proj",".k_proj",".v_proj"))]
    aligned={"vit_b16":"encoder.layers.encoder_layer_11.self_attention.out_proj",
             "vgg16":"features.26","resnet50":"layer4.1.conv3"}[model_key]
    if mode=="paired":
        return [{"name":"identity","layers":[],"scale":0.},
                {"name":"head","layers":names[:1],"scale":None},
                {"name":"aligned","layers":names[:names.index(aligned)+1],"scale":None},
                {"name":"endpoint","layers":names,"scale":None}]
    chosen=[names[0],aligned,names[-1]]
    result=[{"name":"identity","layers":[],"scale":0.}]
    for n in chosen:
        for scale in (.001,.01,.1,1.):
            result.append({"name":f"single_{n}_s{scale:g}","layers":[n],"scale":scale})
    for seed in (1001,1002,1003):
        result.append({"name":f"double_random_{seed}","layers":names,"scale":None,"double_seed":seed})
    return result


def hellinger(a,b,x,y):
    from bwgame.xai.methods.hellinger import HellingerDiagnostic
    result={}
    for game in ("adf","rg"):
        for surv in (False,True):
            key=f"H_{game}_{'surv' if surv else 'plain'}"
            diag=HellingerDiagnostic(a,b,game=game,tau=1.,condition_on_survival=surv)
            try:
                out=diag.compute(x,x,target=y)
                h=float(np.asarray(out["hellinger_distance"].detach().cpu()).reshape(-1)[0])
                if not np.isfinite(h): raise ValueError("Nonfinite Hellinger distance")
                result[key]=h
                if not surv:
                    for m in ("A","B"):
                        result[f"survival_{game}_{m}"]=float(out[f"survival_mass_{m}"].reshape(-1)[0])
                # Path-space diagnostic: measured directly, no map reconstruction claim.
                del out
            except ValueError as exc:
                if not surv or "undefined" not in str(exc): raise
                result[key]=None
                result[key+"_reason"]=str(exc)
            finally:
                del diag
    return result


def faithfulness(model,x,y,a):
    from bwgame.evaluation.quantus_bridge import build_quantus_metric, _compute_truncated_auc
    cfg=load_metric_configs(["PF20"])[0]
    metric=build_quantus_metric(dict(cfg,params={**cfg["params"],"disable_warnings":True}))
    curves=metric(model=model,x_batch=x.detach().cpu().numpy(),y_batch=y.cpu().numpy(),
                  a_batch=a,device=x.device,channel_first=True,softmax=False,batch_size=1)
    c=np.asarray(curves,dtype=float)
    return float(_compute_truncated_auc(c,.2)[0])


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_args(ap)
    ap.add_argument("--mode",choices=("paired","local"),default="paired")
    ap.add_argument("--condition",default=None)
    ap.add_argument("--no-faithfulness",action="store_true")
    args=ap.parse_args()
    device=torch.device(args.device or "cuda")
    a=load_model(args.model,device); b=copy.deepcopy(a)
    original={k:v.detach().cpu().clone() for k,v in a.state_dict().items()}
    ds=dataset(args)
    ids=sample_indices(ds,args.split,args.subset,args.shard,args.shards)
    for cond in conditions(a,args.model,args.mode):
        if args.condition and cond["name"]!=args.condition: continue
        dest=args.out/args.model/args.mode/cond["name"]
        dest.mkdir(parents=True,exist_ok=True)
        a.load_state_dict(original)
        if "double_seed" in cond:
            perturb_parameters(a,original,cond["layers"],cond["double_seed"])
        changed=perturb_parameters(b,original,cond["layers"],args.seed,cond["scale"])
        hashes={"A":digest_state(a),"B":digest_state(b)}
        # Each shard serializes equivalent deterministic tensors under its own name;
        # only shard zero stores the large checkpoint artifact.
        if args.shard==0:
            for label,model in (("A",a),("B",b)):
                path=dest/f"checkpoint_{label}.pt"
                if not path.exists():
                    torch.save({k:v.detach().cpu() for k,v in model.state_dict().items()},path)
        atomic_json(dest/f"manifest_shard{args.shard:03d}.json",{
            "condition":cond,"hashes":hashes,"changed_parameters":changed,
            "targets":"dataset labels, fixed for both diagnostics",
            "qkv":"pretrained in both cascades","seed":args.seed,"indices":ids})
        for i in ids:
            path=dest/f"{i:04d}.json"
            if path.exists():
                old=json.loads(path.read_text())
                if old["hashes"]!=hashes: raise ValueError("Checkpoint mismatch in existing results")
                continue
            t=time.perf_counter()
            x,mask,y,name=image_record(ds,i,device)
            result=hellinger(a,b,x,y)
            with torch.no_grad():
                pa=torch.softmax(a(x),dim=1)[0,y.item()].item()
                pb=torch.softmax(b(x),dim=1)[0,y.item()].item()
            maps={}
            for row in (GRADIENT,ANCHOR):
                from bwgame.evaluation.metrics.sanity_check import compute_similarity
                es=stable_seed(args.seed,name,row["row_label"])
                aa=explain(a,row,x,y,es,allow_zero=True); ab=explain(b,row,x,y,es,allow_zero=True)
                rho=spearmanr(np.abs(aa).ravel(),np.abs(ab).ravel()).statistic
                stability=map_stability(aa,ab) if np.any(aa) and np.any(ab) else {
                    "sensitivity":None,"cosine":None,"top10_overlap":None,
                    "undefined_reason":"zero attribution in at least one checkpoint"}
                maps[row["method"]]={**stability,
                    "spearman_abs":float(rho) if np.isfinite(rho) else None,
                    "hog_pearson":float(compute_similarity("hog_pearson",aa[0],ab[0]))
                        if np.any(aa) and np.any(ab) else None}
                if row["method"]=="rg" and not args.no_faithfulness and np.any(aa) and np.any(ab):
                    with isolated_rng(stable_seed("pf",name)):
                        fa=faithfulness(a,x,y,aa)
                    with isolated_rng(stable_seed("pf",name)):
                        fb=faithfulness(b,x,y,ab)
                    maps["rg"].update(PF20_A=fa,PF20_B=fb,PF20_change=fb-fa)
            # Fixed law negative control: same model and target, regardless of B.
            # Identity condition also measures this explicitly through the DP.
            result["fixed_law_H_analytic"]=0.
            atomic_json(path,{"index":i,"image":name,"target":int(y.item()),
                "hashes":hashes,"condition":cond,"trajectory":result,"maps":maps,
                "probability_A":pa,"probability_B":pb,"prediction_change":abs(pb-pa),
                "seconds":time.perf_counter()-t})
            print(args.model,cond["name"],i,result,flush=True)
            gc.collect()
        if {"A":digest_state(a),"B":digest_state(b)}!=hashes:
            raise RuntimeError("Attribution or diagnostic mutated a checkpoint")


if __name__=="__main__": main()
