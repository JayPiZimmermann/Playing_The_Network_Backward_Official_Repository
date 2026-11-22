"""Validation-selected power sharpening versus RG temperature sweeps.

PF20 is measured once per base explanation; reuse for a power is allowed only
after verifying identical stable rank order of all spatial pixels. AL/PG/TK
use the same nonnegative spatial map and ground-truth mask conventions.
"""
from __future__ import annotations
import argparse
import json
import time
import numpy as np
import torch
from reproduce.common import canonical_params, row_label
from reproduce.control_common import (
    add_args, atomic_json, dataset, explain, image_record, load_model,
    sample_indices, stable_seed)
from reproduce.control_diagnostics import faithfulness


POWERS=(1.,1.5,2.,3.,4.,8.,16.)


def rows_for(selection,model):
    rows={}
    def add(method,p,role):
        p=canonical_params(method,p); label=row_label(method,p)
        r=rows.setdefault(label,{"method":method,"params":p,"row_label":label,"roles":[]})
        if role not in r["roles"]: r["roles"].append(role)
    for method in ("lrp_gamma","attnlrp"):
        for criterion in ("loc","faith"):
            r=selection["methods"].get(method,{}).get(criterion)
            if r: add(method,r["params"],"sharpen-"+criterion)
    for r in selection["methods"]["rg"].get("fixed",[]):
        if r.get("is_anchor"): add("rg",r["params"],"sharpen-anchor")
    for criterion in ("loc","faith"):
        p=selection["methods"]["rg"][criterion]["params"]
        for tau in (.3,.4,.5,.6,.7,.8,.9,1.):
            add("rg",dict(p,routing_temperature=tau),"tau-"+criterion)
    return list(rows.values())


def sharpen_metrics(a,mask,power):
    a=np.abs(np.asarray(a,dtype=np.float64)).squeeze()
    a=a/a.max()
    sharpened=a**power
    # Powers are strictly increasing mathematically; reject numerical rank loss.
    if not np.array_equal(np.argsort(-a.ravel(),kind="stable"),
                          np.argsort(-sharpened.ravel(),kind="stable")):
        raise ValueError("Power sharpening changed ranks numerically")
    inside=np.asarray(mask,dtype=bool).squeeze()
    # Paper protocol / Quantus TopKIntersection: fixed k=1000, not mask size.
    k=min(1000,sharpened.size)
    order=np.argsort(-sharpened.ravel())
    return {"AL":float(sharpened[inside].sum()/sharpened.sum()),
            "PG":float(np.any(inside & (sharpened==sharpened.max()))),
            "TK":float(inside.ravel()[order[:k]].sum()/k)}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_args(ap)
    ap.add_argument("--selection",required=True)
    ap.add_argument("--row",default=None)
    args=ap.parse_args()
    rows=rows_for(json.loads(open(args.selection).read()),args.model)
    ds=dataset(args); device=torch.device(args.device or "cuda")
    model=load_model(args.model,device)
    for row in rows:
        if args.row and row["row_label"]!=args.row: continue
        powers=POWERS if any(role.startswith("sharpen") for role in row["roles"]) else (1.,)
        for i in sample_indices(ds,args.split,args.subset,args.shard,args.shards):
            path=args.out/args.model/args.split/row["row_label"]/f"{i:04d}.json"
            if path.exists(): continue
            x,mask,y,name=image_record(ds,i,device)
            t=time.perf_counter()
            a=explain(model,row,x,y,stable_seed(args.seed,name))
            pf=faithfulness(model,x,y,a)
            scores={str(p):dict(sharpen_metrics(a,mask,p),PF20=pf) for p in powers}
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path.with_suffix(".npz"),map=a)
            atomic_json(path,{"row":row,"index":i,"image":name,"target":int(y.item()),
                "scores":scores,"rank_invariance_verified":True,"seconds":time.perf_counter()-t})
            print(args.model,row["row_label"],args.split,i,scores,flush=True)


if __name__=="__main__": main()
