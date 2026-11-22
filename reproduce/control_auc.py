"""Full deletion/insertion probability and logit AUC.

101 points including both endpoints; ranked spatial pixels replaced/restored
in all RGB channels together; per-channel image-mean baseline. This full-curve
protocol is reported separately from the submitted channel-feature PF20.

The control run reuses the audited attribution-map caches from the sharpening
control. This avoids regenerating explanations and guarantees that monotone
power sharpening reuses exactly the same deletion/insertion curves.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from reproduce.control_common import (
    add_args,atomic_json,dataset,digest_state,image_record,load_model,sample_indices)


def full_curves(model,x,y,a,steps=100,batch_size=8):
    order=np.argsort(-np.abs(a).ravel(),kind="stable")
    base=x.mean(dim=(-2,-1),keepdim=True).expand_as(x).clone()
    original=x.flatten(2); fill=base.flatten(2)
    cuts=np.rint(np.linspace(0,len(order),steps+1)).astype(int)
    result={"fractions":(cuts/len(order)).tolist()}
    for mode in ("deletion","insertion"):
        logits=[]; probabilities=[]
        for start in range(0,len(cuts),batch_size):
            inputs=[]
            for n in cuts[start:start+batch_size]:
                v=(original if mode=="deletion" else fill).clone()
                idx=torch.as_tensor(order[:n].copy(),device=x.device)
                v[:,:,idx]=(fill if mode=="deletion" else original)[:,:,idx]
                inputs.append(v.reshape_as(x))
            with torch.no_grad():
                out=model(torch.cat(inputs))
                logits.extend(out[:,y.item()].cpu().tolist())
                probabilities.extend(torch.softmax(out,dim=1)[:,y.item()].cpu().tolist())
        result[mode+"_logits"]=logits; result[mode+"_probabilities"]=probabilities
        result[mode+"_logit_auc"]=float(np.trapz(logits,result["fractions"]))
        result[mode+"_probability_auc"]=float(np.trapz(probabilities,result["fractions"]))
    return result


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_args(ap)
    ap.add_argument("--map-root",type=Path,required=True,
                    help="Sharpening cache root containing <split>/<row>/*.npz")
    ap.add_argument("--row",required=True,
                    help="Exact cached row label to evaluate")
    ap.add_argument("--steps",type=int,default=100)
    ap.add_argument("--batch-size",type=int,default=16)
    args=ap.parse_args()
    device=torch.device(args.device or "cuda"); model=load_model(args.model,device); ds=dataset(args)
    cache_dir=args.map_root/args.split/args.row
    if not cache_dir.is_dir():
        raise FileNotFoundError(cache_dir)
    cached_json=sorted(cache_dir.glob("[0-9][0-9][0-9][0-9].json"))
    if not cached_json:
        raise FileNotFoundError(f"No cached row records in {cache_dir}")
    row=json.loads(cached_json[0].read_text())["row"]
    if row["row_label"] != args.row:
        raise RuntimeError(f"Cached row mismatch: {row['row_label']} != {args.row}")
    ids=sample_indices(ds,args.split,args.subset,args.shard,args.shards)
    manifest={
        "protocol":"full-spatial-deletion-insertion-v1",
        "model":args.model,"model_state_sha256":digest_state(model),
        "split":args.split,"row":row,"steps":args.steps,
        "points":args.steps+1,"batch_size":args.batch_size,
        "rank":"stable descending absolute spatial attribution",
        "perturbation_unit":"one spatial pixel, all RGB channels together",
        "baseline":"per-image, per-channel spatial mean",
        "scores":["raw target logit","target softmax probability"],
        "auc":"trapezoidal over the full perturbed fraction [0,1]",
        "relation_to_submission":(
            "Separate full spatial-pixel curve; submitted PF5/PF20 used the "
            "Quantus channel-feature curve truncated to 5%/20%."),
        "source_map_root":str(cache_dir.resolve()),"indices":ids,
    }
    dest=args.out/args.model/args.row
    dest.mkdir(parents=True,exist_ok=True)
    atomic_json(dest/f"manifest_shard{args.shard:03d}.json",manifest)
    for i in ids:
        path=dest/f"{i:04d}.json"
        if path.exists():
            old=json.loads(path.read_text())
            if old.get("protocol")==manifest["protocol"] and old.get("steps")==args.steps:
                continue
            raise ValueError(f"Incompatible cached result: {path}")
        source_json=cache_dir/f"{i:04d}.json"
        source_npz=cache_dir/f"{i:04d}.npz"
        if not source_json.exists() or not source_npz.exists():
            raise FileNotFoundError(f"Missing source cache for index {i}")
        source=json.loads(source_json.read_text())
        x,mask,y,name=image_record(ds,i,device)
        if source["index"]!=i or source["image"]!=name or source["target"]!=int(y.item()):
            raise RuntimeError(f"Map/data identity mismatch at index {i}")
        with np.load(source_npz) as archive:
            if "map" not in archive:
                raise KeyError(f"No map array in {source_npz}")
            a=np.asarray(archive["map"],dtype=np.float32)
        curves=full_curves(model,x,y,a,steps=args.steps,batch_size=args.batch_size)
        atomic_json(path,{"protocol":manifest["protocol"],"steps":args.steps,
                          "row":row,"index":i,"image":name,"target":int(y.item()),
                          "source_map_sha256":sha256(source_npz),**curves})
        print(args.model,args.row,i,flush=True)


if __name__=="__main__": main()
