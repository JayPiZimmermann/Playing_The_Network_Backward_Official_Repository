"""Raw signed SG reconstruction and the published pp readout vs autograd."""
from __future__ import annotations
import argparse
import time
import torch
from reproduce.control_common import (
    add_args, atomic_json, dataset, image_record, load_model, sample_indices)


def identity(model,x,target,adf_mode="none"):
    from bwgame.dc_context import dc_decomposition
    model.zero_grad(set_to_none=True)
    inp=x.detach().clone().requires_grad_(True)
    score=model(inp)[0,target.item()]
    g=torch.autograd.grad(score,inp)[0].detach()
    model.zero_grad(set_to_none=True)
    with dc_decomposition(model,x.detach().clone(),backprop_mode="stopping",
                          adf_mode=adf_mode,adf_lambda=0.,adf_input_variance=1.,
                          adf_probabilistic=False) as dc:
        dc.output[0,target.item()].backward()
    rec=dc.reconstruct_gradient().detach()
    pp=dc.sensitivities.delta_pp.detach()
    def errors(a,b):
        return {"max_abs":float((a-b).abs().max()),
                "relative_l2":float((a-b).norm()/b.norm().clamp_min(1e-12)),
                "allclose":bool(torch.allclose(a,b,atol=1e-5,rtol=1e-4))}
    return {"signed_reconstruction":errors(rec,g),"published_pp":errors(pp,g),
            "signed_sum_channels":errors(rec.sum(1),g.sum(1)),
            "signed_maxabs_channels":errors(rec.abs().amax(1),g.abs().amax(1)),
            "published_pp_sum_vs_gradient_maxabs":errors(pp.sum(1),g.abs().amax(1)),
            "gradient_norm":float(g.norm()),"adf_mode":adf_mode}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_args(ap)
    ap.add_argument("--dtype",choices=("float32","float64"),default="float32")
    args=ap.parse_args()
    ds=dataset(args); device=torch.device(args.device or "cuda")
    model=load_model(args.model,device)
    model=model.to(dtype=getattr(torch,args.dtype))
    for i in sample_indices(ds,args.split,args.subset,args.shard,args.shards):
        path=args.out/args.model/args.dtype/f"{i:04d}.json"
        if path.exists(): continue
        x,_,target,name=image_record(ds,i,device)
        x=x.to(dtype=getattr(torch,args.dtype))
        start=time.perf_counter()
        result={mode:identity(model,x,target,mode) for mode in ("none","adf")}
        atomic_json(path,{"image":name,"index":i,"target":int(target.item()),
                          "results":result,"dtype":args.dtype,"seconds":time.perf_counter()-start,
                          "scope":"ReLU theorem on CNNs; conditioned/extended operators on ViT"})
        print(args.model,i,result,flush=True)


if __name__=="__main__": main()
