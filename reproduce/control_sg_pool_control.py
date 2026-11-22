"""Isolate pooling-cache and sensitivity-centering effects on SG pp maps.

Localization-only controls; no claim about deletion, robustness or runtime.
All settings and scalar-output seeds match the existing SG wrapper.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from reproduce.control_common import add_args,atomic_json,dataset,image_record,load_model,normalized_map
from reproduce.control_sharpening import sharpen_metrics


def controlled(model,x,y,params,alpha):
    from bwgame.alignment_cache import AlignmentCache,AlignmentMode
    from bwgame.dc_context import dc_decomposition
    class PoolOnlyCache(AlignmentCache):
        def should_use_cached_mask(self,name):
            return self.get_maxpool_indices(name) is not None
    cache=PoolOnlyCache(mode=AlignmentMode.NONE)
    cache.capture_original(model,x,capture_forward=True,capture_backward=False)
    assert cache.original_input_grad is None
    cache.attach_to_model(model)
    try:
        model.zero_grad(set_to_none=True)
        with dc_decomposition(model,x,backprop_mode="stopping",adf_mode="adf",
                              alpha=alpha,**params) as dc:
            dc.output[0,int(y.item())].backward()
        pp=dc.sensitivities.delta_pp.sum(1).detach().cpu().numpy()
        signed=dc.reconstruct_gradient().abs().amax(1).detach().cpu().numpy()
        return pp,signed
    finally:
        cache.detach_from_model(model)


def main():
    ap=argparse.ArgumentParser(description=__doc__);add_args(ap)
    ap.add_argument("--case",choices=("anchor","loc","faith"),required=True)
    args=ap.parse_args()
    from bwgame.xai.methods.adf_game import AdfGame
    selected=json.loads((Path(__file__).resolve().parents[1]/"results_summary/selection"/args.model/"selected.json").read_text())
    params=({"adf_lambda":0.,"adf_input_variance":1.,"adf_probabilistic":False}
            if args.case=="anchor" else selected["methods"]["sg"][args.case]["params"])
    model=load_model(args.model,torch.device(args.device or "cpu"));ds=dataset(args)
    for i in range(len(ds)):
        path=args.out/args.case/f"{i:04d}.json"
        if path.exists():continue
        started=time.perf_counter();x,mask,y,name=image_record(ds,i,next(model.parameters()).device)
        model.zero_grad(set_to_none=True)
        method=AdfGame(model,x.device)
        for k,v in params.items():setattr(method,k,v)
        legacy=method.attribute(x,y)["pp"];method.cleanup()
        pool,_=controlled(model,x,y,params,0.)
        centered,signed=controlled(model,x,y,params,.5)
        maps={"legacy_pp":legacy,"original_pool_pp":pool,"centered_pp":centered,"centered_signed_maxabs":signed}
        scores={key:sharpen_metrics(a,mask,1.) for key,a in maps.items()}
        norm={key:normalized_map(a) for key,a in maps.items()}
        comparison={key:{"normalized_max_abs_difference":float(np.max(np.abs(a-norm['legacy_pp']))),
                         "normalized_relative_l2_difference":float(np.linalg.norm(a-norm['legacy_pp'])/np.linalg.norm(norm['legacy_pp']))}
                    for key,a in norm.items() if key!="legacy_pp"}
        rec={"case":args.case,"index":i,"image":name,"target":int(y.item()),"params":params,
             "scores":scores,"comparison":comparison,"seconds":time.perf_counter()-started,
             "scope":"float32 CPU, validation; pooling-only retains legacy zero convention"}
        atomic_json(path,rec);print(json.dumps(rec),flush=True)


if __name__=="__main__":main()
