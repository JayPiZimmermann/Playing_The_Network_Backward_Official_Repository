"""Locate SG/autograd divergence by precision, centering and layerwise errors."""
from __future__ import annotations
import argparse
import gc
import time
import torch
from reproduce.control_common import add_args, atomic_json, dataset, image_record, load_model


def trace(model,x,target,alpha=0.,capture=True,adf="none",original_pools=False,strict_zero=False):
    from bwgame.dc_context import dc_decomposition
    refs={}; layer_results={}; handles=[]; phase=["reference"]; pool_cache=None
    def make_hook(name):
        def pre(module,inputs):
            z=inputs[0]
            if not torch.is_tensor(z) or not z.requires_grad: return
            if phase[0]=="reference":
                refs[name]={"value":z.detach().cpu()}
                z.register_hook(lambda g: refs[name].update(gradient=g.detach().cpu()))
            elif z.shape[0]==4 and name in refs:
                q=z.shape[0]//4
                v=(z[:q]-z[q:2*q]).detach().cpu().double()
                orig=refs[name]["value"].double()
                if v.shape!=orig.shape: return
                lr=layer_results.setdefault(name,{})
                lr["forward_max_abs"]=float((v-orig).abs().max())
                if isinstance(module,torch.nn.ReLU):
                    lr["forward_gate_disagreements"]=int(((v>0)!=(orig>0)).sum())
                    lr["original_zero_count"]=int((orig==0).sum())
                if isinstance(module,torch.nn.MaxPool2d):
                    import torch.nn.functional as F
                    _, ref_idx=F.max_pool2d(orig,module.kernel_size,module.stride,module.padding,return_indices=True)
                    _, dc_idx=F.max_pool2d(v,module.kernel_size,module.stride,module.padding,return_indices=True)
                    lr["pool_winner_disagreements"]=int((ref_idx!=dc_idx).sum())
                    values=orig.flatten(2)
                    gaps=(values.gather(2,ref_idx.flatten(2))-values.gather(2,dc_idx.flatten(2)))
                    lr["changed_pool_winner_gap_max"]=float(gaps.abs().max())
                def grad_hook(g):
                    pp,np_,pn,nn=g.chunk(4)
                    rec=(pp.double()-np_.double()-pn.double()+nn.double()).detach().cpu()
                    expected=refs[name]["gradient"].double()
                    lr.update(stream_max_abs=float(g.abs().max()),
                              gradient_max_abs=float(expected.abs().max()),
                              reconstruction_max_abs=float((rec-expected).abs().max()),
                              reconstruction_relative_l2=float((rec-expected).norm()/expected.norm().clamp_min(1e-20)))
                z.register_hook(grad_hook)
        return pre
    for name,module in model.named_modules():
        if isinstance(module,(torch.nn.Conv2d,torch.nn.Linear,torch.nn.ReLU,torch.nn.MaxPool2d)):
            handles.append(module.register_forward_pre_hook(make_hook(name)))
    try:
        model.zero_grad(set_to_none=True)
        inp=x.detach().clone().requires_grad_(True)
        score=model(inp)
        original_output=score.detach().cpu()
        score[0,target.item()].backward()
        ref=inp.grad.detach().clone()
        phase[0]="dc"
        model.zero_grad(set_to_none=True)
        if original_pools:
            from bwgame.alignment_cache import AlignmentCache, AlignmentMode
            # Forward routing only: no reference gradients or backward corrections.
            class PoolOnlyCache(AlignmentCache):
                def should_use_cached_mask(self,name):
                    return self.has_cached_mask(name) if strict_zero else self.get_maxpool_indices(name) is not None
            pool_cache=PoolOnlyCache(mode=AlignmentMode.NONE)
            pool_cache.capture_original(model,x,capture_forward=True,capture_backward=False)
            if strict_zero:
                assert not capture and adf=="none", "Use the forward-only cache for strict ReLU gates"
                for data in pool_cache._layer_data.values():
                    if data.relu_mask is not None:
                        # ReLU output > 0 is exactly the original input > 0.
                        data.relu_mask=data.original_activation>0
            assert pool_cache.original_input_grad is None
            assert all(d.original_gradient is None for d in pool_cache._layer_data.values())
            pool_cache.attach_to_model(model)
        with dc_decomposition(model,x.detach().clone(),backprop_mode="stopping",alpha=alpha,
                              store_activations=capture,adf_mode=adf,adf_lambda=0.,
                              adf_probabilistic=False) as dc:
            output_error=float((dc.output.detach().cpu()-original_output).abs().max())
            dc.output[0,target.item()].backward()
        ss=dc.sensitivities
        rec=dc.reconstruct_gradient()
        rec64=ss.delta_pp.double()-ss.delta_np.double()-ss.delta_pn.double()+ss.delta_nn.double()
        return {"alpha":alpha,"capture_original_gates":capture or strict_zero,"original_pools":original_pools,
                "relu_stop_at_zero":strict_zero,"adf":adf,
                "dtype":str(x.dtype),"output_max_abs":output_error,
                "gradient_max_abs":float(ref.abs().max()),
                "stream_max_abs":max(float(getattr(ss,k).abs().max()) for k in
                                     ("delta_pp","delta_np","delta_pn","delta_nn")),
                "reconstruction_max_abs":float((rec-ref).abs().max()),
                "reconstruction_relative_l2":float((rec-ref).norm()/ref.norm()),
                "final_subtraction_float64_max_abs":float((rec64-ref.double()).abs().max()),
                "layers":layer_results}
    finally:
        if pool_cache is not None: pool_cache.detach_from_model(model)
        for h in handles: h.remove()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_args(ap)
    ap.add_argument("--dtype",choices=("float32","float64"),required=True)
    ap.add_argument("--pool-control",action="store_true")
    ap.add_argument("--indices",type=int,nargs="+",default=[0])
    ap.add_argument("--strict-zero",action="store_true")
    args=ap.parse_args()
    device=torch.device(args.device or "cuda")
    ds=dataset(args); model=load_model(args.model,device).to(dtype=getattr(torch,args.dtype))
    settings=((.5,True,"none"),) if args.pool_control else ((0.,False,"none"),(0.,True,"none"),(0.,True,"adf"),(.5,True,"none"),(.5,True,"adf"))
    if args.strict_zero:
        assert args.pool_control
        settings=((.5,False,"none"),)
    for index in args.indices:
        x,_,y,name=image_record(ds,index,device); x=x.to(dtype=getattr(torch,args.dtype))
        for alpha,capture,adf in settings:
            for pools in ((True,) if args.strict_zero else ((False,True) if args.pool_control else (False,))):
                t=time.perf_counter()
                result=trace(model,x,y,alpha,capture,adf,pools,args.strict_zero)
                result.update(image=name,index=index,target=int(y.item()),seconds=time.perf_counter()-t)
                suffix=f"_pools{pools}_image{index}" if args.pool_control or index else ""
                if args.strict_zero: suffix+="_strictzero"
                path=args.out/args.model/f"{args.dtype}_alpha{alpha}_capture{capture}_{adf}{suffix}.json"
                atomic_json(path,result)
                print({k:v for k,v in result.items() if k!="layers"},flush=True)
                gc.collect()


if __name__=="__main__": main()
