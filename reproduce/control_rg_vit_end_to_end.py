#!/usr/bin/env python3
"""End-to-end ViT-B/16 alpha-beta relevance recovery audit.

Compares Routing Game hooks with an independent explicit reverse recurrence
for the same conditioned local rules. It is not a claim of an unconditional
full-network LRP theorem.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from datetime import datetime, timezone
from pathlib import Path

# Code root (the directory holding bwgame/, reproduce/ and results_summary/).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def stabilize(z, eps):
    return z + eps * torch.where(z == 0, torch.ones_like(z), z.sign())

def ab_linear(x, W, b, r, alpha, beta, eps):
    # Independent positive/negative contribution totals and transpose products.
    wp, wn = W.clamp_min(0), (-W).clamp_min(0)
    xp, xn = x.clamp_min(0), (-x).clamp_min(0)
    cp = xp @ wp.T + xn @ wn.T
    cn = xp @ wn.T + xn @ wp.T
    zp = stabilize(cp + b.clamp_min(0), eps)
    zn = stabilize(cn + (-b).clamp_min(0), eps)
    sp, sn = r / zp, r / zn
    rp = xp * (sp @ wp) + xn * (sp @ wn)
    rn = xp * (sn @ wn) + xn * (sn @ wp)
    return alpha * rp - beta * rn

def composite_value(A, X, W, b, R, alpha, beta, eps):
    # A (H,T,T), X (T,E), W (H,D,E), b (H,D), R (H,T,D).
    ans = torch.zeros_like(X)
    for h in range(A.shape[0]):
        w = W[h]; wp, wn = w.clamp_min(0), (-w).clamp_min(0)
        xp, xn = X.clamp_min(0), (-X).clamp_min(0)
        vp = xp @ wp.T + xn @ wn.T + b[h].clamp_min(0)
        vn = xp @ wn.T + xn @ wp.T + (-b[h]).clamp_min(0)
        zp, zn = stabilize(A[h] @ vp, eps), stabilize(A[h] @ vn, eps)
        sp = A[h].T @ (R[h] / zp); sn = A[h].T @ (R[h] / zn)
        # equivalent to summing signed X*W contribution tensor
        rp = xp * (sp @ wp) + xn * (sp @ wn)
        rn = xp * (sn @ wn) + xn * (sn @ wp)
        ans += alpha * rp - beta * rn
    return ans

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=ROOT/'results/controls/rg_recovery_vit_end_to_end.json')
    p.add_argument('--checkpoint',type=Path,default=Path.home()/'.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth')
    p.add_argument('--images',type=int,default=3)
    p.add_argument('--imagenet-root',default=os.environ.get('IMAGENET_ROOT', str(ROOT/'data'/'imagenet')))
    p.add_argument('--imagenet-s-root',default=os.environ.get('IMAGENET_S_ROOT', str(ROOT/'data'/'imagenet-s')))
    a=p.parse_args()
    import torch
    from torchvision.models import vit_b_16
    from reproduce.common import build_split
    from bwgame.xai.methods.routing_game import _RoutingGameHookManager
    import copy
    from bwgame.composite_replacer import decompose_composites
    from bwgame.functional_replacer import replace_inline_modules, _replace_functional_in_module
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    model=vit_b_16(weights=None).eval()
    model.load_state_dict(torch.load(a.checkpoint,map_location='cpu',weights_only=True))
    model=model.double().eval()
    dataargs=argparse.Namespace(max_images=None,allow_size_mismatch=False,voc_root=None,device='cpu',
                                imagenet_root=a.imagenet_root,imagenet_s_root=a.imagenet_s_root)
    data=build_split('imagenet-s','val',dataargs)
    rec=[]
    for ii in range(min(a.images,len(data))):
      image,_,label,_,name=data[ii]; x=image.unsqueeze(0).double()
      params=[(1.,0.),(2.,1.)]
      for alpha,beta in params:
       for eps in (1e-6,.5):
        start=time.perf_counter()
        routed=copy.deepcopy(model)
        decompose_composites(routed); replace_inline_modules(routed,inplace=True); _replace_functional_in_module(routed); routed.double().eval()
        mgr=_RoutingGameHookManager(routed,lrp_alpha=alpha,lrp_beta=beta,lrp_eps=eps,
                                    routing_temperature=1.,attention_mode='v_only',av_rule='merged')
        mgr.install()
        try:
          if len(mgr._qk_matmuls)!=12 or len(mgr._av_matmuls)!=12:
            raise RuntimeError(f'Expected 12 QK and AV modules, found {len(mgr._qk_matmuls)} and {len(mgr._av_matmuls)}')
          xi=x.clone().requires_grad_(True); logits=routed(xi)
          with torch.no_grad(): plain_logits=model(x)
          forward_error=float((logits.detach()-plain_logits).abs().max())
          if forward_error > 1e-10: raise RuntimeError(f'decomposed forward mismatch {forward_error}')
          target=int(logits.argmax(-1).item())
          seed=torch.zeros_like(logits); seed[0,target]=1.
          logits.backward(gradient=seed); hooked=xi.grad.detach().clone()
        finally: mgr.remove()
        # Fully independent ordinary forward cache and explicit backward rules.
        with torch.no_grad():
          z=model.conv_proj(x).reshape(1,768,196).permute(0,2,1)
          cls=model.class_token.expand(1,-1,-1); h=torch.cat((cls,z),dim=1)+model.encoder.pos_embedding
          blocks=[]
          for layer in model.encoder.layers:
            n1=layer.ln_1(h); qkv=torch.nn.functional.linear(n1,layer.self_attention.in_proj_weight,layer.self_attention.in_proj_bias)
            q,k,v=qkv.chunk(3,-1); q=q.reshape(1,197,12,64).transpose(1,2); k=k.reshape(1,197,12,64).transpose(1,2); v=v.reshape(1,197,12,64).transpose(1,2)
            A=torch.softmax((q@k.transpose(-2,-1))*(64**-.5),-1)
            av=(A@v).transpose(1,2).reshape(1,197,768)
            attnout=layer.self_attention.out_proj(av)
            h1=h+attnout; n2=layer.ln_2(h1); ff=layer.mlp(n2); hout=h1+ff
            blocks.append((h,n1,q,k,v,A,av,h1,n2,hout)); h=hout
          enc=model.encoder.ln(h); lg=model.heads(enc[:,0]); assert int(lg.argmax(-1))==target
          R=torch.zeros_like(lg); R[0,target]=1.
          R=ab_linear(enc[:,0],model.heads.head.weight,model.heads.head.bias,R,alpha,beta,eps)
          # Final norm is a passthrough; class-token readout only.
          Rh=torch.zeros_like(h); Rh[:,0]=R
          for layer,cache in zip(reversed(model.encoder.layers),reversed(blocks)):
            hin,n1,q,k,v,A,av,h1,n2,hout=cache
            # Reverse the MLP residual first. Full relevance reaches each Add operand.
            Rmlp_branch=Rh; Rh1=Rh
            fc1out=layer.mlp[0](n2)
            geluout=layer.mlp[1](fc1out)
            r=ab_linear(geluout[0],layer.mlp[3].weight,layer.mlp[3].bias,Rmlp_branch[0],alpha,beta,eps)
            r=ab_linear(n2[0],layer.mlp[0].weight,layer.mlp[0].bias,r,alpha,beta,eps)
            Rh1=Rh1+r.unsqueeze(0)
            # Then reverse attention residual and its output projection.
            Rbranch=Rh1; Rskip=Rh1
            Ratt=ab_linear(av,layer.self_attention.out_proj.weight,layer.self_attention.out_proj.bias,Rbranch,alpha,beta,eps)
            # Attention probabilities/QK are conditioned constants; route only merged V map.
            Rv=Ratt[0].reshape(197,12,64).permute(1,0,2)
            Rn1=composite_value(A[0],n1[0],layer.self_attention.in_proj_weight[1536:].reshape(12,64,768),layer.self_attention.in_proj_bias[1536:].reshape(12,64),Rv,alpha,beta,eps)
            # LayerNorm passthrough convention.
            Rh=Rskip+Rn1.unsqueeze(0)
            assert Rh.shape == (1,197,768), Rh.shape
          # Ordinary Add backward sends full relevance to the content input.
          Rp=Rh[:,1:,:]
          # Patch projection is conv stride=16; evaluate each patch as a flattened affine map.
          patches=torch.nn.functional.unfold(x,kernel_size=16,stride=16).transpose(1,2)[0]
          W=model.conv_proj.weight.flatten(1); b=model.conv_proj.bias
          assert Rp.shape[1] == patches.shape[0], (Rp.shape, patches.shape)
          patchR=ab_linear(patches,W,b,Rp[0],alpha,beta,eps)
          direct=torch.nn.functional.fold(patchR.T.unsqueeze(0),output_size=(224,224),kernel_size=16,stride=16)
        diff=(hooked-direct).abs(); denom=torch.linalg.vector_norm(direct)+1e-30
        row={'image_index':ii,'image':str(name),'label':int(label),'target_class':target,'target_logit':float(plain_logits[0,target]),'alpha':alpha,'beta':beta,'eps':eps,
             'max_abs_error':float(diff.max()),'mean_abs_error':float(diff.mean()),
             'reference_max_abs':float(direct.abs().max()),'reference_l2':float(torch.linalg.vector_norm(direct)),
             'relative_l2_error':float(torch.linalg.vector_norm(diff)/denom),'finite':bool(torch.isfinite(hooked).all() and torch.isfinite(direct).all()),
             'forward_max_abs_error':forward_error,'passed':bool(torch.isfinite(hooked).all() and torch.isfinite(direct).all() and float(torch.linalg.vector_norm(diff)/denom)<1e-10),'seconds':time.perf_counter()-start}
        rec.append(row); print(json.dumps(row),flush=True)
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'End-to-end pretrained torchvision ViT-B/16 input relevance comparison: RG hooks vs independent direct recurrence using observed fixed QK-softmax, zero Q/K relevance, merged AV+V composite map, full relevance to both residual-addition operands, LN/GELU passthrough, affine alpha-beta rules. Equality is conditional on these local rules and does not assert an unconditional full-ViT LRP theorem.',
      'addition_rule':'Full relevance sent to both operands at each addition by ordinary autograd (trajectory discount 2).','device':'cpu','threads':torch.get_num_threads(),'torch':torch.__version__,'torchvision':__import__('torchvision').__version__,'command':f'PYTHONPATH=code {sys.executable} {Path(__file__)} --images {a.images}',
      'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),'script_sha256':sha(__file__),'images_requested':a.images,
      'attention_modules_per_run':{'qk_matmuls':12,'av_matmuls':12},'checks':rec,'passed':bool(rec) and all(r['passed'] for r in rec)}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('RESULT',report['passed'],len(rec),a.out)
if __name__=='__main__':
    import torch
    main()
