#!/usr/bin/env python3
"""Numerical ViT attention recovery check against independent contributions.

Checks the conditioned composite Value map in every pretrained torchvision
ViT-B/16 block. It does not test a full-model alpha-beta attribution theorem.
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys
from datetime import datetime, timezone
from pathlib import Path
import torch

# Code root (the directory holding bwgame/, reproduce/ and results_summary/).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def composite_reference(A, X, W, b, R, alpha, beta, eps):
    # Factorized evaluation of the explicit c[k,d,e]=X[k,e]W[d,e] rule.
    # The stream totals are sums over e of c^+/c^-; the final input relevance
    # sums c^+/c^- times the independently computed per-output stream factors.
    xp, xn = X.clamp_min(0), (-X).clamp_min(0)
    wp, wn = W.clamp_min(0), (-W).clamp_min(0)
    vp = torch.nn.functional.linear(xp, wp) + torch.nn.functional.linear(xn, wn) + b.clamp_min(0)
    vn = torch.nn.functional.linear(xp, wn) + torch.nn.functional.linear(xn, wp) + (-b).clamp_min(0)
    zp, zn = A @ vp, A @ vn
    zp = zp + eps * torch.where(zp >= 0, 1., -1.)
    zn = zn + eps * torch.where(zn >= 0, 1., -1.)
    sp, sn = A.T @ (R / zp), A.T @ (R / zn)
    pos = xp * torch.nn.functional.linear(sp, wp.T) + xn * torch.nn.functional.linear(sp, wn.T)
    neg = xp * torch.nn.functional.linear(sn, wn.T) + xn * torch.nn.functional.linear(sn, wp.T)
    return alpha * pos - beta * neg

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'results/controls/rg_recovery_vit.json')
    p.add_argument('--checkpoint', type=Path, default=Path.home()/'.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth')
    p.add_argument('--imagenet-root', default=os.environ.get('IMAGENET_ROOT', str(ROOT / 'data' / 'imagenet')))
    p.add_argument('--imagenet-s-root', default=os.environ.get('IMAGENET_S_ROOT', str(ROOT / 'data' / 'imagenet-s')))
    p.add_argument('--images', type=int, default=3)
    a = p.parse_args()
    import torchvision
    from torchvision.models import vit_b_16
    from reproduce.common import build_split
    from bwgame.xai.methods.routing_game import _RoutingGameHookManager
    from bwgame.operations.matmul import DCMatMul
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261120)
    model = vit_b_16(weights=None).eval()
    model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
    model = model.double().eval().requires_grad_(False)
    args = argparse.Namespace(max_images=a.images, allow_size_mismatch=True, voc_root=None, device='cpu',
                              imagenet_root=a.imagenet_root, imagenet_s_root=a.imagenet_s_root)
    data = build_split('imagenet-s', 'val', args)
    # Fixed seed makes signed synthetic block relevance reproducible; image,
    # activations, QK, and all parameters are real. This is block recovery only.
    cases=[]
    for image_i in range(min(a.images, len(data))):
        image, _mask, label, _class_name, image_name = data[image_i]
        image_hash=hashlib.sha256(image.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        x = image.unsqueeze(0).double()
        layers={}
        handles=[]
        for i, layer in enumerate(model.encoder.layers):
            handles.append(layer.register_forward_pre_hook(lambda m, inp, i=i: layers.__setitem__(i, inp[0].detach())))
        with torch.no_grad(): model(x)
        for h in handles: h.remove()
        for bi, layer in enumerate(model.encoder.layers):
            z = layer.ln_1(layers[bi])  # (1,197,768), signed real LayerNorm output
            attn = layer.self_attention
            E=768; H=12; D=64
            qkv = torch.nn.functional.linear(z, attn.in_proj_weight, attn.in_proj_bias)
            q,k,v = (t.detach() for t in qkv.chunk(3, dim=-1))
            q=q.reshape(1,197,H,D).transpose(1,2); k=k.reshape(1,197,H,D).transpose(1,2)
            v=v.reshape(1,197,H,D).transpose(1,2)
            A=torch.softmax((q @ k.transpose(-2,-1)) * (D**-.5), dim=-1)[0]
            X=z[0]
            W=attn.in_proj_weight[2*E:3*E]
            b=attn.in_proj_bias[2*E:3*E]
            # Deterministic signed relevance at the actual AV output, one per head.
            R=torch.randn((H,197,D), dtype=torch.float64, generator=torch.Generator().manual_seed(991+image_i*12+bi))
            for alpha,beta in ((1.,0.),(2.,1.)):
                for eps in (1e-6,.5):
                    with torch.no_grad():
                        ref=sum(composite_reference(A[h], X, W[h*D:(h+1)*D], b[h*D:(h+1)*D], R[h], alpha,beta,eps) for h in range(H))
                    # Exercise actual production AV + Linear backward hooks on
                    # a wrapper carrying the real block weights and actual Q/K.
                    class RealAV(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.v_proj=torch.nn.Linear(E,E,bias=True,dtype=torch.float64)
                            self.qk=DCMatMul(); self.softmax=torch.nn.Softmax(dim=-1); self.av=DCMatMul()
                            with torch.no_grad(): self.v_proj.weight.copy_(W); self.v_proj.bias.copy_(b)
                        def forward(self, xx):
                            vv=self.v_proj(xx).reshape(1,197,H,D).transpose(1,2)
                            aa=self.softmax(self.qk(q,k.transpose(-2,-1))* (D**-.5))
                            return self.av(aa,vv)
                    wrapper=RealAV().eval()
                    mgr=_RoutingGameHookManager(wrapper,lrp_alpha=alpha,lrp_beta=beta,lrp_eps=eps,routing_temperature=1.,attention_mode='v_only',av_rule='merged')
                    mgr.install()
                    try:
                        xx=X.unsqueeze(0).detach().requires_grad_(True)
                        out=wrapper(xx)
                        out.backward(R.unsqueeze(0))
                        got=xx.grad.detach()[0]
                    finally:
                        mgr.remove()
                    diff=(ref-got).abs()
                    relative_l2=float(torch.linalg.vector_norm(diff)/(torch.linalg.vector_norm(ref)+1e-30))
                    cases.append({'image_index':image_i,'image':str(image_name),'transformed_image_sha256':image_hash,
                                  'block':bi,'alpha':alpha,'beta':beta,'eps':eps,
                                  'max_abs_error':float(diff.max()),'mean_abs_error':float(diff.mean()),
                                  'relative_l2_error':relative_l2,
                                  'finite':bool(torch.isfinite(ref).all() and torch.isfinite(got).all()),
                                  'passed':bool(torch.allclose(ref,got,rtol=2e-8,atol=2e-10) and relative_l2 < 1e-10)})
    source_paths=['reproduce/control_rg_attention_recovery.py','bwgame/xai/methods/routing_game.py',
                  'tests/test_attention_ab_recovery.py']
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'Conditioned block-level composite Value recovery only; all 12 torchvision ViT-B/16 blocks, actual pretrained weights and signed real LayerNorm activations, observed QK-softmax recomputed independently. Not full-network recovery or a theorem test.',
            'device':'cpu','threads':torch.get_num_threads(),'seed':20261120,'checkpoint':str(a.checkpoint),'checkpoint_sha256':sha(a.checkpoint),
            'runtime':{'python':sys.version.split()[0],'torch':torch.__version__,'torchvision':torchvision.__version__},
            'script_sha256':sha(__file__),
            'source_sha256':{name:sha(ROOT/name) for name in source_paths},
            'reference_seed_spec':'torch.manual_seed(20261120); each image/block AV relevance R is standard-normal float64 from a fresh Generator seeded 991 + image_index*12 + block_index; no gradient/reference injection.',
            'reference_method':'Independent fused composite alpha-beta-epsilon calculation from sign-separated X, W and bias streams with independently computed A^T(R/Z) factors; compared against input gradient from installed production RG hooks on wrapper with actual block Value weights and frozen observed QK queries/keys.',
            'images_requested':a.images,'checks':cases,'passed':bool(cases) and all(c['passed'] for c in cases)}
    a.out.parent.mkdir(parents=True,exist_ok=True); a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':report['passed'],'n':len(cases),'max_error':max((c['max_abs_error'] for c in cases),default=None),'output':str(a.out)}))

if __name__=='__main__':
    main()
