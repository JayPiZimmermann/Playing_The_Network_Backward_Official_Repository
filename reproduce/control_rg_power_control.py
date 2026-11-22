"""Cache-only post-hoc power control for localization-selected ViT RG."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from reproduce.common import imagenet_s_split
from reproduce.control_common import atomic_json

ROW="rg__av-merged__eps-0.1__lam-0__lamsm-m10__tau-0.4__var-5"
BASELINES={"LRP-gamma":"lrp_gamma__eps-0.1__gamma-2",
           "AttnLRP":"attnlrp__eattn-0.25__gcomp-on__gconv-0.25__glin-0.05"}
POWERS=(.125,.25,.5,.75,1.,1.5,2.,4.,8.,16.)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()

def boot(x,seed=42):
    x=np.asarray(x,dtype=float); rng=np.random.default_rng(seed)
    b=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1)
    return {"mean":float(x.mean()),"ci95":np.quantile(b,[.025,.975]).tolist(),"n":len(x)}

def prepare_map(raw):
    a=np.abs(np.asarray(raw,dtype=np.float64)).squeeze(); a/=a.max()
    return a,np.argsort(-a.ravel(),kind="stable")

def power_scores(a,mask,p,order):
    z=a**p
    sorted_a=a.ravel()[order]; sorted_z=z.ravel()[order]
    if np.any(np.diff(sorted_z)>0): raise ValueError(f"power {p} reversed ranks")
    # A strictly monotone power must preserve tie classes too. This linear check
    # detects floating-point coalescing without sorting all 50,176 pixels again.
    if not np.array_equal(sorted_a[:-1]==sorted_a[1:], sorted_z[:-1]==sorted_z[1:]):
        raise ValueError(f"power {p} changed rank ties")
    inside=np.asarray(mask,dtype=bool).squeeze()
    if inside.ndim == 3:
        inside=inside.any(axis=-1)
    k=min(1000,z.size)
    return {"AL":float(z[inside].sum()/z.sum()),
            "PG":float(np.any(inside&(z==z.max()))),
            "TK":float(inside.ravel()[order[:k]].sum()/k)}

def split_scores(root,split,imroot,imsroot):
    ds=imagenet_s_split(split,imroot,imsroot); src=root/split/ROW
    paths=sorted(src.glob("????.json")); expected=50 if split=="val" else 565
    if len(paths)!=expected: raise ValueError((split,len(paths),expected))
    vals={p:{m:[] for m in ("AL","PG","TK","PF20")} for p in POWERS}; records=[]; sources=[]
    for jp in paths:
        npz=jp.with_suffix(".npz"); old=json.loads(jp.read_text()); idx=int(old["index"])
        _,mask,_,_,_=ds[idx]; raw=np.load(npz)["map"]; one=old["scores"]["1.0"]; row={}
        prepared,order=prepare_map(raw)
        for p in POWERS:
            s=power_scores(prepared,mask,p,order); s["PF20"]=float(one["PF20"]); row[str(p)]=s
            for m,v in s.items(): vals[p][m].append(v)
        for m in ("AL","PG","TK"):
            if abs(row["1.0"][m]-float(one[m]))>1e-9: raise ValueError((jp,m,row["1.0"][m],one[m]))
        records.append({"index":idx,"image":old["image"],"scores":row})
        sources.append({"json":str(jp),"json_sha256":sha(jp),"npz":str(npz),"npz_sha256":sha(npz)})
    agg={}
    for p in POWERS:
        agg[str(p)]={}
        for m,x in vals[p].items():
            stat=boot(x); stat["paired_difference_vs_power_1"]=boot(np.asarray(x)-np.asarray(vals[1.][m])); agg[str(p)][m]=stat
    return {"n":expected,"powers":agg,"per_image":records},sources

def baseline_targets(root):
    out={}
    for name,row in BASELINES.items():
        rr=[json.loads(p.read_text()) for p in sorted((root/"val"/row).glob("????.json"))]
        out[name]={"row":row,"validation_AL":float(np.mean([r["scores"]["1.0"]["AL"] for r in rr]))}
    return out

def baseline_curves(root):
    out={}
    for name,row in BASELINES.items():
        rr=[json.loads(p.read_text()) for p in sorted((root/"test"/row).glob("????.json"))]
        powers=sorted({float(p) for r in rr for p in r["scores"]})
        out[name]={str(p):{"n":len(rr),
                          "AL":float(np.mean([r["scores"][str(p)]["AL"] for r in rr])),
                          "PF20":float(np.mean([r["scores"][str(p)]["PF20"] for r in rr]))}
                   for p in powers}
    return out

def plot(result,path):
    fig,ax=plt.subplots(figsize=(5.8,3.8)); ps=list(POWERS)
    x=[result["test"]["powers"][str(p)]["AL"]["mean"] for p in ps]; y=[result["test"]["powers"][str(p)]["PF20"]["mean"] for p in ps]
    ax.plot(x,y,"o-",label="RG post-hoc power")
    offsets=[(0,8)]*7+[(-26,8),(0,22),(27,8)]
    for p,a,b,offset in zip(ps,x,y,offsets):
        ax.annotate(f"p={p:g}",(a,b),xytext=offset,textcoords="offset points",fontsize=7,ha="center")
    for name,curve in result["baseline_curves"].items():
        points=[curve[k] for k in sorted(curve,key=float)]
        ax.plot([d["AL"] for d in points],[d["PF20"] for d in points],"s--",label=f"{name} power")
    ax.set(xlabel="Attribution Localization (higher is better)",ylabel="PF20 (lower is better)"); ax.grid(alpha=.25); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path,dpi=200); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--cache-root",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); ap.add_argument("--imagenet-root",required=True); ap.add_argument("--imagenet-s-root",required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    val,vs=split_scores(a.cache_root,"val",a.imagenet_root,a.imagenet_s_root); test,ts=split_scores(a.cache_root,"test",a.imagenet_root,a.imagenet_s_root); targets=baseline_targets(a.cache_root); curves=baseline_curves(a.cache_root); matches={}
    for name,t in targets.items():
        p=min([q for q in POWERS if q<=1],key=lambda q:(abs(val["powers"][str(q)]["AL"]["mean"]-t["validation_AL"]),abs(q-1)))
        rr=[json.loads(x.read_text()) for x in sorted((a.cache_root/"test"/t["row"]).glob("????.json"))]
        matches[name]={**t,"selected_RG_power":p,"selected_RG_validation_AL":val["powers"][str(p)]["AL"]["mean"],"selection_rule":"nearest validation AL among RG powers <=1; ties prefer p nearest1","baseline_test_AL":float(np.mean([r["scores"]["1.0"]["AL"] for r in rr])),"baseline_test_PF20":float(np.mean([r["scores"]["1.0"]["PF20"] for r in rr])),"flattened_RG_test":test["powers"][str(p)]}
    result={"protocol":"rg-posthoc-power-control-v1","row":ROW,"transform":"(|a|/max|a|)^p","power_grid":list(POWERS),"rank_invariance":"verified per image; PF20 reused only after verification","selection_scope":"validation only","validation":val,"test":test,"targets":targets,"matches":matches,"baseline_curves":curves}
    atomic_json(a.out/"summary.json",result); atomic_json(a.out/"manifest.json",{"protocol":result["protocol"],"script":str(Path(__file__)),"script_sha256":sha(Path(__file__)),"source_files":{"val":vs,"test":ts}}); plot(result,a.out/"al_pf20_curve.png")
    print(json.dumps({"matches":matches,"test_curve":{str(p):{m:test["powers"][str(p)][m]["mean"] for m in ("AL","PG","TK","PF20")} for p in POWERS}},indent=2))
if __name__=="__main__": main()
