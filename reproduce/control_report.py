"""Aggregate completed control results without selecting on test data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from reproduce.control_common import atomic_json, scalar_stats, stable_seed


def records(path):
    return [json.loads(p.read_text()) for p in sorted(path.glob("[0-9][0-9][0-9][0-9].json"))]


def paired_ci(values,seed=42):
    a=np.asarray(values,dtype=float); a=a[np.isfinite(a)]
    if not len(a): return {"mean":None,"n":0,"ci95":None}
    rng=np.random.default_rng(seed)
    means=a[rng.integers(0,len(a),size=(2000,len(a)))].mean(axis=1)
    return {"mean":float(a.mean()),"n":len(a),"ci95":np.quantile(means,[.025,.975]).tolist()}


def correlation(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    ok=np.isfinite(x)&np.isfinite(y); x,y=x[ok],y[ok]
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return {"rho":None,"n":len(x),"ci95":None}
    rho=float(spearmanr(x,y).statistic)
    rng=np.random.default_rng(42); boot=[]
    for _ in range(1000):
        idx=rng.integers(0,len(x),len(x))
        if np.ptp(x[idx]) and np.ptp(y[idx]): boot.append(spearmanr(x[idx],y[idx]).statistic)
    return {"rho":rho,"n":len(x),"ci95":np.quantile(boot,[.025,.975]).tolist() if boot else None}


def robust_report(root,manifests):
    output={"rows":[],"validation_selections":[],"paired_risk_effects":[]}
    for model_dir in sorted(root.glob("*")):
        if not model_dir.is_dir(): continue
        by_split={}
        for split in ("val","test"):
            by_split[split]={}
            for rowdir in sorted((model_dir/split).glob("*")):
                rr=records(rowdir)
                if not rr: continue
                row=rr[0]["row"]
                scores={k:scalar_stats([r["metrics"].get(k) for r in rr])
                        for k in ("MaxS","AvgS","CosineMean","Top10OverlapMean","AL")}
                entry={"model":model_dir.name,"split":split,"row":row,"n":len(rr),"metrics":scores}
                output["rows"].append(entry)
                by_split[split][rowdir.name]=(entry,rr)
        # Separate fixed-background risk/gate families; require every candidate's
        # 50 validation images before selecting. No test metric enters selection.
        families={}
        for label,(entry,rr) in by_split["val"].items():
            for role in entry["row"].get("roles",[]):
                if role.startswith("baseline"): continue
                families.setdefault((entry["row"]["method"],role),[]).append(label)
        for (method,role),labels in families.items():
            manifest=manifests/f"robustness_{model_dir.name}.json"
            if not manifest.exists(): continue
            expected={r["row_label"] for r in json.loads(manifest.read_text())
                      if r["method"]==method and role in r.get("roles",[])}
            if set(labels)!=expected: continue
            if any(by_split["val"][l][0]["n"]!=50 for l in labels): continue
            valid=[l for l in labels if all(by_split["val"][l][0]["metrics"][k]["coverage"]==1
                                           for k in ("MaxS","AvgS"))]
            if not valid: continue
            ranks=sum(rankdata([by_split["val"][l][0]["metrics"][k]["mean"] for l in valid])
                      for k in ("MaxS","AvgS"))
            chosen=min(zip(ranks,valid),key=lambda q:(q[0],abs(by_split["val"][q[1]][0]["row"]["params"].get(
                "adf_softmax_lambda",0))+abs(by_split["val"][q[1]][0]["row"]["params"].get("adf_lambda",0)),q[1]))[1]
            output["validation_selections"].append({"model":model_dir.name,"method":method,
                "family":role,"selected":chosen,"candidates":labels,
                "test":by_split["test"].get(chosen,(None,))[0]})
        for label,(entry,rr) in by_split["test"].items():
            p=entry["row"]["params"]; risk=p.get("adf_softmax_lambda",0) or p.get("adf_lambda",0)
            if risk==0: continue
            def background(params):
                return {k:v for k,v in params.items() if k not in (
                    "adf_lambda","adf_softmax_lambda","adf_input_variance")}
            for off,(other,oo) in by_split["test"].items():
                op=other["row"]["params"]
                if other["row"]["method"]!=entry["row"]["method"] or op.get("adf_lambda",0) or op.get("adf_softmax_lambda",0): continue
                if background(p)!=background(op): continue
                oi={r["image"]:r for r in oo}
                pairs=[(r,oi[r["image"]]) for r in rr if r["image"] in oi]
                effects={k:paired_ci([a["metrics"][k]-b["metrics"][k] for a,b in pairs
                                     if a["metrics"].get(k) is not None and b["metrics"].get(k) is not None])
                         for k in ("MaxS","AvgS","CosineMean","Top10OverlapMean","AL")}
                output["paired_risk_effects"].append({"model":model_dir.name,"on":label,"off":off,
                                                     "difference":"on minus off","effects":effects})
    return output


def sharpening_report(root):
    out={"curves":[],"validation_selections":[],"matched_validation_AL":[]}
    grouped={}
    for model in root.glob("*"):
        for split in ("val","test"):
            for rowdir in (model/split).glob("*"):
                rr=records(rowdir)
                if not rr: continue
                for power in rr[0]["scores"]:
                    metrics={k:scalar_stats([r["scores"][power][k] for r in rr])
                             for k in ("AL","PG","TK","PF20")}
                    item={"model":model.name,"split":split,"row":rr[0]["row"],
                          "power":float(power),"n":len(rr),"metrics":metrics}
                    out["curves"].append(item)
                    grouped[(model.name,split,rowdir.name,float(power))]=item
    models=set(k[0] for k in grouped)
    for model in models:
        methods=set(i["row"]["method"] for i in out["curves"] if i["model"]==model)
        for method in methods:
            vv=[i for i in out["curves"] if i["model"]==model and i["split"]=="val" and i["row"]["method"]==method]
            from reproduce.control_sharpening import rows_for,POWERS
            source=Path(__file__).resolve().parents[1]/"results_summary"/"selection"/model/"selected.json"
            expected_rows=rows_for(json.loads(source.read_text()),model)
            expected={(r["row_label"],p) for r in expected_rows if r["method"]==method
                      for p in (POWERS if any(role.startswith("sharpen") for role in r["roles"]) else (1.,))}
            if {(i["row"]["row_label"],i["power"]) for i in vv}!=expected: continue
            if not vv or any(i["n"]!=50 for i in vv): continue
            ranks=sum(rankdata([-i["metrics"][k]["mean"] for i in vv]) for k in ("AL","PG","TK"))
            best=min(zip(ranks,vv),key=lambda r:(r[0],abs(r[1]["power"]-1),r[1]["row"]["row_label"]))[1]
            test=grouped.get((model,"test",best["row"]["row_label"],best["power"]))
            out["validation_selections"].append({"model":model,"method":method,"validation":best,"test":test})
            for target in (.55,.60,.65,.70):
                chosen=min(vv,key=lambda i:(abs(i["metrics"]["AL"]["mean"]-target),abs(i["power"]-1)))
                out["matched_validation_AL"].append({"model":model,"method":method,"target_AL":target,
                    "validation":chosen,"test":grouped.get((model,"test",chosen["row"]["row_label"],chosen["power"]))})
    return out


def diagnostic_report(root):
    result=[]
    for model in root.glob("*"):
        for mode in model.glob("*"):
            for cond in mode.glob("*"):
                rr=records(cond)
                if not rr: continue
                keys=[k for k in rr[0]["trajectory"] if k.startswith("H_") and not k.endswith("reason")]
                row={"model":model.name,"mode":mode.name,"condition":cond.name,"n":len(rr),
                     "hellinger":{k:scalar_stats([r["trajectory"].get(k) for r in rr]) for k in keys},
                     "correlations":{}}
                for h in keys:
                    hs=[r["trajectory"].get(h) for r in rr]
                    for method in ("gradient","rg"):
                        for metric in ("spearman_abs","hog_pearson"):
                            row["correlations"][h+"__"+method+"__"+metric]=correlation(hs,[r["maps"][method].get(metric) for r in rr])
                    pf=[abs(r["maps"]["rg"]["PF20_change"]) if "PF20_change" in r["maps"]["rg"] else None for r in rr]
                    row["correlations"][h+"__abs_PF20_change"]=correlation(hs,pf)
                    # Within-condition rank residuals control prediction and map
                    # change. Layer and perturbation magnitude are already fixed.
                    arr=np.array([[r["trajectory"].get(h),pf[i],r["prediction_change"],
                                   r["maps"]["rg"].get("spearman_abs")] for i,r in enumerate(rr)],dtype=float)
                    arr=arr[np.isfinite(arr).all(1)]
                    # A partial rank correlation is undefined if either target
                    # variable is constant (e.g. the identity control).
                    if len(arr)>5 and np.ptp(arr[:,0]) and np.ptp(arr[:,1]):
                        ranks=np.apply_along_axis(rankdata,0,arr)
                        cov=np.column_stack([np.ones(len(ranks)),ranks[:,2:]])
                        residual=ranks[:,:2]-cov@np.linalg.lstsq(cov,ranks[:,:2],rcond=None)[0]
                        rho=np.corrcoef(residual.T)[0,1]
                        row["correlations"][h+"__partial_rank_abs_PF20_change"]={
                            "rho":float(rho) if np.isfinite(rho) else None,"n":len(arr),
                            "controls":["prediction change","RG map Spearman"]}
                    else:
                        row["correlations"][h+"__partial_rank_abs_PF20_change"]={
                            "rho":None,"n":len(arr),
                            "controls":["prediction change","RG map Spearman"],
                            "reason":"undefined because Hellinger or absolute PF20 change is constant"}
                result.append(row)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=Path("results/controls"))
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--manifests",type=Path,
                    default=Path(__file__).resolve().parents[1]/"results_summary"/"controls"/"manifests")
    args=ap.parse_args()
    report={"robustness":robust_report(args.root/"robustness",args.manifests),
            "sharpening":sharpening_report(args.root/"sharpening"),
            "diagnostics":diagnostic_report(args.root/"diagnostics")}
    atomic_json(args.out/"aggregate.json",report)
    lines=["# Control experiment results","", "Only completed image records are included. Partial rows are not final results.","",
           "## Robustness","", "| Model | Split | Row | n | MaxS | AvgS | Cosine | Top-10% overlap |",
           "|---|---|---|---:|---:|---:|---:|---:|"]
    def fmt(x): return "undefined" if x is None else f"{x:.4f}"
    for r in report["robustness"]["rows"]:
        lines.append("| "+" | ".join([r["model"],r["split"],r["row"]["row_label"],str(r["n"])]+
             [fmt(r["metrics"][k]["mean"]) for k in ("MaxS","AvgS","CosineMean","Top10OverlapMean")])+" |")
    lines += ["","## Other experiments","",
              f"Sharpening curve entries: {len(report['sharpening']['curves'])}.",
              f"Diagnostic conditions with results: {len(report['diagnostics'])}.",
              "Full per-condition correlations, paired risk differences and validation selections are in aggregate.json.",
              "No explanation-quality claim follows from Hellinger alone; correlations are descriptive and require inspection of coverage and uncertainty."]
    (args.out/"results.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__": main()
