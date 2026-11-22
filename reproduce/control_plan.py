"""Write the fixed rows of the robustness controls; no configuration search."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from reproduce.ablation_rows import build_rows
from reproduce.common import canonical_params, row_label
from reproduce.control_common import atomic_json


def rows_for(model, selection):
    rows = {}
    def add(method, params, role):
        p=canonical_params(method,params)
        label=row_label(method,p)
        rec=rows.setdefault(label,{"method":method,"params":p,"row_label":label,"roles":[]})
        if role not in rec["roles"]: rec["roles"].append(role)
    if model != "vit_b16":
        for r in build_rows(selection,model):
            if r["sweep"]["param"] in ("lambda_sm","lambda","gate"):
                add(r["method"],r["params"],r["sweep"]["series"])
    anchor={"lrp_alpha":2.,"lrp_beta":1.,"lrp_eps":.5,"routing_temperature":1.,
            "adf_lambda":0.,"adf_softmax_lambda":0.,"adf_input_variance":2.,"av_rule":"merged"}
    if model=="vit_b16":
        center={"lrp_alpha":2.,"lrp_beta":1.,"lrp_eps":.1,"routing_temperature":.4,
                "adf_lambda":0.,"adf_softmax_lambda":0.,"adf_input_variance":5.,"av_rule":"merged"}
        add("rg",center,"selected-localization-risk-off-reference")
        add("rg",dict(center,adf_softmax_lambda=-50.),"validation-selected-localization-risk")
        return list(rows.values())
    else:
        for risk in (0.,-.3,-1.,-2.,-5.):
            add("rg",dict(anchor,adf_lambda=risk),"anchor-risk")
    # Same baseline configurations, with fresh paired-noise measurements.
    for method in ("lrp_gamma","attnlrp"):
        for crit in ("loc","faith"):
            rec=selection["methods"].get(method,{}).get(crit)
            if rec: add(method,rec["params"],"baseline-"+crit)
    return list(rows.values())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection-root",type=Path,default=Path("results_summary/selection"))
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    for model in ("vit_b16","vgg16","resnet50"):
        selection=json.loads((args.selection_root/model/"selected.json").read_text())
        rows=rows_for(model,selection)
        atomic_json(args.out/f"robustness_{model}.json",rows)
        print(model,len(rows),"risk/gate/baseline rows")


if __name__=="__main__": main()
