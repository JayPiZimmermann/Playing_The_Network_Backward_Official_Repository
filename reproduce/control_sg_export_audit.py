"""Measure the unchanged SG wrapper's raw and normalized exported maps."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from reproduce.control_common import add_args,atomic_json,dataset,image_record,load_model


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    add_args(ap)
    args=ap.parse_args()
    from bwgame.xai.methods.adf_game import AdfGame
    from bwgame.dc_context import dc_decomposition
    from quantus.functions.normalise_func import normalise_by_max
    model=load_model(args.model,torch.device(args.device or "cpu"))
    ds=dataset(args);x,mask,y,name=image_record(ds,0,next(model.parameters()).device)
    selected=json.loads((Path(__file__).resolve().parents[1]/"results_summary/selection"/args.model/"selected.json").read_text())
    settings={"hard_gate_lambda_zero":{"adf_lambda":0.,"adf_input_variance":1.,"adf_probabilistic":False}}
    for criterion in ("loc","faith"):
        settings[criterion]=selected["methods"]["sg"][criterion]["params"]
    for label,params in settings.items():
        model.zero_grad(set_to_none=True)
        method=AdfGame(model,x.device)
        for k,v in params.items():setattr(method,k,v)
        a=method.attribute(x,y)["pp"]
        method.cleanup()
        # Independently check the raw sensitivity scale at the hard-gate anchor.
        streams=None
        if label=="hard_gate_lambda_zero":
            model.zero_grad(set_to_none=True)
            with dc_decomposition(model,x,backprop_mode="stopping",adf_mode="adf",
                                  adf_lambda=0.,adf_input_variance=1.,adf_probabilistic=False) as dc:
                dc.output[0,int(y.item())].backward()
            s=dc.sensitivities
            streams={k:float(getattr(s,k).abs().max()) for k in ("delta_pp","delta_np","delta_pn","delta_nn")}
            pp=s.delta_pp.sum(1).detach().cpu().numpy()
            assert np.allclose(a,pp,rtol=1e-5,atol=0.)
        normalized=normalise_by_max(a.copy(),normalise_axes=[1,2])
        rec={"image":name,"target":int(y.item()),"params":params,"device":str(x.device),
             "dtype":str(x.dtype),"raw_min":float(a.min()),"raw_max":float(a.max()),
             "normalized_min":float(normalized.min()),"normalized_max":float(normalized.max()),
             "AL_raw":float(a[0][mask].sum()/a.sum()),
             "AL_normalized":float(normalized[0][mask].sum()/normalized.sum()),
             "stream_maxima":streams,"scope":"Fresh current-code reproduction; not an archived raw map"}
        atomic_json(args.out/f"{label}.json",rec)
        print(label,rec,flush=True)


if __name__=="__main__":main()
