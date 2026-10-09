#!/usr/bin/env python3
"""Write the one-factor ablation rows around the selected configurations.

Usage::

    python -m reproduce.ablation_rows --model vgg16 \\
        --selection results/selection/vgg16/selected.json --out ablations_vgg16.json

For every selected (loc / faith) row of RG, RG+Eq and SG, one
parameter at a time is varied over its validation-grid values while all other
parameters stay at the selected values:

* RG:    tau, epsilon, sigma^2 (only where a risk parameter is non-zero) and
         lambda (CNNs) or lambda_sm (ViT-B/16);
* RG+Eq: tau and lambda_sm (ViT-B/16);
* SG:    lambda, sigma^2 and the noisy gate (VGG-16);
* equivariance (ViT-B/16): RG+Eq at the parameters of each selected RG row;
* attention rule (ViT-B/16): RG with the gradient-fallback attention rule
  (autograd through the attention block with epsilon-LRP on its products)
  instead of Value-only routing, at each selected RG row and at the
  alpha-beta anchor.

Configurations that are inadmissible on validation (``selected.json``,
``inadmissible_rows``) are skipped.

The rows are evaluated on the test split with the localisation and
faithfulness metrics only; nothing is selected from them.  Row labels come
from ``reproduce.common.row_label``, so a point that equals an already
evaluated row is evaluated once.  The output is an ``--extra-rows`` JSON for
``reproduce.evaluate_selected``.  No torch needed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from reproduce.build_grids import (
    CNN_LAMBDA, RG_EPS, RG_TAUS, RG_VARS, RGEQ_LAMBDA_SM, SG_GATE, SG_LAMBDA, SG_VARS,
    VIT_LAMBDA_SM,
)
from reproduce.common import RGEQ_WRAPPER, canonical_params, load_json, row_label

CRITERIA = ("loc", "faith")
DEFAULT_VAR = 2.0     # sigma^2 used when a sweep switches a risk parameter on


def _axes(method: str, model: str) -> List[tuple]:
    vit = model == "vit_b16"
    if method == "rg":
        risk = ("lambda_sm", "adf_softmax_lambda", VIT_LAMBDA_SM) if vit else \
               ("lambda", "adf_lambda", CNN_LAMBDA)
        return [("tau", "routing_temperature", RG_TAUS), risk,
                ("sigma2", "adf_input_variance", RG_VARS), ("eps", "lrp_eps", RG_EPS)]
    if method == "rgeq":
        return [("tau", "routing_temperature", RG_TAUS),
                ("lambda_sm", "adf_softmax_lambda", RGEQ_LAMBDA_SM)]
    if method == "sg":
        return [("lambda", "adf_lambda", SG_LAMBDA), ("sigma2", "adf_input_variance", SG_VARS),
                ("gate", "adf_probabilistic", SG_GATE)]
    return []


def _risk_on(p: Dict[str, Any]) -> bool:
    return any(float(p.get(k, 0.0)) != 0.0
               for k in ("adf_lambda", "adf_softmax_lambda"))


def build_rows(selection: Dict[str, Any], model: str) -> List[Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}

    def _add(method, params, series, param, value, center):
        p = canonical_params(method, params)
        label = row_label(method, p)
        if label in inadmissible:
            return
        e = rows.setdefault(label, {"method": method, "row_label": label, "params": p, "sweeps": []})
        e["sweeps"].append({"series": series, "param": param, "value": value, "center": center})

    def _add_attention(base, tag, center_label):
        # value 0: Value-only routing (the centre row itself), 1: gradient fallback
        for v, mode in ((0.0, "v_only"), (1.0, "gradient_fallback")):
            p = dict(base)
            if mode != "v_only":
                p["attention_mode"] = mode
            _add("rg", p, f"rg:{tag}:attention", "attention", v, center_label)

    methods = selection.get("methods", {})
    # Configurations excluded by the admissibility rule on validation are not ablation points.
    inadmissible = {e["row_label"] for e in selection.get("inadmissible_rows", [])}
    for method in ("rg", "rgeq", "sg"):
        rec = methods.get(method)
        if not rec:
            continue
        centers = {}
        for crit in CRITERIA:
            if crit in rec:
                centers.setdefault(rec[crit]["row_label"], (rec[crit]["params"], []))[1].append(crit)
        for center_label, (base, crits) in centers.items():
            tag = "+".join(crits)
            for name, key, values in _axes(method, model):
                if key == "adf_input_variance" and not _risk_on(base):
                    continue                     # sigma^2 is inert without a risk parameter
                for v in values:
                    p = dict(base)
                    p[key] = v if isinstance(v, bool) else float(v)
                    if _risk_on(p) and "adf_input_variance" not in p:
                        p["adf_input_variance"] = DEFAULT_VAR
                    _add(method, p, f"{method}:{tag}:{name}", name, p[key], center_label)
            if method == "rg" and model == "vit_b16":
                p = dict(RGEQ_WRAPPER, **base)
                _add("rgeq", p, f"rg:{tag}:equivariance", "equivariance", 1.0, center_label)
                _add_attention(base, tag, center_label)
        if method == "rg" and model == "vit_b16":
            for e in rec.get("fixed", []):
                if e.get("is_anchor"):
                    _add_attention(e["params"], "anchor", e["row_label"])
    out = []
    for e in rows.values():
        sw = e.pop("sweeps")
        for s in sw:
            out.append(dict(e, sweep=s))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=("vit_b16", "vgg16", "resnet50"))
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    selection = load_json(args.selection)
    rows = build_rows(selection, args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1) + "\n")
    labels = {r["row_label"] for r in rows}
    print(f"{len(rows)} sweep points, {len(labels)} distinct rows -> {args.out}")


if __name__ == "__main__":
    main()
