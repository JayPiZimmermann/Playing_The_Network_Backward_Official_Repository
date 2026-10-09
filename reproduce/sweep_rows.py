#!/usr/bin/env python3
"""Write the extra rows of the ViT tau / lambda_sm trade-off sweeps.

Usage::

    python -m reproduce.sweep_rows --model vit_b16 \\
        --selection results/selection/vit_b16/selected.json --out sweeps_vit.json

Starting from the loc-selected RG configuration of ``selected.json``:

* ``tau`` series: vary ``routing_temperature`` over the grid values
  (reproduce.build_grids.RG_TAUS) with every other parameter fixed;
* ``lambda_sm`` series: vary ``adf_softmax_lambda`` over the grid values
  (reproduce.build_grids.VIT_LAMBDA_SM) with every other parameter fixed.

Row labels come from ``reproduce.common.row_label``, so a sweep point that
equals an already selected row has the same label and is evaluated once by
``reproduce.evaluate_selected``.  The output is the ``--extra-rows`` JSON
consumed by ``reproduce.evaluate_selected``.  No torch needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reproduce.build_grids import RG_TAUS, VIT_LAMBDA_SM
from reproduce.common import canonical_params, load_json, row_label


def build_sweeps(selection: dict) -> list:
    rec = selection.get("methods", {}).get("rg")
    if not rec or "loc" not in rec:
        raise SystemExit("selected.json has no loc-selected RG row (methods.rg.loc)")
    base = dict(rec["loc"]["params"])
    rows = []
    for series, key, values in (("tau", "routing_temperature", RG_TAUS),
                                ("lambda_sm", "adf_softmax_lambda", VIT_LAMBDA_SM)):
        for v in values:
            p = dict(base)
            p[key] = float(v)
            if key == "adf_softmax_lambda" and float(v) != 0.0 and "adf_input_variance" not in p:
                # The loc row carries no sigma^2 when it is inert (lambda =
                # lambda_sm = 0, see canonical_params); the sweep uses the
                # grid's sigma^2 = 2.
                p["adf_input_variance"] = 2.0
            p = canonical_params("rg", p)
            label = row_label("rg", p)
            rows.append({"method": "rg", "row_label": label, "params": p,
                         "sweep": {"series": series, "param": series, "value": float(v),
                                   "center": rec["loc"]["row_label"]}})
    return rows


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="vit_b16", choices=("vit_b16",))
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    selection = load_json(args.selection)
    if selection.get("model") not in (None, args.model):
        raise SystemExit(f"{args.selection} is for model {selection.get('model')}, not {args.model}")
    rows = build_sweeps(selection)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"Wrote {len(rows)} sweep rows ({len({r['row_label'] for r in rows})} distinct) -> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
