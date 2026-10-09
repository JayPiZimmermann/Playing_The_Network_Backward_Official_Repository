#!/usr/bin/env python3
"""Write the declared ImageNet-S validation grid of one backbone.

Usage::

    python -m reproduce.build_grids --model vit_b16            # -> experiments/grids/imagenet_s_val_vit_b16.csv
    python -m reproduce.build_grids --model vgg16 --out my.csv

One row per (method, configuration), columns
``method, row_label, params_json, is_anchor, tunable``.  The grid follows
the paper's per-method grid table; parameter names are the attribute names
of the registered attribution methods (anything else would be silently
ignored by the evaluator, which ``reproduce.common.validate_params``
rejects at run time).

* Functionally duplicate RG / RG+Eq rows (different sigma^2 while
  lambda = lambda_sm = 0, where the ADF pass that consumes sigma^2 is not
  run) are emitted once (``reproduce.common.canonical_params``).
* ViT-B/16 RG / RG+Eq rows use the attention value-path rule
  ``av_rule='merged'`` (alpha-beta-LRP-eps on the composite map
  X -> O = A (X W_v^T + b_v), the rule of the attention recovery theorem).
  lambda (activation-side risk) is fixed to 0 on ViT-B/16: its GELU hook uses
  the full GELU derivative when lambda != 0, which contradicts the
  conditioned-GELU assumption.  The CNN grids have no attention and therefore
  no av_rule.
* The alpha-beta-LRP anchor (alpha, beta, eps, tau) = (2, 1, 0.5, 1) of RG and
  RG+Eq is part of the method's grid and flagged ``is_anchor=1``.
* Methods with a single configuration and the LIME / KernelSHAP reference
  rows have ``tunable=0``.

This script does not import torch.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from reproduce.common import (
    GRID_DIR, MODEL_METHODS, MODELS, RGEQ_WRAPPER, GridRow, canonical_params, check_model,
    get_spec, row_label, write_grid,
)

# ---------------------------------------------------------------------------
# Declared grid values
# ---------------------------------------------------------------------------
RG_TAUS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
RG_EPS = (0.1, 0.25, 0.5, 1.0)
RG_VARS = (1.0, 2.0, 5.0, 10.0)
RG_ALPHA_BETA = (2.0, 1.0)
VIT_LAMBDA = (0.0,)
VIT_LAMBDA_SM = (-50.0, -10.0, -5.0, -2.0, -1.0, 0.0)
# ViT-B/16 attention @ Value rule of RG / RG+Eq: the merged (composite) rule of
# the attention recovery theorem.
AV_RULES = ("merged",)
CNN_LAMBDA = (-5.0, -2.0, -1.0, -0.3, 0.0)
RG_ANCHOR = {"lrp_alpha": 2.0, "lrp_beta": 1.0, "lrp_eps": 0.5, "routing_temperature": 1.0}

RGEQ_LAMBDA_SM = (0.0, -10.0)
RGEQ_VAR = 2.0
RGEQ_EPS = 0.5

SG_LAMBDA = (-5.0, -2.0, -1.0, -0.3, 0.0)
SG_VARS = (1.0, 2.0, 5.0, 10.0)
SG_GATE = (True, False)

DAVE_N = (25, 50)
DAVE_ANGLE = (0.0, 10.0, 20.0)
DAVE_SHIFT = (0.0, 0.05, 0.1)
DAVE_NOISE = (0.0, 0.25, 0.5)
DAVE_FIXED = {"flip_prob": 0.5, "multiply_input": True, "batch_size": 1}

# LIME / KernelSHAP on 80 SLIC superpixels.
# LIME as the reference lime_image explainer (Ribeiro et al. 2016): N = 1000,
# Ridge surrogate, exponential kernel of width 0.25 on the cosine distance of
# the binary superpixel vectors, removed superpixels filled with their mean colour.
LIME_FIXED = {"n_samples": 1000, "superpixel_method": "slic", "superpixel_n_segments": 80,
              "surrogate": "ridge", "kernel_width": 0.25, "kernel_distance": "binary",
              "fill": "segment_mean"}
# KernelSHAP as shap.KernelExplainer (Lundberg & Lee 2017): N = 2M + 2048,
# removed superpixels filled with the image mean.
KERNELSHAP_FIXED = {"n_samples": "auto", "superpixel_method": "slic", "superpixel_n_segments": 80,
                    "fill": "image_mean"}
SMOOTHGRAD_FIXED = {"nt_samples": 50, "use_abs": False, "use_std_aggregation": False,
                    "channel_reduction": "max_abs"}


def _baseline_grid(method: str, model: str) -> List[Dict[str, Any]]:
    cnn = model in ("vgg16", "resnet50")
    if method in ("tibav", "gae"):
        return [{}]
    if method == "gradient":
        # Simonyan et al. (2014): max over colour channels of |dy/dx|.
        return [{"channel_reduction": "max_abs"}]
    if method in ("gcam", "gcampp", "lcam"):
        # pytorch-grad-cam at the last convolutional block (after its ReLU),
        # bilinear upsampling.
        return [{"target_layer_name": "features.29" if model == "vgg16" else "layer4"}]
    if method == "smoothgrad":
        # Smilkov et al. (2017): N = 50, noise sigma = 10-20 % of the input range.
        return [dict(SMOOTHGRAD_FIXED, stdev_spread=s) for s in (0.1, 0.15, 0.2)]
    if method == "intgrad":
        # Sundararajan et al. (2017): black-image baseline.
        return [{"baseline": "black", "n_steps": n} for n in (20, 50, 100, 300)]
    if method == "deeplift":
        # Shrikumar et al. (2017), Rescale rule (captum), CNNs only.
        if not cnn:
            raise KeyError("DeepLift is evaluated on the CNNs only")
        return [{"rule": "rescale"}]
    if method == "lrp_eps":
        return [{"epsilon": e} for e in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)]
    if method == "lrp_gamma":
        # CNNs: zennit's standard EpsilonGammaBox composite and the Gamma-on-
        # all-layers composite (the class default); selection picks the better
        # one per criterion.  ViT: Gamma on all Linear layers only (the standard
        # composite would reduce to LRP-epsilon there).
        composites = ([{"composite": "epsilon_gamma_box"}, {}] if model in ("vgg16", "resnet50")
                      else [{}])
        return [dict(c, epsilon=e, gamma=g) for c in composites
                for e in (0.1, 0.25, 0.5) for g in (0.1, 0.25, 0.5, 1.0, 2.0)]
    if method == "attnlrp":
        # AttnLRP gamma composite (Achtibat et al. 2024, lxt): Gamma on Conv and
        # Linear, Epsilon on the attention projections; lxt defaults as anchor.
        # (Its epsilon-only variant coincides with LRP-epsilon on ViT-B/16.)
        anchor = {"gamma_composite": True, "lrp_gamma_conv": 0.25, "lrp_gamma_linear": 0.05,
                  "lrp_epsilon_attn": 1e-6}
        return ([dict(anchor)]
                + [dict(anchor, lrp_gamma_linear=g) for g in (0.01, 0.025, 0.1, 0.25)]
                + [dict(anchor, lrp_gamma_conv=g) for g in (0.1, 0.5, 1.0)]
                + [dict(anchor, lrp_epsilon_attn=e) for e in (0.01, 0.05, 0.25)])
    if method == "ar":
        # w = 1 would make the rollout the identity (zero map for every input).
        return [{"residual_weight": w} for w in (0.0, 0.25, 0.5, 0.75)]
    if method == "dave":
        return [dict(DAVE_FIXED, n_samples=n, max_angle=a, max_shift_frac=s, noise_t_max=t)
                for n, a, s, t in itertools.product(DAVE_N, DAVE_ANGLE, DAVE_SHIFT, DAVE_NOISE)]
    if method == "lime":
        return [dict(LIME_FIXED)]
    if method == "kernelshap":
        return [dict(KERNELSHAP_FIXED)]
    if method == "sg":
        return [{"adf_lambda": lam, "adf_input_variance": var, "adf_probabilistic": gate}
                for lam, var, gate in itertools.product(SG_LAMBDA, SG_VARS, SG_GATE)]
    raise KeyError(method)


def _rg_grid(model: str) -> List[Dict[str, Any]]:
    a, b = RG_ALPHA_BETA
    out = []
    if model == "vit_b16":
        combos = itertools.product(RG_TAUS, VIT_LAMBDA, VIT_LAMBDA_SM, RG_VARS, RG_EPS)
        for tau, lam, lsm, var, eps in combos:
            out.append({"lrp_alpha": a, "lrp_beta": b, "lrp_eps": eps, "routing_temperature": tau,
                        "adf_lambda": lam, "adf_softmax_lambda": lsm, "adf_input_variance": var,
                        "av_rule": AV_RULES[0]})
    else:
        for tau, lam, var, eps in itertools.product(RG_TAUS, CNN_LAMBDA, RG_VARS, RG_EPS):
            out.append({"lrp_alpha": a, "lrp_beta": b, "lrp_eps": eps, "routing_temperature": tau,
                        "adf_lambda": lam, "adf_input_variance": var})
    return out


def rg_anchor_params(model: str) -> Dict[str, Any]:
    p = dict(RG_ANCHOR, adf_lambda=0.0)
    if model == "vit_b16":
        p["adf_softmax_lambda"] = 0.0
        p["av_rule"] = AV_RULES[0]
    return p


def _rgeq_grid() -> List[Dict[str, Any]]:
    a, b = RG_ALPHA_BETA
    return [dict(RGEQ_WRAPPER, lrp_alpha=a, lrp_beta=b, lrp_eps=RGEQ_EPS, routing_temperature=tau,
                 adf_lambda=0.0, adf_softmax_lambda=lsm, adf_input_variance=RGEQ_VAR, av_rule=rule)
            for rule in AV_RULES for tau in RG_TAUS for lsm in RGEQ_LAMBDA_SM]


def build_rows(model: str) -> List[GridRow]:
    rows: List[GridRow] = []
    for method in MODEL_METHODS[model]:
        get_spec(method)
        if method == "rg":
            configs = _rg_grid(model)
            anchor = rg_anchor_params(model)
        elif method == "rgeq":
            configs = _rgeq_grid()
            anchor = dict(RGEQ_WRAPPER, **rg_anchor_params(model))
        else:
            configs = _baseline_grid(method, model)
            anchor = None

        # Canonicalise and deduplicate (keeps first occurrence order).
        seen: Dict[str, Dict[str, Any]] = {}
        for p in configs:
            cp = canonical_params(method, p)
            seen.setdefault(row_label(method, cp), cp)
        method_rows = [GridRow(method, lbl, cp) for lbl, cp in seen.items()]

        if anchor is not None:
            anchor_cp = canonical_params(method, anchor)
            anchor_label = row_label(method, anchor_cp)
            if anchor_label not in seen:
                raise AssertionError(f"{method}: anchor {anchor_label} is not in the declared grid")
            # Anchor first: it is also the preferred tie-break row.
            method_rows = ([GridRow(method, anchor_label, anchor_cp, is_anchor=True)]
                           + [r for r in method_rows if r.row_label != anchor_label])

        tunable = len(method_rows) > 1 and method not in ("lime", "kernelshap")
        for r in method_rows:
            r.tunable = tunable
        rows.extend(method_rows)

    labels = [r.row_label for r in rows]
    dup = [l for l, c in Counter(labels).items() if c > 1]
    if dup:
        raise AssertionError(f"duplicate row labels: {dup[:5]}")
    for i, r in enumerate(rows):
        r.order = i
    return rows


def default_grid_path(model: str) -> Path:
    return GRID_DIR / f"imagenet_s_val_{model}.csv"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", type=Path, default=None,
                    help="Output CSV (default: experiments/grids/imagenet_s_val_<model>.csv)")
    args = ap.parse_args(argv)
    check_model(args.model)
    out = args.out or default_grid_path(args.model)
    rows = build_rows(args.model)
    write_grid(out, rows)
    counts = Counter(r.method for r in rows)
    print(f"Wrote {len(rows)} rows for {args.model} -> {out}")
    for m in MODEL_METHODS[args.model]:
        n_anchor = sum(1 for r in rows if r.method == m and r.is_anchor)
        tun = next(r.tunable for r in rows if r.method == m)
        print(f"  {m:12s} {counts[m]:5d} rows  tunable={int(tun)}  anchors={n_anchor}")


if __name__ == "__main__":
    sys.exit(main())
