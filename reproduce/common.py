"""Shared definitions for the ``reproduce/*`` pipeline.

This module is importable without torch: everything that needs torch
(model loading, evaluation) is imported lazily inside functions.

Pipeline (see README.md):

    build_grids -> select_configs (shards, then --aggregate)
                -> evaluate_selected (shards) -> sweep_rows -> evaluate_selected --extra-rows
                -> bootstrap_ci -> runtime_benchmark -> pascal_voc -> make_tables

Terminology
-----------
method key   short name of a method family in the grid CSV (``rg``, ``smoothgrad`` ...);
             maps to a registered attribution method (``MethodSpec.registry``).
row label    deterministic, filesystem-safe identifier of one (method, params) row,
             computed from the canonical params by :func:`row_label`.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent
SEED = 42
METRIC_CFG_DIR = REPO / "bwgame" / "evaluation" / "configs" / "metrics_main_table"
GRID_DIR = REPO / "experiments" / "grids"

# CLI model name -> name understood by bwgame.evaluation.run_eval.load_model
MODELS: Dict[str, str] = {"vit_b16": "vit_b_16", "vgg16": "vgg16", "resnet50": "resnet50"}
MODEL_DISPLAY: Dict[str, str] = {"vit_b16": "ViT-B/16", "vgg16": "VGG-16", "resnet50": "ResNet-50"}

# Expected split sizes of the paper's data (ImageNet-S50 with seed 42).
EXPECTED_SIZES = {("imagenet-s", "val"): 50, ("imagenet-s", "test"): 565}

GRID_COLUMNS = ("method", "row_label", "params_json", "is_anchor", "tunable")


# ---------------------------------------------------------------------------
# Method specifications
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MethodSpec:
    key: str                       # grid 'method' column
    registry: str                  # MethodRegistry name
    display: str                   # LaTeX name in tables
    plain: str                     # plain-text name (CSV / markdown)
    block: str                     # table block: gradient|lrp|attention|cam|perturbation|ours
    component: str = "input"       # evaluated map key
    layers: Optional[Tuple[str, ...]] = None
    ours: bool = False
    # Reference config: "non-default" parameters are counted against it
    # (selection tie-break).  For RG / RG+Eq this is the alpha-beta anchor;
    # for baselines the attribute defaults of the method class.
    defaults: Dict[str, Any] = field(default_factory=dict)
    # Parameters shared by every row of the method; omitted from row labels
    # while they keep this value.
    base: Dict[str, Any] = field(default_factory=dict)


_RG_BASE = {"lrp_alpha": 2.0, "lrp_beta": 1.0}
_RG_ANCHOR = {"lrp_alpha": 2.0, "lrp_beta": 1.0, "lrp_eps": 0.5, "routing_temperature": 1.0,
              "adf_lambda": 0.0, "adf_softmax_lambda": 0.0}
# RG+Eq wrapper settings (as in the paper's RG+Eq rows).
RGEQ_WRAPPER = {"equivariant_averaging": True, "equivariant_n_samples": 50,
                "equivariant_max_angle": 10.0, "equivariant_max_shift_frac": 0.05,
                "equivariant_flip_prob": 0.5}

METHOD_SPECS: Dict[str, MethodSpec] = {s.key: s for s in [
    MethodSpec("gradient", "raw_gradient", "Gradient", "Gradient", "gradient",
               defaults={"channel_reduction": "sum"}),
    MethodSpec("smoothgrad", "smooth_grad", "SmGrad", "SmGrad", "gradient",
               defaults={"nt_samples": 20, "stdevs": 0.1, "use_std_aggregation": False},
               base={"nt_samples": 50, "use_abs": False, "use_std_aggregation": False,
                     "channel_reduction": "max_abs"}),
    MethodSpec("intgrad", "integrated_gradients", "IntGrad", "IntGrad", "gradient",
               defaults={"n_steps": 30, "baseline": "zero"}, base={"baseline": "black"}),
    MethodSpec("deeplift", "deeplift", "DeepLift", "DeepLift", "gradient",
               defaults={"rule": "epsilon", "epsilon": 0.25, "gamma": 0.25}),
    MethodSpec("lrp_eps", "lrp_standard", r"LRP-$\varepsilon$", "LRP-eps", "lrp",
               defaults={"epsilon": 0.25}),
    MethodSpec("lrp_gamma", "gamma_lrp", r"LRP-$\gamma$", "LRP-gamma", "lrp",
               defaults={"epsilon": 0.25, "gamma": 0.25}),
    MethodSpec("attnlrp", "attn_lrp", "AttnLRP", "AttnLRP", "lrp",
               defaults={"gamma_composite": False, "epsilon": 0.25, "lrp_gamma_conv": 0.25,
                         "lrp_gamma_linear": 0.05, "lrp_epsilon_attn": 1e-6}),
    MethodSpec("dave", "dave", "DAVE", "DAVE", "attention",
               defaults={"n_samples": 50, "max_angle": 20.0, "max_shift_frac": 0.1,
                         "flip_prob": 0.5, "noise_t_max": 0.5, "multiply_input": True,
                         "batch_size": 1},
               base={"flip_prob": 0.5, "multiply_input": True, "batch_size": 1}),
    MethodSpec("tibav", "tibav", "TIBAV", "TIBAV", "attention", layers=("input",)),
    MethodSpec("gae", "grad_attention_rollout", "GAE", "GAE", "attention", layers=("input",)),
    MethodSpec("ar", "attention_rollout", "AR", "AR", "attention", layers=("input",),
               defaults={"residual_weight": 0.5}),
    MethodSpec("gcam", "gradcam", "GCAM", "Grad-CAM", "cam"),
    MethodSpec("gcampp", "gradcam_pp", "GCAM++", "GCAM++", "cam"),
    MethodSpec("lcam", "layercam", "LCAM", "LCAM", "cam"),
    MethodSpec("lime", "lime", "LIME", "LIME", "perturbation",
               defaults={"n_samples": 500, "superpixel_method": "grid", "superpixel_n_segments": 80,
                         "surrogate": "lasso", "kernel_width": None}),
    MethodSpec("kernelshap", "kernel_shap", "KernelSHAP", "KernelSHAP", "perturbation",
               defaults={"n_samples": 500, "superpixel_method": "grid", "superpixel_n_segments": 80}),
    MethodSpec("sg", "adf_game", "SG", "SG", "ours", component="pp", ours=True,
               defaults={"adf_lambda": 0.0, "adf_input_variance": 1.0, "adf_probabilistic": True}),
    MethodSpec("rg", "routing_game", "RG", "RG", "ours", ours=True,
               defaults=dict(_RG_ANCHOR), base=dict(_RG_BASE)),
    MethodSpec("rgeq", "routing_game", "RG+Eq", "RG+Eq", "ours", ours=True,
               defaults={**_RG_ANCHOR, **RGEQ_WRAPPER}, base={**_RG_BASE, **RGEQ_WRAPPER}),
]}

# Methods per model, in table order.
MODEL_METHODS: Dict[str, Tuple[str, ...]] = {
    "vit_b16": ("gradient", "smoothgrad", "intgrad",
                "lrp_eps", "lrp_gamma", "attnlrp",
                "dave", "tibav", "gae", "ar",
                "lime", "kernelshap", "rg", "rgeq"),
    "vgg16": ("gradient", "smoothgrad", "intgrad", "deeplift",
              "lrp_eps", "lrp_gamma", "gcam", "gcampp", "lcam",
              "lime", "kernelshap", "sg", "rg"),
    "resnet50": ("gradient", "smoothgrad", "intgrad", "deeplift",
                 "lrp_eps", "lrp_gamma", "gcam", "gcampp", "lcam", "lime", "kernelshap", "rg"),
}

OURS = tuple(k for k, s in METHOD_SPECS.items() if s.ours)


def get_spec(method: str) -> MethodSpec:
    try:
        return METHOD_SPECS[method]
    except KeyError:
        raise KeyError(f"Unknown method key {method!r}; known: {sorted(METHOD_SPECS)}") from None


def check_model(model: str) -> str:
    if model not in MODELS:
        raise SystemExit(f"--model must be one of {sorted(MODELS)}, got {model!r}")
    return model


# ---------------------------------------------------------------------------
# Canonical params and row labels
# ---------------------------------------------------------------------------
def canonical_params(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Return the canonical form of a param dict (drops functionally inert keys).

    RG / RG+Eq: the input variance sigma^2 only enters through the ADF
    forward pass, which runs only if a risk parameter is non-zero
    (``RoutingGame.attribute``).  With lambda = lambda_sm = 0 the variance
    is therefore dropped, so that functionally identical configurations get
    the same row label and are evaluated once.
    Likewise, the softmax risk lambda_sm acts on the Value-path routing of
    ``A·V`` only and is inert under the gradient-fallback attention rule
    (``attention_mode='gradient_fallback'``), so it is set to zero there.
    """
    p = dict(params)
    if method in ("rg", "rgeq"):
        if p.get("attention_mode", "v_only") == "gradient_fallback" and "adf_softmax_lambda" in p:
            p["adf_softmax_lambda"] = 0.0
        risk = [float(p.get(k, 0.0)) for k in ("adf_lambda", "adf_softmax_lambda")]
        if all(r == 0.0 for r in risk):
            p.pop("adf_input_variance", None)
    return p


_ALIASES = {
    "routing_temperature": "tau", "lrp_eps": "eps", "lrp_alpha": "alpha", "lrp_beta": "beta",
    "adf_lambda": "lam", "adf_softmax_lambda": "lamsm", "adf_input_variance": "var",
    "adf_probabilistic": "gate", "nt_samples": "N", "stdevs": "sigma",
    "use_std_aggregation": "std", "n_steps": "steps", "epsilon": "eps", "gamma": "gamma",
    "rule": "rule", "gamma_composite": "gcomp", "lrp_gamma_conv": "gconv",
    "lrp_gamma_linear": "glin", "lrp_epsilon_attn": "eattn", "n_samples": "N",
    "max_angle": "ang", "max_shift_frac": "shift", "noise_t_max": "noise",
    "flip_prob": "flip", "multiply_input": "mulx", "residual_weight": "wr",
    "superpixel_n_segments": "seg", "superpixel_method": "sp", "batch_size": "bs",
    "surrogate": "sur", "kernel_width": "kw", "composite": "comp",
    "channel_reduction": "cr", "stdev_spread": "spread", "use_abs": "useabs", "baseline": "bl",
    "kernel_distance": "kd", "fill": "fill", "target_layer_name": "layer",
    "equivariant_n_samples": "eqN", "equivariant_max_angle": "eqang",
    "equivariant_max_shift_frac": "eqshift", "equivariant_flip_prob": "eqflip",
    "equivariant_averaging": "eq", "av_rule": "av", "attention_mode": "attn",
}


def _fmt_label_value(v: Any) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, (int, float)):
        s = format(float(v), "g") if isinstance(v, float) else str(v)
        return s.replace("-", "m").replace("+", "")
    return re.sub(r"[^A-Za-z0-9.]+", "", str(v))


def row_label(method: str, params: Dict[str, Any]) -> str:
    """Deterministic, filesystem-safe label of a (method, params) row.

    Parameters equal to the method's shared ``base`` value are omitted; the
    rest are listed as ``alias-value`` in alphabetical order of the alias.
    Characters are restricted to ``[A-Za-z0-9._-]`` (negative numbers are
    written with an ``m`` prefix, e.g. ``lamsm-m10``).
    """
    spec = get_spec(method)
    p = canonical_params(method, params)
    parts = []
    for k, v in p.items():
        if k in spec.base and spec.base[k] == v:
            continue
        alias = _ALIASES.get(k, re.sub(r"[^A-Za-z0-9]+", "", k))
        parts.append((alias, _fmt_label_value(v)))
    parts.sort()
    label = method if not parts else method + "__" + "__".join(f"{a}-{v}" for a, v in parts)
    assert re.fullmatch(r"[A-Za-z0-9._-]+", label), label
    return label


def n_nondefault(method: str, params: Dict[str, Any]) -> int:
    """Number of parameters that differ from the method's reference config."""
    spec = get_spec(method)
    p = canonical_params(method, params)
    n = 0
    for k, v in p.items():
        if k in spec.defaults and _same(spec.defaults[k], v):
            continue
        if k in spec.base and _same(spec.base[k], v):
            continue
        n += 1
    # A reference parameter that is absent counts as default.
    return n


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) \
            and not isinstance(b, bool):
        return float(a) == float(b)
    return a == b


def params_json(params: Dict[str, Any]) -> str:
    return json.dumps(params, sort_keys=True)


# ---------------------------------------------------------------------------
# Grid CSV I/O
# ---------------------------------------------------------------------------
@dataclass
class GridRow:
    method: str
    row_label: str
    params: Dict[str, Any]
    is_anchor: bool = False
    tunable: bool = True
    order: int = 0

    @property
    def spec(self) -> MethodSpec:
        return get_spec(self.method)


def read_grid(path: Path) -> List[GridRow]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Grid CSV not found: {path} (run python -m reproduce.build_grids first)")
    rows: List[GridRow] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = set(GRID_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} lacks columns {sorted(missing)} (expected {GRID_COLUMNS})")
        for i, r in enumerate(reader):
            get_spec(r["method"])  # validates the method key
            rows.append(GridRow(
                method=r["method"], row_label=r["row_label"],
                params=json.loads(r["params_json"]),
                is_anchor=bool(int(r["is_anchor"])), tunable=bool(int(r["tunable"])), order=i,
            ))
    labels = [r.row_label for r in rows]
    dup = {l for l in labels if labels.count(l) > 1}
    if dup:
        raise ValueError(f"{path} has duplicate row labels: {sorted(dup)[:5]}")
    return rows


def write_grid(path: Path, rows: Sequence[GridRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(GRID_COLUMNS)
        for r in rows:
            w.writerow([r.method, r.row_label, params_json(r.params), int(r.is_anchor), int(r.tunable)])


def shard(items: Sequence[Any], shard_id: Optional[int], num_shards: Optional[int]) -> List[Any]:
    """Deterministic round-robin shard of ``items`` (item i goes to shard i % n)."""
    if shard_id is None and num_shards is None:
        return list(items)
    if shard_id is None or num_shards is None:
        raise SystemExit("--shard and --num-shards must be given together")
    if not (0 <= shard_id < num_shards):
        raise SystemExit(f"--shard must be in [0, {num_shards}), got {shard_id}")
    return [it for i, it in enumerate(items) if i % num_shards == shard_id]


# ---------------------------------------------------------------------------
# Result files
# ---------------------------------------------------------------------------
def write_json_atomic(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:
        pass
    return str(o)


def load_row_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Load every ``<results_dir>/*.json`` written by the row runner, by row label."""
    out: Dict[str, Dict[str, Any]] = {}
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return out
    for p in sorted(results_dir.glob("*.json")):
        if p.name.endswith(".incomplete.json"):
            continue          # rows that failed the completeness check
        try:
            with open(p) as f:
                obj = json.load(f)
        except Exception as e:
            logger.warning("Skipping unreadable %s: %s", p, e)
            continue
        if isinstance(obj, dict) and obj.get("kind") == "row_result":
            out[obj["row_label"]] = obj
    return out


_PER_IMAGE_CSV: Dict[Path, Dict[str, Dict[str, Any]]] = {}


def _per_image_from_csv(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Per-row per-image arrays from ``<results_dir>/per_image_scores.csv.gz``.

    The compact per-image export shipped in ``results_summary/``: one line per
    row and image, one column per scalar metric (blank = not finite).  Parsed
    once per directory.
    """
    import csv
    import gzip
    import numpy as np

    results_dir = Path(results_dir)
    if results_dir in _PER_IMAGE_CSV:
        return _PER_IMAGE_CSV[results_dir]
    out: Dict[str, Dict[str, Any]] = {}
    p = results_dir / "per_image_scores.csv.gz"
    if p.exists():
        by_label: Dict[str, Dict[str, list]] = {}
        with gzip.open(p, "rt", newline="") as f:
            reader = csv.DictReader(f)
            metrics = [c for c in (reader.fieldnames or []) if c not in ("row_label", "image_id")]
            for r in reader:
                d = by_label.setdefault(r["row_label"], {"image_ids": [], **{m: [] for m in metrics}})
                d["image_ids"].append(r["image_id"])
                for m in metrics:
                    d[m].append(float(r[m]) if r[m] != "" else np.nan)
        for label, d in by_label.items():
            out[label] = {"image_ids": np.array(d["image_ids"], dtype=str),
                          **{m: np.array(d[m], dtype=float) for m in metrics}}
    _PER_IMAGE_CSV[results_dir] = out
    return out


def load_per_image(results_dir: Path, label: str) -> Optional[Dict[str, Any]]:
    """Per-image arrays of a row, or None.

    ``<results_dir>/per_image/<label>.npz`` as written by the evaluator, or the
    row's lines of ``<results_dir>/per_image_scores.csv.gz`` (``results_summary/``).
    """
    import numpy as np
    p = Path(results_dir) / "per_image" / f"{label}.npz"
    if p.exists():
        with np.load(p, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    return _per_image_from_csv(results_dir).get(label)


def load_json(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# LaTeX formatting of configurations
# ---------------------------------------------------------------------------
def _lnum(v: Any) -> str:
    if isinstance(v, bool):
        return r"\mathrm{on}" if v else r"\mathrm{off}"
    s = format(float(v), "g") if isinstance(v, (int, float)) else str(v)
    return "{-}" + s[1:] if s.startswith("-") else s


def config_latex(method: str, params: Dict[str, Any]) -> str:
    """Short LaTeX description of a configuration (e.g. ``$\\tau{=}0.5,...$``)."""
    p = canonical_params(method, params)
    items: List[str] = []
    suffix = ""
    if method in ("rg", "rgeq"):
        items.append(r"\tau{=}" + _lnum(p.get("routing_temperature", 1.0)))
        if float(p.get("adf_lambda", 0.0)) != 0.0:
            items.append(r"\lambda{=}" + _lnum(p["adf_lambda"]))
        if float(p.get("adf_softmax_lambda", 0.0)) != 0.0:
            items.append(r"\lambda_{\mathrm{sm}}{=}" + _lnum(p["adf_softmax_lambda"]))
        if "adf_input_variance" in p:
            items.append(r"\sigma^2{=}" + _lnum(p["adf_input_variance"]))
        if float(p.get("lrp_eps", 0.5)) != 0.5:
            items.append(r"\varepsilon{=}" + _lnum(p["lrp_eps"]))
        if (float(p.get("lrp_alpha", 2.0)), float(p.get("lrp_beta", 1.0))) != (2.0, 1.0):
            items.append(r"(\alpha,\beta){=}(" + _lnum(p.get("lrp_alpha", 1.0)) + ","
                         + _lnum(p.get("lrp_beta", 0.0)) + ")")
        if method == "rgeq":
            items.append(r"N{=}" + _lnum(p.get("equivariant_n_samples", 50)))
        if p.get("attention_mode", "v_only") == "gradient_fallback":
            suffix += ", gradient attention"
    elif method == "sg":
        items.append(r"\lambda{=}" + _lnum(p.get("adf_lambda", 0.0)))
        items.append(r"\sigma^2{=}" + _lnum(p.get("adf_input_variance", 1.0)))
        suffix = ", gate " + ("on" if p.get("adf_probabilistic", True) else "off")
    elif method == "smoothgrad":
        items.append(r"N{=}" + _lnum(p.get("nt_samples", 20)))
        if "stdev_spread" in p:
            items.append(r"\sigma/\mathrm{range}{=}" + _lnum(p["stdev_spread"]))
        else:
            items.append(r"\sigma{=}" + _lnum(p.get("stdevs", 0.1)))
        if p.get("use_std_aggregation", False):
            suffix = ", std"
    elif method == "intgrad":
        items.append(r"N_{\mathrm{steps}}{=}" + _lnum(p.get("n_steps", 30)))
        if p.get("baseline", "zero") == "black":
            suffix = ", black baseline"
    elif method in ("gcam", "gcampp", "lcam"):
        if p.get("target_layer_name"):
            suffix = r"\texttt{" + str(p["target_layer_name"]) + "}"
    elif method == "deeplift":
        rule = p.get("rule", "epsilon")
        if rule == "rescale":
            suffix = "Rescale"
        elif rule == "gamma":
            items.append(r"\gamma{=}" + _lnum(p.get("gamma", 0.25)))
        elif rule == "z_plus":
            items.append(r"z^{+}")
    elif method == "lrp_eps":
        items.append(r"\varepsilon{=}" + _lnum(p.get("epsilon", 0.25)))
    elif method == "lrp_gamma":
        items += [r"\varepsilon{=}" + _lnum(p.get("epsilon", 0.25)), r"\gamma{=}" + _lnum(p.get("gamma", 0.25))]
        if p.get("composite", "all_gamma") == "epsilon_gamma_box":
            suffix = r", $z^{\mathcal{B}}$/$\gamma$/$\varepsilon$ composite"
    elif method == "attnlrp":
        if p.get("gamma_composite", False):
            items += [r"\gamma_{\mathrm{conv}}{=}" + _lnum(p.get("lrp_gamma_conv", 0.25)),
                      r"\gamma_{\mathrm{lin}}{=}" + _lnum(p.get("lrp_gamma_linear", 0.05)),
                      r"\varepsilon_{\mathrm{attn}}{=}" + _lnum(p.get("lrp_epsilon_attn", 0.25))]
        else:
            items.append(r"\varepsilon{=}" + _lnum(p.get("epsilon", 0.25)))
    elif method == "dave":
        items += [r"N{=}" + _lnum(p.get("n_samples", 50)),
                  r"\theta_{\max}{=}" + _lnum(p.get("max_angle", 20.0)),
                  r"s_{\max}{=}" + _lnum(p.get("max_shift_frac", 0.1)),
                  r"t_{\max}{=}" + _lnum(p.get("noise_t_max", 0.5))]
    elif method == "ar":
        items.append(r"w_r{=}" + _lnum(p.get("residual_weight", 0.5)))
    elif method in ("lime", "kernelshap"):
        n = p.get("n_samples", 500)
        items.append(r"N{=}" + (r"2M{+}2048" if n == "auto" else _lnum(n)))
        if p.get("superpixel_method", "grid") != "grid":
            suffix = ", {} superpixels".format(int(p.get("superpixel_n_segments", 80)))
        if p.get("surrogate", "lasso") == "ridge":
            suffix += ", ridge"
    if not items:
        return suffix.lstrip(", ") or "--"
    return "$" + ",".join(items) + "$" + suffix


def config_plain(method: str, params: Dict[str, Any]) -> str:
    """Plain-text configuration (for markdown / logs)."""
    spec = get_spec(method)
    p = canonical_params(method, params)
    shown = {k: v for k, v in p.items() if not (k in spec.base and spec.base[k] == v)}
    if not shown:
        return "default"
    return ", ".join(f"{_ALIASES.get(k, k)}={v}" for k, v in sorted(shown.items()))


# ---------------------------------------------------------------------------
# Dataset / model / evaluation helpers (torch imported lazily)
# ---------------------------------------------------------------------------
def add_data_args(ap) -> None:
    """Dataset-location options shared by the evaluation scripts."""
    ap.add_argument("--imagenet-root", type=str,
                    default=os.environ.get("IMAGENET_ROOT", str(REPO / "data" / "imagenet")),
                    help="ImageNet root containing val/ (default: $IMAGENET_ROOT or data/imagenet).")
    ap.add_argument("--imagenet-s-root", type=str,
                    default=os.environ.get("IMAGENET_S_ROOT", str(REPO / "data" / "imagenet-s")),
                    help="ImageNet-S root (default: $IMAGENET_S_ROOT or data/imagenet-s).")
    ap.add_argument("--voc-root", type=str,
                    default=os.environ.get("VOC_ROOT", str(REPO / "data" / "voc" / "VOCdevkit" / "VOC2012")),
                    help="Pascal VOC 2012 root (default: $VOC_ROOT or data/voc/VOCdevkit/VOC2012).")
    ap.add_argument("--device", type=str, default=None, help="torch device (default: cuda if available)")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Evaluate only the first N images of the split (smoke tests only).")
    ap.add_argument("--allow-size-mismatch", action="store_true",
                    help="Do not abort when the ImageNet-S split size differs from the paper (50 / 565).")
    ap.add_argument("--cache-maps", action="store_true",
                    help="Keep the evaluator's saliency cache (.saliency_cache/*.npz, large).")
    ap.add_argument("--save-heatmaps", action="store_true",
                    help="Write one heatmap PNG per image (off by default).")
    ap.add_argument("--warmup", type=int, default=1,
                    help="Untimed warm-up calls per row before per-image timing (default 1).")


def display_path(path) -> str:
    """Path as recorded in result files: relative to the code root when inside it,
    otherwise only the file name (no machine-specific absolute paths)."""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return p.name


def imagenet_s_split(split: str, imagenet_root: Optional[str] = None,
                     imagenet_s_root: Optional[str] = None, *, seed: int = SEED,
                     n_per_class_val: int = 1, image_size: int = 224,
                     imagenet_s_variant: str = "ImageNetS50") -> Any:
    """ImageNet-S evaluation split shared by every experiment (see ``build_split``).

    The paper protocol fixes seed 42, one validation image per class, 224 px and
    ImageNet-S50; other values are rejected so that no experiment silently uses
    a different split.
    """
    import types
    if (seed, n_per_class_val, image_size, imagenet_s_variant) != (SEED, 1, 224, "ImageNetS50"):
        raise ValueError("The ImageNet-S split is fixed to seed 42, 1 validation image per class, "
                         "224 px, ImageNetS50.")
    args = types.SimpleNamespace(
        imagenet_root=imagenet_root or os.environ.get("IMAGENET_ROOT", str(REPO / "data" / "imagenet")),
        imagenet_s_root=imagenet_s_root or os.environ.get("IMAGENET_S_ROOT",
                                                          str(REPO / "data" / "imagenet-s")),
        voc_root=None)
    return build_split("imagenet-s", split, args)


def build_split(dataset: str, split: str, args) -> Any:
    """Return the evaluation Subset for (dataset, split) with the paper's seeds."""
    from torch.utils.data import Subset
    from bwgame.evaluation.dataset import build_val_split, build_test_split, split_voc_trainval

    if dataset == "imagenet-s":
        from data.imagenet_s import ImageNetS
        full = ImageNetS(imagenet_root=args.imagenet_root, imagenet_s_root=args.imagenet_s_root,
                         split="ImageNetS50", image_size=224)
        if split == "val":
            ds = build_val_split(full, n_per_class=1, seed=SEED)
        elif split == "test":
            ds = build_test_split(full, n_per_class_val=1, seed=SEED)
        else:
            raise ValueError(f"split must be val or test, got {split!r}")
    elif dataset == "voc":
        from data.pascal_voc import PascalVOC2012
        full = PascalVOC2012(voc_root=args.voc_root, image_set="trainval", image_size=224)
        val_ds, test_ds = split_voc_trainval(full, val_fraction=1.0 / 6.0, seed=SEED)
        ds = val_ds if split == "val" else test_ds
    else:
        raise ValueError(f"dataset must be imagenet-s or voc, got {dataset!r}")

    # Evaluability: the annotated object must be visible in the evaluated
    # 224x224 centre crop, otherwise the localisation metrics are undefined
    # for every method (Quantus even scores such an image as a pointing-game
    # hit).  Applied after the seeded split, so the split itself is unchanged.
    visible = [i for i in ds.indices if full.mask_visible(i)]
    if len(visible) != len(ds.indices):
        dropped = [full.samples[i].get("image_name", i) for i in ds.indices if i not in set(visible)]
        logger.info("%s / %s: %d image(s) without a visible object in the crop excluded: %s",
                    dataset, split, len(dropped), dropped)
        ds = Subset(full, visible)

    expected = EXPECTED_SIZES.get((dataset, split))
    if expected is not None and len(ds) != expected:
        msg = (f"{dataset} {split} split has {len(ds)} images, the paper uses {expected}. "
               f"Check the data layout (bash data/download_imagenet_s.sh; "
               f"python data/download_imagenet_val.py ...).")
        if getattr(args, "allow_size_mismatch", False) or getattr(args, "max_images", None):
            logger.warning(msg)
        else:
            raise RuntimeError(msg + " Pass --allow-size-mismatch to continue anyway.")
    if getattr(args, "max_images", None):
        ds = Subset(ds, list(range(min(args.max_images, len(ds)))))
    logger.info("%s / %s: %d images", dataset, split, len(ds))
    return ds


def load_metric_configs(short_names: Sequence[str], nr_samples: Optional[int] = None) -> List[Dict]:
    """Metric configs (paper settings) for the given short names (AL, PG, ...)."""
    from bwgame.evaluation.config_loader import load_metric_configs as _load
    from bwgame.evaluation.metric_spec import SHORT_TO_METRIC
    wanted = {SHORT_TO_METRIC[s] for s in short_names}
    cfgs = [c for c in _load(metrics_dir=METRIC_CFG_DIR) if c["name"] in wanted]
    found = {c["name"] for c in cfgs}
    if found != wanted:
        raise RuntimeError(f"Metric configs missing in {METRIC_CFG_DIR}: {sorted(wanted - found)}")
    if nr_samples is not None:
        for c in cfgs:
            if c["metric_class"] in ("max_sensitivity", "avg_sensitivity"):
                c["params"] = dict(c["params"], nr_samples=int(nr_samples))
    return cfgs


def seed_everything(seed: int = SEED) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Full fp32 on every GPU type: cuDNN uses TF32 by default on Ampere and
    # newer, which would make maps depend on the GPU a row happened to run on.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def validate_params(model, device, spec: MethodSpec, params: Dict[str, Any]) -> None:
    """Raise if a parameter would be silently ignored by the evaluator.

    The evaluator only sets parameters that exist as attributes of the
    method instance; anything else would be a silent no-op.
    """
    from bwgame.xai.registry import MethodRegistry
    from bwgame.xai.wrappers import _WRAPPER_KEYS
    method = MethodRegistry.instantiate(spec.registry, model, device)
    try:
        unknown = [k for k in params if k not in _WRAPPER_KEYS and not hasattr(method, k)]
    finally:
        method.cleanup()
    if unknown:
        raise ValueError(
            f"Parameters {unknown} are not attributes of method {spec.registry!r} "
            f"(grid method {spec.key!r}); they would be ignored. Fix the grid.")


def gpu_info() -> Dict[str, str]:
    import torch
    info = {"torch": torch.__version__, "cuda": str(torch.version.cuda),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    try:
        import quantus
        info["quantus"] = getattr(quantus, "__version__", "?")
    except Exception:
        info["quantus"] = "not installed"
    return info


def finite_stats(values) -> Dict[str, Any]:
    import numpy as np
    a = np.asarray(values, dtype=float)
    fin = a[np.isfinite(a)]
    n = int(fin.size)
    return {
        "mean": float(fin.mean()) if n else float("nan"),
        "std": float(fin.std(ddof=1)) if n > 1 else float("nan"),
        "n": n,
        "coverage": n / a.size if a.size else float("nan"),
    }


class RowRunner:
    """Evaluate grid rows with the bwgame Evaluator and write row results.

    Output per row:
      ``<json_dir>/<row_label>.json``            summary (kind = "row_result")
      ``<out_dir>/per_image/<row_label>.npz``     per-image scores (written by the evaluator)
    """

    def __init__(self, *, model_key: str, dataset: str, split: str, out_dir: Path,
                 json_dir: Path, metric_short: Sequence[str], args,
                 nr_samples: Optional[int] = None,
                 subset_indices: Optional[Sequence[int]] = None):
        import torch
        from bwgame.evaluation.evaluator import Evaluator
        from bwgame.evaluation.run_eval import load_model

        self.model_key = model_key
        self.dataset_name = dataset
        self.split = split
        self.out_dir = Path(out_dir)
        self.json_dir = Path(json_dir)
        self.metric_short = list(metric_short)
        self.args = args
        self.device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_model(MODELS[model_key], self.device)
        self.data = build_split(dataset, split, args)
        if subset_indices is not None:
            from torch.utils.data import Subset
            self.data = Subset(self.data, list(subset_indices))
        self.metric_configs = load_metric_configs(self.metric_short, nr_samples=nr_samples)
        self.evaluator = Evaluator(
            model=self.model, dataset=self.data, device=self.device, output_dir=self.out_dir,
            metric_configs=self.metric_configs, batch_size=1, softmax=True, seed=SEED,
            save_heatmaps=args.save_heatmaps, cache_maps=args.cache_maps, warmup=args.warmup,
        )
        # Evaluator.__init__ decomposes the model; reuse the same instance.
        self.model = self.evaluator.model
        self.info = gpu_info()

    def json_path(self, label: str) -> Path:
        return self.json_dir / f"{label}.json"

    def is_complete(self, label: str) -> bool:
        p = self.json_path(label)
        pi = self.out_dir / "per_image" / f"{label}.npz"
        if not (p.exists() and pi.exists()):
            return False
        try:
            obj = load_json(p)
        except Exception:
            return False
        return set(self.metric_short).issubset(obj.get("metrics", {}).keys())

    def run(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate one row: dict with method, row_label, params (+ metadata)."""
        import numpy as np
        import torch
        from bwgame.evaluation.metric_spec import ALL_SHORT, SHORT_TO_METRIC

        spec = get_spec(row["method"])
        params = dict(row["params"])
        label = row["row_label"]
        validate_params(self.model, self.device, spec, params)
        seed_everything(SEED)  # per-row reseed: results do not depend on shard order
        results = self.evaluator.run_method(
            method_name=spec.registry, method_params=params, component=spec.component,
            layers=list(spec.layers) if spec.layers else None, config_label=label,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        main = [r for r in results if r.get("method") == label]
        if not main:
            raise RuntimeError(f"Evaluator returned no result for {label}: {results}")
        per_image = load_per_image(self.out_dir, label)
        if per_image is None:
            raise RuntimeError(f"Per-image file missing for {label} in {self.out_dir / 'per_image'}")

        # Every metric present in the per-image store is summarised (also those
        # computed by an earlier invocation with a different metric set).
        metrics = {}
        for s in ALL_SHORT:
            name = SHORT_TO_METRIC[s]
            if name in per_image:
                metrics[s] = finite_stats(per_image[name])
            elif s in self.metric_short:
                raise RuntimeError(f"Metric {name} missing in per-image file of {label}")
        meta = {}
        if "meta" in per_image:
            try:
                meta = json.loads(str(per_image["meta"]))
            except Exception:
                meta = {}
        t = np.asarray(per_image.get("time_s", []), dtype=float)
        tf = t[np.isfinite(t)]
        runtime = {"n": int(tf.size), "warmup": int(self.args.warmup), "device": self.info["gpu"]}
        if tf.size:
            runtime.update(mean=float(tf.mean()), median=float(np.median(tf)),
                           q25=float(np.percentile(tf, 25)), q75=float(np.percentile(tf, 75)))

        obj = {
            "kind": "row_result",
            "model": self.model_key, "dataset": self.dataset_name, "split": self.split,
            "method": spec.key, "display": spec.display, "registry_name": spec.registry,
            "component": spec.component, "layers": list(spec.layers) if spec.layers else None,
            "row_label": label, "params": params,
            "config_latex": config_latex(spec.key, params),
            "is_anchor": bool(row.get("is_anchor", False)),
            "tunable": bool(row.get("tunable", True)),
            "selected_by": sorted(set(row.get("selected_by", []))),
            "sweeps": row.get("sweeps", []),
            "n_images": int(len(per_image["image_ids"])),
            "metrics": metrics,
            "map_failures": int(np.asarray(per_image.get("map_failed", [])).sum()),
            "zero_maps": int(np.asarray(per_image.get("map_is_zero", [])).sum()),
            "runtime_s": runtime,
            "curve_len": {k: v for k, v in meta.items()},
            "metric_params": {s: c["params"] for s in self.metric_short for c in self.metric_configs
                              if c["name"] == SHORT_TO_METRIC[s]},
            "environment": self.info,
        }
        if "equivariant_averaging" in params and any(s in metrics for s in ("MaxS", "AvgS")):
            # make_explain_func routes wrapper rows through the same wrapper,
            # so MaxS / AvgS explain the perturbed inputs with the row's method.
            obj["robustness_explain"] = "same wrapper as the evaluated map"
        # Merge metadata of an existing result file (e.g. selected_by of a
        # previous invocation) so that repeated runs only add information.
        old = self.json_path(label)
        if old.exists():
            try:
                prev = load_json(old)
                for k, v in prev.get("metrics", {}).items():
                    if "subset" in v and k in obj["metrics"] and k not in self.metric_short:
                        obj["metrics"][k]["subset"] = v["subset"]
                obj["metric_params"] = {**prev.get("metric_params", {}), **obj["metric_params"]}
                obj["selected_by"] = sorted(set(obj["selected_by"]) | set(prev.get("selected_by", [])))
                obj["sweeps"] = merge_sweeps(prev.get("sweeps", []), obj["sweeps"])
            except Exception:
                pass
        write_json_atomic(self.json_path(label), obj)
        return obj


def merge_sweeps(a: Iterable[Dict], b: Iterable[Dict]) -> List[Dict]:
    seen, out = set(), []
    for s in list(a) + list(b):
        key = json.dumps(s, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


def update_row_metadata(json_path: Path, row: Dict[str, Any]) -> None:
    """Add selected_by / sweeps of ``row`` to an existing row result file."""
    obj = load_json(json_path)
    obj["selected_by"] = sorted(set(obj.get("selected_by", [])) | set(row.get("selected_by", [])))
    obj["sweeps"] = merge_sweeps(obj.get("sweeps", []), row.get("sweeps", []))
    obj["is_anchor"] = bool(obj.get("is_anchor", False) or row.get("is_anchor", False))
    write_json_atomic(json_path, obj)


def makedirs(path: Path, retries: int = 5) -> None:
    """``mkdir -p`` that tolerates concurrent creation on network filesystems.

    Two jobs creating the same directory at the same moment can make
    ``mkdir(exist_ok=True)`` raise FileExistsError while the directory is not
    yet visible as a directory; retry briefly instead of failing the job.
    """
    import time
    for k in range(retries):
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            return
        except FileExistsError:
            if Path(path).is_dir():
                return
            time.sleep(0.2 * (k + 1))
    Path(path).mkdir(parents=True, exist_ok=True)


def setup_logging(log_file: Optional[Path] = None) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        makedirs(Path(log_file).parent)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=logging.INFO, handlers=handlers, force=True,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def fmt_float(v: Optional[float], nd: int) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "nan"
    return f"{v:.{nd}f}"
