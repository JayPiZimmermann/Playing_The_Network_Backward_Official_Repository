#!/usr/bin/env python3
"""Panels of the dense qualitative figure of the appendix (ViT-B/16, ImageNet-S).

Usage::

    python -m reproduce.qualitative_figure \\
        --selection results/selection/vit_b16/selected.json \\
        --out ../paper/figures/qualitative/butterfly --device cuda|cpu \\
        [--examples red_fox,kuvasz] [--methods rg,gradient] [--identify]

For every example of ``EXAMPLES`` (output folder -> ILSVRC2012 validation
image of the ImageNet-S50 pool) the script writes to ``<out>/<example>/``

* ``original.png``: the evaluated 224x224 centre crop
  (``ImageNetS.denormalize``, 8 bit);
* ``mask.png``: the ImageNet-S object mask of that crop (red on black), reduced
  over the colour channels as in ``quantus_bridge.evaluate_saliency_maps``;
* one panel per column of ``COLUMNS`` (``rg_loc.png``, ``gradient.png``, ...).

Configurations: each column uses the method's localisation-selected row of
``selected.json`` (``methods.<m>.loc``) or, for a method without tunable
parameters, its single fixed row (``methods.<m>.fixed``).  No parameter is set
in this script.

Maps are computed as in the evaluation (``reproduce.common.RowRunner``): the
model is loaded with ``load_model`` and prepared by ``Evaluator.__init__``,
the params are checked with ``validate_params``, the seed is reset with
``seed_everything(SEED)`` and the map is ``Evaluator._generate_maps`` (batch
size 1, target = ImageNet-S label, the ``MethodSpec`` component and layers,
the equivariant wrapper for RG+Eq).  Maps below the input resolution (TIBAV:
14x14 patches) are upsampled with nearest neighbour, as the evaluator does
before the metrics.  The seed is reset before every (example, column) pair,
so a panel does not depend on ``--examples`` / ``--methods``.  For the
stochastic methods (SmoothGrad noise, DAVE and RG+Eq transforms) the panels
are therefore deterministic but not bit-identical to the maps of the test
evaluation, where the random state depends on the position of the image in
the split.

Rendering: black background, positive attribution in red and negative
attribution in blue; each sign is divided by its ``PERCENTILE`` = 99th percentile over the image,
clipped to [0, 1] and truncated to 8 bit, so the top 1% of pixels saturate
and concentrated maps stay legible in print.  The same transform is applied
to every column.

``--identify`` ranks, for every example, the ImageNet-S50 images of its class
by pixel MSE between their 224x224 crop and the existing
``<out>/<example>/original.png`` (this is how ``EXAMPLES`` was obtained from
the earlier panels) and exits without writing anything.
"""
from __future__ import annotations

import argparse
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from reproduce.common import (
    MODELS, REPO, SEED, get_spec, imagenet_s_split, load_json, row_label, seed_everything,
    setup_logging, validate_params,
)

logger = logging.getLogger("reproduce.qualitative_figure")

MODEL = "vit_b16"
PERCENTILE = 99.0    # each sign is divided by this percentile of its magnitude (top 1% saturate)

# Output folder -> ILSVRC2012 validation image.  Identified with --identify
# against the earlier panels (identical 8-bit crop, identical mask).  The
# first image belongs to the 565-image test split, the other five to the
# 50-image validation split used for the configuration selection.
EXAMPLES: Dict[str, str] = {
    "red_fox_snow": "ILSVRC2012_val_00025140.JPEG",
    "red_fox": "ILSVRC2012_val_00008113.JPEG",
    "kuvasz": "ILSVRC2012_val_00024981.JPEG",
    "wild_boar": "ILSVRC2012_val_00045758.JPEG",
    "american_black_bear": "ILSVRC2012_val_00012507.JPEG",
    "airliner": "ILSVRC2012_val_00008292.JPEG",
}

# (panel file stem, method key of reproduce.common.METHOD_SPECS), in figure order.
COLUMNS = (
    ("rg_loc", "rg"),
    ("rgeq_loc", "rgeq"),
    ("gradient", "gradient"),
    ("smooth_grad", "smoothgrad"),
    ("integrated_gradients", "intgrad"),
    ("dave", "dave"),
    ("lrp_gamma", "lrp_gamma"),
    ("attn_lrp", "attnlrp"),
    ("tibav", "tibav"),
)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------
def selected_row(selection: Dict[str, Any], method: str) -> Dict[str, Any]:
    """The method's ``loc`` row of ``selected.json``, else its single ``fixed`` row."""
    entry = selection.get("methods", {}).get(method)
    if entry is None:
        raise KeyError(f"{method!r} is not in the selection")
    if entry.get("loc"):
        row, role = entry["loc"], "loc"
    else:
        fixed = entry.get("fixed") or []
        if len(fixed) != 1:
            raise ValueError(f"{method!r} has no loc row and {len(fixed)} fixed rows")
        row, role = fixed[0], "fixed"
    label = row["row_label"]
    if row_label(method, row["params"]) != label:
        logger.warning("%s: row label %s differs from the canonical label %s of its params",
                       method, label, row_label(method, row["params"]))
    return {"method": method, "row_label": label, "params": dict(row["params"]), "role": role}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def to_input_resolution(a: np.ndarray, hw: Sequence[int]) -> np.ndarray:
    """Nearest-neighbour upsampling of a (h, w) map, as in ``evaluate_saliency_maps``."""
    a = np.asarray(a)
    if tuple(a.shape) == tuple(hw):
        return a
    import cv2
    return cv2.resize(a.astype(np.float32), (int(hw[1]), int(hw[0])), interpolation=cv2.INTER_NEAREST)


def render_signed(a: np.ndarray, percentile: float = PERCENTILE) -> np.ndarray:
    """(H, W) map -> (H, W, 3) uint8: red = positive, blue = negative part.

    Each sign is divided by the ``percentile``-th percentile of its magnitude
    over the image (its maximum if that percentile is zero), clipped to [0, 1]
    and truncated to 8 bit; the green channel is zero.  The same display
    transform is used for every method, so a few extreme pixels cannot darken
    a concentrated map.
    """
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    rgb = np.zeros(a.shape + (3,), dtype=np.uint8)
    for channel, part in ((0, np.clip(a, 0.0, None)), (2, np.clip(-a, 0.0, None))):
        peak = part.max()
        if peak > 0:
            scale = float(np.percentile(part, percentile)) or peak
            rgb[..., channel] = (np.clip(part / scale, 0.0, 1.0) * 255).astype(np.uint8)
    return rgb


def render_mask(mask: np.ndarray) -> np.ndarray:
    """Binary (H, W) or (H, W, 3) mask -> red on black (H, W, 3) uint8."""
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m.any(axis=-1)      # as evaluate_saliency_maps does for RGB masks
    rgb = np.zeros(m.shape + (3,), dtype=np.uint8)
    rgb[..., 0] = np.where(m, 255, 0)
    return rgb


def render_original(pool, image) -> np.ndarray:
    """Normalised (3, H, W) tensor -> (H, W, 3) uint8 via ``ImageNetS.denormalize``."""
    import torch
    return (pool.denormalize(image) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()


def map_summary(a: np.ndarray) -> str:
    """Short description of a map for the log (sign balance, degenerate maps)."""
    a = np.asarray(a, dtype=np.float64)
    pos, neg = a.max(initial=0.0), -a.min(initial=0.0)
    notes = []
    if not np.any(a):
        notes.append("ALL-ZERO")
    elif np.ptp(a) == 0:
        notes.append("CONSTANT")
    if pos > 0 and 0 < neg < 1e-3 * pos:
        notes.append("negative part < 1e-3 of the positive maximum (shown at full blue)")
    if neg > 0 and 0 < pos < 1e-3 * neg:
        notes.append("positive part < 1e-3 of the negative maximum (shown at full red)")
    return (f"min={a.min():.3g} max={a.max():.3g} pos={np.mean(a > 0):.2f} neg={np.mean(a < 0):.2f}"
            + (" [" + "; ".join(notes) + "]" if notes else ""))


def _save_png(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


# ---------------------------------------------------------------------------
# Identification of the example images
# ---------------------------------------------------------------------------
def identify(pool, split_of: Dict[int, str], out: Path, examples: Sequence[str]) -> None:
    """Log the three ImageNet-S50 images of each example's class closest to original.png."""
    from PIL import Image
    index = {s["image_name"]: i for i, s in enumerate(pool.samples)}
    for ex in examples:
        ref_path = out / ex / "original.png"
        if not ref_path.exists():
            logger.warning("%s: %s does not exist", ex, ref_path)
            continue
        ref = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.float64)
        wnid = pool.samples[index[EXAMPLES[ex]]]["wnid"]
        scores = sorted(
            (float(np.mean((render_original(pool, pool[i][0]).astype(np.float64) - ref) ** 2)), i)
            for i, s in enumerate(pool.samples) if s["wnid"] == wnid)
        for mse, i in scores[:3]:
            logger.info("%s: MSE %.3f  %s (%s split)", ex, mse, pool.samples[i]["image_name"],
                        split_of.get(i, "no"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_list(value: Optional[str], allowed: Sequence[str], what: str) -> List[str]:
    if not value:
        return list(allowed)
    items = [v.strip() for v in value.split(",") if v.strip()]
    unknown = [v for v in items if v not in allowed]
    if unknown:
        raise SystemExit(f"unknown {what} {unknown}; choose from {list(allowed)}")
    return [v for v in allowed if v in items]      # figure order


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Panels of the dense qualitative figure (ViT-B/16).")
    ap.add_argument("--selection", type=Path, default=None, help="selected.json of vit_b16")
    ap.add_argument("--out", type=Path, required=True, help="figure folder (one sub-folder per example)")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available)")
    ap.add_argument("--examples", default=None, help=f"comma-separated subset of {list(EXAMPLES)}")
    ap.add_argument("--methods", default=None,
                    help=f"comma-separated subset of the method keys {[m for _, m in COLUMNS]}")
    ap.add_argument("--map-cache", type=Path, default=None,
                    help="directory for the raw maps (<example>/<column>.npy); cached maps are re-rendered")
    ap.add_argument("--identify", action="store_true",
                    help="rank the candidate images against the existing original.png and exit")
    ap.add_argument("--imagenet-root", default=os.environ.get("IMAGENET_ROOT", str(REPO / "data" / "imagenet")))
    ap.add_argument("--imagenet-s-root",
                    default=os.environ.get("IMAGENET_S_ROOT", str(REPO / "data" / "imagenet-s")))
    args = ap.parse_args(argv)
    if args.selection is None and not args.identify:
        ap.error("--selection is required (except with --identify)")
    setup_logging()
    examples = _parse_list(args.examples, list(EXAMPLES), "examples")
    methods = _parse_list(args.methods, [m for _, m in COLUMNS], "methods")

    # The ImageNet-S50 pool and splits exactly as in the evaluation.
    test = imagenet_s_split("test", args.imagenet_root, args.imagenet_s_root)
    val = imagenet_s_split("val", args.imagenet_root, args.imagenet_s_root)
    pool = test.dataset
    split_of = {**{i: "validation" for i in val.indices}, **{i: "test" for i in test.indices}}
    if args.identify:
        identify(pool, split_of, args.out, examples)
        return
    index = {s["image_name"]: i for i, s in enumerate(pool.samples)}
    missing = [EXAMPLES[e] for e in examples if EXAMPLES[e] not in index]
    if missing:
        raise SystemExit(f"images not in the ImageNet-S50 pool: {missing}")

    selection = load_json(args.selection)
    if selection.get("model") not in (None, MODEL):
        raise SystemExit(f"{args.selection} is for model {selection.get('model')}, not {MODEL}")
    rows = {m: selected_row(selection, m) for m in methods}
    for m in methods:
        logger.info("%-10s %-5s %s %s", m, rows[m]["role"], rows[m]["row_label"], rows[m]["params"])

    import torch
    from torch.utils.data import Subset
    from bwgame.evaluation.evaluator import Evaluator
    from bwgame.evaluation.run_eval import load_model

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    positions = [index[EXAMPLES[e]] for e in examples]
    model = load_model(MODELS[MODEL], device)
    t_start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="qualitative_") as work:
        # Evaluator.__init__ prepares the model as in the evaluation
        # (decomposed attention, functional ops as modules); nothing is written.
        ev = Evaluator(model=model, dataset=Subset(pool, positions), device=device,
                       output_dir=Path(work), metric_configs=[], batch_size=1, softmax=True,
                       seed=SEED, save_heatmaps=False, cache_maps=False, warmup=0)
        for k, ex in enumerate(examples):
            image, mask, label, class_name, name = pool[positions[k]]
            with torch.no_grad():
                pred = int(ev.model(image.unsqueeze(0).to(device)).argmax(dim=1))
            logger.info("%s: %s (%s split), label %d (%s), top-1 prediction %d%s", ex, name,
                        split_of.get(positions[k], "no"), label, class_name, pred,
                        "" if pred == label else "  [MISCLASSIFIED; maps explain the label]")
            folder = args.out / ex
            _save_png(folder / "original.png", render_original(pool, image))
            _save_png(folder / "mask.png", render_mask(mask))
            for stem, m in COLUMNS:
                if m not in methods:
                    continue
                cache = args.map_cache / ex / f"{stem}.npy" if args.map_cache else None
                if cache is not None and cache.exists():
                    a = np.load(cache)
                    _save_png(folder / f"{stem}.png", render_signed(a))
                    logger.info("%s / %-20s cached  %s", ex, stem, map_summary(a))
                    continue
                spec, params = get_spec(m), rows[m]["params"]
                validate_params(ev.model, device, spec, params)
                seed_everything(SEED)
                ev._indices = [k]          # _generate_maps runs over ev._indices only
                t0 = time.perf_counter()
                named, images, _, _, _, _, ids, n_failed = ev._generate_maps(
                    spec.registry, dict(params), spec.component,
                    list(spec.layers) if spec.layers else None)
                seconds = time.perf_counter() - t0
                if n_failed or spec.component not in named or list(ids) != [name]:
                    raise RuntimeError(f"{ex} / {rows[m]['row_label']}: map generation failed")
                a = to_input_resolution(named[spec.component][0], images.shape[-2:])
                if not np.isfinite(a).all():
                    raise RuntimeError(f"{ex} / {rows[m]['row_label']}: non-finite map")
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache, a)
                _save_png(folder / f"{stem}.png", render_signed(a))
                logger.info("%s / %-20s %6.1fs  %s", ex, stem, seconds, map_summary(a))
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    logger.info("Done in %.1f min: %d examples x %d methods -> %s",
                (time.perf_counter() - t_start) / 60, len(examples), len(methods), args.out)


if __name__ == "__main__":
    main()
