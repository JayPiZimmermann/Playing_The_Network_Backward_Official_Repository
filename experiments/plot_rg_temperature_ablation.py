#!/usr/bin/env python3
"""Qualitative RG temperature sweep on three fixed ImageNet-S examples.

Run from the playing_backward checkout with its project environment. Raw maps
are cached per image and temperature; the final grid uses the same signed,
per-image 99th percentile display transform as reproduce/qualitative_figure.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = {
    "red_fox_snow": "ILSVRC2012_val_00025140.JPEG",
    "kuvasz": "ILSVRC2012_val_00024981.JPEG",
    "airliner": "ILSVRC2012_val_00008292.JPEG",
}
DEFAULT_TEMPS = (0.0, 0.2, 0.4, 0.7, 1.0)
PARAMS = {"lrp_alpha": 2.0, "lrp_beta": 1.0, "lrp_eps": 0.1,
          "adf_lambda": 0.0, "adf_softmax_lambda": -10.0,
          "adf_input_variance": 5.0, "av_rule": "merged"}


def signed_rgb(a):
    a = np.nan_to_num(np.asarray(a, dtype=np.float64))
    rgb = np.zeros((*a.shape, 3), np.uint8)
    for channel, part in ((0, np.maximum(a, 0)), (2, np.maximum(-a, 0))):
        peak = float(part.max(initial=0))
        if peak:
            p99 = float(np.percentile(part, 99))
            scale = p99 if p99 > 0 else peak
            rgb[..., channel] = (np.clip(part / scale, 0, 1) * 255).astype(np.uint8)
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "tex/figures/rg_temperature")
    ap.add_argument("--map-cache", type=Path, default=ROOT / "reports/evaluation-low-temperature/prepared_evaluator_maps")
    ap.add_argument("--report", type=Path, default=ROOT / "reports/evaluation-low-temperature/qualitative.json")
    code_root = Path(__file__).resolve().parents[1]
    ap.add_argument("--imagenet-root",
                    default=os.environ.get("IMAGENET_ROOT", str(code_root / "data" / "imagenet")))
    ap.add_argument("--imagenet-s-root",
                    default=os.environ.get("IMAGENET_S_ROOT", str(code_root / "data" / "imagenet-s")))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--temperatures", default=','.join(map(str, DEFAULT_TEMPS)))
    ap.add_argument("--render-only", action="store_true", help="render existing caches without loading the model")
    args = ap.parse_args()
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    temps = [float(t) for t in args.temperatures.split(',')]
    args.out.mkdir(parents=True, exist_ok=True)
    cache_dir = args.map_cache
    if args.render_only:
        metadata = json.loads(args.report.read_text())
        # The zero-temperature probe remains in the evidence cache but is
        # omitted from the requested display grid.
        render_cached(args, temps, Image, plt, metadata)
        return
    import sys
    sys.path.insert(0, str(ROOT / "code/playing_backward"))
    from reproduce.common import imagenet_s_split, seed_everything
    from bwgame.evaluation.evaluator import Evaluator
    from bwgame.evaluation.run_eval import load_model
    import torch
    from torch.utils.data import Subset
    test = imagenet_s_split("test", args.imagenet_root, args.imagenet_s_root)
    val = imagenet_s_split("val", args.imagenet_root, args.imagenet_s_root)
    pool = test.dataset
    index = {s["image_name"]: i for i, s in enumerate(pool.samples)}
    ids = [index[EXAMPLES[k]] for k in EXAMPLES]
    missing = [EXAMPLES[k] for k in EXAMPLES if EXAMPLES[k] not in index]
    if missing:
        raise RuntimeError(f"Examples absent from ImageNet-S pool: {missing}")
    device = torch.device(args.device)
    model = load_model("vit_b_16", device)
    backend_diff = subprocess.check_output(
        ["git", "-C", str(ROOT / "code/playing_backward"), "diff", "HEAD", "--binary"])
    split_by_index = {i: "test" for i in test.indices}
    split_by_index.update({i: "validation" for i in val.indices})
    metadata = {"model": "torchvision ViT_B_16_Weights.IMAGENET1K_V1", "examples": {},
                "temperatures": temps, "fixed_params": PARAMS,
                "map_generation": "Evaluator._generate_maps on Evaluator-prepared model",
                "display": "signed 99th-percentile per-image transform; red positive, blue negative",
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "playing_backward_git_commit": subprocess.check_output(
                    ["git", "-C", str(ROOT / "code/playing_backward"), "rev-parse", "HEAD"], text=True).strip(),
                "playing_backward_worktree_diff_sha256": hashlib.sha256(backend_diff).hexdigest()}
    with tempfile.TemporaryDirectory(prefix="rg_temperature_") as work:
        ev = Evaluator(model=model, dataset=Subset(pool, ids), device=device,
                       output_dir=Path(work), metric_configs=[], batch_size=1, softmax=True,
                       seed=42, save_heatmaps=False, cache_maps=False, warmup=0)
        for row, (key, filename) in enumerate(EXAMPLES.items()):
            image, mask, label, class_name, name = pool[ids[row]]
            original = (pool.denormalize(image) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            Image.fromarray(original).save(args.out / f"{key}_original.png")
            metadata["examples"][key] = {"filename": name, "label": int(label), "class_name": class_name,
                                         "split": split_by_index.get(ids[row], "unassigned"),
                                         "cache": {}}
            for tau in temps:
                stem = f"{key}_tau_{tau:g}.npy"
                path = cache_dir / stem
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    params = dict(PARAMS, routing_temperature=tau)
                    seed_everything(42)
                    ev._indices = [row]
                    start = time.perf_counter()
                    named, images, _, _, _, _, got_ids, failed = ev._generate_maps(
                        "routing_game", params, "input", None)
                    seconds = time.perf_counter() - start
                    if failed or list(got_ids) != [name] or "input" not in named:
                        raise RuntimeError(f"Map generation failed: {key} tau={tau}")
                    a = np.asarray(named["input"][0], dtype=np.float32)
                    if a.shape != original.shape[:2]:
                        import cv2
                        a = cv2.resize(a, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
                    if not np.isfinite(a).all():
                        raise RuntimeError(f"Non-finite map: {key} tau={tau}")
                    np.save(path, a)
                else:
                    seconds = None
                a = np.load(path)
                # Preserve every finite nonzero map, including very small
                # values: small magnitude alone does not establish residue.
                nonzero = bool(np.any(a != 0))
                metadata["examples"][key]["cache"][str(tau)] = {
                    "path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "seconds_this_run": seconds, "min": float(a.min()), "max": float(a.max()),
                    "max_abs": float(np.max(np.abs(a), initial=0)), "nonzero": nonzero,
                    "all_entries_exactly_zero": not nonzero}
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    render_cached(args, temps, Image, plt, metadata)


def render_cached(args, temps, Image, plt, metadata):
    cache_dir = args.map_cache
    metadata["displayed_temperatures"] = temps
    nrows, ncols = len(EXAMPLES), len(temps) + 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.15*ncols, 2.0*nrows), squeeze=False)
    for r, (key, _) in enumerate(EXAMPLES.items()):
        orig = np.asarray(Image.open(args.out / f"{key}_original.png").convert("RGB"))
        axes[r, 0].imshow(orig)
        axes[r, 0].set_title("Original" if r == 0 else "", fontsize=9)
        axes[r, 0].set_ylabel(key.replace('_', ' '), fontsize=9)
        for c, tau in enumerate(temps, 1):
            a = np.load(cache_dir / f"{key}_tau_{tau:g}.npy")
            axes[r, c].imshow(signed_rgb(a))
            axes[r, c].set_title(rf"$\tau={tau:g}$" if r == 0 else "", fontsize=9)
            if not np.any(a != 0):
                axes[r, c].text(.5, .5, "Zero attribution", color="white", ha="center", va="center",
                                fontsize=8, transform=axes[r, c].transAxes)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(pad=.25)
    fig.savefig(args.out / "rg_temperature_sweep.png", dpi=300, bbox_inches="tight")
    fig.savefig(args.out / "rg_temperature_sweep.pdf", bbox_inches="tight")
    plt.close(fig)
    metadata["figure_sha256"] = {
        suffix: hashlib.sha256((args.out / f"rg_temperature_sweep.{suffix}").read_bytes()).hexdigest()
        for suffix in ("png", "pdf")}
    metadata["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"report": str(args.report), "figure": str(args.out / 'rg_temperature_sweep.png'),
                      "images": list(EXAMPLES.values()), "temperatures": temps}, indent=2))


if __name__ == "__main__":
    main()
