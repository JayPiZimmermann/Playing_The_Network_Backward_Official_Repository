"""Hellinger distance between two independently random-initialised models.

Baseline for the pretrained-vs-randomised Hellinger sanity check: if two
networks that share only the architecture (no trained weights) are already
close in trajectory distribution, then the pretrained-vs-randomised gap we
report has to be read against that baseline.

Protocol
========
For each architecture, run N rounds. Each round:
  * build two fresh random copies of the network: every Conv2d / Linear
    layer of the pretrained checkpoint is re-initialized as at the end of
    the cascading randomization (different RNG seeds so the draws are
    independent);
  * sample one image from the ImageNet-S test split (deterministic per round);
  * compute Hellinger (plain + survival-conditioned) for both games.

Outputs (one file for all architectures, column ``model``; in --results-dir,
default results_double_random/):
  per_round.csv         one row per round / game / mode
  aggregate.csv         mean + std across rounds per (arch, game, mode)
"""

from __future__ import annotations

import os
import argparse
import csv
import gc
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import (
    ResNet50_Weights,
    VGG16_Weights,
    ViT_B_16_Weights,
    resnet50,
    vgg16,
    vit_b_16,
)

# Make the repo root importable so the `data` package (and any other
# sibling packages relied on by the evaluation imports below) resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bwgame.evaluation.metrics.sanity_check import (
    _get_randomizable_layers,
    _randomize_layer,
)
from bwgame.xai.methods.hellinger import HellingerDiagnostic


_HERE = Path(__file__).resolve().parent
_DEFAULT_RESULTS_DIR = _HERE / "results_double_random"
_DEFAULT_IMAGENET_ROOTS = tuple(p for p in (os.environ.get("IMAGENET_ROOT"),) if p) + (
    str(Path(__file__).resolve().parents[2] / "data" / "imagenet"),
)
_DEFAULT_IMAGENET_S_ROOTS = tuple(p for p in (os.environ.get("IMAGENET_S_ROOT"),) if p) + (
    str(Path(__file__).resolve().parents[2] / "data" / "imagenet-s"),
)

# (ctor, pretrained-weights enum) pairs; the pretrained weights are the basis
# of the re-initialization (see ``_fresh_random_model``).
_MODEL_FACTORY = {
    "vgg16":   (vgg16,    VGG16_Weights.DEFAULT),
    "resnet50":(resnet50, ResNet50_Weights.DEFAULT),
    "vit_b_16":(vit_b_16, ViT_B_16_Weights.DEFAULT),
}

_GAMES = ("adf", "rg")


def _resolve(user, candidates, label):
    for c in ([user] if user else list(candidates)):
        if c and Path(c).is_dir():
            return c
    raise FileNotFoundError(f"{label}: none of {candidates}")


def _load_images(cfg) -> Tuple[torch.Tensor, List[dict]]:
    imagenet = _resolve(cfg.imagenet_root, _DEFAULT_IMAGENET_ROOTS, "ImageNet")
    imagenet_s = _resolve(cfg.imagenet_s_root, _DEFAULT_IMAGENET_S_ROOTS, "ImageNet-S")
    from reproduce.common import imagenet_s_split
    eval_ds = imagenet_s_split("test", imagenet, imagenet_s, seed=cfg.seed,
                               n_per_class_val=cfg.n_per_class_val, image_size=cfg.image_size,
                               imagenet_s_variant=cfg.imagenet_s_split)
    imgs, recs = [], []
    for idx in range(len(eval_ds)):
        img, _m, label, cls, name = eval_ds[idx]
        imgs.append(img)
        recs.append({"index": idx, "id": Path(name).stem, "class": cls, "label": int(label)})
    return torch.stack(imgs).contiguous(), recs


def _fresh_random_model(name: str, device: torch.device, seed: int) -> nn.Module:
    """A new random network with the given architecture.

    Loads the pretrained checkpoint and re-initialises every Conv2d / Linear
    weight via ``_randomize_layer`` (kaiming_normal_ on weights,
    normal(std=0.01) on biases), the "fully randomised" endpoint of the
    cascading sanity check.  This gives a non-degenerate forward pass on all
    three architectures (with PyTorch's default init the ViT head is
    all-zero, and two such copies are indistinguishable).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    ctor, weights = _MODEL_FACTORY[name]
    model = ctor(weights=weights).to(device).eval()
    for layer in _get_randomizable_layers(model):
        _randomize_layer(model, layer)
    return model


def _compute_one(
    model_A: nn.Module,
    model_B: nn.Module,
    game: str,
    image: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """Return {hellinger, hellinger_live} for one image and one game."""
    diag = HellingerDiagnostic(model_A, model_B=model_B, game=game, tau=1.0)
    live = HellingerDiagnostic(
        model_A, model_B=model_B, game=game, tau=1.0,
        condition_on_survival=True,
    )
    x = image.unsqueeze(0)
    tgt = target.unsqueeze(0)
    def _scalar(v):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        return float(np.asarray(v).reshape(-1)[0])
    try:
        res = diag.compute(x, x, target=tgt)
        h = _scalar(res["hellinger_distance"])
    except Exception as e:
        logging.getLogger("double_random").warning(
            "plain H failed (game=%s): %s", game, e
        )
        h = float("nan")
    try:
        res_live = live.compute(x, x, target=tgt)
        h_live = _scalar(res_live["hellinger_distance"])
    except Exception as e:
        logging.getLogger("double_random").warning(
            "live H failed (game=%s): %s", game, e
        )
        h_live = float("nan")
    del diag, live
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"h_plain": h, "h_live": h_live}


def run(cfg) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("double_random")
    device = torch.device(cfg.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    images_cpu, recs = _load_images(cfg)
    n_img = images_cpu.shape[0]
    log.info("Loaded %d ImageNet-S test images", n_img)

    rng = np.random.default_rng(cfg.seed)
    # Deterministic image indices, one per round.
    image_picks = rng.integers(0, n_img, size=cfg.num_rounds).tolist()

    per_round_path = cfg.results_dir / "per_round.csv"
    agg_path = cfg.results_dir / "aggregate.csv"
    header = ["model", "round", "image_idx", "image_id", "game", "h_plain", "h_live"]

    # Overwrite fresh; we append per model.
    with per_round_path.open("w", newline="") as f:
        csv.writer(f).writerow(header)

    for model_name in cfg.models:
        log.info("=== %s ===", model_name)
        t0 = time.perf_counter()
        for r in range(cfg.num_rounds):
            idx = image_picks[r]
            image = images_cpu[idx].to(device)
            # Two fresh random nets with independent seeds.
            seed_A = cfg.seed * 1_000_000 + r * 2
            seed_B = seed_A + 1
            model_A = _fresh_random_model(model_name, device, seed_A)
            model_B = _fresh_random_model(model_name, device, seed_B)
            # Target = ImageNet-S ground-truth label (from the data record).
            # We need a meaningful class index; a random model's argmax is
            # essentially noise, and for ViT in particular it produces the
            # cemetery-collapse degenerate case where Hellinger is trivially
            # 0 (both nets route all mass to the absorbing state).
            target = torch.tensor([recs[idx]["label"]], device=device)

            rows = []
            for game in _GAMES:
                r_res = _compute_one(model_A, model_B, game, image, target)
                rows.append([model_name, r, int(idx), recs[idx]["id"], game,
                             r_res["h_plain"], r_res["h_live"]])
            with per_round_path.open("a", newline="") as f:
                csv.writer(f).writerows(rows)

            del model_A, model_B, target
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (r + 1) % 10 == 0 or r + 1 == cfg.num_rounds:
                dt = time.perf_counter() - t0
                log.info("  %s round %d/%d  (elapsed %.1fs, ETA %.1fs)",
                         model_name, r + 1, cfg.num_rounds, dt,
                         dt * (cfg.num_rounds - r - 1) / max(r + 1, 1))

        # Summarise for this model.
        _write_aggregate(per_round_path, agg_path)

    log.info("Done. per-round CSV -> %s, aggregate CSV -> %s", per_round_path, agg_path)


def _write_aggregate(per_round_path: Path, agg_path: Path) -> None:
    import pandas as pd
    df = pd.read_csv(per_round_path)
    agg = (
        df.groupby(["model", "game"])[["h_plain", "h_live"]]
          .agg(["mean", "std", "count"])
          .reset_index()
    )
    # Flatten multiindex columns
    agg.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in agg.columns
    ]
    agg.to_csv(agg_path, index=False, float_format="%.6f")


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=list(_MODEL_FACTORY.keys()),
                   choices=list(_MODEL_FACTORY.keys()))
    p.add_argument("--num-rounds", type=int, default=500)
    p.add_argument("--results-dir", type=Path, default=_DEFAULT_RESULTS_DIR)
    p.add_argument("--imagenet-root", type=str, default=None)
    p.add_argument("--imagenet-s-root", type=str, default=None)
    p.add_argument("--imagenet-s-split", type=str, default="ImageNetS50")
    p.add_argument("--n-per-class-val", type=int, default=1)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run(_parse_args())
