"""Numerically compare Routing Game and Zennit alpha-beta LRP on real VGG16.

Reference calculation is Zennit's AlphaBeta composite with an output seed of
one at the selected ImageNet logit. The RGB input tensor is compared before
any channel or spatial reduction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

import torch
from reproduce.control_common import atomic_json


# Code root (the directory holding bwgame/, reproduce/ and results_summary/).
REPO = Path(__file__).resolve().parents[1]
# Image identifiers and labels are taken from the completed SG validation
# audit shipped with the control summaries; override with --sg-audit.
SG_AUDIT = REPO / "results_summary/controls/sg_validation_audit.json"
SETTINGS = [(1.0, 0.0), (2.0, 1.0)]
EPSILONS = [1e-6, 0.5]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def zennit_attribution(model, x, target, alpha, beta, eps):
    from zennit.composites import SpecialFirstLayerMapComposite
    from zennit.rules import AlphaBeta

    layer_map = [
        (torch.nn.Conv2d, AlphaBeta(alpha=alpha, beta=beta, stabilizer=eps)),
        (torch.nn.Linear, AlphaBeta(alpha=alpha, beta=beta, stabilizer=eps)),
    ]
    composite = SpecialFirstLayerMapComposite(layer_map=layer_map,
                                               first_map=layer_map, canonizers=[])
    inp = x.detach().clone().requires_grad_(True)
    with composite.context(model) as wrapped:
        logits = wrapped(inp)
        seed = torch.zeros_like(logits)
        seed[0, target] = 1.0
        logits.backward(gradient=seed)
    return inp.grad.detach(), logits.detach()


def routing_attribution(model, x, target, alpha, beta, eps):
    from bwgame.xai.methods.routing_game import _RoutingGameHookManager

    manager = _RoutingGameHookManager(model=model, lrp_alpha=alpha, lrp_beta=beta,
                                      lrp_eps=eps, routing_temperature=1.0,
                                      adf_lambda=0.0)
    manager.install()
    try:
        inp = x.detach().clone().requires_grad_(True)
        logits = model(inp)
        seed = torch.zeros_like(logits)
        seed[0, target] = 1.0
        logits.backward(gradient=seed)
        return inp.grad.detach(), logits.detach()
    finally:
        manager.remove()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--imagenet-root", type=Path,
                    default=Path(os.environ.get("IMAGENET_ROOT", str(REPO / "data" / "imagenet"))))
    ap.add_argument("--imagenet-s-root", type=Path,
                    default=Path(os.environ.get("IMAGENET_S_ROOT", str(REPO / "data" / "imagenet-s"))))
    ap.add_argument("--sg-audit", type=Path, default=SG_AUDIT)
    ap.add_argument("--out", type=Path, default=REPO / "results/controls/rg_recovery_vgg16.json")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)

    audit = json.loads(args.sg_audit.read_text())
    # These are the first three deterministic ImageNet-S50 rows from the
    # completed SG validation audit, retaining its image IDs and labels.
    source_key = "float32_centered_original_pools_True"
    source = audit[source_key]["records"][:args.samples]
    from data.imagenet_s import ImageNetS
    ds = ImageNetS(imagenet_root=str(args.imagenet_root),
                   imagenet_s_root=str(args.imagenet_s_root), split="ImageNetS50",
                   image_size=224)
    by_name = {row["image_name"]: i for i, row in enumerate(ds.samples)}
    from torchvision.models import VGG16_Weights
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / Path(
        VGG16_Weights.IMAGENET1K_V1.url).name
    from bwgame.evaluation.run_eval import load_model
    net = load_model("vgg16", device)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Loaded torchvision checkpoint not found for provenance: {checkpoint}")
    checkpoint_hash = sha256(checkpoint)
    records = []
    base = {
        "audit": "actual pretrained torchvision VGG16, ImageNet-S50 SG validation examples",
        "reference": "zennit.rules.AlphaBeta in SpecialFirstLayerMapComposite",
        "reference_independent_of_routing_game": True,
        "seed": "unit gradient at selected target logit (same in both implementations)",
        "bias_policy": "Zennit AlphaBeta zero_params=None; its standard positive/negative bias modifiers; same model biases in RG",
        "pooling": "same realized torch max-pool winner indices in each forward; no cached/backward reference alignment",
        "dtype_comparison": "each implementation evaluated independently at the same dtype",
        "device": str(device),
        "tf32": False,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "sg_manifest": str(args.sg_audit),
        "sg_manifest_sha256": sha256(args.sg_audit),
        "sg_source_key": source_key,
        "settings": [{"alpha": a, "beta": b, "epsilon": e}
                     for a,b in SETTINGS for e in EPSILONS],
        "torch_version": torch.__version__,
        "torchvision_version": importlib.metadata.version("torchvision"),
        "zennit_version": importlib.metadata.version("zennit"),
        "script_sha256": sha256(Path(__file__).resolve()),
        "routing_game_source_sha256": sha256(REPO / "bwgame/xai/methods/routing_game.py"),
        "reference_test_source_sha256": sha256(REPO / "tests/test_routing_game_lrp_match.py"),
    }
    for row in source:
        name, target = row["image"], int(row["target"])
        if name not in by_name:
            raise FileNotFoundError(f"SG audit image missing from ImageNet-S dataset: {name}")
        ds_i = by_name[name]
        sample = ds[ds_i]
        x = sample[0][None].to(device)
        actual_path = next(p for p in (args.imagenet_root / "val" / name,
                                       args.imagenet_root / "val" / ds.samples[ds_i]["wnid"] / name)
                           if p.exists())
        for dtype in (torch.float32, torch.float64):
            net_dtype = net.to(dtype=dtype)
            x_dtype = x.to(dtype=dtype)
            for alpha, beta in SETTINGS:
                for eps in EPSILONS:
                    started = time.time()
                    rg, rg_logits = routing_attribution(net_dtype, x_dtype, target,
                                                        alpha, beta, eps)
                    zn, zn_logits = zennit_attribution(net_dtype, x_dtype, target,
                                                       alpha, beta, eps)
                    delta = (rg - zn).detach().double()
                    reference = zn.detach().double()
                    rec = {
                        "image": name, "sample_index_in_sg_source": source.index(row),
                        "target": target, "dataset_target": int(sample[2]),
                        "input_sha256": sha256(actual_path), "dtype": str(dtype),
                        "alpha": alpha, "beta": beta, "epsilon": eps,
                        "input_shape": list(x.shape),
                        "rg_target_logit": float(rg_logits[0,target]),
                        "zennit_target_logit": float(zn_logits[0,target]),
                        "forward_logit_max_absolute_error": float((rg_logits - zn_logits).abs().max()),
                        "max_absolute_error": float(delta.abs().max()),
                        "relative_l2_error": float(delta.norm() / reference.norm().clamp_min(1e-30)),
                        "reference_l2": float(reference.norm()),
                        "rg_max_abs": float(rg.detach().double().abs().max()),
                        "zennit_max_abs": float(reference.abs().max()),
                        "finite": bool(torch.isfinite(rg).all() and torch.isfinite(zn).all()),
                        "pass": bool(torch.isfinite(rg).all() and torch.isfinite(zn).all()
                                     and (delta.abs().max() < (1e-5 if dtype == torch.float32 else 1e-12))),
                        "elapsed_seconds": time.time() - started,
                    }
                    records.append(rec)
                    atomic_json(args.out, {**base, "n_checks": len(records), "records": records})
                    print(json.dumps(rec), flush=True)
            net = net.to(dtype=torch.float32)
    print(f"Wrote {args.out} ({len(records)} comparisons)", flush=True)


if __name__ == "__main__":
    main()
