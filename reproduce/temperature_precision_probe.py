"""Compare native-precision RG maps at low temperatures on three fixed ViT images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = {
    "red_fox_snow": "ILSVRC2012_val_00025140.JPEG",
    "kuvasz": "ILSVRC2012_val_00024981.JPEG",
    "airliner": "ILSVRC2012_val_00008292.JPEG",
    "fulltest_zero_04677": "ILSVRC2012_val_00004677.JPEG",
    "fulltest_zero_06007": "ILSVRC2012_val_00006007.JPEG",
    "fulltest_zero_13623": "ILSVRC2012_val_00013623.JPEG",
}
PARAMS = {
    "lrp_alpha": 2.0, "lrp_beta": 1.0, "lrp_eps": 0.1,
    "adf_lambda": 0.0, "adf_softmax_lambda": -10.0,
    "adf_input_variance": 5.0, "av_rule": "merged",
}


def stats(a: np.ndarray) -> dict:
    a = np.asarray(a)
    finite = np.isfinite(a)
    abs_a = np.abs(a[finite])
    positive = abs_a[abs_a > 0]
    return {
        "dtype": str(a.dtype), "shape": list(a.shape),
        "finite_count": int(finite.sum()), "element_count": int(a.size),
        "nonfinite_count": int((~finite).sum()),
        "all_finite": bool(finite.all()),
        "all_zero": bool(finite.all() and np.all(a == 0)),
        "max_abs": float(abs_a.max(initial=0.0)),
        "min_positive_abs": float(positive.min()) if positive.size else None,
        "nonzero_count": int(np.count_nonzero(a)),
    }


def normalized_agreement(left: np.ndarray, right: np.ndarray) -> dict:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    result = {"cosine": None, "normalized_l2": None, "both_nonzero": bool(na > 0 and nb > 0)}
    if na > 0 and nb > 0:
        an, bn = a / na, b / nb
        result["cosine"] = float(np.dot(an, bn))
        result["normalized_l2"] = float(np.linalg.norm(an - bn))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    code_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--imagenet-root",
                        default=os.environ.get("IMAGENET_ROOT", str(code_root / "data" / "imagenet")))
    parser.add_argument("--imagenet-s-root",
                        default=os.environ.get("IMAGENET_S_ROOT", str(code_root / "data" / "imagenet-s")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/evaluation-low-temperature/precision_probe.json")
    parser.add_argument("--maps", type=Path, default=ROOT / "code/playing_backward/results/evaluation-low-temperature/precision_probe_maps")
    parser.add_argument("--temperatures", default="0.1")
    args = parser.parse_args()

    import torch
    import sys
    sys.path.insert(0, str(ROOT / "code/playing_backward"))
    from reproduce.common import imagenet_s_split
    from bwgame.evaluation.evaluator import Evaluator
    from bwgame.evaluation.run_eval import load_model
    from bwgame.xai.methods import routing_game
    from bwgame.xai.methods.routing_game import RoutingGame

    torch.set_num_threads(2)
    device = torch.device(args.device)
    temperatures = [float(t) for t in args.temperatures.split(",")]
    test = imagenet_s_split("test", args.imagenet_root, args.imagenet_s_root)
    pool = test.dataset
    by_name = {item["image_name"]: i for i, item in enumerate(pool.samples)}
    missing = [name for name in EXAMPLES.values() if name not in by_name]
    if missing:
        raise RuntimeError(f"Images missing from canonical ImageNet-S test split: {missing}")

    args.maps.mkdir(parents=True, exist_ok=True)
    # Retain the exact input examples and labels; cast only tensor precision.
    examples = {}
    for key, name in EXAMPLES.items():
        image, mask, label, class_name, image_name = pool[by_name[name]]
        examples[key] = {"image": image.unsqueeze(0), "label": torch.tensor([label], device=device),
                         "filename": image_name, "class_name": class_name}

    old_tensor_to_numpy = routing_game.tensor_to_numpy
    routing_game.tensor_to_numpy = lambda t: t.detach().cpu().numpy()
    report = {
        "model": "torchvision ViT_B_16_Weights.IMAGENET1K_V1",
        "torch_version": torch.__version__,
        "probe_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "routing_game_source_sha256": hashlib.sha256(Path(routing_game.__file__).read_bytes()).hexdigest(),
        "checkpoint_dtype": "loaded torchvision weights; cast to requested precision for each pass",
        "split": "canonical ImageNet-S50 test split, three fixed qualitative and three full-evaluator zero-map IDs",
        "examples": {k: {"filename": v["filename"], "class_name": v["class_name"], "label": int(v["label"].item())}
                     for k, v in examples.items()},
        "fixed_params": PARAMS, "temperatures": temperatures,
        "maps": {}, "agreement": {},
        "model_preparation": "Evaluator.__init__ decompose_composites + replace_inline_modules + _replace_functional_in_module",
        "note": "Maps are captured before tensor_to_numpy float32 conversion; no threshold or per-map scaling is applied.",
    }
    try:
        for precision in ("float32", "float64"):
            dtype = torch.float32 if precision == "float32" else torch.float64
            model = load_model("vit_b_16", device).eval()
            # Mirror Evaluator.__init__: its decomposition/replacements are
            # required for RoutingGame's hooks to see ViT attention internals.
            subset = torch.utils.data.Subset(pool, [by_name[next(iter(EXAMPLES.values()))]])
            with tempfile.TemporaryDirectory(prefix="rg_precision_prepare_") as work:
                evaluator = Evaluator(model=model, dataset=subset, device=device,
                                      output_dir=Path(work), metric_configs=[], batch_size=1,
                                      softmax=True, seed=42, save_heatmaps=False,
                                      cache_maps=False, warmup=0)
            # Decomposition may create fresh submodules, so cast after its
            # replacement pass to ensure every generated module is converted.
            model = evaluator.model.to(dtype=dtype).eval()
            for key, example in examples.items():
                x = example["image"].to(device=device, dtype=dtype)
                target = example["label"]
                for tau in temperatures:
                    method = RoutingGame(model, device)
                    for name, value in PARAMS.items():
                        setattr(method, name, value)
                    method.routing_temperature = tau
                    maps = method.attribute(x, target, layers=["input"])
                    arr = np.asarray(maps["input"])
                    path = args.maps / f"{key}_{precision}_tau_{tau:g}.npy"
                    np.save(path, arr)
                    report["maps"][f"{key}/{precision}/tau={tau:g}"] = {
                        **stats(arr), "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    print(f"{key} {precision} tau={tau:g}: {report['maps'][f'{key}/{precision}/tau={tau:g}']}", flush=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        for key in examples:
            for tau in temperatures:
                a = np.load(args.maps / f"{key}_float32_tau_{tau:g}.npy")
                b = np.load(args.maps / f"{key}_float64_tau_{tau:g}.npy")
                report["agreement"][f"{key}/tau={tau:g}"] = normalized_agreement(a, b)
                peak = float(np.max(np.abs(b), initial=0.0))
                if peak > 0 and np.isfinite(peak):
                    scaled = (b / peak).astype(np.float32)
                    report.setdefault("fp64_positive_rescale_to_fp32", {})[f"{key}/tau={tau:g}"] = {
                        "positive_scale": peak,
                        "scaled_map": stats(scaled),
                        "agreement_to_native_fp64": normalized_agreement(scaled, b),
                    }
                else:
                    report.setdefault("fp64_positive_rescale_to_fp32", {})[f"{key}/tau={tau:g}"] = {
                        "positive_scale": None, "scaled_map": None,
                        "agreement_to_native_fp64": normalized_agreement(np.zeros_like(b), b),
                    }
    finally:
        routing_game.tensor_to_numpy = old_tensor_to_numpy
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
