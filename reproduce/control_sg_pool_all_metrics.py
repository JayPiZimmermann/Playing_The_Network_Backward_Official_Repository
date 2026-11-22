"""Evaluate original-forward MaxPool routing for SG on every paper metric.

The only attribution change is the source of MaxPool winner indices: the
legacy SG recomputes them from ``pos - neg``; the controlled version reuses
indices captured from the model's ordinary forward pass.  Sensitivity
centering and the legacy ReLU zero convention are deliberately unchanged.

Results are resumable through the Evaluator's map, metric, curve, and
per-image caches.  The summary reports paired bootstrap intervals for
``controlled - legacy`` on the same 50 validation images and (for
sensitivity) the same content-seeded perturbations.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from reproduce.common import build_split, load_metric_configs, seed_everything
from reproduce.control_common import add_args, atomic_json, digest_state, load_model


SHORT_TO_LONG = {
    "AL": "attribution_localisation",
    "PG": "pointing_game",
    "TK": "top_k_intersection",
    "PF5": "pixel_flipping_abs_5pct",
    "PF20": "pixel_flipping_abs_20pct",
    "Sel": "selectivity",
    "MaxS": "max_sensitivity",
    "AvgS": "avg_sensitivity",
    "time_s": "time_s",
}
CASES = ("anchor", "loc", "faith")
PROTOCOL = "sg-original-pool-indices-all-paper-metrics-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _register_pool_method():
    """Register the controlled method without modifying the library registry."""
    import bwgame.xai  # populate the normal registry
    from bwgame.xai.methods.adf_game import AdfGame
    from bwgame.xai.registry import MethodRegistry

    @MethodRegistry.register
    class PoolCachedAdfGame(AdfGame):
        @classmethod
        def config(cls):
            return dataclasses.replace(
                super().config(), name="adf_game_original_pool_indices",
                description="SG with MaxPool indices captured from the original forward pass",
            )

        def attribute(self, x, target, layers=None):
            from bwgame.alignment_cache import AlignmentCache, AlignmentMode
            from bwgame.dc_context import dc_decomposition
            from bwgame.xai.postprocess import tensor_to_numpy

            class PoolOnlyCache(AlignmentCache):
                def should_use_cached_mask(self, name):
                    # Deliberately do not reuse ReLU masks: this experiment
                    # isolates pooling and keeps the submitted >=0 convention.
                    return self.get_maxpool_indices(name) is not None

            maps = []
            for b in range(x.shape[0]):
                xb = x[b:b + 1].detach().clone()
                yb = int(target[b].item())
                cache = PoolOnlyCache(mode=AlignmentMode.NONE)
                cache.capture_original(
                    self.model, xb, capture_forward=True, capture_backward=False,
                )
                if cache.original_input_grad is not None:
                    raise AssertionError("Pooling-only cache captured a reference gradient")
                if any(d.original_gradient is not None for d in cache._layer_data.values()):
                    raise AssertionError("Pooling-only cache contains reference gradients")
                cache.attach_to_model(self.model)
                try:
                    self.model.zero_grad(set_to_none=True)
                    with dc_decomposition(
                        self.model, xb, backprop_mode="stopping",
                        adf_mode="adf", adf_lambda=self.adf_lambda,
                        adf_input_variance=self.adf_input_variance,
                        adf_probabilistic=self.adf_probabilistic,
                        entropy_bias_adapt=self.entropy_lambda,
                        alpha=0.0,
                    ) as dc:
                        dc.output[0, yb].backward()
                    maps.append(tensor_to_numpy(dc.sensitivities.delta_pp.sum(dim=1)))
                finally:
                    cache.detach_from_model(self.model)
            return {"pp": np.concatenate(maps, axis=0)}

    return PoolCachedAdfGame.config().name


def _case_params(selection: dict, case: str) -> dict:
    if case == "anchor":
        return {"adf_lambda": 0.0, "adf_input_variance": 1.0,
                "adf_probabilistic": False}
    return dict(selection["methods"]["sg"][case]["params"])


def _bootstrap_mean(values: np.ndarray, seed: int = 42) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"mean": None, "ci95": None, "n": 0}
    rng = np.random.default_rng(seed)
    boot = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(1)
    return {"mean": float(values.mean()),
            "ci95": np.quantile(boot, [.025, .975]).tolist(), "n": int(len(values))}


def _load_store(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def summarize_case(case_dir: Path, params: dict) -> dict:
    paths = {
        "legacy_pp": case_dir / "evaluator/per_image/legacy_pp.npz",
        "original_pool_pp": case_dir / "evaluator/per_image/original_pool_pp.npz",
    }
    stores = {name: _load_store(path) for name, path in paths.items()}
    ids = stores["legacy_pp"]["image_ids"].astype(str)
    if not np.array_equal(ids, stores["original_pool_pp"]["image_ids"].astype(str)):
        raise RuntimeError("Per-image cache IDs differ between paired methods")
    rows = {}
    for row, store in stores.items():
        rows[row] = {}
        for short, long in SHORT_TO_LONG.items():
            if long not in store:
                raise RuntimeError(f"Missing {long} in {paths[row]}")
            rows[row][short] = _bootstrap_mean(store[long])
    deltas = {}
    for short, long in SHORT_TO_LONG.items():
        a = np.asarray(stores["legacy_pp"][long], dtype=float)
        b = np.asarray(stores["original_pool_pp"][long], dtype=float)
        ok = np.isfinite(a) & np.isfinite(b)
        deltas[short] = _bootstrap_mean(b[ok] - a[ok])
    return {"params": params, "n": int(len(ids)), "rows": rows,
            "paired_deltas": deltas,
            "delta_definition": "original_pool_pp - legacy_pp"}


def _dataset_fingerprint(ds) -> tuple[list[str], str]:
    """Hash the exact transformed inputs, masks, labels and ordered image IDs."""
    digest = hashlib.sha256()
    image_ids = []
    for i in range(len(ds)):
        image, mask, label, cls, name = ds[i]
        image_ids.append(str(name))
        digest.update(str(name).encode())
        digest.update(str(int(label)).encode())
        digest.update(np.ascontiguousarray(image.numpy()).tobytes())
        digest.update(np.ascontiguousarray(np.asarray(mask)).tobytes())
    return image_ids, digest.hexdigest()


def _manifest(args, model, ds, cases, pool_method) -> dict:
    here = Path(__file__).resolve()
    code_root = here.parents[1]
    files = sorted((code_root / "bwgame").rglob("*.py"))
    files += [here, code_root / "reproduce/common.py",
              code_root / "reproduce/control_common.py"]
    image_ids, data_sha = _dataset_fingerprint(ds)
    return {
        "protocol": PROTOCOL, "model": args.model, "split": args.split,
        "n_images": len(ds), "cases": cases, "pool_method": pool_method,
        "model_state_sha256": digest_state(model),
        "source_sha256": {str(p.relative_to(code_root)): _sha256(p) for p in files},
        "ordered_image_ids": image_ids,
        "transformed_dataset_sha256": data_sha,
        "metric_configs": load_metric_configs(
            ["AL", "PG", "TK", "PF5", "PF20", "Sel", "MaxS", "AvgS"],
            nr_samples=args.nr_samples),
        "pooling_change_only": True,
        "sensitivity_alpha": 0.0,
        "relu_zero_convention": "legacy >= 0 in both rows",
        "target": "dataset label in both rows",
        "sensitivity_noise": f"paper content-seeded uniform +/-0.2; {args.nr_samples} draws",
        "top_k": 1000,
        "runtime_note": "controlled row includes an extra original-forward index-capture pass; unoptimized control",
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_args(ap)
    ap.set_defaults(model="vgg16", split="val")
    ap.add_argument("--case", choices=(*CASES, "all"), default="all")
    ap.add_argument("--nr-samples", type=int, default=50)
    args = ap.parse_args()
    if args.model != "vgg16" or args.split != "val":
        raise ValueError("This control is defined for VGG-16 on the 50-image validation split")
    cases = CASES if args.case == "all" else (args.case,)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_everything(args.seed)
    model = load_model(args.model, device)
    ds = build_split("imagenet-s", args.split, args)
    pool_method = _register_pool_method()
    selection_path = Path(__file__).resolve().parents[1] / "results_summary/selection/vgg16/selected.json"
    selection = json.loads(selection_path.read_text())
    params_by_case = {case: _case_params(selection, case) for case in cases}
    manifest = _manifest(args, model, ds, params_by_case, pool_method)
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise RuntimeError(f"Existing cache has a different manifest: {manifest_path}")
    atomic_json(manifest_path, manifest)

    from bwgame.evaluation.evaluator import Evaluator
    metrics = manifest["metric_configs"]
    for case in cases:
        case_dir = args.out / case
        evaluator = Evaluator(
            model=model, dataset=ds, device=device, output_dir=case_dir / "evaluator",
            metric_configs=metrics, batch_size=1, softmax=True, seed=args.seed,
            save_heatmaps=False, cache_maps=True, warmup=args.warmup,
        )
        params = params_by_case[case]
        for registry, label in (("adf_game", "legacy_pp"),
                                (pool_method, "original_pool_pp")):
            seed_everything(args.seed)
            started = time.perf_counter()
            evaluator.run_method(registry, params, component="pp", config_label=label)
            print(json.dumps({"case": case, "row": label,
                              "elapsed_s": time.perf_counter() - started}), flush=True)
        atomic_json(case_dir / "summary.json", summarize_case(case_dir, params))

    available = {}
    for case in CASES:
        path = args.out / case / "summary.json"
        if path.exists():
            available[case] = json.loads(path.read_text())
    atomic_json(args.out / "summary.json", {"protocol": manifest, "cases": available})


if __name__ == "__main__":
    main()
