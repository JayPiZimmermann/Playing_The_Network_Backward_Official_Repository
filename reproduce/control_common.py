"""Shared, deterministic helpers for the control experiments.

Outputs are separate from the submitted results. Seeds depend on image identity,
not shard order; model randomization never uses the explanation RNG stream.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from reproduce.common import MODELS, build_split, get_spec, seed_everything


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts).encode()).digest()[:4], "little")


@contextlib.contextmanager
def isolated_rng(seed):
    py, np_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(py)
            np.random.set_state(np_state)


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def digest_state(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def prepare_model(model):
    from bwgame.functional_replacer import replace_functional_with_modules
    replace_functional_with_modules(model, inplace=True)
    return model.eval()


def load_model(model_key, device):
    from bwgame.evaluation.run_eval import load_model as load
    seed_everything(42)
    return prepare_model(load(MODELS[model_key], device))


def explain(model, row, x, target, seed, allow_zero=False):
    import bwgame.xai  # noqa: F401
    from bwgame.xai.registry import MethodRegistry
    from bwgame.xai.wrappers import _WRAPPER_KEYS, equivariant_average
    spec = get_spec(row["method"])
    params = {**spec.base, **row["params"]}
    model.zero_grad(set_to_none=True)
    with isolated_rng(seed):
        if params.get("equivariant_averaging"):
            maps = equivariant_average(
                spec.registry, model, x.device, x, target,
                list(spec.layers) if spec.layers else None,
                n_samples=params.get("equivariant_n_samples", 50),
                max_angle=params.get("equivariant_max_angle", 10),
                max_shift_frac=params.get("equivariant_max_shift_frac", .05),
                flip_prob=params.get("equivariant_flip_prob", .5), method_params=params)
        else:
            method = MethodRegistry.instantiate(spec.registry, model, x.device)
            try:
                for k, v in params.items():
                    if k not in _WRAPPER_KEYS:
                        if not hasattr(method, k):
                            raise ValueError(f"Unknown parameter {k} for {spec.registry}")
                        setattr(method, k, v)
                maps = method.attribute(x, target, list(spec.layers) if spec.layers else None)
            finally:
                method.cleanup()
    a = np.asarray(maps[spec.component], dtype=np.float32)
    if a.shape[-2:] != tuple(x.shape[-2:]):
        a = torch.nn.functional.interpolate(torch.from_numpy(a).unsqueeze(1),
                                            size=x.shape[-2:], mode="nearest")[:, 0].numpy()
    if not np.isfinite(a).all() or (not allow_zero and not np.any(a)):
        raise ValueError(f"Nonfinite or zero explanation: {row['row_label']}")
    return a


def sample_indices(ds, split, subset=200, shard=0, shards=1):
    ids = np.arange(len(ds))
    if split == "test" and subset and subset < len(ds):
        ids = np.sort(np.random.RandomState(0).choice(len(ds), subset, replace=False))
    return ids[shard::shards].tolist()


def dataset(args):
    return build_split("imagenet-s", args.split, args)


def scalar_stats(values):
    a = np.asarray(values, dtype=float)
    good = a[np.isfinite(a)]
    if not len(good):
        return {"mean": None, "n": 0, "coverage": 0.0}
    return {"mean": float(good.mean()), "std": float(good.std(ddof=1)) if len(good)>1 else 0.,
            "n": len(good), "coverage": len(good)/len(a)}


def normalized_map(a):
    a = np.abs(np.asarray(a, dtype=float))
    scale = a.max()
    if not np.isfinite(a).all() or scale == 0:
        raise ValueError("Nonfinite or zero attribution")
    return a / scale


def map_stability(a, b):
    a = normalized_map(a).ravel()
    if not np.isfinite(b).all(): raise ValueError("Nonfinite attribution")
    if not np.any(b):
        return {"sensitivity":1.,"cosine":None,"top10_overlap":None}
    b = normalized_map(b).ravel()
    an, bn = np.linalg.norm(a), np.linalg.norm(b)
    k = max(1, int(np.ceil(.1 * len(a))))
    # Stable sorting gives reproducible ties; record this convention in protocol.
    ai, bi = np.argsort(-a, kind="stable")[:k], np.argsort(-b, kind="stable")[:k]
    return {"sensitivity": float(np.linalg.norm(a-b)/an),
            "cosine": float(np.dot(a,b)/(an*bn)),
            "top10_overlap": float(len(np.intersect1d(ai,bi))/k)}


def perturb_parameters(model, original, names, seed, scale=None):
    """Restore then randomize named layers with an independent seed per tensor.

    scale=None uses the paper's Kaiming/normal reinitialization. Otherwise add
    scale times Gaussian noise times the tensor RMS (floor 1e-6).
    """
    model.load_state_dict(original, strict=True)
    changed = []
    with torch.no_grad():
        for name in names:
            module = model.get_submodule(name)
            for local, p in module.named_parameters():
                full = name + "." + local
                g = torch.Generator(device=p.device).manual_seed(stable_seed(seed, full))
                if scale is None:
                    if p.ndim >= 2:
                        torch.nn.init.kaiming_normal_(p, generator=g)
                    else:
                        torch.nn.init.normal_(p, std=.01, generator=g)
                else:
                    rms = original[full].float().square().mean().sqrt().clamp_min(1e-6).to(p.device)
                    p.add_(torch.randn(p.shape, generator=g, device=p.device, dtype=p.dtype)*rms*scale)
                changed.append(full)
    return sorted(changed)


def image_record(ds, index, device):
    x, mask, label, cls, name = ds[index]
    mask=np.asarray(mask).astype(bool)
    if mask.ndim==3 and mask.shape[-1]==3:
        mask=mask.any(axis=-1)  # same RGB-mask convention as quantus_bridge
    return x[None].to(device), mask.squeeze(), \
        torch.tensor([label], device=device), str(name)


def add_args(parser):
    from reproduce.common import add_data_args
    add_data_args(parser)
    parser.add_argument("--model", choices=list(MODELS), default="vit_b16")
    parser.add_argument("--split", choices=("val","test"), default="test")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--subset", type=int, default=200)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
