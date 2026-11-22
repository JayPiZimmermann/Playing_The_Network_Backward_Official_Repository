"""Recheck fixed p=1.5 RG-anchor AL from cached maps, with paired intervals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from reproduce.control_common import atomic_json
from reproduce.control_localization_power_curves import load_masks


ANCHOR = "rg__av-merged__eps-0.5__lam-0__lamsm-0__tau-1"
SELECTED = "rg__av-merged__eps-0.1__lam-0__lamsm-m10__tau-0.4__var-5"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap(values):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(42)
    means = np.concatenate([
        x[rng.integers(0, len(x), size=(250, len(x)))].mean(axis=1)
        for _ in range(40)
    ])
    return {"n": len(x), "mean": float(x.mean()),
            "ci95": np.quantile(means, [.025, .975]).tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imagenet-root", required=True)
    ap.add_argument("--imagenet-s-root", required=True)
    args = ap.parse_args()
    summary = {"protocol": "rg-anchor-fixed-power-1.5-v1", "row": ANCHOR,
               "transform": "(|a| / max|a|)^p", "powers": [1.0, 1.5],
               "selection": "p=1.5 fixed by user; no exponent search or test selection",
               "bootstrap": {"resamples": 10000, "seed": 42,
                             "unit": "paired image", "interval": "percentile 95%"}}
    manifest = {"script_sha256": digest(__file__), "cache_root": str(args.cache_root.resolve()),
                "sources": {}}
    for split, expected in (("val", 50), ("test", 565)):
        masks = load_masks(split, args.imagenet_root, args.imagenet_s_root)
        paths = sorted((args.cache_root / split / ANCHOR).glob("????.json"))
        assert len(paths) == expected == len(masks)
        records, sources, params = [], [], None
        for index, jp in enumerate(paths):
            meta = json.loads(jp.read_text())
            datum = masks[index]
            assert int(meta["index"]) == int(jp.stem) == index
            assert meta["image"] == datum["image"] and int(meta["target"]) == datum["target"]
            assert meta["row"]["row_label"] == ANCHOR
            if params is None:
                params = meta["row"]["params"]
            assert meta["row"]["params"] == params
            npz = jp.with_suffix(".npz")
            with np.load(npz) as cached:
                a = np.abs(np.asarray(cached["map"], dtype=np.float64)).squeeze()
            assert a.shape == datum["mask"].shape == (224, 224)
            assert np.isfinite(a).all() and a.max() > 0
            a /= a.max()
            sharp = a ** 1.5
            order = np.argsort(a.ravel(), kind="stable")
            v, w = a.ravel()[order], sharp.ravel()[order]
            assert np.all(np.diff(w) >= 0)
            assert np.array_equal(v[1:] == v[:-1], w[1:] == w[:-1])
            al = [float(m[datum["mask"]].sum() / m.sum()) for m in (a, sharp)]
            for power, value in zip(("1.0", "1.5"), al):
                assert abs(value - float(meta["scores"][power]["AL"])) < 1e-12
            selected_path = args.cache_root / split / SELECTED / jp.name
            selected = json.loads(selected_path.read_text())
            assert selected["index"] == index and selected["image"] == meta["image"]
            assert selected["target"] == meta["target"]
            records.append({"index": index, "image": meta["image"], "target": meta["target"],
                            "AL_p1": al[0], "AL_p1.5": al[1], "AL_delta": al[1] - al[0],
                            "selected_RG_AL_p1": float(selected["scores"]["1.0"]["AL"]),
                            "PF20_unchanged": float(meta["scores"]["1.0"]["PF20"])})
            sources.append({"index": index, "json": str(jp), "json_sha256": digest(jp),
                            "map": str(npz), "map_sha256": digest(npz),
                            "mask_sha256": datum["mask_sha256"],
                            "selected_json_sha256": digest(selected_path)})
        stats = {key: bootstrap([r[key] for r in records])
                 for key in ("AL_p1", "AL_p1.5", "AL_delta", "selected_RG_AL_p1", "PF20_unchanged")}
        stats["sharpened_anchor_minus_selected_RG_AL"] = bootstrap([
            r["AL_p1.5"] - r["selected_RG_AL_p1"] for r in records])
        summary[split] = {"n": expected, "params": params, "stats": stats,
                          "rank_and_tie_invariance_verified": True,
                          "cached_AL_agreement_tolerance": 1e-12, "per_image": records}
        manifest["sources"][split] = sources
        print(split, json.dumps(stats), flush=True)
    atomic_json(args.out / "summary.json", summary)
    atomic_json(args.out / "manifest.json", manifest)


if __name__ == "__main__":
    main()
