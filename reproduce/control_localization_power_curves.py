"""Cache-only post-hoc attribution-localization power curves.

The protocol is frozen in results_summary/controls/localization_power_curves/protocol.md.
No model inference or test-set selection occurs here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reproduce.common import imagenet_s_split
from reproduce.control_common import atomic_json


POWERS = (0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0)
TABLE_POWERS = (0.5, 1.0, 1.5, 2.0, 4.0, 16.0, "inf")
MAIN_ROWS = {
    "LRP-gamma (loc)": "lrp_gamma__eps-0.1__gamma-2",
    "LRP-gamma (faith)": "lrp_gamma__eps-0.1__gamma-0.1",
    "AttnLRP (loc)": "attnlrp__eattn-0.25__gcomp-on__gconv-0.25__glin-0.05",
    "AttnLRP (faith)": "attnlrp__eattn-0.01__gcomp-on__gconv-0.25__glin-0.05",
    "RG (loc)": "rg__av-merged__eps-0.1__lam-0__lamsm-m10__tau-0.4__var-5",
    "RG (faith)": "rg__av-merged__eps-0.1__lam-0__lamsm-m5__tau-0.7__var-5",
    "RG anchor": "rg__av-merged__eps-0.5__lam-0__lamsm-0__tau-1",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_array(array: np.ndarray) -> str:
    a = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def power_metrics(raw: np.ndarray, mask: np.ndarray, power) -> dict:
    a = np.abs(np.asarray(raw, dtype=np.float64)).squeeze()
    inside = np.asarray(mask, dtype=bool).squeeze()
    if a.ndim != 2 or inside.shape != a.shape:
        raise ValueError((a.shape, inside.shape))
    if not np.isfinite(a).all() or a.max() <= 0:
        raise ValueError("nonfinite or zero attribution")
    a /= a.max()
    if power == "zero":
        z = (a > 0).astype(np.float64)
    elif power == "inf":
        z = (a == 1).astype(np.float64)
    else:
        z = np.zeros_like(a)
        support = a > 0
        z[support] = np.exp(float(power) * np.log(a[support]))
    total = z.sum()
    square = np.square(z).sum()
    if not np.isfinite(total) or total <= 0 or not np.isfinite(square) or square <= 0:
        raise ValueError((power, total, square))
    mask_fraction = float(inside.mean())
    al = float(z[inside].sum() / total)
    return {
        "AL": al,
        "mask_fraction": mask_fraction,
        "enrichment": al / mask_fraction,
        "effective_support_fraction": float(total * total / (z.size * square)),
        "max_set_inside_fraction": float(inside[a == 1].mean()),
        "pointing_game_any_max": float(inside[a == 1].any()),
    }


def bootstrap_matrix(values: np.ndarray, seed: int = 42) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(values.shape)
    rng = np.random.default_rng(seed)
    means = np.empty((10000, values.shape[1]), dtype=np.float64)
    # Chunking avoids a 10k x 565 x n_metric temporary.
    for start in range(0, 10000, 250):
        ids = rng.integers(0, values.shape[0], size=(250, values.shape[0]))
        means[start:start + 250] = values[ids].mean(axis=1)
    return {
        "mean": values.mean(axis=0).tolist(),
        "ci95_low": np.quantile(means, 0.025, axis=0).tolist(),
        "ci95_high": np.quantile(means, 0.975, axis=0).tolist(),
        "n": int(values.shape[0]),
    }


def load_masks(split: str, image_root: str, mask_root: str):
    ds = imagenet_s_split(split, image_root, mask_root)
    out = {}
    for index in range(len(ds)):
        _, mask, target, _, name = ds[index]
        m = np.asarray(mask, dtype=bool).squeeze()
        if m.ndim == 3:
            m = m.any(axis=0) if m.shape[0] <= 4 else m.any(axis=-1)
        # ImageNet-S includes a few legitimate full-image masks. AL is then
        # identically one; retain them and expose mask_fraction=1 rather than
        # silently changing coverage. Empty masks remain invalid.
        if m.shape != (224, 224) or not m.any():
            raise ValueError((split, index, m.shape, int(m.sum())))
        out[index] = {
            "mask": m,
            "mask_sha256": digest_array(m),
            "target": int(target),
            "image": str(name),
        }
    return out


def complete_rows(root: Path, split: str, expected: int):
    rows = []
    for directory in sorted((root / split).iterdir()):
        if not directory.is_dir():
            continue
        js = sorted(directory.glob("????.json"))
        nz = sorted(directory.glob("????.npz"))
        if len(js) == expected and len(nz) == expected:
            rows.append(directory.name)
    return rows


def process_split(cache_root: Path, split: str, masks: dict, rows: list[str], out: Path):
    expected = 50 if split == "val" else 565
    powers = ["zero"] + list(POWERS) + ["inf"]
    metrics = ("AL", "mask_fraction", "enrichment", "effective_support_fraction")
    per_image_path = out / f"per_image_{split}.csv"
    aggregates = {}
    sources = []
    with per_image_path.open("w", newline="") as handle:
        fieldnames = ["row", "index", "image", "target", "power"] + list(metrics) + [
            "max_set_inside_fraction", "pointing_game_any_max", "map_sha256", "mask_sha256"
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            directory = cache_root / split / row
            records = []
            source_digest = hashlib.sha256()
            for json_path in sorted(directory.glob("????.json")):
                index = int(json_path.stem)
                npz_path = json_path.with_suffix(".npz")
                meta = json.loads(json_path.read_text())
                datum = masks[index]
                if int(meta["index"]) != index or int(meta["target"]) != datum["target"] or meta["image"] != datum["image"]:
                    raise ValueError((row, index, meta["image"], datum["image"]))
                raw = np.load(npz_path)["map"]
                map_hash = sha256(npz_path)
                source_digest.update(index.to_bytes(4, "little"))
                source_digest.update(bytes.fromhex(map_hash))
                one = None
                image_rows = []
                for power in powers:
                    result = power_metrics(raw, datum["mask"], power)
                    if power == 1.0:
                        one = result
                    output_row = {
                        "row": row, "index": index, "image": datum["image"], "target": datum["target"],
                        "power": str(power), **result, "map_sha256": map_hash,
                        "mask_sha256": datum["mask_sha256"],
                    }
                    writer.writerow(output_row)
                    image_rows.append([result[m] for m in metrics])
                submitted_al = float(meta["scores"]["1.0"]["AL"])
                if one is None or abs(one["AL"] - submitted_al) > 1e-9:
                    raise ValueError((row, index, one["AL"] if one else None, submitted_al))
                records.append(image_rows)
            values = np.asarray(records, dtype=np.float64)  # image x power x metric
            if values.shape != (expected, len(powers), len(metrics)):
                raise ValueError((row, values.shape))
            stat = bootstrap_matrix(values.reshape(expected, -1))
            mean = np.asarray(stat["mean"]).reshape(len(powers), len(metrics))
            low = np.asarray(stat["ci95_low"]).reshape(len(powers), len(metrics))
            high = np.asarray(stat["ci95_high"]).reshape(len(powers), len(metrics))
            aggregates[row] = {
                "n": expected,
                "powers": {
                    str(power): {
                        metric: {"mean": float(mean[i, j]), "ci95": [float(low[i, j]), float(high[i, j])]}
                        for j, metric in enumerate(metrics)
                    } for i, power in enumerate(powers)
                },
                "ordered_map_digest": source_digest.hexdigest(),
            }
            sources.append({"row": row, "n": expected, "ordered_map_digest": source_digest.hexdigest()})
    return aggregates, sources


def plot_curves(result: dict, out: Path):
    labels = list(MAIN_ROWS)
    colors = plt.get_cmap("tab10")
    xlabels = [str(p) for p in POWERS]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for i, label in enumerate(labels):
        row = MAIN_ROWS[label]
        powers = result["test"][row]["powers"]
        al = [powers[str(p)]["AL"]["mean"] for p in POWERS]
        eff = [powers[str(p)]["effective_support_fraction"]["mean"] for p in POWERS]
        ax1.plot(range(len(POWERS)), al, marker="o", ms=3, color=colors(i), label=label)
        ax2.plot(eff, al, marker="o", ms=3, color=colors(i), label=label)
    ax1.set_xticks(range(len(POWERS)), xlabels, rotation=45)
    ax1.set(xlabel="Power p", ylabel="Attribution Localization")
    ax2.set(xlabel="Mean effective support fraction", ylabel="Attribution Localization")
    ax2.invert_xaxis()
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
    ax1.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"localization_power_curves.{ext}", dpi=220)
    plt.close(fig)


def write_table(result: dict, out: Path):
    lines = ["| Row | " + " | ".join(f"p={p}" for p in TABLE_POWERS) + " |",
             "|---|" + "---:|" * len(TABLE_POWERS)]
    for label, row in MAIN_ROWS.items():
        powers = result["test"][row]["powers"]
        cells = [f"{powers[str(p)]['AL']['mean']:.3f}" for p in TABLE_POWERS]
        lines.append("| " + label + " | " + " | ".join(cells) + " |")
    lines += ["", "Values are mean held-out AL over 565 images. No power is selected from test results."]
    (out / "main_table.md").write_text("\n".join(lines) + "\n")


def paired_main_comparisons(per_image_csv: Path) -> dict:
    """Paired AL differences, each named row minus RG localization."""
    wanted = set(MAIN_ROWS.values())
    data = {}
    with per_image_csv.open() as handle:
        for record in csv.DictReader(handle):
            if record["row"] not in wanted:
                continue
            key = (record["row"], record["power"])
            data.setdefault(key, {})[int(record["index"])] = float(record["AL"])
    reference = MAIN_ROWS["RG (loc)"]
    out = {}
    for label, row in MAIN_ROWS.items():
        if row == reference:
            continue
        out[label] = {}
        for power in [str(p) for p in POWERS] + ["inf"]:
            a, b = data[(row, power)], data[(reference, power)]
            if set(a) != set(b) or len(a) != 565:
                raise ValueError((label, power, len(a), len(b)))
            ids = sorted(a)
            delta = np.asarray([a[i] - b[i] for i in ids], dtype=np.float64)[:, None]
            stat = bootstrap_matrix(delta)
            out[label][power] = {
                "mean": stat["mean"][0],
                "ci95": [stat["ci95_low"][0], stat["ci95_high"][0]],
                "n": 565,
                "direction": "named row minus RG localization",
            }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imagenet-root", required=True)
    parser.add_argument("--imagenet-s-root", required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    masks = {split: load_masks(split, args.imagenet_root, args.imagenet_s_root) for split in ("val", "test")}
    test_rows = complete_rows(args.cache_root, "test", 565)
    val_rows = complete_rows(args.cache_root, "val", 50)
    rows = sorted(set(test_rows) & set(val_rows))
    if len(rows) != 21 or not set(MAIN_ROWS.values()).issubset(rows):
        raise ValueError((len(rows), sorted(set(MAIN_ROWS.values()) - set(rows))))
    result = {
        "protocol": "posthoc-localization-power-curves-v1",
        "selection": "none; complete frozen grid reported",
        "powers": ["zero"] + list(POWERS) + ["inf"],
        "main_rows": MAIN_ROWS,
        "rows": rows,
    }
    manifests = {}
    for split in ("val", "test"):
        result[split], manifests[split] = process_split(args.cache_root, split, masks[split], rows, args.out)
    result["paired_test_AL_vs_RG_loc"] = paired_main_comparisons(args.out / "per_image_test.csv")
    atomic_json(args.out / "summary.json", result)
    atomic_json(args.out / "manifest.json", {
        "protocol": result["protocol"], "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()), "cache_root": str(args.cache_root.resolve()),
        "sources": manifests,
        "mask_digests": {split: hashlib.sha256("".join(masks[split][i]["mask_sha256"] for i in sorted(masks[split])).encode()).hexdigest() for split in masks},
    })
    write_table(result, args.out)
    plot_curves(result, args.out)
    print(json.dumps({"rows": len(rows), "test_n": 565, "table": str(args.out / "main_table.md")}, indent=2))


if __name__ == "__main__":
    main()
