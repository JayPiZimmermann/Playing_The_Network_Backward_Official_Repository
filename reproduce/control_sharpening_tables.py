"""Generate the fixed-map equal-power sharpening ablation table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ATTN_ROW = "attnlrp__eattn-0.01__gcomp-on__gconv-0.25__glin-0.05"
RG_ROW = "rg__av-merged__eps-0.1__lam-0__lamsm-m10__tau-0.4__var-5"
RG_ANCHOR_ROW = "rg__av-merged__eps-0.5__lam-0__lamsm-0__tau-1"


def load(path: Path):
    return json.loads(path.read_text())


def curve(aggregate, row_label, power):
    hits = [
        item for item in aggregate["sharpening"]["curves"]
        if item["model"] == "vit_b16"
        and item["split"] == "test"
        and item["row"]["row_label"] == row_label
        and item["power"] == power
    ]
    if len(hits) != 1:
        raise RuntimeError(f"Expected one curve for {row_label}, p={power}; found {len(hits)}")
    if hits[0]["n"] != 565:
        raise RuntimeError("Unexpected test-set size")
    return hits[0]["metrics"]


def collect(rg_summary, anchor_summary, aggregate, matched):
    if rg_summary["protocol"] != "rg-posthoc-power-control-v1":
        raise RuntimeError("Unexpected RG power-control protocol")
    if rg_summary["row"] != RG_ROW or rg_summary["rank_invariance"] != "verified per image; PF20 reused only after verification":
        raise RuntimeError("RG row or rank-invariance audit mismatch")
    if matched["reference"]["row_label"] != RG_ROW:
        raise RuntimeError("Matched-control RG row differs")
    if (anchor_summary["protocol"] != "rg-anchor-fixed-power-1.5-v1"
            or anchor_summary["row"] != RG_ANCHOR_ROW):
        raise RuntimeError("Unexpected RG-anchor sharpening protocol or row")
    attn_match = next(x for x in matched["matches"] if x["method"] == "AttnLRP")
    if attn_match["row_label"] != ATTN_ROW or attn_match["power"] != 1.5:
        raise RuntimeError("Matched-control AttnLRP selection differs")

    attn_1 = curve(aggregate, ATTN_ROW, 1.0)
    attn_15 = curve(aggregate, ATTN_ROW, 1.5)
    rg_1 = rg_summary["test"]["powers"]["1.0"]
    rg_15 = rg_summary["test"]["powers"]["1.5"]
    anchor = anchor_summary["test"]["stats"]
    for item in (rg_1, rg_15):
        if item["AL"]["n"] != 565 or item["PF20"]["n"] != 565:
            raise RuntimeError("Unexpected RG test-set size")
    if any(anchor[key]["n"] != 565 for key in ("AL_p1", "AL_p1.5", "PF20_unchanged")):
        raise RuntimeError("Unexpected RG-anchor test-set size")

    # Cross-source checks prevent silently joining different selected rows.
    tol = 1e-12
    if abs(attn_15["AL"]["mean"] - attn_match["test"]["AL"]) > tol:
        raise RuntimeError("AttnLRP matched result differs from aggregate curve")
    if abs(rg_1["AL"]["mean"] - matched["reference"]["test"]["AL"]) > tol:
        raise RuntimeError("RG reference differs from power-control curve")

    return [
        ("AttnLRP", "Unmodified", 1.0, attn_1["AL"]["mean"], attn_1["PF20"]["mean"]),
        ("AttnLRP", "Power-sharpened", 1.5, attn_15["AL"]["mean"], attn_15["PF20"]["mean"]),
        ("RG anchor", "Unmodified", 1.0, anchor["AL_p1"]["mean"], anchor["PF20_unchanged"]["mean"]),
        ("RG anchor", "Power-sharpened", 1.5, anchor["AL_p1.5"]["mean"], anchor["PF20_unchanged"]["mean"]),
        ("RG (loc)", "Unmodified", 1.0, rg_1["AL"]["mean"], rg_1["PF20"]["mean"]),
        ("RG (loc)", "Power-sharpened", 1.5, rg_15["AL"]["mean"], rg_15["PF20"]["mean"]),
    ]


def render(rows):
    lines = [
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Method & Map readout & $p$ & AL $\uparrow$ & PF$_{20}$ $\downarrow$ \\",
        r"\midrule",
    ]
    for method, readout, power, al, pf20 in rows:
        lines.append(f"{method} & {readout} & ${power:g}$ & ${al:.6f}$ & ${pf20:.6f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rg-summary", type=Path, required=True)
    ap.add_argument("--anchor-summary", type=Path, required=True)
    ap.add_argument("--aggregate", type=Path, required=True)
    ap.add_argument("--matched", type=Path, required=True)
    ap.add_argument("--tex", type=Path, required=True)
    args = ap.parse_args()
    rows = collect(load(args.rg_summary), load(args.anchor_summary),
                   load(args.aggregate), load(args.matched))
    args.tex.parent.mkdir(parents=True, exist_ok=True)
    args.tex.write_text(render(rows))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
