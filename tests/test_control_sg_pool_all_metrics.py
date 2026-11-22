import numpy as np

from reproduce.control_sg_pool_all_metrics import (
    SHORT_TO_LONG, _bootstrap_mean, summarize_case,
)


def test_bootstrap_summary_is_deterministic_and_uses_finite_values():
    a = np.array([1.0, 2.0, np.nan, 3.0])
    first = _bootstrap_mean(a)
    second = _bootstrap_mean(a)
    assert first == second
    assert first["mean"] == 2.0
    assert first["n"] == 3
    assert first["ci95"][0] <= 2.0 <= first["ci95"][1]


def test_summary_reads_every_paper_metric_and_pairs_by_image(tmp_path):
    case = tmp_path / "case" / "evaluator" / "per_image"
    case.mkdir(parents=True)
    ids = np.array(["a", "b"])
    base = {"image_ids": ids}
    for i, long in enumerate(SHORT_TO_LONG.values()):
        base[long] = np.array([i + 1.0, i + 2.0])
    np.savez_compressed(case / "legacy_pp.npz", **base)
    changed = dict(base)
    for long in SHORT_TO_LONG.values():
        changed[long] = base[long] + 0.25
    np.savez_compressed(case / "original_pool_pp.npz", **changed)
    out = summarize_case(tmp_path / "case", {"adf_lambda": 0.0})
    assert set(out["rows"]["legacy_pp"]) == set(SHORT_TO_LONG)
    assert set(out["paired_deltas"]) == set(SHORT_TO_LONG)
    assert all(v["mean"] == 0.25 for v in out["paired_deltas"].values())
    assert out["n"] == 2
