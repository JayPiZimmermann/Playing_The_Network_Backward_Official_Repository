"""Inference respects independent images when each image has several class records."""

import json
import math

import numpy as np
import pytest

from reproduce import bootstrap_ci as bc


def _enumerated_plan(ids):
    plan = bc.BootstrapPlan(ids, n_boot=4, seed=0, clustered=True)
    plan.indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int32)
    return plan


def test_unequal_image_sizes_preserve_record_weighted_estimand():
    plan = _enumerated_plan(["a__cat", "a__dog", "b__cat"])
    values = np.array([0.0, 2.0, 8.0])
    expected_draws = np.array([1.0, 10.0 / 3.0, 10.0 / 3.0, 8.0])

    np.testing.assert_allclose(plan.bootstrap_means(values), expected_draws)
    mean, lo, hi, n = plan.ci(values)
    assert n == 3
    assert plan.n_units == 2
    assert mean == pytest.approx(10.0 / 3.0)
    np.testing.assert_allclose([lo, hi], np.percentile(expected_draws, [2.5, 97.5]))
    assert mean != pytest.approx(4.5)  # The mean of image means is a different estimand.


def test_uniformly_duplicating_class_records_does_not_add_independent_images():
    original = _enumerated_plan(["a__cat", "a__dog", "b__cat"])
    duplicate = _enumerated_plan(
        ["a__cat", "a__dog", "a__bird", "a__horse", "b__cat", "b__dog"]
    )
    draws = original.bootstrap_means(np.array([0.0, 2.0, 8.0]))
    duplicated_draws = duplicate.bootstrap_means(np.array([0.0, 2.0, 0.0, 2.0, 8.0, 8.0]))
    np.testing.assert_array_equal(draws, duplicated_draws)
    assert original.n_units == duplicate.n_units == 2


def test_single_class_images_have_the_same_ci_as_iid_image_resampling():
    values = np.array([1.5, -2.0, 7.0, 0.0])
    ids = ["a__cat", "b__cat", "c__dog", "d__bird"]
    plan = bc.BootstrapPlan(ids, n_boot=257, seed=17, clustered=True)
    expected = bc.ci_from_indices(values, bc.boot_indices(len(ids), 257, 17))
    np.testing.assert_array_equal(plan.ci(values), expected)


def test_cluster_draws_are_deterministic():
    ids = ["a__cat", "a__dog", "b__cat", "c__bird"]
    values = np.array([0.0, 2.0, 8.0, 4.0])
    first = bc.BootstrapPlan(ids, n_boot=129, seed=9, clustered=True)
    second = bc.BootstrapPlan(ids, n_boot=129, seed=9, clustered=True)
    np.testing.assert_array_equal(first.bootstrap_means(values), second.bootstrap_means(values))


def test_class_separator_is_removed_only_at_the_end():
    plan = bc.BootstrapPlan(
        ["part__a__cat", "part__a__dog", "part__b__cat"],
        n_boot=32, seed=0, clustered=True,
    )
    assert plan.n_units == 2


def test_two_records_from_one_informative_image_do_not_support_inference():
    plan = _enumerated_plan(["a__cat", "a__dog", "b__cat"])
    values = np.array([0.0, 1.0, np.nan])
    mean, lo, hi, n = plan.ci(values)
    assert mean == 0.5
    assert n == 2
    assert math.isnan(lo) and math.isnan(hi)
    _, p, n_pair = plan.compare(values, np.array([0.0, 0.0, np.nan]), "PG")
    assert math.isnan(p)
    assert n_pair == 2


def test_resamples_with_no_finite_records_do_not_produce_inference():
    plan = bc.BootstrapPlan(["a__cat", "b__cat", "c__cat"], n_boot=4, seed=0, clustered=True)
    plan.indices = np.full((4, 3), 2, dtype=np.int32)
    values = np.array([1.0, 2.0, np.nan])

    mean, lo, hi, n = plan.ci(values)
    delta, p, n_pair = plan.compare(values, np.array([0.0, 0.0, np.nan]), "AL")

    assert mean == delta == 1.5
    assert n == n_pair == 2
    assert math.isnan(lo) and math.isnan(hi)
    assert math.isnan(p)


def test_paired_missing_records_use_common_support_and_cluster_draws():
    plan = _enumerated_plan(["a__cat", "a__dog", "b__cat", "b__dog"])
    a = np.array([3.0, np.nan, 1.0, 6.0])
    b = np.array([1.0, 8.0, 1.0, np.nan])

    delta, p, n = plan.compare(a, b, "AL")

    assert n == 2
    assert delta == 1.0
    # Paired differences have draws [2, 1, 1, 0], with two null-centered
    # deviations at least as large as the observed difference of one.
    assert p == pytest.approx(3.0 / 5.0)


def test_clustered_pointing_game_uses_bootstrap_not_rowwise_mcnemar(monkeypatch):
    from bwgame.evaluation import stats

    def forbidden(*args, **kwargs):
        pytest.fail("Dependent image-class records cannot use ordinary McNemar inference")

    monkeypatch.setattr(stats, "mcnemar_test", forbidden)
    plan = _enumerated_plan(["a__cat", "a__dog", "b__cat"])
    delta, p, n = plan.compare(np.array([1.0, 1.0, 1.0]), np.zeros(3), "PG")

    assert delta == 1.0
    assert n == 3
    assert p == pytest.approx(1.0 / 5.0)


def test_identical_paired_scores_have_unit_p_value():
    plan = _enumerated_plan(["a__cat", "a__dog", "b__cat"])
    values = np.array([0.0, 1.0, 0.0])
    assert plan.compare(values, values.copy(), "PG") == (0.0, 1.0, 3)


@pytest.mark.parametrize("metric", bc.METRICS)
@pytest.mark.parametrize("missing", [False, True])
def test_iid_image_comparisons_equal_the_legacy_statistic(metric, missing):
    rng = np.random.RandomState(31)
    if metric == "PG":
        a = rng.binomial(1, 0.8, size=37).astype(float)
        b = rng.binomial(1, 0.6, size=37).astype(float)
    else:
        a = rng.normal(0.4, 1.2, size=37)
        b = rng.normal(0.0, 1.2, size=37)
    if missing:
        a[3] = np.nan
        b[11] = np.inf
    plan = bc.BootstrapPlan([f"image_{i}" for i in range(37)], n_boot=257, seed=13)

    assert plan.compare(a, b, metric) == bc.compare(a, b, metric, n_boot=257, seed=13)


def test_winner_comparisons_use_holm_over_all_pairs_including_invalid_tests():
    from bwgame.evaluation.stats import holm_correction

    labels = ["a", "b", "c", "d"]
    scores = {
        label: {"AL": np.full(3, value)}
        for label, value in zip(labels, [3.0, 2.0, 1.0, np.nan])
    }
    array_label = {id(scores[label]["AL"]): label for label in labels}
    raw = {
        ("a", "b"): 0.009, ("a", "c"): 0.020, ("a", "d"): np.nan,
        ("b", "c"): 0.001, ("b", "d"): np.nan, ("c", "d"): np.nan,
    }

    class PairTests:
        def __init__(self):
            self.called = []

        def compare(self, a, b, metric):
            pair = tuple(sorted((array_label[id(a)], array_label[id(b)])))
            self.called.append(pair)
            return float(a[0] - b[0]), raw[pair], len(a)

    plan = PairTests()
    result = bc.significance_vs_best(scores, labels, "AL", 100, 0, plan=plan)
    expected = dict(zip(raw, holm_correction([p if np.isfinite(p) else 1.0 for p in raw.values()])))

    assert sorted(plan.called) == sorted(raw)
    assert {r["row"] for r in result} == {"b", "c", "d"}
    for row in result:
        assert row["best_row"] == "a"
        assert row["family_size"] == 6
        assert row["p_holm"] == pytest.approx(expected[("a", row["row"])])
    by_label = {r["row"]: r for r in result}
    assert by_label["b"]["significant"]
    assert not by_label["c"]["significant"]
    assert not by_label["d"]["significant"]


@pytest.mark.parametrize(
    "dataset, metadata, expected_mean",
    [
        ("voc", None, 10.0 / 3.0),
        ("imagenet-s", None, 999.0),
        ("voc", {"sampling_unit": "image_cluster", "n_boot": 64, "seed": 0}, 999.0),
        ("voc", {"sampling_unit": "image", "n_boot": 64, "seed": 0}, 10.0 / 3.0),
        ("voc", {"sampling_unit": "image_cluster", "n_boot": 63, "seed": 0}, 10.0 / 3.0),
        ("voc", {"sampling_unit": "image_cluster", "n_boot": 64, "seed": 1}, 10.0 / 3.0),
    ],
)
def test_resultset_only_reuses_ci_cache_with_the_correct_sampling_unit(
    tmp_path, monkeypatch, dataset, metadata, expected_mean,
):
    from reproduce import make_tables

    rows = {"row": {"kind": "row_result", "row_label": "row", "dataset": dataset}}
    ids = np.array(["a__cat", "a__dog", "b__cat"])
    scores = {"row": {"AL": np.array([0.0, 2.0, 8.0])}}
    monkeypatch.setattr(make_tables, "load_row_results", lambda path: rows)
    monkeypatch.setattr(bc, "load_scores", lambda path: (rows, ids, scores))
    (tmp_path / "ci.csv").write_text("row,metric,mean,lo,hi,n\nrow,AL,999,998,1000,3\n")
    if metadata is not None:
        (tmp_path / "bootstrap_metadata.json").write_text(json.dumps(metadata))

    result = make_tables.ResultSet(tmp_path, None, None, n_boot=64, seed=0).ci("row", "AL")

    assert result[0] == pytest.approx(expected_mean)
    if expected_mean != 999.0:
        assert result[1] < 10.0
        assert result[2] < 10.0
    else:
        assert result == (999.0, 998.0, 1000.0)
