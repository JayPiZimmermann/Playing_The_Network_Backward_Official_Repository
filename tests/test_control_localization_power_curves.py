import numpy as np

from reproduce.control_localization_power_curves import power_metrics


def test_power_endpoints_and_effective_support():
    a = np.array([[1.0, 0.5], [0.0, 1.0]])
    mask = np.array([[True, False], [False, False]])
    zero = power_metrics(a, mask, "zero")
    inf = power_metrics(a, mask, "inf")
    assert np.isclose(zero["AL"], 1 / 3)
    assert np.isclose(zero["effective_support_fraction"], 3 / 4)
    assert np.isclose(inf["AL"], 1 / 2)
    assert np.isclose(inf["effective_support_fraction"], 1 / 2)
    assert inf["pointing_game_any_max"] == 1


def test_power_can_raise_or_lower_localization_without_rank_change():
    mask = np.array([[True, False], [False, False]])
    raising = np.array([[1.0, 0.7], [0.6, 0.5]])
    lowering = np.array([[0.7, 1.0], [0.6, 0.5]])
    assert power_metrics(raising, mask, 4)["AL"] > power_metrics(raising, mask, 1)["AL"]
    assert power_metrics(lowering, mask, 4)["AL"] < power_metrics(lowering, mask, 1)["AL"]


def test_enrichment_is_one_for_constant_map():
    a = np.ones((2, 3))
    mask = np.array([[True, False, False], [True, False, False]])
    result = power_metrics(a, mask, 2)
    assert np.isclose(result["enrichment"], 1)
    assert np.isclose(result["effective_support_fraction"], 1)
