import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from reproduce.control_rank_localization import Q_GRID, rank_localization_metrics


def test_perfect_reverse_and_constant_rankings():
    mask = np.array([[1, 1, 0, 0]], dtype=bool)
    perfect = rank_localization_metrics(np.array([[4., 3., 2., 1.]]), mask)
    reverse = rank_localization_metrics(np.array([[1., 2., 3., 4.]]), mask)
    constant = rank_localization_metrics(np.ones((1, 4)), mask)
    assert perfect["AUROC"] == 1 and perfect["AP"] == 1
    assert reverse["AUROC"] == 0
    assert constant["AUROC"] == .5
    assert constant["AP"] == .5
    assert np.allclose(constant["topq_precision"], .5)
    assert np.allclose(constant["topq_mask_recall"], Q_GRID)
    assert np.allclose(constant["topq_outside_fraction"], Q_GRID)


def test_auc_ap_match_sklearn_with_ties_and_power_invariance():
    score = np.array([0., .1, .1, .4, .8, .8, 1.])
    mask = np.array([0, 1, 0, 1, 0, 1, 1], dtype=bool)
    got = rank_localization_metrics(score, mask)
    assert np.isclose(got["AUROC"], roc_auc_score(mask, score))
    assert np.isclose(got["AP"], average_precision_score(mask, score))
    for power in (.25, .5, 1.5, 3.):
        powered = rank_localization_metrics(score ** power, mask)
        assert np.isclose(powered["AUROC"], got["AUROC"])
        assert np.isclose(powered["AP"], got["AP"])
        assert np.allclose(powered["topq_precision"], got["topq_precision"])


def test_fractional_tie_boundary_is_order_invariant():
    score = np.array([1., .5, .5, .5, 0.])
    mask = np.array([1, 1, 0, 0, 0], dtype=bool)
    a = rank_localization_metrics(score, mask, q_grid=np.array([.4]))
    perm = np.array([0, 3, 1, 2, 4])
    b = rank_localization_metrics(score[perm], mask[perm], q_grid=np.array([.4]))
    assert a["topq_precision"] == b["topq_precision"]
    assert a["topq_mask_recall"] == b["topq_mask_recall"]


def test_invalid_heatmap_and_masks_are_explicit():
    import pytest
    with pytest.raises(ValueError, match="zero heatmap"):
        rank_localization_metrics(np.zeros(4), np.array([1, 0, 0, 0]))
    degenerate = rank_localization_metrics(np.arange(4.), np.zeros(4, dtype=bool))
    assert degenerate["AUROC"] is None
    assert degenerate["median_rank_inside"] is None
    assert "positives=0" in degenerate["invalid_pairwise_reason"]


def test_all_positive_retains_defined_metrics():
    result = rank_localization_metrics(np.arange(1., 5.), np.ones(4, dtype=bool))
    assert result["AUROC"] is None and result["median_rank_outside"] is None
    assert result["AP"] == 1
    assert np.allclose(result["topq_mask_recall"], Q_GRID)
