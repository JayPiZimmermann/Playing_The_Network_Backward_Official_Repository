"""Paired significance tests used by the table scripts.

- Continuous per-image metrics: paired bootstrap test of the mean difference
- Binary hit metric (pointing game): McNemar's test
- Multiple comparisons: Holm correction
"""

from typing import Dict, List

import numpy as np
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Paired bootstrap test
# ---------------------------------------------------------------------------

def paired_bootstrap_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sided paired bootstrap test for difference in means.

    Tests H0: mean(A) == mean(B) against H1: mean(A) != mean(B).

    Args:
        scores_a: Per-example scores for method A.
        scores_b: Per-example scores for method B.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        Dict with: observed_diff, p_value, ci_lower, ci_upper (95% CI).
    """
    assert len(scores_a) == len(scores_b), "Must have paired samples"
    n = len(scores_a)
    diffs = scores_a - scores_b
    observed_diff = float(np.mean(diffs))

    rng = np.random.RandomState(seed)
    boot_diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_diffs[i] = np.mean(diffs[idx])

    # Two-sided p-value: proportion of bootstrap diffs at least as extreme
    centered = boot_diffs - observed_diff
    p_value = float(np.mean(np.abs(centered) >= np.abs(observed_diff)))

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    return {
        "observed_diff": observed_diff,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


# ---------------------------------------------------------------------------
# McNemar's test (for binary metrics like Pointing Game)
# ---------------------------------------------------------------------------

def mcnemar_test(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
) -> Dict[str, float]:
    """McNemar's test for paired binary outcomes.

    Args:
        hits_a: Binary array (1 = correct, 0 = incorrect) for method A.
        hits_b: Binary array (1 = correct, 0 = incorrect) for method B.

    Returns:
        Dict with: statistic, p_value, odds_ratio.
    """
    assert len(hits_a) == len(hits_b)
    hits_a = hits_a.astype(bool)
    hits_b = hits_b.astype(bool)

    # Discordant pairs
    b = np.sum(hits_a & ~hits_b)  # A correct, B wrong
    c = np.sum(~hits_a & hits_b)  # A wrong, B correct

    # McNemar's chi-squared with continuity correction
    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "odds_ratio": 1.0}

    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(sp_stats.chi2.sf(stat, df=1))
    odds = float(b / c) if c > 0 else float("inf")

    return {"statistic": float(stat), "p_value": p, "odds_ratio": odds}


# ---------------------------------------------------------------------------
# Multiple comparison corrections
# ---------------------------------------------------------------------------

def holm_correction(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction.

    Args:
        p_values: List of raw p-values.

    Returns:
        List of corrected p-values (same order as input).
    """
    n = len(p_values)
    if n == 0:
        return []

    # Sort by p-value
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]

    corrected = np.empty(n)
    for i in range(n):
        corrected[i] = sorted_p[i] * (n - i)

    # Enforce monotonicity
    for i in range(1, n):
        corrected[i] = max(corrected[i], corrected[i - 1])

    # Cap at 1.0
    corrected = np.minimum(corrected, 1.0)

    # Unsort
    result = np.empty(n)
    result[order] = corrected
    return result.tolist()
