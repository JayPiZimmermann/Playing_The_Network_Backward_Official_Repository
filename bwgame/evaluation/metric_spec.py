"""Canonical metric names, short labels and optimisation directions.

The evaluator emits one column per metric config ``name`` (see
``configs/metrics_main_table``).  Every selection / ranking / table script
must use :data:`METRIC_DIRECTION` (or :func:`metric_direction`) so that the
direction of a metric is defined in exactly one place.  Unknown metric
names raise ``KeyError`` instead of silently defaulting to a direction.
"""

from typing import Dict

# Column name emitted by the evaluator -> 'max' (higher is better) or
# 'min' (lower is better).
METRIC_DIRECTION: Dict[str, str] = {
    # Localisation (higher is better)
    "attribution_localisation": "max",
    "pointing_game": "max",
    "top_k_intersection": "max",
    # Faithfulness (lower is better: logit AUC under most-relevant-first removal)
    "pixel_flipping_abs_5pct": "min",
    "pixel_flipping_abs_20pct": "min",
    "selectivity": "min",
    # Robustness (lower is better)
    "max_sensitivity": "min",
    "avg_sensitivity": "min",
}

# Short labels used in the paper tables -> evaluator column names.
SHORT_TO_METRIC: Dict[str, str] = {
    "AL": "attribution_localisation",
    "PG": "pointing_game",
    "TK": "top_k_intersection",
    "PF5": "pixel_flipping_abs_5pct",
    "PF20": "pixel_flipping_abs_20pct",
    "Sel": "selectivity",
    "MaxS": "max_sensitivity",
    "AvgS": "avg_sensitivity",
}

# Order used in every table.
ALL_SHORT = ("AL", "PG", "TK", "PF5", "PF20", "Sel", "MaxS", "AvgS")


def metric_direction(name: str) -> str:
    """Return 'max' or 'min' for an evaluator column or a short label.

    Raises:
        KeyError: if the metric has no registered direction.
    """
    key = SHORT_TO_METRIC.get(name, name)
    try:
        return METRIC_DIRECTION[key]
    except KeyError:
        raise KeyError(
            f"No optimisation direction registered for metric {name!r}. "
            f"Add it to bwgame.evaluation.metric_spec.METRIC_DIRECTION "
            f"(known: {sorted(METRIC_DIRECTION)})."
        ) from None
