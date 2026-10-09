"""Cascading randomisation: single steps equal the same steps of the full cascade.

Attribution sampling (SmoothGrad here) must not advance the random stream
that re-initialises the layers, so a step computed on its own sees the same
randomised model as inside the full cascade.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn

import bwgame.xai  # noqa: F401  (registers the methods)
from bwgame.evaluation.metrics.sanity_check import iter_cascading_randomization


def _net():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 5)).eval()


@pytest.mark.parametrize("method,params", [("smooth_grad", {"nt_samples": 3, "stdevs": 0.2}),
                                           ("raw_gradient", {})])
def test_single_step_equals_full_cascade(method, params):
    x = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(1))
    y = torch.tensor([1, 3])
    kw = dict(method_params=params, similarity_metrics=("spearman_abs",), seed=7,
              device=torch.device("cpu"), images=x, labels=y, method_name=method)
    full = {s.step_idx: s.similarities["spearman_abs"]
            for s in iter_cascading_randomization(_net(), **kw)}
    last = max(full)
    only = [s for s in iter_cascading_randomization(
        _net(), compute_step=lambda k, _layer: k == last, **kw)]
    assert [s.step_idx for s in only] == [last]
    np.testing.assert_array_equal(only[0].similarities["spearman_abs"], full[last])
