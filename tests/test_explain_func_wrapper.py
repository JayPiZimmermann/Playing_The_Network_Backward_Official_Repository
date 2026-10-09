"""The robustness explain_func must use the same wrapper as the evaluated row.

For an equivariant-averaging (RG+Eq) row, MaxS / AvgS must explain the
perturbed inputs with the equivariant average, not with the plain inner method.
"""
import random

import numpy as np
import torch
import torch.nn as nn

import bwgame.xai  # noqa: F401  (registers the methods)
from bwgame.evaluation.quantus_bridge import make_explain_func
from bwgame.xai.wrappers import equivariant_average


def _tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(4), nn.Flatten(), nn.Linear(64, 5),
    ).eval()


EQ = {"equivariant_averaging": True, "equivariant_n_samples": 4,
      "equivariant_max_angle": 10.0, "equivariant_max_shift_frac": 0.05,
      "equivariant_flip_prob": 0.5}


def test_explain_func_uses_equivariant_wrapper():
    model = _tiny_model()
    x = np.random.RandomState(0).randn(1, 3, 16, 16).astype(np.float32)
    y = np.array([2])

    random.seed(123)
    got = make_explain_func("raw_gradient", model, torch.device("cpu"), dict(EQ))(model, x, y)

    random.seed(123)
    ref = equivariant_average("raw_gradient", model, torch.device("cpu"),
                              torch.from_numpy(x), torch.from_numpy(y), None,
                              n_samples=4, max_angle=10.0, max_shift_frac=0.05,
                              flip_prob=0.5, method_params={})["input"]
    np.testing.assert_allclose(got, np.asarray(ref).reshape(got.shape), rtol=1e-5, atol=1e-7)

    plain = make_explain_func("raw_gradient", model, torch.device("cpu"), {})(model, x, y)
    assert not np.allclose(got, plain), "wrapper row must not be explained with the plain method"
