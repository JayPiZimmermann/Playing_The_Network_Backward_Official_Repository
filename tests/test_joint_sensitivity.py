"""MaxSensitivity and AvgSensitivity computed in one pass equal the two separate metrics."""
import numpy as np
import torch
import torch.nn as nn

import bwgame.xai  # noqa: F401  (registers the methods)
from bwgame.evaluation import quantus_bridge as qb

CFGS = [
    {"name": "max_sensitivity", "metric_class": "max_sensitivity", "category": "robustness",
     "params": {"nr_samples": 3, "abs": True, "normalise": True}},
    {"name": "avg_sensitivity", "metric_class": "avg_sensitivity", "category": "robustness",
     "params": {"nr_samples": 3, "abs": True, "normalise": True}},
]


def test_joint_equals_separate(monkeypatch):
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(2),
                          nn.Flatten(), nn.Linear(16, 5)).eval()
    x = np.random.RandomState(0).randn(3, 3, 16, 16).astype(np.float32)
    y = np.array([0, 2, 4])
    ef = qb.make_explain_func("raw_gradient", model, torch.device("cpu"), {})
    a = ef(model, x, y)
    monkeypatch.setattr(qb, "JOINT_SENSITIVITY", False)
    sep = qb.evaluate_saliency_maps(model, x, y, a, CFGS, explain_func=ef, device="cpu")
    monkeypatch.setattr(qb, "JOINT_SENSITIVITY", True)
    joint = qb.evaluate_saliency_maps(model, x, y, a, CFGS, explain_func=ef, device="cpu")
    for name in ("max_sensitivity", "avg_sensitivity"):
        np.testing.assert_array_equal(sep[f"{name}_per_image"], joint[f"{name}_per_image"])
    assert np.all(joint["max_sensitivity_per_image"] >= joint["avg_sensitivity_per_image"])
