"""Shared fixtures of the LRP baseline tests: small model factories and weight
initialization."""

import os
import sys

import pytest
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def init_random_biases(model: nn.Module) -> None:
    """Initialize weights/biases for stable gradients."""
    scale_factor = 3.0
    for module in model.modules():
        with torch.no_grad():
            if hasattr(module, "weight") and module.weight is not None:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    fan_in = module.weight.shape[1] if len(module.weight.shape) > 1 else module.weight.shape[0]
                    fan_out = module.weight.shape[0]
                    std = scale_factor * (2.0 / (fan_in + fan_out)) ** 0.5
                    module.weight.normal_(0.0, std)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.weight.uniform_(0.9, 1.1)
            if hasattr(module, "bias") and module.bias is not None:
                if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                    module.bias.uniform_(0.0, 0.1)
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.bias.uniform_(-0.05, 0.05)


# =============================================================================
# Model Factories
# =============================================================================

def MLP_2layer():
    return nn.Sequential(
        nn.Linear(12, 8), nn.ReLU(),
        nn.Linear(8, 4),
    )


def MLP_3layer():
    return nn.Sequential(
        nn.Linear(12, 8), nn.ReLU(),
        nn.Linear(8, 8), nn.ReLU(),
        nn.Linear(8, 4),
    )


def CNN_2conv():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
        nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 2),
    )


def CNN_stride_pool():
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 4),
    )


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    return 42
