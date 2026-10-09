"""
Test DC decomposition on CNN models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
from utils import run_model_tests, run_output_layer_tests


# =============================================================================
# Model Definitions
# =============================================================================

def CNN_2conv():
    return nn.Sequential(
        nn.Conv2d(1, 2, 2), nn.ReLU(),
        nn.Conv2d(2, 4, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 1)
    )

def CNN_2conv_maxpool():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, stride=2), nn.ReLU(),
        nn.Conv2d(4, 8, 2), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )

def CNN_deep_relu():
    return nn.Sequential(
        nn.Conv2d(1, 4, 2), nn.ReLU(),
        nn.Conv2d(4, 8, 2), nn.ReLU(),
        nn.Conv2d(8, 4, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 1)
    )

def CNN_inner_maxpool():
    return nn.Sequential(
        nn.Conv2d(1, 4, 2), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4, 8, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )

def CNN_inner_avgpool():
    return nn.Sequential(
        nn.Conv2d(1, 4, 2), nn.ReLU(),
        nn.AvgPool2d(2),
        nn.Conv2d(4, 8, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )

def CNN_BN_Relu():
    model = nn.Sequential(
        nn.Conv2d(1, 4, 2),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.Conv2d(4, 8, 3),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )
    model.eval()
    return model

def CNN_stride_padding():
    return nn.Sequential(
        nn.Conv2d(1, 4, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(4, 8, 2, stride=2), nn.ReLU(),
        nn.Conv2d(8, 8, 2, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(8, 4, 2), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, 1)
    )

def CNN_Pointwise():
    model = nn.Sequential(
        nn.Conv2d(1, 16, 3), nn.ReLU(),
        nn.Conv2d(16, 8, 1),  # Pointwise conv
        nn.BatchNorm2d(8), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )
    model.eval()
    return model

def CNN_VGG_Block():
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 1)
    )
    model.eval()
    return model


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    "CNN_2conv": (CNN_2conv, (1, 1, 8, 8)),
    "CNN_2conv_maxpool": (CNN_2conv_maxpool, (1, 1, 16, 16)),
    "CNN_deep_relu": (CNN_deep_relu, (1, 1, 8, 8)),
    "CNN_inner_maxpool": (CNN_inner_maxpool, (1, 1, 8, 8)),
    "CNN_inner_avgpool": (CNN_inner_avgpool, (1, 1, 8, 8)),
    "CNN_BN_Relu": (CNN_BN_Relu, (1, 1, 8, 8)),
    "CNN_stride_padding": (CNN_stride_padding, (1, 1, 32, 32)),
    "CNN_Pointwise": (CNN_Pointwise, (1, 1, 16, 16)),
    "CNN_VGG_Block": (CNN_VGG_Block, (1, 1, 16, 16)),
}


# Manual output_layer selection tests
OUTPUT_LAYER_MODELS = {
    # Last layer (explicit)
    "CNN_2conv@Linear[6]":        (CNN_2conv,       (1, 1, 8, 8),   lambda m: m[6]),
    "CNN_BN_Relu@Linear[8]":      (CNN_BN_Relu,     (1, 1, 8, 8),   lambda m: m[8]),
    "CNN_VGG_Block@Linear[9]":    (CNN_VGG_Block,   (1, 1, 16, 16), lambda m: m[9]),
    "CNN_deep_relu@Linear[8]":    (CNN_deep_relu,   (1, 1, 8, 8),   lambda m: m[8]),

    # Intermediate layers
    "CNN_2conv@Conv[0]":          (CNN_2conv,       (1, 1, 8, 8),   lambda m: m[0]),
    "CNN_2conv@Conv[2]":          (CNN_2conv,       (1, 1, 8, 8),   lambda m: m[2]),
    "CNN_BN_Relu@Conv[0]":        (CNN_BN_Relu,     (1, 1, 8, 8),   lambda m: m[0]),
    "CNN_BN_Relu@Conv[3]":        (CNN_BN_Relu,     (1, 1, 8, 8),   lambda m: m[3]),
    "CNN_deep_relu@Conv[0]":      (CNN_deep_relu,   (1, 1, 8, 8),   lambda m: m[0]),
    "CNN_deep_relu@Conv[4]":      (CNN_deep_relu,   (1, 1, 8, 8),   lambda m: m[4]),
}


def test_models():
    assert run_model_tests(MODELS, title="CNN Model Tests")


def test_output_layer_selection():
    assert run_output_layer_tests(OUTPUT_LAYER_MODELS, title="CNN Manual Output Layer Selection Tests")


if __name__ == "__main__":
    success1 = run_model_tests(MODELS, title="CNN Model Tests")

    print()
    success2 = run_output_layer_tests(
        OUTPUT_LAYER_MODELS,
        title="CNN Manual Output Layer Selection Tests",
    )

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(f"Standard Tests:      {'PASS' if success1 else 'FAIL'}")
    print(f"Output Layer Tests:  {'PASS' if success2 else 'FAIL'}")

    sys.exit(0 if success1 else 1)
