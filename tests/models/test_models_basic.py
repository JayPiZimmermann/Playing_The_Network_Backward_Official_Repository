"""
Test basic layer functionality with realistic models (all with 1 output dim).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch.nn as nn
from utils import run_model_tests, run_output_layer_tests


# =============================================================================
# Model Definitions - All end with 1 output dimension
# =============================================================================

# Linear models
def linear_shallow():
    return nn.Sequential(
        nn.Linear(10, 1)
    )

def linear_2layer():
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )

def linear_3layer():
    return nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1)
    )

def linear_5layer():
    return nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )

def linear_with_bn():
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    model.eval()
    return model

def linear_with_dropout():
    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, 1)
    )
    model.eval()
    return model


# Conv models
def conv_shallow():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 1)
    )
    model.eval()
    return model

def conv_3layer():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 1)
    )
    model.eval()
    return model

def conv_with_bn():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, 1)
    )
    model.eval()
    return model

def conv_with_maxpool():
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 1)
    )
    model.eval()
    return model

def conv_deep():
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(128, 1)
    )
    model.eval()
    return model


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    # Linear models
    'Linear_shallow': (linear_shallow, (2, 10)),
    'Linear_2layer': (linear_2layer, (2, 10)),
    'Linear_3layer': (linear_3layer, (2, 10)),
    'Linear_5layer': (linear_5layer, (2, 20)),
    'Linear_with_BN': (linear_with_bn, (4, 20)),
    'Linear_with_Dropout': (linear_with_dropout, (4, 20)),

    # Conv models
    'Conv_shallow': (conv_shallow, (2, 3, 16, 16)),
    'Conv_3layer': (conv_3layer, (2, 3, 16, 16)),
    'Conv_with_BN': (conv_with_bn, (2, 3, 16, 16)),
    'Conv_with_MaxPool': (conv_with_maxpool, (2, 3, 32, 32)),
    'Conv_deep': (conv_deep, (2, 3, 32, 32)),
}


# Manual output_layer selection tests
OUTPUT_LAYER_MODELS = {
    # Last layer (explicit)
    "Linear_2layer@Linear[2]":    (linear_2layer,   (2, 10),          lambda m: m[2]),
    "Linear_3layer@Linear[4]":    (linear_3layer,   (2, 10),          lambda m: m[4]),
    "Conv_shallow@Linear[4]":     (conv_shallow,    (2, 3, 16, 16),   lambda m: m[4]),
    "Conv_3layer@Linear[8]":      (conv_3layer,     (2, 3, 16, 16),   lambda m: m[8]),

    # Intermediate layers
    "Linear_2layer@Linear[0]":    (linear_2layer,   (2, 10),          lambda m: m[0]),
    "Linear_3layer@Linear[0]":    (linear_3layer,   (2, 10),          lambda m: m[0]),
    "Linear_3layer@Linear[2]":    (linear_3layer,   (2, 10),          lambda m: m[2]),
    "Conv_shallow@Conv[0]":       (conv_shallow,    (2, 3, 16, 16),   lambda m: m[0]),
    "Conv_3layer@Conv[0]":        (conv_3layer,     (2, 3, 16, 16),   lambda m: m[0]),
    "Conv_3layer@Conv[2]":        (conv_3layer,     (2, 3, 16, 16),   lambda m: m[2]),
}


def test_models():
    assert run_model_tests(MODELS, title="Basic Layer Tests")


def test_output_layer_selection():
    assert run_output_layer_tests(OUTPUT_LAYER_MODELS, title="Basic Layers Manual Output Layer Selection Tests")


if __name__ == "__main__":
    success1 = run_model_tests(MODELS, title="Basic Layer Tests")

    print()
    success2 = run_output_layer_tests(
        OUTPUT_LAYER_MODELS,
        title="Basic Layers Manual Output Layer Selection Tests",
    )

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(f"Standard Tests:      {'PASS' if success1 else 'FAIL'}")
    print(f"Output Layer Tests:  {'PASS' if success2 else 'FAIL'}")

    sys.exit(0 if success1 else 1)
