"""
Test DC decomposition on MLP models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.nn as nn
from utils import run_model_tests, run_output_layer_tests


# =============================================================================
# Model Definitions
# =============================================================================

def MLP_1layer():
    return nn.Sequential(nn.Linear(3, 2))

def MLP_1layer_relu():
    return nn.Sequential(nn.Linear(3, 2), nn.ReLU())

def MLP_2layer():
    return nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 1))

def MLP_3layer():
    return nn.Sequential(nn.Linear(3, 2), nn.ReLU(), nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 1))

def MLP_4layer():
    return nn.Sequential(
        nn.Linear(3, 4), nn.ReLU(),
        nn.Linear(4, 3), nn.ReLU(),
        nn.Linear(3, 2), nn.ReLU(),
        nn.Linear(2, 1)
    )

def MLP_5layer():
    return nn.Sequential(
        nn.Linear(3, 4), nn.ReLU(),
        nn.Linear(4, 4), nn.ReLU(),
        nn.Linear(4, 3), nn.ReLU(),
        nn.Linear(3, 2), nn.ReLU(),
        nn.Linear(2, 1)
    )


# =============================================================================
# Test Configuration
# =============================================================================

MODELS = {
    "MLP_1layer": (MLP_1layer, (1, 3)),
    "MLP_1layer_relu": (MLP_1layer_relu, (1, 3)),
    "MLP_2layer": (MLP_2layer, (1, 3)),
    "MLP_3layer": (MLP_3layer, (1, 3)),
    "MLP_4layer": (MLP_4layer, (1, 3)),
    "MLP_5layer": (MLP_5layer, (1, 3)),
}


# Manual output_layer selection tests
# For Sequential models: m[i] selects the i-th module
OUTPUT_LAYER_MODELS = {
    # Last layer (explicit) - should match auto-detection
    "MLP_1layer@Linear[0]":     (MLP_1layer,      (1, 3), lambda m: m[0]),
    "MLP_1layer_relu@ReLU[1]":  (MLP_1layer_relu,  (1, 3), lambda m: m[1]),
    "MLP_2layer@Linear[2]":     (MLP_2layer,       (1, 3), lambda m: m[2]),
    "MLP_3layer@Linear[4]":     (MLP_3layer,       (1, 3), lambda m: m[4]),
    "MLP_5layer@Linear[8]":     (MLP_5layer,       (1, 3), lambda m: m[8]),

    # Intermediate layers
    "MLP_2layer@Linear[0]":     (MLP_2layer,       (1, 3), lambda m: m[0]),
    "MLP_3layer@Linear[0]":     (MLP_3layer,       (1, 3), lambda m: m[0]),
    "MLP_3layer@Linear[2]":     (MLP_3layer,       (1, 3), lambda m: m[2]),
    "MLP_5layer@Linear[0]":     (MLP_5layer,       (1, 3), lambda m: m[0]),
    "MLP_5layer@Linear[4]":     (MLP_5layer,       (1, 3), lambda m: m[4]),
}


def test_models():
    assert run_model_tests(MODELS, title="MLP Model Tests")


def test_output_layer_selection():
    assert run_output_layer_tests(OUTPUT_LAYER_MODELS, title="MLP Manual Output Layer Selection Tests")


if __name__ == "__main__":
    success1 = run_model_tests(MODELS, title="MLP Model Tests")

    print()
    success2 = run_output_layer_tests(
        OUTPUT_LAYER_MODELS,
        title="MLP Manual Output Layer Selection Tests",
    )

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(f"Standard Tests:      {'PASS' if success1 else 'FAIL'}")
    print(f"Output Layer Tests:  {'PASS' if success2 else 'FAIL'}")

    sys.exit(0 if success1 else 1)
