#!/usr/bin/env python3
"""Test DC decomposition on U-Net like models with skip connections."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from tests.utils.dc_tester import test_model_with_loss as _test_model_with_loss

def smooth_l1_loss(output, target):
    """Smooth L1 / Huber loss."""
    return F.smooth_l1_loss(output, target)

def mse_loss(output, target):
    """Mean squared error loss."""
    return F.mse_loss(output, target)

class SimpleUNet(nn.Module):
    """Simplified U-Net with skip connections - minimal version of DeepSegNet."""

    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 32 + 32 from skip
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)

        # Decoder with skip
        u1 = self.up1(e2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return d1

class UNetWithBatchNorm(nn.Module):
    """U-Net with BatchNorm in every encoder/decoder block."""

    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 32 + 32 from skip
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)

        # Decoder with skip
        u1 = self.up1(e2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        return d1


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls,loss_fn,loss_name", [
    (SimpleUNet, mse_loss, "mse"),
    (SimpleUNet, smooth_l1_loss, "smooth_l1"),
    (UNetWithBatchNorm, mse_loss, "mse"),
    (UNetWithBatchNorm, smooth_l1_loss, "smooth_l1"),
])
def test_unet_like_dc_decomposition(model_cls, loss_fn, loss_name):
    torch.manual_seed(42)
    model = model_cls(in_channels=3, num_classes=5)
    x = torch.randn(2, 3, 32, 32)
    target = torch.randn(2, 5, 32, 32)
    name = f"{model_cls.__name__}_{loss_name}"
    result = _test_model_with_loss(model, x, target, loss_fn, name)
    assert result.success, (
        f"{name} failed: fwd_rel={result.fwd_rel_error:.2e}, "
        f"bwd_rel={result.bwd_rel_error:.2e}, err={result.error_message}"
    )


# ---------------------------------------------------------------------------
# Standalone script
# ---------------------------------------------------------------------------

def main():
    print("Testing U-Net like models with skip connections")
    print("=" * 60)

    models = {
        "SimpleUNet_MSE": (
            SimpleUNet(in_channels=3, num_classes=5),
            torch.randn(2, 3, 32, 32),
            torch.randn(2, 5, 32, 32),
            mse_loss,
        ),
        "SimpleUNet_SmoothL1": (
            SimpleUNet(in_channels=3, num_classes=5),
            torch.randn(2, 3, 32, 32),
            torch.randn(2, 5, 32, 32),
            smooth_l1_loss,
        ),
        "UNetBN_MSE": (
            UNetWithBatchNorm(in_channels=3, num_classes=5),
            torch.randn(2, 3, 32, 32),
            torch.randn(2, 5, 32, 32),
            mse_loss,
        ),
        "UNetBN_SmoothL1": (
            UNetWithBatchNorm(in_channels=3, num_classes=5),
            torch.randn(2, 3, 32, 32),
            torch.randn(2, 5, 32, 32),
            smooth_l1_loss,
        ),
    }

    results = []

    for name, (model, x, target, loss_fn) in models.items():
        print(f"\nTesting {name}...")

        try:
            result = _test_model_with_loss(model, x, target, loss_fn, name)
            results.append(result)
            print(f"  Result: {'PASS' if result.success else 'FAIL'}")
            print(f"    Fwd: rel={result.fwd_rel_error:.2e}")
            print(f"    Bwd: rel={result.bwd_rel_error:.2e}")
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for r in results if r.success)
    total = len(results)
    print(f"Passed: {passed}/{total}")

if __name__ == "__main__":
    main()
