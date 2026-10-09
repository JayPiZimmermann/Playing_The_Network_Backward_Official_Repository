#!/usr/bin/env python3
"""Test DC decomposition on models with ConvTranspose2d (deconvolution) layers."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
from tests.utils.dc_tester import test_model as _test_model

class SimpleDecoder(nn.Module):
    """Simple decoder with ConvTranspose2d layers."""

    def __init__(self, in_channels=128, out_channels=3):
        super().__init__()
        # Encoder (downsampling)
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
        )

        # Decoder (upsampling with deconvolution)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 64, 3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, 3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x

class SimpleConvTransposeOnly(nn.Module):
    """Very simple model with just ConvTranspose2d."""

    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1)
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        return x

class EncoderDecoder(nn.Module):
    """Encoder-decoder with BatchNorm to test interaction."""

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2)

        # Decoder
        self.up = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 1),
        )

    def forward(self, x):
        # Encoder
        x = self.enc1(x)
        x = self.pool(x)

        # Decoder
        x = self.up(x)
        x = self.dec(x)
        return x


# ---------------------------------------------------------------------------
# Pytest tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_cls,input_shape,name", [
    (SimpleConvTransposeOnly, (1, 16, 4, 4), "SimpleConvTransposeOnly"),
    (lambda: SimpleDecoder(in_channels=3), (1, 3, 16, 16), "SimpleDecoder"),
    (EncoderDecoder, (1, 3, 16, 16), "EncoderDecoder"),
])
def test_deconv_dc_decomposition(model_cls, input_shape, name):
    torch.manual_seed(42)
    model = model_cls() if callable(model_cls) else model_cls
    x = torch.randn(*input_shape)
    result = _test_model(model, x, name)
    assert result.success, (
        f"{name} failed: fwd_rel={result.fwd_rel_error:.2e}, "
        f"bwd_rel={result.bwd_rel_error:.2e}, err={result.error_message}"
    )


# ---------------------------------------------------------------------------
# Standalone script
# ---------------------------------------------------------------------------

def main():
    print("Testing ConvTranspose2d with DC decomposition")
    print("=" * 60)

    models = {
        "SimpleConvTransposeOnly": (
            SimpleConvTransposeOnly(),
            torch.randn(1, 16, 4, 4)
        ),
        "SimpleDecoder": (
            SimpleDecoder(in_channels=3),
            torch.randn(1, 3, 16, 16)
        ),
        "EncoderDecoder": (
            EncoderDecoder(),
            torch.randn(1, 3, 16, 16)
        ),
    }

    results = []

    for name, (model, x) in models.items():
        print(f"\nTesting {name}...")
        print(f"Input shape: {x.shape}")

        with torch.no_grad():
            out = model(x)
            print(f"Output shape: {out.shape}")

        try:
            result = _test_model(model, x, name)
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
