"""
Test DC decomposition on ResNet models with residual connections.

These models use torch.relu() and + operations which get replaced by
prepare_model_for_dc().
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from utils import run_model_tests, run_output_layer_tests


# =============================================================================
# Model Definitions
# =============================================================================

class ResBlock(nn.Module):
    """Standard ResBlock using torch.relu() and + (standard PyTorch)."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity  # Residual connection
        out = torch.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResBlockNoBN(nn.Module):
    """ResBlock without BatchNorm."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        identity = x
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        out = torch.relu(out + identity)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResBlockInternal(nn.Module):
    """Internal ResBlock (no final fc) for use in SimpleResNet."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = torch.relu(out)
        return out


class SimpleResNet(nn.Module):
    """Simple ResNet with multiple blocks."""
    def __init__(self, num_blocks=2, channels=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.blocks = nn.Sequential(*[ResBlockInternal(channels) for _ in range(num_blocks)])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.stem(x)
        x = torch.relu(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResBlockMinimal(nn.Module):
    """Minimal ResBlock without BatchNorm and pools - just conv+relu+add+relu."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        identity = x
        out = torch.relu(self.conv1(x))
        out = torch.relu(out + identity)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResBlockWithoutFinalRelu(nn.Module):
    """Residual block without the ReLU after the addition."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        identity = x
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity  # No ReLU after addition
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class IdentityBlock(nn.Module):
    """Just identity connection without any convolutions."""
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        identity = x
        out = torch.relu(x + identity)  # Just add input to itself and ReLU
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SingleResNet(nn.Module):
    """ResNet with just one ResBlockInternal to test if it behaves like SimpleResNet."""
    def __init__(self, channels=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        self.block = ResBlockInternal(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.stem(x)
        x = torch.relu(x)
        x = self.block(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResBlockWithStem(nn.Module):
    """ResBlock with a stem like SimpleResNet to test if stem affects caching."""
    def __init__(self, channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),  # 1x1 conv as minimal stem
            nn.BatchNorm2d(channels)
        )
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        x = self.stem(x)
        x = torch.relu(x)
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = torch.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResBlockInputModified(nn.Module):
    """Residual block whose identity branch is a 1x1 convolution of the input."""
    def __init__(self, channels):
        super().__init__()
        self.input_transform = nn.Conv2d(channels, channels, 1, bias=False)  # 1x1 to modify input
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x):
        identity = self.input_transform(x)  # Modify input before using as identity
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        out = torch.relu(out + identity)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResBlock_single():
    model = ResBlock(16)
    model.eval()
    return model

def ResBlockNoBN_single():
    model = ResBlockNoBN(16)
    model.eval()
    return model

def ResNet_2blocks():
    model = SimpleResNet(num_blocks=2, channels=16)
    model.eval()
    return model

def ResNet_3blocks():
    model = SimpleResNet(num_blocks=3, channels=16)
    model.eval()
    return model

def ResBlockMinimal_single():
    model = ResBlockMinimal(16)
    model.eval()
    return model

def ResBlockWithoutFinalRelu_single():
    model = ResBlockWithoutFinalRelu(16)
    model.eval()
    return model

def IdentityBlock_single():
    model = IdentityBlock(16)
    model.eval()
    return model

def SingleResNet_single():
    model = SingleResNet(16)
    model.eval()
    return model

def ResBlockWithStem_single():
    model = ResBlockWithStem(16)
    model.eval()
    return model

def ResNet_1block():
    model = SimpleResNet(num_blocks=1, channels=16)
    model.eval()
    return model

def ResBlockInputModified_single():
    model = ResBlockInputModified(16)
    model.eval()
    return model


# =============================================================================
# Test Configuration
# =============================================================================

# Residual-block variants
MODELS = {
    # ReLU after the addition, raw input as identity
    "ResBlockNoBN": (ResBlockNoBN_single, (1, 16, 8, 8)),
    "IdentityBlock": (IdentityBlock_single, (1, 16, 8, 8)),

    # no ReLU after the addition
    "ResBlockWithoutFinalRelu": (ResBlockWithoutFinalRelu_single, (1, 16, 8, 8)),

    # with a preprocessing stem
    "SingleResNet": (SingleResNet_single, (1, 3, 16, 16)),
    "ResNet_1block": (ResNet_1block, (1, 3, 16, 16)),
    "ResBlockWithStem": (ResBlockWithStem_single, (1, 16, 8, 8)),

    # 1x1-transformed input as identity
    "ResBlockInputModified": (ResBlockInputModified_single, (1, 16, 8, 8)),
}


# Manual output_layer selection tests
OUTPUT_LAYER_MODELS = {
    # Last layer (explicit) - fc is the auto-detected last layer
    "ResBlockWithoutFinalRelu@fc":  (ResBlockWithoutFinalRelu_single, (1, 16, 8, 8), lambda m: m.fc),
    "SingleResNet@fc":              (SingleResNet_single, (1, 3, 16, 16), lambda m: m.fc),
    "IdentityBlock@fc":             (IdentityBlock_single, (1, 16, 8, 8), lambda m: m.fc),

    # Intermediate layers (must be after residual connections to avoid boundary crossing)
    "ResBlockWithoutFinalRelu@pool": (ResBlockWithoutFinalRelu_single, (1, 16, 8, 8), lambda m: m.pool),
    "SingleResNet@stem[0]":           (SingleResNet_single, (1, 3, 16, 16), lambda m: m.stem[0]),
    "IdentityBlock@pool":             (IdentityBlock_single, (1, 16, 8, 8), lambda m: m.pool),
}


def test_models():
    assert run_model_tests(MODELS, title="ResNet Model Tests")


def test_output_layer_selection():
    assert run_output_layer_tests(OUTPUT_LAYER_MODELS, title="ResNet Manual Output Layer Selection Tests")


if __name__ == "__main__":
    success1 = run_model_tests(MODELS, title="ResNet Model Tests")

    print()
    success2 = run_output_layer_tests(
        OUTPUT_LAYER_MODELS,
        title="ResNet Manual Output Layer Selection Tests",
    )

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(f"Standard Tests:      {'PASS' if success1 else 'FAIL'}")
    print(f"Output Layer Tests:  {'PASS' if success2 else 'FAIL'}")

    sys.exit(0 if success1 else 1)
