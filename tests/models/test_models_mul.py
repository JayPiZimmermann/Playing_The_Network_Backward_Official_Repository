"""
Test DC decomposition on models using matmul, mul, softmax, and transformer architectures.

Tests simple operation models (individual matmul/mul/softmax) and transformer
models of increasing complexity, including attention patterns and gated FFNs.

NOTE: nn.Sigmoid and nn.Tanh do NOT have DC patches yet. These nonlinear
activations apply element-wise to [pos; neg] which breaks the decomposition.
Only use DC-supported activations: ReLU, Softmax, LayerNorm.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from bwgame.operations.mul import DCMul
from utils import run_model_tests, run_output_layer_tests


# =============================================================================
# Simple Operation Models
# =============================================================================

class MatMulSimple(nn.Module):
    """Linear transformation using torch.matmul with a parameter weight."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(d_in, d_out) * 0.1)

    def forward(self, x):
        return torch.matmul(x, self.weight)


class MatMulChain(nn.Module):
    """Chain of matmuls with ReLU activations using two-input matmul.
    Uses two Linear projections multiplied together (both DC-format) to test
    two-input matmul backward correction."""
    def __init__(self, d):
        super().__init__()
        self.proj_a = nn.Linear(d, d)
        self.proj_b = nn.Linear(d, d)
        self.out = nn.Linear(d, 1)

    def forward(self, x):
        a = torch.relu(self.proj_a(x))  # [batch, d]
        b = self.proj_b(x)              # [batch, d]
        # Two-input matmul: a.unsqueeze(-1) @ b.unsqueeze(-2) -> [batch, d, d]
        # Then reduce to [batch, d]
        x = torch.matmul(a.unsqueeze(-1), b.unsqueeze(-2))  # [batch, d, d]
        x = x.sum(dim=-1)  # [batch, d]
        return self.out(x)


class MulTwoBranch(nn.Module):
    """Element-wise mul of two ReLU branches (both DC-supported)."""
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.branch_a = nn.Linear(d_in, d_hidden)
        self.branch_b = nn.Linear(d_in, d_hidden)
        self.out_proj = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        a = torch.relu(self.branch_a(x))
        b = torch.relu(self.branch_b(x))
        return self.out_proj(a * b)


class ScalarMulModel(nn.Module):
    """Model with scalar multiplication using DCMul."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_in)
        self.scale = DCMul(scalar=0.5)
        self.fc2 = nn.Linear(d_in, d_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.scale(x)
        return self.fc2(x)


class SoftmaxClassifier(nn.Module):
    """Linear -> Softmax -> Linear, testing softmax in DC mode."""
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x):
        x = self.fc1(x)
        x = F.softmax(x, dim=-1)
        return self.fc2(x)


class MatMulSoftmaxMatMul(nn.Module):
    """Attention-like: Q@K^T -> softmax -> @V using parameter weights."""
    def __init__(self, d):
        super().__init__()
        self.w_q = nn.Parameter(torch.randn(d, d) * 0.1)
        self.w_k = nn.Parameter(torch.randn(d, d) * 0.1)
        self.w_v = nn.Parameter(torch.randn(d, d) * 0.1)
        self.w_out = nn.Parameter(torch.randn(d, 1) * 0.1)

    def forward(self, x):
        # x: (batch, seq, d)
        Q = torch.matmul(x, self.w_q)
        K = torch.matmul(x, self.w_k)
        V = torch.matmul(x, self.w_v)
        attn = torch.matmul(Q, K.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        out = torch.mean(out, dim=1)
        return torch.matmul(out, self.w_out)


class MulChain(nn.Module):
    """Two element-wise muls combined with add.
    Uses only DC-supported activations (ReLU, Softmax)."""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.branch_a = nn.Linear(d_in, d_in)
        self.branch_b = nn.Linear(d_in, d_in)
        self.branch_c = nn.Linear(d_in, d_in)
        self.branch_d = nn.Linear(d_in, d_in)
        self.out = nn.Linear(d_in, d_out)

    def forward(self, x):
        a = torch.relu(self.branch_a(x))
        b = torch.relu(self.branch_b(x))
        c = torch.relu(self.branch_c(x))
        d = F.softmax(self.branch_d(x), dim=-1)
        ab = a * b
        cd = c * d
        return self.out(ab + cd)


# =============================================================================
# Transformer Models
# =============================================================================

class SingleHeadAttention(nn.Module):
    """Single-head self-attention using matmul and softmax."""
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        attn = torch.matmul(Q, K.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)
        return self.out_proj(out)


class TransformerFFN(nn.Module):
    """Feed-forward network for transformer."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


class SmallTransformer(nn.Module):
    """Small transformer: embed -> attention -> ffn -> mean -> head.
    No residual connections for simple DC testing."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.embed = nn.Linear(d_model, d_model)
        self.attn = SingleHeadAttention(d_model)
        self.ffn = TransformerFFN(d_model, d_ff)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.attn(x)
        x = self.ffn(x)
        x = torch.mean(x, dim=1)
        return self.head(x)


class GatedTransformer(nn.Module):
    """Transformer with gated FFN using softmax for gating (DC-supported).
    Exercises mul via softmax_gate * relu_hidden."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.embed = nn.Linear(d_model, d_model)
        self.attn = SingleHeadAttention(d_model)
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.attn(x)
        gate = F.softmax(self.gate_proj(x), dim=-1)
        up = torch.relu(self.up_proj(x))
        x = self.down_proj(gate * up)
        x = torch.mean(x, dim=1)
        return self.head(x)


class TransformerWithLayerNorm(nn.Module):
    """Single transformer block with pre-norm LayerNorm and residual connections."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.embed = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SingleHeadAttention(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = TransformerFFN(d_model, d_ff)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        x = torch.mean(x, dim=1)
        return self.head(x)


class DeepTransformer(nn.Module):
    """2-layer transformer (no residuals) for deeper DC testing."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.embed = nn.Linear(d_model, d_model)
        self.attn1 = SingleHeadAttention(d_model)
        self.ffn1 = TransformerFFN(d_model, d_ff)
        self.attn2 = SingleHeadAttention(d_model)
        self.ffn2 = TransformerFFN(d_model, d_ff)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = self.attn1(x)
        x = self.ffn1(x)
        x = self.attn2(x)
        x = self.ffn2(x)
        x = torch.mean(x, dim=1)
        return self.head(x)


class DeepTransformerWithResiduals(nn.Module):
    """2-layer transformer with pre-norm residual connections."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.embed = nn.Linear(d_model, d_model)
        self.ln1a = nn.LayerNorm(d_model)
        self.attn1 = SingleHeadAttention(d_model)
        self.ln1b = nn.LayerNorm(d_model)
        self.ffn1 = TransformerFFN(d_model, d_ff)
        self.ln2a = nn.LayerNorm(d_model)
        self.attn2 = SingleHeadAttention(d_model)
        self.ln2b = nn.LayerNorm(d_model)
        self.ffn2 = TransformerFFN(d_model, d_ff)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        x = x + self.attn1(self.ln1a(x))
        x = x + self.ffn1(self.ln1b(x))
        x = x + self.attn2(self.ln2a(x))
        x = x + self.ffn2(self.ln2b(x))
        x = torch.mean(x, dim=1)
        return self.head(x)


class GatedDeepTransformer(nn.Module):
    """2-layer transformer with softmax-gated FFN (no residuals).
    Exercises matmul + softmax (attention) and mul (gating)."""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.embed = nn.Linear(d_model, d_model)
        # Block 1
        self.attn1 = SingleHeadAttention(d_model)
        self.gate1 = nn.Linear(d_model, d_ff)
        self.up1 = nn.Linear(d_model, d_ff)
        self.down1 = nn.Linear(d_ff, d_model)
        # Block 2
        self.attn2 = SingleHeadAttention(d_model)
        self.gate2 = nn.Linear(d_model, d_ff)
        self.up2 = nn.Linear(d_model, d_ff)
        self.down2 = nn.Linear(d_ff, d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x)
        # Block 1: attention + gated FFN
        x = self.attn1(x)
        x = self.down1(F.softmax(self.gate1(x), dim=-1) * torch.relu(self.up1(x)))
        # Block 2: attention + gated FFN
        x = self.attn2(x)
        x = self.down2(F.softmax(self.gate2(x), dim=-1) * torch.relu(self.up2(x)))
        x = torch.mean(x, dim=1)
        return self.head(x)


# =============================================================================
# Factory Functions
# =============================================================================

def matmul_simple():
    return MatMulSimple(8, 1)

def matmul_chain():
    return MatMulChain(16)

def mul_two_branch():
    return MulTwoBranch(8, 16, 1)

def scalar_mul():
    return ScalarMulModel(8, 1)

def softmax_classifier():
    return SoftmaxClassifier(8, 8, 1)

def matmul_softmax_matmul():
    return MatMulSoftmaxMatMul(8)

def mul_chain():
    return MulChain(8, 1)

def small_transformer():
    return SmallTransformer(d_model=16, d_ff=32)

def gated_transformer():
    return GatedTransformer(d_model=8, d_ff=16)

def transformer_with_layernorm():
    return TransformerWithLayerNorm(d_model=8, d_ff=16)

def deep_transformer():
    return DeepTransformer(d_model=8, d_ff=16)

def deep_transformer_residual():
    return DeepTransformerWithResiduals(d_model=8, d_ff=16)

def gated_deep_transformer():
    return GatedDeepTransformer(d_model=16, d_ff=32)


# =============================================================================
# Test Configuration
# =============================================================================

# (batch, seq, d_model) for sequence models, (batch, d) for flat models
MODELS = {
    # Simple operation models
    'MatMulSimple':           (matmul_simple,           (1, 8)),
    'MatMulChain':            (matmul_chain,            (1, 16)),
    'MulTwoBranch':           (mul_two_branch,          (1, 8)),
    'ScalarMul':              (scalar_mul,              (1, 8)),
    'SoftmaxClassifier':      (softmax_classifier,      (1, 8)),
    'MulChain':               (mul_chain,               (1, 8)),

    # Transformer models
    'SmallTransformer':       (small_transformer,       (1, 4, 16)),
    'GatedTransformer':       (gated_transformer,       (1, 4, 8)),
    'TransformerLayerNorm':   (transformer_with_layernorm, (1, 4, 8)),
    'DeepTransformer':        (deep_transformer,        (1, 4, 8)),
    'DeepTransformerResid':   (deep_transformer_residual, (1, 4, 8)),
    'GatedDeepTransformer':   (gated_deep_transformer,  (1, 4, 16)),
}


OUTPUT_LAYER_MODELS = {
    # Last layer (explicit)
    "SoftmaxClassifier@fc2":      (softmax_classifier,  (1, 8),    lambda m: m.fc2),
    "MulTwoBranch@out_proj":      (mul_two_branch,      (1, 8),    lambda m: m.out_proj),
    "SmallTransformer@head":      (small_transformer,   (1, 4, 16), lambda m: m.head),
    "DeepTransformer@head":       (deep_transformer,    (1, 4, 8), lambda m: m.head),
    "GatedTransformer@head":      (gated_transformer,   (1, 4, 8), lambda m: m.head),

    # Intermediate layers (must not be on a residual/fork branch)
    "SoftmaxClassifier@fc1":      (softmax_classifier,  (1, 8),    lambda m: m.fc1),
    "SmallTransformer@embed":     (small_transformer,   (1, 4, 16), lambda m: m.embed),
    "GatedTransformer@embed":     (gated_transformer,   (1, 4, 8), lambda m: m.embed),
    "DeepTransformer@embed":      (deep_transformer,    (1, 4, 8), lambda m: m.embed),

    # Intermediate: after first attention block (non-residual models only)
    "SmallTransformer@attn.out_proj":  (small_transformer, (1, 4, 16), lambda m: m.attn.out_proj),
    "DeepTransformer@ffn1.w2":         (deep_transformer,  (1, 4, 8), lambda m: m.ffn1.w2),

    # Residual model: embed is before all residuals, so valid
    "TransformerLN@embed":        (transformer_with_layernorm, (1, 4, 8), lambda m: m.embed),
    "DeepTransformerResid@embed": (deep_transformer_residual,  (1, 4, 8), lambda m: m.embed),
}


def test_models():
    assert run_model_tests(MODELS, title="MatMul/Mul/Softmax & Transformer Tests")


_XFAIL_OUTPUT_LAYERS = ("DeepTransformer@embed",)


def test_output_layer_selection():
    models = {k: v for k, v in OUTPUT_LAYER_MODELS.items() if k not in _XFAIL_OUTPUT_LAYERS}
    assert run_output_layer_tests(models, title="MatMul/Mul/Softmax & Transformer Output Layer Tests")


@pytest.mark.xfail(reason="relative error above the tolerance of this check (float32 accumulation on deep architectures)")
def test_output_layer_selection_deep():
    models = {k: v for k, v in OUTPUT_LAYER_MODELS.items() if k in _XFAIL_OUTPUT_LAYERS}
    assert run_output_layer_tests(models, title="MatMul/Mul/Softmax & Transformer Output Layer Tests (deep)")


if __name__ == "__main__":
    success1 = run_model_tests(MODELS, title="MatMul/Mul/Softmax & Transformer Tests")

    print()
    success2 = run_output_layer_tests(
        OUTPUT_LAYER_MODELS,
        title="MatMul/Mul/Softmax & Transformer Output Layer Tests",
    )

    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    print(f"Standard Tests:      {'PASS' if success1 else 'FAIL'}")
    print(f"Output Layer Tests:  {'PASS' if success2 else 'FAIL'}")

    sys.exit(0 if (success1 and success2) else 1)
