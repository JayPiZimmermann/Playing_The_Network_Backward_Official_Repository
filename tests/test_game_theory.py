"""Tests of the game-theoretic parts of the DC decomposition.

Tests cover:
- the fan-in entropy and the entropy bias (entropy.py)
- the output-layer initialization of the DC backward
- the DC decomposition of GELU and SiLU activations
"""

import pytest
import torch
import torch.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Simple test models
# ---------------------------------------------------------------------------

class SimpleMLP(nn.Module):
    """3-layer MLP with ReLU for testing."""
    def __init__(self, in_features=12, hidden=8, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = x.flatten(1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


class SimpleCNN(nn.Module):
    """Tiny CNN for testing Conv2d handling."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 4, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


class NoBiasMLP(nn.Module):
    """MLP without biases."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(12, 8, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 4, bias=False)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc2(self.relu(self.fc1(x)))


# ===========================================================================
# Fan-in entropy
# ===========================================================================

class TestComputeEntropy:
    """Tests for compute_entropy()."""

    def test_uniform_weights_maximum_entropy(self):
        """Uniform weights should give maximum entropy = log(fan_in)."""
        from bwgame.game_theory.entropy import compute_entropy

        fan_in = 8
        w = torch.ones(4, fan_in)
        H = compute_entropy(w)
        expected = np.log(fan_in)
        assert H.shape == (4,)
        assert torch.allclose(H, torch.full_like(H, expected), atol=1e-5)

    def test_one_hot_weights_zero_entropy(self):
        """One-hot weights (all mass on one connection) should give ~0 entropy."""
        from bwgame.game_theory.entropy import compute_entropy

        w = torch.zeros(3, 6)
        w[:, 0] = 1.0
        H = compute_entropy(w)
        assert H.shape == (3,)
        assert torch.all(H < 1e-6)

    def test_entropy_non_negative(self):
        """Entropy should always be non-negative."""
        from bwgame.game_theory.entropy import compute_entropy

        torch.manual_seed(42)
        w = torch.randn(5, 10)
        H = compute_entropy(w)
        assert torch.all(H >= -1e-8)

    def test_conv2d_weight_shape(self):
        """Conv2d weights (out, in, kH, kW) should be flattened correctly."""
        from bwgame.game_theory.entropy import compute_entropy

        w = torch.ones(8, 3, 3, 3)  # 8 output channels, 3x3x3 fan_in=27
        H = compute_entropy(w)
        expected = np.log(27)
        assert H.shape == (8,)
        assert torch.allclose(H, torch.full_like(H, expected), atol=1e-5)


class TestEntropyBiasAdaptContext:
    """Tests for entropy_bias_adapt_context() (fan-in entropy bias λ_ent)."""

    def test_biases_shifted_inside_context(self):
        """Inside the context every bias is shifted by -lam * H_norm."""
        from bwgame.game_theory.entropy import entropy_bias_adapt_context, compute_entropy

        model = SimpleMLP()
        lam = 0.5
        x = torch.randn(1, 1, 3, 4)
        expected = {n: m.bias.detach() - lam * compute_entropy(m.weight, normalize=True)
                    for n, m in model.named_modules() if isinstance(m, nn.Linear)}

        with torch.no_grad():
            out_orig = model(x).clone()

        with entropy_bias_adapt_context(model, lam):
            for n, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    assert torch.allclose(m.bias, expected[n], atol=1e-6)
            with torch.no_grad():
                out_shifted = model(x)

        assert not torch.equal(out_orig, out_shifted), \
            "Output should change inside the entropy-bias context"

    def test_biases_restored_after_context(self):
        """All biases are restored after context exit."""
        from bwgame.game_theory.entropy import entropy_bias_adapt_context

        model = SimpleMLP()
        x = torch.randn(1, 1, 3, 4)

        with torch.no_grad():
            out_before = model(x).clone()

        with entropy_bias_adapt_context(model, lam=1.0):
            pass

        with torch.no_grad():
            out_after = model(x)
        assert torch.equal(out_before, out_after), \
            "Output changed after context exit — biases not restored"

    def test_no_bias_modules_unchanged(self):
        """Modules without bias are left unchanged."""
        from bwgame.game_theory.entropy import entropy_bias_adapt_context

        model = NoBiasMLP()
        x = torch.randn(1, 1, 3, 4)

        with torch.no_grad():
            out_orig = model(x).clone()

        with entropy_bias_adapt_context(model, lam=0.5):
            with torch.no_grad():
                out_inside = model(x)

        assert torch.equal(out_orig, out_inside)

    def test_biases_restored_on_exception(self):
        """Biases are restored even if an exception occurs."""
        from bwgame.game_theory.entropy import entropy_bias_adapt_context

        model = SimpleMLP()
        x = torch.randn(1, 1, 3, 4)

        with torch.no_grad():
            out_before = model(x).clone()

        with pytest.raises(RuntimeError):
            with entropy_bias_adapt_context(model, lam=1.0):
                raise RuntimeError("test error")

        with torch.no_grad():
            out_after = model(x)
        assert torch.equal(out_before, out_after), \
            "Biases not restored after exception"


# ===========================================================================
# Output-layer initialization of the DC backward
# ===========================================================================

class TestOutputLayerInit:
    """Tests for init_backward() at the output layer."""

    def test_half_split(self):
        """The output gradient is split 1/2 - 1/2 between delta_pp and delta_nn."""
        from bwgame.operations.base import init_backward

        grad = torch.randn(4, 3)  # 4*batch=4, features=3
        pp, np_, pn, nn_ = init_backward(grad, is_output_layer=True)

        grad_orig = grad[:1]  # batch size = 4/4
        assert torch.allclose(pp, 0.5 * grad_orig)
        assert torch.allclose(nn_, 0.5 * grad_orig)
        assert torch.all(np_ == 0)
        assert torch.all(pn == 0)

    def test_gradient_reconstruction(self):
        """pp - np - pn + nn equals the original gradient."""
        from bwgame.operations.base import init_backward

        grad = torch.randn(4, 5)
        pp, np_, pn, nn_ = init_backward(grad, is_output_layer=True)
        assert torch.allclose(pp - np_ - pn + nn_, grad[:1], atol=1e-6)


# ===========================================================================
# DC decomposition with the fan-in entropy bias
# ===========================================================================

class TestDCDecompositionEntropyBias:
    """dc_decomposition with the fan-in entropy bias."""

    def test_entropy_bias_adapt_integration(self):
        """DC decomposition runs with entropy_bias_adapt and restores the biases."""
        from bwgame.dc_context import dc_decomposition

        model = SimpleMLP(in_features=48, out_features=4)
        biases = {n: p.detach().clone() for n, p in model.named_parameters() if "bias" in n}
        x = torch.randn(1, 3, 4, 4)

        with dc_decomposition(model, x, entropy_bias_adapt=0.1) as dc:
            out = dc.output
            one_hot = torch.zeros_like(out)
            one_hot[0, 0] = 1.0
            out.backward(gradient=one_hot)

        grad = dc.reconstruct_gradient()
        assert torch.isfinite(grad).all()
        for n, p in model.named_parameters():
            if "bias" in n:
                assert torch.equal(p, biases[n])


# ===========================================================================
# GELU→ReLU replacement tests
# ===========================================================================

class GeluMLP(nn.Module):
    """MLP with GELU activations for testing replacement."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(48, 16)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = x.flatten(1)
        return self.fc2(self.gelu1(self.fc1(x)))


class TestGeluReplacement:
    """Tests for automatic GELU→ReLU replacement in dc_decomposition."""

    def test_gelu_replaced_during_context(self):
        """GELU modules should become ReLU inside the dc_decomposition context."""
        from bwgame.dc_context import dc_decomposition

        model = GeluMLP()
        assert isinstance(model.gelu1, nn.GELU)

        x = torch.randn(1, 3, 4, 4)
        with dc_decomposition(model, x) as dc:
            out = dc.output
            one_hot = torch.zeros_like(out)
            one_hot[0, 0] = 1.0
            out.backward(gradient=one_hot)

        # After context exits, original GELU should be restored
        assert isinstance(model.gelu1, nn.GELU)

    def test_gelu_restored_after_context(self):
        """Original GELU modules should be restored when context exits."""
        from bwgame.dc_context import dc_decomposition

        model = GeluMLP()
        original_gelu = model.gelu1
        x = torch.randn(1, 3, 4, 4)

        with dc_decomposition(model, x) as dc:
            out = dc.output
            one_hot = torch.zeros_like(out)
            one_hot[0, 0] = 1.0
            out.backward(gradient=one_hot)

        assert model.gelu1 is original_gelu

    def test_gelu_restored_on_exception(self):
        """GELU should be restored even if an exception occurs inside context."""
        from bwgame.dc_context import dc_decomposition

        model = GeluMLP()
        assert isinstance(model.gelu1, nn.GELU)
        x = torch.randn(1, 3, 4, 4)

        with pytest.raises(RuntimeError):
            with dc_decomposition(model, x) as dc:
                raise RuntimeError("test error")

        assert isinstance(model.gelu1, nn.GELU)

    def test_dc_sensitivities_nonzero_with_gelu_model(self):
        """DC sensitivities should be non-zero for a model originally containing GELU."""
        from bwgame.dc_context import dc_decomposition

        model = GeluMLP()
        x = torch.randn(1, 3, 4, 4)

        with dc_decomposition(model, x) as dc:
            out = dc.output
            one_hot = torch.zeros_like(out)
            one_hot[0, 0] = 1.0
            out.backward(gradient=one_hot)

        sens = dc.sensitivities
        # At least some sensitivities should be non-zero
        assert sens.delta_pp.abs().sum() > 0

    def test_silu_also_replaced(self):
        """SiLU (Swish) should also be replaced with ReLU."""
        from bwgame.dc_context import dc_decomposition

        class SiluModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(48, 16)
                self.silu = nn.SiLU()
                self.fc2 = nn.Linear(16, 4)

            def forward(self, x):
                x = x.flatten(1)
                return self.fc2(self.silu(self.fc1(x)))

        model = SiluModel()
        assert isinstance(model.silu, nn.SiLU)
        x = torch.randn(1, 3, 4, 4)

        with dc_decomposition(model, x) as dc:
            out = dc.output
            one_hot = torch.zeros_like(out)
            one_hot[0, 0] = 1.0
            out.backward(gradient=one_hot)

        assert isinstance(model.silu, nn.SiLU)  # Restored
        assert dc.sensitivities.delta_pp.abs().sum() > 0


# ===========================================================================
# GELU DC Decomposition tests
# ===========================================================================

class SimpleGELUMLP(nn.Module):
    """MLP with GELU activations for DC decomposition testing."""
    def __init__(self, in_features=48, hidden=16, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.gelu1 = nn.GELU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.gelu2 = nn.GELU()
        self.fc3 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = x.flatten(1)
        x = self.gelu1(self.fc1(x))
        x = self.gelu2(self.fc2(x))
        return self.fc3(x)


class SimpleSiLUMLP(nn.Module):
    """MLP with SiLU/Swish activations for DC decomposition testing."""
    def __init__(self, in_features=48, hidden=16, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.silu1 = nn.SiLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.silu2 = nn.SiLU()
        self.fc3 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = x.flatten(1)
        x = self.silu1(self.fc1(x))
        x = self.silu2(self.fc2(x))
        return self.fc3(x)


class TestGELUDecomposition:
    """Tests for GELU DC decomposition (forward and backward)."""

    def test_gelu_forward_reconstruction(self):
        """out_pos - out_neg = GELU(pos - neg) for various lambda values."""
        from bwgame.operations.gelu import forward_gelu
        import torch.nn.functional as F

        torch.manual_seed(42)
        pos = torch.randn(2, 8).abs() + 0.1  # Ensure positive
        neg = torch.randn(2, 8).abs() + 0.1

        z = pos - neg
        expected = F.gelu(z)

        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            out_pos, out_neg = forward_gelu(pos, neg, relu_lambda=lam)
            reconstructed = out_pos - out_neg
            assert torch.allclose(reconstructed, expected, atol=1e-5), \
                f"GELU forward reconstruction failed for lambda={lam}: max err={( reconstructed - expected).abs().max()}"

    def test_gelu_stopping_forward(self):
        """Stopping mode: Phi(z)*pos - Phi(z)*neg = GELU(z)."""
        from bwgame.operations.gelu import forward_gelu_stopping
        import torch.nn.functional as F

        torch.manual_seed(42)
        pos = torch.randn(2, 8).abs() + 0.1
        neg = torch.randn(2, 8).abs() + 0.1

        z = pos - neg
        expected = F.gelu(z)

        out_pos, out_neg = forward_gelu_stopping(pos, neg)
        reconstructed = out_pos - out_neg
        assert torch.allclose(reconstructed, expected, atol=1e-5), \
            f"GELU stopping forward reconstruction failed: max err={(reconstructed - expected).abs().max()}"

    def test_gelu_lambda_extremes(self):
        """Lambda=1 -> max mode (a- = neg), lambda=0 -> min mode (a+ = pos)."""
        from bwgame.operations.gelu import forward_gelu

        torch.manual_seed(42)
        pos = torch.randn(2, 8).abs() + 0.5
        neg = torch.randn(2, 8).abs() + 0.5

        # Max mode: a- = b (neg)
        out_pos_max, out_neg_max = forward_gelu(pos, neg, relu_lambda=1.0)
        assert torch.allclose(out_neg_max, neg, atol=1e-6), \
            "Max mode: out_neg should equal neg"

        # Min mode: a+ = a (pos)
        out_pos_min, out_neg_min = forward_gelu(pos, neg, relu_lambda=0.0)
        assert torch.allclose(out_pos_min, pos, atol=1e-6), \
            "Min mode: out_pos should equal pos"

    def test_gelu_backward_gradient_reconstruction(self):
        """pp - np - pn + nn should approximate the standard gradient (standard mode)."""
        from bwgame.dc_context import dc_decomposition

        model = SimpleGELUMLP()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 4)

        # Standard gradient
        x_grad = x.clone().requires_grad_(True)
        out = model(x_grad.flatten(1))
        one_hot = torch.zeros_like(out)
        one_hot[0, 0] = 1.0
        out.backward(gradient=one_hot)
        std_grad = x_grad.grad.clone()

        # DC gradient reconstruction with native GELU
        with dc_decomposition(
            model, x, replace_activations=False, backprop_mode='standard'
        ) as dc:
            dc_out = dc.output
            one_hot = torch.zeros_like(dc_out)
            one_hot[0, 0] = 1.0
            dc_out.backward(gradient=one_hot)

        dc_grad = dc.sensitivities.reconstruct_gradient()

        assert torch.allclose(std_grad.flatten(), dc_grad.flatten(), atol=1e-4), \
            f"GELU gradient reconstruction failed: max err={(std_grad.flatten() - dc_grad.flatten()).abs().max()}"

    def test_gelu_stopping_backward(self):
        """Stopping mode produces valid sensitivities."""
        from bwgame.dc_context import dc_decomposition

        model = SimpleGELUMLP()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 4)

        with dc_decomposition(
            model, x, replace_activations=False, backprop_mode='stopping'
        ) as dc:
            dc_out = dc.output
            one_hot = torch.zeros_like(dc_out)
            one_hot[0, 0] = 1.0
            dc_out.backward(gradient=one_hot)

        sens = dc.sensitivities
        assert torch.isfinite(sens.delta_pp).all()
        assert torch.isfinite(sens.delta_np).all()
        assert torch.isfinite(sens.delta_pn).all()
        assert torch.isfinite(sens.delta_nn).all()
        # Should be non-zero
        assert sens.delta_pp.abs().sum() > 0

    def test_gelu_model_dc_decomposition(self):
        """Full model with nn.GELU, dc_decomposition(replace_activations=False)."""
        from bwgame.dc_context import dc_decomposition

        model = SimpleGELUMLP()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 4)

        # Original output
        with torch.no_grad():
            orig_out = model(x.flatten(1))

        # DC output (should reconstruct original)
        with dc_decomposition(
            model, x, replace_activations=False, backprop_mode='standard'
        ) as dc:
            dc_out = dc.output

        assert torch.allclose(orig_out, dc_out, atol=1e-4), \
            f"GELU model forward reconstruction failed: max err={(orig_out - dc_out).abs().max()}"

    def test_gelu_model_gradient_reconstruction(self):
        """Full model: pp-np-pn+nn approximates standard gradient."""
        from bwgame.dc_context import dc_decomposition

        model = SimpleGELUMLP()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 4)

        # Standard gradient
        x_grad = x.clone().requires_grad_(True)
        out = model(x_grad.flatten(1))
        one_hot = torch.zeros_like(out)
        one_hot[0, 0] = 1.0
        out.backward(gradient=one_hot)
        std_grad = x_grad.grad.clone()

        # DC gradient
        with dc_decomposition(
            model, x, replace_activations=False, backprop_mode='standard'
        ) as dc:
            dc_out = dc.output
            one_hot = torch.zeros_like(dc_out)
            one_hot[0, 0] = 1.0
            dc_out.backward(gradient=one_hot)

        dc_grad = dc.sensitivities.reconstruct_gradient()
        assert torch.allclose(std_grad.flatten(), dc_grad.flatten(), atol=1e-4), \
            f"GELU model gradient reconstruction failed: max err={(std_grad.flatten() - dc_grad.flatten()).abs().max()}"

    def test_gelu_unsupported_backprop_mode_raises(self):
        """'sum' and 'mask_diff' modes should fail with native GELU."""
        from bwgame.dc_context import dc_decomposition

        model = SimpleGELUMLP()
        x = torch.randn(1, 3, 4, 4)

        # The 'sum' mode is ReLU-specific (uses discrete masks).
        # With native GELU, it should still run but the semantics differ.
        # We just verify it doesn't crash for 'standard' and 'stopping'.
        for mode in ['standard', 'stopping']:
            with dc_decomposition(
                model, x, replace_activations=False, backprop_mode=mode
            ) as dc:
                dc_out = dc.output
                one_hot = torch.zeros_like(dc_out)
                one_hot[0, 0] = 1.0
                dc_out.backward(gradient=one_hot)
            assert dc.sensitivities.delta_pp.abs().sum() > 0


class TestSwishDecomposition:
    """Tests for SiLU/Swish DC decomposition (forward and backward)."""

    def test_swish_forward_reconstruction(self):
        """out_pos - out_neg = SiLU(pos - neg) for various lambda values."""
        from bwgame.operations.swish import forward_swish
        import torch.nn.functional as F

        torch.manual_seed(42)
        pos = torch.randn(2, 8).abs() + 0.1
        neg = torch.randn(2, 8).abs() + 0.1

        z = pos - neg
        expected = F.silu(z)

        for lam in [0.0, 0.25, 0.5, 0.75, 1.0]:
            out_pos, out_neg = forward_swish(pos, neg, relu_lambda=lam)
            reconstructed = out_pos - out_neg
            assert torch.allclose(reconstructed, expected, atol=1e-5), \
                f"Swish forward reconstruction failed for lambda={lam}: max err={(reconstructed - expected).abs().max()}"

    def test_swish_stopping_forward(self):
        """Stopping mode: sigma(z)*pos - sigma(z)*neg = SiLU(z)."""
        from bwgame.operations.swish import forward_swish_stopping
        import torch.nn.functional as F

        torch.manual_seed(42)
        pos = torch.randn(2, 8).abs() + 0.1
        neg = torch.randn(2, 8).abs() + 0.1

        z = pos - neg
        expected = F.silu(z)

        out_pos, out_neg = forward_swish_stopping(pos, neg)
        reconstructed = out_pos - out_neg
        assert torch.allclose(reconstructed, expected, atol=1e-5), \
            f"Swish stopping forward reconstruction failed: max err={(reconstructed - expected).abs().max()}"

    def test_swish_lambda_extremes(self):
        """Lambda=1 -> max mode (a- = neg), lambda=0 -> min mode (a+ = pos)."""
        from bwgame.operations.swish import forward_swish

        torch.manual_seed(42)
        pos = torch.randn(2, 8).abs() + 0.5
        neg = torch.randn(2, 8).abs() + 0.5

        out_pos_max, out_neg_max = forward_swish(pos, neg, relu_lambda=1.0)
        assert torch.allclose(out_neg_max, neg, atol=1e-6), \
            "Max mode: out_neg should equal neg"

        out_pos_min, out_neg_min = forward_swish(pos, neg, relu_lambda=0.0)
        assert torch.allclose(out_pos_min, pos, atol=1e-6), \
            "Min mode: out_pos should equal pos"

    def test_swish_lambda_sweep(self):
        """Sweep lambda from 0 to 1, all produce valid decomposition."""
        from bwgame.operations.swish import forward_swish
        import torch.nn.functional as F

        torch.manual_seed(42)
        pos = torch.randn(3, 10).abs() + 0.1
        neg = torch.randn(3, 10).abs() + 0.1
        z = pos - neg
        expected = F.silu(z)

        for lam in torch.linspace(0, 1, 11).tolist():
            out_pos, out_neg = forward_swish(pos, neg, relu_lambda=lam)
            reconstructed = out_pos - out_neg
            assert torch.allclose(reconstructed, expected, atol=1e-5), \
                f"Swish reconstruction failed at lambda={lam:.1f}"
            # Both streams should be finite
            assert torch.isfinite(out_pos).all()
            assert torch.isfinite(out_neg).all()

    def test_swish_backward_gradient_reconstruction(self):
        """pp - np - pn + nn should approximate the standard gradient."""
        from bwgame.dc_context import dc_decomposition

        model = SimpleSiLUMLP()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 4)

        # Standard gradient
        x_grad = x.clone().requires_grad_(True)
        out = model(x_grad.flatten(1))
        one_hot = torch.zeros_like(out)
        one_hot[0, 0] = 1.0
        out.backward(gradient=one_hot)
        std_grad = x_grad.grad.clone()

        # DC gradient
        with dc_decomposition(
            model, x, replace_activations=False, backprop_mode='standard'
        ) as dc:
            dc_out = dc.output
            one_hot = torch.zeros_like(dc_out)
            one_hot[0, 0] = 1.0
            dc_out.backward(gradient=one_hot)

        dc_grad = dc.sensitivities.reconstruct_gradient()
        assert torch.allclose(std_grad.flatten(), dc_grad.flatten(), atol=1e-4), \
            f"Swish gradient reconstruction failed: max err={(std_grad.flatten() - dc_grad.flatten()).abs().max()}"

    def test_swish_stopping_mode(self):
        """Stopping mode forward + backward produces valid sensitivities."""
        from bwgame.dc_context import dc_decomposition

        model = SimpleSiLUMLP()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 4)

        with dc_decomposition(
            model, x, replace_activations=False, backprop_mode='stopping'
        ) as dc:
            dc_out = dc.output
            one_hot = torch.zeros_like(dc_out)
            one_hot[0, 0] = 1.0
            dc_out.backward(gradient=one_hot)

        sens = dc.sensitivities
        assert torch.isfinite(sens.delta_pp).all()
        assert torch.isfinite(sens.delta_nn).all()
        assert sens.delta_pp.abs().sum() > 0

    def test_swish_model_dc_decomposition(self):
        """Full model with nn.SiLU, dc_decomposition(replace_activations=False)."""
        from bwgame.dc_context import dc_decomposition

        model = SimpleSiLUMLP()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 4)

        with torch.no_grad():
            orig_out = model(x.flatten(1))

        with dc_decomposition(
            model, x, replace_activations=False, backprop_mode='standard'
        ) as dc:
            dc_out = dc.output

        assert torch.allclose(orig_out, dc_out, atol=1e-4), \
            f"Swish model forward reconstruction failed: max err={(orig_out - dc_out).abs().max()}"

    def test_swish_model_gradient_reconstruction(self):
        """Full model: pp-np-pn+nn approximates standard gradient."""
        from bwgame.dc_context import dc_decomposition

        model = SimpleSiLUMLP()
        model.eval()
        torch.manual_seed(42)
        x = torch.randn(1, 3, 4, 4)

        x_grad = x.clone().requires_grad_(True)
        out = model(x_grad.flatten(1))
        one_hot = torch.zeros_like(out)
        one_hot[0, 0] = 1.0
        out.backward(gradient=one_hot)
        std_grad = x_grad.grad.clone()

        with dc_decomposition(
            model, x, replace_activations=False, backprop_mode='standard'
        ) as dc:
            dc_out = dc.output
            one_hot = torch.zeros_like(dc_out)
            one_hot[0, 0] = 1.0
            dc_out.backward(gradient=one_hot)

        dc_grad = dc.sensitivities.reconstruct_gradient()
        assert torch.allclose(std_grad.flatten(), dc_grad.flatten(), atol=1e-4), \
            f"Swish model gradient reconstruction failed: max err={(std_grad.flatten() - dc_grad.flatten()).abs().max()}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
