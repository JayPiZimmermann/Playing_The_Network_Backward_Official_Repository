"""Routing Game (RG) attribution, implemented with backward hooks.

The backward pass of every Linear / Conv2d layer is replaced by the
alpha-beta-LRP-eps rule with the Gibbs routing temperature tau of the Routing
Game (the sign-split activations, weights and biases are raised to the power
1/tau).  At tau = 1 and without risk parameters the method is
alpha-beta-LRP (zennit's ``AlphaBeta`` rule on the same layers).  Analytical
modes (paper appendix "Analytical Modes"):

  - activation-side risk aversion ``adf_lambda`` with input variance
    ``adf_input_variance``: the activation derivatives are evaluated at the
    ADF confidence bound z_eff = mu + lambda * sigma, with the smooth gate
    Phi(z_eff / sigma) (``adf_probabilistic``) or the hard gate 1{z_eff > 0};
  - softmax risk ``adf_softmax_lambda`` (ViTs): the attention of the Value
    routing is recomputed from the logits shifted by lambda_sm * sigma_k of
    the key tokens (backward only, the forward pass is unchanged);
  - fan-in entropy bias ``entropy_lambda``: b_j -= lambda_ent * H~_j.

Vision Transformers: LayerNorm is detached, the softmax and the Q/K product
pass no relevance and the Value path is routed with the composite rule of
the attention recovery theorem (``av_rule='merged'``,
:meth:`_RoutingGameHookManager._merged_value_backward`).  With
``attention_mode='gradient_fallback'`` every attention product uses
epsilon-LRP and the softmax standard autograd instead.

BatchNorm layers are merged into the preceding convolution (zennit-style
canonization) and treated with the epsilon rule.
"""

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _power_routing_shares(mass: Tensor, coefficient: Tensor, tau: float,
                          eps: float, dim: int = -1) -> Tensor:
    """Normalized limit of ``coefficient * mass**(1/tau)`` with outside eps.

    At tau=0 the epsilon term is an unpowered outside option, as in RG's
    implemented denominator.  Consequently masses below one lose to eps,
    masses equal to one tie with it, and masses above one dominate it.  Zero
    epsilon leaves the largest positive mass (or an empty route) instead.
    The log-domain finite-temperature branch avoids overflow for small tau.
    """
    if tau < 0:
        raise ValueError("routing_temperature must be nonnegative")
    mass = mass.clamp_min(0)
    coefficient = coefficient.clamp_min(0)
    log_mass = torch.where(mass > 0, mass.log(), -torch.inf)
    log_coeff = torch.where(coefficient > 0, coefficient.log(), -torch.inf)
    if tau == 0:
        positive = (mass > 0) & (coefficient > 0)
        maximum = torch.where(positive, log_mass, -torch.inf).amax(dim=dim, keepdim=True)
        if eps > 0:
            # Compare mass to the unit threshold.  For mass=1, coefficients
            # and epsilon retain their usual proportional competition.
            active = positive & ((log_mass == maximum) & (maximum > 0))
            tied = positive & (log_mass == 0) & (maximum == 0)
            out_wins = maximum < 0
            edge_weight = torch.where(active, coefficient, torch.zeros_like(coefficient))
            edge_weight = torch.where(tied, coefficient, edge_weight)
            out_weight = torch.where(out_wins, torch.full_like(maximum, float(eps)),
                                     torch.zeros_like(maximum))
            out_weight = torch.where(maximum == 0, out_weight + eps, out_weight)
        else:
            active = positive & (log_mass == maximum) & torch.isfinite(maximum)
            edge_weight = torch.where(active, coefficient, torch.zeros_like(coefficient))
            out_weight = torch.zeros_like(maximum)
        total = edge_weight.sum(dim=dim, keepdim=True) + out_weight
        return torch.where(total > 0, edge_weight / total.clamp_min(torch.finfo(mass.dtype).tiny),
                           torch.zeros_like(edge_weight))

    edge_log = log_coeff + log_mass / tau
    norm = torch.logsumexp(edge_log, dim=dim, keepdim=True)
    if eps > 0:
        norm = torch.logaddexp(norm, mass.new_tensor(float(eps)).log())
    finite_norm = torch.isfinite(norm)
    return torch.where(finite_norm, torch.exp(edge_log - norm), torch.zeros_like(edge_log))
from ..base import AttributionMethod
from ..postprocess import tensor_to_numpy
from ..registry import MethodRegistry
from ..types import MethodCategory, MethodConfig

from bwgame.operations.matmul import DCMatMul

logger = logging.getLogger(__name__)

_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

ATTENTION_MODES = ('v_only', 'gradient_fallback')
AV_RULES = ('merged',)


def _normal_pdf(z: Tensor) -> Tensor:
    return _INV_SQRT_2PI * torch.exp(-0.5 * z * z)


def _conv_output_padding(input_shape, output_shape, kernel_size, stride, padding, dilation):
    """Compute output_padding for conv_transpose2d so it maps back to input_shape.

    conv_transpose2d output: H = (H_in-1)*s - 2*p + d*(k-1) + op + 1
    We want H = input_shape, H_in = output_shape (of the forward conv).
    """
    op = []
    for i in range(2):
        h_target = input_shape[i + 2]
        h_no_op = ((output_shape[i + 2] - 1) * stride[i]
                    - 2 * padding[i] + dilation[i] * (kernel_size[i] - 1) + 1)
        op.append(h_target - h_no_op)
    return tuple(op)


# =========================================================================
# Hook manager
# =========================================================================

class _RoutingGameHookManager:
    """Forward + backward hooks implementing the Routing Game backward pass:
    alpha-beta LRP with Gibbs temperature routing, ADF risk-modified
    activation derivatives and the attention rules of the RG on ViTs.
    """

    def __init__(
        self,
        model: nn.Module,
        lrp_alpha: float = 2.0,
        lrp_beta: float = 1.0,
        lrp_eps: float = 0.5,
        routing_temperature: float = 1.0,
        adf_lambda: float = 0.0,
        adf_input_variance: float = 1.0,
        adf_probabilistic: bool = True,
        force_adf_activation_gate: bool = False,
        adf_softmax_lambda: float = 0.0,
        attention_mode: str = 'v_only',
        av_rule: str = 'merged',
    ):
        self.model = model
        if attention_mode not in ATTENTION_MODES:
            raise ValueError(f"attention_mode must be one of {ATTENTION_MODES}, got {attention_mode!r}")
        # Relevance rule for the value path of O = A·V (attention_mode
        # 'v_only'): 'merged' is alpha-beta-LRP-eps on the composite map
        # X -> O = A (X W_v^T + b_v) (see _merged_value_backward).
        if av_rule not in AV_RULES:
            raise ValueError(f"av_rule must be one of {AV_RULES}, got {av_rule!r}")
        self.av_rule = av_rule
        # Per-stream relevance handed from the attention @ Value product to the
        # Value projection (keyed by the projection's module name)
        self._merged_value_relevance: Dict[str, tuple] = {}
        self.lrp_alpha = lrp_alpha
        self.lrp_beta = lrp_beta
        self.lrp_eps = lrp_eps
        self.routing_temperature = routing_temperature
        self.adf_lambda = adf_lambda
        self.adf_input_variance = adf_input_variance
        self.adf_probabilistic = adf_probabilistic
        # Diagnostic control: keep the ADF variance pass and probabilistic
        # activation gate active at lambda=0.  The default is false, preserving
        # the submitted implementation exactly.  This is used only to separate
        # gate smoothing from the confidence-bound shift in matched ablations.
        self.force_adf_activation_gate = force_adf_activation_gate
        self.adf_softmax_lambda = adf_softmax_lambda
        self.attention_mode = attention_mode
        self._bn_backup: List[dict] = []  # saved BN state for unmerge

        self._hooks: List[torch.utils.hooks.RemovableHook] = []
        self._activations: Dict[str, Tensor] = {}
        self._adf_variances: Dict[str, Tensor] = {}
        self._inplace_relus: List[nn.ReLU] = []
        self._patched_residual_blocks = []

        # Attention module identification
        self._qk_matmuls: Dict[str, nn.Module] = {}  # name -> module
        self._av_matmuls: Dict[str, nn.Module] = {}   # name -> module
        self._softmax_modules: Dict[str, nn.Module] = {}  # name -> module

    def _identify_attention_modules(self):
        """Identify QK matmul (index 0), attn@V matmul (index 1),
        and softmax modules within each attention block."""
        matmul_count: Dict[str, int] = {}  # parent -> count
        for name, module in self.model.named_modules():
            if isinstance(module, DCMatMul):
                parts = name.rsplit('.', 1)
                parent = parts[0] if len(parts) > 1 else ''
                idx = matmul_count.get(parent, 0)
                matmul_count[parent] = idx + 1
                if idx == 0:
                    self._qk_matmuls[name] = module
                elif idx == 1:
                    self._av_matmuls[name] = module
            elif isinstance(module, nn.Softmax):
                self._softmax_modules[name] = module

    def run_adf_pass(self, x: Tensor):
        """Forward-only ADF pass that stores the per-layer variances.

        Uses the stacked [mean; variance] ADF forward through the patched
        model, so that every operation (including Add / skip connections,
        LayerNorm with the delta method, Softmax and MatMul) propagates the
        variance.  The per-layer variances are read from DC_LAYER_DATA and
        stored in ``_adf_variances``.
        """
        from bwgame.dc_context import run_adf_forward
        from bwgame.patcher import patch_model, unpatch_model
        from bwgame.game_theory.game_context import (
            DC_LAYER_DATA, cleanup_layer_data,
        )
        from bwgame.operations.base import adf_propagation_context

        patch_model(self.model)
        try:
            with torch.no_grad(), adf_propagation_context('delta_method'):
                run_adf_forward(
                    self.model, x,
                    input_variance=self.adf_input_variance,
                )

                for name, m in self.model.named_modules():
                    if not name:
                        continue
                    ld = getattr(m, DC_LAYER_DATA, None)
                    if ld is not None and ld.adf_variance is not None:
                        self._adf_variances[name] = ld.adf_variance.detach()
        finally:
            unpatch_model(self.model)
            cleanup_layer_data(self.model)

    def _compute_z_eff(self, mean: Tensor, var: Tensor) -> Tensor:
        """Risk-modified pre-activation (confidence bound) z_eff = mu + lambda * sigma."""
        if self.adf_lambda == 0.0:
            return mean
        std = torch.sqrt(var.clamp(min=1e-8))
        return mean + self.adf_lambda * std

    @staticmethod
    def _linear_power_route(act: Tensor, weight: Tensor, bias: Optional[Tensor],
                            grad_out: Tensor, tau: float, eps: float,
                            cross: bool = False) -> Tensor:
        """Route linear relevance through normalized per-edge power masses."""
        shape = act.shape
        x = act.reshape(-1, shape[-1])
        r = grad_out.reshape(-1, grad_out.shape[-1])
        xp, xn = torch.relu(x), torch.relu(-x)
        wp, wn = torch.relu(weight), torch.relu(-weight)
        relevance = torch.zeros_like(x)
        # Bound peak memory for wide ViT projections by processing output
        # features in chunks.  Each output is normalized independently.
        for start in range(0, weight.shape[0], 32):
            stop = min(start + 32, weight.shape[0])
            w = weight[start:stop]
            if cross:
                first, second = xp[:, None, :] * torch.relu(-w)[None, :, :], xn[:, None, :] * torch.relu(w)[None, :, :]
                b = None if bias is None else (-bias[start:stop]).clamp_min(0)
            else:
                first, second = xp[:, None, :] * torch.relu(w)[None, :, :], xn[:, None, :] * torch.relu(-w)[None, :, :]
                b = None if bias is None else bias[start:stop].clamp_min(0)
            mass = torch.cat((first, second), dim=-1)
            if b is not None:
                mass = torch.cat((mass, b[None, :, None].expand(mass.shape[0], -1, -1)), dim=-1)
            shares = _power_routing_shares(mass, torch.ones_like(mass), tau, eps)
            routed = shares[..., :2 * x.shape[-1]] * r[:, start:stop].unsqueeze(-1)
            relevance += (routed[..., :x.shape[-1]].sum(dim=1)
                          + routed[..., x.shape[-1]:].sum(dim=1))
        return relevance.reshape(shape)

    @staticmethod
    def _conv_power_route(act: Tensor, weight: Tensor, bias: Optional[Tensor],
                          grad_out: Tensor, mod: nn.Conv2d, tau: float,
                          eps: float, cross: bool = False) -> Tensor:
        """Conv2d analogue of normalized per-edge power routing.

        Spatial output locations are processed in small chunks so patch
        embeddings do not materialize the full output-by-input edge tensor.
        """
        bsz, cin, hin, win = act.shape
        groups = mod.groups
        cin_g, cout_g = cin // groups, mod.out_channels // groups
        unfolded = F.unfold(act, mod.kernel_size, dilation=mod.dilation,
                            padding=mod.padding, stride=mod.stride)
        locs = unfolded.shape[-1]
        out_cols = grad_out.flatten(2).transpose(1, 2)
        khkw = mod.kernel_size[0] * mod.kernel_size[1]
        relevance_cols = []
        for group in range(groups):
            xg = unfolded[:, group * cin_g * khkw:(group + 1) * cin_g * khkw].transpose(1, 2)
            wg = weight[group * cout_g:(group + 1) * cout_g].reshape(cout_g, -1)
            bg = None if bias is None else bias[group * cout_g:(group + 1) * cout_g]
            xout = []
            for start in range(0, locs, 8):
                stop = min(start + 8, locs)
                x = xg[:, start:stop]
                xp, xn = torch.relu(x), torch.relu(-x)
                wp, wn = torch.relu(wg), torch.relu(-wg)
                if cross:
                    first = xp[:, :, None, :] * wn[None, None, :, :]
                    second = xn[:, :, None, :] * wp[None, None, :, :]
                    b = None if bg is None else (-bg).clamp_min(0)
                else:
                    first = xp[:, :, None, :] * wp[None, None, :, :]
                    second = xn[:, :, None, :] * wn[None, None, :, :]
                    b = None if bg is None else bg.clamp_min(0)
                mass = torch.cat((first, second), dim=-1)
                if b is not None:
                    mass = torch.cat((mass, b[None, None, :, None].expand(x.shape[0], x.shape[1], -1, -1)), dim=-1)
                shares = _power_routing_shares(mass, torch.ones_like(mass), tau, eps)
                routed = shares[..., :2 * x.shape[-1]] * out_cols[:, start:stop, group * cout_g:(group + 1) * cout_g].unsqueeze(-1)
                xout.append(routed[..., :x.shape[-1]].sum(dim=2)
                            + routed[..., x.shape[-1]:].sum(dim=2))
            relevance_cols.append(torch.cat(xout, dim=1).transpose(1, 2))
        relevance_unfolded = torch.cat(relevance_cols, dim=1)
        return F.fold(relevance_unfolded, output_size=(hin, win), kernel_size=mod.kernel_size,
                      dilation=mod.dilation, padding=mod.padding, stride=mod.stride)

    def _stabilize(self, z, eps):
        """Stabilize denominator for LRP division: z + ε·sign(z), with +ε for exact zeros."""
        return z + eps * ((z == 0).float() + z.sign())

    @staticmethod
    def _divide_relevance(relevance, denominator):
        """Empty routing streams absorb their incoming relevance."""
        nonempty = denominator != 0
        safe_denominator = torch.where(nonempty, denominator, torch.ones_like(denominator))
        return torch.where(nonempty, relevance / safe_denominator, torch.zeros_like(relevance))

    # -----------------------------------------------------------------
    # BatchNorm canonization
    # -----------------------------------------------------------------
    def _merge_batchnorm(self):
        """Merge BatchNorm into preceding Conv/Linear (zennit-style canonization).

        Modifies weights in-place so LRP sees the merged parameters.
        Call _unmerge_batchnorm() after backward to restore originals.
        """
        self._bn_backup.clear()

        # Build ordered list of leaf modules (no children)
        leaf_list = [(n, m) for n, m in self.model.named_modules()
                     if sum(1 for _ in m.children()) == 0]

        for i, (name, mod) in enumerate(leaf_list):
            if not isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d)):
                continue

            # Find the preceding Conv/Linear in execution order
            lin_mod = None
            lin_name = None
            for j in range(i - 1, -1, -1):
                prev_name, prev_mod = leaf_list[j]
                if isinstance(prev_mod, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                    lin_name, lin_mod = prev_name, prev_mod
                    break
                # Stop searching if we hit another non-trivial layer
                if not isinstance(prev_mod, (nn.Identity,)):
                    break

            if lin_mod is None:
                continue

            # Save originals
            backup = {
                'bn_name': name, 'bn_mod': mod,
                'lin_name': lin_name, 'lin_mod': lin_mod,
                'lin_weight': lin_mod.weight.data.clone(),
                'lin_bias': lin_mod.bias.data.clone() if lin_mod.bias is not None else None,
                'bn_running_mean': mod.running_mean.clone(),
                'bn_running_var': mod.running_var.clone(),
                'bn_weight': mod.weight.data.clone() if mod.weight is not None else None,
                'bn_bias': mod.bias.data.clone() if mod.bias is not None else None,
                'bn_eps': mod.eps,
            }
            self._bn_backup.append(backup)

            # Merge: W_new = (gamma / sqrt(var + eps)) * W_old
            #         b_new = gamma * (b_old - mean) / sqrt(var + eps) + beta
            std = torch.sqrt(mod.running_var + mod.eps)
            gamma = mod.weight.data if mod.weight is not None else torch.ones_like(std)
            beta = mod.bias.data if mod.bias is not None else torch.zeros_like(std)
            scale = gamma / std

            if isinstance(lin_mod, (nn.Conv2d, nn.Conv1d)):
                # scale shape: (out_channels,) → reshape for conv weight
                shape = [1] * lin_mod.weight.dim()
                shape[0] = -1
                lin_mod.weight.data.mul_(scale.view(*shape))
            else:
                lin_mod.weight.data.mul_(scale.unsqueeze(1))

            if lin_mod.bias is not None:
                lin_mod.bias.data = scale * (lin_mod.bias.data - mod.running_mean) + beta
            else:
                lin_mod.bias = nn.Parameter(scale * (-mod.running_mean) + beta)

            # Neutralize BN: set to identity (eps > 0 is required by PyTorch)
            mod.running_mean.zero_()
            mod.running_var.fill_(1.0 - mod.eps)  # var + eps = 1.0 → scale = gamma/1.0 = gamma
            if mod.weight is not None:
                mod.weight.data.fill_(1.0)
            if mod.bias is not None:
                mod.bias.data.zero_()

    def _unmerge_batchnorm(self):
        """Restore original Conv/Linear and BatchNorm parameters."""
        for backup in reversed(self._bn_backup):
            lin_mod = backup['lin_mod']
            bn_mod = backup['bn_mod']
            lin_mod.weight.data.copy_(backup['lin_weight'])
            if backup['lin_bias'] is not None:
                if lin_mod.bias is not None:
                    lin_mod.bias.data.copy_(backup['lin_bias'])
            else:
                lin_mod.bias = None
            bn_mod.running_mean.copy_(backup['bn_running_mean'])
            bn_mod.running_var.copy_(backup['bn_running_var'])
            if backup['bn_weight'] is not None and bn_mod.weight is not None:
                bn_mod.weight.data.copy_(backup['bn_weight'])
            if backup['bn_bias'] is not None and bn_mod.bias is not None:
                bn_mod.bias.data.copy_(backup['bn_bias'])
            bn_mod.eps = backup['bn_eps']
        self._bn_backup.clear()

    def install(self) -> "_RoutingGameHookManager":
        """Register forward (cache activations) + backward (LRP rules) hooks."""
        self._hooks.clear()
        self._activations.clear()

        # BatchNorm canonization
        self._merge_batchnorm()

        self._identify_attention_modules()

        # Disable inplace ReLU
        self._inplace_relus.clear()
        for module in self.model.modules():
            if isinstance(module, nn.ReLU) and module.inplace:
                module.inplace = False
                self._inplace_relus.append(module)

        # Patch residual blocks to use non-inplace add. torchvision's
        # Bottleneck/BasicBlock uses ``out += identity``, which modifies a
        # tensor wrapped by PyTorch's BackwardHookFunction (inserted around
        # any module carrying a ``register_full_backward_hook``) — that trips
        # "Output 0 of BackwardHookFunctionBackward is a view and is being
        # modified inplace". Swapping in a patched forward with
        # ``out = out + identity`` avoids the collision.
        self._patched_residual_blocks = []
        try:
            from torchvision.models.resnet import BasicBlock, Bottleneck
            _resnet_block_types = (BasicBlock, Bottleneck)
        except Exception:
            _resnet_block_types = ()

        def _make_bottleneck_forward(block):
            def forward(x):
                identity = x
                out = block.conv1(x)
                out = block.bn1(out)
                out = block.relu(out)
                out = block.conv2(out)
                out = block.bn2(out)
                out = block.relu(out)
                out = block.conv3(out)
                out = block.bn3(out)
                if block.downsample is not None:
                    identity = block.downsample(x)
                out = out + identity
                out = block.relu(out)
                return out
            return forward

        def _make_basicblock_forward(block):
            def forward(x):
                identity = x
                out = block.conv1(x)
                out = block.bn1(out)
                out = block.relu(out)
                out = block.conv2(out)
                out = block.bn2(out)
                if block.downsample is not None:
                    identity = block.downsample(x)
                out = out + identity
                out = block.relu(out)
                return out
            return forward

        for name, module in self.model.named_modules():
            if not _resnet_block_types or not isinstance(module, _resnet_block_types):
                continue
            orig = module.forward
            if type(module).__name__ == 'Bottleneck':
                module.forward = _make_bottleneck_forward(module)
            else:
                module.forward = _make_basicblock_forward(module)
            self._patched_residual_blocks.append((module, orig))

        for name, module in self.model.named_modules():
            if not name:
                continue

            if isinstance(module, nn.Linear):
                self._install_linear_hooks(name, module)
            elif isinstance(module, nn.Conv2d):
                self._install_conv2d_hooks(name, module)
            elif isinstance(module, nn.GELU):
                self._install_gelu_hooks(name, module)
            elif isinstance(module, nn.ReLU):
                self._install_relu_hooks(name, module)
            elif isinstance(module, nn.LayerNorm):
                self._install_layernorm_hooks(name, module)
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # Merged BN is identity but zennit applies epsilon-LRP to it.
                # Install matching LRP hooks so relevance is attenuated identically.
                self._install_bn_lrp_hooks(name, module)
            elif isinstance(module, nn.Softmax):
                self._install_softmax_hooks(name, module)
            elif isinstance(module, DCMatMul):
                self._install_matmul_hooks(name, module)

        return self

    def remove(self):
        """Clean up all hooks, restore inplace ReLU, unpatch residuals, and unmerge BN."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._activations.clear()
        for m in self._inplace_relus:
            m.inplace = True
        self._inplace_relus.clear()
        for block, orig_forward in getattr(self, '_patched_residual_blocks', []):
            block.forward = orig_forward
        self._patched_residual_blocks = []
        if self._bn_backup:
            self._unmerge_batchnorm()

    # -----------------------------------------------------------------
    # Linear layer hooks
    # -----------------------------------------------------------------
    def _install_linear_hooks(self, name: str, module: nn.Module):
        """Install forward (cache input, clone output) + backward (LRP) hooks."""

        def _fwd_hook(mod, inp, out, mod_name=name):
            self._activations[mod_name] = inp[0].detach()
            return out.clone()

        h_fwd = module.register_forward_hook(_fwd_hook)
        self._hooks.append(h_fwd)

        def _bwd_hook(mod, grad_input, grad_output, mod_name=name):
            if mod_name not in self._activations:
                return None

            act = self._activations[mod_name]   # (B, ..., in_features)
            weight = mod.weight.data             # (out_features, in_features)
            bias = mod.bias.data if mod.bias is not None else None
            grad_out = grad_output[0]            # (B, ..., out_features)

            tau = self.routing_temperature
            eps = self.lrp_eps
            alpha = self.lrp_alpha
            beta = self.lrp_beta

            merged = self._merged_value_relevance.pop(mod_name, None)
            if tau < 0.1:
                if merged is not None:
                    s_pos, s_neg = merged
                    lrp_grad = (alpha * self._linear_power_route(act, weight, bias, s_pos, tau, 0.0)
                                - beta * self._linear_power_route(act, weight, bias, s_neg, tau, 0.0, cross=True))
                else:
                    r_alpha = self._linear_power_route(act, weight, bias, grad_out, tau, eps)
                    r_beta = self._linear_power_route(act, weight, bias, grad_out, tau, eps, cross=True)
                    lrp_grad = alpha * r_alpha - beta * r_beta
                return (lrp_grad,) if len(grad_input) == 1 else (lrp_grad,) + grad_input[1:]

            # Positive/negative decomposition
            x_pos = torch.relu(act)
            x_neg = torch.relu(-act)
            w_pos = torch.relu(weight)
            w_neg = torch.relu(-weight)

            # Temperature: raise to power 1/τ
            if tau != 1.0:
                inv_tau = 1.0 / tau
                x_pos_t = x_pos.pow(inv_tau)
                x_neg_t = x_neg.pow(inv_tau)
                w_pos_t = w_pos.pow(inv_tau)
                w_neg_t = w_neg.pow(inv_tau)
            else:
                x_pos_t = x_pos
                x_neg_t = x_neg
                w_pos_t = w_pos
                w_neg_t = w_neg

            # Bias pos/neg decomposition
            b_pos_t = None
            b_neg_t = None
            if bias is not None:
                b_pos = bias.clamp(min=0)
                b_neg = bias.clamp(max=0)
                if tau != 1.0:
                    b_pos_t = b_pos.pow(inv_tau)
                    b_neg_t = -((-b_neg).pow(inv_tau))
                else:
                    b_pos_t = b_pos
                    b_neg_t = b_neg

            if merged is not None:
                # Value projection under the merged rule: the attention step has
                # already divided by the composite stream totals Z^kappa + eps and
                # applied A^T; route each stream to the contributions of its own
                # sign (alpha, beta applied once, no second stabiliser).
                s_pos, s_neg = merged
                r_alpha = (x_pos_t * torch.matmul(s_pos, w_pos_t)
                           + x_neg_t * torch.matmul(s_pos, w_neg_t))
                r_beta = (x_pos_t * torch.matmul(s_neg, w_neg_t)
                          + x_neg_t * torch.matmul(s_neg, w_pos_t))
                lrp_grad = alpha * r_alpha - beta * r_beta
                if len(grad_input) == 1:
                    return (lrp_grad,)
                return (lrp_grad,) + grad_input[1:]

            # Same-sign denominator
            z_pos = (torch.matmul(x_pos_t, w_pos_t.t())
                     + torch.matmul(x_neg_t, w_neg_t.t()))
            if b_pos_t is not None:
                z_pos = z_pos + b_pos_t

            # Cross-sign denominator
            z_neg = (torch.matmul(x_pos_t, w_neg_t.t())
                     + torch.matmul(x_neg_t, w_pos_t.t()))
            if b_neg_t is not None:
                z_neg = z_neg - b_neg_t

            z_pos = self._stabilize(z_pos, eps)
            z_neg = self._stabilize(z_neg, eps)

            # α path: same-sign routing
            s_pos = self._divide_relevance(grad_out, z_pos)
            r_alpha = (x_pos_t * torch.matmul(s_pos, w_pos_t)
                       + x_neg_t * torch.matmul(s_pos, w_neg_t))

            # β path: cross-sign routing
            s_neg = self._divide_relevance(grad_out, z_neg)
            r_beta = (x_pos_t * torch.matmul(s_neg, w_neg_t)
                      + x_neg_t * torch.matmul(s_neg, w_pos_t))

            lrp_grad = alpha * r_alpha - beta * r_beta

            if len(grad_input) == 1:
                return (lrp_grad,)
            else:
                return (lrp_grad,) + grad_input[1:]

        h_bwd = module.register_full_backward_hook(_bwd_hook)
        self._hooks.append(h_bwd)

    # -----------------------------------------------------------------
    # Conv2d layer hooks
    # -----------------------------------------------------------------
    def _install_conv2d_hooks(self, name: str, module: nn.Module):
        """Install forward (cache input) + backward (LRP) hooks for Conv2d."""

        def _fwd_hook(mod, inp, out, mod_name=name):
            self._activations[mod_name] = inp[0].detach()
            return out.clone()

        h_fwd = module.register_forward_hook(_fwd_hook)
        self._hooks.append(h_fwd)

        def _bwd_hook(mod, grad_input, grad_output, mod_name=name):
            if mod_name not in self._activations:
                return None

            act = self._activations[mod_name]   # (B, C_in, H, W)
            weight = mod.weight.data             # (C_out, C_in/groups, kH, kW)
            bias = mod.bias.data if mod.bias is not None else None
            grad_out = grad_output[0]            # (B, C_out, H', W')

            tau = self.routing_temperature
            eps = self.lrp_eps
            stride = mod.stride
            padding = mod.padding
            dilation = mod.dilation
            groups = mod.groups

            conv_kw = dict(stride=stride, padding=padding,
                           dilation=dilation, groups=groups)

            alpha = self.lrp_alpha
            beta = self.lrp_beta

            if tau < 0.1:
                r_alpha = self._conv_power_route(act, weight, bias, grad_out, mod, tau, eps)
                r_beta = self._conv_power_route(act, weight, bias, grad_out, mod, tau, eps, cross=True)
                lrp_grad = alpha * r_alpha - beta * r_beta
                return (lrp_grad,) if len(grad_input) == 1 else (lrp_grad,) + grad_input[1:]

            # Positive/negative decomposition
            x_pos = torch.relu(act)
            x_neg = torch.relu(-act)
            w_pos = torch.relu(weight)
            w_neg = torch.relu(-weight)

            # Temperature
            if tau != 1.0:
                inv_tau = 1.0 / tau
                x_pos_t = x_pos.pow(inv_tau)
                x_neg_t = x_neg.pow(inv_tau)
                w_pos_t = w_pos.pow(inv_tau)
                w_neg_t = w_neg.pow(inv_tau)
            else:
                x_pos_t = x_pos
                x_neg_t = x_neg
                w_pos_t = w_pos
                w_neg_t = w_neg

            # Bias pos/neg decomposition
            b_pos_t = None
            b_neg_t = None
            if bias is not None:
                b_pos_b = bias.clamp(min=0)
                b_neg_b = bias.clamp(max=0)
                if tau != 1.0:
                    b_pos_t = b_pos_b.pow(inv_tau)
                    b_neg_t = -((-b_neg_b).pow(inv_tau))
                else:
                    b_pos_t = b_pos_b
                    b_neg_t = b_neg_b

            # Same-sign denominator
            z_pos = (F.conv2d(x_pos_t, w_pos_t, bias=None, **conv_kw)
                     + F.conv2d(x_neg_t, w_neg_t, bias=None, **conv_kw))
            if b_pos_t is not None:
                z_pos = z_pos + b_pos_t.view(1, -1, 1, 1)

            # Cross-sign denominator
            z_neg = (F.conv2d(x_pos_t, w_neg_t, bias=None, **conv_kw)
                     + F.conv2d(x_neg_t, w_pos_t, bias=None, **conv_kw))
            if b_neg_t is not None:
                z_neg = z_neg - b_neg_t.view(1, -1, 1, 1)

            z_pos = self._stabilize(z_pos, eps)
            z_neg = self._stabilize(z_neg, eps)

            # Backproject via transposed convolution
            tp_kw = dict(stride=stride, padding=padding,
                         dilation=dilation, groups=groups,
                         output_padding=_conv_output_padding(
                             act.shape, z_pos.shape, mod.kernel_size,
                             stride, padding, dilation))

            # α path: same-sign routing
            s_pos = self._divide_relevance(grad_out, z_pos)
            r_alpha = (x_pos_t * F.conv_transpose2d(s_pos, w_pos_t, **tp_kw)
                       + x_neg_t * F.conv_transpose2d(s_pos, w_neg_t, **tp_kw))

            # β path: cross-sign routing
            s_neg = self._divide_relevance(grad_out, z_neg)
            r_beta = (x_pos_t * F.conv_transpose2d(s_neg, w_neg_t, **tp_kw)
                      + x_neg_t * F.conv_transpose2d(s_neg, w_pos_t, **tp_kw))

            lrp_grad = alpha * r_alpha - beta * r_beta

            if len(grad_input) == 1:
                return (lrp_grad,)
            else:
                return (lrp_grad,) + grad_input[1:]

        h_bwd = module.register_full_backward_hook(_bwd_hook)
        self._hooks.append(h_bwd)

    # -----------------------------------------------------------------
    # BatchNorm LRP hooks (merged BN — matches zennit's Epsilon on merged BN)
    # -----------------------------------------------------------------
    def _install_bn_lrp_hooks(self, name: str, module: nn.Module):
        """Apply epsilon-LRP to a (merged, now identity) BatchNorm module.

        After merge BN has gamma=1, beta=0, mean=0, var≈1, so it computes
        identity in forward.  But zennit still applies Epsilon to it, so we
        replicate that: z = x, s = R / (z + ε·sign(z)), R_in = x · s.
        """
        def _fwd_hook(mod, inp, out, mod_name=name):
            self._activations[mod_name] = inp[0].detach()
            return out.clone()

        h_fwd = module.register_forward_hook(_fwd_hook)
        self._hooks.append(h_fwd)

        def _bwd_hook(mod, grad_input, grad_output, mod_name=name):
            if mod_name not in self._activations:
                return None
            x = self._activations[mod_name]
            grad_out = grad_output[0]
            eps = self.lrp_eps
            z = self._stabilize(x, eps)
            s = self._divide_relevance(grad_out, z)
            new_grad = x * s
            if len(grad_input) == 1:
                return (new_grad,)
            return (new_grad,) + grad_input[1:]

        h_bwd = module.register_full_backward_hook(_bwd_hook)
        self._hooks.append(h_bwd)

    # -----------------------------------------------------------------
    # Activation layer hooks (GELU, ReLU)
    # -----------------------------------------------------------------
    def _install_gelu_hooks(self, name: str, module: nn.Module):
        """GELU backward: derivative at risk-modified z_eff."""

        def _fwd_hook(mod, inp, out, mod_name=name):
            self._activations[mod_name] = inp[0].detach()
            return out.clone()

        h_fwd = module.register_forward_hook(_fwd_hook)
        self._hooks.append(h_fwd)

        def _bwd_hook(mod, grad_input, grad_output, mod_name=name):
            if mod_name not in self._activations:
                return None

            pre_act = self._activations[mod_name]
            grad_out = grad_output[0]

            # Get ADF variance for this layer
            var = self._adf_variances.get(mod_name)
            if var is not None and (self.adf_lambda != 0.0 or self.force_adf_activation_gate):
                z_eff = self._compute_z_eff(pre_act, var)
                if self.adf_probabilistic:
                    # Smooth GELU derivative: gelu'(z) = Φ(z) + z·φ(z)
                    phi = torch.special.ndtr(z_eff)
                    pdf = _normal_pdf(z_eff)
                    gelu_deriv = phi + z_eff * pdf
                else:
                    # Hard threshold approximation (step function at z=0)
                    gelu_deriv = (z_eff > 0).float()
                new_grad = grad_out * gelu_deriv
            elif not self.adf_probabilistic:
                # Hard threshold requested — override with step function
                new_grad = grad_out * (pre_act > 0).float()
            else:
                # Pass() — identity gradient (standard LRP for activations)
                new_grad = grad_out

            if len(grad_input) == 1:
                return (new_grad,)
            else:
                return (new_grad,) + grad_input[1:]

        h_bwd = module.register_full_backward_hook(_bwd_hook)
        self._hooks.append(h_bwd)

    def _install_relu_hooks(self, name: str, module: nn.Module):
        """ReLU backward: firing probability Φ(z_eff/σ) via ADF variance."""

        def _fwd_hook(mod, inp, out, mod_name=name):
            self._activations[mod_name] = inp[0].detach()
            return out.clone()

        h_fwd = module.register_forward_hook(_fwd_hook)
        self._hooks.append(h_fwd)

        def _bwd_hook(mod, grad_input, grad_output, mod_name=name):
            if mod_name not in self._activations:
                return None

            pre_act = self._activations[mod_name]
            grad_out = grad_output[0]

            var = self._adf_variances.get(mod_name)
            if var is not None and (self.adf_lambda != 0.0 or self.force_adf_activation_gate):
                z_eff = self._compute_z_eff(pre_act, var)
                if self.adf_probabilistic:
                    std = torch.sqrt(var.clamp(min=1e-8))
                    relu_deriv = torch.special.ndtr(z_eff / std)
                else:
                    relu_deriv = (z_eff > 0).float()
                new_grad = grad_out * relu_deriv
                if len(grad_input) == 1:
                    return (new_grad,)
                return (new_grad,) + grad_input[1:]
            # No ADF risk — let autograd handle ReLU normally (0/1 indicator)
            return None

        h_bwd = module.register_full_backward_hook(_bwd_hook)
        self._hooks.append(h_bwd)

    # -----------------------------------------------------------------
    # LayerNorm hooks
    # -----------------------------------------------------------------
    def _install_layernorm_hooks(self, name: str, module: nn.Module):
        """LayerNorm: identity pass-through (detached statistics)."""

        def _bwd_hook(mod, grad_input, grad_output):
            if len(grad_input) == 3:
                return (grad_output[0],) + grad_input[1:]
            return grad_output

        h_bwd = module.register_full_backward_hook(_bwd_hook)
        self._hooks.append(h_bwd)

    # -----------------------------------------------------------------
    # Softmax hooks
    # -----------------------------------------------------------------
    def _install_softmax_hooks(self, name: str, module: nn.Module):
        """Attention softmax: no relevance ('v_only'); standard autograd
        ('gradient_fallback').

        With ``adf_softmax_lambda`` != 0 the pre-softmax logits are cached;
        the risk-shifted attention is computed in the attention @ Value
        backward hook, so the forward pass is unchanged.
        """
        if name not in self._softmax_modules:
            return

        if self.attention_mode == 'v_only':
            def _bwd_hook(mod, grad_input, grad_output):
                return (torch.zeros_like(grad_input[0]),)
            h_bwd = module.register_full_backward_hook(_bwd_hook)
            self._hooks.append(h_bwd)

        if self.adf_softmax_lambda != 0.0:
            _sm_name = name

            def _capture_logits(mod, inp, out, sm_name=_sm_name):
                self._activations[sm_name + ':logits'] = inp[0].detach()

            h_fwd = module.register_forward_hook(_capture_logits)
            self._hooks.append(h_fwd)

    def _compute_risk_attention(self, av_matmul_name: str, A_orig: Tensor) -> Optional[Tensor]:
        """Recompute attention weights with ADF-shifted logits for risk routing.

        Uses the cached pre-softmax logits of the softmax in the same
        attention block and the per-key-token ADF standard deviation of the
        Q projection (feature-averaged variance, see
        :meth:`_find_variance_for_softmax`).  The logits are shifted by
        lambda_sm * sigma_k and the softmax is recomputed, changing where
        relevance flows without altering the forward pass.  lambda_sm < 0
        moves attention away from high-uncertainty keys.
        """
        # Find the softmax in the same attention block
        parts = av_matmul_name.rsplit('.', 1)
        parent = parts[0] if len(parts) > 1 else ''

        logits = None
        for sm_name in self._softmax_modules:
            if sm_name.startswith(parent + '.'):
                logits = self._activations.get(sm_name + ':logits')
                break

        if logits is None:
            return None

        # Each key token has a different uncertainty, so the shift varies
        # per key and the softmax re-ranks (it is invariant only to uniform
        # shifts).
        var_key = None
        for sm_name in self._softmax_modules:
            if sm_name.startswith(parent + '.'):
                var_key = self._find_variance_for_softmax(sm_name)
                break

        if var_key is None:
            return None
        var = self._adf_variances.get(var_key)
        if var is None:
            return None

        # var shape: (B, T, D) from q_proj/k_proj
        # Reduce to per-token: (B, T) → broadcast to (B, 1, 1, T_k)
        # so each KEY gets its own uncertainty in the logit matrix.
        v_tok = var.mean(dim=-1)  # (B, T)
        std_tok = torch.sqrt(v_tok.clamp(min=1e-8))  # (B, T)
        # Broadcast: (B, T) → (B, 1, 1, T_k) to vary across key dim
        std_bcast = std_tok.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_k)
        try:
            std_bcast = std_bcast.expand_as(logits)  # (B, H, T_q, T_k)
        except RuntimeError:
            return None

        shifted_logits = logits + self.adf_softmax_lambda * std_bcast
        return F.softmax(shifted_logits, dim=-1)

    def _find_variance_for_softmax(self, softmax_name: str) -> Optional[str]:
        """Name of the Q projection of the attention block of ``softmax_name``.

        Its feature-averaged ADF variance at a key token is the key
        uncertainty of the risk-averse attention; None if the projection has
        no stored variance.
        """
        parts = softmax_name.rsplit('.', 1)
        parent = parts[0] if len(parts) > 1 else ''
        q_proj = parent + '.q_proj'
        return q_proj if q_proj in self._adf_variances else None

    # -----------------------------------------------------------------
    # DCMatMul hooks (attention)
    # -----------------------------------------------------------------
    def _install_matmul_hooks(self, name: str, module: nn.Module):
        """Install hooks on DCMatMul modules based on attention_mode."""
        # gradient_fallback: epsilon-LRP on all matmul ops (as zennit's
        # DCMatMul hooks)
        if self.attention_mode == 'gradient_fallback':
            self._install_matmul_lrp_hooks(name, module)
            return

        if name in self._qk_matmuls:
            self._install_qk_matmul_hooks(name, module)
        elif name in self._av_matmuls:
            self._install_av_matmul_hooks(name, module)

    def _install_matmul_lrp_hooks(self, name: str, module: nn.Module):
        """Epsilon-LRP for DCMatMul (relevance to the first operand)."""
        def _fwd_hook(mod, inp, out, mod_name=name):
            self._activations[mod_name] = tuple(
                t.detach() if isinstance(t, Tensor) else t for t in inp)

        h_fwd = module.register_forward_hook(_fwd_hook)
        self._hooks.append(h_fwd)

        def _bwd_hook(mod, grad_input, grad_output, mod_name=name):
            if mod_name not in self._activations:
                return None
            cached = self._activations[mod_name]
            A = cached[0]
            # Resolve B (handles DCMatMul's weight/operand/transpose modes)
            if mod._has_weight:
                B_raw = mod.weight.data
                B = B_raw.t() if not mod.transpose_b else B_raw
            elif len(cached) > 1 and cached[1] is not None:
                B_raw = cached[1]
                B = B_raw.transpose(-2, -1) if mod.transpose_b else B_raw
            elif mod._operand is not None:
                B_raw = mod._operand.detach()
                B = B_raw.transpose(-2, -1) if mod.transpose_b else B_raw
            else:
                return None

            grad_out = grad_output[0]
            eps = self.lrp_eps
            z = torch.matmul(A, B)
            # Stabilizer eps * sign(z), with the sign of exact zeros taken as positive
            z = z + eps * z.sign().clamp(min=1e-12)
            s = grad_out / z
            R_A = torch.matmul(s, B.transpose(-2, -1)) * A
            return (R_A,) + grad_input[1:]

        h_bwd = module.register_full_backward_hook(_bwd_hook)
        self._hooks.append(h_bwd)

    def _install_qk_matmul_hooks(self, name: str, module: nn.Module):
        """QK matmul: zero gradient to both Q and K."""
        def _bwd_hook(mod, grad_input, grad_output):
            return tuple(torch.zeros_like(g) if g is not None else None
                         for g in grad_input)
        h_bwd = module.register_full_backward_hook(_bwd_hook)
        self._hooks.append(h_bwd)

    def _merged_value_backward(self, av_name: str, A: torch.Tensor, V: torch.Tensor,
                               R_O: torch.Tensor, eps: float) -> torch.Tensor:
        """alpha-beta-LRP on the composite map X -> O = A (X W_v^T + b_v).

        Every Value entry keeps its two stream masses
        v~^+ = sum_e [x_e W_de]^+ + [b_d]^+ and v~^- = sum_e [x_e W_de]^- + [b_d]^-
        (the Stopping-Decomposition pair of the entry, computed from the Value
        projection's input) instead of being netted to v and re-split by its sign.
        The stream totals Z^kappa = A v~^kappa + eps are formed once for the whole
        block; the per-stream quantities S^kappa = A^T (R_O / Z^kappa) are handed to
        the Value projection, which routes stream kappa only to contributions of
        sign kappa.  Returns the relevance of the Value entries,
        alpha v~^+ S^+ - beta v~^- S^- (used only to keep autograd flowing).
        """
        parent = av_name.rsplit('.', 1)[0] if '.' in av_name else ''
        vp_name = f"{parent}.v_proj" if parent else "v_proj"
        vp = self.model.get_submodule(vp_name)
        x = self._activations.get(vp_name)
        if x is None or not isinstance(vp, nn.Linear):
            raise RuntimeError(f"av_rule='merged': no cached input of {vp_name}")
        tau = self.routing_temperature
        if tau < 0.1:
            # Keep the composite A·V normalization and the local Value
            # projection decomposition in log space.  At tau=0 the local
            # tie count is part of the A-weighted key competition; the paired
            # Linear hook then divides each selected key's share uniformly
            # among those tied input edges.
            x_pos, x_neg = torch.relu(x), torch.relu(-x)
            weight = vp.weight.data
            n_heads = V.shape[-3] if V.dim() == 4 else 1
            value_dim = V.shape[-1]
            A_heads = A.unsqueeze(1) if A.dim() == 3 else A
            R_heads = R_O.unsqueeze(1) if R_O.dim() == 3 else R_O
            if A_heads.shape[1] == 1 and n_heads > 1:
                A_heads = A_heads.expand(-1, n_heads, -1, -1)
            if R_heads.shape[1] == 1 and n_heads > 1:
                R_heads = R_heads.expand(-1, n_heads, -1, -1)

            def route_head(head, cross):
                start, stop = head * value_dim, (head + 1) * value_dim
                w = weight[start:stop]
                wp, wn = torch.relu(w), torch.relu(-w)
                if cross:
                    first = x_pos[:, :, None, :] * wn[None, None, :, :]
                    second = x_neg[:, :, None, :] * wp[None, None, :, :]
                    bias = None if vp.bias is None else (-vp.bias.data[start:stop]).clamp_min(0)
                else:
                    first = x_pos[:, :, None, :] * wp[None, None, :, :]
                    second = x_neg[:, :, None, :] * wn[None, None, :, :]
                    bias = None if vp.bias is None else vp.bias.data[start:stop].clamp_min(0)
                edges = torch.cat((first, second), dim=-1)
                if bias is not None:
                    edges = torch.cat((edges, bias[None, None, :, None].expand(edges.shape[0], edges.shape[1], -1, -1)), dim=-1)
                log_edges = torch.where(edges > 0, edges.log(), -torch.inf)
                A_head = A_heads[:, head:head + 1]
                R_head = R_heads[:, head:head + 1]
                if tau == 0:
                    local_max = log_edges.amax(dim=-1)
                    ties = (log_edges == local_max.unsqueeze(-1)) & torch.isfinite(local_max.unsqueeze(-1))
                    tie_count = ties.sum(dim=-1).to(log_edges.dtype).unsqueeze(1)
                    local_max = local_max.unsqueeze(1)
                    local_mass = torch.where(torch.isfinite(local_max), local_max.exp(), torch.zeros_like(local_max))
                    coeff = A_head.unsqueeze(-1) * tie_count.unsqueeze(-3)
                    key_share = _power_routing_shares(local_mass.unsqueeze(-3).expand_as(coeff), coeff,
                                                       0.0, eps, dim=-2)
                else:
                    local_log_total = torch.logsumexp(log_edges / tau, dim=-1).unsqueeze(1)
                    scores = A_head.unsqueeze(-1).log() + local_log_total.unsqueeze(-3)
                    norm = torch.logsumexp(scores, dim=-2, keepdim=True)
                    if eps > 0:
                        norm = torch.logaddexp(norm, scores.new_tensor(float(eps)).log())
                    key_share = torch.where(torch.isfinite(norm), torch.exp(scores - norm), torch.zeros_like(scores))
                s_heads = (key_share * R_head.unsqueeze(-2)).sum(dim=-3)
                return s_heads.squeeze(1)

            s_pos = torch.cat([route_head(head, False) for head in range(n_heads)], dim=-1)
            s_neg = torch.cat([route_head(head, True) for head in range(n_heads)], dim=-1)
            self._merged_value_relevance[vp_name] = (s_pos, s_neg)
            return torch.zeros_like(V)

        x_pos, x_neg = torch.relu(x), torch.relu(-x)
        w_pos, w_neg = torch.relu(vp.weight.data), torch.relu(-vp.weight.data)
        if tau != 1.0:
            inv_tau = 1.0 / tau
            x_pos, x_neg = x_pos.pow(inv_tau), x_neg.pow(inv_tau)
            w_pos, w_neg = w_pos.pow(inv_tau), w_neg.pow(inv_tau)
        v_pos = torch.matmul(x_pos, w_pos.t()) + torch.matmul(x_neg, w_neg.t())
        v_neg = torch.matmul(x_pos, w_neg.t()) + torch.matmul(x_neg, w_pos.t())
        if vp.bias is not None:
            b = vp.bias.data
            b_pos, b_neg = b.clamp(min=0), (-b).clamp(min=0)
            if tau != 1.0:
                b_pos, b_neg = b_pos.pow(inv_tau), b_neg.pow(inv_tau)
            v_pos, v_neg = v_pos + b_pos, v_neg + b_neg

        def to_heads(t):   # (B, S, H*d) -> layout of V, e.g. (B, H, S, d)
            if t.dim() == V.dim():
                return t
            return t.unflatten(-1, (V.shape[-3], V.shape[-1])).transpose(-3, -2)

        def from_heads(t):
            if t.dim() == x.dim():
                return t
            return t.transpose(-3, -2).flatten(-2)

        v_pos, v_neg = to_heads(v_pos), to_heads(v_neg)
        At = A.transpose(-2, -1)

        def route_stream(value_mass):
            denominator = self._stabilize(torch.matmul(A, value_mass), eps)
            scaled_relevance = self._divide_relevance(R_O, denominator)
            return torch.matmul(At, scaled_relevance)

        s_pos, s_neg = route_stream(v_pos), route_stream(v_neg)
        self._merged_value_relevance[vp_name] = (from_heads(s_pos), from_heads(s_neg))
        return self.lrp_alpha * v_pos * s_pos - self.lrp_beta * v_neg * s_neg

    def _install_av_matmul_hooks(self, name: str, module: nn.Module):
        """Attention @ Value matmul: the Value path is routed with the merged
        rule; the attention matrix A >= 0 is a fixed policy (no relevance).

        With ADF softmax risk, A is recomputed from the shifted logits,
        changing where relevance flows in the backward pass without
        altering the forward computation.
        """

        # Cache A (attention weights) and V
        def _fwd_hook(mod, inp, out, mod_name=name):
            if len(inp) >= 2:
                self._activations[f"{mod_name}:A"] = inp[0].detach()
                self._activations[f"{mod_name}:V"] = inp[1].detach()
            return out.clone()

        h_fwd = module.register_forward_hook(_fwd_hook)
        self._hooks.append(h_fwd)

        def _bwd_hook(mod, grad_input, grad_output, mod_name=name):
            A = self._activations.get(f"{mod_name}:A")
            V = self._activations.get(f"{mod_name}:V")

            # Apply ADF softmax risk: recompute attention from shifted logits
            if self.adf_softmax_lambda != 0.0 and A is not None:
                A_risk = self._compute_risk_attention(mod_name, A)
                if A_risk is not None:
                    A = A_risk
            if A is None or V is None:
                return None

            R_V = self._merged_value_backward(mod_name, A, V, grad_output[0], self.lrp_eps)
            # Zero gradient to A (attention weights are fixed policy)
            R_A = torch.zeros_like(A)
            if len(grad_input) >= 2:
                return (R_A, R_V) + grad_input[2:]
            return (R_A, R_V)

        h_bwd = module.register_full_backward_hook(_bwd_hook)
        self._hooks.append(h_bwd)


# =========================================================================
# Routing Game attribution method
# =========================================================================

@MethodRegistry.register
class RoutingGame(AttributionMethod):
    """Routing Game attribution: alpha-beta LRP with Gibbs routing
    temperature, ADF risk aversion and the RG attention rules (see the
    module docstring)."""

    @classmethod
    def config(cls) -> MethodConfig:
        return MethodConfig(
            name="routing_game",
            category=MethodCategory.GAME_THEORY,
            class_specific=True,
            supports_batch=True,
            requires_grad=True,
            per_layer=False,
            description=(
                "Routing game: hook-based alpha-beta LRP with ADF "
                "risk aversion and attention mode control"
            ),
            reference="game_theory.routing",
        )

    def __init__(self, model: nn.Module, device: torch.device):
        super().__init__(model, device)
        # LRP parameters: the alpha-beta-LRP-eps anchor of the paper
        self.lrp_alpha = 2.0
        self.lrp_beta = 1.0
        self.lrp_eps = 0.5
        # Routing temperature tau (Gibbs measure)
        self.routing_temperature = 1.0
        # Activation-side risk aversion lambda and input variance sigma^2
        self.adf_lambda = 0.0
        self.adf_input_variance = 1.0
        # Smooth (Phi) or hard (step) gate of the risk-modified activations
        self.adf_probabilistic = True
        # Diagnostic-only matched-gate control.  The public/default path keeps
        # this false, so lambda=0 remains the ordinary RG anchor.
        self._force_adf_activation_gate = False
        # Softmax risk lambda_sm (attention routing)
        self.adf_softmax_lambda = 0.0
        # Attention mode: 'v_only' or 'gradient_fallback'
        self.attention_mode = 'v_only'
        # Value-path rule of O = A·V (composite rule, see _merged_value_backward)
        self.av_rule = 'merged'
        # Fan-in entropy bias lambda_ent: b_j -= lambda_ent * H~_j
        self.entropy_lambda = 0.0

    def attribute(
        self,
        x: Tensor,
        target: Tensor,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        if layers and any(layer != "input" for layer in layers):
            raise NotImplementedError("routing_game: per-layer maps are not available")
        self.model.eval()
        B = x.size(0)

        logger.info(
            "Routing Game: α=%.2f β=%.2f ε=%.3f τ=%.2f λ=%.3f λ_sm=%.3f "
            "attn_mode=%s av_rule=%s",
            self.lrp_alpha, self.lrp_beta, self.lrp_eps,
            self.routing_temperature, self.adf_lambda, self.adf_softmax_lambda,
            self.attention_mode, self.av_rule,
        )

        hook_mgr = _RoutingGameHookManager(
            self.model,
            lrp_alpha=self.lrp_alpha,
            lrp_beta=self.lrp_beta,
            lrp_eps=self.lrp_eps,
            routing_temperature=self.routing_temperature,
            adf_lambda=self.adf_lambda,
            adf_input_variance=self.adf_input_variance,
            adf_probabilistic=self.adf_probabilistic,
            force_adf_activation_gate=self._force_adf_activation_gate,
            adf_softmax_lambda=self.adf_softmax_lambda,
            attention_mode=self.attention_mode,
            av_rule=self.av_rule,
        )

        # Step 1: ADF forward pass to collect variances (needed only when a
        # risk parameter is active).
        if (self.adf_lambda != 0 or self.adf_softmax_lambda != 0
                or self._force_adf_activation_gate):
            hook_mgr.run_adf_pass(x)

        # Step 2: LRP hooks.
        hook_mgr.install()

        try:
            # Step 3: Forward + backward (with the optional entropy bias shift)
            from bwgame.game_theory.entropy import entropy_bias_adapt_context
            from contextlib import nullcontext

            entropy_ctx = (entropy_bias_adapt_context(self.model, self.entropy_lambda)
                           if self.entropy_lambda != 0.0 else nullcontext())

            with entropy_ctx:
                x_input = x.detach().clone().requires_grad_(True)
                output = self.model(x_input)

                one_hot = torch.zeros_like(output)
                for b in range(B):
                    one_hot[b, target[b]] = 1.0

                output.backward(gradient=one_hot)

            # Step 4: Extract relevance from input gradient
            relevance = x_input.grad
            spatial = relevance.sum(dim=1)  # Single signed map

        finally:
            hook_mgr.remove()

        return {"input": tensor_to_numpy(spatial)}
