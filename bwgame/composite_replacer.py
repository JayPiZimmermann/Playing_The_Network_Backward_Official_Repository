"""Decompose composite nn modules into sub-modules for DC compatibility.

Composite modules like nn.MultiheadAttention wrap multiple operations
(linear projections, matmul, softmax) inside a single opaque forward().
This module replaces them with equivalent implementations using explicit
sub-modules that torch.fx can trace and the DC patcher can reach.

Usage:
    from bwgame.composite_replacer import decompose_composites
    decompose_composites(model)  # In-place replacement
"""

import torch
import torch.nn as nn
from typing import Optional


class DecomposedMultiheadAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention using explicit sub-modules.

    Decomposes the opaque F.multi_head_attention_forward into:
    - nn.Linear for Q, K, V, output projections
    - torch.matmul for attention scores (replaced by DCMatMul via fx)
    - torch.softmax for attention weights (replaced by nn.Softmax via fx)
    - nn.Dropout for attention dropout

    Respects ALL nn.MultiheadAttention constructor parameters:
    - embed_dim, num_heads, dropout, bias
    - add_bias_kv, add_zero_attn
    - kdim, vdim (cross-attention with different key/value dims)
    - batch_first
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, \
            "embed_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5
        self.batch_first = batch_first
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        # Explicit Q, K, V projection layers (replacing packed in_proj_weight)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(dropout)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = nn.Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.register_parameter('bias_k', None)
            self.register_parameter('bias_v', None)

    def forward(self, query, key, value,
                key_padding_mask=None, need_weights=False,
                attn_mask=None, average_attn_weights=True,
                is_causal=False):
        """
        Forward pass matching nn.MultiheadAttention signature exactly.

        All parameters are accepted for API compatibility. The mask parameters
        (key_padding_mask, attn_mask, is_causal) and need_weights are accepted
        but not used - the torch.fx traced version only includes the core
        attention path. This is appropriate for DC decomposition where masks
        would need separate DC-aware handling anyway.
        """
        # batch_first: input is (N, S, E). Otherwise (S, N, E) → transpose.
        # self.batch_first is a Python bool → concrete at trace time, no issue.
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Project Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # add_bias_kv: append learned bias as extra key/value position
        # self.add_bias_kv is a Python bool → concrete at trace time.
        if self.add_bias_kv:
            K = torch.cat([K, self.bias_k.repeat(K.size(0), 1, 1)], dim=1)
            V = torch.cat([V, self.bias_v.repeat(V.size(0), 1, 1)], dim=1)

        # Reshape for multi-head: split embed_dim into (num_heads, head_dim)
        # (N, S, E) → (N, S, H, D) → (N, H, S, D)
        Q = Q.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)
        K = K.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)
        V = V.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)

        # add_zero_attn: append zeros as extra key/value position
        # self.add_zero_attn is a Python bool → concrete at trace time.
        if self.add_zero_attn:
            zero = torch.zeros_like(K[:, :, :1, :])
            K = torch.cat([K, zero], dim=-2)
            V = torch.cat([V, zero], dim=-2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum
        output = torch.matmul(attn_weights, V)

        # Merge heads: (N, H, L, D) → (N, L, H, D) → (N, L, E)
        output = output.transpose(-3, -2).flatten(-2)

        # Output projection
        output = self.out_proj(output)

        # Convert back if not batch_first
        if not self.batch_first:
            output = output.transpose(0, 1)

        return output, None

    @classmethod
    def from_multihead_attention(cls, mha: nn.MultiheadAttention):
        """Create from existing nn.MultiheadAttention, copying all weights."""
        decomposed = cls(
            embed_dim=mha.embed_dim,
            num_heads=mha.num_heads,
            dropout=mha.dropout,
            bias=mha.in_proj_bias is not None,
            add_bias_kv=mha.bias_k is not None,
            add_zero_attn=mha.add_zero_attn,
            kdim=mha.kdim,
            vdim=mha.vdim,
            batch_first=mha.batch_first,
        )

        E = mha.embed_dim

        with torch.no_grad():
            # Q, K, V projection weights
            if mha._qkv_same_embed_dim:
                # Packed in_proj_weight: (3*E, E) → split into Q, K, V
                decomposed.q_proj.weight.copy_(mha.in_proj_weight[:E])
                decomposed.k_proj.weight.copy_(mha.in_proj_weight[E:2*E])
                decomposed.v_proj.weight.copy_(mha.in_proj_weight[2*E:])
            else:
                # Separate projection weights
                decomposed.q_proj.weight.copy_(mha.q_proj_weight)
                decomposed.k_proj.weight.copy_(mha.k_proj_weight)
                decomposed.v_proj.weight.copy_(mha.v_proj_weight)

            # Q, K, V biases (always packed as in_proj_bias regardless of _qkv_same_embed_dim)
            if mha.in_proj_bias is not None:
                decomposed.q_proj.bias.copy_(mha.in_proj_bias[:E])
                decomposed.k_proj.bias.copy_(mha.in_proj_bias[E:2*E])
                decomposed.v_proj.bias.copy_(mha.in_proj_bias[2*E:])

            # Output projection
            decomposed.out_proj.weight.copy_(mha.out_proj.weight)
            if mha.out_proj.bias is not None:
                decomposed.out_proj.bias.copy_(mha.out_proj.bias)

            # Bias K/V (for add_bias_kv)
            if mha.bias_k is not None:
                decomposed.bias_k.copy_(mha.bias_k)
                decomposed.bias_v.copy_(mha.bias_v)

        # Move decomposed module to the same device as the original
        device = next(mha.parameters()).device
        decomposed = decomposed.to(device)

        return decomposed


class DecomposedTimmAttention(nn.Module):
    """
    Drop-in replacement for timm's Attention using explicit sub-modules.

    Splits the combined qkv Linear into separate q_proj, k_proj, v_proj,
    and replaces the fused scaled_dot_product_attention with explicit
    torch.matmul and torch.softmax calls (which torch.fx can then replace
    with DCMatMul and nn.Softmax modules).

    Preserves all original sub-modules (q_norm, k_norm, attn_drop, norm,
    proj, proj_drop) so they appear correctly in the layer tree.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        scale: float,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.scale = scale

        # Separate Q, K, V projections (split from combined qkv)
        self.q_proj = nn.Linear(dim, self.attn_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, self.attn_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, self.attn_dim, bias=qkv_bias)

        # Kept from original timm Attention
        self.q_norm = nn.Identity()
        self.k_norm = nn.Identity()
        self.attn_drop = nn.Dropout(0)
        self.norm = nn.Identity()
        self.proj = nn.Linear(self.attn_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(0)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape

        # Project Q, K, V separately
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head: (B, N, attn_dim) → (B, N, H, D) → (B, H, N, D)
        q = q.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        k = k.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        v = v.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)

        # Optional QK normalization (Identity for most DINO models)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Scaled dot-product attention (explicit, not fused)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum
        output = torch.matmul(attn_weights, v)

        # Merge heads: (B, H, N, D) → (B, N, H, D) → (B, N, attn_dim)
        output = output.transpose(1, 2).flatten(-2)

        # Scale normalization (Identity for most DINO models)
        output = self.norm(output)

        # Output projection
        output = self.proj(output)
        output = self.proj_drop(output)

        return output

    @classmethod
    def from_timm_attention(cls, attn) -> "DecomposedTimmAttention":
        """Create from existing timm Attention, copying all weights."""
        dim = attn.qkv.in_features
        attn_dim = attn.num_heads * attn.head_dim
        dim_out = attn.proj.out_features

        decomposed = cls(
            dim=dim,
            num_heads=attn.num_heads,
            head_dim=attn.head_dim,
            scale=attn.scale,
            qkv_bias=attn.qkv.bias is not None,
            proj_bias=attn.proj.bias is not None,
        )
        # Fix output dim if it differs from input dim
        if dim_out != dim:
            decomposed.proj = nn.Linear(attn_dim, dim_out,
                                        bias=attn.proj.bias is not None)

        with torch.no_grad():
            # Split qkv weights: (3*attn_dim, dim) → Q, K, V
            qkv_w = attn.qkv.weight
            decomposed.q_proj.weight.copy_(qkv_w[:attn_dim])
            decomposed.k_proj.weight.copy_(qkv_w[attn_dim:2*attn_dim])
            decomposed.v_proj.weight.copy_(qkv_w[2*attn_dim:])

            if attn.qkv.bias is not None:
                qkv_b = attn.qkv.bias
                decomposed.q_proj.bias.copy_(qkv_b[:attn_dim])
                decomposed.k_proj.bias.copy_(qkv_b[attn_dim:2*attn_dim])
                decomposed.v_proj.bias.copy_(qkv_b[2*attn_dim:])

            # Copy output projection
            decomposed.proj.weight.copy_(attn.proj.weight)
            if attn.proj.bias is not None:
                decomposed.proj.bias.copy_(attn.proj.bias)

        # Copy sub-modules directly (preserves norms, dropout rates)
        decomposed.q_norm = attn.q_norm
        decomposed.k_norm = attn.k_norm
        decomposed.attn_drop = attn.attn_drop
        decomposed.proj_drop = attn.proj_drop
        decomposed.norm = attn.norm

        # Move decomposed module to the same device as the original
        device = next(attn.parameters()).device
        decomposed = decomposed.to(device)

        return decomposed


def _is_timm_attention(module: nn.Module) -> bool:
    """Detect timm's Attention class by duck-typing."""
    return (
        hasattr(module, 'qkv') and isinstance(module.qkv, nn.Linear) and
        hasattr(module, 'num_heads') and hasattr(module, 'head_dim') and
        hasattr(module, 'proj') and isinstance(module.proj, nn.Linear) and
        hasattr(module, 'attn_drop') and
        not isinstance(module, DecomposedTimmAttention)
    )


def decompose_composites(model: nn.Module) -> nn.Module:
    """
    Replace composite modules with decomposed equivalents (in-place).

    Currently handles:
    - nn.MultiheadAttention → DecomposedMultiheadAttention
    - timm Attention → DecomposedTimmAttention

    This must run BEFORE torch.fx tracing so that the decomposed modules'
    internal operations (matmul, softmax) are visible to the tracer.
    """
    _decompose_recursive(model)
    return model


def _decompose_recursive(module: nn.Module):
    """Recursively find and replace composite modules."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name,
                    DecomposedMultiheadAttention.from_multihead_attention(child))
        elif _is_timm_attention(child):
            setattr(module, name,
                    DecomposedTimmAttention.from_timm_attention(child))
        else:
            _decompose_recursive(child)
