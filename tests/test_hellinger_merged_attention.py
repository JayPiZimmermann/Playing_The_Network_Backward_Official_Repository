"""Hellinger trajectory distance through an attention block (RG kernel).

The RG kernel routes the attention product O = A V, V = x W_v^T + b_v, with
the merged Value law: at O_td the trajectory splits 1/2-1/2 into the streams
kappa (the - stream switches the player), moves to the Value entry (k, d)
with probability A_tk v~^kappa_kd / sum_k' A_tk' v~^kappa_k'd,
v~^kappa_kd = sum_e [x_ke W_de]^kappa, and stays in stream kappa: it moves to
the projection input (k, e) with probability [x_ke W_de]^kappa / v~^kappa_kd.

Temperature tau raises A and every signed product to 1/tau; the bias is not
a routing target (as in every RG linear kernel).  The reference enumerates
every trajectory of a tiny block (read-out Linear at token 0 -> attention ->
Value projection -> input) with its probability under each model.  A
trajectory is identified by its streams and units (the player labels are a
function of the streams); absorption at a stream with zero total is a
separate cemetery trajectory per (unit, stream).  The dynamic program must
reproduce the per-input masses, the Bhattacharyya sub-mass
sum_paths sqrt(P_A P_B) and the cemetery masses.

A residual block with ``nn.MultiheadAttention`` (decomposed by the
diagnostic) must give the same result as the same block with its attention
matrix supplied as a constant: the sweep follows the Value path only and
the residual state waits for the block input.  CPU only, fast.
"""

import copy
import math

import pytest
import torch
import torch.nn as nn

from bwgame.functional_replacer import replace_functional_with_modules
from bwgame.operations.matmul import DCMatMul
from bwgame.xai.methods.hellinger import HellingerDiagnostic

DEAD = "cemetery"
EPS = 1e-12            # dead-stream threshold of the dynamic program
N_TOKENS, D_IN, N_OUT = 5, 6, 3
TARGET = 1


class TinyAttention(nn.Module):
    """Single-head block, constant attention from buffers, read-out at token 0."""

    heads = 1

    def __init__(self, d_in=D_IN, d=6, n_tokens=N_TOKENS, n_out=N_OUT):
        super().__init__()
        self.v_proj = nn.Linear(d_in, d, bias=True)
        self.qk = DCMatMul()
        self.softmax = nn.Softmax(dim=-1)
        self.av = DCMatMul()
        self.out_proj = nn.Linear(d, n_out, bias=True)
        self.register_buffer("q", torch.randn(n_tokens, 4))
        self.register_buffer("k", torch.randn(n_tokens, 4))

    def forward(self, x):   # (B, T, d_in)
        A = self.softmax(self.qk(self.q, self.k.t())).expand(x.shape[0], -1, -1)
        O = self.av(A, self.v_proj(x))
        return self.out_proj(O)[:, 0, :]


class TinyMultiHeadAttention(nn.Module):
    """Multi-head block laid out as the decomposed nn.MultiheadAttention."""

    heads = 2

    def __init__(self, d_in=D_IN, d_head=3, n_tokens=N_TOKENS, n_out=N_OUT):
        super().__init__()
        self.dh = d_head
        self.v_proj = nn.Linear(d_in, self.heads * d_head, bias=True)
        self.qk = DCMatMul()
        self.softmax = nn.Softmax(dim=-1)
        self.av = DCMatMul()
        self.out_proj = nn.Linear(self.heads * d_head, n_out, bias=True)
        self.register_buffer("q", torch.randn(self.heads, n_tokens, 4))
        self.register_buffer("k", torch.randn(self.heads, n_tokens, 4))

    def forward(self, x):   # (B, T, d_in)
        V = self.v_proj(x).unflatten(-1, (self.heads, self.dh)).transpose(-3, -2)
        A = self.softmax(self.qk(self.q, self.k.transpose(-2, -1))).expand(x.shape[0], -1, -1, -1)
        O = self.av(A, V).transpose(-3, -2).flatten(-2)
        return self.out_proj(O)[:, 0, :]


class NormedValueAttention(TinyAttention):
    """The Value operand passes through a LayerNorm: not a view of v_proj."""

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(6)

    def forward(self, x):
        A = self.softmax(self.qk(self.q, self.k.t())).expand(x.shape[0], -1, -1)
        O = self.av(A, self.norm(self.v_proj(x)))
        return self.out_proj(O)[:, 0, :]


# ---- explicit trajectory laws ------------------------------------------------

def _block_tensors(model):
    """(attention (H, T, T), W_v, b_v, W_o) of a tiny block."""
    A = torch.softmax(model.q @ model.k.transpose(-2, -1), dim=-1)
    A = A.reshape(model.heads, N_TOKENS, N_TOKENS)
    return (A.detach(), model.v_proj.weight.detach(), model.v_proj.bias.detach(),
            model.out_proj.weight.detach())


def _streams(c, tau):
    """(([c]^+)^(1/tau), ([c]^-)^(1/tau)) of a tensor of signed products."""
    return c.clamp(min=0) ** (1.0 / tau), (-c).clamp(min=0) ** (1.0 / tau)


def _readout_step(X, W_v, b_v, A, W_o, tau):
    """First step (RG linear kernel of the read-out at token 0).

    Returns the explicit law of this step as a list of
    (kappa1, j, probability) and the cemetery entries."""
    H = A.shape[0]
    dh = W_v.shape[0] // H
    V = X @ W_v.t() + b_v                                              # (T, D)
    O0 = torch.stack([A[j // dh, 0] @ V[:, j] for j in range(W_v.shape[0])])
    steps, dead = [], {}
    for s1, u1 in enumerate(_streams(O0 * W_o[TARGET], tau)):
        if u1.sum() <= EPS:
            dead[(s1, DEAD)] = 0.5
            continue
        for j in range(W_v.shape[0]):
            steps.append((s1, j, 0.5 * (u1[j] / u1.sum()).item()))
    return V, steps, dead


def _merged_law(X, W_v, b_v, A, W_o, tau):
    """{trajectory: probability} of the merged law."""
    H = A.shape[0]
    dh = W_v.shape[0] // H
    T, E = X.shape
    _, steps, law = _readout_step(X, W_v, b_v, A, W_o, tau)
    for s1, j, p1 in steps:
        a = A[j // dh, 0] ** (1.0 / tau)                                # (T,)
        for s2, u in enumerate(_streams(X * W_v[j], tau)):              # u[k, e]
            v = u.sum(dim=1)                                            # v~[k, j]
            Z = (a * v).sum()
            if Z <= EPS:
                law[(s1, j, s2, DEAD)] = p1 * 0.5
                continue
            for k in range(T):
                for e in range(E):
                    if u[k, e] > 0:
                        p2 = 0.5 * (a[k] * v[k] / Z).item()
                        law[(s1, j, s2, k, e)] = p1 * p2 * (u[k, e] / v[k]).item()
    return law


def _occupations(law_A, law_B, T, E):
    """Per-input masses, Bhattacharyya sub-mass and cemetery (mu_A, mu_B, beta)."""
    mu_A = torch.zeros(T, E, dtype=torch.float64)
    mu_B = torch.zeros(T, E, dtype=torch.float64)
    beta = torch.zeros(T, E, dtype=torch.float64)
    cem = torch.zeros(3, dtype=torch.float64)
    for path in set(law_A) | set(law_B):
        pA, pB = law_A.get(path, 0.0), law_B.get(path, 0.0)
        g = math.sqrt(pA * pB)
        if path[-1] == DEAD:
            cem += torch.tensor([pA, pB, g], dtype=torch.float64)
        else:
            k, e = path[-2], path[-1]
            mu_A[k, e] += pA
            mu_B[k, e] += pB
            beta[k, e] += g
    return mu_A, mu_B, beta, cem


# ---- helpers -------------------------------------------------------------------

def _pair(model_cls, seed, scale=0.7):
    """Model A and a perturbed copy B (weights and attention both differ)."""
    torch.manual_seed(seed)
    model_A = model_cls().double().eval()
    model_B = copy.deepcopy(model_A)
    with torch.no_grad():
        for t in list(model_B.parameters()) + list(model_B.buffers()):
            t.add_(scale * torch.randn_like(t))
    return model_A, model_B


def _check_against_law(model_A, model_B, x_A, x_B, tau, law_fn):
    tensors_A, tensors_B = _block_tensors(model_A), _block_tensors(model_B)
    r = HellingerDiagnostic(model_A, model_B, game="rg", tau=tau).compute(
        x_A, x_B, target=TARGET,
    )
    laws = []
    for b in range(x_A.shape[0]):
        A_a, Wv_a, bv_a, Wo_a = tensors_A
        A_b, Wv_b, bv_b, Wo_b = tensors_B
        law_A = law_fn(x_A[b], Wv_a, bv_a, A_a, Wo_a, tau)
        law_B = law_fn(x_B[b], Wv_b, bv_b, A_b, Wo_b, tau)
        laws.append((law_A, law_B))
        assert math.isclose(sum(law_A.values()), 1.0, rel_tol=1e-12)
        assert math.isclose(sum(law_B.values()), 1.0, rel_tol=1e-12)
        mu_A, mu_B, beta, cem = _occupations(law_A, law_B, *x_A.shape[1:])
        close = dict(rtol=1e-9, atol=1e-12)
        assert torch.allclose(r["mu_A_input"][b].double(), mu_A, **close)
        assert torch.allclose(r["mu_B_input"][b].double(), mu_B, **close)
        assert torch.allclose(r["beta_input"][b].double(), beta, **close)
        dp_cem = torch.stack([r["mu_A_cem"][b, 0], r["mu_B_cem"][b, 0], r["beta_cem"][b, 0]]).double()
        assert torch.allclose(dp_cem, cem, **close)
        bc = beta.sum() + cem[2]
        assert math.isclose(r["bhattacharyya"][b].item(), bc.item(), rel_tol=1e-9, abs_tol=1e-12)
        assert math.isclose(r["h_squared"][b].item(), 1.0 - bc.item(), rel_tol=1e-9, abs_tol=1e-12)
    return r, laws


# ---- merged law ------------------------------------------------------------------

@pytest.mark.parametrize("model_cls", [TinyAttention, TinyMultiHeadAttention])
@pytest.mark.parametrize("tau", [1.0, 0.5, 2.0])
def test_merged_matches_trajectory_enumeration(model_cls, tau):
    model_A, model_B = _pair(model_cls, seed=0)
    torch.manual_seed(1)
    x_A = torch.randn(2, N_TOKENS, D_IN, dtype=torch.float64)
    x_B = x_A + 0.5 * torch.randn_like(x_A)
    r, _ = _check_against_law(model_A, model_B, x_A, x_B, tau, _merged_law)
    assert (r["h_squared"] > 1e-3).all()


@pytest.mark.parametrize("model_cls", [TinyAttention, TinyMultiHeadAttention])
def test_merged_dead_streams_go_to_cemetery(model_cls):
    """Non-negative inputs and rows of W_v with one sign: the - stream of those
    Value features has zero total.  B differs from A only in row 0 of W_v, so
    feature 0 is dead in A only and feature 1 is dead in both models (and
    reached through the same read-out stream, hence a positive overlap)."""
    model_A, _ = _pair(model_cls, seed=2)
    with torch.no_grad():
        model_A.v_proj.weight[0].abs_()
        model_A.v_proj.weight[1].abs_()
    model_B = copy.deepcopy(model_A)
    with torch.no_grad():
        model_B.v_proj.weight[0, ::2].neg_()
    torch.manual_seed(3)
    x = torch.randn(1, N_TOKENS, D_IN, dtype=torch.float64).abs()
    r, [(law_A, law_B)] = _check_against_law(model_A, model_B, x, x, 1.0, _merged_law)
    dead_A = {p[:3] for p in law_A if len(p) == 4 and p[-1] == DEAD}
    dead_B = {p[:3] for p in law_B if len(p) == 4 and p[-1] == DEAD}
    assert {(s1, j, 1) for s1 in (0, 1) for j in (0, 1)} <= dead_A
    assert {(s1, 1, 1) for s1 in (0, 1)} <= dead_B
    assert not any(j == 0 for _, j, _ in dead_B)
    assert r["mu_A_cem"].item() > 0.0 and r["mu_B_cem"].item() > 0.0
    assert r["beta_cem"].item() > 0.0


def test_merged_identical_models_zero_distance():
    model, _ = _pair(TinyMultiHeadAttention, seed=4)
    x = torch.randn(2, N_TOKENS, D_IN, dtype=torch.float64)
    r = HellingerDiagnostic(model, game="rg").compute(x, x, target=TARGET)
    assert torch.allclose(r["h_squared"], torch.zeros_like(r["h_squared"]), atol=1e-12)
    total = r["mu_A_input"].flatten(1).sum(1) + r["mu_A_cem"][:, 0]
    assert torch.allclose(total.double(), torch.ones(2, dtype=torch.float64), atol=1e-12)


# ---- Value operand ----------------------------------------------------------------

def test_merged_requires_value_view_of_linear():
    torch.manual_seed(9)
    model = NormedValueAttention().double().eval()
    x = torch.randn(1, N_TOKENS, D_IN, dtype=torch.float64)
    with pytest.raises(ValueError, match="not a layout view of an nn.Linear output"):
        HellingerDiagnostic(model, game="rg").compute(x, target=TARGET)


# ---- nn.MultiheadAttention block ---------------------------------------------------

DIM = 8   # 2 heads of width 4: the key transpose has T * DIM entries per sample


class MHABlock(nn.Module):
    """Pre-norm residual attention block, read-out at token 0."""

    def __init__(self, dim=DIM, heads=2, n_out=N_OUT):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.head = nn.Linear(dim, n_out)

    def forward(self, x):
        h = self.ln(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        return self.head((x + a)[:, 0])


class ConstAttentionBlock(nn.Module):
    """The block of ``MHABlock`` with the attention matrix given as a constant."""

    def __init__(self, block, A):
        super().__init__()
        dim = block.attn.embed_dim
        self.heads, self.dh = block.attn.num_heads, dim // block.attn.num_heads
        self.ln = copy.deepcopy(block.ln)
        self.v_proj = nn.Linear(dim, dim)
        with torch.no_grad():
            self.v_proj.weight.copy_(block.attn.in_proj_weight[2 * dim:])
            self.v_proj.bias.copy_(block.attn.in_proj_bias[2 * dim:])
        self.av = DCMatMul()
        self.out_proj = copy.deepcopy(block.attn.out_proj)
        self.head = copy.deepcopy(block.head)
        self.register_buffer("A", A)

    def forward(self, x):
        h = self.ln(x)
        V = self.v_proj(h).unflatten(-1, (self.heads, self.dh)).transpose(-3, -2)
        O = self.av(self.A, V).transpose(-3, -2).flatten(-2)
        return self.head((x + self.out_proj(O))[:, 0])


@pytest.mark.parametrize("game", ["adf", "rg"])
def test_multihead_attention_block_matches_constant_attention(game):
    torch.manual_seed(11)
    block_A = MHABlock().double().eval()
    block_B = copy.deepcopy(block_A)
    with torch.no_grad():
        for p in block_B.parameters():
            p.add_(0.3 * torch.randn_like(p))
        # float32-representable weights: the decomposition below passes them
        # through float32 projections
        for p in list(block_A.parameters()) + list(block_B.parameters()):
            p.copy_(p.float())
    x_A = torch.randn(2, N_TOKENS, DIM, dtype=torch.float64)
    x_B = torch.randn(2, N_TOKENS, DIM, dtype=torch.float64)
    twins = []
    for block, x in ((block_A, x_A), (block_B, x_B)):
        with torch.no_grad():
            h = block.ln(x)
            _, A = block.attn(h, h, h, need_weights=True, average_attn_weights=False)
        twins.append(ConstAttentionBlock(block, A).double().eval())
        # Decompose here (the diagnostic's own call is then a no-op) and
        # restore float64: the decomposed projections are built in float32.
        replace_functional_with_modules(block, inplace=True)
        block.double()

    r = HellingerDiagnostic(block_A, block_B, game=game).compute(x_A, x_B, target=TARGET)
    r_ref = HellingerDiagnostic(twins[0], twins[1], game=game).compute(
        x_A, x_B, target=TARGET,
    )
    for key in ("mu_A_input", "mu_B_input", "beta_input", "mu_A_cem", "mu_B_cem", "beta_cem", "h_squared"):
        assert torch.allclose(r[key].double(), r_ref[key].double(), rtol=1e-9, atol=1e-12), key
    visited = set(r["per_module_h2"])
    assert "attn.v_proj" in visited
    assert not visited & {"attn.q_proj", "attn.k_proj"}
