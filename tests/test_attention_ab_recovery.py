"""RG attention value routing recovers alpha-beta-LRP-eps on the conditioned map.

For an attention block X -> V = X W_v^T + b_v -> O = A V -> Y = O W_o^T + b_o
with a fixed, row-stochastic attention matrix A >= 0 (the "conditioned"
map, A detached), the Routing Game at tau = 1 (``av_rule='merged'``) must
return exactly the relevance of alpha-beta-LRP-eps on the composite map
X -> O = A (X W_v^T + b_v) (alpha = 2, beta = 1, eps = 0.5), the target rule
of the attention recovery theorem.  The references below are independent,
direct implementations with explicit contribution tensors, not the hook
code.  CPU only, fast.
"""

import pytest
import torch
import torch.nn as nn

from bwgame.operations.matmul import DCMatMul
from bwgame.xai.methods.routing_game import _RoutingGameHookManager

ALPHA, BETA, EPS = 2.0, 1.0, 0.5


class TinyAttention(nn.Module):
    """Single attention block with a constant attention matrix.

    Two DCMatMul modules in the same parent: the first is identified as the
    QK product and the second as the attention @ value product, exactly as
    in a decomposed nn.MultiheadAttention.
    """

    def __init__(self, d: int = 6, n_tokens: int = 5, n_out: int = 3):
        super().__init__()
        self.v_proj = nn.Linear(d, d, bias=True)
        self.qk = DCMatMul()
        self.softmax = nn.Softmax(dim=-1)
        self.av = DCMatMul()
        self.out_proj = nn.Linear(d, n_out, bias=True)
        self.register_buffer("q", torch.randn(n_tokens, 4))
        self.register_buffer("k", torch.randn(n_tokens, 4))

    def attention(self) -> torch.Tensor:
        # softmax of random logits: row-stochastic, >= 0, independent of X
        return self.softmax(self.qk(self.q, self.k.t()))

    def forward(self, x):  # x: (B, T, d)
        A = self.attention().expand(x.shape[0], -1, -1)
        O = self.av(A, self.v_proj(x))
        return self.out_proj(O)[:, 0, :]  # read-out at token 0


# ---- independent reference ------------------------------------------------

def _ab_linear(x, W, b, R_out, alpha, beta, eps):
    """alpha-beta-LRP-eps for y = x W^T + b on rows of x (bias absorbs)."""
    contrib = x[..., None, :] * W                       # (..., out, in)
    cp, cn = contrib.clamp(min=0), (-contrib).clamp(min=0)
    zp = cp.sum(-1) + b.clamp(min=0) + eps              # (..., out)
    zn = cn.sum(-1) + (-b).clamp(min=0) + eps
    return (alpha * (cp * (R_out / zp)[..., None]).sum(-2)
            - beta * (cn * (R_out / zn)[..., None]).sum(-2))


def _ab_attention_value(A, V, R_O, alpha, beta, eps):
    """alpha-beta-LRP-eps for O = A V with A >= 0 fixed (relevance to V only)."""
    contrib = A[:, :, None] * V[None, :, :]             # (i, j, d): A_ij V_jd
    cp, cn = contrib.clamp(min=0), (-contrib).clamp(min=0)
    zp = cp.sum(1) + eps                                 # (i, d)
    zn = cn.sum(1) + eps
    r_pos = (cp * (R_O / zp)[:, None, :]).sum(0)         # (j, d)
    r_neg = (cn * (R_O / zn)[:, None, :]).sum(0)
    return alpha * r_pos - beta * r_neg


def _layerwise_reference(model, x, target):
    """alpha-beta-LRP-eps applied layer by layer (Value entries netted by sign)."""
    A = model.attention().detach()
    W_v, b_v = model.v_proj.weight.detach(), model.v_proj.bias.detach()
    W_o, b_o = model.out_proj.weight.detach(), model.out_proj.bias.detach()
    X = x[0]
    V = X @ W_v.t() + b_v
    O = A @ V
    R_Y = torch.zeros(O.shape[0], W_o.shape[0], dtype=x.dtype)
    R_Y[0, target] = 1.0                                 # seed at token 0
    R_O = _ab_linear(O, W_o, b_o, R_Y, ALPHA, BETA, EPS)
    R_V = _ab_attention_value(A, V, R_O, ALPHA, BETA, EPS)
    return _ab_linear(X, W_v, b_v, R_V, ALPHA, BETA, EPS)[None]


def _routing_game(model, x, target, eps=EPS):
    mgr = _RoutingGameHookManager(
        model=model, lrp_alpha=ALPHA, lrp_beta=BETA, lrp_eps=eps,
        routing_temperature=1.0, adf_lambda=0.0, attention_mode="v_only",
        av_rule="merged",
    )
    mgr.install()
    try:
        x_in = x.detach().clone().requires_grad_(True)
        out = model(x_in)
        seed = torch.zeros_like(out)
        seed[0, target] = 1.0
        out.backward(gradient=seed)
        return x_in.grad.detach()
    finally:
        mgr.remove()


@pytest.fixture
def setup():
    torch.manual_seed(0)
    model = TinyAttention().double().eval()
    x = torch.randn(1, 5, 6, dtype=torch.float64)       # signed inputs
    return model, x


@pytest.mark.parametrize("av_rule", ["bogus", ""])
def test_invalid_av_rule_rejected(av_rule):
    with pytest.raises(ValueError):
        _RoutingGameHookManager(model=nn.Linear(2, 2), av_rule=av_rule)


def test_routing_game_default_is_merged():
    from bwgame.xai.methods.routing_game import RoutingGame
    assert RoutingGame(nn.Linear(2, 2), torch.device("cpu")).av_rule == "merged"


# ---- merged (composite) Value rule --------------------------------------------

def _composite_reference(A, X, W, b, R_O, alpha, beta, eps):
    """alpha-beta-LRP-eps on the composite map X -> O = A (X W^T + b), one head.

    A: (T, T) >= 0, X: (T, E), W: (D, E), b: (D,), R_O: (T, D).  Every product
    A_tk X_ke W_de is one contribution; the bias enters the stream totals only.
    """
    c = X[:, None, :] * W[None, :, :]                       # (k, d, e): X_ke W_de
    cp, cn = c.clamp(min=0), (-c).clamp(min=0)
    vp = cp.sum(-1) + b.clamp(min=0)                        # (k, d) stream masses
    vn = cn.sum(-1) + (-b).clamp(min=0)
    zp, zn = A @ vp + eps, A @ vn + eps                     # (t, d)
    sp, sn = A.t() @ (R_O / zp), A.t() @ (R_O / zn)         # (k, d)
    return alpha * (cp * sp[:, :, None]).sum(1) - beta * (cn * sn[:, :, None]).sum(1)


class TinyMultiHeadAttention(nn.Module):
    """Multi-head block laid out as the decomposed nn.MultiheadAttention."""

    def __init__(self, d_in=6, heads=2, d_head=3, n_tokens=5, n_out=3):
        super().__init__()
        self.h, self.dh = heads, d_head
        self.v_proj = nn.Linear(d_in, heads * d_head, bias=True)
        self.qk = DCMatMul()
        self.softmax = nn.Softmax(dim=-1)
        self.av = DCMatMul()
        self.out_proj = nn.Linear(heads * d_head, n_out, bias=True)
        self.register_buffer("q", torch.randn(heads, n_tokens, 4))
        self.register_buffer("k", torch.randn(heads, n_tokens, 4))

    def attention(self):
        return self.softmax(self.qk(self.q, self.k.transpose(-2, -1)))   # (H, T, T)

    def forward(self, x):   # (B, T, d_in)
        V = self.v_proj(x).unflatten(-1, (self.h, self.dh)).transpose(-3, -2)
        A = self.attention().expand(x.shape[0], -1, -1, -1)
        O = self.av(A, V).transpose(-3, -2).flatten(-2)
        return self.out_proj(O)[:, 0, :]


@pytest.mark.parametrize("target", [0, 1, 2])
def test_merged_matches_composite_rule(setup, target):
    model, x = setup
    got = _routing_game(model, x, target)
    A = model.attention().detach()
    W_v, b_v = model.v_proj.weight.detach(), model.v_proj.bias.detach()
    W_o, b_o = model.out_proj.weight.detach(), model.out_proj.bias.detach()
    X = x[0]
    O = A @ (X @ W_v.t() + b_v)
    R_Y = torch.zeros(O.shape[0], W_o.shape[0], dtype=x.dtype)
    R_Y[0, target] = 1.0
    R_O = _ab_linear(O, W_o, b_o, R_Y, ALPHA, BETA, EPS)
    ref = _composite_reference(A, X, W_v, b_v, R_O, ALPHA, BETA, EPS)[None]
    assert torch.allclose(got, ref, rtol=1e-8, atol=1e-10), (got - ref).abs().max()
    assert not torch.allclose(got, _layerwise_reference(model, x, target))


def test_merged_multi_head_matches_composite_rule():
    torch.manual_seed(1)
    model = TinyMultiHeadAttention().double().eval()
    x = torch.randn(1, 5, 6, dtype=torch.float64)
    got = _routing_game(model, x, 1)
    A = model.attention().detach()                           # (H, T, T)
    W_v, b_v = model.v_proj.weight.detach(), model.v_proj.bias.detach()
    W_o, b_o = model.out_proj.weight.detach(), model.out_proj.bias.detach()
    X = x[0]
    V = (X @ W_v.t() + b_v).unflatten(-1, (2, 3)).transpose(0, 1)       # (H, T, dh)
    O = (A @ V).transpose(0, 1).flatten(-2)                              # (T, H*dh)
    R_Y = torch.zeros(O.shape[0], W_o.shape[0], dtype=x.dtype)
    R_Y[0, 1] = 1.0
    R_O = _ab_linear(O, W_o, b_o, R_Y, ALPHA, BETA, EPS).unflatten(-1, (2, 3)).transpose(0, 1)
    ref = sum(_composite_reference(A[h], X, W_v[3 * h:3 * h + 3], b_v[3 * h:3 * h + 3],
                                   R_O[h], ALPHA, BETA, EPS) for h in range(2))[None]
    assert torch.allclose(got, ref, rtol=1e-8, atol=1e-10), (got - ref).abs().max()


class FixedValueBlock(nn.Module):
    def __init__(self, attention, weight, bias):
        super().__init__()
        self.v_proj = nn.Linear(weight.shape[1], weight.shape[0], bias=True).double()
        with torch.no_grad():
            self.v_proj.weight.copy_(weight)
            self.v_proj.bias.copy_(bias)
        self.qk = DCMatMul()
        self.av = DCMatMul()
        self.register_buffer("attention", attention)

    def forward(self, x):
        self.qk(self.attention, torch.eye(self.attention.shape[-1], dtype=x.dtype))
        return self.av(self.attention, self.v_proj(x))


def _explicit_composite_relevance(attention, x, weight, bias, seed, eps):
    result = torch.zeros_like(x)
    for t in range(attention.shape[0]):
        for d in range(weight.shape[0]):
            products = attention[t, :, None] * x * weight[d]
            for sign, multiplier in ((1, ALPHA), (-1, -BETA)):
                contributions = (sign * products).clamp(min=0)
                total = contributions.sum() + (sign * bias[d]).clamp(min=0) + eps
                if total > 0:
                    result += multiplier * contributions / total * seed[t, d]
    return result


@pytest.mark.parametrize("eps", [0.0, 0.5])
@pytest.mark.parametrize("all_zero", [False, True])
def test_merged_empty_streams_match_completed_composite_rule(eps, all_zero):
    attention = torch.tensor([[1.0, 0.0], [0.25, 0.75]], dtype=torch.float64)
    x = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 1.0]], dtype=torch.float64)
    weight = torch.tensor([[1.0, 2.0, 0.0], [-1.0, 0.0, -2.0],
                           [0.0, -1.0, 0.0], [1.0, -1.0, 0.5]], dtype=torch.float64)
    bias = torch.tensor([0.5, -0.25, 0.0, 0.0], dtype=torch.float64)
    if all_zero:
        weight.zero_()
        bias.zero_()
    seed = torch.tensor([[1.0, -2.0, 3.0, -0.5], [-1.0, 0.0, 0.5, 2.0]],
                        dtype=torch.float64)
    model = FixedValueBlock(attention, weight, bias).eval()
    manager = _RoutingGameHookManager(model, lrp_alpha=ALPHA, lrp_beta=BETA, lrp_eps=eps)
    manager.install()
    try:
        inputs = x.unsqueeze(0).clone().requires_grad_(True)
        model(inputs).backward(seed.unsqueeze(0))
        got = inputs.grad[0]
    finally:
        manager.remove()
    expected = _explicit_composite_relevance(attention, x, weight, bias, seed, eps)
    assert torch.isfinite(got).all()
    assert torch.allclose(got, expected, rtol=1e-12, atol=1e-12)
    if all_zero:
        assert torch.count_nonzero(got) == 0


@pytest.mark.parametrize("eps", [1e-8, 0.5, 2.0])
def test_merged_positive_stabilizer_preserves_previous_arithmetic(setup, eps):
    model, x = setup
    manager = _RoutingGameHookManager(model, lrp_alpha=ALPHA, lrp_beta=BETA, lrp_eps=eps)
    manager._activations["v_proj"] = x
    attention = model.attention().detach()
    values = model.v_proj(x).detach()
    seed = torch.linspace(-1, 1, values.numel(), dtype=x.dtype).reshape_as(values)
    weight, bias = model.v_proj.weight.detach(), model.v_proj.bias.detach()
    xp, xn = torch.relu(x), torch.relu(-x)
    wp, wn = torch.relu(weight), torch.relu(-weight)
    vp = xp @ wp.t() + xn @ wn.t() + bias.clamp(min=0)
    vn = xp @ wn.t() + xn @ wp.t() + (-bias).clamp(min=0)
    sp = attention.t() @ (seed / manager._stabilize(attention @ vp, eps))
    sn = attention.t() @ (seed / manager._stabilize(attention @ vn, eps))
    expected = ALPHA * vp * sp - BETA * vn * sn
    got = manager._merged_value_backward("av", attention, values, seed, eps)
    cached_positive, cached_negative = manager._merged_value_relevance["v_proj"]
    assert torch.equal(got, expected)
    assert torch.equal(cached_positive, sp)
    assert torch.equal(cached_negative, sn)


@pytest.mark.parametrize("all_zero", [False, True])
def test_zero_stabilizer_attention_with_affine_readout(all_zero):
    model = TinyAttention(d=2, n_tokens=2, n_out=1).double().eval()
    with torch.no_grad():
        model.q.zero_()
        model.k.zero_()
        for layer in (model.v_proj, model.out_proj):
            layer.weight.fill_(0.0 if all_zero else 1.0)
            layer.bias.zero_()
    x = torch.ones(1, 2, 2, dtype=torch.float64)
    got = _routing_game(model, x, 0, eps=0.0)
    expected = torch.zeros_like(x) if all_zero else torch.ones_like(x)
    assert torch.isfinite(got).all()
    assert torch.equal(got, expected)
