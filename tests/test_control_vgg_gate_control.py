"""Invariants for the diagnostic VGG RG gate-matched control."""

import torch


def _relu_gradient(x, *, risk, force_gate):
    from bwgame.xai.methods.routing_game import _RoutingGameHookManager

    model = torch.nn.Sequential(torch.nn.ReLU())
    manager = _RoutingGameHookManager(
        model,
        adf_lambda=risk,
        adf_input_variance=1.0,
        adf_probabilistic=True,
        force_adf_activation_gate=force_gate,
    )
    # Isolate the backward gate from the ADF forward implementation: the full
    # experiment runs that pass, while this unit test fixes sigma=1 exactly.
    manager._adf_variances["0"] = torch.ones_like(x)
    manager.install()
    try:
        x_input = x.detach().clone().requires_grad_(True)
        model(x_input).sum().backward()
        return x_input.grad.detach()
    finally:
        manager.remove()


def test_forced_zero_risk_gate_is_probit_of_original_preactivation():
    x = torch.tensor([[-1.0, 0.0, 1.0]])
    got = _relu_gradient(x, risk=0.0, force_gate=True)
    # With sigma=1 and lambda=0, the diagnostic gate is Phi(z/sigma).  This
    # also checks that it uses the captured ordinary preactivation z.
    torch.testing.assert_close(got, torch.special.ndtr(x))


def test_ordinary_zero_risk_path_remains_the_hard_relu_gradient():
    x = torch.tensor([[-1.0, 0.0, 1.0]])
    got = _relu_gradient(x, risk=0.0, force_gate=False)
    torch.testing.assert_close(got, torch.tensor([[0.0, 0.0, 1.0]]))


def test_force_flag_does_not_change_nonzero_risk_path():
    x = torch.tensor([[-1.0, 0.0, 1.0]])
    legacy = _relu_gradient(x, risk=-2.0, force_gate=False)
    forced = _relu_gradient(x, risk=-2.0, force_gate=True)
    torch.testing.assert_close(forced, legacy, rtol=0.0, atol=0.0)


def test_force_flag_triggers_variance_pass_at_zero_risk(monkeypatch):
    from bwgame.xai.methods.routing_game import RoutingGame, _RoutingGameHookManager

    calls = []
    monkeypatch.setattr(
        _RoutingGameHookManager,
        "run_adf_pass",
        lambda self, x: calls.append(tuple(x.shape)),
    )
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1)
    ).eval()
    x, target = torch.ones(1, 2), torch.tensor([0])

    ordinary = RoutingGame(model, torch.device("cpu"))
    ordinary.attribute(x, target)
    assert calls == []

    forced = RoutingGame(model, torch.device("cpu"))
    forced._force_adf_activation_gate = True
    forced.attribute(x, target)
    assert calls == [(1, 2)]

