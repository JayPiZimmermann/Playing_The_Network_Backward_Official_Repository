"""zennit LRP calls leave the model's modules unchanged.

zennit's batch-norm merge and parameter modifiers replace ``weight`` /
``bias`` of the merged convolutions.  After an LRP call the registered
parameters must again be the tensors the forward pass reads, so in-place
parameter changes (cascading randomization) change the model's output.
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("zennit")
torchvision = pytest.importorskip("torchvision")

from bwgame.xai.methods.lrp_library import GammaLRP, LRPEpsilon  # noqa: E402


def _model():
    torch.manual_seed(0)
    model = torchvision.models.resnet18(weights=None, num_classes=10).eval()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0.5, 1.5)
            nn.init.uniform_(m.bias, -0.1, 0.1)
            m.running_mean.uniform_(-0.1, 0.1)
            m.running_var.uniform_(0.5, 1.5)
    return model


@pytest.mark.parametrize("cls", [GammaLRP, LRPEpsilon])
def test_lrp_call_leaves_modules_unchanged(cls):
    model = _model()
    x = torch.randn(2, 3, 64, 64)
    target = torch.tensor([1, 2])
    params_before = dict(model.named_parameters())
    buffers_before = dict(model.named_buffers())
    with torch.no_grad():
        y0 = model(x)

    cls(model, torch.device("cpu")).attribute(x, target)

    assert dict(model.named_parameters()).keys() == params_before.keys()
    assert all(p is params_before[n] for n, p in model.named_parameters())
    assert all(b is buffers_before[n] for n, b in model.named_buffers())
    for m in model.modules():
        shadowed = (set(m._parameters) | set(m._buffers)) & set(m.__dict__)
        assert not shadowed, (type(m).__name__, shadowed)
    with torch.no_grad():
        assert torch.equal(model(x), y0)


def test_parameter_change_after_lrp_call_reaches_forward():
    """Re-initializing ``parameters()``, as the cascading randomization does, changes the output."""
    model = _model()
    x = torch.randn(1, 3, 64, 64)
    GammaLRP(model, torch.device("cpu")).attribute(x, torch.tensor([3]))
    with torch.no_grad():
        y0 = model(x)
        for p in model.layer4[1].conv2.parameters():
            nn.init.normal_(p)
        y1 = model(x)
    assert not torch.allclose(y0, y1)
