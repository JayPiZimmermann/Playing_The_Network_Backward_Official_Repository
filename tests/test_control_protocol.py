"""Scientific invariants of the new paired experiments."""
import copy
import numpy as np
import torch
import quantus
from reproduce.control_common import isolated_rng, perturb_parameters, digest_state
from reproduce.control_robustness import RobustnessMetrics


def test_joint_robustness_matches_independent_quantus():
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(12,2)).eval()
    x=np.linspace(.1,1.2,12,dtype=np.float32).reshape(1,3,2,2)
    def explanation(model,inputs,targets,**kwargs):
        return (inputs**2).sum(1)
    common=dict(model=model,x_batch=x,y_batch=np.array([0]),a_batch=explanation(None,x,None),
                explain_func=explanation,device="cpu",channel_first=True,batch_size=1)
    cfg=dict(nr_samples=4,abs=True,normalise=True,disable_warnings=True)
    with isolated_rng(23):
        mx=quantus.MaxSensitivity(**cfg)(**common)
    with isolated_rng(23):
        av=quantus.AvgSensitivity(**cfg)(**common)
    joint=RobustnessMetrics(**cfg)
    with isolated_rng(23):
        got=joint(**common)
    np.testing.assert_allclose(got,mx,rtol=1e-6,atol=1e-7)
    np.testing.assert_allclose(joint.extra["AvgS"],av[0],rtol=1e-6,atol=1e-7)


def test_randomization_independent_of_explanations_and_layer_order():
    model=torch.nn.Sequential(torch.nn.Linear(4,5),torch.nn.ReLU(),torch.nn.Linear(5,2))
    original=copy.deepcopy(model.state_dict())
    perturb_parameters(model,original,["0","2"],17)
    first=digest_state(model)
    with isolated_rng(99):
        torch.rand(500); np.random.rand(400)
    perturb_parameters(model,original,["2","0"],17)
    assert digest_state(model)==first
    perturb_parameters(model,original,["0","2"],17,scale=0.)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in original.items())


def test_sg_signed_reconstruction_and_published_component_are_distinct():
    from reproduce.control_sg_identity import identity
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,1,bias=False)).double()
    with torch.no_grad(): model[1].weight.copy_(torch.tensor([[1.,-2.,3.,-4.]]))
    result=identity(model,torch.ones(1,1,2,2,dtype=torch.float64),torch.tensor([0]))
    assert result["signed_reconstruction"]["allclose"]
    assert not result["published_pp"]["allclose"]


def test_sg_layerwise_centering_prevents_deep_signed_stream_cancellation():
    from bwgame.dc_context import dc_decomposition
    # Orthogonal signed weights keep the actual derivative bounded, while
    # absolute path masses grow by about 1.4 per layer.
    model=torch.nn.Sequential(*[torch.nn.Linear(2,2,bias=False) for _ in range(80)])
    with torch.no_grad():
        for layer in model:
            layer.weight.copy_(torch.tensor([[.8,.6],[-.6,.8]]))
    x=torch.ones(1,2,requires_grad=True)
    ref=torch.autograd.grad(model(x)[0,0],x)[0]
    errors={}
    for alpha in (0.,.5):
        with dc_decomposition(model,x.detach(),backprop_mode="stopping",alpha=alpha,
                              adf_probabilistic=False) as dc:
            dc.output[0,0].backward()
        errors[alpha]=float((dc.reconstruct_gradient()-ref).norm())
    assert errors[0.]>.1
    assert errors[.5]<1e-5


def test_sg_original_forward_masks_with_autograd_zero_convention():
    from reproduce.control_sg_debug import trace
    model=torch.nn.Sequential(torch.nn.Linear(2,2,bias=False),torch.nn.ReLU(),
                              torch.nn.Linear(2,1,bias=False)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(2))
        model[2].weight.fill_(1.)
    x=torch.tensor([[0.,1.]],dtype=torch.float64)
    ordinary=trace(model,x,torch.tensor([0]),alpha=.5,capture=True)
    strict=trace(model,x,torch.tensor([0]),alpha=.5,capture=False,
                 original_pools=True,strict_zero=True)
    assert ordinary["reconstruction_max_abs"]==1.
    assert strict["reconstruction_max_abs"]==0.


def test_power_sharpening_preserves_ranking_but_changes_mass():
    from reproduce.control_sharpening import sharpen_metrics
    a=np.array([[[1.,2.],[3.,4.]]]); mask=np.array([[False,False],[False,True]])
    plain=sharpen_metrics(a,mask,1.)
    sharp=sharpen_metrics(a,mask,4.)
    assert sharp["AL"]>plain["AL"]
    assert sharp["PG"]==plain["PG"]
    assert sharp["TK"]==plain["TK"]


def test_sharpening_topk_uses_paper_k1000_not_mask_area():
    from reproduce.control_sharpening import sharpen_metrics
    a=np.arange(1.,1002.).reshape(1,1,1001)
    mask=np.zeros((1,1001),dtype=bool);mask[0,-1]=True
    result=sharpen_metrics(a,mask,1.)
    assert result["TK"]==.001


def test_insertion_deletion_endpoints_and_zero_baseline_auc():
    from reproduce.control_auc import full_curves
    model=torch.nn.Sequential(torch.nn.Flatten(),torch.nn.Linear(4,2,bias=False))
    x=torch.tensor([[[[0.,1.],[2.,3.]]]])
    c=full_curves(model,x,torch.tensor([0]),np.array([[[1.,2.],[3.,4.]]]),steps=4)
    np.testing.assert_allclose(c["deletion_logits"][0],c["insertion_logits"][-1])
    np.testing.assert_allclose(c["deletion_logits"][-1],c["insertion_logits"][0])
    assert c["fractions"]==[0.,.25,.5,.75,1.]
