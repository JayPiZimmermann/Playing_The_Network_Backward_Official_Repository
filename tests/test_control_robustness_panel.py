import numpy as np
from reproduce.control_robustness_panel import (
    bootstrap_mean, draw_noise, noise_seed, stability_vector, summarize_prefix)


def test_noise_is_paired_nested_and_variance_matched():
    shape=(1,3,128,128); r=.2
    u=draw_noise(shape,"uniform",r,noise_seed(42,"image","uniform",r,0))
    g=draw_noise(shape,"gaussian",r,noise_seed(42,"image","gaussian",r,0))
    assert np.array_equal(u,draw_noise(shape,"uniform",r,noise_seed(42,"image","uniform",r,0)))
    assert abs(u.std()-r/np.sqrt(3))<.002
    assert abs(g.std()-r/np.sqrt(3))<.002
    assert not np.array_equal(u,g)


def test_metrics_and_prefixes():
    a=np.array([[[1.,2.],[3.,4.]]]); b=a.copy()
    got=stability_vector(a,b)
    assert got["sensitivity"]==0 and got["mass_tv"]==0
    assert np.isclose(got["cosine"],1) and np.isclose(got["spearman"],1)
    vectors={k:list(range(200)) for k in (
        "sensitivity","cosine","spearman","top10_overlap","mass_tv",
        "prediction_agreement","target_probability_change")}
    s=summarize_prefix(vectors,50)
    assert s["sensitivity"]["max"]==49 and s["sensitivity"]["n"]==50
    assert s["sensitivity"]["q95"]<50
    b=bootstrap_mean([1,2,3])
    assert b["mean"]==2 and b["n"]==3 and b["ci95"][0]<=2<=b["ci95"][1]
