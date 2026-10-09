# Validation selection: vgg16 (imagenet-s, val)

Grid: `experiments/grids/imagenet_s_val_vgg16.csv` - 635 of 635 configurations evaluated.

loc = min rank-sum over {AL, PG, TK}; faith = min rank-sum over {PF5, PF20, Sel} (ranks within each method).

| Method | Criterion | Row label | Config | AL | PG | TK | PF5 | PF20 | Sel | Loc rank-sum | Faith rank-sum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gradient (1/1) | fixed | `gradient__cr-maxabs` | cr=max_abs | 0.532 | 0.760 | 0.699 | 14.13 | 10.55 | 6.184 | 3.0 | 3.0 |
| SmGrad (3/3) | loc | `smoothgrad__spread-0.1` | spread=0.1 | 0.530 | 0.740 | 0.715 | 12.96 | 7.93 | 5.445 | 3.0 | 3.0 |
| SmGrad (3/3) | faith | `smoothgrad__spread-0.1` | spread=0.1 | 0.530 | 0.740 | 0.715 | 12.96 | 7.93 | 5.445 | 3.0 | 3.0 |
| IntGrad (4/4) | loc | `intgrad__steps-20` | steps=20 | 0.557 | 0.800 | 0.702 | 15.15 | 11.64 | 6.123 | 6.0 | 3.0 |
| IntGrad (4/4) | faith | `intgrad__steps-20` | steps=20 | 0.557 | 0.800 | 0.702 | 15.15 | 11.64 | 6.123 | 6.0 | 3.0 |
| DeepLift (1/1) | fixed | `deeplift__rule-rescale` | rule=rescale | 0.546 | 0.680 | 0.639 | 12.19 | 8.65 | 5.714 | 3.0 | 3.0 |
| LRP-eps (7/7) | loc | `lrp_eps__eps-0.5` | eps=0.5 | 0.601 | 0.800 | 0.718 | 11.82 | 7.72 | 5.304 | 6.0 | 9.0 |
| LRP-eps (7/7) | faith | `lrp_eps__eps-1` | eps=1.0 | 0.630 | 0.780 | 0.727 | 11.60 | 7.23 | 5.130 | 6.0 | 4.0 |
| LRP-gamma (30/30) | loc | `lrp_gamma__comp-epsilongammabox__eps-0.5__gamma-0.25` | comp=epsilon_gamma_box, eps=0.5, gamma=0.25 | 0.614 | 0.880 | 0.719 | 14.19 | 9.05 | 4.858 | 13.0 | 43.0 |
| LRP-gamma (30/30) | faith | `lrp_gamma__eps-0.25__gamma-0.1` | eps=0.25, gamma=0.1 | 0.610 | 0.840 | 0.689 | 11.31 | 6.83 | 4.793 | 36.0 | 5.0 |
| Grad-CAM (1/1) | fixed | `gcam__layer-features.29` | layer=features.29 | 0.589 | 0.820 | 0.826 | 15.95 | 11.69 | 5.407 | 3.0 | 3.0 |
| GCAM++ (1/1) | fixed | `gcampp__layer-features.29` | layer=features.29 | 0.558 | 0.800 | 0.806 | 15.97 | 12.08 | 5.517 | 3.0 | 3.0 |
| LCAM (1/1) | fixed | `lcam__layer-features.29` | layer=features.29 | 0.553 | 0.760 | 0.770 | 15.69 | 11.10 | 5.117 | 3.0 | 3.0 |
| LIME (1/1) | fixed | `lime__N-1000__fill-segmentmean__kd-binary__kw-0.25__seg-80__sp-slic__sur-ridge` | fill=segment_mean, kd=binary, kw=0.25, N=1000, sp=slic, seg=80, sur=ridge | 0.588 | 0.920 | 0.785 | 16.38 | 12.32 | 5.572 | 3.0 | 3.0 |
| KernelSHAP (1/1) | fixed | `kernelshap__N-auto__fill-imagemean__seg-80__sp-slic` | fill=image_mean, N=auto, sp=slic, seg=80 | 0.557 | 0.980 | 0.828 | 16.21 | 12.43 | 5.896 | 3.0 | 3.0 |
| SG (40/40) | loc | `sg__gate-on__lam-m1__var-5` | var=5.0, lam=-1.0, gate=True | 0.568 | 0.880 | 0.731 | 16.34 | 12.46 | 6.008 | 37.0 | 10.0 |
| SG (40/40) | faith | `sg__gate-off__lam-m0.3__var-1` | var=1.0, lam=-0.3, gate=False | 0.561 | 0.780 | 0.733 | 16.23 | 12.12 | 6.021 | 49.5 | 6.0 |
| RG (544/544) | loc | `rg__eps-0.1__lam-0__tau-0.3` | lam=0.0, eps=0.1, tau=0.3 | 0.663 | 0.800 | 0.660 | 13.32 | 8.55 | 5.181 | 78.0 | 194.0 |
| RG (544/544) | faith | `rg__eps-0.1__lam-0__tau-0.8` | lam=0.0, eps=0.1, tau=0.8 | 0.624 | 0.760 | 0.671 | 12.79 | 8.04 | 4.861 | 173.5 | 5.0 |
| RG (544/544) | anchor | `rg__eps-0.5__lam-0__tau-1` | lam=0.0, eps=0.5, tau=1.0 | 0.604 | 0.800 | 0.671 | 13.14 | 8.48 | 4.949 | 339.0 | 60.0 |

**Inadmissible rows (5, excluded from ranking)**: `sg__gate-off__lam-m5__var-1` (all-zero map on 1 image(s)), `sg__gate-off__lam-m5__var-2` (all-zero map on 24 image(s)), `sg__gate-off__lam-m5__var-5` (all-zero map on 50 image(s)), `sg__gate-off__lam-m5__var-10` (all-zero map on 50 image(s)), `sg__gate-off__lam-m2__var-10` (all-zero map on 10 image(s))
