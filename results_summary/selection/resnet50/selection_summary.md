# Validation selection: resnet50 (imagenet-s, val)

Grid: `experiments/grids/imagenet_s_val_resnet50.csv` - 595 of 595 configurations evaluated.

loc = min rank-sum over {AL, PG, TK}; faith = min rank-sum over {PF5, PF20, Sel} (ranks within each method).

| Method | Criterion | Row label | Config | AL | PG | TK | PF5 | PF20 | Sel | Loc rank-sum | Faith rank-sum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gradient (1/1) | fixed | `gradient__cr-maxabs` | cr=max_abs | 0.497 | 0.580 | 0.583 | 6.31 | 6.02 | 4.839 | 3.0 | 3.0 |
| SmGrad (3/3) | loc | `smoothgrad__spread-0.15` | spread=0.15 | 0.555 | 0.820 | 0.755 | 6.16 | 5.18 | 4.384 | 4.0 | 5.0 |
| SmGrad (3/3) | faith | `smoothgrad__spread-0.1` | spread=0.1 | 0.543 | 0.800 | 0.711 | 6.08 | 5.27 | 4.363 | 8.0 | 4.0 |
| IntGrad (4/4) | loc | `intgrad__steps-100` | steps=100 | 0.528 | 0.680 | 0.607 | 6.31 | 5.94 | 4.864 | 3.0 | 6.0 |
| IntGrad (4/4) | faith | `intgrad__steps-50` | steps=50 | 0.523 | 0.560 | 0.596 | 6.24 | 5.96 | 4.946 | 7.5 | 6.0 |
| DeepLift (1/1) | fixed | `deeplift__rule-rescale` | rule=rescale | 0.544 | 0.680 | 0.638 | 6.02 | 5.37 | 4.305 | 3.0 | 3.0 |
| LRP-eps (7/7) | loc | `lrp_eps__eps-0.5` | eps=0.5 | 0.603 | 0.700 | 0.693 | 6.02 | 5.27 | 4.096 | 5.5 | 9.0 |
| LRP-eps (7/7) | faith | `lrp_eps__eps-2` | eps=2.0 | 0.624 | 0.680 | 0.657 | 5.98 | 4.93 | 4.037 | 8.0 | 4.0 |
| LRP-gamma (30/30) | loc | `lrp_gamma__comp-epsilongammabox__eps-0.5__gamma-2` | comp=epsilon_gamma_box, eps=0.5, gamma=2.0 | 0.686 | 0.860 | 0.784 | 5.74 | 4.51 | 3.743 | 12.5 | 33.0 |
| LRP-gamma (30/30) | faith | `lrp_gamma__comp-epsilongammabox__eps-0.5__gamma-0.5` | comp=epsilon_gamma_box, eps=0.5, gamma=0.5 | 0.679 | 0.820 | 0.762 | 5.72 | 4.53 | 3.734 | 36.0 | 22.5 |
| Grad-CAM (1/1) | fixed | `gcam__layer-layer4` | layer=layer4 | 0.647 | 0.860 | 0.849 | 6.43 | 6.07 | 4.269 | 3.0 | 3.0 |
| GCAM++ (1/1) | fixed | `gcampp__layer-layer4` | layer=layer4 | 0.614 | 0.780 | 0.797 | 6.47 | 6.13 | 4.432 | 3.0 | 3.0 |
| LCAM (1/1) | fixed | `lcam__layer-layer4` | layer=layer4 | 0.632 | 0.840 | 0.840 | 6.43 | 6.12 | 4.348 | 3.0 | 3.0 |
| LIME (1/1) | fixed | `lime__N-1000__fill-segmentmean__kd-binary__kw-0.25__seg-80__sp-slic__sur-ridge` | fill=segment_mean, kd=binary, kw=0.25, N=1000, sp=slic, seg=80, sur=ridge | 0.554 | 0.960 | 0.709 | 6.17 | 5.83 | 4.593 | 3.0 | 3.0 |
| KernelSHAP (1/1) | fixed | `kernelshap__N-auto__fill-imagemean__seg-80__sp-slic` | fill=image_mean, N=auto, sp=slic, seg=80 | 0.560 | 0.940 | 0.803 | 6.11 | 5.65 | 4.521 | 3.0 | 3.0 |
| RG (544/544) | loc | `rg__eps-1__lam-0__tau-0.9` | lam=0.0, eps=1.0, tau=0.9 | 0.699 | 0.780 | 0.727 | 5.93 | 4.76 | 3.860 | 19.0 | 224.0 |
| RG (544/544) | faith | `rg__eps-0.25__lam-m1__tau-1__var-1` | var=1.0, lam=-1.0, eps=0.25, tau=1.0 | 0.649 | 0.720 | 0.705 | 5.84 | 4.70 | 3.843 | 466.0 | 38.0 |
| RG (544/544) | anchor | `rg__eps-0.5__lam-0__tau-1` | lam=0.0, eps=0.5, tau=1.0 | 0.667 | 0.760 | 0.722 | 5.91 | 4.75 | 3.814 | 212.5 | 140.0 |
