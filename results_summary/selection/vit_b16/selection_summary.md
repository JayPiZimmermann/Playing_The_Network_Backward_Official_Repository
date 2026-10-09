# Validation selection: vit_b16 (imagenet-s, val)

Grid: `experiments/grids/imagenet_s_val_vit_b16.csv` - 791 of 791 configurations evaluated.

loc = min rank-sum over {AL, PG, TK}; faith = min rank-sum over {PF5, PF20, Sel} (ranks within each method).

| Method | Criterion | Row label | Config | AL | PG | TK | PF5 | PF20 | Sel | Loc rank-sum | Faith rank-sum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gradient (1/1) | fixed | `gradient__cr-maxabs` | cr=max_abs | 0.431 | 0.520 | 0.446 | 8.50 | 8.06 | 5.896 | 3.0 | 3.0 |
| SmGrad (3/3) | loc | `smoothgrad__spread-0.2` | spread=0.2 | 0.515 | 0.720 | 0.684 | 8.24 | 7.15 | 4.911 | 3.5 | 3.0 |
| SmGrad (3/3) | faith | `smoothgrad__spread-0.2` | spread=0.2 | 0.515 | 0.720 | 0.684 | 8.24 | 7.15 | 4.911 | 3.5 | 3.0 |
| IntGrad (4/4) | loc | `intgrad__steps-100` | steps=100 | 0.461 | 0.540 | 0.473 | 8.46 | 8.04 | 5.651 | 3.5 | 4.0 |
| IntGrad (4/4) | faith | `intgrad__steps-100` | steps=100 | 0.461 | 0.540 | 0.473 | 8.46 | 8.04 | 5.651 | 3.5 | 4.0 |
| LRP-eps (7/7) | loc | `lrp_eps__eps-0.25` | eps=0.25 | 0.581 | 0.820 | 0.728 | 8.05 | 7.10 | 4.623 | 5.5 | 7.0 |
| LRP-eps (7/7) | faith | `lrp_eps__eps-0.1` | eps=0.1 | 0.572 | 0.820 | 0.729 | 7.84 | 6.91 | 4.611 | 6.5 | 3.0 |
| LRP-gamma (15/15) | loc | `lrp_gamma__eps-0.1__gamma-2` | eps=0.1, gamma=2.0 | 0.637 | 0.880 | 0.796 | 8.01 | 6.70 | 4.348 | 6.0 | 25.0 |
| LRP-gamma (15/15) | faith | `lrp_gamma__eps-0.1__gamma-0.1` | eps=0.1, gamma=0.1 | 0.605 | 0.840 | 0.760 | 7.83 | 6.53 | 4.489 | 32.0 | 12.0 |
| AttnLRP (11/11) | loc | `attnlrp__eattn-0.25__gcomp-on__gconv-0.25__glin-0.05` | gcomp=True, eattn=0.25, gconv=0.25, glin=0.05 | 0.639 | 0.920 | 0.787 | 7.86 | 6.74 | 4.472 | 3.0 | 29.0 |
| AttnLRP (11/11) | faith | `attnlrp__eattn-0.01__gcomp-on__gconv-0.25__glin-0.05` | gcomp=True, eattn=0.01, gconv=0.25, glin=0.05 | 0.623 | 0.860 | 0.778 | 7.72 | 6.41 | 4.298 | 14.0 | 7.0 |
| DAVE (54/54) | loc | `dave__N-50__ang-0__noise-0.5__shift-0` | ang=0.0, shift=0.0, N=50, noise=0.5 | 0.483 | 0.700 | 0.565 | 8.20 | 7.56 | 5.183 | 14.0 | 6.0 |
| DAVE (54/54) | faith | `dave__N-50__ang-0__noise-0.5__shift-0` | ang=0.0, shift=0.0, N=50, noise=0.5 | 0.483 | 0.700 | 0.565 | 8.20 | 7.56 | 5.183 | 14.0 | 6.0 |
| TIBAV (1/1) | fixed | `tibav` | default | 0.617 | 0.900 | 0.758 | 8.07 | 7.49 | 4.775 | 3.0 | 3.0 |
| GAE (1/1) | fixed | `gae` | default | 0.525 | 0.760 | 0.610 | 8.08 | 7.52 | 5.018 | 3.0 | 3.0 |
| AR (4/4) | loc | `ar__wr-0.75` | wr=0.75 | 0.454 | 0.560 | 0.512 | 8.39 | 7.98 | 5.094 | 3.5 | 3.0 |
| AR (4/4) | faith | `ar__wr-0.75` | wr=0.75 | 0.454 | 0.560 | 0.512 | 8.39 | 7.98 | 5.094 | 3.5 | 3.0 |
| LIME (1/1) | fixed | `lime__N-1000__fill-segmentmean__kd-binary__kw-0.25__seg-80__sp-slic__sur-ridge` | fill=segment_mean, kd=binary, kw=0.25, N=1000, sp=slic, seg=80, sur=ridge | 0.553 | 0.940 | 0.706 | 8.28 | 7.60 | 5.002 | 3.0 | 3.0 |
| KernelSHAP (1/1) | fixed | `kernelshap__N-auto__fill-imagemean__seg-80__sp-slic` | fill=image_mean, N=auto, sp=slic, seg=80 | 0.576 | 0.960 | 0.804 | 8.15 | 7.43 | 4.924 | 3.0 | 3.0 |
| RG (672/672) | loc | `rg__av-merged__eps-0.1__lam-0__lamsm-m10__tau-0.4__var-5` | var=5.0, lam=0.0, lamsm=-10.0, av=merged, eps=0.1, tau=0.4 | 0.679 | 0.860 | 0.750 | 8.22 | 7.56 | 4.890 | 47.0 | 788.0 |
| RG (672/672) | faith | `rg__av-merged__eps-0.1__lam-0__lamsm-m5__tau-0.7__var-5` | var=5.0, lam=0.0, lamsm=-5.0, av=merged, eps=0.1, tau=0.7 | 0.541 | 0.720 | 0.702 | 8.16 | 6.97 | 4.676 | 1156.0 | 22.0 |
| RG (672/672) | anchor | `rg__av-merged__eps-0.5__lam-0__lamsm-0__tau-1` | lam=0.0, lamsm=0.0, av=merged, eps=0.5, tau=1.0 | 0.505 | 0.720 | 0.668 | 8.26 | 7.17 | 4.706 | 1420.0 | 344.0 |
| RG+Eq (16/16) | loc | `rgeq__av-merged__eps-0.5__lam-0__lamsm-m10__tau-0.5__var-2` | var=2.0, lam=0.0, lamsm=-10.0, av=merged, eps=0.5, tau=0.5 | 0.673 | 0.840 | 0.800 | 8.37 | 7.66 | 4.767 | 6.5 | 29.0 |
| RG+Eq (16/16) | faith | `rgeq__av-merged__eps-0.5__lam-0__lamsm-m10__tau-0.7__var-2` | var=2.0, lam=0.0, lamsm=-10.0, av=merged, eps=0.5, tau=0.7 | 0.636 | 0.820 | 0.787 | 8.37 | 7.59 | 4.656 | 14.0 | 11.0 |
| RG+Eq (16/16) | anchor | `rgeq__av-merged__eps-0.5__lam-0__lamsm-0__tau-1` | lam=0.0, lamsm=0.0, av=merged, eps=0.5, tau=1.0 | 0.515 | 0.740 | 0.712 | 8.40 | 7.66 | 4.723 | 47.0 | 31.0 |
