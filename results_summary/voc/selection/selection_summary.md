# Validation selection: vit_b16 (voc, val)

Grid: `experiments/grids/imagenet_s_val_vit_b16.csv` - 103 of 103 configurations evaluated.

loc = min rank-sum over {AL, PG, TK}; faith = min rank-sum over {PF5, PF20, Sel} (ranks within each method).

| Method | Criterion | Row label | Config | AL | PG | TK | PF5 | PF20 | Sel | Loc rank-sum | Faith rank-sum |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gradient (1/1) | fixed | `gradient__cr-maxabs` | cr=max_abs | 0.295 | 0.344 | 0.349 | 3.45 | 3.20 | 2.353 | 3.0 | 3.0 |
| SmGrad (3/3) | loc | `smoothgrad__spread-0.1` | spread=0.1 | 0.347 | 0.586 | 0.515 | 3.22 | 2.71 | 1.931 | 5.0 | 7.0 |
| SmGrad (3/3) | faith | `smoothgrad__spread-0.15` | spread=0.15 | 0.349 | 0.559 | 0.511 | 3.23 | 2.65 | 1.920 | 6.0 | 5.0 |
| IntGrad (4/4) | loc | `intgrad__steps-300` | steps=300 | 0.290 | 0.379 | 0.350 | 3.43 | 3.21 | 2.333 | 5.0 | 6.0 |
| IntGrad (4/4) | faith | `intgrad__steps-100` | steps=100 | 0.290 | 0.381 | 0.349 | 3.43 | 3.21 | 2.333 | 6.0 | 5.0 |
| LRP-eps (7/7) | loc | `lrp_eps__eps-0.1` | eps=0.1 | 0.383 | 0.631 | 0.527 | 3.04 | 2.62 | 1.940 | 7.0 | 4.0 |
| LRP-eps (7/7) | faith | `lrp_eps__eps-0.1` | eps=0.1 | 0.383 | 0.631 | 0.527 | 3.04 | 2.62 | 1.940 | 7.0 | 4.0 |
| LRP-gamma (15/15) | loc | `lrp_gamma__eps-0.1__gamma-1` | eps=0.1, gamma=1.0 | 0.408 | 0.594 | 0.544 | 2.94 | 2.31 | 1.694 | 13.0 | 6.0 |
| LRP-gamma (15/15) | faith | `lrp_gamma__eps-0.1__gamma-1` | eps=0.1, gamma=1.0 | 0.408 | 0.594 | 0.544 | 2.94 | 2.31 | 1.694 | 13.0 | 6.0 |
| AttnLRP (11/11) | loc | `attnlrp__eattn-0.05__gcomp-on__gconv-0.25__glin-0.05` | gcomp=True, eattn=0.05, gconv=0.25, glin=0.05 | 0.407 | 0.595 | 0.541 | 2.83 | 2.24 | 1.715 | 5.5 | 5.0 |
| AttnLRP (11/11) | faith | `attnlrp__eattn-0.05__gcomp-on__gconv-0.25__glin-0.05` | gcomp=True, eattn=0.05, gconv=0.25, glin=0.05 | 0.407 | 0.595 | 0.541 | 2.83 | 2.24 | 1.715 | 5.5 | 5.0 |
| DAVE (54/54) | loc | `dave__N-50__ang-0__noise-0.25__shift-0` | ang=0.0, shift=0.0, N=50, noise=0.25 | 0.338 | 0.567 | 0.468 | 3.37 | 3.03 | 2.168 | 4.5 | 22.0 |
| DAVE (54/54) | faith | `dave__N-50__ang-0__noise-0.5__shift-0` | ang=0.0, shift=0.0, N=50, noise=0.5 | 0.337 | 0.549 | 0.464 | 3.35 | 2.98 | 2.150 | 15.5 | 8.0 |
| TIBAV (1/1) | fixed | `tibav` | default | 0.420 | 0.737 | 0.572 | 3.05 | 2.69 | 1.733 | 3.0 | 3.0 |
| GAE (1/1) | fixed | `gae` | default | 0.358 | 0.732 | 0.545 | 3.01 | 2.65 | 1.753 | 3.0 | 3.0 |
| AR (4/4) | loc | `ar__wr-0.75` | wr=0.75 | 0.276 | 0.486 | 0.428 | 3.61 | 3.45 | 2.150 | 3.0 | 3.0 |
| AR (4/4) | faith | `ar__wr-0.75` | wr=0.75 | 0.276 | 0.486 | 0.428 | 3.61 | 3.45 | 2.150 | 3.0 | 3.0 |
| LIME (1/1) | fixed | `lime__N-1000__fill-segmentmean__kd-binary__kw-0.25__seg-80__sp-slic__sur-ridge` | fill=segment_mean, kd=binary, kw=0.25, N=1000, sp=slic, seg=80, sur=ridge | 0.351 | 0.797 | 0.526 | 3.40 | 3.17 | 2.135 | 3.0 | 3.0 |
| KernelSHAP (1/1) | fixed | `kernelshap__N-auto__fill-imagemean__seg-80__sp-slic` | fill=image_mean, N=auto, sp=slic, seg=80 | 0.341 | 0.779 | 0.552 | 3.39 | 3.04 | 2.108 | 3.0 | 3.0 |
