# Post-hoc localization power-curve protocol

Status: frozen before inspecting the aggregate results.

- Data: the 565 held-out ImageNet-S test images and the 50 validation images.
- Maps: the absolute spatial maps already cached by the sharpening control run. No model inference or test-set selection is performed.
- Main rows: the validation-selected localization and faithfulness rows of LRP-gamma, AttnLRP and RG, plus the RG anchor (seven rows total).
- Diagnostic rows: the complete cached ViT temperature sweeps, reported without selecting a temperature from test results (21 rows total including the main rows).
- Powers: `0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8, 16`, applied as `(abs(a) / max(abs(a)))**p` in float64.
- Endpoints: `p -> 0+` assigns equal mass to every strictly positive-support pixel; `p -> infinity` assigns equal mass to the maximum-valued tie set.
- Primary quantity: Attribution Localization (AL), the fraction of transformed attribution mass inside the object mask.
- Context quantities: mask area fraction; AL divided by mask area fraction; and the effective-support fraction `(sum a)^2 / (N * sum a^2)`. These describe concentration and do not measure classifier faithfulness.
- Inference: image-level means and 95% percentile-bootstrap intervals from 10,000 resamples with seed 42. Pixels are never treated as independent samples.
- Full-image masks remain in AL and concentration summaries (AL is necessarily one for those images); their absence of outside pixels matters only to separate inside-versus-outside rank statistics.
- Main table powers: `0.5, 1, 1.5, 2, 4, 16, infinity`. All frozen powers are retained in machine-readable results and curves.
- Interpretation: a strictly increasing power changes cardinal mass concentration but preserves mathematical pixel ordering. AL need not be monotone in power. The `p -> infinity` value is the fraction of tied maxima inside the mask, which differs from the paper's any-maximum-inside Pointing Game convention when maxima are tied.

The analysis is post-hoc and supplemental. It is not used to select a method, power or temperature, and it does not establish explanation faithfulness.
