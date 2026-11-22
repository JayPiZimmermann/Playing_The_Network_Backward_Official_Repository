# Result summaries

Compact copies of the evaluation outputs the paper's tables are built from.

- `test/<model>/<row>.json` (ImageNet-S test split, 565 images; `vit_b16`, `vgg16`,
  `resnet50`) and `voc/test/<row>.json` (Pascal VOC 2012, 2,743 image–class records from 2,293 images): one file per
  evaluated row (configuration) with the metric means, the configuration, the selection
  roles and the per-image runtime statistics.
- `test/<model>/per_image_scores.csv.gz` and `voc/test/per_image_scores.csv.gz`: one line
  per method row and evaluation record with the scalar value of every metric (blank = not finite;
  MaxS / AvgS are computed on the fixed 200-image subset and blank elsewhere). The
  confidence intervals and the paired tests are computed from these files.
- `selection/<model>/` and `voc/selection/`: the validation selection
  (`selected.json`: the `loc`, `faith`, fixed and anchor rows of every method;
  `val_scores.csv`: validation means of every grid row; `selection_summary.md`).
- `runtime/<model>.csv`: the per-image runtime benchmark of the selected rows (t/img).
- `controls/`: aggregated outputs of the additional controls (`reproduce/control_*.py`);
  see `controls/README.md`.
- `low_temperature/vit_b16/`: the routing temperature 0.2 point of the ViT-B/16 temperature
  sweep, in the layout of `test/vit_b16/`. It extends the one-factor sweep of the
  localization-selected Routing Game configuration below its grid
  (`experiments/grids/vit_rg_low_temperature.json`).

Thirteen summary-backed outputs regenerate from this directory on a CPU in a few
minutes with the default 10,000 bootstrap resamples (`--n-boot 1000` for a quick
check with different Monte Carlo precision):

    python -m reproduce.make_tables --results results_summary --out tables --strict

`python -m reproduce.bootstrap_ci --results results_summary/test/<model> --out-dir tables/ci/<model>`
recomputes the intervals and tests of one setting. The randomization, Hellinger and pointing-game
granularity tables and the figures come from the `experiments/` runs (see the top-level
README) and are not covered by this directory.

For VOC, use `python -m reproduce.bootstrap_ci --results results_summary/voc/test --out-dir tables/ci/voc`.
This resamples whole images while retaining their class records and the pairing
between methods. The reported point estimand remains the image–class record mean.
