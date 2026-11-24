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
recomputes the intervals and tests of one setting.

## Diagnostic measurements

The randomization, Hellinger and pointing-game granularity tables are not computed from
`test/`. Their recorded measurements are in `diagnostics/`, written by
`python -m reproduce.collect_diagnostics` from the outputs of the `experiments/` runs and
of `reproduce.pg_granularity`. The six outputs `sanity_check_tabular.tex`,
`double_random_tabular.tex`, `per_image_corr_tabular.tex`, `input_baseline_tabular.tex`,
`diagnostic_numbers.tex` and `pg_granularity_tabular.tex` regenerate from that directory.

| File under `diagnostics/` | Content | Output |
|---|---|---|
| `sanity_check_randomization/avg.csv.gz` | mean map similarity per model, method row, metric and cascade step | `sanity_check_tabular.tex`, `\sanity…` macros |
| `sanity_check_randomization/per_image.csv.gz` | per-image map similarity (Spearman of absolute maps, HOG) at full randomization and at the step aligned with the Hellinger cascade | constant-map marks of `sanity_check_tabular.tex`, `per_image_corr_tabular.tex` |
| `hellinger_randomization/avg.csv.gz`, `per_image.csv.gz` | Hellinger distance and its survival-conditioned variant per cascade step, and per image | cascade figure, `per_image_corr_tabular.tex`, `\perImage…` macros |
| `double_random/aggregate.csv.gz`, `per_round.csv.gz` | double-randomization reference, aggregated and per round | `double_random_tabular.tex`, `\dr…` macros, reference lines of the cascade figure |
| `hellinger_input_noise/avg.csv.gz`, `clean_pair_baselines.csv.gz`, `random_baselines.csv.gz` | Hellinger distance per noise level and for the reference pairs | input-noise figure, `input_baseline_tabular.tex` |
| `pg_granularity_per_image.csv.gz` | per model, reported row and test image: pointing-game hit, single-pixel hit fraction, hit on the 16-pixel grid, size of the maximum set | `pg_granularity_tabular.tex` and its paired tests |
| `MANIFEST.json` | source path, size and SHA-256 of every file | |

The tables regenerate on a CPU:

    python -m reproduce.diagnostic_figures --out tables
    python -m reproduce.pg_granularity --results results_summary --out tables

The pointing-game table, its statistics and its paired tests are computed from the
recorded per-image hit values.

For VOC, use `python -m reproduce.bootstrap_ci --results results_summary/voc/test --out-dir tables/ci/voc`.
This resamples whole images while retaining their class records and the pairing
between methods. The reported point estimand remains the image–class record mean.
