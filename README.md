# Playing the Network Backward: code

Reference implementation and reproduction scripts for the paper *Playing the Network
Backward: A Game-Theoretic Attribution Framework* https://arxiv.org/abs/2605.06212. The Python
package is `bwgame`; the Routing Game (RG), its equivariant variant (RG+Eq) and the
Stopping Game (SG) are registered attribution methods in `bwgame/xai/methods`.

## Setup

Requires Python >= 3.10 (tested with Python 3.12) and a CUDA GPU for the full runs.

```bash
python -m venv .venv && source .venv/bin/activate
# torch: pick the wheel index that matches your CUDA driver, e.g. cu124
pip install "torch>=2.0,<3" "torchvision>=0.15,<1" --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e ".[dev]"        # package + test dependencies (pytest, zennit, lxt)
```

`requirements.txt` uses compatible-release pins with upper bounds where a new major
version changes the API or the metric definitions (notably `quantus~=0.6.0`). The exact
environment of the reported run is recorded in **`requirements-lock.txt`**, produced with
`pip freeze > requirements-lock.txt` in the evaluation environment; install from it
(`pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126`)
to match the recorded package versions. Hardware, accelerator kernels and
nondeterministic operations can still affect fresh evaluations; report numerical
tolerances when comparing independently recomputed scores.

Pretrained backbones are downloaded by torchvision with pinned weight enums:
`VGG16_Weights.IMAGENET1K_V1`, `ResNet50_Weights.IMAGENET1K_V2`, `ViT_B_16_Weights.IMAGENET1K_V1`.

## Data

### ImageNet-S50 + ILSVRC2012 validation images

```bash
bash data/download_imagenet_s.sh                          # ImageNet-S50 masks -> data/imagenet-s/
python data/download_imagenet_val.py /path/to/ILSVRC2012_img_val.tar   # only the needed images -> data/imagenet/val/
# or: python data/download_imagenet_val.py --link /path/to/existing/imagenet/val
```

The ILSVRC2012 validation images require a (free) login at <https://image-net.org>.
`data.imagenet_s.ImageNetS` keeps the 616 masks whose ImageNet-S class equals the ILSVRC2012 label;
`build_val_split` (seed 42) holds out 50 images (one per class) for configuration
selection and `build_test_split` returns the remaining 566 images, of which the 565 whose
object is visible in the evaluated crop form the **565-image test split**. The
scripts abort if the split sizes differ (`--allow-size-mismatch` to override).
Other locations can be given with `--imagenet-root` / `--imagenet-s-root` or the
environment variables `IMAGENET_ROOT` / `IMAGENET_S_ROOT`.

### Pascal VOC 2012

`python -m reproduce.pascal_voc` runs `data/download_pascal_voc.sh` if
`data/voc/VOCdevkit/VOC2012` is missing (or set `--voc-root` / `VOC_ROOT`). The
trainval pool is split per image into val:test = 1:5 (seed 42).

## Reproduction

All commands run from this directory. Every evaluation script can use a single GPU or
several GPUs via `--shard i --num-shards n` (one process per shard, disjoint output
files) and is resumable with `--resume`. `M` is one of `vit_b16`, `vgg16`, `resnet50`.

| Paper artifact | Command(s) | Output |
|---|---|---|
| Declared validation grids (grid table of the protocol appendix) | `python -m reproduce.build_grids --model M` (the CSVs are shipped in `experiments/grids/`) | `experiments/grids/imagenet_s_val_M.csv` |
| Validation selection tables | `python -m reproduce.select_configs --model M --grid experiments/grids/imagenet_s_val_M.csv --split val --out results/selection/M [--shard i --num-shards n] --resume`, then the same command with `--aggregate` | `results/selection/M/{val_scores.csv, selected.json, selection_summary.md}`; table `val_selection_M_tabular.tex` |
| Main ViT table, test tables with CIs | `python -m reproduce.evaluate_selected --model M --selection results/selection/M/selected.json --split test --out results/test/M --robustness-subset 200 --resume`; `python -m reproduce.bootstrap_ci --results results/test/M --n-boot 10000 --seed 0`; `python -m reproduce.runtime_benchmark --model M --selection results/selection/M/selected.json --n-images 50 --warmup 5 --out results/runtime/M.csv` | `main_vit_tabular.tex`, `test_M_ci_longtable.tex` |
| One-factor ablations, attention-rule table (appendix) | `python -m reproduce.ablation_rows --model M --selection results/selection/M/selected.json --out results/ablations_M.json` (ViT-B/16: plus the tau / lambda_sm sweep rows of `python -m reproduce.sweep_rows --model vit_b16 --selection results/selection/vit_b16/selected.json --out results/sweeps_vit.json`); `python -m reproduce.evaluate_selected --model M --selection results/selection/M/selected.json --extra-rows results/ablations_M.json --split test --out results/test/M --skip-robustness --resume` | `ablations_M_longtable.tex`, `attention_rule_vit_b16_tabular.tex` |
| VOC transfer | `python -m reproduce.pascal_voc --stage tune [--shard i --num-shards n] --resume`; `--stage select`; `--stage test --selection results/selection/vit_b16/selected.json --resume` | `results/voc/`; `test_voc_vit_b16_ci_longtable.tex` |
| Pointing game at matched granularity (appendix) | `python -m reproduce.pg_granularity --results results --out results/tables` (CPU; uses the maps cached by `evaluate_selected --cache-maps`) | `pg_granularity_tabular.tex`, `results/analysis/pg_granularity_*.csv`, per-image measurements `results/analysis/pg_granularity_per_image.csv.gz` |
| Archive the diagnostic measurements | `python -m reproduce.collect_diagnostics` (CPU; after the randomisation runs and `pg_granularity`) | `results_summary/diagnostics/` |
| Tables from the shipped results (no GPU) | `python -m reproduce.make_tables --results results_summary --out tables --strict` (CPU, a few minutes) | all `.tex` tables above, `numbers.tex` |
| Tables and numbers in the text | `python -m reproduce.make_tables --results results --out results/tables` (after the steps above for all models and VOC) | the `.tex` files above, `numbers.tex` |
| Randomisation figures / tables | `python -m reproduce.hellinger_sanity --block all` (blocks: `sanity`, `hellinger`, `double-random`, `input-noise`), then `python -m reproduce.diagnostic_figures --paper <paper dir>` | under `results/` (`results_double_random/` for the double-random block) of each experiment; `sanity_check_tabular.tex`, `double_random_tabular.tex`, `per_image_corr_tabular.tex`, `input_baseline_tabular.tex`, `diagnostic_numbers.tex` and the data of the two randomisation figures |
| Qualitative figure (appendix) | `python -m reproduce.qualitative_figure --selection results/selection/vit_b16/selected.json --out <dir>` | one PNG per example and method |

`python -m reproduce.main_tables_imagenet_s --models vit_b16,vgg16,resnet50` runs the grid,
selection, test-evaluation, bootstrap, runtime and table steps above sequentially on one GPU.
`make_tables` works with partial results and writes red `??` placeholders for missing
numbers. Table cells are test-split means; **bold** marks the best row of a column and
`≈` marks a comparison with the observed best whose adjusted p-value is at least
0.05; this does not establish equivalence. Holm correction covers every unordered
pair of eligible rows in the displayed metric column before winner comparisons
are reported. ImageNet-S uses paired image bootstrap tests and McNemar's test for
binary PG. VOC uses paired image-cluster bootstrap tests for every metric, including
PG: each sampled image retains all of its class records, and means remain weighted
by image–class records. The default is 10,000 resamples with seed 0 and 95%
percentile intervals. VOC p-values use null-centered bootstrap errors with the
plus-one Monte Carlo correction. `bootstrap_ci` records the sampling design in
`bootstrap_metadata.json`; `make_tables` recomputes VOC intervals if cached CIs lack
matching image-cluster metadata.

Protocol in short: every grid row is evaluated on the 50-image validation split with
AL, PG, TK, PF5, PF20 and Sel; per method, `loc` minimises the rank-sum over
{AL, PG, TK} and `faith` the rank-sum over {PF5, PF20, Sel} (ties: anchor/default
first, then fewer non-default parameters). The selected rows, the RG / RG+Eq
alpha-beta-LRP anchor and the fixed LIME / KernelSHAP rows are evaluated on the test
split with all eight metrics. Per-image scores are stored in
`results/test/M/per_image/<row>.npz`. The `runtime_s` block of a result file is the
wall-clock time of the evaluation run (GPU types vary between rows); the t/img column of
the paper comes from `reproduce.runtime_benchmark` on one GPU type.

The paper's own runs ship in compact form in `results_summary/` (row JSONs, per-image
scores, validation selections and runtime benchmarks; see `results_summary/README.md`):
`python -m reproduce.make_tables --results results_summary --out tables --strict`
regenerates 13 outputs without a GPU: the main table, four test CI tables, three
validation-selection tables, three ablation tables, the attention-rule table and
`numbers.tex`.

Six diagnostic outputs are computed from the recorded measurements of the
`experiments/` runs rather than from `results_summary/test`: `sanity_check_tabular.tex`,
`double_random_tabular.tex`, `per_image_corr_tabular.tex`, `input_baseline_tabular.tex`,
`diagnostic_numbers.tex` and `pg_granularity_tabular.tex`. They regenerate without a GPU,
the dataset or cached attribution maps from `results_summary/diagnostics/`:

```bash
python -m reproduce.diagnostic_figures --out tables                        # randomisation and Hellinger tables, macros
python -m reproduce.pg_granularity --results results_summary --out tables  # pointing game at matched granularity
```

`results_summary/diagnostics/` holds the gzip-compressed CSV measurements (per image, per
round and per noise level) and is written by `python -m reproduce.collect_diagnostics`
after the experiment runs; `results_summary/README.md` lists the files and the table each
one feeds. The measurements of the paper's runs ship in this directory, so the two
commands above regenerate all six outputs from it. The
pointing-game table is then recomputed from recorded per-image hit values, which checks
the reported statistics and paired tests but not the map processing; the latter needs the
maps cached by `evaluate_selected --cache-maps`.

### Compute estimate

Rough estimates from the per-image times of the methods (single modern GPU):
validation grids ~9 GPU-hours (ViT-B/16, 791 rows), ~6 (VGG-16, 635 rows) and ~5
(ResNet-50, 595 rows). On the test split the robustness metrics dominate: MaxS and
AvgS re-explain every image 2 x 50 times, i.e. about 16 GPU-hours per row and per
second of attribution time per image (e.g. ~55 h for a LIME row, ~3 h for an RG row);
the full test evaluation of all three models is several hundred GPU-hours.
`--skip-robustness`, `--robustness-nr-samples N` and `--robustness-subset N` (fixed
random subset of test images, identical for all rows) reduce this. The VOC transfer
(tuning on 556 and testing on 2,743 samples, no MaxS/AvgS) is ~150 GPU-hours.

## Additional controls

The scripts `reproduce/control_*.py` implement the additional controls reported with the
thesis: fixed power sharpening, full deletion and insertion curves, risk and gate
robustness contrasts, the Stopping Game pooling and signed-gradient audits, the numerical
alpha-beta-LRP recovery checks, and the checkpoint-aligned randomization and local
Hellinger controls. Their aggregated outputs ship in `results_summary/controls/` (see its
README for the mapping from files to controls). The recovery checks use three ImageNet-S
images each and the pretrained torchvision weights:

```bash
python -m reproduce.control_rg_recovery            # VGG-16 against Zennit alpha-beta LRP
python -m reproduce.control_rg_vit_end_to_end      # ViT-B/16, end to end
python -m reproduce.control_rg_attention_recovery  # ViT-B/16, every attention block
```

They write to `results/controls/` by default and read the datasets from the locations of the
Data section.

`experiments/plot_rg_temperature_ablation.py` renders the qualitative routing-temperature
sweep and `experiments/grids/vit_rg_low_temperature.json` lists the additional
low-temperature rows; `reproduce/temperature_precision_probe.py` compares precisions at low
temperatures. These three files assume that this directory is `code/playing_backward`
inside the thesis repository.

## Testing

```bash
pytest                   # all tests (pytest.ini: testpaths = tests)
pytest -m "not slow"     # skip tests that download pretrained weights
```

## Code layout

```
bwgame/
  dc_context.py, patcher.py, operations/   DC decomposition of the network (forward/backward kernels)
  game_theory/                             Fan-in entropy bias, per-layer activation / ADF cache
  xai/methods/                             Attribution methods (RG = routing_game, SG = adf_game, baselines)
  xai/wrappers.py                          Equivariant averaging wrapper (RG+Eq)
  evaluation/                              Evaluator, Quantus bridge, splits, statistics (see its README)
reproduce/
  build_grids.py, select_configs.py, evaluate_selected.py, ablation_rows.py, sweep_rows.py,
  bootstrap_ci.py, runtime_benchmark.py, make_tables.py, pg_granularity.py, pascal_voc.py,
  main_tables_imagenet_s.py, hellinger_sanity.py, diagnostic_figures.py, collect_diagnostics.py,
  qualitative_figure.py, common.py
  control_*.py                            Additional controls (see above)
  temperature_precision_probe.py           Low-temperature precision comparison
experiments/
  grids/                                   Declared validation grids (CSV), low-temperature rows (JSON)
  plot_rg_temperature_ablation.py          Qualitative routing-temperature sweep
  sanity_check_randomization/, hellinger_*/  Randomisation and Hellinger diagnostics
results_summary/                           Row results, per-image scores, selections and runtime benchmarks of the paper's runs (see its README)
  controls/                                Aggregated outputs of the additional controls
tools/export_results_summary.py            Export of results_summary/ from a full results/ directory
data/
  download_imagenet_s.sh, download_imagenet_val.py, download_pascal_voc.sh
  imagenet_s.py, pascal_voc.py, imagenet_class_index.json
tests/                                     Unit and integration tests
```

## License

Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0); see [LICENSE](LICENSE).
The file `bwgame/xai/methods/_chefer_vit_lrp.py` is adapted from third-party code and stays under the
MIT License reproduced in its header.
