# bwgame evaluation framework

Evaluation of attribution methods with [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus)
metrics. The paper's tables are produced by the `reproduce/*` scripts (see the top-level
`README.md`); this package provides the evaluator they use.

Requires Python >= 3.10 (tested with 3.12) and the dependencies in `requirements.txt`.

## Structure

```
evaluation/
├── __init__.py              # Package entry point (Evaluator, split builders, metric configs)
├── run_eval.py              # load_model: the three pretrained backbones
├── evaluator.py             # Evaluator: method-first loop, caching, per-image persistence
├── quantus_bridge.py        # Quantus metric construction, explain function, aggregation
├── metric_spec.py           # Metric names, short labels (AL, PG, ...) and directions
├── dataset.py               # 50-image val split (1 per class, seed 42) / 565-image test split, VOC split
├── config_loader.py         # Metric-config loading
├── stats.py                 # Paired bootstrap, McNemar, Holm
├── metrics/sanity_check.py  # Cascading parameter randomisation (Adebayo et al., 2018)
└── configs/
    └── metrics_main_table/  # Metric set of the paper (default of load_metric_configs)
```

## Metrics used in the paper (`configs/metrics_main_table`)

| Short | Config name | Direction |
|---|---|---|
| AL | `attribution_localisation` | higher is better |
| PG | `pointing_game` | higher is better |
| TK | `top_k_intersection` | higher is better |
| PF5 / PF20 | `pixel_flipping_abs_5pct` / `pixel_flipping_abs_20pct` | lower is better |
| Sel | `selectivity` | lower is better |
| MaxS / AvgS | `max_sensitivity` / `avg_sensitivity` | lower is better |

Directions are defined once in `metric_spec.py`. The evaluator also records the per-image
wall-clock time of every map (CUDA-synchronised, optional untimed warm-up).

## Outputs of the evaluator

For an output directory `OUT` and a config label `LABEL`:

```
OUT/
├── metrics_cache/LABEL.json     # metric means, stds and coverage (fraction of images with a score)
├── per_image/LABEL.npz          # image_ids, one array per metric (per-image value; for pixel flipping
│                                #   and selectivity the per-image AUC), time_s, map_failed, map_is_zero
├── .saliency_cache/LABEL.npz    # cached maps (only if cache_maps=True)
└── heatmaps/LABEL/              # PNG heatmaps (only if save_heatmaps=True)
```

If an attribution map cannot be computed for an image, the map is stored as NaN, the image is
excluded from every metric (reported as NaN in `per_image/`), the failure is counted in
`map_failures` and logged, and the `<metric>_coverage` columns drop below 1.

Method parameters are set as attributes of the registered method (`bwgame/xai/methods`); the
`reproduce/*` scripts reject parameters that are not attributes of the method.
