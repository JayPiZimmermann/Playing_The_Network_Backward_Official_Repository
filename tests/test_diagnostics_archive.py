"""The diagnostic tables recomputed from the archived measurements equal the ones
computed from the experiment outputs."""
import numpy as np
import pandas as pd

from reproduce import collect_diagnostics, diagnostic_figures, pg_granularity
from reproduce.common import MODEL_METHODS, get_spec


def _entries(rng, n=40):
    methods = MODEL_METHODS["vgg16"]
    ours = next(m for m in methods if get_spec(m).ours)
    base = [m for m in methods if not get_spec(m).ours][:2]
    ids = np.asarray([f"img{i:03d}" for i in range(n)])
    out = []
    for method in [ours] + base:
        frac = rng.random(n)
        v = {"PG": (rng.random(n) > 0.4).astype(float), "PG_1px": frac,
             "PG_16px": (rng.random(n) > 0.5).astype(float),
             "max_px": rng.integers(1, 500, n).astype(float)}
        out.append((method, "loc", f"{method}__loc", ids, v))
    return out


def test_pg_granularity_per_image_roundtrip(tmp_path):
    entries = {"vgg16": _entries(np.random.default_rng(0))}
    path = tmp_path / pg_granularity.PER_IMAGE_FILE
    pg_granularity.write_per_image(entries, path)
    back = pg_granularity.read_per_image(path)
    for (m, r, l, ids, v), (m2, r2, l2, ids2, v2) in zip(entries["vgg16"], back["vgg16"]):
        assert (m, r, l) == (m2, r2, l2) and np.array_equal(ids, ids2)
        for k in v:
            assert np.array_equal(v[k], v2[k])          # exact, not approximate
    assert pg_granularity.summarise("vgg16", entries["vgg16"]) == \
        pg_granularity.summarise("vgg16", back["vgg16"])


def _experiment_outputs(root, rng):
    models = [m for m, _ in diagnostic_figures.MODELS]
    ids = [f"img{i:03d}" for i in range(12)]
    h_rows, s_rows = [], []
    for model in models:
        hl, sl = diagnostic_figures.ALIGNED_STEPS[model]
        for game in ("adf", "rg"):
            for step, layer in enumerate((hl, "other")):
                for i in ids:
                    h_rows.append(dict(model=model, game=game, step_idx=step, last_randomized_layer=layer,
                                       sample_id=i, hellinger_distance=rng.random(), hellinger_live=rng.random()))
        for label in ("gradient__fixed", "rg__anchor"):
            for metric in ("spearman_abs", "hog_pearson", "ssim"):
                for step, layer in enumerate(("early", sl, "final")):
                    for i in ids:
                        s_rows.append(dict(model=model, method_label=label, method_name=label, metric=metric,
                                           step_idx=step, num_randomized_layers=step + 1,
                                           last_randomized_layer=layer, sample_id=i, similarity=rng.random()))
    S = pd.DataFrame(s_rows)
    savg = S.groupby(["model", "method_label", "metric", "step_idx"], as_index=False).similarity.mean() \
        .rename(columns={"similarity": "mean"})
    H = pd.DataFrame(h_rows)
    dr = pd.DataFrame([dict(model=m, game=g, h_plain_mean=0.95, h_plain_std=0.01, h_plain_count=5,
                            h_live_mean=1.0, h_live_std=0.0, h_live_count=5)
                       for m in models for g in ("adf", "rg")])
    rounds = pd.DataFrame([dict(model=m, game="rg", sample_id=i, h_plain=0.9) for m in models for i in ids])
    base = pd.DataFrame([dict(model=m, game=g, baseline=k, mean_H=0.5, std_H=0.1, mean_Hlive=0.9, std_Hlive=0.05)
                         for m in models for g in ("adf", "rg")
                         for k in ("clean_intraclass", "clean_interclass")])
    files = {"sanity_check_randomization/results/avg.csv": savg,
             "sanity_check_randomization/results/per_image.csv": S,
             "hellinger_randomization/results/avg.csv": H.groupby(["model", "game", "step_idx"],
                                                                 as_index=False).hellinger_distance.mean(),
             "hellinger_randomization/results/per_image.csv": H,
             "hellinger_randomization/results_double_random/aggregate.csv": dr,
             "hellinger_randomization/results_double_random/per_round.csv": rounds,
             "hellinger_input_noise/results/avg.csv": base,
             "hellinger_input_noise/results/clean_pair_baselines.csv": base,
             "hellinger_input_noise/results/random_baselines.csv": base.assign(baseline="random_vs_random")}
    for rel, df in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(root / rel, index=False)
    return len(S)


def _tables(out, hellinger, double_random, input_noise, sanity):
    diagnostic_figures.main(["--out", str(out), "--hellinger", str(hellinger),
                             "--double-random", str(double_random),
                             "--input-noise", str(input_noise), "--sanity", str(sanity)])
    return {p.name: p.read_text() for p in sorted(out.glob("*.tex"))}


def test_archived_measurements_reproduce_the_tables(tmp_path):
    exp = tmp_path / "experiments"
    n_full = _experiment_outputs(exp, np.random.default_rng(1))
    arch = tmp_path / "diagnostics"
    missing = collect_diagnostics.collect(tmp_path / "results", arch, exp)
    assert missing == [str(tmp_path / "results" / "analysis" / pg_granularity.PER_IMAGE_FILE)]
    kept = pd.read_csv(arch / "sanity_check_randomization" / "per_image.csv.gz")
    assert 0 < len(kept) < n_full and set(kept.metric) == set(collect_diagnostics.SANITY_METRICS)
    raw = _tables(tmp_path / "t_raw", exp / "hellinger_randomization/results",
                  exp / "hellinger_randomization/results_double_random",
                  exp / "hellinger_input_noise/results", exp / "sanity_check_randomization/results")
    shipped = _tables(tmp_path / "t_arch", arch / "hellinger_randomization", arch / "double_random",
                      arch / "hellinger_input_noise", arch / "sanity_check_randomization")
    assert raw == shipped and len(raw) == 5
    assert "??" not in raw["per_image_corr_tabular.tex"]
