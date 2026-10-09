"""CPU tests of reproduce.qualitative_figure (no data, no pretrained weights)."""
import numpy as np
import pytest

from reproduce.common import METHOD_SPECS, MODEL_METHODS, row_label
from reproduce.qualitative_figure import (
    COLUMNS, EXAMPLES, PERCENTILE, render_mask, render_signed, selected_row, to_input_resolution,
)


def test_render_signed_normalises_each_sign_by_its_percentile():
    a = np.array([[0.0, 1.0, 2.0, -4.0, -1.0]])
    rgb = render_signed(a)
    assert rgb.dtype == np.uint8 and rgb.shape == (1, 5, 3)
    pos, neg = np.clip(a[0], 0, None), np.clip(-a[0], 0, None)
    expected_r = (np.clip(pos / np.percentile(pos, PERCENTILE), 0, 1) * 255).astype(np.uint8)
    expected_b = (np.clip(neg / np.percentile(neg, PERCENTILE), 0, 1) * 255).astype(np.uint8)
    np.testing.assert_array_equal(rgb[0, :, 0], expected_r)
    np.testing.assert_array_equal(rgb[0, :, 2], expected_b)
    assert rgb[0, 2, 0] == 255 and rgb[0, 3, 2] == 255      # values above the percentile saturate
    assert not rgb[..., 1].any()


def test_render_signed_falls_back_to_the_maximum_for_sparse_maps():
    a = np.zeros((20, 20))
    a[3, 4] = 0.5                                            # 99th percentile of the positive part is 0
    rgb = render_signed(a)
    assert rgb[3, 4].tolist() == [255, 0, 0] and rgb.sum() == 255


def test_render_signed_degenerate_maps():
    assert not render_signed(np.zeros((4, 4))).any()
    rgb = render_signed(np.array([[np.nan, 1.0]]))
    assert rgb[0, 0].tolist() == [0, 0, 0] and rgb[0, 1].tolist() == [255, 0, 0]


def test_percentile_100_matches_the_evaluator_heatmap_export(tmp_path):
    from PIL import Image
    from bwgame.evaluation.evaluator import Evaluator
    a = np.random.RandomState(0).normal(size=(1, 16, 16)).astype(np.float32)
    ev = Evaluator.__new__(Evaluator)          # only output_dir is used by the export
    ev.output_dir = tmp_path
    ev._save_heatmap_images("toy", {"input": a}, None)
    ref = np.asarray(Image.open(tmp_path / "heatmaps" / "toy" / "input" / "img0000.png"), dtype=int)
    # float32 (evaluator) vs float64 (here) can differ by one 8-bit level
    assert np.abs(render_signed(a[0], percentile=100.0).astype(int) - ref).max() <= 1


def test_to_input_resolution_is_nearest_neighbour():
    patches = np.arange(14 * 14, dtype=np.float32).reshape(14, 14)
    up = to_input_resolution(patches, (224, 224))
    assert up.shape == (224, 224)
    np.testing.assert_array_equal(up[::16, ::16], patches)
    np.testing.assert_array_equal(up[15::16, 15::16], patches)
    full = np.ones((224, 224))
    assert to_input_resolution(full, (224, 224)) is full


def test_render_mask_reduces_colour_channels():
    m = np.zeros((2, 2, 3), dtype=bool)
    m[0, 1, 2] = True
    rgb = render_mask(m)
    assert rgb[0, 1].tolist() == [255, 0, 0] and rgb.sum() == 255


def test_selected_row_prefers_loc_then_single_fixed():
    loc = {"epsilon": 0.1, "gamma": 2.0}
    sel = {"methods": {
        "lrp_gamma": {"loc": {"row_label": row_label("lrp_gamma", loc), "params": loc},
                      "fixed": []},
        "tibav": {"fixed": [{"row_label": "tibav", "params": {}}]},
        "gae": {"fixed": []},
    }}
    r = selected_row(sel, "lrp_gamma")
    assert r["role"] == "loc" and r["params"] == loc
    r = selected_row(sel, "tibav")
    assert r["role"] == "fixed" and r["row_label"] == "tibav" and r["params"] == {}
    with pytest.raises(ValueError):
        selected_row(sel, "gae")
    with pytest.raises(KeyError):
        selected_row(sel, "rg")


def test_columns_and_examples():
    stems = [s for s, _ in COLUMNS]
    assert len(set(stems)) == len(stems) == 9
    for _, m in COLUMNS:
        assert m in METHOD_SPECS and m in MODEL_METHODS["vit_b16"]
    assert len(EXAMPLES) == 6 and len(set(EXAMPLES.values())) == 6
