"""Tests for the Tk hand-correction editor's non-GUI helpers.

The module is skipped when tkinter is unavailable; no window is ever opened.
"""

import importlib.util

import numpy as np
import pytest
from PIL import Image

from tests.synth import PAGE_H, PAGE_W, synthetic_page

if importlib.util.find_spec("tkinter") is None:
    pytest.skip("tkinter is unavailable", allow_module_level=True)

from missilenotes.geometry import order_corners  # noqa: E402
from missilenotes.marker import rotate_points  # noqa: E402
from tools import warp_editor  # noqa: E402


def test_quad_for_editing_maps_into_upright_frame():
    synth = synthetic_page(rotation=90, seed=2)
    # The pipeline's rotation is the CCW turn that makes the page upright.
    rotation = (360 - synth.rotation) % 360
    upright_shape = (synth.image.shape[1], synth.image.shape[0], 3)

    quad = warp_editor.quad_for_editing(
        synth.quad, rotation, synth.image.shape, upright_shape
    )

    expected = order_corners(
        rotate_points(synth.quad, rotation, synth.image.shape)
    )
    assert quad.shape == (4, 2)
    assert np.allclose(quad, expected)


def test_failed_scan_starts_from_whole_upright_image():
    upright_shape = (300, 200, 3)  # H, W
    quad = warp_editor.quad_for_editing(None, 0, upright_shape, upright_shape)

    assert np.allclose(
        order_corners(quad),
        [[0, 0], [199, 0], [199, 299], [0, 299]],
    )


def test_flatten_page_warps_and_binarizes():
    synth = synthetic_page(rotation=0, tilt=0.0, seed=4)
    flat = warp_editor.flatten_page(synth.image, synth.quad, binary=True)

    assert abs(flat.shape[1] / flat.shape[0] - PAGE_W / PAGE_H) < 0.05
    assert flat.dtype == np.uint8
    assert set(np.unique(flat)) <= {0, 255}
    assert (flat == 255).mean() > 0.5  # paper is the majority


def test_flatten_page_can_return_grayscale():
    synth = synthetic_page(rotation=0, tilt=0.0, seed=5)
    flat = warp_editor.flatten_page(synth.image, synth.quad, binary=False)

    assert flat.ndim == 2
    assert len(np.unique(flat)) > 2


def test_load_rgb_reads_png(tmp_path):
    synth = synthetic_page(rotation=0, seed=6)
    path = tmp_path / "page.png"
    Image.fromarray(synth.image).save(path)

    loaded = warp_editor.load_rgb(path)

    assert loaded.dtype == np.uint8
    assert loaded.shape == synth.image.shape
    assert np.array_equal(loaded, synth.image)
