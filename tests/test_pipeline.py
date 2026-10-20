"""End-to-end scan tests.

The fake segmenter is the key: it returns the synthetic ground-truth mask for
the exact frame it receives, so any mistake in the pipeline's rotate-into-
model-frame / rotate-back bookkeeping shows up as a misaligned mask and a bad
quad.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from missilenotes.geometry import order_corners
from missilenotes.pipeline import ScanResult, scan
from tests.synth import synthetic_page

ROOT = Path(__file__).resolve().parents[1]
TFLITE_WEIGHTS = (
    ROOT / "experiments/model_horizontal/fairscan-segmentation-model.tflite"
)
HAS_LITERT = importlib.util.find_spec("ai_edge_litert") is not None
ROTATIONS = [0, 90, 180, 270]


class GroundTruthSegmenter:
    """Return the synth mask rotated into the model frame it is handed.

    ``synthetic_page(rotation=r)`` puts the page ``r`` degrees CCW in the
    photo. The pipeline rotates by ``360 - r`` to upright, then 90 CW into the
    model frame; the mask needs that same total rotation.
    """

    def __init__(self, synth):
        upright_ccw = (360 - synth.rotation) % 360
        model_ccw = upright_ccw - 90
        self.mask = np.rot90(synth.mask, (model_ccw // 90) % 4)

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        assert img_rgb.shape[:2] == self.mask.shape
        return self.mask.astype(np.float32)


class ExplodingSegmenter:
    def predict(self, img_rgb):
        raise AssertionError("segmenter must not run without a marker")


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_scan_recovers_page_in_original_coords(rotation):
    synth = synthetic_page(rotation=rotation, seed=rotation + 3)
    result = scan(synth.image, GroundTruthSegmenter(synth))

    assert result.ok, result.reason
    assert result.rotation == (360 - rotation) % 360
    assert result.scan is not None
    assert result.scan.dtype == np.uint8
    assert set(np.unique(result.scan)) <= {0, 255}
    assert result.flat_rgb is not None
    assert result.flat_rgb.shape[2] == 3
    assert result.quad is not None

    actual = order_corners(result.quad)
    expected = order_corners(synth.quad)
    diagonal = float(np.linalg.norm(expected[0] - expected[2]))
    error = float(np.linalg.norm(actual - expected, axis=1).max())
    assert error <= 0.02 * diagonal, f"{error:.1f}px > 2% of {diagonal:.1f}px"

    expected_aspect = synth.page_size[0] / synth.page_size[1]
    actual_aspect = result.flat_rgb.shape[1] / result.flat_rgb.shape[0]
    assert abs(actual_aspect / expected_aspect - 1.0) <= 0.03


def test_no_marker_fails_without_running_model():
    synth = synthetic_page(with_marker=False)

    result = scan(synth.image, ExplodingSegmenter())

    assert result.ok is False
    assert "marker" in result.reason.lower()
    assert result.scan is None
    assert result.flat_rgb is None
    assert result.quad is None
    assert "predict" not in result.timings


@pytest.mark.skipif(
    not HAS_LITERT or not TFLITE_WEIGHTS.is_file(),
    reason="ai-edge-litert or the horizontal .tflite weights are unavailable",
)
def test_real_tflite_smoke():
    from missilenotes.segment import Segmenter

    synth = synthetic_page(rotation=0, seed=1)
    result = scan(synth.image, Segmenter(TFLITE_WEIGHTS))

    assert isinstance(result, ScanResult)
    assert result.timings["predict"] > 0.0
