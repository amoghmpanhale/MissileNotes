"""End-to-end scan tests.

The fake segmenter is the key: it returns the synthetic ground-truth mask for
the exact frame it receives, so any mistake in the pipeline's rotate-into-
model-frame / rotate-back bookkeeping shows up as a misaligned mask and a bad
quad.
"""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from missilenotes.enhance import binarize
from missilenotes.geometry import order_corners
from missilenotes.pipeline import ScanResult, scan
from tests.synth import PAGE_H, PAGE_W, _placement, synthetic_page

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


def _page_mask_to_flat(mask, synth, result, seed):
    """Map a page-space synth mask into the pipeline's flat scan frame."""
    assert synth.rotation == 0, "helper assumes an already-upright page"
    assert result.scan is not None and result.quad is not None
    homography, _, _ = _placement(
        PAGE_W, PAGE_H, synth.rotation, synth.tilt, seed, None
    )
    in_photo = cv2.warpPerspective(
        mask.astype(np.uint8),
        homography,
        (synth.image.shape[1], synth.image.shape[0]),
        flags=cv2.INTER_NEAREST,
    )
    height, width = result.scan.shape
    src = order_corners(result.quad).astype(np.float32)
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        np.float32,
    )
    to_flat = cv2.getPerspectiveTransform(src, dst)
    flat = cv2.warpPerspective(
        in_photo, to_flat, (width, height), flags=cv2.INTER_NEAREST
    )
    return flat > 0


def _erode(mask, size=3):
    kernel = np.ones((size, size), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel) > 0


def test_normalize_illumination_recovers_hard_cast_shadow():
    seed = 7
    synth = synthetic_page(seed=seed, shadow=0.55, annotate=True)
    result = scan(synth.image, GroundTruthSegmenter(synth), normalize=True)
    raw = scan(synth.image, GroundTruthSegmenter(synth), normalize=False)
    assert result.ok and raw.ok

    ink = _page_mask_to_flat(synth.ink_mask, synth, result, seed)
    rules = _page_mask_to_flat(synth.rule_mask, synth, result, seed)
    marker = _page_mask_to_flat(synth.marker_mask, synth, result, seed)
    ink &= ~marker
    rules &= ~marker
    paper = ~(ink | rules | marker)
    # Erode the paper mask so antialiased edges of rules, their reprojection
    # neighbourhood and the page border do not count as "plain paper".
    paper_core = _erode(paper)
    assert ink.sum() > 1000 and paper_core.sum() > 100_000

    ink_fraction = (result.scan[ink] == 0).mean()
    paper_fraction = (result.scan[paper_core] == 255).mean()
    raw_ink = (raw.scan[ink] == 0).mean()
    raw_paper = (raw.scan[paper_core] == 255).mean()

    assert ink_fraction >= 0.95, ink_fraction
    assert paper_fraction >= 0.98, paper_fraction
    # The adaptive threshold already handles smooth shading; the point of
    # normalization is the hard shadow edge. Count decisions rather than
    # averaging two fractions with very different pixel counts.
    normalized_correct = int((result.scan[ink] == 0).sum()) + int(
        (result.scan[paper_core] == 255).sum()
    )
    raw_correct = int((raw.scan[ink] == 0).sum()) + int(
        (raw.scan[paper_core] == 255).sum()
    )
    assert normalized_correct > raw_correct, (raw_paper, paper_fraction)
    assert paper_fraction > raw_paper
    # Ink is already found well by the adaptive threshold; normalization must
    # not lose it.
    assert ink_fraction >= raw_ink - 0.01, (raw_ink, ink_fraction)


def test_normalize_illumination_does_not_speckle_the_marker():
    seed = 7
    shadowed = synthetic_page(seed=seed, shadow=0.55, annotate=True)
    plain = synthetic_page(seed=seed, annotate=True)
    result = scan(
        shadowed.image, GroundTruthSegmenter(shadowed), normalize=True
    )
    reference = scan(plain.image, GroundTruthSegmenter(plain))
    assert result.ok and reference.ok

    marker = _page_mask_to_flat(
        shadowed.marker_mask, shadowed, result, seed
    )
    assert marker.sum() > 1000
    agreement = (result.scan[marker] == reference.scan[marker]).mean()
    assert agreement >= 0.95, agreement


def test_remove_rules_keeps_handwriting_even_on_a_rule():
    seed = 7
    synth = synthetic_page(seed=seed, annotate=True)
    before = scan(synth.image, GroundTruthSegmenter(synth))
    result = scan(
        synth.image, GroundTruthSegmenter(synth), remove_rules=True
    )
    assert result.ok and before.ok

    ink = _page_mask_to_flat(synth.ink_mask, synth, result, seed)
    rules = _page_mask_to_flat(synth.rule_mask, synth, result, seed)
    marker = _page_mask_to_flat(synth.marker_mask, synth, result, seed)
    ink &= ~marker
    rules &= ~marker
    rule_only = rules & ~ink
    ink_on_rule = ink & rules

    assert rule_only.sum() > 1000
    assert ink_on_rule.sum() > 50, "seed must put handwriting on a rule"
    assert (before.scan[rule_only] == 0).mean() > 0.8, "rules start out ink"

    assert (result.scan[rule_only] == 255).mean() >= 0.98
    assert (result.scan[ink] == 0).mean() >= 0.97
    assert (result.scan[ink_on_rule] == 0).mean() >= 0.97

    # Every surviving dash fragment would be a connected component lying
    # entirely inside a (slightly widened) ground-truth rule band. Handwriting
    # crossing a rule extends beyond the band and is not counted.
    band = cv2.dilate(
        rules.astype(np.uint8), np.ones((5, 5), np.uint8)
    ) > 0
    ink_mask = result.scan == 0
    count, labels = cv2.connectedComponents(ink_mask.astype(np.uint8), 8)
    fragments = 0
    for i in range(1, count):
        component = labels == i
        if not np.any(component & ~band):
            fragments += 1
    assert fragments <= 5, f"{fragments} leftover components inside rule bands"


@pytest.mark.parametrize("rotation,seed", [(0, 3), (90, 4), (180, 5), (270, 6)])
def test_default_output_unchanged_and_new_stages_are_opt_in(rotation, seed):
    synth = synthetic_page(rotation=rotation, seed=seed)
    segmenter = GroundTruthSegmenter(synth)

    default = scan(synth.image, segmenter)
    explicit = scan(
        synth.image, segmenter, normalize=False, remove_rules=False
    )

    assert np.array_equal(default.scan, explicit.scan)
    assert "normalize" not in default.timings
    assert "remove_rules" not in default.timings

    legacy = cv2.cvtColor(default.flat_rgb, cv2.COLOR_RGB2GRAY)
    assert np.array_equal(default.scan, binarize(legacy))


def test_optional_stages_are_timed():
    synth = synthetic_page(seed=2)
    result = scan(
        synth.image,
        GroundTruthSegmenter(synth),
        normalize=True,
        remove_rules=True,
    )

    assert result.ok
    assert result.timings["normalize"] >= 0.0
    assert result.timings["remove_rules"] >= 0.0


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
