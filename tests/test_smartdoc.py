"""Tests for the SmartDoc benchmark's scoring and marker compositing."""

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from missilenotes.marker import detect_marker, rotate_points, upright_rotation

_SPEC = importlib.util.spec_from_file_location(
    "smartdoc", Path(__file__).resolve().parents[1] / "scripts/smartdoc.py"
)
smartdoc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smartdoc)

W, H = 2100.0, 2970.0
# An upright A4 page seen with mild perspective in a 1920x1080 frame.
QUAD = np.array([[760, 200], [1180, 190], [1260, 800], [700, 820]], np.float64)


def test_jaccard_of_ground_truth_is_one():
    assert smartdoc.jaccard(QUAD, QUAD, W, H) == pytest.approx(1.0)


def test_jaccard_scores_a_failed_scan_zero():
    assert smartdoc.jaccard(None, QUAD, W, H) == 0.0


def test_jaccard_is_measured_in_the_document_frame():
    # Detect the top half of the page, defined in document units, then map it
    # into the frame: the score must be exactly 1/2 whatever the perspective.
    half = np.array([[0, 0], [W, 0], [W, H / 2], [0, H / 2]], np.float32)
    to_frame = np.linalg.inv(smartdoc.doc_homography(QUAD, W, H))
    detected = cv2.perspectiveTransform(half.reshape(-1, 1, 2), to_frame)
    assert smartdoc.jaccard(detected.reshape(-1, 2), QUAD, W, H) == pytest.approx(
        0.5, abs=1e-4
    )


@pytest.mark.parametrize("turn", [0, 90, 180, 270])
def test_composited_marker_agrees_with_ground_truth_rotation(turn):
    frame = np.full((1080, 1920, 3), 200, np.uint8)
    # Turn the whole frame so the page is no longer upright.
    image = np.ascontiguousarray(np.rot90(frame, turn // 90))
    quad = rotate_points(QUAD, turn, frame.shape).astype(np.float64)

    composite = smartdoc.composite_marker(image, quad, W, H)
    marker = detect_marker(cv2.cvtColor(composite, cv2.COLOR_RGB2GRAY))

    assert marker is not None and marker.id == smartdoc.MARKER_ID
    assert upright_rotation(marker) == smartdoc.gt_rotation(quad) == (360 - turn) % 360
