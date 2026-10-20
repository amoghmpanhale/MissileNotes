import cv2
import numpy as np
import pytest

from missilenotes.geometry import clean_mask, order_corners, page_quad, warp
from missilenotes.marker import Marker, detect_marker, rotate_image, rotate_points, upright_rotation
from tests.synth import PAGE_H, PAGE_W, synthetic_page

ROTATIONS = [0, 90, 180, 270]
TILTS = [-24.0, -12.0, 0.0, 12.0, 24.0]


def _upright(synth):
    marker = detect_marker(synth.gray)
    assert marker is not None
    deg = upright_rotation(marker)
    image = rotate_image(synth.image, deg)
    gray = rotate_image(synth.gray, deg)
    mask = rotate_image(synth.mask, deg).astype(bool)
    quad = rotate_points(synth.quad, deg, synth.image.shape)
    marker = Marker(
        corners=rotate_points(marker.corners, deg, synth.image.shape),
        id=marker.id,
    )
    return image, gray, mask, quad, marker


def _corner_error(actual, expected):
    a = order_corners(actual)
    b = order_corners(expected)
    diag = float(np.linalg.norm(b[0] - b[2]))
    return float(np.linalg.norm(a - b, axis=1).max()), diag


@pytest.mark.parametrize("rotation", ROTATIONS)
@pytest.mark.parametrize("tilt", TILTS)
def test_page_quad_recovered(rotation, tilt):
    synth = synthetic_page(rotation=rotation, tilt=tilt, seed=rotation + int(tilt) + 40)
    image, gray, mask, quad, marker = _upright(synth)

    result = page_quad(mask, marker, gray)

    assert result.quad is not None, result.reason
    assert result.confidence > 0.6
    error, diag = _corner_error(result.quad, quad)
    assert error <= 0.02 * diag, f"{error:.1f}px > 2% of {diag:.1f}px"


def test_spurious_blob_and_holes_do_not_change_quad():
    clean = synthetic_page(rotation=0, tilt=10, seed=11, contaminate=False)
    dirty = synthetic_page(rotation=0, tilt=10, seed=11, contaminate=True)

    # Same rendering: only the model-like mask differs.
    assert np.array_equal(clean.image, dirty.image)
    assert dirty.mask.sum() != clean.mask.sum()

    _, gray_c, mask_c, quad_c, marker_c = _upright(clean)
    _, gray_d, mask_d, quad_d, marker_d = _upright(dirty)
    result_c = page_quad(mask_c, marker_c, gray_c)
    result_d = page_quad(mask_d, marker_d, gray_d)

    assert result_c.quad is not None, result_c.reason
    assert result_d.quad is not None, result_d.reason
    error, diag = _corner_error(result_d.quad, result_c.quad)
    assert error <= 0.01 * diag

    cleaned = clean_mask(mask_d, marker_d)
    n, _ = cv2.connectedComponents(cleaned.astype(np.uint8))
    assert n == 2, "spurious island must be dropped"
    assert abs(cleaned.sum() - mask_c.sum()) <= 0.01 * mask_c.sum()


def test_no_page_region_has_reason():
    gray = np.full((200, 240), 255, np.uint8)
    result = page_quad(np.zeros_like(gray, bool), None, gray)
    assert result.quad is None
    assert result.confidence == 0.0
    assert "no page region" in result.reason


def test_no_angle_source_has_reason():
    gray = np.full((300, 300), 255, np.uint8)  # no ruled lines, no marker
    mask = np.zeros_like(gray, bool)
    mask[50:250, 60:240] = True

    result = page_quad(mask, None, gray)
    assert result.quad is None
    assert "angle" in result.reason


@pytest.mark.parametrize("rotation", ROTATIONS)
def test_warp_preserves_page_aspect(rotation):
    synth = synthetic_page(rotation=rotation, tilt=12, seed=rotation + 90)
    image, gray, mask, quad, marker = _upright(synth)
    result = page_quad(mask, marker, gray)
    assert result.quad is not None, result.reason

    flat = warp(image, result.quad)
    expected = PAGE_W / PAGE_H
    assert abs((flat.shape[1] / flat.shape[0]) / expected - 1.0) <= 0.03
