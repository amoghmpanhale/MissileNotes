import numpy as np
import pytest

from missilenotes.marker import (
    Marker,
    detect_marker,
    rotate_image,
    rotate_points,
    upright_rotation,
)
from tests.synth import synthetic_page


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("tilt", [0.0, 12.0, -12.0])
def test_marker_detected_and_upright_rotation(rotation, tilt):
    synth = synthetic_page(rotation=rotation, tilt=tilt, seed=rotation + 7)
    marker = detect_marker(synth.gray)

    assert marker is not None
    assert marker.id == 7
    assert marker.corners.shape == (4, 2)
    # The photo was turned `rotation` degrees CCW, so undo that.
    assert upright_rotation(marker) == (360 - rotation) % 360


def test_no_marker_returns_none():
    blank = np.full((240, 320), 255, np.uint8)
    assert detect_marker(blank) is None
    assert detect_marker(synthetic_page(with_marker=False).gray) is None


@pytest.mark.parametrize("deg", [0, 90, 180, 270])
def test_rotate_points_track_image_rotation(deg):
    img = np.zeros((7, 11), np.uint8)
    img[2, 8] = 255

    rotated = rotate_image(img, deg)
    point = rotate_points([[8, 2]], deg, img.shape)[0]

    expected_shape = (11, 7) if deg in (90, 270) else (7, 11)
    assert rotated.shape == expected_shape
    assert rotated[int(point[1]), int(point[0])] == 255


def test_upright_rotation_uses_marker_frame_not_image_frame():
    # Canonical TL/TR/BR/BL of a marker printed upside-down on an upright page.
    corners = np.array(
        [[10, 10], [0, 10], [0, 0], [10, 0]], np.float32
    )
    assert upright_rotation(Marker(corners=corners, id=3)) == 0


def test_upright_rotation_after_quarter_turn():
    corners = np.array(
        [[10, 10], [0, 10], [0, 0], [10, 0]], np.float32
    )
    # Screen-CCW 90 degrees: (x, y) -> (y, -x).
    turned = np.column_stack([corners[:, 1], -corners[:, 0]]).astype(np.float32)
    assert upright_rotation(Marker(corners=turned, id=3)) == 270
