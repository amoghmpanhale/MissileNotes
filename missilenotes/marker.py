"""ArUco marker detection and rotation of the photo into an upright frame.

The marker is printed upside-down relative to the page, so page-up is the
marker's down direction.  Every rotation here is expressed in degrees
counter-clockwise as seen on screen (image coordinates, y pointing down).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

_DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
_DETECTOR = cv2.aruco.ArucoDetector(
    _DICTIONARY, cv2.aruco.DetectorParameters()
)


@dataclass(frozen=True)
class Marker:
    """A detected marker. ``corners`` is (4, 2) float32, TL/TR/BR/BL in the
    marker's own canonical frame (not necessarily the image's top-left)."""

    corners: np.ndarray
    id: int


def detect_marker(gray: np.ndarray) -> Marker | None:
    """Find the ArUco marker in a grayscale image, once.

    Returns the lowest-id marker if several are visible, or ``None`` when the
    photo contains no marker. Callers carry ``marker.corners`` through
    :func:`rotate_points` instead of re-detecting after a rotation.
    """
    gray = np.asarray(gray)
    corners, ids, _ = _DETECTOR.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    i = int(np.argmin(ids.ravel()))
    marker_id = int(np.asarray(ids).reshape(-1)[i])
    return Marker(
        corners=np.asarray(corners[i], np.float32).reshape(4, 2), id=marker_id
    )


def upright_rotation(marker: Marker) -> int:
    """Degrees CCW (0/90/180/270) that make the page upright.

    ``corners[0] - corners[3]`` is the marker's up vector. The page is the
    other way up, so we need to rotate the marker's up vector to point down
    (+y) in image coordinates; quantising to the nearest quadrant absorbs the
    small in-plane tilt that the ruled-line refinement handles later.
    """
    c = np.asarray(marker.corners, np.float64)
    ux, uy = c[0] - c[3]
    angle = np.degrees(np.arctan2(uy, ux))
    return int(round((angle - 90.0) / 90.0)) * 90 % 360


def rotate_image(img: np.ndarray, deg: float) -> np.ndarray:
    """Rotate an image CCW by ``deg`` with an enlarged canvas.

    Multiples of 90 use exact array rotations so pixels are never resampled;
    other angles use an affine warp around the image centre.
    """
    deg = float(deg) % 360.0
    if abs(deg - round(deg)) < 1e-9 and int(round(deg)) % 90 == 0:
        k = (int(round(deg)) // 90) % 4
        return np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))

    h, w = img.shape[:2]
    m, new_w, new_h = _rotation_matrix(deg, w, h)
    if img.dtype == bool:
        warped = cv2.warpAffine(
            img.astype(np.uint8), m, (new_w, new_h),
            flags=cv2.INTER_NEAREST,
        )
        return warped.astype(bool)
    return cv2.warpAffine(img, m, (new_w, new_h), flags=cv2.INTER_LINEAR)


def rotate_points(
    pts: np.ndarray, deg: float, shape: tuple[int, ...]
) -> np.ndarray:
    """Rotate (N, 2) x/y points by ``deg`` CCW into the rotated frame.

    ``shape`` is the original image shape (H, W[, C]). The result lands in the
    same enlarged frame that :func:`rotate_image` produces, so marker corners,
    page quads and masks all stay registered.
    """
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    h, w = shape[:2]
    k = int(round(float(deg))) % 360
    if abs(float(deg) - round(float(deg))) < 1e-9 and k % 90 == 0:
        x, y = pts[:, 0], pts[:, 1]
        if k == 0:
            out = np.column_stack([x, y])
        elif k == 90:
            out = np.column_stack([y, (w - 1) - x])
        elif k == 180:
            out = np.column_stack([(w - 1) - x, (h - 1) - y])
        else:
            out = np.column_stack([(h - 1) - y, x])
        return out.astype(np.float32)

    m, _, _ = _rotation_matrix(float(deg), w, h)
    return (pts @ m[:, :2].T + m[:, 2]).astype(np.float32)


def _rotation_matrix(deg: float, w: int, h: int) -> tuple[np.ndarray, int, int]:
    """Expanded affine matrix for a CCW rotation by ``deg``."""
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(round(h * sin + w * cos))
    new_h = int(round(h * cos + w * sin))
    m[0, 2] += new_w / 2.0 - w / 2.0
    m[1, 2] += new_h / 2.0 - h / 2.0
    return m, new_w, new_h
