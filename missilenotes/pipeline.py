"""End-to-end scan: photo in, flat upright page out.

The pipeline is deliberately numpy-in/numpy-out so the mobile port can reuse
the same stage order. The caller injects a segmenter (anything with
``predict(img_rgb) -> HxW probability``) so tests can use a fake and devices
can swap in their own backend.

Stage order and frames:

1. Detect the ArUco marker in the original photo and rotate the photo upright
   (marker down, page up). ``ScanResult.rotation`` records this CCW angle.
2. Rotate another 90 degrees clockwise into the horizontal model's frame,
   predict, then undo the rotation to get a mask in the upright frame.
3. Recover the page quad, warp it upright, and binarize. Optional stages can
   flatten lighting first and remove printed rules afterwards.
4. Express the quad back in the original photo's coordinates so callers can
   overlay it on the photo they passed in.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .enhance import binarize, normalize_illumination
from .enhance import remove_rules as remove_rule_lines
from .geometry import page_quad, warp
from .marker import (
    Marker,
    detect_marker,
    rotate_image,
    rotate_points,
    upright_rotation,
)


@dataclass
class ScanResult:
    """Outcome of :func:`scan`.

    ``scan`` is the binarized uint8 image (0 = ink, 255 = paper) or ``None``
    when binarization was disabled. ``flat_rgb`` is always the upright warped
    colour image when warping succeeded. ``quad`` is (4, 2) TL/TR/BR/BL in the
    original photo's pixel coordinates. ``rotation`` is the CCW angle applied
    to the photo to make the page upright. ``timings`` holds milliseconds per
    stage that actually ran.
    """

    ok: bool
    reason: str
    scan: np.ndarray | None
    flat_rgb: np.ndarray | None
    quad: np.ndarray | None
    rotation: int
    confidence: float
    timings: dict[str, float]
    # Thresholded page mask in the original photo frame; only used by
    # debug_overlay, so it is kept out of repr/compare.
    mask: np.ndarray | None = field(default=None, repr=False, compare=False)


def scan(
    img_rgb: np.ndarray,
    segmenter,
    *,
    threshold: float = 0.5,
    binarize_output: bool = True,
    normalize: bool = False,
    remove_rules: bool = False,
) -> ScanResult:
    """Scan one RGB photo. Never raises for a bad photo.

    Missing markers, pages the model cannot find, and unusable geometry all
    return ``ok=False`` with a human-readable reason. Only programmer errors
    (wrong array shape/dtype) raise.

    ``normalize`` flattens uneven lighting and shadows before binarizing.
    ``remove_rules`` clears printed ruled/margin lines after binarizing; it is
    ignored when ``binarize_output`` is false. Both default to off so the
    default output matches the historical pipeline exactly.
    """
    img_rgb = _validate(img_rgb)
    timings: dict[str, float] = {}

    start = time.perf_counter()
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    marker = detect_marker(gray)
    timings["marker"] = _elapsed_ms(start)

    if marker is None:
        return ScanResult(
            False, "no ArUco marker found", None, None, None, 0, 0.0, timings
        )

    rotation = upright_rotation(marker)
    upright = rotate_image(img_rgb, rotation)
    upright_marker = Marker(
        corners=rotate_points(marker.corners, rotation, img_rgb.shape),
        id=marker.id,
    )
    upright_gray = cv2.cvtColor(upright, cv2.COLOR_RGB2GRAY)

    # The horizontal model expects the page turned 90 degrees clockwise from
    # upright; -90 is a clockwise quarter turn in rotate_image's CCW convention.
    model_img = rotate_image(upright, -90)

    start = time.perf_counter()
    prob_model = np.asarray(segmenter.predict(model_img), np.float32)
    timings["predict"] = _elapsed_ms(start)
    if prob_model.shape != model_img.shape[:2]:
        raise ValueError(
            "segmenter.predict must return an HxW map matching its input; "
            f"got {prob_model.shape} for {model_img.shape[:2]}"
        )

    # Undo the 90 degrees clockwise turn: 90 CCW returns to the upright frame.
    prob_upright = np.rot90(prob_model, 1)
    mask_upright = prob_upright > threshold

    start = time.perf_counter()
    quad_result = page_quad(mask_upright, upright_marker, upright_gray)
    timings["geometry"] = _elapsed_ms(start)

    # Keep a mask overlay in the frame of the photo the caller passed in.
    mask_original = rotate_image(mask_upright.astype(np.uint8), -rotation) > 0

    if quad_result.quad is None:
        return ScanResult(
            False,
            quad_result.reason,
            None,
            None,
            None,
            rotation,
            quad_result.confidence,
            timings,
            mask_original,
        )

    quad_original = rotate_points(quad_result.quad, -rotation, upright.shape)

    start = time.perf_counter()
    flat_rgb = warp(upright, quad_result.quad)
    timings["warp"] = _elapsed_ms(start)

    scan_image = None
    if binarize_output:
        flat_gray = cv2.cvtColor(flat_rgb, cv2.COLOR_RGB2GRAY)
        if normalize:
            start = time.perf_counter()
            flat_gray = normalize_illumination(flat_gray)
            timings["normalize"] = _elapsed_ms(start)

        start = time.perf_counter()
        scan_image = binarize(flat_gray)
        timings["binarize"] = _elapsed_ms(start)

        if remove_rules:
            start = time.perf_counter()
            scan_image = remove_rule_lines(flat_rgb, scan_image)
            timings["remove_rules"] = _elapsed_ms(start)

    return ScanResult(
        True,
        quad_result.reason,
        scan_image,
        flat_rgb,
        quad_original,
        rotation,
        quad_result.confidence,
        timings,
        mask_original,
    )


def debug_overlay(img_rgb: np.ndarray, result: ScanResult) -> np.ndarray:
    """Return a copy of ``img_rgb`` with the page mask, quad and marker drawn."""
    out = img_rgb.copy()
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    if result.mask is not None:
        contours, _ = cv2.findContours(
            result.mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(out, contours, -1, (0, 170, 255), 2)

    if result.quad is not None:
        quad = np.asarray(result.quad, np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [quad], True, (0, 255, 0), 3)

    marker = detect_marker(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY))
    if marker is not None:
        corners = np.asarray(marker.corners, np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [corners], True, (255, 0, 255), 2)

    return out


def _validate(img_rgb: np.ndarray) -> np.ndarray:
    if not isinstance(img_rgb, np.ndarray):
        raise TypeError("img_rgb must be a numpy array")
    if img_rgb.dtype != np.uint8:
        raise ValueError(f"img_rgb must be uint8, got {img_rgb.dtype}")
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"img_rgb must be HxWx3 RGB, got {img_rgb.shape}")
    if img_rgb.shape[0] == 0 or img_rgb.shape[1] == 0:
        raise ValueError("img_rgb must have non-zero height and width")
    return img_rgb


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0
