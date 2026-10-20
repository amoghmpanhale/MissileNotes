"""Turn a flat grayscale scan into clean black ink on white paper."""

from __future__ import annotations

import cv2
import numpy as np


def binarize(
    gray: np.ndarray, block: int = 41, C: int = 15, min_area: int = 6
) -> np.ndarray:
    """Return an HxW uint8 image with 0 = ink and 255 = paper.

    A local Gaussian threshold handles uneven lighting and shadows; the
    connected-component pass removes specks smaller than a pen stroke.
    """
    gray = np.asarray(gray, np.uint8)
    ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block | 1,
        C,
    )  # 255 where ink
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    ink = np.where(keep[labels], 255, 0).astype(np.uint8)
    return 255 - ink


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Flatten paper brightness so ink separates cleanly in shadow.

    A large median estimates the paper: its window is wider than a pen stroke,
    so thin ink is replaced by nearby paper, but it keeps a step edge (a hard
    cast shadow) sharp because the median picks the majority side instead of
    averaging across it. Dividing by that background makes both the lit and
    shadowed paper white. A blur or a wide morphological kernel would smear
    the shadow edge into a seam that the divider turns into a false ink band,
    which is why the edge-preserving median is used.
    """
    gray = np.asarray(gray, np.uint8)
    height, width = gray.shape[:2]
    limit = min(height, width)
    size = (limit // 10) | 1
    size = min(max(3, size), 99, limit if limit % 2 else limit - 1)
    background = cv2.medianBlur(gray, size)
    background = np.maximum(background, 1)
    return cv2.divide(gray, background, scale=255)


def remove_rules(flat_rgb: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """Clear ruled and margin lines from a binarized scan, sparing ink.

    A rule is two things at once: geometrically long and thin, and printed in
    a rule colour. Morphological opening with page-spanning kernels finds the
    first; HSV thresholds find the second. Clearing only the intersection
    keeps dark handwriting that happens to form a long stroke, and keeps pen
    strokes crossing a rule because those pixels are ink-coloured, not
    rule-coloured. A full-width fallback handles faded rules with little
    colour left, but still refuses to erase very dark (HSV value) pixels.
    """
    flat_rgb = np.asarray(flat_rgb, np.uint8)
    binary = np.asarray(binary, np.uint8)
    if flat_rgb.shape[:2] != binary.shape[:2]:
        raise ValueError("flat_rgb and binary must have the same HxW shape")

    height, width = binary.shape[:2]
    ink = 255 - binary

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(15, width // 4), 1)
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(15, height // 4))
    )
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, horizontal_kernel) > 0
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, vertical_kernel) > 0
    line_like = horizontal | vertical

    hsv = cv2.cvtColor(flat_rgb, cv2.COLOR_RGB2HSV)
    hue, saturation, value = cv2.split(hsv)
    blue_rule = (
        (hue >= 85) & (hue <= 130) & (saturation >= 45) & (value >= 120)
    )
    red_rule = (
        ((hue <= 8) | (hue >= 172)) & (saturation >= 70) & (value >= 90)
    )
    rule_colour = blue_rule | red_rule

    # Antialiasing leaves coloured rule pixels just beside the binary line.
    line_like = cv2.dilate(
        line_like.astype(np.uint8), np.ones((3, 3), np.uint8)
    ) > 0
    remove = line_like & rule_colour

    # Faded rules may have almost no colour left. A row/column almost entirely
    # covered by ink is a full-width rule; clear it except where the source
    # pixel is genuinely dark, which is the handwriting crossing it.
    full_row = (ink.mean(axis=1) / 255.0) > 0.5
    full_column = (ink.mean(axis=0) / 255.0) > 0.5
    full_width = np.zeros_like(line_like)
    full_width[full_row, :] = True
    full_width[:, full_column] = True
    remove |= full_width & line_like & (value >= 90)

    # Antialiased stroke edges blend ink with rule colour, so they can look
    # like a faded rule. Protect a one-pixel rim around genuinely dark pixels:
    # it costs almost no rule removal but keeps handwriting that crosses a
    # rule intact.
    dark_ink = (value < 90).astype(np.uint8)
    protected = cv2.dilate(dark_ink, np.ones((3, 3), np.uint8)) > 0
    remove &= ~protected

    out = binary.copy()
    out[remove] = 255
    return out
