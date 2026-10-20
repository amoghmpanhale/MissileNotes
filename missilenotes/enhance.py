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
    shadowed paper white. A blur or a wide median alone would smear the shadow
    edge into a seam that the divider turns into a false ink band, hence the
    edge-preserving median.

    A large solid dark mark (the ArUco marker) is wider than the median window
    and would be treated as paper, making its interior speckled after
    division. Before the median, a morphological close fills dark features
    narrower than the blob kernel (marker modules, pen strokes) with
    surrounding paper; a cast shadow is much wider than that kernel and stays
    in the estimate, so it is still divided out.
    """
    gray = np.asarray(gray, np.uint8)
    height, width = gray.shape[:2]
    limit = min(height, width)

    def _odd(value: int, low: int, high: int) -> int:
        size = min(max(low, value | 1), high)
        if size % 2 == 0:
            size -= 1
        return min(size, limit if limit % 2 else max(1, limit - 1))

    background_size = _odd(limit // 10, 3, 99)
    blob_size = _odd(limit // 15, 15, 99)
    blob_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (blob_size, blob_size)
    )
    background_source = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, blob_kernel)
    background = cv2.medianBlur(background_source, background_size)
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
    colour left, but still refuses to erase very dark (HSV value) pixels. A
    second pass then removes short line-shaped leftovers lying wholly inside
    a detected rule band (jittered dashes, a dotted margin), which a single
    long-kernel opening misses, while strokes crossing the band survive.
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

    # Rule bands: rows/columns carrying many rule-coloured or long-line
    # pixels. Opening sees only unbroken runs, so a jittered or faded rule can
    # leave short dashes and a dotted margin; these bands let a second pass
    # find them.
    row_rule = (
        (rule_colour.sum(axis=1) >= 0.25 * width)
        | (horizontal.sum(axis=1) >= 0.25 * width)
        | full_row
    )
    column_rule = (
        (rule_colour.sum(axis=0) >= 0.25 * height)
        | (vertical.sum(axis=0) >= 0.25 * height)
        | full_column
    )
    band_rows = cv2.dilate(
        row_rule.astype(np.uint8)[:, None], np.ones((5, 1), np.uint8)
    ).ravel() > 0
    band_columns = cv2.dilate(
        column_rule.astype(np.uint8)[None, :], np.ones((1, 5), np.uint8)
    ).ravel() > 0
    in_band = band_rows[:, None] | band_columns[None, :]

    # Light rule-coloured pixels inside a band are rule even when broken into
    # dashes; dark pixels (handwriting) fail this test.
    remove |= in_band & rule_colour & (value >= 150)

    out = binary.copy()
    out[remove] = 255

    # Finally clear small line-shaped components lying wholly inside a rule
    # band. A handwriting stroke crossing a rule extends outside the band, so
    # its component is not touched.
    leftover = 255 - out
    count, labels, stats, _ = cv2.connectedComponentsWithStats(leftover, 8)
    max_area = max(64, int(0.004 * height * width))
    for i in range(1, count):
        x, y, width_i, height_i, area = stats[i]
        if area > max_area:
            continue
        horizontal_fragment = (
            height_i <= max(4, int(0.30 * width_i))
            and band_rows[y : y + height_i].all()
        )
        vertical_fragment = (
            width_i <= max(4, int(0.30 * height_i))
            and band_columns[x : x + width_i].all()
        )
        if horizontal_fragment or vertical_fragment:
            out[labels == i] = 255
    return out
