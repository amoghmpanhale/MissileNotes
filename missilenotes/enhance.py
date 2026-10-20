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
