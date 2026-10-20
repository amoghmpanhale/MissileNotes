"""Reduce a blobby page mask to a clean, straight four-corner page quad.

The model gives a probability mask that covers the page well but has jagged
edges and occasional stray blobs.  Ruled notebook lines give the page's true
in-plane angle; the mask outline gives its extent; the ArUco marker pins the
corner it is stuck to.  When those signals do not agree we say so instead of
quietly returning a bogus rectangle.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .marker import Marker

ANGLE_TOL = 6.0  # degrees; line segments this close to the dominant angle are ruled lines
PCTL = (0.5, 99.5)  # robust page extent in the deskewed frame
MIN_IOU = 0.5  # below this the quad is not a credible page outline


@dataclass(frozen=True)
class QuadResult:
    """Outcome of :func:`page_quad`.

    ``quad`` is (4, 2) float32 ordered TL/TR/BR/BL, or ``None``. ``reason`` is
    always human readable and explains a failure; ``confidence`` is the IoU
    between the quad and the cleaned page mask, in [0, 1].
    """

    quad: np.ndarray | None
    confidence: float
    reason: str


def clean_mask(mask: np.ndarray, marker: Marker | None = None) -> np.ndarray:
    """Keep one page island and fill it solid.

    With a marker, keep the island that contains its centre (the marker sits
    on the page); otherwise keep the largest island. Interior holes are filled
    through the outer contour, exactly like ``keep_marker_island`` did.
    """
    mask = np.asarray(mask).astype(bool)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return mask

    chosen = None
    if marker is not None:
        cx, cy = np.asarray(marker.corners, np.float64).mean(axis=0)
        inside = [
            c
            for c in contours
            if cv2.pointPolygonTest(c, (float(cx), float(cy)), False) >= 0
        ]
        if inside:
            chosen = max(inside, key=cv2.contourArea)
    if chosen is None:
        chosen = max(contours, key=cv2.contourArea)

    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, [chosen], -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def quad_mask(quad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Fill a (4, 2) quad into a solid bool mask of shape HxW."""
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.asarray(quad, np.int32), 1)
    return m.astype(bool)


def ruled_lines(gray: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Hough segments whose pixels lie on the page region, as an (N, 4) array."""
    edges = cv2.Canny(np.asarray(gray, np.uint8), 50, 150)
    edges[~np.asarray(region, bool)] = 0
    segs = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=max(24, gray.shape[1] // 6),
        maxLineGap=20,
    )
    return np.empty((0, 4)) if segs is None else segs.reshape(-1, 4)


def dominant_angle(segs: np.ndarray) -> tuple[float, np.ndarray]:
    """Median angle (deg, folded to -90..90) of the biggest parallel cluster."""
    segs = np.asarray(segs, np.float64)
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    ang = (ang + 90.0) % 180.0 - 90.0
    hist, edges = np.histogram(ang, bins=36, range=(-90.0, 90.0))
    center = edges[np.argmax(hist)] + 2.5
    keep = np.abs((ang - center + 90.0) % 180.0 - 90.0) < ANGLE_TOL
    return float(np.median(ang[keep])), keep


def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL via the sum/diff trick."""
    pts = np.asarray(pts, np.float32)
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    return np.array(
        [pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]],
        np.float32,
    )


def warp(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Perspective-warp the quad to a flat, front-on rectangle.

    Output size follows the quad's own edge lengths, so the page keeps its
    aspect ratio.
    """
    q = order_corners(quad)
    tl, tr, br, bl = q
    width = int(round(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl))))
    height = int(round(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl))))
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        np.float32,
    )
    m = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(np.asarray(img), m, (width, height))


def page_quad(
    mask: np.ndarray, marker: Marker | None, gray: np.ndarray
) -> QuadResult:
    """Recover the page quad from a model mask, marker and grayscale photo.

    All inputs share one frame (call :func:`missilenotes.marker.rotate_image`
    and ``rotate_points`` first). Returns a :class:`QuadResult`; never raises
    for an unusable image, and never silently substitutes a different region.
    """
    mask = np.asarray(mask).astype(bool)
    gray = np.asarray(gray)
    if mask.shape != gray.shape[:2]:
        raise ValueError("mask and gray must have the same HxW shape")

    region = clean_mask(mask, marker)
    if int(region.sum()) < max(64, int(0.002 * region.size)):
        return QuadResult(None, 0.0, "no page region found in the mask")

    segs = ruled_lines(gray, region)
    theta = None
    source = ""
    if len(segs) >= 3:
        candidate, keep = dominant_angle(segs)
        if int(keep.sum()) >= 3:
            theta, source = candidate, "ruled lines"
    if theta is None and marker is not None:
        theta = _marker_angle(marker)
        source = "marker orientation (no ruled lines)"
    if theta is None:
        return QuadResult(
            None, 0.0, "cannot determine page angle: no ruled lines or marker"
        )

    ys, xs = np.nonzero(region)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    center = np.array([gray.shape[1] / 2.0, gray.shape[0] / 2.0])

    deskewed = _rotate_about(pts, -theta, center)
    lo = np.percentile(deskewed, PCTL[0], axis=0)
    hi = np.percentile(deskewed, PCTL[1], axis=0)
    if marker is not None:
        # The marker sits on a page corner: never let the extent cut into it.
        anchors = _rotate_about(
            np.asarray(marker.corners, np.float64), -theta, center
        )
        lo = np.minimum(lo, anchors.min(axis=0))
        hi = np.maximum(hi, anchors.max(axis=0))

    box = np.array(
        [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]]
    )
    quad = _rotate_about(box, theta, center).astype(np.float32)

    if not np.isfinite(quad).all() or abs(cv2.contourArea(quad)) < 1.0:
        return QuadResult(None, 0.0, "page extent is degenerate")

    shape = gray.shape[:2]
    qmask = quad_mask(quad, shape)
    union = np.logical_or(qmask, region).sum()
    iou = float(np.logical_and(qmask, region).sum() / union) if union else 0.0
    if iou < MIN_IOU:
        return QuadResult(
            None, iou, f"quad disagrees with the page mask (IoU {iou:.2f})"
        )

    return QuadResult(order_corners(quad), iou, f"ok ({source})")


def _rotate_about(pts: np.ndarray, deg: float, center: np.ndarray) -> np.ndarray:
    """Rotate points by ``deg`` mathematically about ``center`` (y-down frame)."""
    r = np.deg2rad(deg)
    rot = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return (np.asarray(pts, np.float64) - center) @ rot.T + center


def _marker_angle(marker: Marker) -> float:
    """Page horizontal angle implied by the marker's orientation.

    Page-up is marker-down, so the page's right direction is page-up rotated
    +90 degrees in image coordinates.
    """
    c = np.asarray(marker.corners, np.float64)
    up = c[3] - c[0]  # page-up (marker-down)
    horizontal = np.array([-up[1], up[0]])
    angle = np.degrees(np.arctan2(horizontal[1], horizontal[0]))
    return float((angle + 90.0) % 180.0 - 90.0)
