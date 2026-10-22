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
# degrees; how far ruled lines may move the marker/image-axes page angle
RULE_AGREE_TOL = 20.0
MIN_IOU = 0.5  # below this the quad is not a credible page outline


@dataclass(frozen=True)
class QuadResult:
    """Outcome of :func:`page_quad`.

    ``quad`` is (4, 2) float32 ordered TL/TR/BR/BL in the page's own axes
    (page-up, which may be rotated relative to the image), or ``None``.
    ``reason`` is always human readable and explains a failure; ``confidence``
    is the IoU between the quad and the cleaned page mask, in [0, 1].
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
    # Under perspective the near edge is longer than the far one; averaging
    # opposite sides recovers the page's true width/height better than max().
    width = int(
        round((np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2.0)
    )
    height = int(
        round((np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2.0)
    )
    width, height = max(width, 1), max(height, 1)
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

    contour = _page_contour(region)
    if contour is None or len(contour) < 8:
        return QuadResult(None, 0.0, "page region has no usable outline")

    # The marker gives a coarse page angle; ruled lines may only refine it.
    # Hough also finds phantom diagonals through tables and text grids, and a
    # 45-degree theta makes corner ordering collapse. No marker means the
    # caller already rotated the frame upright (scan's rotation=), so the
    # image axes are the reference.
    if marker is not None:
        theta, ref_name = _marker_angle(marker), "marker orientation"
    else:
        theta, ref_name = 0.0, "image axes"
    source = f"{ref_name} (no ruled lines)"
    segs = ruled_lines(gray, region)
    if len(segs) >= 3:
        candidate, keep = dominant_angle(segs)
        if int(keep.sum()) >= 3:
            # Rules may be horizontal or vertical; compare modulo 90 degrees.
            offset = (candidate - theta + 45.0) % 90.0 - 45.0
            if abs(offset) <= RULE_AGREE_TOL:
                theta, source = theta + offset, "ruled lines"
            else:
                source = f"{ref_name} (ruled lines disagree)"

    # Seed with a four-vertex contour approximation, then fit each side to the
    # outline points and intersect adjacent lines for sub-pixel corners.
    quad = _order_by_angle(_initial_quad(contour), theta)
    outline = contour.reshape(-1, 2).astype(np.float64)
    refined = _refine_quad(outline, quad)
    if refined is None:
        refined = quad
        source += "; edges not refined"
    refined = _order_by_angle(refined, theta)

    # The marker is stuck on a page corner: expanding that corner (never
    # shrinking it) keeps the page quad from cutting through the marker.
    contained = _contain_marker(refined, marker)
    if contained is None:
        return QuadResult(None, 0.0, "page quad cannot include the marker")
    refined = contained

    area = abs(cv2.contourArea(refined.reshape(-1, 1, 2).astype(np.float32)))
    if not np.isfinite(refined).all() or area < 1.0:
        return QuadResult(None, 0.0, "page quad is degenerate")

    shape = gray.shape[:2]
    qmask = quad_mask(refined, shape)
    union = np.logical_or(qmask, region).sum()
    iou = float(np.logical_and(qmask, region).sum() / union) if union else 0.0
    if iou < MIN_IOU:
        return QuadResult(
            None, iou, f"quad disagrees with the page mask (IoU {iou:.2f})"
        )

    return QuadResult(refined.astype(np.float32), iou, f"ok ({source})")


def _page_contour(region: np.ndarray) -> np.ndarray | None:
    """Largest outer contour of the cleaned page region, every boundary pixel."""
    contours, _ = cv2.findContours(
        region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def _initial_quad(contour: np.ndarray) -> np.ndarray:
    """Four corner candidates from a convex hull, falling back to minAreaRect."""
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    for frac in np.linspace(0.005, 0.08, 16):
        approx = cv2.approxPolyDP(hull, float(frac) * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float64)
        if len(approx) < 4:
            break
    return cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64)


def _order_by_angle(quad: np.ndarray, theta: float) -> np.ndarray:
    """Order corners TL/TR/BR/BL in the page's own axes.

    ``theta`` is the page's horizontal angle, so the sum/diff trick works for
    any camera rotation instead of only for an axis-aligned page.
    """
    r = np.deg2rad(theta)
    h = np.array([np.cos(r), np.sin(r)])
    v = np.array([-np.sin(r), np.cos(r)])
    u = np.asarray(quad, np.float64) @ h
    w = np.asarray(quad, np.float64) @ v
    s = u + w
    d = u - w
    q = np.asarray(quad, np.float64)
    return np.array([q[s.argmin()], q[d.argmax()], q[s.argmax()], q[d.argmin()]])


def _line_coeffs(line: np.ndarray) -> np.ndarray:
    """Homogeneous (a, b, c) for a line given as (vx, vy, x0, y0)."""
    vx, vy, x0, y0 = line
    return np.array([-vy, vx, vy * x0 - vx * y0], np.float64)


def _line_through(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    d = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    norm = float(np.linalg.norm(d))
    if norm < 1e-9:
        return None
    return np.array([d[0] / norm, d[1] / norm, a[0], a[1]], np.float64)


def _intersect_lines(l1: np.ndarray, l2: np.ndarray) -> np.ndarray | None:
    p = np.cross(_line_coeffs(l1), _line_coeffs(l2))
    if abs(p[2]) < 1e-9:
        return None
    return p[:2] / p[2]


def _refine_quad(
    points: np.ndarray, quad: np.ndarray, iterations: int = 3
) -> np.ndarray | None:
    """Robustly fit each side of ``quad`` to ``points`` and re-intersect.

    Points are assigned to the closest side by segment distance, ignoring the
    corner neighbourhoods so adjacent edges do not steal each other's points.
    Returns ``None`` if any side is under-determined or the result runs away.
    """
    quad = np.asarray(quad, np.float64)
    diag = float(np.linalg.norm(quad[2] - quad[0]))
    for _ in range(iterations):
        # Nearest side per point, using the whole segment so a point is never
        # handed to a far side just because it is past its own corner.
        assign = np.full(len(points), -1, np.int32)
        best = np.full(len(points), np.inf, np.float64)
        for i in range(4):
            a, b = quad[i], quad[(i + 1) % 4]
            seg = b - a
            length2 = float(seg @ seg)
            if length2 < 1e-12:
                return None
            t = ((points - a) @ seg) / length2
            inside = (t >= 0.0) & (t <= 1.0)
            dist = np.linalg.norm(points - (a + t[:, None] * seg), axis=1)
            better = inside & (dist < best)
            assign[better] = i
            best[better] = dist[better]

        lines = []
        for i in range(4):
            members = np.nonzero(assign == i)[0]
            if len(members) >= 6:
                a, b = quad[i], quad[(i + 1) % 4]
                seg = b - a
                t = ((points[members] - a) @ seg) / float(seg @ seg)
                # Corner neighbourhoods belong to two sides; drop them.
                members = members[(t >= 0.05) & (t <= 0.95)]
            side = points[members]
            if len(side) >= 6:
                vx, vy, x0, y0 = cv2.fitLine(
                    side.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01
                ).ravel()
                lines.append(np.array([vx, vy, x0, y0], np.float64))
            else:
                line = _line_through(quad[i], quad[(i + 1) % 4])
                if line is None:
                    return None
                lines.append(line)

        corners = []
        for i in range(4):
            corner = _intersect_lines(lines[(i - 1) % 4], lines[i])
            if corner is None:
                return None
            corners.append(corner)
        refined = np.array(corners, np.float64)

        if not np.isfinite(refined).all():
            return None
        if not cv2.isContourConvex(refined.reshape(-1, 1, 2).astype(np.float32)):
            return None
        if float(np.linalg.norm(refined - quad, axis=1).max()) > 0.1 * diag:
            return None
        quad = refined
    return quad


def _contain_marker(
    quad: np.ndarray, marker: Marker | None
) -> np.ndarray | None:
    """Expand the corner nearest the marker until all its corners are inside."""
    if marker is None:
        return quad
    corners = np.asarray(marker.corners, np.float64).reshape(-1, 2)
    poly = quad.reshape(-1, 1, 2).astype(np.float32)
    if all(
        cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0
        for x, y in corners
    ):
        return quad

    centre = quad.mean(axis=0)
    index = int(np.argmin(np.linalg.norm(quad - corners.mean(axis=0), axis=1)))
    outward = quad[index] - centre
    if float(np.linalg.norm(outward)) < 1e-9:
        return None
    for step in np.linspace(0.0, 1.0, 51):
        candidate = quad.copy()
        candidate[index] = quad[index] + outward * step
        candidate_poly = candidate.reshape(-1, 1, 2).astype(np.float32)
        if not cv2.isContourConvex(candidate_poly):
            break
        if all(
            cv2.pointPolygonTest(candidate_poly, (float(x), float(y)), False) >= 0
            for x, y in corners
        ):
            return candidate
    return None


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
