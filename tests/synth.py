"""Render synthetic notebook pages so geometry can be tested without a model.

A page is drawn upright (paper, blue rules, red margin, handwriting, the real
ArUco marker upside-down in a corner), then placed on a darker background with
a known projective homography, an in-plane rotation plus small tilt, uneven
lighting, blur and noise. The returned mask mimics a segmentation model:
jagged page outline, small holes and a spurious background blob.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

PAGE_W, PAGE_H = 480, 640
MARKER_ID = 7
_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)


@dataclass(frozen=True)
class SyntheticPage:
    image: np.ndarray  # RGB uint8 HxWx3
    gray: np.ndarray  # uint8 HxW
    quad: np.ndarray  # (4, 2) float32 ground-truth page corners
    mask: np.ndarray  # bool HxW, model-like page mask
    rotation: int
    tilt: float
    page_size: tuple[int, int]  # (width, height) of the paper


def synthetic_page(
    rotation: int = 0,
    tilt: float = 0.0,
    seed: int = 0,
    marker_id: int = MARKER_ID,
    with_marker: bool = True,
    contaminate: bool = True,
) -> SyntheticPage:
    """Build one synthetic photo.

    ``rotation`` is one of 0/90/180/270 degrees CCW; ``tilt`` is a small extra
    in-plane rotation (degrees) left for the ruled-line refinement to find.
    """
    if rotation % 90 != 0:
        raise ValueError("rotation must be a multiple of 90")

    page = _render_page(seed, marker_id, with_marker)
    src = np.array(
        [[0, 0], [PAGE_W - 1, 0], [PAGE_W - 1, PAGE_H - 1], [0, PAGE_H - 1]],
        np.float64,
    )
    homography, quad, (width, height) = _placement(
        PAGE_W, PAGE_H, rotation, tilt, seed
    )

    warped = cv2.warpPerspective(
        page,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    alpha = cv2.warpPerspective(
        np.ones((PAGE_H, PAGE_W), np.float32),
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )[..., None]
    background = np.empty((height, width, 3), np.float32)
    background[:] = (54, 50, 46)
    image = alpha * warped.astype(np.float32) + (1.0 - alpha) * background
    image = _photometric(image, seed)

    mask = _model_mask((height, width), quad, seed, contaminate)
    return SyntheticPage(
        image=image,
        gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY),
        quad=quad.astype(np.float32),
        mask=mask,
        rotation=rotation,
        tilt=float(tilt),
        page_size=(PAGE_W, PAGE_H),
    )


def _render_page(seed: int, marker_id: int, with_marker: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    page = np.empty((PAGE_H, PAGE_W, 3), np.uint8)
    page[:] = (246, 245, 240)
    page = np.clip(
        page.astype(np.int16) + rng.normal(0, 2, page.shape[:2])[..., None],
        0,
        255,
    ).astype(np.uint8)

    margin_x = 72
    for y in range(96, PAGE_H - 24, 34):
        cv2.line(page, (20, y), (PAGE_W - 20, y), (120, 165, 205), 2, cv2.LINE_AA)
    cv2.line(
        page,
        (margin_x, 20),
        (margin_x, PAGE_H - 20),
        (205, 95, 85),
        2,
        cv2.LINE_AA,
    )

    ink = (42, 45, 55)
    for _ in range(7):
        y = int(rng.uniform(110, PAGE_H - 70))
        x0 = int(rng.uniform(margin_x + 18, PAGE_W * 0.45))
        length = int(rng.uniform(80, max(90, PAGE_W - x0 - 30)))
        pts = []
        for i in range(6):
            x = x0 + length * i / 5
            yy = y + (6 if i % 2 else -6) + int(rng.normal(0, 3))
            pts.append([int(x), int(yy)])
        cv2.polylines(
            page, [np.array(pts, np.int32)], False, ink, 3, cv2.LINE_AA
        )

    if with_marker:
        tile = _marker_tile(marker_id)
        th, tw = tile.shape
        x0 = PAGE_W - tw - 14
        y0 = 14
        page[y0 : y0 + th, x0 : x0 + tw] = cv2.cvtColor(
            tile, cv2.COLOR_GRAY2RGB
        )
    return page


def _marker_tile(marker_id: int) -> np.ndarray:
    bits = cv2.aruco.generateImageMarker(_DICT, int(marker_id), 110)
    tile = cv2.copyMakeBorder(
        bits, 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=255
    )
    # Printed upside-down on the page: page-up is marker-down.
    return cv2.rotate(tile, cv2.ROTATE_180)


def _placement(
    page_w: int, page_h: int, rotation: int, tilt: float, seed: int
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Compose rotation + perspective and fit the result on a canvas."""
    rng = np.random.default_rng(seed + 1000)
    cx, cy = (page_w - 1) / 2.0, (page_h - 1) / 2.0
    phi = np.deg2rad(rotation + tilt)
    cos, sin = np.cos(phi), np.sin(phi)
    rot = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], np.float64)
    to_origin = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], np.float64)
    from_origin = np.array([[1, 0, cx], [0, 1, cy], [0, 0, 1]], np.float64)
    px, py = rng.uniform(-1, 1, 2) * 7e-5
    perspective = np.array(
        [[1, 0, 0], [0, 1, 0], [px, py, 1]], np.float64
    )
    homography = from_origin @ perspective @ rot @ to_origin

    src = np.array(
        [[0, 0], [page_w - 1, 0], [page_w - 1, page_h - 1], [0, page_h - 1]],
        np.float64,
    )
    placed = _apply(homography, src)
    margin = 110
    min_x, min_y = placed.min(axis=0)
    max_x, max_y = placed.max(axis=0)
    shift = np.array(
        [[1, 0, margin - min_x], [0, 1, margin - min_y], [0, 0, 1]],
        np.float64,
    )
    homography = shift @ homography
    width = int(np.ceil(max_x - min_x + 2 * margin))
    height = int(np.ceil(max_y - min_y + 2 * margin))
    return homography, _apply(homography, src), (width, height)


def _photometric(image: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 2000)
    height, width = image.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    gx = xx / max(width - 1, 1) - 0.5
    gy = yy / max(height - 1, 1) - 0.5
    gain = 1.0 + 0.16 * gx + 0.10 * gy - 0.06 * (gx**2 + gy**2)
    out = image * gain[..., None]
    out = cv2.GaussianBlur(out, (0, 0), 1.0)
    out = out + rng.normal(0, 3.0, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _model_mask(
    shape: tuple[int, int], quad: np.ndarray, seed: int, contaminate: bool
) -> np.ndarray:
    rng = np.random.default_rng(seed + 3000)
    height, width = shape
    diag = float(np.linalg.norm(quad[2] - quad[0]))
    points = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        edge = b - a
        normal = np.array([-edge[1], edge[0]]) / max(np.linalg.norm(edge), 1.0)
        for t in np.linspace(0.0, 1.0, 10, endpoint=False):
            offset = rng.normal(0, 0.004 * diag)
            offset += 0.003 * diag * np.sin(2 * np.pi * 2 * t + i)
            points.append(a + edge * t + normal * offset)
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)

    if contaminate:
        for _ in range(3):
            u, v = rng.uniform(0.15, 0.85, 2)
            p = (
                (1 - u) * (1 - v) * quad[0]
                + u * (1 - v) * quad[1]
                + u * v * quad[2]
                + (1 - u) * v * quad[3]
            )
            cv2.circle(
                mask, (int(p[0]), int(p[1])), max(3, int(diag * 0.006)), 0, -1
            )

        radius = max(4, int(diag * 0.03))
        candidates = [
            (radius + 4, radius + 4),
            (width - radius - 5, radius + 4),
            (width - radius - 5, height - radius - 5),
            (radius + 4, height - radius - 5),
            (width // 2, radius + 4),
            (width // 2, height - radius - 5),
        ]
        poly = np.asarray(quad, np.float32).reshape(-1, 1, 2)
        best = max(
            candidates,
            key=lambda p: cv2.pointPolygonTest(
                poly, (float(p[0]), float(p[1])), True
            ),
        )
        if cv2.pointPolygonTest(poly, (float(best[0]), float(best[1])), True) < (
            -0.02 * diag
        ):
            cv2.circle(mask, best, radius, 1, -1)

    return mask.astype(bool)


def _apply(homography: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64)
    homo = np.column_stack([pts, np.ones(len(pts))]) @ homography.T
    return homo[:, :2] / homo[:, 2:3]
