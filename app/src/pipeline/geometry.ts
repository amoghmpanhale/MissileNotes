// Pure geometry helpers with NO native imports — safe to unit-test in plain
// Node (see scan.test.ts). scan.ts imports these and layers the fast-opencv-
// dependent steps (approxPolyDP sweep, minAreaRect, warpPerspective) on top.
export type Point = [number, number];

function argMin(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] < values[best]) best = i;
  return best;
}

function argMax(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

/**
 * Port of order_corners(): order 4 points as TL, TR, BR, BL via the
 * sum/diff trick (TL = min x+y, BR = max x+y, TR = max x-y, BL = min x-y).
 */
export function orderCorners(pts: Point[]): [Point, Point, Point, Point] {
  if (pts.length !== 4) {
    throw new Error(`orderCorners: expected exactly 4 points, got ${pts.length}`);
  }
  const sums = pts.map(([x, y]) => x + y);
  const diffs = pts.map(([x, y]) => x - y);
  return [pts[argMin(sums)], pts[argMax(diffs)], pts[argMax(sums)], pts[argMin(diffs)]];
}

export function dist(a: Point, b: Point): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

// ponytail: cv2.pointPolygonTest has no fast-opencv binding (checked
// Shape.ts — findContours/convexHull/approxPolyDP/minAreaRect are there,
// pointPolygonTest is not). Since we already have contour points as plain JS
// arrays (from PointVectorOfVectors, the only findContours output container
// that exposes coordinates rather than opaque Mat handles), a standard
// ray-casting test is a direct, dependency-free substitute.
export function pointInPolygon([x, y]: Point, poly: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    const crosses = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (crosses) inside = !inside;
  }
  return inside;
}

// ponytail: fast-opencv's convexHull(points: Mat, hull: Mat) round-trips
// through opaque Mat handles with no documented way to read the resulting
// hull's point coordinates back into JS (Mat's toJSValue is for image data/
// base64, not point lists — only Point/PointVector/PointVectorOfVectors
// expose {x,y} arrays). approxPolyDP and arcLength DO accept PointVector
// input directly and are used via fast-opencv in scan.ts (per the brief),
// but the hull itself is a well-known, easily-verified algorithm (Andrew's
// monotone chain), so it's computed here in plain JS instead of fighting the
// Mat/PointVector marshalling for zero behavioral difference.
export function convexHull(points: Point[]): Point[] {
  const pts = [...points].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o: Point, a: Point, b: Point) =>
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);

  const lower: Point[] = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }
  const upper: Point[] = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

/** Port of cv2.boxPoints(minAreaRect) — matches OpenCV's RotatedRect::points()
 * formula exactly (angle in degrees). */
export function boxPointsFromRotatedRect(rr: {
  centerX: number;
  centerY: number;
  width: number;
  height: number;
  angle: number;
}): [Point, Point, Point, Point] {
  const angle = (rr.angle * Math.PI) / 180;
  const b = Math.cos(angle) * 0.5;
  const a = Math.sin(angle) * 0.5;
  const { centerX: cx, centerY: cy, width: w, height: h } = rr;
  const p0: Point = [cx - a * h - b * w, cy + b * h - a * w];
  const p1: Point = [cx + a * h - b * w, cy - b * h - a * w];
  const p2: Point = [2 * cx - p0[0], 2 * cy - p0[1]];
  const p3: Point = [2 * cx - p1[0], 2 * cy - p1[1]];
  return [p0, p1, p2, p3];
}
