// Marker detection, ported from segment_pages.py's use of cv2.aruco
// (DICT_6X6_50) — but react-native-fast-opencv has no ArUco bindings (see
// task-2.1-report.md), so this uses js-aruco2 (pure JS) with a custom
// DICT_6X6_50 codebook generated from OpenCV ground truth (dict6x6_50.ts).
// js-aruco2 checks all 4 rotations of the detected bit pattern internally
// against the codebook, so a single canonical code per marker is enough.
import { AR } from 'js-aruco2';

import { DICT_6X6_50, DICT_6X6_50_NAME } from './dict6x6_50';

// ponytail: js-aruco2 has no exported TS types; the shapes below are the
// subset of AR.Detector/AR.Marker's runtime shape (see
// node_modules/js-aruco2/src/aruco.js) that we actually read.
type RawMarker = {
  id: number;
  hammingDistance: number;
  corners: { x: number; y: number }[];
};

if (!AR.DICTIONARIES[DICT_6X6_50_NAME]) {
  AR.DICTIONARIES[DICT_6X6_50_NAME] = DICT_6X6_50;
}

// One detector instance, reused across calls — construction parses/validates
// the whole codebook (50 codes), no need to redo that per photo.
const detector = new AR.Detector({ dictionaryName: DICT_6X6_50_NAME });

export type DetectedMarker = {
  /** TL, TR, BR, BL in the marker's own frame (matches cv2.aruco order). */
  corners: [number, number][];
  id: number;
};

/**
 * Detect the DICT_6X6_50 marker in an RGBA pixel buffer.
 * Mirrors segment_pages.py's `_ARUCO.detectMarkers(...)` call sites: returns
 * null (no marker) rather than throwing, since every caller in the Python
 * reference treats "no marker" as a valid, handled case.
 */
export function detectMarker(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number
): DetectedMarker | null {
  const markers = detector.detect({
    width,
    height,
    data: rgba,
  }) as RawMarker[];

  if (markers.length === 0) return null;

  // Multiple quad-shaped candidates in a photo can decode to *something* in
  // the codebook at high Hamming distance (noise) — see aruco.test.ts's
  // IMG_4905 case (a spurious id=43 at distance 10 alongside the real id=0
  // at distance 0). Prefer the lowest-distance (most confident) match, same
  // as js-aruco2's own internal per-candidate tie-break in getMarker().
  let best = markers[0];
  for (const m of markers) {
    if (m.hammingDistance < best.hammingDistance) best = m;
  }

  return {
    id: best.id,
    corners: best.corners.map((c) => [c.x, c.y] as [number, number]),
  };
}

/**
 * Port of segment_pages.py's upright(): the marker is printed upside-down
 * relative to the upright page, so 'page up' is the marker's down direction.
 * Returns the snapped rotation (degrees, clockwise) needed to bring the page
 * upright, given the marker's own 4 corners (TL,TR,BR,BL).
 *
 *   ux, uy = c[0] - c[3]       # marker-up = TL - BL, in y-down pixel coords
 *   theta = round((90 - degrees(atan2(uy, -ux))) / 90) * 90 % 360
 */
export function uprightAngle(markerCorners: [number, number][]): 0 | 90 | 180 | 270 {
  const [tlx, tly] = markerCorners[0];
  const [blx, bly] = markerCorners[3];
  const ux = tlx - blx;
  const uy = tly - bly;
  const theta = (Math.round((90 - (Math.atan2(uy, -ux) * 180) / Math.PI) / 90) * 90 + 360) % 360;
  return theta as 0 | 90 | 180 | 270;
}
