// Port of experiments/segment_pages.py + experiments/segment_pages_horizontal.py
// (the HORIZONTAL model's frame handling: upright -> rotate 90 CW into the
// model frame -> infer -> rotate mask 90 CCW back to upright -> keep_marker_
// island -> page_quad -> warp -> grayscale). Native pixel ops go through
// react-native-fast-opencv, segmentation through react-native-fast-tflite,
// ArUco detection through aruco.ts (js-aruco2, since fast-opencv has none).
//
// Two kinds of functions live here:
//  - Pure geometry (orderCorners, pointInPolygon, convexHull, boxPoints,
//    approxQuadFromHull/pageQuadFromContourPoints): operate on plain JS point
//    arrays. orderCorners has no native calls at all and is unit-tested in
//    Node (scan.test.ts). The others call into fast-opencv for
//    approxPolyDP/arcLength/minAreaRect (per the brief) and so need the
//    native module, but the hull/point-in-polygon math itself is plain JS —
//    see the ponytail notes below for why.
//  - The native pipeline (maskFor, keepMarkerIsland, pageQuad, warpQuad,
//    processScan): Mat in, Mat out, only runs on-device.
import type { Mat } from 'react-native-fast-opencv';
import {
  BorderTypes,
  ColorConversionCodes,
  ContourApproximationModes,
  DataTypes,
  DecompTypes,
  InterpolationFlags,
  LineTypes,
  MorphShapes,
  MorphTypes,
  ObjectType,
  OpenCV,
  RetrievalModes,
  RotateFlags,
  ThresholdTypes,
} from 'react-native-fast-opencv';
import type { TensorflowModel } from 'react-native-fast-tflite';

import { detectMarker, uprightAngle, type DetectedMarker } from './aruco';
import {
  boxPointsFromRotatedRect,
  convexHull,
  dist,
  orderCorners,
  pointInPolygon,
  type Point,
} from './geometry';

// Re-exported for callers/tests that only need the geometry (see geometry.ts
// for the actual implementations and pure-Node unit tests in scan.test.ts).
export { orderCorners };
export type { Point };

/** Port of page_quad()'s hull->approxPolyDP sweep, given an already-built
 * convex hull. Split out from pageQuadFromContourPoints so the sweep (which
 * needs fast-opencv) is isolated from the pure-JS hull step above. */
function approxQuadFromHull(hull: Point[]): Point[] | null {
  const point2fs = hull.map(([x, y]) => OpenCV.createObject(ObjectType.Point2f, x, y));
  const hullPV = OpenCV.createObject(ObjectType.Point2fVector, point2fs);
  const peri = (OpenCV.invoke('arcLength', hullPV, true) as { value: number }).value;

  const STEPS = 30;
  for (let i = 0; i < STEPS; i++) {
    const f = 0.01 + ((0.12 - 0.01) * i) / (STEPS - 1); // np.linspace(0.01, 0.12, 30)
    const approx = OpenCV.createObject(ObjectType.PointVector);
    OpenCV.invoke('approxPolyDP', hullPV, approx, f * peri, true);
    const { array } = OpenCV.toJSValue(approx);
    if (array.length === 4) return array.map((p): Point => [p.x, p.y]);
  }

  // Fallback: tightest enclosing rotated rect. minAreaRect needs points as a
  // Mat (not PointVector, per fast-opencv's Shape.ts signature).
  const flat = hull.flatMap(([x, y]) => [x, y]);
  const mat = OpenCV.createObject(ObjectType.Mat, hull.length, 1, DataTypes.CV_32FC2, flat);
  const rrHandle = OpenCV.invoke('minAreaRect', mat);
  const rr = OpenCV.toJSValue(rrHandle);
  return boxPointsFromRotatedRect(rr);
}

/**
 * Port of page_quad(): convex-hull the mask's contour points together with
 * the marker's own 4 corners (a guaranteed page-corner anchor), then
 * approxPolyDP-sweep down to a quad (fallback: tightest enclosing rect).
 */
export function pageQuadFromContourPoints(
  points: Point[],
  markerCorners: Point[] | null
): Point[] | null {
  if (points.length === 0) return null;
  const allPoints = markerCorners ? points.concat(markerCorners) : points;
  const hull = convexHull(allPoints);
  if (hull.length < 3) return null;
  return approxQuadFromHull(hull);
}

/**
 * Port of warp(): perspective-warp the quad region of `mat` to a flat,
 * front-on rectangle sized to the quad's own edge lengths.
 */
export function warpQuad(mat: Mat, quad: Point[]): Mat {
  const [tl, tr, br, bl] = orderCorners(quad);
  const w = Math.max(1, Math.round(Math.max(dist(br, bl), dist(tr, tl))));
  const h = Math.max(1, Math.round(Math.max(dist(tr, br), dist(tl, bl))));

  const srcPV = OpenCV.createObject(
    ObjectType.Point2fVector,
    [tl, tr, br, bl].map(([x, y]) => OpenCV.createObject(ObjectType.Point2f, x, y))
  );
  const dstPV = OpenCV.createObject(
    ObjectType.Point2fVector,
    ([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]] as Point[]).map(([x, y]) =>
      OpenCV.createObject(ObjectType.Point2f, x, y)
    )
  );
  const M = OpenCV.invoke('getPerspectiveTransform', srcPV, dstPV, DecompTypes.DECOMP_LU);

  const dst = OpenCV.createObject(ObjectType.Mat, h, w, DataTypes.CV_8UC3);
  const size = OpenCV.createObject(ObjectType.Size, w, h);
  const borderValue = OpenCV.createObject(ObjectType.Scalar, 0, 0, 0);
  OpenCV.invoke(
    'warpPerspective',
    mat,
    dst,
    M,
    size,
    InterpolationFlags.INTER_LINEAR,
    BorderTypes.BORDER_CONSTANT,
    borderValue
  );
  return dst;
}

// ============================================================================
// Native pipeline — on-device only.
// ============================================================================

const MODEL_SIZE = 512; // matches segment_pages.py's SIZE
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

function findContoursJS(mask: Mat): Point[][] {
  const contours = OpenCV.createObject(ObjectType.PointVectorOfVectors);
  OpenCV.invoke(
    'findContours',
    mask,
    contours,
    RetrievalModes.RETR_EXTERNAL,
    ContourApproximationModes.CHAIN_APPROX_SIMPLE
  );
  const { array } = OpenCV.toJSValue(contours);
  return array.map((pts) => pts.map((p): Point => [p.x, p.y]));
}

function detectMarkerOnMat(mat: Mat): DetectedMarker | null {
  const rgba = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_8U);
  OpenCV.invoke('cvtColor', mat, rgba, ColorConversionCodes.COLOR_BGR2RGBA);
  const { buffer, cols, rows } = OpenCV.matToBuffer(rgba, 'uint8');
  return detectMarker(buffer, cols, rows);
}

/** img.rotate(theta, expand=True) where theta is uprightAngle()'s CCW degrees. */
function rotateUprightCCW(mat: Mat, degrees: 0 | 90 | 180 | 270): Mat {
  if (degrees === 0) return mat;
  const dst = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_8U);
  const flag =
    degrees === 90
      ? RotateFlags.ROTATE_90_COUNTERCLOCKWISE
      : degrees === 180
        ? RotateFlags.ROTATE_180
        : RotateFlags.ROTATE_90_CLOCKWISE; // 270 CCW == 90 CW
  OpenCV.invoke('rotate', mat, dst, flag);
  return dst;
}

function rotate90CW(mat: Mat): Mat {
  const dst = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_8U);
  OpenCV.invoke('rotate', mat, dst, RotateFlags.ROTATE_90_CLOCKWISE);
  return dst;
}

function rotate90CCW(mat: Mat): Mat {
  const dst = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_8U);
  OpenCV.invoke('rotate', mat, dst, RotateFlags.ROTATE_90_COUNTERCLOCKWISE);
  return dst;
}

/**
 * Port of mask_for(): resize to the model's input resolution, normalize
 * (ImageNet mean/std), run tflite inference, sigmoid, resize the probability
 * map back to `mat`'s own resolution, threshold > 124/255, then CLOSE+OPEN
 * with an odd elliptical kernel ~1.5% of the short side.
 */
async function maskFor(model: TensorflowModel, mat: Mat): Promise<Mat> {
  const { cols: W, rows: H } = OpenCV.toJSValue(mat);

  const inputShape = model.inputs[0]?.shape ?? [1, MODEL_SIZE, MODEL_SIZE, 3];
  const modelH = inputShape[1] ?? MODEL_SIZE;
  const modelW = inputShape[2] ?? MODEL_SIZE;

  const resized = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_8U);
  OpenCV.invoke(
    'resize',
    mat,
    resized,
    OpenCV.createObject(ObjectType.Size, modelW, modelH),
    0,
    0,
    InterpolationFlags.INTER_LINEAR
  );
  const rgb = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_8U);
  OpenCV.invoke('cvtColor', resized, rgb, ColorConversionCodes.COLOR_BGR2RGB);

  const { buffer } = OpenCV.matToBuffer(rgb, 'uint8');
  // ponytail/assumption: buffer is HWC (row-major, channel-last), and the
  // model's input tensor is NHWC — the tflite-standard layout, and the one
  // that lets us normalize in place with no transpose. Only verifiable
  // on-device (no device available here) — see task-2.2-report.md.
  const input = new Float32Array(buffer.length);
  for (let i = 0; i < buffer.length; i++) {
    const c = i % 3;
    input[i] = (buffer[i] / 255 - MEAN[c]) / STD[c];
  }

  const outputs = await model.run([input.buffer]);
  const raw = new Float32Array(outputs[0]);
  const outShape = model.outputs[0]?.shape ?? [1, modelH, modelW, 1];
  const outH = outShape[1] ?? modelH;
  const outW = outShape[2] ?? modelW;

  const probBytes = new Array<number>(raw.length);
  for (let i = 0; i < raw.length; i++) {
    const sigmoid = 1 / (1 + Math.exp(-raw[i]));
    probBytes[i] = Math.round(sigmoid * 255);
  }
  const probMat = OpenCV.createObject(ObjectType.Mat, outH, outW, DataTypes.CV_8U, probBytes);

  const probFull = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_8U);
  OpenCV.invoke(
    'resize',
    probMat,
    probFull,
    OpenCV.createObject(ObjectType.Size, W, H),
    0,
    0,
    InterpolationFlags.INTER_LINEAR
  );

  const mask = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_8U);
  OpenCV.invoke('threshold', probFull, mask, 124, 1, ThresholdTypes.THRESH_BINARY);

  const k = Math.max(3, Math.trunc(0.015 * Math.min(W, H)) | 1); // odd, >= 3
  const kernel = OpenCV.invoke(
    'getStructuringElement',
    MorphShapes.MORPH_ELLIPSE,
    OpenCV.createObject(ObjectType.Size, k, k)
  );
  OpenCV.invoke('morphologyEx', mask, mask, MorphTypes.MORPH_CLOSE, kernel);
  OpenCV.invoke('morphologyEx', mask, mask, MorphTypes.MORPH_OPEN, kernel);

  return mask;
}

/**
 * Port of keep_marker_island(): keep only the mask island the marker sits
 * inside, filled solid. `mask` and the marker's corners must share a frame.
 *
 * ponytail deviation: the Python falls back to "keep the largest island"
 * when no marker is found. That branch is unreachable here — processScan
 * already bails out (returns the original photo) before calling this, the
 * moment marker detection fails. See processScan's top-level try/catch.
 */
function keepMarkerIsland(mask: Mat, markerCorners: Point[], width: number, height: number): Mat {
  const contours = findContoursJS(mask);
  if (contours.length === 0) return mask;

  const cx = markerCorners.reduce((s, [x]) => s + x, 0) / markerCorners.length;
  const cy = markerCorners.reduce((s, [, y]) => s + y, 0) / markerCorners.length;
  const chosen = contours.find((c) => pointInPolygon([cx, cy], c)) ?? contours[0];

  const filled = OpenCV.createObject(
    ObjectType.Mat,
    height,
    width,
    DataTypes.CV_8U,
    new Array(width * height).fill(0)
  );
  const pv = OpenCV.createObject(
    ObjectType.Point2fVector,
    chosen.map(([x, y]) => OpenCV.createObject(ObjectType.Point2f, x, y))
  );
  // ponytail: cv2.drawContours(..., FILLED) correctly fills concave shapes;
  // fast-opencv only exposes fillConvexPoly/fillPoly (MatVector-based, see
  // Drawing.ts). After CLOSE+OPEN the island is a clean, near-convex page
  // blob, so fillConvexPoly is a reasonable, documented approximation.
  OpenCV.invoke(
    'fillConvexPoly',
    filled,
    pv,
    OpenCV.createObject(ObjectType.Scalar, 1),
    LineTypes.FILLED
  );
  return filled;
}

function pageQuad(mask: Mat, markerCorners: Point[]): Point[] | null {
  const contours = findContoursJS(mask);
  if (contours.length === 0) return null;
  return pageQuadFromContourPoints(contours.flat(), markerCorners);
}

let cachedModel: TensorflowModel | null = null;
async function getModel(): Promise<TensorflowModel> {
  if (!cachedModel) {
    const { loadTensorflowModel } = await import('react-native-fast-tflite');
    cachedModel = await loadTensorflowModel(
      // eslint-disable-next-line @typescript-eslint/no-require-imports
      require('../../assets/model/fairscan-segmentation-model.tflite'),
      []
    );
  }
  return cachedModel;
}

/**
 * Runs everything through page_quad() (upright -> model-frame mask -> island
 * -> quad) but stops short of warping. `quad`'s coordinates are in the
 * *upright* frame — the same frame experiments/export_goldens.py's
 * quad.json goldens are recorded in — so this is what the /parity screen
 * calls to diff against the goldens without needing to decode the warped
 * output back into quad coordinates.
 */
async function computeUprightQuad(photoUri: string): Promise<{ upright: Mat; quad: Point[] }> {
  const FileSystem = await import('expo-file-system/legacy');
  const base64 = await FileSystem.readAsStringAsync(photoUri, { encoding: 'base64' });
  const src = OpenCV.base64ToMat(base64);

  // upright(): detect the marker in the as-shot frame, snap to the nearest
  // 90 deg so the page reads upright.
  const preMarker = detectMarkerOnMat(src);
  if (!preMarker) throw new Error('no marker detected (pre-rotation frame)');
  const angle = uprightAngle(preMarker.corners);
  const upright = rotateUprightCCW(src, angle);

  // Re-detect on the upright frame — keep_marker_island/page_quad both
  // re-detect on `img` in the Python reference; we do it once here and
  // reuse the corners for both (same deterministic detector, same image).
  const postMarker = detectMarkerOnMat(upright);
  if (!postMarker) throw new Error('no marker detected (upright frame)');

  const horiz = rotate90CW(upright); // -> model's expected frame
  const model = await getModel();
  const maskHoriz = await maskFor(model, horiz);
  const maskUpright = rotate90CCW(maskHoriz); // back to upright frame

  const { cols: uW, rows: uH } = OpenCV.toJSValue(upright);
  const island = keepMarkerIsland(maskUpright, postMarker.corners, uW, uH);
  const quad = pageQuad(island, postMarker.corners);
  if (!quad) throw new Error('no page quad found');

  return { upright, quad };
}

async function runPipeline(photoUri: string): Promise<string> {
  const FileSystem = await import('expo-file-system/legacy');
  const { upright, quad } = await computeUprightQuad(photoUri);

  const warped = warpQuad(upright, quad);
  const gray = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_8U);
  OpenCV.invoke('cvtColor', warped, gray, ColorConversionCodes.COLOR_BGR2GRAY);

  const outPath = `${FileSystem.cacheDirectory}scan-${Date.now()}-${Math.round(
    Math.random() * 1e6
  )}.jpg`;
  OpenCV.saveMatToFile(gray, outPath, 'jpeg', 0.92);
  return outPath;
}

/**
 * Dev/parity-only entry point: returns the detected page quad's 4 corners in
 * the upright frame (same frame as experiments/export_goldens.py's
 * `<name>.quad.json`), or null on any failure. Used by the /parity screen to
 * diff against the golden quads — see task-2.2-report.md for exact steps.
 */
export async function detectPageQuadForParity(photoUri: string): Promise<Point[] | null> {
  try {
    const { quad } = await computeUprightQuad(photoUri);
    return quad;
  } catch (err) {
    console.warn('[pipeline] detectPageQuadForParity failed:', err);
    return null;
  } finally {
    OpenCV.clearBuffers();
  }
}

/**
 * Run the full horizontal-model scan pipeline on a captured photo and return
 * a file:// URI of the flattened, grayscale page. Never throws — any failure
 * (no marker, no quad, inference error, decode error) falls back to the
 * original photo so a scan is never blocked on CV/ML failure.
 */
export async function processScan(photoUri: string): Promise<string> {
  try {
    return await runPipeline(photoUri);
  } catch (err) {
    console.warn('[pipeline] processScan failed, keeping original photo:', err);
    return photoUri;
  } finally {
    OpenCV.clearBuffers();
  }
}
