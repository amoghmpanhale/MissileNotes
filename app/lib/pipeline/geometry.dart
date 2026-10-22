import 'dart:math' as math;

import 'package:opencv_dart/opencv.dart' as cv;

import 'marker.dart';
import 'types.dart';

const double angleTolerance = 6.0;
// Degrees: how far ruled lines may move the marker/image-axes page angle.
const double ruleAgreeTolerance = 20.0;
const double minIou = 0.5;

/// Outcome of [pageQuad].  `quad` is TL/TR/BR/BL in the page's own axes, or
/// null.  `confidence` is the IoU between the quad and the cleaned page mask.
class QuadResult {
  const QuadResult(this.quad, this.confidence, this.reason);

  final List<Pt>? quad;
  final double confidence;
  final String reason;
}

/// Keep one page island and fill it solid.
///
/// With a marker, keep the island containing its centre; otherwise keep the
/// largest island.  Interior holes are filled through the outer contour.
cv.Mat cleanMask(cv.Mat mask, [Marker? marker]) {
  final (contours, hierarchy) = cv.findContours(
    mask,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE,
  );
  hierarchy.dispose();
  try {
    if (contours.isEmpty) return mask.clone();

    int? chosen;
    if (marker != null) {
      final centre = _mean(marker.corners);
      var bestArea = double.negativeInfinity;
      for (var i = 0; i < contours.length; i++) {
        final contour = contours[i];
        if (cv.pointPolygonTest(
              contour,
              cv.Point2f(centre.x, centre.y),
              false,
            ) >=
            0) {
          final area = cv.contourArea(contour);
          if (area > bestArea) {
            bestArea = area;
            chosen = i;
          }
        }
      }
    }
    if (chosen == null) {
      var bestArea = double.negativeInfinity;
      for (var i = 0; i < contours.length; i++) {
        final area = cv.contourArea(contours[i]);
        if (area > bestArea) {
          bestArea = area;
          chosen = i;
        }
      }
    }

    final filled = cv.Mat.zeros(mask.rows, mask.cols, cv.MatType.CV_8UC1);
    final contour = contours[chosen!];
    final polygons = cv.VecVecPoint.generate(1, (_) => contour, dispose: false);
    try {
      cv.drawContours(filled, polygons, -1, cv.Scalar(1), thickness: cv.FILLED);
    } finally {
      polygons.dispose();
    }
    return filled;
  } finally {
    contours.dispose();
  }
}

/// Fill a quad into a solid 0/1 mask of shape HxW.
cv.Mat quadMask(List<Pt> quad, int height, int width) {
  final mask = cv.Mat.zeros(height, width, cv.MatType.CV_8UC1);
  final points = _vecPoint(quad);
  final polygons = cv.VecVecPoint.generate(1, (_) => points, dispose: false);
  try {
    cv.fillPoly(mask, polygons, cv.Scalar(1));
  } finally {
    polygons.dispose();
    points.dispose();
  }
  return mask;
}

/// Hough segments whose pixels lie on the page region, as (N, 4) tuples.
List<List<double>> ruledLines(cv.Mat gray, cv.Mat region) {
  final edges = cv.canny(gray, 50, 150);
  try {
    final edgeData = edges.data;
    final regionData = region.data;
    for (var i = 0; i < edgeData.length; i++) {
      if (regionData[i] == 0) edgeData[i] = 0;
    }

    final minLineLength = math.max(24, gray.cols ~/ 6).toDouble();
    final lines = cv.HoughLinesP(
      edges,
      1,
      math.pi / 180.0,
      80,
      minLineLength: minLineLength,
      maxLineGap: 20,
    );
    try {
      if (lines.rows == 0) return const [];
      final ints = lines.data.buffer.asInt32List(
        lines.data.offsetInBytes,
        lines.rows * 4,
      );
      return [
        for (var i = 0; i < lines.rows; i++)
          [
            ints[i * 4].toDouble(),
            ints[i * 4 + 1].toDouble(),
            ints[i * 4 + 2].toDouble(),
            ints[i * 4 + 3].toDouble(),
          ],
      ];
    } finally {
      lines.dispose();
    }
  } finally {
    edges.dispose();
  }
}

/// Median angle (degrees, folded to -90..90) of the biggest parallel cluster.
/// Returns the angle and how many segments belong to the cluster.
(double, int) dominantAngle(List<List<double>> segments) {
  final angles = [
    for (final s in segments)
      _foldDegrees(math.atan2(s[3] - s[1], s[2] - s[0]) * 180.0 / math.pi),
  ];

  final histogram = List<int>.filled(36, 0);
  for (final angle in angles) {
    final bin = ((angle + 90.0) / 5.0).floor().clamp(0, 35);
    histogram[bin]++;
  }
  var bestBin = 0;
  for (var i = 1; i < histogram.length; i++) {
    if (histogram[i] > histogram[bestBin]) bestBin = i;
  }
  final centre = -90.0 + bestBin * 5.0 + 2.5;

  final kept = [
    for (final angle in angles)
      if ((_foldDegrees(angle - centre)).abs() < angleTolerance) angle,
  ];
  return (_median(kept), kept.length);
}

/// Order 4 points as TL, TR, BR, BL via the sum/diff trick.
List<Pt> orderCorners(List<Pt> points) {
  final sums = [for (final p in points) p.x + p.y];
  final diffs = [for (final p in points) p.x - p.y];
  return [
    points[_argMin(sums)],
    points[_argMax(diffs)],
    points[_argMax(sums)],
    points[_argMin(diffs)],
  ];
}

/// Perspective-warp the quad to a flat, front-on rectangle.  Output size
/// averages opposite edges, so the page keeps its aspect ratio.
cv.Mat warp(cv.Mat image, List<Pt> quad) {
  final ordered = orderCorners(quad);
  final tl = ordered[0];
  final tr = ordered[1];
  final br = ordered[2];
  final bl = ordered[3];

  final width = math.max(
    1,
    roundHalfEven((Pt.distance(tr, tl) + Pt.distance(br, bl)) / 2.0),
  );
  final height = math.max(
    1,
    roundHalfEven((Pt.distance(bl, tl) + Pt.distance(br, tr)) / 2.0),
  );

  final dst = [
    const Pt(0, 0),
    Pt(width - 1.0, 0),
    Pt(width - 1.0, height - 1.0),
    Pt(0, height - 1.0),
  ];
  final srcPoints = _vecPoint2f(ordered);
  final dstPoints = _vecPoint2f(dst);
  final transform = cv.getPerspectiveTransform2f(srcPoints, dstPoints);
  try {
    return cv.warpPerspective(image, transform, (width, height));
  } finally {
    transform.dispose();
    srcPoints.dispose();
    dstPoints.dispose();
  }
}

/// Recover the page quad from a model mask, marker and grayscale photo.
///
/// All inputs share one frame.  Returns a [QuadResult], never throws for an
/// unusable photo, and never silently substitutes a different region.
QuadResult pageQuad(cv.Mat mask, Marker? marker, cv.Mat gray) {
  if (mask.rows != gray.rows || mask.cols != gray.cols) {
    throw ArgumentError('mask and gray must have the same HxW shape');
  }

  final region = cleanMask(mask, marker);
  try {
    final regionData = region.data;
    var regionCount = 0;
    for (final value in regionData) {
      if (value != 0) regionCount++;
    }
    final minimum = math.max(64, (0.002 * regionData.length).truncate());
    if (regionCount < minimum) {
      return const QuadResult(null, 0.0, 'no page region found in the mask');
    }

    final contour = _pageContour(region);
    if (contour == null || contour.length < 8) {
      return const QuadResult(null, 0.0, 'page region has no usable outline');
    }

    // The marker gives a coarse page angle; ruled lines may only refine it.
    // Hough also finds phantom diagonals through tables and text grids, and a
    // 45-degree theta makes corner ordering collapse. No marker means the
    // caller already rotated the frame upright, so the image axes are the
    // reference.
    final refName = marker != null ? 'marker orientation' : 'image axes';
    var theta = marker != null ? _markerAngle(marker) : 0.0;
    var source = '$refName (no ruled lines)';
    final segments = ruledLines(gray, region);
    if (segments.length >= 3) {
      final (candidate, kept) = dominantAngle(segments);
      if (kept >= 3) {
        // Rules may be horizontal or vertical; compare modulo 90 degrees.
        final offset = (candidate - theta + 45.0) % 90.0 - 45.0;
        if (offset.abs() <= ruleAgreeTolerance) {
          theta += offset;
          source = 'ruled lines';
        } else {
          source = '$refName (ruled lines disagree)';
        }
      }
    }

    var quad = _orderByAngle(_initialQuad(contour), theta);
    final refined = _refineQuad(contour, quad);
    if (refined == null) {
      source += '; edges not refined';
    } else {
      quad = refined;
    }
    quad = _orderByAngle(quad, theta);

    final contained = _containMarker(quad, marker);
    if (contained == null) {
      return const QuadResult(null, 0.0, 'page quad cannot include the marker');
    }
    quad = contained;

    final area = _contourArea(quad).abs();
    if (!quad.every((p) => p.x.isFinite && p.y.isFinite) || area < 1.0) {
      return const QuadResult(null, 0.0, 'page quad is degenerate');
    }

    final qmask = quadMask(quad, gray.rows, gray.cols);
    try {
      final quadData = qmask.data;
      var intersection = 0;
      var union = 0;
      for (var i = 0; i < quadData.length; i++) {
        final inQuad = quadData[i] != 0;
        final inRegion = regionData[i] != 0;
        if (inQuad || inRegion) union++;
        if (inQuad && inRegion) intersection++;
      }
      final iou = union == 0 ? 0.0 : intersection / union;
      if (iou < minIou) {
        return QuadResult(
          null,
          iou,
          'quad disagrees with the page mask '
          '(IoU ${iou.toStringAsFixed(2)})',
        );
      }
      return QuadResult(quad, iou, 'ok ($source)');
    } finally {
      qmask.dispose();
    }
  } finally {
    region.dispose();
  }
}

List<Pt>? _pageContour(cv.Mat region) {
  final (contours, hierarchy) = cv.findContours(
    region,
    cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_NONE,
  );
  hierarchy.dispose();
  try {
    if (contours.isEmpty) return null;
    var best = 0;
    var bestArea = double.negativeInfinity;
    for (var i = 0; i < contours.length; i++) {
      final area = cv.contourArea(contours[i]);
      if (area > bestArea) {
        bestArea = area;
        best = i;
      }
    }
    return _fromVecPoint(contours[best]);
  } finally {
    contours.dispose();
  }
}

/// Four corner candidates from a convex hull, falling back to minAreaRect.
List<Pt> _initialQuad(List<Pt> contour) {
  final contourPoints = _vecPoint(contour);
  final hullMat = cv.Mat.empty();
  cv.VecPoint? hull;
  try {
    cv.convexHull(contourPoints, hull: hullMat);
    hull = cv.VecPoint.fromMat(hullMat);
    final perimeter = cv.arcLength(hull, true);

    for (var i = 0; i < 16; i++) {
      final fraction = 0.005 + (0.08 - 0.005) * i / 15.0;
      final approx = cv.approxPolyDP(hull, fraction * perimeter, true);
      try {
        if (approx.length == 4) return _fromVecPoint(approx);
        if (approx.length < 4) break;
      } finally {
        approx.dispose();
      }
    }

    final rect = cv.minAreaRect(hull);
    try {
      final box = cv.boxPoints(rect);
      try {
        return _fromVecPoint2f(box);
      } finally {
        box.dispose();
      }
    } finally {
      rect.dispose();
    }
  } finally {
    hull?.dispose();
    hullMat.dispose();
    contourPoints.dispose();
  }
}

/// Order corners TL/TR/BR/BL in the page's own axes, where [theta] is the
/// page's horizontal angle.
List<Pt> _orderByAngle(List<Pt> quad, double theta) {
  final radians = theta * math.pi / 180.0;
  final horizontal = Pt(math.cos(radians), math.sin(radians));
  final vertical = Pt(-math.sin(radians), math.cos(radians));

  final u = [for (final p in quad) p.dot(horizontal)];
  final w = [for (final p in quad) p.dot(vertical)];
  final sums = [for (var i = 0; i < 4; i++) u[i] + w[i]];
  final diffs = [for (var i = 0; i < 4; i++) u[i] - w[i]];
  return [
    quad[_argMin(sums)],
    quad[_argMax(diffs)],
    quad[_argMax(sums)],
    quad[_argMin(diffs)],
  ];
}

List<double> _lineCoeffs(List<double> line) {
  final vx = line[0];
  final vy = line[1];
  final x0 = line[2];
  final y0 = line[3];
  return [-vy, vx, vy * x0 - vx * y0];
}

List<double>? _lineThrough(Pt a, Pt b) {
  final dx = b.x - a.x;
  final dy = b.y - a.y;
  final norm = math.sqrt(dx * dx + dy * dy);
  if (norm < 1e-9) return null;
  return [dx / norm, dy / norm, a.x, a.y];
}

Pt? _intersectLines(List<double> first, List<double> second) {
  final a = _lineCoeffs(first);
  final b = _lineCoeffs(second);
  final cross = [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
  if (cross[2].abs() < 1e-9) return null;
  return Pt(cross[0] / cross[2], cross[1] / cross[2]);
}

/// Robustly fit each side of [quad] to [points] and re-intersect.
///
/// Points are assigned to the closest side by segment distance, ignoring
/// corner neighbourhoods so adjacent edges do not steal each other's points.
List<Pt>? _refineQuad(List<Pt> points, List<Pt> quad, {int iterations = 3}) {
  var current = quad;
  final diagonal = Pt.distance(current[2], current[0]);

  for (var iteration = 0; iteration < iterations; iteration++) {
    final assign = List<int>.filled(points.length, -1);
    final best = List<double>.filled(points.length, double.infinity);

    for (var i = 0; i < 4; i++) {
      final a = current[i];
      final b = current[(i + 1) % 4];
      final seg = b - a;
      final lengthSquared = seg.dot(seg);
      if (lengthSquared < 1e-12) return null;

      for (var p = 0; p < points.length; p++) {
        final t = (points[p] - a).dot(seg) / lengthSquared;
        if (t < 0.0 || t > 1.0) continue;
        final distance = (points[p] - (a + seg * t)).norm;
        if (distance < best[p]) {
          assign[p] = i;
          best[p] = distance;
        }
      }
    }

    final lines = <List<double>>[];
    for (var i = 0; i < 4; i++) {
      var members = [
        for (var p = 0; p < points.length; p++)
          if (assign[p] == i) p,
      ];
      if (members.length >= 6) {
        final a = current[i];
        final b = current[(i + 1) % 4];
        final seg = b - a;
        final lengthSquared = seg.dot(seg);
        members = [
          for (final p in members)
            if ((points[p] - a).dot(seg) / lengthSquared >= 0.05 &&
                (points[p] - a).dot(seg) / lengthSquared <= 0.95)
              p,
        ];
      }

      if (members.length >= 6) {
        final side = _vecPoint2f([for (final p in members) points[p]]);
        final fitted = cv.fitLine2f(side, cv.DIST_HUBER, 0, 0.01, 0.01);
        try {
          final values = fitted.data.buffer.asFloat32List(
            fitted.data.offsetInBytes,
            fitted.total,
          );
          lines.add([
            values[0].toDouble(),
            values[1].toDouble(),
            values[2].toDouble(),
            values[3].toDouble(),
          ]);
        } finally {
          fitted.dispose();
          side.dispose();
        }
      } else {
        final line = _lineThrough(current[i], current[(i + 1) % 4]);
        if (line == null) return null;
        lines.add(line);
      }
    }

    final corners = <Pt>[];
    for (var i = 0; i < 4; i++) {
      final corner = _intersectLines(lines[(i - 1) % 4], lines[i]);
      if (corner == null) return null;
      corners.add(corner);
    }
    final refined = corners;

    if (!refined.every((p) => p.x.isFinite && p.y.isFinite)) return null;
    if (!_isConvex(refined)) return null;
    final maxMove = [
      for (var i = 0; i < 4; i++) Pt.distance(refined[i], current[i]),
    ].reduce(math.max);
    if (maxMove > 0.1 * diagonal) return null;
    current = refined;
  }
  return current;
}

/// Expand the corner nearest the marker until all marker corners are inside.
List<Pt>? _containMarker(List<Pt> quad, Marker? marker) {
  if (marker == null) return quad;
  if (_allInside(quad, marker.corners)) return quad;

  final centre = _mean(quad);
  final markerCentre = _mean(marker.corners);
  var index = 0;
  var bestDistance = double.infinity;
  for (var i = 0; i < 4; i++) {
    final distance = Pt.distance(quad[i], markerCentre);
    if (distance < bestDistance) {
      bestDistance = distance;
      index = i;
    }
  }

  final outward = quad[index] - centre;
  if (outward.norm < 1e-9) return null;

  for (var step = 0; step <= 50; step++) {
    final candidate = List<Pt>.of(quad);
    candidate[index] = quad[index] + outward * (step / 50.0);
    if (!_isConvex(candidate)) break;
    if (_allInside(candidate, marker.corners)) return candidate;
  }
  return null;
}

double _markerAngle(Marker marker) {
  final c = marker.corners;
  final up = c[3] - c[0]; // page-up (marker-down)
  final horizontal = Pt(-up.y, up.x);
  final angle = math.atan2(horizontal.y, horizontal.x) * 180.0 / math.pi;
  return (angle + 90.0) % 180.0 - 90.0;
}

bool _allInside(List<Pt> quad, List<Pt> points) {
  final polygon = _vecPoint2f(quad);
  try {
    for (final point in points) {
      if (cv.pointPolygonTest2f(polygon, cv.Point2f(point.x, point.y), false) <
          0) {
        return false;
      }
    }
    return true;
  } finally {
    polygon.dispose();
  }
}

bool _isConvex(List<Pt> quad) {
  final points = _vecPoint2f(quad);
  try {
    return cv.isContourConvex2f(points);
  } finally {
    points.dispose();
  }
}

double _contourArea(List<Pt> quad) {
  final points = _vecPoint2f(quad);
  try {
    return cv.contourArea2f(points);
  } finally {
    points.dispose();
  }
}

double _foldDegrees(double angle) => (angle + 90.0) % 180.0 - 90.0;

double _median(List<double> values) {
  final sorted = List<double>.of(values)..sort();
  final middle = sorted.length ~/ 2;
  if (sorted.length.isOdd) return sorted[middle];
  return (sorted[middle - 1] + sorted[middle]) / 2.0;
}

Pt _mean(List<Pt> points) {
  var x = 0.0;
  var y = 0.0;
  for (final point in points) {
    x += point.x;
    y += point.y;
  }
  return Pt(x / points.length, y / points.length);
}

int _argMin(List<double> values) {
  var index = 0;
  for (var i = 1; i < values.length; i++) {
    if (values[i] < values[index]) index = i;
  }
  return index;
}

int _argMax(List<double> values) {
  var index = 0;
  for (var i = 1; i < values.length; i++) {
    if (values[i] > values[index]) index = i;
  }
  return index;
}

cv.VecPoint _vecPoint(List<Pt> points) => cv.VecPoint.generate(
  points.length,
  (i) => cv.Point(points[i].x.toInt(), points[i].y.toInt()),
);

cv.VecPoint2f _vecPoint2f(List<Pt> points) => cv.VecPoint2f.generate(
  points.length,
  (i) => cv.Point2f(points[i].x, points[i].y),
);

List<Pt> _fromVecPoint(cv.VecPoint points) {
  final result = <Pt>[];
  for (final point in points) {
    result.add(Pt(point.x.toDouble(), point.y.toDouble()));
  }
  return result;
}

List<Pt> _fromVecPoint2f(cv.VecPoint2f points) {
  final result = <Pt>[];
  for (final point in points) {
    result.add(Pt(point.x, point.y));
  }
  return result;
}
