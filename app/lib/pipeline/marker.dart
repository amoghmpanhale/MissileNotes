import 'dart:math' as math;

import 'package:opencv_dart/opencv.dart' as cv;

import 'types.dart';

/// A detected ArUco marker.  Corners are TL/TR/BR/BL in the marker's own
/// canonical frame, which is not necessarily the photo's top-left.
class Marker {
  const Marker(this.corners, this.id);

  final List<Pt> corners;
  final int id;
}

final _dictionary =
    cv.ArucoDictionary.predefined(cv.PredefinedDictionaryType.DICT_6X6_50);
final _detectorParameters = cv.ArucoDetectorParameters.empty();
final _detector = cv.ArucoDetector.create(_dictionary, _detectorParameters);

/// Find the marker in a grayscale image, once.
///
/// Returns the lowest-id marker when several are visible, or null when the
/// photo has no marker.  Callers carry the corners through [rotatePoints]
/// rather than re-detecting after a rotation.
Marker? detectMarker(cv.Mat gray) {
  final (corners, ids, rejected) = _detector.detectMarkers(gray);
  try {
    if (ids.isEmpty) return null;

    var best = 0;
    for (var i = 1; i < ids.length; i++) {
      if (ids[i] < ids[best]) best = i;
    }
    final marker = corners[best];
    final points = <Pt>[
      for (var i = 0; i < marker.length; i++) Pt(marker[i].x, marker[i].y),
    ];
    return Marker(points, ids[best]);
  } finally {
    corners.dispose();
    ids.dispose();
    rejected.dispose();
  }
}

/// Degrees CCW (0/90/180/270) that make the page upright.
///
/// The marker is printed upside-down, so page-up is marker-down: rotate the
/// marker's up vector to point down (+y) and quantise to a quadrant.
int uprightRotation(Marker marker) {
  final c = marker.corners;
  final ux = c[0].x - c[3].x;
  final uy = c[0].y - c[3].y;
  final angle = math.atan2(uy, ux) * 180.0 / math.pi;
  return (roundHalfEven((angle - 90.0) / 90.0) * 90) % 360;
}

/// Rotate an image CCW by [deg] with an enlarged canvas.
///
/// Multiples of 90 are exact array rotations, so no pixels are resampled;
/// other angles use an affine warp around the image centre.
cv.Mat rotateImage(cv.Mat image, double deg) {
  final normalized = deg % 360.0;
  final rounded = normalized.round();
  if ((normalized - rounded).abs() < 1e-9 && rounded % 90 == 0) {
    final k = (rounded ~/ 90) % 4;
    return switch (k) {
      0 => image.clone(),
      1 => image.rotate(cv.ROTATE_90_COUNTERCLOCKWISE),
      2 => image.rotate(cv.ROTATE_180),
      _ => image.rotate(cv.ROTATE_90_CLOCKWISE),
    };
  }

  final height = image.rows;
  final width = image.cols;
  final matrix = _rotationMatrix(normalized, width, height);
  final newWidth = matrix.newWidth;
  final newHeight = matrix.newHeight;
  final transform = cv.Mat.fromList(2, 3, cv.MatType.CV_64FC1, matrix.values);
  try {
    return cv.warpAffine(
      image,
      transform,
      (newWidth, newHeight),
      flags: cv.INTER_LINEAR,
    );
  } finally {
    transform.dispose();
  }
}

/// Rotate (N, 2) points by [deg] CCW into the same enlarged frame as
/// [rotateImage].  [width]/[height] are the original image dimensions.
List<Pt> rotatePoints(
  List<Pt> points,
  double deg,
  int width,
  int height,
) {
  final k = deg.round() % 360;
  if ((deg - deg.round()).abs() < 1e-9 && k % 90 == 0) {
    return [
      for (final p in points)
        switch (k) {
          0 => Pt(p.x, p.y),
          90 => Pt(p.y, (width - 1) - p.x),
          180 => Pt((width - 1) - p.x, (height - 1) - p.y),
          _ => Pt((height - 1) - p.y, p.x),
        },
    ];
  }

  final m = _rotationMatrix(deg % 360.0, width, height).values;
  return [
    for (final p in points)
      Pt(m[0] * p.x + m[1] * p.y + m[2], m[3] * p.x + m[4] * p.y + m[5]),
  ];
}

class _Affine {
  const _Affine(this.values, this.newWidth, this.newHeight);

  final List<double> values;
  final int newWidth;
  final int newHeight;
}

/// OpenCV's expanded `getRotationMatrix2D` matrix, built from its formula so
/// the result matches the Python helper exactly.
_Affine _rotationMatrix(double deg, int width, int height) {
  final radians = deg * math.pi / 180.0;
  final alpha = math.cos(radians);
  final beta = math.sin(radians);
  final cx = width / 2.0;
  final cy = height / 2.0;

  final cos = alpha.abs();
  final sin = beta.abs();
  final newWidth = roundHalfEven(height * sin + width * cos);
  final newHeight = roundHalfEven(height * cos + width * sin);

  return _Affine(
    [
      alpha,
      beta,
      (1.0 - alpha) * cx - beta * cy + newWidth / 2.0 - cx,
      -beta,
      alpha,
      beta * cx + (1.0 - alpha) * cy + newHeight / 2.0 - cy,
    ],
    newWidth,
    newHeight,
  );
}
