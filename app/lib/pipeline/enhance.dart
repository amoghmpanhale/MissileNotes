import 'dart:typed_data';

import 'package:opencv_dart/opencv.dart' as cv;

/// Turn a flat grayscale scan into clean black ink on white paper.
///
/// A local Gaussian threshold handles uneven lighting; the connected-component
/// pass removes specks smaller than a pen stroke.  Returns 0 for ink and 255
/// for paper.
cv.Mat binarize(
  cv.Mat gray, {
  int block = 41,
  int c = 15,
  int minArea = 6,
}) {
  final ink = cv.adaptiveThreshold(
    gray,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY_INV,
    block | 1,
    c.toDouble(),
  );

  final labels = cv.Mat.empty();
  final stats = cv.Mat.empty();
  final centroids = cv.Mat.empty();
  try {
    final count = cv.connectedComponentsWithStats(
      ink,
      labels,
      stats,
      centroids,
      8,
      cv.MatType.CV_32S,
      cv.CCL_DEFAULT,
    );

    final statsInts = stats.data.buffer
        .asInt32List(stats.data.offsetInBytes, stats.total);
    final keep = List<bool>.filled(count, false);
    for (var label = 1; label < count; label++) {
      // Connected-component stats columns are left, top, width, height, area.
      keep[label] = statsInts[label * 5 + 4] >= minArea;
    }

    final labelInts = labels.data.buffer
        .asInt32List(labels.data.offsetInBytes, labels.total);
    final output = Uint8List(labelInts.length);
    for (var i = 0; i < labelInts.length; i++) {
      output[i] = keep[labelInts[i]] ? 0 : 255;
    }
    return cv.Mat.fromList(gray.rows, gray.cols, cv.MatType.CV_8UC1, output);
  } finally {
    ink.dispose();
    labels.dispose();
    stats.dispose();
    centroids.dispose();
  }
}
