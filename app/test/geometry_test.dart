import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:missilenotes/pipeline/geometry.dart';
import 'package:missilenotes/pipeline/marker.dart';
import 'package:missilenotes/pipeline/types.dart';
import 'package:opencv_dart/opencv.dart' as cv;

void main() {
  test('orderCorners returns TL, TR, BR, BL', () {
    final points = [
      const Pt(100, 80),
      const Pt(10, 10),
      const Pt(10, 80),
      const Pt(100, 10),
    ];
    final ordered = orderCorners(points);
    expect((ordered[0].x, ordered[0].y), (10, 10));
    expect((ordered[1].x, ordered[1].y), (100, 10));
    expect((ordered[2].x, ordered[2].y), (100, 80));
    expect((ordered[3].x, ordered[3].y), (10, 80));
  });

  test('warp keeps the quad aspect ratio', () {
    final image = cv.Mat.create(
      rows: 400,
      cols: 300,
      r: 240,
      g: 240,
      b: 240,
      type: cv.MatType.CV_8UC3,
    );
    final quad = [
      const Pt(10, 20),
      const Pt(310, 20),
      const Pt(310, 420),
      const Pt(10, 420),
    ];
    final flat = warp(image, quad);
    try {
      expect(flat.cols, 300);
      expect(flat.rows, 400);
      expect(flat.channels, 3);
    } finally {
      flat.dispose();
      image.dispose();
    }
  });

  test('cleanMask keeps the island around the marker', () {
    const height = 200;
    const width = 200;
    final pixels = Uint8List(height * width);
    for (var row = 20; row < 180; row++) {
      for (var col = 20; col < 80; col++) {
        pixels[row * width + col] = 1;
      }
      for (var col = 100; col < 190; col++) {
        pixels[row * width + col] = 1;
      }
    }
    final mask =
        cv.Mat.fromList(height, width, cv.MatType.CV_8UC1, pixels);

    final largest = cleanMask(mask);
    final withMarker = cleanMask(
      mask,
      const Marker([
        Pt(40, 90),
        Pt(60, 90),
        Pt(60, 110),
        Pt(40, 110),
      ], 7),
    );
    try {
      expect(_count(largest), 14400);
      expect(_count(withMarker), 9600);
    } finally {
      largest.dispose();
      withMarker.dispose();
      mask.dispose();
    }
  });

  test('pageQuad explains an empty mask', () {
    final gray = cv.Mat.fromScalar(
      200,
      240,
      cv.MatType.CV_8UC1,
      cv.Scalar(255),
    );
    final mask = cv.Mat.zeros(200, 240, cv.MatType.CV_8UC1);
    try {
      final result = pageQuad(mask, null, gray);
      expect(result.quad, isNull);
      expect(result.confidence, 0.0);
      expect(result.reason, contains('no page region'));
    } finally {
      gray.dispose();
      mask.dispose();
    }
  });

  test('pageQuad explains a missing angle source', () {
    final gray = cv.Mat.fromScalar(
      300,
      300,
      cv.MatType.CV_8UC1,
      cv.Scalar(255),
    );
    final mask = cv.Mat.zeros(300, 300, cv.MatType.CV_8UC1);
    final pixels = mask.data;
    for (var row = 50; row < 250; row++) {
      for (var col = 60; col < 240; col++) {
        pixels[row * 300 + col] = 1;
      }
    }
    try {
      final result = pageQuad(mask, null, gray);
      expect(result.quad, isNull);
      expect(result.reason, contains('angle'));
    } finally {
      gray.dispose();
      mask.dispose();
    }
  });

  test('dominantAngle picks the biggest parallel cluster', () {
    final segments = <List<double>>[
      [0, 0, 100, 17.6],
      [0, 20, 100, 37.6],
      [0, 40, 100, 57.6],
      [0, 60, 100, 77.6],
      [0, 0, 60, 71.5],
    ];
    final (angle, kept) = dominantAngle(segments);
    expect(angle, closeTo(10.0, 0.5));
    expect(kept, 4);
  });
}

int _count(cv.Mat mask) {
  var count = 0;
  for (final value in mask.data) {
    if (value != 0) count++;
  }
  return count;
}
