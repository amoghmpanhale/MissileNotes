import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:missilenotes/pipeline/enhance.dart';
import 'package:opencv_dart/opencv.dart' as cv;

void main() {
  test('binarize keeps ink, drops specks and paper stays white', () {
    final pixels = Uint8List(60 * 60)..fillRange(0, 60 * 60, 200);
    for (var row = 20; row < 40; row++) {
      for (var col = 30; col < 33; col++) {
        pixels[row * 60 + col] = 30;
      }
    }
    pixels[0] = 20;

    final gray = cv.Mat.fromList(60, 60, cv.MatType.CV_8UC1, pixels);
    final out = binarize(gray, block: 11, c: 5, minArea: 6);
    try {
      expect(out.rows, 60);
      expect(out.cols, 60);
      expect(out.type.value, cv.MatType.CV_8UC1.value);
      expect(out.atU8(30, i1: 31), 0);
      expect(out.atU8(0, i1: 0), 255);
      expect(out.atU8(5, i1: 55), 255);

      final values = out.data.toSet();
      expect(values.difference({0, 255}), isEmpty);
    } finally {
      out.dispose();
      gray.dispose();
    }
  });
}
