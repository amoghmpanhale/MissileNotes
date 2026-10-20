import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:missilenotes/pipeline/segmenter.dart';
import 'package:missilenotes/pipeline/types.dart';
import 'package:opencv_dart/opencv.dart' as cv;

void main() {
  test('preprocess produces NHWC ImageNet-normalised floats', () {
    final rgb = cv.Mat.fromList(
      2,
      2,
      cv.MatType.CV_8UC3,
      [255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0],
    );
    final input = preprocess(rgb);
    try {
      expect(input, hasLength(256 * 256 * 3));
      expect(input[0], closeTo((1.0 - 0.485) / 0.229, 1e-5));
      expect(input[1], closeTo((0.0 - 0.456) / 0.224, 1e-5));
      expect(input[2], closeTo((0.0 - 0.406) / 0.225, 1e-5));
    } finally {
      rgb.dispose();
    }
  });

  test('postprocess resizes and clips probabilities', () {
    final input = Float32List(256 * 256)
      ..fillRange(0, 128 * 256, 1.5)
      ..fillRange(128 * 256, 256 * 256, -0.5);
    final map = postprocess(input, 300, 200);
    expect(map.width, 200);
    expect(map.height, 300);
    expect(map.data, hasLength(200 * 300));
    expect(map.data.every((value) => value >= 0.0 && value <= 1.0), isTrue);
    expect(map.data.first, 1.0);
    expect(map.data.last, 0.0);
  });

  test('ProbMap.rot90Ccw matches numpy.rot90', () {
    final map = ProbMap(
      3,
      2,
      Float32List.fromList([1, 2, 3, 4, 5, 6]),
    );
    final rotated = map.rot90Ccw();
    expect(rotated.width, 2);
    expect(rotated.height, 3);
    expect(rotated.data, [3, 6, 2, 5, 1, 4]);
  });
}
