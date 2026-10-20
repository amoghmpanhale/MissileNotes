import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv.dart' as cv;

void main() {
  test('opencv native asset loads', () {
    final mat = cv.Mat.zeros(8, 8, cv.MatType.CV_8UC1);
    expect(mat.rows, 8);
    mat.dispose();
  });
}
