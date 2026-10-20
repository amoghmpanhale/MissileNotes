import 'package:flutter_test/flutter_test.dart';
import 'package:missilenotes/pipeline/marker.dart';
import 'package:missilenotes/pipeline/types.dart';
import 'package:opencv_dart/opencv.dart' as cv;

import 'fixture_utils.dart';

Marker markerWithUp(double dx, double dy) {
  // corners[0] - corners[3] is the marker's up vector.
  return Marker([
    Pt(0, 0),
    Pt(1, 0),
    Pt(1, 1),
    Pt(-dx, -dy),
  ], 7);
}

void main() {
  test('upright rotation quantises the marker up vector', () {
    expect(uprightRotation(markerWithUp(0, 1)), 0);
    expect(uprightRotation(markerWithUp(1, 0)), 270);
    expect(uprightRotation(markerWithUp(0, -1)), 180);
    expect(uprightRotation(markerWithUp(-1, 0)), 90);
  });

  test('rotatePoints follows numpy.rot90', () {
    const point = [Pt(1, 2)];
    const width = 6;
    const height = 4;

    Pt rotated(int degrees) => rotatePoints(
          point,
          degrees.toDouble(),
          width,
          height,
        ).single;

    expect((rotated(0).x, rotated(0).y), (1, 2));
    expect((rotated(90).x, rotated(90).y), (2, 4));
    expect((rotated(180).x, rotated(180).y), (4, 1));
    expect((rotated(270).x, rotated(270).y), (1, 1));
  });

  test('rotateImage and rotatePoints agree on a 90 degree turn', () {
    final data = [for (var i = 0; i < 15; i++) i];
    final image = cv.Mat.fromList(3, 5, cv.MatType.CV_8UC1, data);
    final rotated = rotateImage(image, 90);
    try {
      expect(rotated.rows, 5);
      expect(rotated.cols, 3);
      expect(rotated.atU8(2, i1: 1), 7);
      expect(rotated.atU8(0, i1: 2), 14);
    } finally {
      rotated.dispose();
      image.dispose();
    }
  });

  test('finds the marker in a fixture photo and none without one', () {
    final withMarker = readGray(Fixture.loadAll()[0].photo);
    final withoutMarker = readGray(Fixture.loadAll()[5].photo);
    try {
      final marker = detectMarker(withMarker);
      expect(marker, isNotNull);
      expect(marker!.id, 7);
      expect(marker.corners, hasLength(4));
      expect(detectMarker(withoutMarker), isNull);
    } finally {
      withMarker.dispose();
      withoutMarker.dispose();
    }
  });
}
