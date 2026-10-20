import 'dart:io';
import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:missilenotes/pipeline/geometry.dart';
import 'package:missilenotes/pipeline/pipeline.dart';
import 'package:missilenotes/pipeline/segmenter.dart';
import 'package:missilenotes/pipeline/types.dart';

import 'fixture_utils.dart';

/// tflite_flutter 0.12.1 opens
/// `${Directory(Platform.resolvedExecutable).parent.path}/blobs/libtensorflowlite_c-linux.so`
/// on Linux. `flutter_tester` has no `blobs/` directory and the package does
/// not bundle the native library, so this test is device/CI only.
final File _hostLibrary = File(
  '${File(Platform.resolvedExecutable).parent.path}'
  '/blobs/libtensorflowlite_c-linux.so',
);

void main() {
  final fixture = Fixture.loadAll().first;
  final model =
      File('../experiments/model_horizontal/fairscan-segmentation-model.tflite');

  test(
    'real TFLite scan matches the Python fixture',
    () async {
      final photo = readRgb(fixture.photo);
      final segmenter = TfliteSegmenter.fromFile(model);
      try {
        final result = await scan(photo, segmenter);
        try {
          expect(result.ok, isTrue, reason: result.reason);
          expect(result.rotation, fixture.rotation);

          final expectedQuad = orderCorners(fixture.quad!);
          final actualQuad = orderCorners(result.quad!);
          final diagonal = Pt.distance(expectedQuad[0], expectedQuad[2]);
          var worst = 0.0;
          for (var i = 0; i < 4; i++) {
            worst = math.max(worst, Pt.distance(actualQuad[i], expectedQuad[i]));
          }
          expect(worst, lessThanOrEqualTo(0.05 * diagonal));

          final expected = readGray(fixture.expectedScan!);
          try {
            final actual = result.scan!;
            expect(actual.rows, expected.rows);
            expect(actual.cols, expected.cols);
            var same = 0;
            for (var i = 0; i < actual.data.length; i++) {
              if (actual.data[i] == expected.data[i]) same++;
            }
            expect(same / actual.data.length, greaterThanOrEqualTo(0.98));
          } finally {
            expected.dispose();
          }
        } finally {
          result.scan?.dispose();
          result.flatRgb?.dispose();
          result.mask?.dispose();
        }
      } finally {
        photo.dispose();
        segmenter.dispose();
      }
    },
    skip: Platform.isLinux && !_hostLibrary.existsSync()
        ? 'flutter_tester has no blobs/libtensorflowlite_c-linux.so; '
            'tflite_flutter cannot load the native LiteRT library'
        : null,
  );
}
