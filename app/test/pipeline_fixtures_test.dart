import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:missilenotes/pipeline/geometry.dart';
import 'package:missilenotes/pipeline/pipeline.dart';
import 'package:missilenotes/pipeline/types.dart';
import 'package:opencv_dart/opencv.dart' as cv;

import 'fixture_utils.dart';

void main() {
  final fixtures = Fixture.loadAll();

  test('fixture set is complete', () {
    expect(fixtures, hasLength(6));
    expect([for (final f in fixtures) f.index], [1, 2, 3, 4, 5, 6]);
  });

  test('binarizeOutput=false keeps the colour warp only', () async {
    final fixture = fixtures.first;
    final photo = readRgb(fixture.photo);
    final result = await scan(
      photo,
      FixtureSegmenter(readProb(fixture.probModel!)),
      binarizeOutput: false,
    );
    try {
      expect(result.ok, isTrue);
      expect(result.flatRgb, isNotNull);
      expect(result.scan, isNull);
      expect(result.timings.keys, isNot(contains('binarize')));
    } finally {
      photo.dispose();
      result.flatRgb?.dispose();
      result.mask?.dispose();
    }
  });

  for (final fixture in fixtures) {
    test('Dart and Python agree on ${fixture.name}', () async {
      final photo = readRgb(fixture.photo);
      final segmenter = fixture.probModel == null
          ? ExplodingSegmenter()
          : FixtureSegmenter(readProb(fixture.probModel!));

      final result = await scan(photo, segmenter);
      try {
        expect(result.ok, fixture.ok, reason: 'Dart: ${result.reason}');
        expect(result.rotation, fixture.rotation);

        if (!fixture.ok) {
          expect(result.scan, isNull);
          expect(result.quad, isNull);
          // Same failure class: both sides name the missing marker.
          expect(result.reason.toLowerCase(), contains('marker'));
          expect(result.timings.keys.toSet(), {'marker'});
          return;
        }

        expect(result.scan, isNotNull);
        expect(result.flatRgb, isNotNull);
        expect(result.quad, isNotNull);
        expect(result.timings.keys, containsAll([
          'marker',
          'predict',
          'geometry',
          'warp',
          'binarize',
        ]));

        final actualQuad = orderCorners(result.quad!);
        final expectedQuad = orderCorners(fixture.quad!);
        final diagonal = Pt.distance(expectedQuad[0], expectedQuad[2]);
        var worst = 0.0;
        for (var i = 0; i < 4; i++) {
          worst = math.max(worst, Pt.distance(actualQuad[i], expectedQuad[i]));
        }
        expect(
          worst,
          lessThanOrEqualTo(0.01 * diagonal),
          reason: 'worst corner ${worst.toStringAsFixed(2)} px '
              'vs diagonal ${diagonal.toStringAsFixed(2)} px',
        );

        final expectedScan = readGray(fixture.expectedScan!);
        try {
          final expectedShape = fixture.scanShape!;
          var actual = result.scan!;
          cv.Mat? resized;
          if (actual.rows != expectedShape[0] ||
              actual.cols != expectedShape[1]) {
            resized = cv.resize(
              actual,
              (expectedShape[1], expectedShape[0]),
              interpolation: cv.INTER_NEAREST,
            );
            actual = resized;
          }
          try {
            expect(actual.rows, expectedScan.rows);
            expect(actual.cols, expectedScan.cols);
            final a = actual.data;
            final b = expectedScan.data;
            var same = 0;
            for (var i = 0; i < a.length; i++) {
              if (a[i] == b[i]) same++;
            }
            final agreement = same / a.length;
            expect(
              agreement,
              greaterThanOrEqualTo(0.98),
              reason: 'binary scan agreement '
                  '${(agreement * 100).toStringAsFixed(2)}%',
            );
          } finally {
            resized?.dispose();
          }
        } finally {
          expectedScan.dispose();
        }
      } finally {
        photo.dispose();
        result.scan?.dispose();
        result.flatRgb?.dispose();
        result.mask?.dispose();
      }
    });
  }
}
