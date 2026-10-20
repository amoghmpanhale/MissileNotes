import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:missilenotes/pipeline/segmenter.dart';
import 'package:missilenotes/pipeline/types.dart';
import 'package:opencv_dart/opencv.dart' as cv;

final Directory fixtureDir = Directory('test/fixtures');

class Fixture {
  Fixture(this.payload);

  final Map<String, dynamic> payload;

  int get index => payload['index'] as int;
  String get name => 'fixture_$index';

  File get photo => File('${fixtureDir.path}/${payload['photo']}');
  File? get probModel {
    final name = payload['prob_model'] as String?;
    return name == null ? null : File('${fixtureDir.path}/$name');
  }

  File? get expectedScan {
    final name = payload['expected_scan'] as String?;
    return name == null ? null : File('${fixtureDir.path}/$name');
  }

  bool get ok => payload['ok'] as bool;
  String get reason => payload['reason'] as String;
  int get rotation => payload['rotation'] as int;
  double get confidence => (payload['confidence'] as num).toDouble();
  List<Pt>? get quad {
    final raw = payload['quad'] as List<dynamic>?;
    if (raw == null) return null;
    return [
      for (final point in raw)
        Pt(
          ((point as List<dynamic>)[0] as num).toDouble(),
          (point[1] as num).toDouble(),
        ),
    ];
  }

  List<int>? get scanShape => (payload['scan_shape'] as List<dynamic>?)
      ?.map((value) => value as int)
      .toList();

  static List<Fixture> loadAll() {
    final files = fixtureDir
        .listSync()
        .whereType<File>()
        .where((file) => RegExp(r'fixture_\d+\.json$').hasMatch(file.path))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    return [
      for (final file in files)
        Fixture(
          jsonDecode(file.readAsStringSync()) as Map<String, dynamic>,
        ),
    ];
  }
}

/// Decode a PNG as RGB, regardless of how it was encoded.
cv.Mat readRgb(File file) {
  final bgr = cv.imdecode(file.readAsBytesSync(), cv.IMREAD_COLOR);
  try {
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB);
  } finally {
    bgr.dispose();
  }
}

cv.Mat readGray(File file) {
  return cv.imdecode(file.readAsBytesSync(), cv.IMREAD_GRAYSCALE);
}

ProbMap readProb(File file) {
  final gray = readGray(file);
  try {
    final data = Float32List(gray.rows * gray.cols);
    final pixels = gray.data;
    for (var i = 0; i < pixels.length; i++) {
      data[i] = pixels[i] / 255.0;
    }
    return ProbMap(gray.cols, gray.rows, data);
  } finally {
    gray.dispose();
  }
}

/// Returns the saved Python probability map for the model-frame image.
class FixtureSegmenter implements Segmenter {
  FixtureSegmenter(this.map);

  final ProbMap map;

  @override
  Future<ProbMap> predict(cv.Mat imageRgb) async {
    if (imageRgb.rows != map.height || imageRgb.cols != map.width) {
      throw StateError(
        'fixture map is ${map.height}x${map.width}, '
        'pipeline asked for ${imageRgb.rows}x${imageRgb.cols}',
      );
    }
    return ProbMap(map.width, map.height, Float32List.fromList(map.data));
  }
}

class ExplodingSegmenter implements Segmenter {
  @override
  Future<ProbMap> predict(cv.Mat imageRgb) {
    throw StateError('segmenter must not run without a marker');
  }
}
