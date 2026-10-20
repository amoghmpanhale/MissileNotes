import 'package:opencv_dart/opencv.dart' as cv;

import 'enhance.dart';
import 'geometry.dart';
import 'marker.dart';
import 'segmenter.dart';
import 'types.dart';

/// Outcome of [scan].
///
/// [scan] is the binarized page (0 = ink, 255 = paper), [flatRgb] the upright
/// warped colour image, and [quad] is TL/TR/BR/BL in the original photo's
/// coordinates.  [rotation] is the CCW angle applied to make the page upright.
/// [timings] holds milliseconds per stage that actually ran.
class ScanResult {
  const ScanResult({
    required this.ok,
    required this.reason,
    required this.scan,
    required this.flatRgb,
    required this.quad,
    required this.rotation,
    required this.confidence,
    required this.timings,
    this.mask,
  });

  final bool ok;
  final String reason;
  final cv.Mat? scan;
  final cv.Mat? flatRgb;
  final List<Pt>? quad;
  final int rotation;
  final double confidence;
  final Map<String, double> timings;

  /// Thresholded page mask in the original photo frame; used for overlays.
  final cv.Mat? mask;
}

/// Scan one RGB photo.  Never throws for a bad photo.
///
/// Missing markers, pages the model cannot find, and unusable geometry return
/// `ok=false` with a human-readable reason.  Only programmer errors (wrong
/// Mat type/shape) throw.
Future<ScanResult> scan(
  cv.Mat imageRgb,
  Segmenter segmenter, {
  double threshold = 0.5,
  bool binarizeOutput = true,
}) async {
  _validate(imageRgb);
  final timings = <String, double>{};

  var markerTimer = Stopwatch()..start();
  final gray = cv.cvtColor(imageRgb, cv.COLOR_RGB2GRAY);
  final marker = detectMarker(gray);
  gray.dispose();
  timings['marker'] = _elapsedMs(markerTimer);

  if (marker == null) {
    return ScanResult(
      ok: false,
      reason: 'no ArUco marker found',
      scan: null,
      flatRgb: null,
      quad: null,
      rotation: 0,
      confidence: 0.0,
      timings: timings,
    );
  }

  final rotation = uprightRotation(marker);
  final upright = rotateImage(imageRgb, rotation.toDouble());
  final uprightMarker = Marker(
    rotatePoints(
      marker.corners,
      rotation.toDouble(),
      imageRgb.cols,
      imageRgb.rows,
    ),
    marker.id,
  );
  final uprightGray = cv.cvtColor(upright, cv.COLOR_RGB2GRAY);

  // The horizontal model expects the page turned 90 degrees clockwise from
  // upright; -90 is a clockwise quarter turn in rotateImage's CCW convention.
  final modelImage = rotateImage(upright, -90);

  var predictTimer = Stopwatch()..start();
  final probModel = await segmenter.predict(modelImage);
  timings['predict'] = _elapsedMs(predictTimer);
  if (probModel.width != modelImage.cols ||
      probModel.height != modelImage.rows) {
    throw ArgumentError(
      'segmenter.predict must return an HxW map matching its input; '
      'got ${probModel.height}x${probModel.width} for '
      '${modelImage.rows}x${modelImage.cols}',
    );
  }

  // Undo the clockwise turn: one CCW quarter turn returns to upright.
  final probUpright = probModel.rot90Ccw();
  final maskUpright = _threshold(probUpright, threshold);

  var geometryTimer = Stopwatch()..start();
  final quadResult = pageQuad(maskUpright, uprightMarker, uprightGray);
  timings['geometry'] = _elapsedMs(geometryTimer);

  // Keep a mask overlay in the frame of the photo the caller passed in.
  final maskOriginal = rotateImage(maskUpright, -rotation.toDouble());

  if (quadResult.quad == null) {
    maskUpright.dispose();
    uprightGray.dispose();
    modelImage.dispose();
    upright.dispose();
    return ScanResult(
      ok: false,
      reason: quadResult.reason,
      scan: null,
      flatRgb: null,
      quad: null,
      rotation: rotation,
      confidence: quadResult.confidence,
      timings: timings,
      mask: maskOriginal,
    );
  }

  final quadOriginal = rotatePoints(
    quadResult.quad!,
    -rotation.toDouble(),
    upright.cols,
    upright.rows,
  );

  var warpTimer = Stopwatch()..start();
  final flatRgb = warp(upright, quadResult.quad!);
  timings['warp'] = _elapsedMs(warpTimer);

  cv.Mat? scanImage;
  if (binarizeOutput) {
    var binarizeTimer = Stopwatch()..start();
    final flatGray = cv.cvtColor(flatRgb, cv.COLOR_RGB2GRAY);
    scanImage = binarize(flatGray);
    flatGray.dispose();
    timings['binarize'] = _elapsedMs(binarizeTimer);
  }

  maskUpright.dispose();
  uprightGray.dispose();
  modelImage.dispose();
  upright.dispose();

  return ScanResult(
    ok: true,
    reason: quadResult.reason,
    scan: scanImage,
    flatRgb: flatRgb,
    quad: quadOriginal,
    rotation: rotation,
    confidence: quadResult.confidence,
    timings: timings,
    mask: maskOriginal,
  );
}

cv.Mat _threshold(ProbMap prob, double threshold) {
  final mask = cv.Mat.zeros(prob.height, prob.width, cv.MatType.CV_8UC1);
  final data = mask.data;
  for (var i = 0; i < prob.data.length; i++) {
    if (prob.data[i] > threshold) data[i] = 1;
  }
  return mask;
}

void _validate(cv.Mat image) {
  if (image.isEmpty) {
    throw ArgumentError('imageRgb must be non-empty');
  }
  if (image.channels != 3 ||
      image.type.value != cv.MatType.CV_8UC3.value) {
    throw ArgumentError(
      'imageRgb must be HxWx3 uint8 RGB, got '
      '${image.rows}x${image.cols} channels=${image.channels} '
      'type=${image.type.value}',
    );
  }
}

double _elapsedMs(Stopwatch timer) {
  timer.stop();
  return timer.elapsedMicroseconds / 1000.0;
}
