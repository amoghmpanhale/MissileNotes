import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:opencv_dart/opencv.dart' as cv;
import 'package:tflite_flutter/tflite_flutter.dart' as tfl;

import 'types.dart';

const int inputSize = 256;
const List<double> imagenetMean = [0.485, 0.456, 0.406];
const List<double> imagenetStd = [0.229, 0.224, 0.225];

/// Anything that maps an RGB image to a probability map at the same HxW.
abstract class Segmenter {
  Future<ProbMap> predict(cv.Mat imageRgb);
}

/// FairScan segmentation with the shipped `.tflite` graph.
///
/// The graph takes NHWC float32 `1x256x256x3` and returns raw logits
/// `1x256x256x1`.  ImageNet normalization and the sigmoid are applied here,
/// outside the graph, exactly like the Python reference.
///
/// `tflite_flutter` is used instead of a newer LiteRT-specific plugin because
/// it is the maintained Flutter binding for the LiteRT C API, ships plugins
/// for Android and iOS, and exposes the raw tensor contract this port needs.
class TfliteSegmenter implements Segmenter {
  TfliteSegmenter._(this._interpreter) {
    final input = _interpreter.getInputTensor(0);
    final output = _interpreter.getOutputTensor(0);
    if (!_sameShape(input.shape, [1, inputSize, inputSize, 3]) ||
        input.type != tfl.TensorType.float32) {
      throw ArgumentError(
        'unexpected TFLite input ${input.shape} ${input.type}; '
        'expected [1, 256, 256, 3] float32',
      );
    }
    if (!_sameShape(output.shape, [1, inputSize, inputSize, 1]) ||
        output.type != tfl.TensorType.float32) {
      throw ArgumentError(
        'unexpected TFLite output ${output.shape} ${output.type}; '
        'expected [1, 256, 256, 1] float32',
      );
    }
  }

  factory TfliteSegmenter.fromFile(File model) {
    final options = tfl.InterpreterOptions()..threads = 2;
    return TfliteSegmenter._(
      tfl.Interpreter.fromFile(model, options: options),
    );
  }

  factory TfliteSegmenter.fromBytes(Uint8List model) {
    final options = tfl.InterpreterOptions()..threads = 2;
    return TfliteSegmenter._(
      tfl.Interpreter.fromBuffer(model, options: options),
    );
  }

  final tfl.Interpreter _interpreter;

  void dispose() => _interpreter.close();

  @override
  Future<ProbMap> predict(cv.Mat imageRgb) async {
    final inputImage = preprocess(imageRgb);
    final input = _interpreter.getInputTensor(0);
    input.data = inputImage.buffer.asUint8List();

    _interpreter.invoke();

    final output = _interpreter.getOutputTensor(0);
    final logits = output.data.buffer.asFloat32List(
      output.data.offsetInBytes,
      output.numElements(),
    );
    final prob = Float32List(logits.length);
    for (var i = 0; i < logits.length; i++) {
      prob[i] = _sigmoid(logits[i]);
    }
    return postprocess(prob, imageRgb.rows, imageRgb.cols);
  }
}

/// Resize to 256x256 and apply ImageNet normalization, returning NHWC floats.
Float32List preprocess(cv.Mat imageRgb, {int size = inputSize}) {
  if (imageRgb.channels != 3 ||
      imageRgb.type.value != cv.MatType.CV_8UC3.value) {
    throw ArgumentError(
      'imageRgb must be uint8 RGB, got ${imageRgb.rows}x${imageRgb.cols} '
      'channels=${imageRgb.channels} type=${imageRgb.type.value}',
    );
  }

  final resized = cv.resize(
    imageRgb,
    (size, size),
    interpolation: cv.INTER_CUBIC,
  );
  try {
    final pixels = resized.data;
    final output = Float32List(size * size * 3);
    var index = 0;
    for (var i = 0; i < pixels.length; i += 3) {
      output[index++] = (pixels[i] / 255.0 - imagenetMean[0]) / imagenetStd[0];
      output[index++] =
          (pixels[i + 1] / 255.0 - imagenetMean[1]) / imagenetStd[1];
      output[index++] =
          (pixels[i + 2] / 255.0 - imagenetMean[2]) / imagenetStd[2];
    }
    return output;
  } finally {
    resized.dispose();
  }
}

/// Resize a raw probability map to the input resolution and clip to [0, 1].
ProbMap postprocess(Float32List prob, int height, int width) {
  const size = inputSize;
  if (height == size && width == size) {
    return ProbMap(width, height, _clip(prob));
  }

  final source = cv.Mat.fromList(size, size, cv.MatType.CV_32FC1, prob);
  final resized = cv.Mat.empty();
  try {
    cv.resize(
      source,
      (width, height),
      dst: resized,
      interpolation: cv.INTER_CUBIC,
    );
    final values = resized.data.buffer
        .asFloat32List(resized.data.offsetInBytes, resized.total)
        .toList(growable: false);
    return ProbMap(
      width,
      height,
      _clip(Float32List.fromList(values)),
    );
  } finally {
    source.dispose();
    resized.dispose();
  }
}

Float32List _clip(Float32List values) {
  for (var i = 0; i < values.length; i++) {
    values[i] = values[i].clamp(0.0, 1.0).toDouble();
  }
  return values;
}

double _sigmoid(double value) => 1.0 / (1.0 + math.exp(-value));

bool _sameShape(List<int> actual, List<int> expected) {
  if (actual.length != expected.length) return false;
  for (var i = 0; i < actual.length; i++) {
    if (actual[i] != expected[i]) return false;
  }
  return true;
}
