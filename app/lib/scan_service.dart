import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:opencv_dart/opencv.dart' as cv;

import 'pipeline/pipeline.dart';
import 'pipeline/segmenter.dart';
import 'scan_outcome.dart';

const _modelAsset = 'assets/fairscan-segmentation-model.tflite';

/// Runs scans on a long-lived worker isolate.
///
/// The TFLite interpreter is loaded once, inside the worker, and reused for
/// every scan.  OpenCV and TFLite work then stays off the UI thread, and the
/// native Mats never cross the isolate boundary: only encoded bytes and plain
/// doubles come back.
class ScanService {
  ScanService._({
    required this._isolate,
    required this._commands,
    required this._responses,
    required this._errors,
    required this._responseSubscription,
    required this._errorSubscription,
    required this._pending,
  });

  static const _startupTimeout = Duration(seconds: 60);
  static const _scanTimeout = Duration(minutes: 2);

  final Isolate _isolate;
  final SendPort _commands;
  final ReceivePort _responses;
  final ReceivePort _errors;
  final StreamSubscription<dynamic> _responseSubscription;
  final StreamSubscription<dynamic> _errorSubscription;
  final Map<int, Completer<ScanOutcome>> _pending;

  int _nextId = 0;

  static Future<ScanService> start() async {
    final model = await rootBundle.load(_modelAsset);
    final responses = ReceivePort();
    final errors = ReceivePort();
    final isolate = await Isolate.spawn(
      _scanWorkerMain,
      responses.sendPort,
      onError: errors.sendPort,
      debugName: 'missilenotes-scan',
    );

    final commands = Completer<SendPort>();
    final ready = Completer<void>();
    final pending = <int, Completer<ScanOutcome>>{};

    void failEverything(Object error, [StackTrace? stack]) {
      if (!ready.isCompleted) ready.completeError(error, stack);
      for (final completer in pending.values) {
        if (!completer.isCompleted) completer.completeError(error, stack);
      }
      pending.clear();
    }

    final responseSubscription = responses.listen((message) {
      if (message is SendPort) {
        if (!commands.isCompleted) commands.complete(message);
      } else if (message is _ModelReady) {
        if (!ready.isCompleted) ready.complete();
      } else if (message is _ModelFailed) {
        if (!ready.isCompleted) {
          ready.completeError(StateError(message.message));
        }
      } else if (message is _ScanDone) {
        pending.remove(message.id)?.complete(message.outcome);
      } else if (message is _ScanFailed) {
        pending.remove(message.id)?.completeError(StateError(message.message));
      }
    });

    final errorSubscription = errors.listen((message) {
      final detail = message is List && message.isNotEmpty
          ? message.first
          : 'unknown';
      failEverything(StateError('The scan worker stopped: $detail'));
    });

    try {
      final commandPort = await commands.future.timeout(_startupTimeout);
      commandPort.send(
        _LoadModel(
          model.buffer.asUint8List(model.offsetInBytes, model.lengthInBytes),
        ),
      );
      await ready.future.timeout(_startupTimeout);
    } catch (error, stack) {
      await responseSubscription.cancel();
      await errorSubscription.cancel();
      responses.close();
      errors.close();
      isolate.kill(priority: Isolate.immediate);
      Error.throwWithStackTrace(error, stack);
    }

    return ScanService._(
      isolate: isolate,
      commands: await commands.future,
      responses: responses,
      errors: errors,
      responseSubscription: responseSubscription,
      errorSubscription: errorSubscription,
      pending: pending,
    );
  }

  Future<ScanOutcome> scan(Uint8List photoBytes) {
    final id = _nextId++;
    final completer = Completer<ScanOutcome>();
    _pending[id] = completer;
    _commands.send(_ScanJob(id, photoBytes));
    return completer.future.timeout(
      _scanTimeout,
      onTimeout: () {
        _pending.remove(id);
        throw TimeoutException('The scan took too long.');
      },
    );
  }

  void dispose() {
    _isolate.kill(priority: Isolate.immediate);
    _responseSubscription.cancel();
    _errorSubscription.cancel();
    _responses.close();
    _errors.close();
  }
}

class _LoadModel {
  const _LoadModel(this.model);

  final Uint8List model;
}

class _ModelReady {
  const _ModelReady();
}

class _ModelFailed {
  const _ModelFailed(this.message);

  final String message;
}

class _ScanJob {
  const _ScanJob(this.id, this.photo);

  final int id;
  final Uint8List photo;
}

class _ScanDone {
  const _ScanDone(this.id, this.outcome);

  final int id;
  final ScanOutcome outcome;
}

class _ScanFailed {
  const _ScanFailed(this.id, this.message);

  final int id;
  final String message;
}

Future<void> _scanWorkerMain(SendPort mainPort) async {
  final commands = ReceivePort();
  mainPort.send(commands.sendPort);

  TfliteSegmenter? segmenter;
  await for (final message in commands) {
    if (message is _LoadModel) {
      try {
        segmenter = TfliteSegmenter.fromBytes(message.model);
        mainPort.send(const _ModelReady());
      } catch (error) {
        mainPort.send(_ModelFailed('$error'));
      }
    } else if (message is _ScanJob) {
      final activeSegmenter = segmenter;
      if (activeSegmenter == null) {
        mainPort.send(
          _ScanFailed(message.id, 'The scan model has not loaded yet.'),
        );
        continue;
      }
      try {
        final outcome = await _scanPhoto(message.photo, activeSegmenter);
        mainPort.send(_ScanDone(message.id, outcome));
      } catch (error) {
        mainPort.send(_ScanFailed(message.id, '$error'));
      }
    }
  }
}

Future<ScanOutcome> _scanPhoto(
  Uint8List photoBytes,
  Segmenter segmenter,
) async {
  final decoded = img.decodeImage(photoBytes);
  if (decoded == null) {
    throw const FormatException('That file is not a photo this app can read.');
  }

  // The pipeline expects an upright image, like the Python CLI's EXIF
  // transpose.  Camera and library photos carry their orientation in EXIF.
  final oriented = img.bakeOrientation(decoded);
  final rgb = oriented.getBytes(order: img.ChannelOrder.rgb);
  final photo = cv.Mat.fromList(
    oriented.height,
    oriented.width,
    cv.MatType.CV_8UC3,
    rgb,
  );
  final originalJpeg = img.encodeJpg(oriented, quality: 85);

  try {
    final result = await scan(photo, segmenter);
    try {
      final quad = result.quad == null
          ? null
          : [
              for (final point in result.quad!) ...[point.x, point.y],
            ];
      final scanPng = result.scan == null ? null : _encodePng(result.scan!);
      return ScanOutcome(
        ok: result.ok,
        reason: result.reason,
        scanPng: scanPng,
        originalJpeg: originalJpeg,
        originalWidth: oriented.width,
        originalHeight: oriented.height,
        quad: quad,
        rotation: result.rotation,
        confidence: result.confidence,
        timings: result.timings,
      );
    } finally {
      result.scan?.dispose();
      result.flatRgb?.dispose();
      result.mask?.dispose();
    }
  } finally {
    photo.dispose();
  }
}

Uint8List _encodePng(cv.Mat image) {
  final (ok, bytes) = cv.imencode('.png', image);
  if (!ok) {
    throw StateError('The scanned page could not be encoded.');
  }
  return bytes;
}
