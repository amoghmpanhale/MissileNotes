import 'dart:typed_data';

/// A scan result in a form the UI can hold.
///
/// The pipeline returns OpenCV Mats, which cannot leave the worker isolate.
/// The worker therefore encodes the scan and the original photo as PNG/JPEG
/// bytes and flattens the page corners into plain doubles.
class ScanOutcome {
  const ScanOutcome({
    required this.ok,
    required this.reason,
    this.scanPng,
    this.originalJpeg,
    this.originalWidth = 0,
    this.originalHeight = 0,
    this.quad,
    this.rotation = 0,
    this.confidence = 0,
    this.timings = const {},
  });

  factory ScanOutcome.failure(String reason) =>
      ScanOutcome(ok: false, reason: reason);

  final bool ok;
  final String reason;

  /// Binarized page as PNG (0 = ink, 255 = paper), when [ok].
  final Uint8List? scanPng;

  /// EXIF-upright photo the pipeline saw, for the debug quad overlay.
  final Uint8List? originalJpeg;
  final int originalWidth;
  final int originalHeight;

  /// Page corners in [originalJpeg] coordinates: TL, TR, BR, BL as x/y pairs.
  final List<double>? quad;

  /// Counter-clockwise rotation the pipeline applied to make the page upright.
  final int rotation;
  final double confidence;
  final Map<String, double> timings;
}
