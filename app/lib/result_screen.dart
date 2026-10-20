import 'dart:io';

import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

import 'scan_outcome.dart';

class ResultScreen extends StatefulWidget {
  const ResultScreen({super.key, required this.outcome, this.onRetry});

  final ScanOutcome outcome;

  /// Called by "Try again".  When absent the screen simply pops.
  final VoidCallback? onRetry;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  bool _showOriginal = false;

  bool get _canShowOriginal =>
      widget.outcome.originalJpeg != null && widget.outcome.quad != null;

  void _retry() {
    final onRetry = widget.onRetry;
    if (onRetry != null) {
      onRetry();
    } else {
      Navigator.of(context).maybePop();
    }
  }

  Future<void> _share() async {
    final bytes = widget.outcome.scanPng;
    if (bytes == null) {
      return;
    }
    try {
      final directory = await getTemporaryDirectory();
      final file = File(
        '${directory.path}/missilenotes-'
        '${DateTime.now().millisecondsSinceEpoch}.png',
      );
      await file.writeAsBytes(bytes, flush: true);
      await SharePlus.instance.share(
        ShareParams(
          files: [XFile(file.path, mimeType: 'image/png')],
          subject: 'MissileNotes scan',
        ),
      );
    } catch (error) {
      if (!mounted) {
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not share the scan: $error')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final outcome = widget.outcome;
    return Scaffold(
      appBar: AppBar(title: Text(outcome.ok ? 'Scan' : 'Scan failed')),
      body: SafeArea(
        child: outcome.ok ? _buildSuccess(context) : _buildFailure(context),
      ),
    );
  }

  Widget _buildSuccess(BuildContext context) {
    final outcome = widget.outcome;
    final textTheme = Theme.of(context).textTheme;
    return Column(
      children: [
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 12, 20, 0),
          child: Row(
            children: [
              Icon(Icons.check_circle, color: Colors.green.shade600),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  outcome.confidence > 0
                      ? 'Page found · ${(outcome.confidence * 100).round()}% '
                            'confidence'
                      : 'Page found',
                  style: textTheme.bodyMedium,
                ),
              ),
            ],
          ),
        ),
        Expanded(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Container(
              clipBehavior: Clip.antiAlias,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: Theme.of(context).dividerColor),
              ),
              child: _showOriginal
                  ? _DebugOverlay(outcome: outcome)
                  : Image.memory(
                      outcome.scanPng!,
                      fit: BoxFit.contain,
                      gaplessPlayback: true,
                    ),
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 0, 20, 20),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  onPressed: _share,
                  icon: const Icon(Icons.ios_share),
                  label: const Text('Share'),
                ),
              ),
              if (_canShowOriginal)
                TextButton.icon(
                  onPressed: () =>
                      setState(() => _showOriginal = !_showOriginal),
                  icon: Icon(
                    _showOriginal ? Icons.description : Icons.photo_outlined,
                  ),
                  label: Text(_showOriginal ? 'Show scan' : 'Show original'),
                ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildFailure(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.find_in_page_outlined,
              size: 56,
              color: Theme.of(context).colorScheme.primary,
            ),
            const SizedBox(height: 16),
            Text(
              "Couldn't find the page",
              style: textTheme.titleLarge,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              widget.outcome.reason,
              style: textTheme.bodyMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 24),
            FilledButton.icon(
              onPressed: _retry,
              icon: const Icon(Icons.refresh),
              label: const Text('Try again'),
            ),
          ],
        ),
      ),
    );
  }
}

class _DebugOverlay extends StatelessWidget {
  const _DebugOverlay({required this.outcome});

  final ScanOutcome outcome;

  @override
  Widget build(BuildContext context) {
    final jpeg = outcome.originalJpeg!;
    final quad = outcome.quad!;
    final imageSize = Size(
      outcome.originalWidth.toDouble(),
      outcome.originalHeight.toDouble(),
    );

    return LayoutBuilder(
      builder: (context, constraints) {
        final fitted = applyBoxFit(
          BoxFit.contain,
          imageSize,
          constraints.biggest,
        );
        final destination = Alignment.center.inscribe(
          fitted.destination,
          Offset.zero & constraints.biggest,
        );
        return Stack(
          fit: StackFit.expand,
          children: [
            Image.memory(jpeg, fit: BoxFit.contain, gaplessPlayback: true),
            Positioned.fromRect(
              rect: destination,
              child: CustomPaint(
                key: const Key('debug-quad'),
                painter: _PageQuadPainter(quad: quad, imageSize: imageSize),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _PageQuadPainter extends CustomPainter {
  const _PageQuadPainter({required this.quad, required this.imageSize});

  final List<double> quad;
  final Size imageSize;

  @override
  void paint(Canvas canvas, Size size) {
    if (quad.length < 8) {
      return;
    }
    final points = [
      for (var i = 0; i < 8; i += 2)
        Offset(
          quad[i] / imageSize.width * size.width,
          quad[i + 1] / imageSize.height * size.height,
        ),
    ];
    final stroke = Paint()
      ..color = Colors.redAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;
    canvas.drawPath(Path()..addPolygon(points, true), stroke);
    for (final point in points) {
      canvas.drawCircle(point, 6, Paint()..color = Colors.redAccent);
    }
  }

  @override
  bool shouldRepaint(_PageQuadPainter oldDelegate) =>
      oldDelegate.quad != quad || oldDelegate.imageSize != imageSize;
}
