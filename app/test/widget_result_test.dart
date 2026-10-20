import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image/image.dart' as img;
import 'package:missilenotes/result_screen.dart';
import 'package:missilenotes/scan_outcome.dart';

Uint8List _png(int width, int height) =>
    Uint8List.fromList(img.encodePng(img.Image(width: width, height: height)));

Uint8List _jpeg(int width, int height) =>
    Uint8List.fromList(img.encodeJpg(img.Image(width: width, height: height)));

void main() {
  testWidgets('success shows the scan, share, and a debug toggle', (
    tester,
  ) async {
    final outcome = ScanOutcome(
      ok: true,
      reason: 'page found',
      scanPng: _png(8, 8),
      originalJpeg: _jpeg(8, 8),
      originalWidth: 8,
      originalHeight: 8,
      quad: const [0, 0, 8, 0, 8, 8, 0, 8],
      confidence: 0.93,
    );

    await tester.pumpWidget(MaterialApp(home: ResultScreen(outcome: outcome)));
    await tester.pumpAndSettle();

    expect(find.text('Share'), findsOneWidget);
    expect(find.text('Show original'), findsOneWidget);
    expect(find.byType(Image), findsOneWidget);

    await tester.tap(find.text('Show original'));
    await tester.pumpAndSettle();

    expect(find.text('Show scan'), findsOneWidget);
    expect(find.byKey(const Key('debug-quad')), findsOneWidget);
  });

  testWidgets('failure shows the reason and a retry action', (tester) async {
    const outcome = ScanOutcome(ok: false, reason: 'no ArUco marker found');

    await tester.pumpWidget(
      const MaterialApp(home: ResultScreen(outcome: outcome)),
    );

    expect(find.text('no ArUco marker found'), findsOneWidget);
    expect(find.text('Try again'), findsOneWidget);
    expect(find.text('Share'), findsNothing);
  });
}
