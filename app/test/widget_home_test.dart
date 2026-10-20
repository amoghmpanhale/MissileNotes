import 'package:flutter_test/flutter_test.dart';
import 'package:missilenotes/main.dart';

void main() {
  testWidgets('home screen offers camera and library', (tester) async {
    await tester.pumpWidget(const MissileNotesApp());

    expect(find.text('Take photo'), findsOneWidget);
    expect(find.text('Choose from library'), findsOneWidget);
  });
}
