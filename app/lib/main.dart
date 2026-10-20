import 'package:flutter/material.dart';

import 'home_screen.dart';

void main() {
  runApp(const MissileNotesApp());
}

class MissileNotesApp extends StatelessWidget {
  const MissileNotesApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MissileNotes',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
      ),
      home: const HomeScreen(),
    );
  }
}
