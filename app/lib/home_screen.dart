import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';

import 'result_screen.dart';
import 'scan_outcome.dart';
import 'scan_service.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final ImagePicker _picker = ImagePicker();
  Future<ScanService>? _worker;
  bool _busy = false;
  String _progress = '';

  Future<ScanService> _ensureWorker() => _worker ??= ScanService.start();

  Future<void> _pickAndScan(ImageSource source) async {
    if (_busy) {
      return;
    }
    setState(() {
      _busy = true;
      _progress = source == ImageSource.camera
          ? 'Opening camera…'
          : 'Opening photo library…';
    });

    ScanOutcome? outcome;
    try {
      final picked = await _picker.pickImage(source: source);
      if (picked != null) {
        if (mounted) {
          setState(() => _progress = 'Scanning page…');
        }
        final bytes = await picked.readAsBytes();
        final worker = await _ensureWorker();
        outcome = await worker.scan(bytes);
      }
    } catch (error) {
      // A failed start leaves a dead future in the cache; try again next time.
      _worker = null;
      outcome = ScanOutcome.failure(_friendlyMessage(error));
    } finally {
      if (mounted) {
        setState(() {
          _busy = false;
          _progress = '';
        });
      }
    }

    final result = outcome;
    if (!mounted || result == null) {
      return;
    }
    await Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => ResultScreen(outcome: result)),
    );
  }

  String _friendlyMessage(Object error) {
    if (error is PlatformException) {
      return error.message ?? 'The photo could not be opened.';
    }
    return 'The photo could not be scanned: $error';
  }

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Scaffold(
      appBar: AppBar(title: const Text('MissileNotes')),
      body: Stack(
        children: [
          SafeArea(
            child: ListView(
              padding: const EdgeInsets.all(24),
              children: [
                const SizedBox(height: 24),
                Icon(
                  Icons.document_scanner_outlined,
                  size: 88,
                  color: Theme.of(context).colorScheme.primary,
                ),
                const SizedBox(height: 16),
                Text(
                  'Scan a notebook page',
                  style: textTheme.headlineSmall,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 8),
                Text(
                  'Print the MissileNotes marker, stick it on a page corner, '
                  'then photograph the page.',
                  style: textTheme.bodyMedium,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 40),
                FilledButton.icon(
                  onPressed: () => _pickAndScan(ImageSource.camera),
                  icon: const Icon(Icons.photo_camera_outlined),
                  label: const Text('Take photo'),
                  style: FilledButton.styleFrom(
                    minimumSize: const Size.fromHeight(52),
                  ),
                ),
                const SizedBox(height: 12),
                OutlinedButton.icon(
                  onPressed: () => _pickAndScan(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library_outlined),
                  label: const Text('Choose from library'),
                  style: OutlinedButton.styleFrom(
                    minimumSize: const Size.fromHeight(52),
                  ),
                ),
              ],
            ),
          ),
          if (_busy)
            Positioned.fill(
              child: ColoredBox(
                color: Colors.black54,
                child: Center(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const CircularProgressIndicator(),
                      const SizedBox(height: 16),
                      Text(
                        _progress,
                        style: const TextStyle(color: Colors.white),
                      ),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}
