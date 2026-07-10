// ponytail: dev-only screen, not part of the shipped app (no link from any
// tab — open via the dev menu / deep link `app://parity`, or temporarily
// point a Settings link at it, same pattern spike.tsx used for Task 2.1).
//
// Golden-parity check for Task 2.2: runs the on-device pipeline
// (src/pipeline/scan.ts) against the bundled fixture photos and compares the
// detected page quad against experiments/export_goldens.py's Python-computed
// goldens (bundled at __fixtures__/goldens/*.quad.json). This is the one
// check that genuinely can't run in this environment (no device/simulator) —
// see task-2.2-report.md for why, and for what a PASS here confirms
// (morphology/warp/tflite native-op parity) that the Node tests can't.
import { useEffect, useState } from 'react';
import { Image, ScrollView, StyleSheet } from 'react-native';
import { Asset } from 'expo-asset';

import { Text, View } from '@/components/Themed';
import { detectPageQuadForParity, orderCorners, processScan, type Point } from '@/src/pipeline/scan';

/* eslint-disable @typescript-eslint/no-require-imports */
const FIXTURES: { name: string; module: number; golden: { corners: Point[] | null } }[] = [
  {
    name: 'IMG_4905',
    module: require('../src/pipeline/__fixtures__/IMG_4905.png'),
    golden: require('../src/pipeline/__fixtures__/goldens/IMG_4905.quad.json'),
  },
  {
    name: 'IMG_4906',
    module: require('../src/pipeline/__fixtures__/IMG_4906.png'),
    golden: require('../src/pipeline/__fixtures__/goldens/IMG_4906.quad.json'),
  },
  {
    name: 'IMG_4907',
    module: require('../src/pipeline/__fixtures__/IMG_4907.png'),
    golden: require('../src/pipeline/__fixtures__/goldens/IMG_4907.quad.json'),
  },
];
/* eslint-enable @typescript-eslint/no-require-imports */

const PASS_THRESHOLD_PX = 5;

type Result = {
  name: string;
  maxErrorPx: number | null;
  pass: boolean;
  detail: string;
  warpedUri: string | null;
};

function maxCornerDistance(a: Point[], b: Point[]): number {
  let max = 0;
  for (let i = 0; i < 4; i++) {
    const d = Math.hypot(a[i][0] - b[i][0], a[i][1] - b[i][1]);
    if (d > max) max = d;
  }
  return max;
}

export default function ParityScreen() {
  const [results, setResults] = useState<Result[]>([]);
  const [running, setRunning] = useState(true);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      const out: Result[] = [];
      for (const fixture of FIXTURES) {
        if (cancelled) return;
        try {
          const asset = Asset.fromModule(fixture.module);
          await asset.downloadAsync();
          const uri = asset.localUri ?? asset.uri;

          const detected = await detectPageQuadForParity(uri);
          const golden = fixture.golden.corners;

          if (!detected || !golden) {
            out.push({
              name: fixture.name,
              maxErrorPx: null,
              pass: false,
              detail: `detected=${detected ? 'quad' : 'null'} golden=${golden ? 'quad' : 'null'}`,
              warpedUri: null,
            });
          } else {
            const orderedDetected = orderCorners(detected);
            const orderedGolden = orderCorners(golden);
            const maxErrorPx = maxCornerDistance(orderedDetected, orderedGolden);
            out.push({
              name: fixture.name,
              maxErrorPx,
              pass: maxErrorPx <= PASS_THRESHOLD_PX,
              detail: `detected=${JSON.stringify(orderedDetected)} golden=${JSON.stringify(orderedGolden)}`,
              warpedUri: null,
            });
          }

          // Also run the full pipeline for the warped-image preview.
          const warpedUri = await processScan(uri);
          const last = out[out.length - 1];
          if (last) last.warpedUri = warpedUri === uri ? null : warpedUri;
        } catch (err) {
          out.push({
            name: fixture.name,
            maxErrorPx: null,
            pass: false,
            detail: String(err instanceof Error ? (err.stack ?? err.message) : err),
            warpedUri: null,
          });
        }
        if (!cancelled) setResults([...out]);
      }
      if (!cancelled) setRunning(false);
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Task 2.2 golden parity</Text>
      <Text style={styles.subtitle}>
        Compares src/pipeline/scan.ts's detected page quad against
        experiments/export_goldens.py's OpenCV-computed goldens. PASS = max
        corner error ≤ {PASS_THRESHOLD_PX}px.
      </Text>
      {running && <Text>Running…</Text>}
      {results.map((r) => (
        <View key={r.name} style={styles.block}>
          <Text style={[styles.label, r.pass ? styles.pass : styles.fail]}>
            {r.pass ? 'PASS' : 'FAIL'} — {r.name}
            {r.maxErrorPx != null ? ` (max error ${r.maxErrorPx.toFixed(1)}px)` : ''}
          </Text>
          <Text style={styles.detail}>{r.detail}</Text>
          {r.warpedUri && (
            <Image source={{ uri: r.warpedUri }} style={styles.warped} resizeMode="contain" />
          )}
        </View>
      ))}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 16,
    gap: 12,
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  subtitle: {
    fontSize: 12,
    opacity: 0.7,
    marginBottom: 8,
  },
  block: {
    borderWidth: StyleSheet.hairlineWidth,
    borderColor: '#888',
    borderRadius: 8,
    padding: 10,
    gap: 8,
  },
  label: {
    fontWeight: '600',
  },
  pass: {
    color: '#2a9d3f',
  },
  fail: {
    color: '#c0392b',
  },
  detail: {
    fontFamily: 'monospace',
    fontSize: 11,
  },
  warped: {
    width: '100%',
    height: 300,
    backgroundColor: '#222',
  },
});
