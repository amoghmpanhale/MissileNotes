// ponytail: throwaway diagnostic scaffolding for Task 2.1 (native module spike).
// Delete this screen (and the Settings tab link to it) once Task 2.2 lands real
// camera + pipeline screens that exercise the same libraries for real.
//
// What this proves, on a physical device/simulator running the DEV CLIENT
// (not Expo Go — these are native modules, Expo Go can't load them):
//   1. react-native-fast-tflite loads the bundled FairScan .tflite model and
//      reports its input/output tensor shapes.
//   2. It runs one inference against a bundled fixture photo and reports
//      output length + timing.
//   3. react-native-fast-opencv's native binding loads and can decode/resize/
//      threshold/find-contours on the same fixture.
//   4. ArUco (DICT_6X6_50) detection: fast-opencv has NO aruco API (checked
//      its ObjectDetection.ts — only matchTemplate/phaseCorrelate are
//      exposed). This is a documented gap, not a bug in this screen. See the
//      fallback note rendered below and task-2.1-report.md.
//
// Everything is wrapped in try/catch per stage so a failure in one library
// still lets you see the others' results — this is meant to be read as a
// diagnostic dump, not a pass/fail test.

import { useEffect, useState } from 'react';
import { ScrollView, StyleSheet } from 'react-native';
import { Text, View } from '@/components/Themed';

type LogLine = { label: string; ok: boolean; detail: string };

export default function SpikeScreen() {
  const [lines, setLines] = useState<LogLine[]>([]);

  useEffect(() => {
    let cancelled = false;
    const log = (label: string, ok: boolean, detail: string) => {
      if (cancelled) return;
      setLines((prev) => [...prev, { label, ok, detail }]);
    };

    (async () => {
      // --- Stage 1: fast-tflite model load ---------------------------------
      let model: Awaited<ReturnType<typeof import('react-native-fast-tflite').loadTensorflowModel>> | null = null;
      try {
        const { loadTensorflowModel } = await import('react-native-fast-tflite');
        const t0 = Date.now();
        model = await loadTensorflowModel(
          // eslint-disable-next-line @typescript-eslint/no-require-imports
          require('../assets/model/fairscan-segmentation-model.tflite'),
          []
        );
        const loadMs = Date.now() - t0;
        log(
          'tflite: model load',
          true,
          `loaded in ${loadMs}ms\ninputs: ${JSON.stringify(model.inputs)}\noutputs: ${JSON.stringify(model.outputs)}`
        );
      } catch (e) {
        log('tflite: model load', false, String(e instanceof Error ? e.stack ?? e.message : e));
      }

      // --- Stage 2: fast-opencv native binding + decode a fixture ----------
      let grayBase64: string | null = null;
      let fixtureCols = 0;
      let fixtureRows = 0;
      try {
        const { Asset } = await import('expo-asset');
        const FileSystemLegacy = await import('expo-file-system/legacy');
        const OpenCVModule = await import('react-native-fast-opencv');
        const { OpenCV, ObjectType, ColorConversionCodes, DataTypes } = OpenCVModule;

        // eslint-disable-next-line @typescript-eslint/no-require-imports
        const asset = Asset.fromModule(require('../src/pipeline/__fixtures__/IMG_4905.png'));
        await asset.downloadAsync();
        if (!asset.localUri) throw new Error('fixture asset has no localUri after downloadAsync()');

        const base64 = await FileSystemLegacy.readAsStringAsync(asset.localUri, {
          encoding: 'base64',
        });

        const srcMat = OpenCV.base64ToMat(base64);
        const srcInfo = OpenCV.toJSValue(srcMat);
        fixtureCols = srcInfo.cols;
        fixtureRows = srcInfo.rows;

        const grayMat = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_8U);
        OpenCV.invoke('cvtColor', srcMat, grayMat, ColorConversionCodes.COLOR_BGR2GRAY);

        const edgesMat = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_8U);
        OpenCV.invoke('Canny', grayMat, edgesMat, 75, 150);

        const contours = OpenCV.createObject(ObjectType.PointVectorOfVectors);
        const hierarchy = OpenCV.createObject(ObjectType.Mat, 0, 0, DataTypes.CV_32S);
        OpenCV.invoke('findContoursWithHierarchy', edgesMat, contours, hierarchy, 0, 1);
        const contoursValue = OpenCV.toJSValue(contours as never) as unknown as { array: unknown[] };

        const grayResult = OpenCV.toJSValue(grayMat, 'png');
        grayBase64 = grayResult.base64;

        log(
          'opencv: native binding + decode + ops',
          true,
          `decoded fixture IMG_4905.png: ${fixtureCols}x${fixtureRows}\n` +
            `cvtColor -> gray ok, Canny -> edges ok, findContours -> ${
              Array.isArray(contoursValue?.array) ? contoursValue.array.length : '?'
            } contours`
        );

        OpenCV.clearBuffers();
      } catch (e) {
        log('opencv: native binding + decode + ops', false, String(e instanceof Error ? e.stack ?? e.message : e));
      }

      // --- Stage 3: one tflite inference against the fixture ---------------
      try {
        if (!model) throw new Error('skipped: model did not load in stage 1');
        const inputTensor = model.inputs[0];
        if (!inputTensor) throw new Error('model reports no input tensors');

        // Model expects NHWC float32, e.g. [1, 512, 512, 3] (see
        // experiments/cache_masks.py SIZE=512). We don't have a real resize-
        // to-model-input path wired yet (that's Task 2.2's job) so this is a
        // zero-filled buffer of the CORRECT shape/dtype — it proves the JSI
        // round-trip (alloc, run, read output) works, not that the mask is
        // meaningful.
        const totalElements = inputTensor.shape.reduce((a, b) => a * b, 1);
        const input = new Float32Array(totalElements); // zero-filled
        const t0 = Date.now();
        const outputs = await model.run([input.buffer]);
        const runMs = Date.now() - t0;
        const out = outputs[0];
        log(
          'tflite: run() on zero-filled input',
          true,
          `ran in ${runMs}ms, output byteLength=${out?.byteLength ?? 'n/a'} (elements=${
            out ? out.byteLength / 4 : 'n/a'
          })`
        );
      } catch (e) {
        log('tflite: run() on zero-filled input', false, String(e instanceof Error ? e.stack ?? e.message : e));
      }

      // --- Stage 4: ArUco — documented gap, not attempted ------------------
      log(
        'opencv: ArUco DICT_6X6_50 detection',
        false,
        'NOT ATTEMPTED — react-native-fast-opencv has no aruco bindings ' +
          '(its ObjectDetection.ts only exposes matchTemplate/phaseCorrelate; ' +
          'no ArucoDetector/detectMarkers anywhere in the package).\n' +
          'Fallback decision (see task-2.1-report.md): use a pure-JS ArUco ' +
          'decoder (e.g. js-aruco2) fed by a grayscale buffer pulled via ' +
          "OpenCV's matToBuffer, wired up in Task 2.2. Python reference " +
          'detector (experiments/, DICT_6X6_50) confirms all 3 bundled ' +
          'fixtures contain marker id 0.'
      );
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>Task 2.1 native spike</Text>
      <Text style={styles.subtitle}>
        Requires the dev client — this screen will crash-loop in Expo Go.
      </Text>
      {lines.length === 0 && <Text>Running...</Text>}
      {lines.map((l, i) => (
        <View key={i} style={styles.block}>
          <Text style={[styles.label, l.ok ? styles.ok : styles.fail]}>
            {l.ok ? 'OK' : 'FAIL'} — {l.label}
          </Text>
          <Text style={styles.detail}>{l.detail}</Text>
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
  },
  label: {
    fontWeight: '600',
    marginBottom: 4,
  },
  ok: {
    color: '#2a9d3f',
  },
  fail: {
    color: '#c0392b',
  },
  detail: {
    fontFamily: 'monospace',
    fontSize: 12,
  },
});
