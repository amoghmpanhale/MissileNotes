// Crux test for the option-A architecture decision: react-native-fast-opencv
// has no ArUco bindings, so marker detection runs on js-aruco2 (pure JS) with
// a custom DICT_6X6_50 codebook generated from OpenCV ground truth
// (dict6x6_50.ts / experiments/export_goldens.py). This proves that codebook
// + bit-ordering are correct by decoding the same bundled fixture photos
// OpenCV decoded, in pure Node (no native modules — fast-opencv/fast-tflite
// cannot run outside the app, so scan.ts's native path is NOT imported here).
import fs from 'fs';
import path from 'path';
import { PNG } from 'pngjs';

import { detectMarker, uprightAngle } from './aruco';

const FIXTURES_DIR = path.join(__dirname, '__fixtures__');
const GOLDENS_DIR = path.join(FIXTURES_DIR, 'goldens');
const FIXTURE_NAMES = ['IMG_4905', 'IMG_4906', 'IMG_4907'];

function loadPngAsRgba(name: string) {
  const png = PNG.sync.read(fs.readFileSync(path.join(FIXTURES_DIR, `${name}.png`)));
  return { data: png.data, width: png.width, height: png.height };
}

function loadGoldenMarker(name: string): { id: number; corners: [number, number][] } {
  return JSON.parse(fs.readFileSync(path.join(GOLDENS_DIR, `${name}.marker.json`), 'utf8'));
}

describe('detectMarker (js-aruco2 + DICT_6X6_50 codebook) vs. OpenCV goldens', () => {
  it.each(FIXTURE_NAMES)('%s: detected id matches OpenCV-detected id', (name) => {
    const { data, width, height } = loadPngAsRgba(name);
    const golden = loadGoldenMarker(name);

    const detected = detectMarker(data, width, height);

    expect(detected).not.toBeNull();
    expect(detected!.id).toBe(golden.id);
  });

  it.each(FIXTURE_NAMES)('%s: detected corners are within ~10px of OpenCV goldens', (name) => {
    const { data, width, height } = loadPngAsRgba(name);
    const golden = loadGoldenMarker(name);

    const detected = detectMarker(data, width, height)!;

    for (let i = 0; i < 4; i++) {
      const [dx, dy] = detected.corners[i];
      const [gx, gy] = golden.corners[i];
      const dist = Math.hypot(dx - gx, dy - gy);
      expect(dist).toBeLessThanOrEqual(10);
    }
  });
});

describe('uprightAngle', () => {
  // uprightAngle only reads corners[0] (TL) and corners[3] (BL) — TR/BR are
  // filled with placeholder points since the function under test ignores
  // them. marker-up = TL - BL, in y-down pixel coords (screen space).
  const markerWithUp = (ux: number, uy: number): [number, number][] => {
    const bl: [number, number] = [50, 50];
    const tl: [number, number] = [50 + ux, 50 + uy];
    return [tl, [0, 0], [0, 0], bl];
  };

  // Expected values cross-checked against segment_pages.py's own formula:
  // round((90 - degrees(atan2(uy, -ux))) / 90) * 90 % 360.
  it('marker-up = screen-up -> page needs 180', () => {
    expect(uprightAngle(markerWithUp(0, -20))).toBe(180);
  });

  it('marker-up = screen-right -> page needs 270', () => {
    expect(uprightAngle(markerWithUp(20, 0))).toBe(270);
  });

  it('marker-up = screen-down -> page needs 0 (already upright)', () => {
    expect(uprightAngle(markerWithUp(0, 20))).toBe(0);
  });

  it('marker-up = screen-left -> page needs 90', () => {
    expect(uprightAngle(markerWithUp(-20, 0))).toBe(90);
  });
});
