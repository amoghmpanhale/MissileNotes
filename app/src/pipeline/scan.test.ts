// Pure-geometry unit tests — no native imports (fast-opencv/fast-tflite
// can't run in Node). Imports directly from geometry.ts (not scan.ts, which
// pulls in react-native-fast-opencv/-fast-tflite at module load and would
// crash under plain Node/jest). The other exported helpers in scan.ts
// (pageQuadFromContourPoints, warpQuad, processScan) call into fast-opencv/
// fast-tflite and are exercised on-device via the /parity screen instead
// (see task-2.2-report.md for exact steps).
import { orderCorners } from './geometry';

describe('orderCorners', () => {
  it('orders a shuffled square as TL, TR, BR, BL', () => {
    const tl: [number, number] = [10, 10];
    const tr: [number, number] = [110, 12];
    const br: [number, number] = [108, 112];
    const bl: [number, number] = [8, 109];

    // Feed them in a shuffled order — the function must not depend on input
    // order.
    const shuffled = [br, tl, bl, tr];
    const [outTl, outTr, outBr, outBl] = orderCorners(shuffled);

    expect(outTl).toEqual(tl);
    expect(outTr).toEqual(tr);
    expect(outBr).toEqual(br);
    expect(outBl).toEqual(bl);
  });

  it('handles an axis-aligned rectangle', () => {
    const pts: [number, number][] = [
      [0, 100], // BL
      [0, 0], // TL
      [200, 0], // TR
      [200, 100], // BR
    ];
    const [tl, tr, br, bl] = orderCorners(pts);
    expect(tl).toEqual([0, 0]);
    expect(tr).toEqual([200, 0]);
    expect(br).toEqual([200, 100]);
    expect(bl).toEqual([0, 100]);
  });

  it('throws on anything other than 4 points', () => {
    expect(() => orderCorners([[0, 0], [1, 1], [2, 2]])).toThrow();
  });
});
