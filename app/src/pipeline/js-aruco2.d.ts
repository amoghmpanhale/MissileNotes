// js-aruco2 ships no TypeScript types (plain CommonJS, see
// node_modules/js-aruco2/src/aruco.js). This declares only the surface this
// app actually calls: a custom-dictionary detector that takes a width/height
// RGBA buffer and returns marker candidates with id, corners, and Hamming
// distance to the nearest codebook entry.
declare module 'js-aruco2' {
  export namespace AR {
    const DICTIONARIES: Record<string, { nBits: number; tau?: number; codeList: (number | string)[] }>;

    class Detector {
      constructor(config?: { dictionaryName?: string; maxHammingDistance?: number });
      detect(image: { width: number; height: number; data: Uint8ClampedArray | Uint8Array }): {
        id: number;
        hammingDistance: number;
        corners: { x: number; y: number }[];
      }[];
    }
  }
}
