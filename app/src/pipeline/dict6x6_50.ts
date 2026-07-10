// GENERATED — do not hand-edit. Source of truth is
// experiments/export_goldens.py's export_dict6x6_50(), which renders each of
// OpenCV's 50 canonical DICT_6X6_50 markers (cv2.aruco.generateImageMarker)
// and samples the interior 6x6 cell grid the same way js-aruco2's detector
// does (mean intensity of each cell vs. threshold, row-major), so bit-order
// matches js-aruco2's AR.Dictionary.find() bit-for-bit. Regenerate via
// `python3 experiments/export_goldens.py`, which writes
// app/src/pipeline/__fixtures__/goldens/dict6x6_50.json; this file just wraps
// that JSON with the shape js-aruco2's AR.DICTIONARIES entries expect
// ({ nBits, codeList }), and adds our own dictionary name.
//
// Verified against ground truth: app/src/pipeline/aruco.test.ts feeds the
// three bundled fixtures through js-aruco2 with this exact codebook and
// asserts the detected marker id matches the OpenCV-detected id from
// __fixtures__/goldens/*.marker.json (all three: id 0, hammingDistance 0,
// corners within ~1px of OpenCV's).
import codebook from './__fixtures__/goldens/dict6x6_50.json';

export const DICT_6X6_50_NAME = 'DICT_6X6_50';

export const DICT_6X6_50 = {
  nBits: codebook.nBits,
  codeList: codebook.codeList,
};
