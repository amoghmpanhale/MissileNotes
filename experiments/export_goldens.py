"""Golden-parity harness for Task 2.2 (on-device pipeline port).

Runs the Python reference pipeline (segment_pages_horizontal's frame
handling: upright -> rotate 90 CW into model frame -> infer -> rotate mask
back -> keep_marker_island -> page_quad -> warp -> grayscale) against the
bundled app fixtures, and writes goldens the TS port can be checked against:

  app/src/pipeline/__fixtures__/goldens/<name>.quad.json    - 4 ordered
      corners (TL,TR,BR,BL) in the fixture's own pixel coordinates
  app/src/pipeline/__fixtures__/goldens/<name>.warped.png   - flattened,
      grayscale output (matches segment_pages_horizontal.py's preview)
  app/src/pipeline/__fixtures__/goldens/<name>.marker.json  - OpenCV-detected
      ArUco marker id + its 4 corner pixel coords (TL,TR,BR,BL, marker's own
      frame) in the upright image
  app/src/pipeline/__fixtures__/goldens/dict6x6_50.json     - the DICT_6X6_50
      codebook in js-aruco2's dictionary format ({ nBits, codeList }), so
      app/src/pipeline/dict6x6_50.ts is generated from ground truth rather
      than hand-typed.

Run from repo root: `python3 experiments/export_goldens.py`
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import experiments.segment_pages as sp  # noqa: E402

# Same swap segment_pages_horizontal.py does: point load_model() at the
# horizontal weights instead of forking the whole module.
sp.MODEL = str(REPO_ROOT / "experiments" / "model_horizontal" / "fairscan-segmentation-model.pt")

FIXTURES_DIR = REPO_ROOT / "app" / "src" / "pipeline" / "__fixtures__"
GOLDENS_DIR = FIXTURES_DIR / "goldens"
FIXTURE_NAMES = ["IMG_4905", "IMG_4906", "IMG_4907"]


def order_corners_list(quad):
    """order_corners but returns a plain list of [x, y] pairs for JSON."""
    ordered = sp.order_corners(quad)
    return [[float(x), float(y)] for x, y in ordered]


def export_fixture(model, name):
    path = FIXTURES_DIR / f"{name}.png"
    img = sp.upright(Image.open(path).convert("RGB"))  # upright frame
    horiz = img.rotate(-90, expand=True)  # 90 deg CW -> model frame

    mask = sp.mask_for(model, horiz)  # mask in horizontal (model) frame
    mask = np.rot90(mask)  # CCW 90 -> back to upright frame
    mask = sp.keep_marker_island(mask, img)
    quad = sp.page_quad(mask, img)

    corners, ids, _ = sp._ARUCO.detectMarkers(
        cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)
    )
    marker_id = int(np.asarray(ids).reshape(-1)[0]) if ids is not None else None
    marker_corners = corners[0][0].tolist() if ids is not None else None

    quad_json = order_corners_list(quad) if quad is not None else None

    scan = sp.warp(img, quad) if quad is not None else np.asarray(img)
    gray = (scan @ [0.299, 0.587, 0.114]).astype(np.uint8)

    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(gray).save(GOLDENS_DIR / f"{name}.warped.png")
    with open(GOLDENS_DIR / f"{name}.quad.json", "w") as f:
        json.dump({"corners": quad_json}, f, indent=2)
    with open(GOLDENS_DIR / f"{name}.marker.json", "w") as f:
        json.dump({"id": marker_id, "corners": marker_corners}, f, indent=2)

    print(
        f"{name}: marker id={marker_id} quad={'found' if quad_json else 'NONE'} "
        f"warped={gray.shape}"
    )


def export_dict6x6_50():
    """Emit the DICT_6X6_50 codebook in js-aruco2's dictionary format.

    js-aruco2's detector reads a warped, Otsu-thresholded grayscale patch of
    the marker, samples the interior N x N cells (skipping the black border),
    and calls a cell "1" if its mean intensity is above threshold (mostly
    white), row-major (row i outer, column j inner) - see getMarker() in
    node_modules/js-aruco2/src/aruco.js. To match that convention exactly (so
    bit-order lines up bit-for-bit) we render each of OpenCV's 50 canonical
    markers with cv2.aruco.generateImageMarker, sample the same interior grid
    the same way, and flatten it the same way js-aruco2 does -- rather than
    reading cv2's internal byte-packed bit layout, which uses its own
    (different, undocumented-in-python) packing order.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    n_bits = 36
    mark_size = 6  # sqrt(36)
    cell_px = 10
    side_px = (mark_size + 2) * cell_px  # +2 for the 1-module black border

    code_list = []
    for marker_id in range(50):
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
        bits = []
        for i in range(mark_size):
            row_bits = []
            for j in range(mark_size):
                y0 = (i + 1) * cell_px
                x0 = (j + 1) * cell_px
                cell = img[y0 : y0 + cell_px, x0 : x0 + cell_px]
                row_bits.append(1 if cell.mean() > 127 else 0)
            bits.append(row_bits)
        bitstring = "".join(str(b) for row in bits for b in row)
        code_list.append(hex(int(bitstring, 2)))

    GOLDENS_DIR.mkdir(parents=True, exist_ok=True)
    with open(GOLDENS_DIR / "dict6x6_50.json", "w") as f:
        json.dump({"nBits": n_bits, "codeList": code_list}, f, indent=2)
    print(f"dict6x6_50: wrote {len(code_list)} codes, nBits={n_bits}")


def main():
    export_dict6x6_50()
    model = sp.load_model()
    for name in FIXTURE_NAMES:
        export_fixture(model, name)


if __name__ == "__main__":
    main()
