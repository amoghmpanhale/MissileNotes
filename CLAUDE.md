# MissileNotes

Cross-platform (iPhone + Android) document-scanning app. The user prints an
ArUco marker, sticks it on a page corner, and photographs the page; the app
finds, flattens, cleans, and cloud-saves the scan. End goal: on-device
scanning + cloud sync + AI cleanup, setup being just a printed marker.

`missilenotes/` is the tested Python reference pipeline; `app/` is the Flutter
app with a Dart port of it (core scan path only — `normalize`/`remove_rules`
are Python-only). The Python package is the source of truth: change behavior
there first, then port and refresh the parity fixtures.

## Layout

- `app/` — Flutter iOS + Android app. `lib/pipeline/` is the Dart port;
  `lib/scan_service.dart` runs scans on a worker isolate; `home_screen.dart` /
  `result_screen.dart` are capture → scan → share. `test/fixtures/` holds
  parity fixtures from `scripts/export_fixtures.py`. Test with
  `flutter test --concurrency=1` (the real-model test skips on Linux).

- `missilenotes/` — the reference pipeline: `marker.py` (ArUco + rotation),
  `geometry.py` (mask → page quad → warp), `enhance.py` (binarize),
  `segment.py` (`.pt`/`.tflite` segmenter), `pipeline.py` (`scan`),
  `__main__.py` (CLI).
- `scripts/eval.py` — folder eval: `results.csv` + `contact_sheet.png`.
- `scripts/smartdoc.py` — ICDAR'15 SmartDoc benchmark (`fetch`, `run`); data
  lands in gitignored `experiments/data/smartdoc/`, reports in `eval_out/`.
  It's a test set: never tune parameters against it.
- `tools/warp_editor.py` — Tk editor for scans the pipeline got wrong.
- `tests/` — pytest suite; `tests/synth.py` generates synthetic pages.
- `experiments/model/`, `experiments/model_horizontal/` — FairScan weights
  (`.pt` = PyTorch, `.tflite` = on-device). **On Git LFS.** The package uses
  the horizontal model.
- `experiments/data/` — printable `aruco.png` / `aruco.svg`. Scans and cached
  outputs are **gitignored**; don't commit them.

## Conventions

- The ArUco dict in use is `DICT_6X6_50`. The marker is printed upside-down
  relative to the upright page, so page-up = marker-down.
- Model contract: 256×256 RGB input. The `.tflite` is NHWC float32
  `1×256×256×3`, with ImageNet normalization applied **outside** the graph and
  raw logits on output (sigmoid in Python). Don't feed 512×512: it produces a
  different mask and breaks parity with the on-device graph.
- The photo is rotated upright via the marker, then 90° CW into the horizontal
  model frame; the mask is rotated back before geometry. Keep points in sync
  with `marker.rotate_points`.
- Models are Git LFS. New `.pt/.pth/.onnx/.tflite` files auto-track via
  `.gitattributes`. Run `git lfs install` once per machine.
- Use the worker venv at `.venv` for tests and scripts. Deps live in
  `pyproject.toml`.
- Core stack: OpenCV (contrib, for `cv2.aruco`), numpy, Pillow, pillow-heif;
  optional torch/segmentation-models-pytorch or ai-edge-litert.

## Don't commit

`.claude/`, `__pycache__/`, `.venv/`, `experiments/data/` scans, `out/`,
`eval_out/`, `*_scan.png`, `*_debug.png`, `.DS_Store` — all gitignored. Keep it
that way.
