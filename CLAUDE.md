# rocket-killer

Cross-platform (iPhone + Android) document-scanning app. The user prints an
ArUco marker, sticks it on a page corner, and photographs the page; the app
finds, flattens, cleans, and cloud-saves the scan. End goal: on-device
scanning + cloud sync + AI cleanup, setup being just a printed marker.

**Today there is no app** — only `experiments/`, a Python CV sandbox that
prototypes the pipeline before it gets ported to mobile.

## Layout

- `experiments/` — Python pipeline scripts (see README for what each does).
- `experiments/model/`, `experiments/model_horizontal/` — FairScan weights
  (`.pt` = PyTorch, `.tflite` = on-device). **On Git LFS.**
- `experiments/data/` — scan images, photos, cached masks. **Gitignored**
  (except the printable `aruco.png` / `aruco.svg`). Don't commit scans.

## Conventions

- The model is expensive: run inference once via `cache_masks.py` and read the
  cached probability PNGs downstream. `refine_page.py` never loads the model.
- The ArUco dict in use is `DICT_6X6_50`. The marker is printed upside-down
  relative to the upright page, so page-up = marker-down.
- Models are Git LFS. New `.pt/.pth/.onnx/.tflite` files auto-track via
  `.gitattributes`. Run `git lfs install` once per machine.
- Deps: `experiments/requirements.txt`. Core stack: OpenCV (contrib, for
  `cv2.aruco`), PyTorch, segmentation-models-pytorch, Pillow.

## Don't commit

`.claude/`, `__pycache__/`, `experiments/data/` scans, `.DS_Store` — all
gitignored. Keep it that way.
