# rocket-killer

Rocketbooks suck. This is better.

A phone app (iPhone + Android) that turns any printed page into a clean,
flat, cloud-saved scan — you just print an [ArUco](https://en.wikipedia.org/wiki/ArUco)
marker, stick it in a page corner, and point your camera. The marker gives the
app a reliable anchor to find the page, get its orientation, and flatten it,
so scans come out square and legible without a flatbed or careful framing.

**Goal:** cross-platform mobile app doing on-device scanning + cloud sync +
the fun AI stuff (page segmentation, deskew, binarization), with the only
setup being a printed marker on the page corners.

## Status

Early. Everything so far lives in [`experiments/`](experiments/) — a Python
sandbox for the computer-vision pipeline that will eventually be ported to the
phone. There is no app yet.

## The pipeline (in `experiments/`)

The FairScan segmentation model (`.pt` PyTorch weights, `.tflite` for on-device)
finds the page; classic OpenCV cleans it up.

1. `heic2png.py` — convert iPhone HEIC captures to PNG.
2. `aruco_keypoints.py` — detect the ArUco marker and its corners.
3. `cache_masks.py` — run the model once per image, cache the probability maps.
4. `segment_pages.py` — segment + auto-rotate a page upright from the marker.
5. `refine_page.py` — snap the blobby mask to a straight page quad using ruled lines.
6. `warp_editor.py` / `binarize_warp_editor.py` — Tk tools to hand-tune the quad and flatten.
7. `binarize.py` — turn the flat grayscale scan into crisp black-ink-on-white.

## Setup

```bash
cd experiments
pip install -r requirements.txt
```

## Models (Git LFS)

The AI models are stored in [Git LFS](https://git-lfs.com/), not in the repo
blob. Install LFS **before** cloning/pulling or you'll get pointer files instead
of real weights:

```bash
git lfs install
git clone <repo-url>          # or, if already cloned: git lfs pull
```
