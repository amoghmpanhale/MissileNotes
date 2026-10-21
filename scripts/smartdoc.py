"""Benchmark the page quad on ICDAR'15 SmartDoc Challenge 1.

SmartDoc frames are phone-video shots of printed documents with ground-truth
page corners, but no ArUco marker. Two modes are scored on the same frames:

- ``nomarker``: ``scan(rotation=...)`` with the upright rotation taken from the
  ground truth (the marker's only job there), so the quad comes from
  segmentation and geometry alone.
- ``marker``: a real DICT_6X6_50 marker is warped onto the page's top-right
  corner with the ground-truth homography, then the unmodified pipeline runs.

The score is the competition's Jaccard index, computed in the document's own
reference frame (0.1 mm units) so frames shot at different distances weigh the
same. A failed scan scores 0, as in the competition.

    python scripts/smartdoc.py fetch
    python scripts/smartdoc.py run --step 10 -o eval_out/smartdoc

Dataset: CC BY 4.0, Burie et al., "ICDAR2015 Competition on Smartphone
Document Capture and OCR (SmartDoc)", ICDAR 2015.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import multiprocessing as mp
import re
import statistics
import sys
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from missilenotes.pipeline import scan  # noqa: E402
from missilenotes.segment import Segmenter  # noqa: E402

RELEASE = "https://github.com/jchazalon/smartdoc15-ch1-dataset/releases/download/v2.0.0"
FRAMES_SHA256 = "3acb8be143fc86c507d90d298097cba762e91a3abf7e2d35ccd5303e13a79eae"
DEFAULT_DATA = ROOT / "experiments/data/smartdoc"
DEFAULT_WEIGHTS = (
    ROOT / "experiments/model_horizontal/fairscan-segmentation-model.tflite"
)
MODES = ("nomarker", "marker")

# Composited marker, in document units (0.1 mm): 30 mm of bits, a 5 mm white
# border, 10 mm in from the page's top-right corner.
MARKER_ID = 7
MARKER_BITS = 300
MARKER_BORDER = 50
MARKER_MARGIN = 100
_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)


# --- ground truth and scoring -------------------------------------------------


def gt_quad(row: dict) -> np.ndarray:
    """Ground-truth corners as (4, 2) TL/TR/BR/BL in frame pixels."""
    return np.array(
        [[float(row[f"{c}_x"]), float(row[f"{c}_y"])] for c in ("tl", "tr", "br", "bl")],
        np.float64,
    )


def gt_rotation(quad: np.ndarray) -> int:
    """CCW degrees (0/90/180/270) that turn the page upright.

    Same quantisation as ``marker.upright_rotation``: rotate the page's up
    vector (BL -> TL) to point up (-y) on screen.
    """
    ux, uy = quad[0] - quad[3]
    angle = np.degrees(np.arctan2(uy, ux))
    return int(round((angle + 90.0) / 90.0)) * 90 % 360


def doc_homography(quad: np.ndarray, width: float, height: float) -> np.ndarray:
    """Homography from frame pixels to the document's reference frame."""
    rect = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float32)
    return cv2.getPerspectiveTransform(quad.astype(np.float32), rect)


def jaccard(
    detected: np.ndarray | None, gt: np.ndarray, width: float, height: float
) -> float:
    """SmartDoc Jaccard index of a detected quad, in the document frame."""
    if detected is None:
        return 0.0
    h = doc_homography(gt, width, height)
    pts = cv2.perspectiveTransform(
        np.asarray(detected, np.float32).reshape(-1, 1, 2), h
    ).reshape(-1, 2)
    if not np.isfinite(pts).all():
        return 0.0
    # A projected convex quad stays convex unless it crosses the horizon,
    # which only a wildly wrong detection does; the hull keeps that scorable.
    poly = cv2.convexHull(pts.astype(np.float32))
    rect = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], np.float32
    ).reshape(-1, 1, 2)
    inter, _ = cv2.intersectConvexConvex(poly, rect)
    union = cv2.contourArea(poly) + width * height - inter
    return float(inter / union) if union > 0 else 0.0


# --- marker compositing -------------------------------------------------------


def composite_marker(
    image: np.ndarray, quad: np.ndarray, width: float, height: float
) -> np.ndarray:
    """Warp a printed marker onto the page's top-right corner.

    Printed upside-down like the real one: page-up is marker-down.
    """
    bits = cv2.aruco.generateImageMarker(_DICT, MARKER_ID, MARKER_BITS)
    tile = cv2.copyMakeBorder(
        bits, *[MARKER_BORDER] * 4, cv2.BORDER_CONSTANT, value=255
    )
    tile = cv2.rotate(tile, cv2.ROTATE_180)
    side = tile.shape[0]
    x0 = width - MARKER_MARGIN - side
    y0 = MARKER_MARGIN

    tile_to_doc = np.array([[1, 0, x0], [0, 1, y0], [0, 0, 1]], np.float64)
    doc_to_frame = np.linalg.inv(doc_homography(quad, width, height))
    h = doc_to_frame @ tile_to_doc
    size = (image.shape[1], image.shape[0])

    warped = cv2.warpPerspective(tile, h, size, flags=cv2.INTER_AREA)
    alpha = cv2.warpPerspective(
        np.full_like(tile, 255), h, size, flags=cv2.INTER_AREA
    ).astype(np.float32)[..., None] / 255.0
    out = image.astype(np.float32) * (1 - alpha) + warped[..., None] * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


# --- dataset ------------------------------------------------------------------


def fetch(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / "frames.tar.gz"
    if not archive.is_file():
        print(f"downloading {RELEASE}/frames.tar.gz (~1 GB) ...")
        urllib.request.urlretrieve(f"{RELEASE}/frames.tar.gz", archive)
    digest = hashlib.sha256()
    with archive.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != FRAMES_SHA256:
        raise SystemExit(f"checksum mismatch for {archive}; delete it and retry")
    if not (dest / "metadata.csv.gz").is_file():
        print("extracting ...")
        with tarfile.open(archive) as tar:
            tar.extractall(dest, filter="data")
    print(f"SmartDoc frames ready in {dest}")


def load_rows(data: Path, step: int) -> list[dict]:
    with gzip.open(data / "metadata.csv.gz", "rt", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if (int(r["frame_index"]) - 1) % step == 0]


# --- evaluation ---------------------------------------------------------------

_SEGMENTER = None
_DATA: Path | None = None


def _init_worker(weights: str, data: str) -> None:
    global _SEGMENTER, _DATA
    cv2.setNumThreads(1)
    _SEGMENTER = Segmenter(weights)
    _DATA = Path(data)


def _evaluate(row: dict) -> list[dict]:
    image = np.asarray(Image.open(_DATA / row["image_path"]).convert("RGB"))
    gt = gt_quad(row)
    w, h = float(row["model_width"]), float(row["model_height"])
    out = []
    for mode in MODES:
        if mode == "nomarker":
            result = scan(
                image, _SEGMENTER, rotation=gt_rotation(gt), binarize_output=False
            )
        else:
            result = scan(
                composite_marker(image, gt, w, h), _SEGMENTER, binarize_output=False
            )
        out.append(
            {
                "mode": mode,
                "image_path": row["image_path"],
                "background": row["bg_name"],
                "model": row["model_name"],
                "modeltype": row["modeltype_name"],
                "ok": int(result.ok),
                "jaccard": round(jaccard(result.quad, gt, w, h), 5),
                "reason": result.reason,
                "quad": " ".join(f"{v:.1f}" for v in np.ravel(result.quad))
                if result.quad is not None
                else "",
                "ms": round(sum(result.timings.values()), 1),
            }
        )
    return out


def _summary(results: list[dict], step: int, frames: int) -> str:
    lines = [
        "# SmartDoc 2015 Challenge 1",
        "",
        f"{frames} frames (every {step}th of 24,889). Jaccard index in the "
        "document frame; failed scans score 0.",
        "",
        "| mode | mean JI | median JI | failed | JI < 0.9 |",
        "|---|---|---|---|---|",
    ]
    by_bg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_type: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for mode in MODES:
        rows = [r for r in results if r["mode"] == mode]
        ji = [r["jaccard"] for r in rows]
        failed = sum(1 - r["ok"] for r in rows)
        low = sum(j < 0.9 for j in ji)
        lines.append(
            f"| {mode} | {statistics.fmean(ji):.4f} | {statistics.median(ji):.4f} "
            f"| {failed / len(ji):.1%} | {low / len(ji):.1%} |"
        )
        for r in rows:
            by_bg[r["background"]][mode].append(r["jaccard"])
            by_type[r["modeltype"]][mode].append(r["jaccard"])

    for title, groups in (("Background", by_bg), ("Document type", by_type)):
        lines += ["", f"| {title.lower()} | " + " | ".join(MODES) + " |",
                  "|---|" + "---|" * len(MODES)]
        for key in sorted(groups):
            cells = " | ".join(f"{statistics.fmean(groups[key][m]):.4f}" for m in MODES)
            lines.append(f"| {key} | {cells} |")

    reasons: dict[str, int] = defaultdict(int)
    for r in results:
        if not r["ok"]:
            reason = re.sub(r" \(IoU [\d.]+\)", "", r["reason"])
            reasons[f"{r['mode']}: {reason}"] += 1
    if reasons:
        lines += ["", "Failure reasons:", ""]
        lines += [f"- {k} ({v})" for k, v in sorted(reasons.items(), key=lambda kv: -kv[1])]
    return "\n".join(lines) + "\n"


def _contact_sheet(results: list[dict], data: Path, path: Path, mode: str) -> None:
    """Worst 8, median 2 and best 2 frames: ground truth blue, detection green."""
    rows = sorted((r for r in results if r["mode"] == mode), key=lambda r: r["jaccard"])
    mid = len(rows) // 2
    picks = rows[:8] + rows[mid : mid + 2] + rows[-2:]
    thumb_w, thumb_h, cols = 480, 270, 3
    sheet = Image.new(
        "RGB", (cols * thumb_w, -(-len(picks) // cols) * (thumb_h + 20)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    meta = {r["image_path"]: r for r in load_rows(data, 1)}
    for i, r in enumerate(picks):
        image = np.asarray(Image.open(data / r["image_path"]).convert("RGB")).copy()
        gt = gt_quad(meta[r["image_path"]])
        cv2.polylines(image, [gt.astype(np.int32)], True, (40, 90, 255), 4)
        if r["quad"]:
            quad = np.array(r["quad"].split(), float).reshape(4, 2)
            cv2.polylines(image, [quad.astype(np.int32)], True, (0, 220, 0), 4)
        tile = Image.fromarray(image).resize((thumb_w, thumb_h))
        x, y = (i % cols) * thumb_w, (i // cols) * (thumb_h + 20)
        sheet.paste(tile, (x, y + 20))
        draw.text((x + 4, y + 4), f"JI {r['jaccard']:.3f}  {r['image_path']}", fill="black")
    sheet.save(path)


def run(data: Path, out: Path, weights: Path, step: int, workers: int) -> None:
    rows = load_rows(data, step)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(rows)} frames x {len(MODES)} modes on {workers} workers")
    results: list[dict] = []
    with mp.Pool(workers, _init_worker, (str(weights), str(data))) as pool:
        for i, frame in enumerate(pool.imap(_evaluate, rows, chunksize=4), 1):
            results.extend(frame)
            if i % 100 == 0:
                print(f"  {i}/{len(rows)}", flush=True)

    with (out / "results.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    summary = _summary(results, step, len(rows))
    (out / "summary.md").write_text(summary)
    for mode in MODES:
        _contact_sheet(results, data, out / f"contact_sheet_{mode}.png", mode)
    print(summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch", help="download and unpack the SmartDoc frames")
    f.add_argument("--data", type=Path, default=DEFAULT_DATA)
    r = sub.add_parser("run", help="score both modes and write a report")
    r.add_argument("--data", type=Path, default=DEFAULT_DATA)
    r.add_argument("-o", "--out", type=Path, default=ROOT / "eval_out/smartdoc")
    r.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    r.add_argument("--step", type=int, default=10, help="use every Nth frame")
    r.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1))
    args = parser.parse_args(argv)

    if args.command == "fetch":
        fetch(args.data)
    else:
        run(args.data, args.out, args.weights, args.step, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
