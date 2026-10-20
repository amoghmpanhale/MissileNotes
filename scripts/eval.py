"""Scan a folder and write a metrics CSV plus a visual contact sheet.

    python scripts/eval.py [FOLDER] -o OUTDIR [--weights PATH] [--no-binarize]

FOLDER defaults to ``experiments/data/samples``. Failures are recorded rather
than aborting the run, so a partially bad folder still produces a usable
report.
"""

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from pathlib import Path

# Running a script puts scripts/ on sys.path, not the repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pillow_heif
from PIL import Image, ImageDraw, ImageOps

from missilenotes.pipeline import ScanResult, debug_overlay, scan
from missilenotes.segment import Segmenter

DEFAULT_FOLDER = ROOT / "experiments/data/samples"
DEFAULT_WEIGHTS = (
    ROOT / "experiments/model_horizontal/fairscan-segmentation-model.tflite"
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".heic", ".heif"}
STAGES = ("marker", "predict", "geometry", "warp", "binarize")
THUMB = (260, 340)
PAD = 12
LABEL_H = 38


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", nargs="?", type=Path, default=DEFAULT_FOLDER)
    parser.add_argument("-o", "--out", type=Path, default=Path("eval_out"))
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--no-binarize", action="store_true")
    args = parser.parse_args(argv)

    if not args.folder.is_dir():
        print(f"folder not found: {args.folder}", file=sys.stderr)
        return 2
    files = sorted(p for p in args.folder.iterdir() if _is_image(p))
    if not files:
        print(f"no images found in {args.folder}", file=sys.stderr)
        return 1
    if not args.weights.is_file():
        print(f"weights not found: {args.weights}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    segmenter = Segmenter(args.weights)

    rows: list[dict] = []
    entries: list[tuple[Path, np.ndarray | None, ScanResult | None]] = []
    for path in files:
        try:
            image = _load_rgb(path)
        except Exception as exc:
            print(f"{path.name}: FAIL | cannot read image: {exc}")
            rows.append(_row(path, False, f"cannot read image: {exc}", 0.0, {}))
            entries.append((path, None, None))
            continue

        result = scan(image, segmenter, binarize_output=not args.no_binarize)
        total_ms = sum(result.timings.values())
        status = "ok" if result.ok else "FAIL"
        print(
            f"{path.name}: {status} | {result.reason} | "
            f"conf={result.confidence:.2f} | {total_ms:.0f} ms"
        )
        rows.append(
            _row(
                path,
                result.ok,
                result.reason,
                result.confidence,
                result.timings,
            )
        )
        entries.append((path, image, result))

    _write_csv(args.out / "results.csv", rows)
    _contact_sheet(entries, args.out / "contact_sheet.png")
    print(f"wrote {args.out / 'results.csv'} and {args.out / 'contact_sheet.png'}")
    return 0


def _row(
    path: Path,
    ok: bool,
    reason: str,
    confidence: float,
    timings: dict[str, float],
) -> dict:
    row = {
        "file": path.name,
        "ok": int(ok),
        "reason": reason,
        "confidence": f"{confidence:.4f}",
    }
    for stage in STAGES:
        row[stage] = f"{timings.get(stage, 0.0):.2f}"
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _contact_sheet(
    entries: list[tuple[Path, np.ndarray | None, ScanResult | None]],
    path: Path,
) -> None:
    width = 2 * THUMB[0] + 3 * PAD
    row_h = LABEL_H + THUMB[1] + PAD
    sheet = Image.new("RGB", (width, len(entries) * row_h + PAD), "white")
    draw = ImageDraw.Draw(sheet)

    y = PAD
    for name, image, result in entries:
        ok = bool(result and result.ok)
        status = "OK" if ok else "FAIL"
        reason = result.reason if result else "image could not be loaded"
        draw.text((PAD, y), f"{name.name}  [{status}]", fill=(20, 20, 20))
        draw.text(
            (PAD, y + 16),
            textwrap.shorten(reason, width=72, placeholder="..."),
            fill=(110 if ok else 170, 40 if ok else 0, 40 if ok else 0),
        )

        left = debug_overlay(image, result) if result else image
        sheet.paste(_thumb(left, "no image"), (PAD, y + LABEL_H))

        scan_panel = _scan_panel(result)
        sheet.paste(
            scan_panel, (PAD + THUMB[0] + PAD, y + LABEL_H)
        )
        y += row_h

    sheet.save(path)


def _scan_panel(result: ScanResult | None) -> Image.Image:
    if result is None or not result.ok:
        return _placeholder("scan unavailable")
    output = result.scan if result.scan is not None else result.flat_rgb
    if output is None:
        return _placeholder("scan unavailable")
    return _thumb(output, "scan")


def _thumb(image: np.ndarray | None, label: str) -> Image.Image:
    if image is None:
        return _placeholder(label)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=2)
    panel = Image.fromarray(image).convert("RGB")
    panel.thumbnail(THUMB, Image.LANCZOS)
    canvas = Image.new("RGB", THUMB, (238, 238, 238))
    canvas.paste(
        panel,
        ((THUMB[0] - panel.width) // 2, (THUMB[1] - panel.height) // 2),
    )
    return canvas


def _placeholder(label: str) -> Image.Image:
    tile = Image.new("RGB", THUMB, (238, 238, 238))
    draw = ImageDraw.Draw(tile)
    draw.text(
        (PAD, PAD),
        textwrap.fill(label, width=32),
        fill=(150, 150, 150),
    )
    return tile


def _load_rgb(path: Path) -> np.ndarray:
    pillow_heif.register_heif_opener()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return np.asarray(image.convert("RGB"))


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


if __name__ == "__main__":
    raise SystemExit(main())
