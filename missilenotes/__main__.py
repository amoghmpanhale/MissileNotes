"""Command line: ``python -m missilenotes scan IN [IN...] -o OUTDIR``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageOps

from .pipeline import debug_overlay, scan
from .segment import Segmenter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = (
    ROOT / "experiments/model_horizontal/fairscan-segmentation-model.tflite"
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".heic", ".heif"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m missilenotes")
    sub = parser.add_subparsers(dest="command", required=True)

    scan_parser = sub.add_parser("scan", help="scan photos of marked pages")
    scan_parser.add_argument("inputs", nargs="+", type=Path)
    scan_parser.add_argument("-o", "--out", required=True, type=Path)
    scan_parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    scan_parser.add_argument("--no-binarize", action="store_true")
    scan_parser.add_argument(
        "--normalize",
        action="store_true",
        help="flatten uneven lighting and cast shadows before binarizing",
    )
    scan_parser.add_argument(
        "--remove-rules",
        action="store_true",
        help="remove printed ruled/margin lines from the binarized scan",
    )
    scan_parser.add_argument("--debug", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "scan":
        return _run_scan(args)
    return 2


def _run_scan(args) -> int:
    if not args.weights.is_file():
        print(f"weights not found: {args.weights}", file=sys.stderr)
        return 2

    files: list[Path] = []
    failed = False
    for path in args.inputs:
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if _is_image(p)))
        elif _is_image(path):
            files.append(path)
        else:
            print(f"{path}: not an image or directory", file=sys.stderr)
            failed = True

    if not files:
        print("no images found", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    segmenter = Segmenter(args.weights)

    for path in files:
        try:
            image = _load_rgb(path)
        except Exception as exc:  # a corrupt photo must not stop the batch
            print(f"{path.name}: FAIL | cannot read image: {exc} | conf=0.00 | 0 ms")
            failed = True
            continue

        result = scan(
            image,
            segmenter,
            binarize_output=not args.no_binarize,
            normalize=args.normalize,
            remove_rules=args.remove_rules,
        )
        total_ms = sum(result.timings.values())
        status = "ok" if result.ok else "FAIL"
        print(
            f"{path.name}: {status} | {result.reason} | "
            f"conf={result.confidence:.2f} | {total_ms:.0f} ms"
        )

        if result.ok:
            output = result.scan if result.scan is not None else result.flat_rgb
            Image.fromarray(output).save(args.out / f"{path.stem}_scan.png")
        if args.debug:
            overlay = debug_overlay(image, result)
            Image.fromarray(overlay).save(args.out / f"{path.stem}_debug.png")

        failed = failed or not result.ok

    return 1 if failed else 0


def _load_rgb(path: Path) -> np.ndarray:
    pillow_heif.register_heif_opener()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return np.asarray(image.convert("RGB"))


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


if __name__ == "__main__":
    raise SystemExit(main())
