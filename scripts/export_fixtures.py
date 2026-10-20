"""Export shared Python/Dart parity fixtures.

The Dart port is checked against the Python reference on the same synthetic
photos.  Each fixture contains the photo, the real model's probability map in
the horizontal model frame, the Python scan result as JSON, and the expected
binarized scan.

The probability PNG is 8-bit, so the Python result is computed from the same
quantised map the Dart ``FakeSegmenter`` will return.  That keeps the two sides
comparable without hiding a rounding difference at the 0.5 threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from missilenotes.geometry import order_corners  # noqa: E402
from missilenotes.marker import (  # noqa: E402
    detect_marker,
    rotate_image,
    upright_rotation,
)
from missilenotes.pipeline import scan  # noqa: E402
from tests.synth import synthetic_page  # noqa: E402

WEIGHTS = ROOT / "experiments/model_horizontal/fairscan-segmentation-model.tflite"
DEFAULT_OUT = ROOT / "app/test/fixtures"

# Rotation is CCW in the photo.  Case 5 is the 20% keystone, case 6 has no
# marker so the pipeline must fail before touching the model.
CASES = (
    {"index": 1, "rotation": 0, "tilt": 7.0, "seed": 101, "perspective": None},
    {"index": 2, "rotation": 90, "tilt": -6.0, "seed": 202, "perspective": None},
    {"index": 3, "rotation": 180, "tilt": 5.0, "seed": 303, "perspective": None},
    {"index": 4, "rotation": 270, "tilt": -8.0, "seed": 404, "perspective": None},
    {"index": 5, "rotation": 0, "tilt": 4.0, "seed": 505, "perspective": 0.2},
    {
        "index": 6,
        "rotation": 0,
        "tilt": 6.0,
        "seed": 606,
        "perspective": None,
        "with_marker": False,
    },
)


class _QuantisedSegmenter:
    """Wrap the real model and quantise exactly like the 8-bit fixture PNG."""

    def __init__(self, inner):
        self.inner = inner

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        prob = np.asarray(self.inner.predict(img_rgb), np.float32)
        return (np.rint(prob * 255.0) / 255.0).astype(np.float32)


def _resize_longest(img: np.ndarray, longest: int) -> np.ndarray:
    height, width = img.shape[:2]
    scale = longest / max(height, width)
    if scale >= 1.0:
        return img
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def _trim_low_bits(img: np.ndarray, keep_bits: int = 6) -> np.ndarray:
    """Drop imperceptible low colour bits so the committed PNGs stay small.

    Sensor noise lives in the low bits and dominates PNG size.  The Python
    result is computed from this same trimmed photo, so parity is unaffected.
    """
    mask = (0xFF << (8 - keep_bits)) & 0xFF
    return (img & mask).astype(np.uint8)


def _model_frame(img_rgb: np.ndarray):
    """The frame the horizontal model sees, plus the upright rotation.

    Mirrors the first half of ``missilenotes.pipeline.scan``; the exporter
    reuses it so the saved probability map has exactly the orientation the
    Dart ``FakeSegmenter`` is handed at test time.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    marker = detect_marker(gray)
    if marker is None:
        return None, 0
    rotation = upright_rotation(marker)
    upright = rotate_image(img_rgb, rotation)
    return rotate_image(upright, -90), rotation


def _write_png(path: Path, image: np.ndarray, compression: int = 9) -> None:
    ok, encoded = cv2.imencode(
        ".png", image, [cv2.IMWRITE_PNG_COMPRESSION, compression]
    )
    if not ok:
        raise RuntimeError(f"could not encode {path}")
    path.write_bytes(encoded.tobytes())


def _json_quad(quad: np.ndarray | None):
    if quad is None:
        return None
    return [[float(x), float(y)] for x, y in order_corners(quad)]


def export_fixtures(
    out_dir: Path | str = DEFAULT_OUT,
    *,
    size: int = 760,
    compression: int = 9,
) -> list[Path]:
    """Write all fixtures into ``out_dir`` and return their JSON paths."""
    if not WEIGHTS.is_file():
        raise FileNotFoundError(
            f"{WEIGHTS} is missing; run `git lfs pull` first"
        )

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from missilenotes.segment import Segmenter

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("fixture_*"):
        old.unlink()

    segmenter = _QuantisedSegmenter(Segmenter(WEIGHTS))
    written: list[Path] = []

    for case in CASES:
        index = case["index"]
        synth = synthetic_page(
            rotation=case["rotation"],
            tilt=case["tilt"],
            seed=case["seed"],
            with_marker=case.get("with_marker", True),
            perspective=case["perspective"],
        )
        photo = _trim_low_bits(_resize_longest(synth.image, size))
        photo_path = out_dir / f"fixture_{index}_photo.png"
        _write_png(photo_path, photo, compression)

        model_img, rotation = _model_frame(photo)
        prob_model = None
        prob_path = None
        if model_img is not None:
            prob_model = np.asarray(segmenter.predict(model_img), np.float32)
            prob_u8 = np.clip(np.rint(prob_model * 255.0), 0, 255).astype(np.uint8)
            prob_path = out_dir / f"fixture_{index}_prob_model.png"
            _write_png(prob_path, prob_u8, compression)

        result = scan(photo, segmenter)
        expected_path = None
        if result.scan is not None:
            expected_path = out_dir / f"fixture_{index}_expected.png"
            _write_png(expected_path, result.scan, compression)

        payload = {
            "index": index,
            "photo": photo_path.name,
            "photo_shape": [int(photo.shape[0]), int(photo.shape[1]), 3],
            "prob_model": prob_path.name if prob_path else None,
            "model_shape": (
                [int(prob_model.shape[0]), int(prob_model.shape[1])]
                if prob_model is not None
                else None
            ),
            "expected_scan": expected_path.name if expected_path else None,
            "ok": bool(result.ok),
            "reason": result.reason,
            "rotation": int(result.rotation),
            "quad": _json_quad(result.quad),
            "confidence": float(result.confidence),
            "scan_shape": (
                [int(result.scan.shape[0]), int(result.scan.shape[1])]
                if result.scan is not None
                else None
            ),
            "flat_shape": (
                [int(x) for x in result.flat_rgb.shape]
                if result.flat_rgb is not None
                else None
            ),
            "threshold": 0.5,
        }
        json_path = out_dir / f"fixture_{index}.json"
        json_path.write_text(json.dumps(payload, indent=2) + "\n")
        written.append(json_path)
        print(
            f"fixture {index}: ok={payload['ok']} rotation={payload['rotation']} "
            f"reason={payload['reason']!r}"
        )

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--size", type=int, default=760)
    parser.add_argument("--compression", type=int, default=9)
    args = parser.parse_args(argv)

    written = export_fixtures(
        args.out, size=args.size, compression=args.compression
    )
    total = sum(p.stat().st_size for p in args.out.glob("fixture_*"))
    print(f"wrote {len(written)} fixtures to {args.out} ({total / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
