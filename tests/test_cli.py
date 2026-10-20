"""Subprocess tests for the CLI and the folder eval script."""

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

from tests.synth import synthetic_page

ROOT = Path(__file__).resolve().parents[1]
TFLITE_WEIGHTS = (
    ROOT / "experiments/model_horizontal/fairscan-segmentation-model.tflite"
)
HAS_LITERT = importlib.util.find_spec("ai_edge_litert") is not None
needs_model = pytest.mark.skipif(
    not HAS_LITERT or not TFLITE_WEIGHTS.is_file(),
    reason="ai-edge-litert or the horizontal .tflite weights are unavailable",
)


def _write_synth(path: Path, rotation: int = 0, seed: int = 0) -> Path:
    Image.fromarray(synthetic_page(rotation=rotation, seed=seed).image).save(path)
    return path


def _run(*args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )


@needs_model
def test_cli_scans_synth_image(tmp_path):
    image = _write_synth(tmp_path / "photo.png")
    out = tmp_path / "out"

    proc = _run(
        "-m", "missilenotes", "scan", image, "-o", out,
        "--weights", TFLITE_WEIGHTS, "--debug",
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (out / "photo_scan.png").is_file()
    assert (out / "photo_debug.png").is_file()
    assert "photo.png: ok" in proc.stdout


@needs_model
def test_eval_writes_csv_and_contact_sheet(tmp_path):
    folder = tmp_path / "images"
    folder.mkdir()
    for i, rotation in enumerate((0, 90, 180)):
        _write_synth(folder / f"page_{i}.png", rotation=rotation, seed=i)
    out = tmp_path / "eval"

    proc = _run(
        "scripts/eval.py", folder, "-o", out, "--weights", TFLITE_WEIGHTS
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    with (out / "results.csv").open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert {"file", "ok", "reason", "confidence", "marker", "predict"} <= set(
        rows[0]
    )
    assert (out / "contact_sheet.png").is_file()
