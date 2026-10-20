"""Checks the shared fixture exporter used by the Dart parity tests."""

import importlib.util
import json
from pathlib import Path

import cv2
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "scripts/export_fixtures.py"
WEIGHTS = ROOT / "experiments/model_horizontal/fairscan-segmentation-model.tflite"

pytestmark = pytest.mark.skipif(
    not WEIGHTS.is_file(),
    reason="horizontal .tflite weights are unavailable (run git lfs pull)",
)


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_fixtures", EXPORTER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    out = tmp_path_factory.mktemp("fixtures")
    exporter = _load_exporter()
    exporter.export_fixtures(out, size=400)
    return out


def _read_json(path):
    return json.loads(path.read_text())


def _shape(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert image is not None, f"could not decode {path}"
    return image.shape


def test_writes_six_complete_fixtures(exported):
    jsons = sorted(exported.glob("fixture_*.json"))
    assert len(jsons) == 6
    assert [p.name for p in jsons] == [f"fixture_{i}.json" for i in range(1, 7)]

    for path in jsons:
        payload = _read_json(path)
        assert set(payload) >= {
            "index",
            "photo",
            "photo_shape",
            "prob_model",
            "model_shape",
            "expected_scan",
            "ok",
            "reason",
            "rotation",
            "quad",
            "confidence",
            "scan_shape",
            "flat_shape",
            "threshold",
        }
        assert isinstance(payload["ok"], bool)
        assert isinstance(payload["reason"], str) and payload["reason"]
        assert payload["rotation"] in (0, 90, 180, 270)
        assert payload["threshold"] == 0.5

        photo = exported / payload["photo"]
        assert photo.is_file()
        assert not photo.name.endswith(("_scan.png", "_debug.png"))
        assert list(_shape(photo)[:2]) == payload["photo_shape"][:2]

        if payload["prob_model"] is not None:
            prob = exported / payload["prob_model"]
            assert prob.is_file()
            assert list(_shape(prob)) == payload["model_shape"]
        if payload["expected_scan"] is not None:
            scan_png = exported / payload["expected_scan"]
            assert scan_png.is_file()
            assert list(_shape(scan_png)) == payload["scan_shape"]


def test_expected_results(exported):
    payloads = [_read_json(p) for p in sorted(exported.glob("fixture_*.json"))]
    assert [p["ok"] for p in payloads] == [True, True, True, True, True, False]
    assert [p["rotation"] for p in payloads[:5]] == [0, 270, 180, 90, 0]
    assert all(p["prob_model"] is not None for p in payloads[:5])
    assert all(p["quad"] is not None for p in payloads[:5])

    markerless = payloads[5]
    assert "marker" in markerless["reason"].lower()
    assert markerless["prob_model"] is None
    assert markerless["scan_shape"] is None
    assert markerless["quad"] is None


def test_fixture_budget(exported):
    total = sum(p.stat().st_size for p in exported.glob("fixture_*"))
    assert total < 3_000_000, f"fixtures are {total / 1e6:.2f} MB"
