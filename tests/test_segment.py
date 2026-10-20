"""Tests for the FairScan segmentation wrapper.

The synthetic photo contains a white page with dark text lines, slightly
rotated, on a darker textured background. It is generated in-process so no
scan images need to be committed.
"""

import importlib.util
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from missilenotes.segment import INPUT_SIZE, Segmenter, preprocess

ROOT = Path(__file__).resolve().parents[1]
PT_WEIGHTS = ROOT / "experiments/model_horizontal/fairscan-segmentation-model.pt"
TFLITE_WEIGHTS = ROOT / "experiments/model_horizontal/fairscan-segmentation-model.tflite"

HAS_TORCH = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("segmentation_models_pytorch") is not None
)
HAS_LITERT = importlib.util.find_spec("ai_edge_litert") is not None

BACKENDS = [
    pytest.param(PT_WEIGHTS, HAS_TORCH, id="torch"),
    pytest.param(TFLITE_WEIGHTS, HAS_LITERT, id="litert"),
]


def synthetic_photo() -> np.ndarray:
    """~1000x750 RGB photo: a rotated paper page on a textured dark surface."""
    rng = np.random.default_rng(7)
    h, w = 750, 1000
    background = np.clip(rng.normal(55, 20, (h, w, 3)), 0, 255).astype(np.uint8)
    background = cv2.GaussianBlur(background, (7, 7), 0)

    page = np.full((560, 420, 3), 245, np.uint8)
    cv2.rectangle(page, (30, 30), (390, 530), (70, 70, 70), 2)
    for y in range(60, 520, 42):
        cv2.line(page, (45, y), (375, y), (30, 30, 30), 2)
    rotation = cv2.getRotationMatrix2D((210, 280), -11, 1.0)
    rotated = cv2.warpAffine(
        page, rotation, (420, 560), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
    )
    page_mask = cv2.warpAffine(
        np.full((560, 420), 255, np.uint8),
        rotation,
        (420, 560),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    ).astype(bool)

    x0, y0 = 290, 95
    roi = background[y0 : y0 + 560, x0 : x0 + 420]
    background[y0 : y0 + 560, x0 : x0 + 420] = np.where(
        page_mask[..., None], rotated, roi
    )
    return background


@pytest.fixture(scope="module")
def photo() -> np.ndarray:
    return synthetic_photo()


@pytest.mark.parametrize("weights_path,available", BACKENDS)
def test_prediction_contract(weights_path, available, photo):
    if not available:
        pytest.skip(f"runtime for {weights_path.suffix} is not installed")

    segmenter = Segmenter(weights_path)
    prob = segmenter.predict(photo)

    assert segmenter.input_size == INPUT_SIZE == 256
    assert prob.shape == photo.shape[:2]
    assert prob.dtype == np.float32
    assert np.isfinite(prob).all()
    assert float(prob.min()) >= 0.0 and float(prob.max()) <= 1.0
    # The page must be found for the backend parity check to mean anything.
    assert float((prob > 0.5).mean()) > 0.1


def test_tflite_tensor_contract():
    if not HAS_LITERT:
        pytest.skip("ai-edge-litert is not installed")

    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(TFLITE_WEIGHTS))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    assert list(inp["shape"]) == [1, 256, 256, 3]
    assert inp["dtype"] == np.float32
    assert inp["quantization"] == (0.0, 0)
    assert list(out["shape"]) == [1, 256, 256, 1]
    assert out["dtype"] == np.float32
    assert out["quantization"] == (0.0, 0)

    # Feed the documented (externally normalized) NHWC input and verify the
    # raw output is logits, not probabilities: this photo drives it well
    # outside [0, 1].
    x = preprocess(synthetic_photo())[None]
    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    raw = interpreter.get_tensor(out["index"])[0, :, :, 0]
    assert raw.min() < 0.0 and raw.max() > 1.0


@pytest.mark.skipif(
    not (HAS_TORCH and HAS_LITERT), reason="both torch and ai-edge-litert are required"
)
def test_torch_and_tflite_agree(photo):
    probs = {}
    for name, weights in (("torch", PT_WEIGHTS), ("litert", TFLITE_WEIGHTS)):
        segmenter = Segmenter(weights)
        segmenter.predict(photo)  # warm up lazy imports / allocator
        start = time.perf_counter()
        probs[name] = segmenter.predict(photo)
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"{name} inference: {elapsed_ms:.1f} ms")

    torch_prob, tflite_prob = probs["torch"], probs["litert"]
    mean_abs_diff = float(np.abs(torch_prob - tflite_prob).mean())
    torch_mask, tflite_mask = torch_prob > 0.5, tflite_prob > 0.5
    union = np.logical_or(torch_mask, tflite_mask).sum()
    iou = float(np.logical_and(torch_mask, tflite_mask).sum() / union) if union else 1.0
    print(f"mean abs diff: {mean_abs_diff:.5f}, IoU: {iou:.5f}")

    assert mean_abs_diff < 0.02
    assert iou > 0.97
