"""FairScan page segmentation for the PyTorch and TFLite weights.

Both backends expose the same contract through :class:`Segmenter`: an
``HxWx3`` uint8 RGB image goes in, an ``HxW`` float32 probability map in
``[0, 1]`` at the original resolution comes out.

Model/tensor contract (verified against the shipped horizontal weights):

PyTorch (``.pt``)
    * Architecture: DeepLabV3+ / mobilenet_v2, one output channel.
    * Input: NCHW float32, ``1x3x256x256``. RGB, scaled to ``[0, 1]``, then
      normalized with ImageNet mean ``[0.485, 0.456, 0.406]`` and std
      ``[0.229, 0.224, 0.225]``.
    * Output: ``1x1x256x256`` float32 **logits**. The sigmoid is applied by
      this module, not by the checkpoint.

TFLite (``.tflite``)
    * Input: NHWC float32, ``1x256x256x3``. The graph contains no
      normalization ops; the caller must apply the same ImageNet
      normalization as above (verified: pre-normalized input makes the TFLite
      logits match the PyTorch logits to within float noise).
    * Output: ``1x256x256x1`` float32 **logits**; the graph ends in a
      ``RESIZE_BILINEAR`` + conv and has no ``LOGISTIC``/sigmoid op.
    * Both tensors are plain float32 (no quantization).

The TFLite file pins the input to 256x256, so 256 is the wrapper's input size
for *both* backends. The original prototype scripts, now removed, fed
512x512; running the same PyTorch weights at 512 produces a materially
different mask, while at 256 the two backends agree. Keeping one shared size
is what makes the reference implementation and the on-device model
interchangeable.

Post-processing: sigmoid -> resize (bicubic) back to the input image's
resolution -> clip to ``[0, 1]``.
"""

from pathlib import Path

import numpy as np
from PIL import Image

INPUT_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_TORCH_SUFFIX = ".pt"
_LITERT_SUFFIX = ".tflite"


def preprocess(img_rgb: np.ndarray, size: int = INPUT_SIZE) -> np.ndarray:
    """Resize an HxWx3 uint8 RGB image and apply ImageNet normalization.

    Returns an HxWx3 float32 array in NCHW-agnostic HWC layout.
    """
    if not isinstance(img_rgb, np.ndarray):
        raise TypeError("img_rgb must be a numpy array")
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError(f"img_rgb must be HxWx3, got {img_rgb.shape}")
    if img_rgb.dtype != np.uint8:
        raise ValueError(f"img_rgb must be uint8, got {img_rgb.dtype}")
    if img_rgb.shape[0] == 0 or img_rgb.shape[1] == 0:
        raise ValueError("img_rgb must have non-zero height and width")

    # PIL's bicubic resize matches the resize used by the reference scripts.
    resized = Image.fromarray(img_rgb).resize((size, size), Image.BICUBIC)
    x = np.asarray(resized, dtype=np.float32) / 255.0
    return (x - MEAN) / STD


def postprocess(prob: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Resize a probability map to ``out_hw`` and clip it to ``[0, 1]``."""
    h, w = out_hw
    if prob.shape != (h, w):
        resized = Image.fromarray(prob.astype(np.float32), mode="F").resize(
            (w, h), Image.BICUBIC
        )
        prob = np.asarray(resized, dtype=np.float32)
    return np.clip(prob, 0.0, 1.0).astype(np.float32, copy=False)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64, copy=False)
    return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)


class Segmenter:
    """Run FairScan page segmentation with either backend.

    ``weights_path`` selects the backend by extension: ``.pt`` uses PyTorch,
    ``.tflite`` uses LiteRT. Torch prefers CUDA, then MPS, then CPU.
    """

    def __init__(self, weights_path: str | Path):
        self.weights_path = Path(weights_path)
        suffix = self.weights_path.suffix.lower()
        self.input_size = INPUT_SIZE
        self.device = None
        self._model = None
        self._interpreter = None
        self._input_index = None
        self._output_index = None

        if suffix == _TORCH_SUFFIX:
            self.backend = "torch"
            self._load_torch()
        elif suffix == _LITERT_SUFFIX:
            self.backend = "litert"
            self._load_litert()
        else:
            raise ValueError(
                f"unsupported weights extension {suffix!r}; "
                f"expected {_TORCH_SUFFIX!r} or {_LITERT_SUFFIX!r}"
            )

    def _load_torch(self) -> None:
        import torch
        import segmentation_models_pytorch as smp

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        model = smp.DeepLabV3Plus(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        self._model = model.eval().to(self.device)

    def _load_litert(self) -> None:
        from ai_edge_litert.interpreter import Interpreter

        interpreter = Interpreter(model_path=str(self.weights_path))
        interpreter.allocate_tensors()

        inp = interpreter.get_input_details()[0]
        out = interpreter.get_output_details()[0]
        expected_in = [1, self.input_size, self.input_size, 3]
        expected_out = [1, self.input_size, self.input_size, 1]
        if list(inp["shape"]) != expected_in or inp["dtype"] != np.float32:
            raise ValueError(
                f"unexpected TFLite input {inp['shape']} {inp['dtype']}; "
                f"expected {expected_in} float32"
            )
        if list(out["shape"]) != expected_out or out["dtype"] != np.float32:
            raise ValueError(
                f"unexpected TFLite output {out['shape']} {out['dtype']}; "
                f"expected {expected_out} float32"
            )

        self._interpreter = interpreter
        self._input_index = inp["index"]
        self._output_index = out["index"]

    def _infer_torch(self, x_hwc: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(x_hwc.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            logits = self._model(x.to(self.device).contiguous())
        return logits[0, 0].cpu().numpy()

    def _infer_litert(self, x_hwc: np.ndarray) -> np.ndarray:
        self._interpreter.set_tensor(self._input_index, x_hwc[None])
        self._interpreter.invoke()
        out = self._interpreter.get_tensor(self._output_index)
        return np.array(out[0, :, :, 0], dtype=np.float32, copy=True)

    def predict(self, img_rgb: np.ndarray) -> np.ndarray:
        """Return a float32 HxW probability map in ``[0, 1]``."""
        x = preprocess(img_rgb, self.input_size)
        if self.backend == "torch":
            logits = self._infer_torch(x)
        else:
            logits = self._infer_litert(x)
        return postprocess(_sigmoid(logits), img_rgb.shape[:2])
