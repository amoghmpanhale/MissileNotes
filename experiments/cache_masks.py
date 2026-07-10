"""Run FairScan inference ONCE per sample and cache the probability map.
Everything downstream (refine_page.py) reads these PNGs, never the model.

    python cache_masks.py              # all data/samples/*.png
    python cache_masks.py img.png ...  # specific files
"""
import sys, glob, os, time
import numpy as np
import torch
from PIL import Image
import segmentation_models_pytorch as smp

MODEL = "model/fairscan-segmentation-model.pt"
CACHE = "data/cache"
SIZE = 512
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)


def load_model():
    m = smp.DeepLabV3Plus(encoder_name="mobilenet_v2", encoder_weights=None,
                          in_channels=3, classes=1)
    m.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=False))
    return m.eval().to(DEVICE)


def prob_for(model, img):
    """Return uint8 probability map (0-255) at the image's original resolution."""
    x = torch.from_numpy(np.asarray(img.resize((SIZE, SIZE)), np.float32) / 255.0).to(DEVICE)
    x = ((x.permute(2, 0, 1) - MEAN) / STD).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu()
    prob = Image.fromarray((prob.numpy() * 255).astype(np.uint8)).resize(img.size)
    return prob


def main(paths):
    os.makedirs(CACHE, exist_ok=True)
    model = load_model()
    for p in paths:
        out = os.path.join(CACHE, os.path.splitext(os.path.basename(p))[0] + ".png")
        if os.path.exists(out):
            print(f"skip {out} (exists)")
            continue
        t = time.perf_counter()
        prob_for(model, Image.open(p).convert("RGB")).save(out)
        print(f"{os.path.basename(p)} -> {out}  ({time.perf_counter()-t:.2f}s)")


if __name__ == "__main__":
    main(sys.argv[1:] or sorted(glob.glob("data/samples/*.png")))
