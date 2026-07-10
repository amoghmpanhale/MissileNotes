#!/usr/bin/env python3
from pathlib import Path
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

src, dst = Path("data"), Path("data_png")
dst.mkdir(exist_ok=True)

for f in list(src.glob("*.heic")) + list(src.glob("*.HEIC")):
    Image.open(f).save(dst / f"{f.stem}.png")
