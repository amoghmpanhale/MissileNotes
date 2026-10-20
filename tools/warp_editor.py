"""Hand-correct scans that the automatic pipeline got wrong.

    python -m tools.warp_editor IMG...

The initial quad comes from :func:`missilenotes.pipeline.scan`, mapped back
into the upright photo frame. Drag the four corners, then press Enter (or
Space) to preview the flattened page. ``b`` toggles binarized/grayscale,
Left/Right step through the photos, and ``s`` saves the current view next to
the original as ``<stem>_scan.png``.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pillow_heif
import tkinter as tk
from PIL import Image, ImageOps, ImageTk

from missilenotes.enhance import binarize
from missilenotes.geometry import order_corners, warp
from missilenotes.marker import rotate_image, rotate_points
from missilenotes.pipeline import ScanResult, scan
from missilenotes.segment import Segmenter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = (
    ROOT / "experiments/model_horizontal/fairscan-segmentation-model.tflite"
)

MAX_W, MAX_H = 1100, 850
CORNER_HIT = 15


def load_rgb(path: Path) -> np.ndarray:
    """Read a photo (HEIC included) as an upright RGB array."""
    pillow_heif.register_heif_opener()
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return np.asarray(image.convert("RGB"))


def quad_for_editing(
    quad: np.ndarray | None,
    rotation: int,
    original_shape: tuple[int, ...],
    upright_shape: tuple[int, ...],
) -> np.ndarray:
    """Map the pipeline quad (original photo coords) into the upright frame.

    ``rotation`` is the CCW angle the pipeline applied. When the pipeline
    failed to fit a page (``quad is None``) the editor starts from the whole
    upright image so the corners can still be dragged.
    """
    if quad is not None:
        return order_corners(rotate_points(quad, rotation, original_shape))

    h, w = upright_shape[:2]
    return np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32
    )


def flatten_page(
    image_rgb: np.ndarray, quad: np.ndarray, binary: bool = False
) -> np.ndarray:
    """Warp ``quad`` flat and return grayscale or binarized ink-on-white."""
    flat = warp(image_rgb, order_corners(quad))
    gray = cv2.cvtColor(flat, cv2.COLOR_RGB2GRAY)
    return binarize(gray) if binary else gray


@dataclass
class Page:
    """One photo, the upright copy the user edits, and its quad."""

    path: Path
    image: np.ndarray
    quad: np.ndarray
    scale: float
    result: ScanResult


class Editor:
    def __init__(self, root: tk.Tk, paths: list[Path], segmenter: Segmenter):
        self.root = root
        self.paths = paths
        self.segmenter = segmenter
        self.index = 0
        self.mode = "edit"
        self.binary = True
        self.drag: int | None = None
        self._pages: dict[int, Page] = {}
        self._photo = None

        self.canvas = tk.Canvas(root, width=MAX_W, height=MAX_H, bg="gray20")
        self.canvas.pack()
        self.status = tk.StringVar()
        tk.Label(root, textvariable=self.status, anchor="w").pack(fill="x")

        self.canvas.bind("<Button-1>", self._press)
        self.canvas.bind("<B1-Motion>", self._motion)
        self.canvas.bind(
            "<ButtonRelease-1>", lambda _e: setattr(self, "drag", None)
        )
        root.bind("<Left>", lambda _e: self.step(-1))
        root.bind("<Right>", lambda _e: self.step(1))
        root.bind("<Return>", lambda _e: self.toggle_mode())
        root.bind("<space>", lambda _e: self.toggle_mode())
        root.bind("b", lambda _e: self.toggle_binary())
        root.bind("s", lambda _e: self.save())
        self.render()

    def _load(self, index: int) -> Page:
        if index in self._pages:
            return self._pages[index]

        path = self.paths[index]
        original = load_rgb(path)
        result = scan(original, self.segmenter)
        upright = rotate_image(original, result.rotation)
        quad = quad_for_editing(
            result.quad, result.rotation, original.shape, upright.shape
        )
        h, w = upright.shape[:2]
        page = Page(
            path=path,
            image=upright,
            quad=quad.astype(np.float32),
            scale=min(MAX_W / w, MAX_H / h, 1.0),
            result=result,
        )
        self._pages[index] = page
        return page

    def render(self) -> None:
        page = self._load(self.index)
        view = "preview" if self.mode == "warp" else "edit"
        note = "" if page.result.ok else f"  [auto failed: {page.result.reason}]"
        self.root.title(
            f"{self.index + 1}/{len(self.paths)}  {page.path.name}  "
            f"({view}){note}"
        )
        if self.mode == "warp":
            self._show_warp(page)
        else:
            self._show_edit(page)
        self.status.set(
            "drag corners | Enter: preview | Left/Right: photo | "
            "b: binarize | s: save"
        )

    def _show_edit(self, page: Page) -> None:
        h, w = page.image.shape[:2]
        size = (max(1, round(w * page.scale)), max(1, round(h * page.scale)))
        display = Image.fromarray(page.image).resize(size, Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)

        pts = page.quad * page.scale
        loop = np.vstack([pts, pts[0]])
        self.canvas.create_line(
            *loop.ravel().tolist(), fill="lime", width=2
        )
        for x, y in pts:
            self.canvas.create_oval(
                x - 6, y - 6, x + 6, y + 6, fill="yellow", outline="black"
            )

    def _show_warp(self, page: Page) -> None:
        flat = flatten_page(page.image, page.quad, self.binary)
        display = Image.fromarray(flat)
        scale = min(MAX_W / display.width, MAX_H / display.height, 1.0)
        display = display.resize(
            (round(display.width * scale), round(display.height * scale)),
            Image.LANCZOS,
        )
        self._photo = ImageTk.PhotoImage(display)
        self.canvas.delete("all")
        self.canvas.create_image(
            MAX_W // 2, MAX_H // 2, anchor="center", image=self._photo
        )

    def step(self, delta: int) -> None:
        self.index = max(0, min(len(self.paths) - 1, self.index + delta))
        self.mode = "edit"
        self.drag = None
        self.render()

    def toggle_mode(self) -> None:
        self.mode = "warp" if self.mode == "edit" else "edit"
        self.drag = None
        self.render()

    def toggle_binary(self) -> None:
        self.mode = "warp"
        self.binary = not self.binary
        self.render()

    def save(self) -> None:
        page = self._load(self.index)
        output = flatten_page(page.image, page.quad, self.binary)
        path = page.path.with_name(f"{page.path.stem}_scan.png")
        Image.fromarray(output).save(path)
        self.status.set(f"saved {path}")

    def _press(self, event: tk.Event) -> None:
        if self.mode != "edit":
            return
        page = self._load(self.index)
        pts = page.quad * page.scale
        dist = np.sum((pts - [event.x, event.y]) ** 2, axis=1)
        corner = int(np.argmin(dist))
        self.drag = corner if dist[corner] <= CORNER_HIT**2 else None

    def _motion(self, event: tk.Event) -> None:
        if self.drag is None or self.mode != "edit":
            return
        page = self._load(self.index)
        h, w = page.image.shape[:2]
        page.quad[self.drag] = (
            min(max(event.x / page.scale, 0.0), w - 1.0),
            min(max(event.y / page.scale, 0.0), h - 1.0),
        )
        self._show_edit(page)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.warp_editor",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    missing = [path for path in args.images if not path.is_file()]
    if missing:
        print(f"not a file: {missing[0]}", file=sys.stderr)
        return 2
    if not args.weights.is_file():
        print(f"weights not found: {args.weights}", file=sys.stderr)
        return 2

    root = tk.Tk()
    Editor(root, list(args.images), Segmenter(args.weights))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
