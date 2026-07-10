"""Tkinter editor for the horizontal FairScan pipeline.

Flow per image: refine the quad by dragging the yellow corners, then generate
the warp to see the flattened grayscale scan, then move to the next image.
Left/Right arrows (or the bottom buttons) step through every screen, so you
can go back and fix a bad corner. Nothing is saved.

    python warp_editor.py                 # all data/samples/*.png
    python warp_editor.py img1.png img2.png
"""
import sys, glob, os
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import experiments.segment_pages as sp

sp.MODEL = "model_horizontal/fairscan-segmentation-model.pt"  # horizontal weights

MAXW, MAXH = 1100, 850   # display box; image scaled to fit, quad edited in this frame
HIT = 15                 # px radius to grab a corner


def quad_for(model, img):
    """Initial page quad (4x2, image coords) via the horizontal pipeline."""
    horiz = img.rotate(-90, expand=True)
    mask = np.rot90(sp.mask_for(model, horiz))
    mask = sp.keep_marker_island(mask, img)
    quad = sp.page_quad(mask, img)
    if quad is None:  # fallback: whole-image rectangle so the dots still exist
        w, h = img.size
        quad = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    return quad


class Editor:
    def __init__(self, root, paths):
        self.paths = paths
        self.model = sp.load_model()
        self.imgs, self.scales, self.pts = {}, {}, {}  # lazily filled, keyed by image idx
        self.s = 0                                      # screen index: image = s//2, mode edit/warp
        self.drag = None

        self.canvas = tk.Canvas(root, width=MAXW, height=MAXH, bg="gray20")
        self.canvas.pack()
        bar = tk.Frame(root); bar.pack(fill="x")
        tk.Button(bar, text="◀ Back", command=lambda: self.step(-1)).pack(side="left", expand=True, fill="x")
        tk.Button(bar, text="Generate warp / Next ▶", command=lambda: self.step(1)).pack(side="left", expand=True, fill="x")
        self.canvas.bind("<Button-1>", self.press)
        self.canvas.bind("<B1-Motion>", self.motion)
        self.canvas.bind("<ButtonRelease-1>", lambda e: setattr(self, "drag", None))
        root.bind("<Left>", lambda e: self.step(-1))
        for k in ("<Right>", "<Return>", "<space>"):
            root.bind(k, lambda e: self.step(1))
        self.root = root
        self.render()

    def load(self, idx):
        """Ensure image, scale and editable corner points exist for image idx."""
        if idx in self.imgs:
            return
        img = sp.upright(Image.open(self.paths[idx]).convert("RGB"))
        w, h = img.size
        scale = min(MAXW / w, MAXH / h, 1.0)
        self.imgs[idx] = img
        self.scales[idx] = scale
        self.pts[idx] = (quad_for(self.model, img) * scale).tolist()  # display coords

    def step(self, d):
        self.s = max(0, min(2 * len(self.paths) - 1, self.s + d))
        self.render()

    def render(self):
        idx, warp_mode = self.s // 2, self.s % 2
        self.load(idx)
        name = os.path.basename(self.paths[idx])
        self.root.title(f"{idx+1}/{len(self.paths)}  {name}  ({'warp' if warp_mode else 'refine'})")
        self.canvas.delete("all")
        if warp_mode:
            self.show_warp(idx)
        else:
            self.show_edit(idx)

    def show_edit(self, idx):
        disp = self.imgs[idx].resize((round(self.imgs[idx].width * self.scales[idx]),
                                      round(self.imgs[idx].height * self.scales[idx])))
        self.photo = ImageTk.PhotoImage(disp)  # keep ref alive
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        pts = self.pts[idx]
        loop = pts + [pts[0]]
        for (x0, y0), (x1, y1) in zip(loop, loop[1:]):
            self.canvas.create_line(x0, y0, x1, y1, fill="lime", width=2)
        for x, y in pts:
            self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill="yellow", outline="black")

    def show_warp(self, idx):
        quad = np.array(self.pts[idx], np.float32) / self.scales[idx]  # back to image coords
        scan = sp.warp(self.imgs[idx], quad)
        gray = (scan @ [0.299, 0.587, 0.114]).astype(np.uint8)
        pim = Image.fromarray(gray)
        s = min(MAXW / pim.width, MAXH / pim.height, 1.0)
        pim = pim.resize((round(pim.width * s), round(pim.height * s)))
        self.photo = ImageTk.PhotoImage(pim)
        self.canvas.create_image(MAXW // 2, MAXH // 2, image=self.photo)

    def press(self, e):
        if self.s % 2:  # no dragging on the warp screen
            return
        pts = self.pts[self.s // 2]
        d = [(e.x - x) ** 2 + (e.y - y) ** 2 for x, y in pts]
        i = int(np.argmin(d))
        self.drag = i if d[i] <= HIT ** 2 else None

    def motion(self, e):
        if self.drag is None:
            return
        self.pts[self.s // 2][self.drag] = [min(max(e.x, 0), MAXW), min(max(e.y, 0), MAXH)]
        self.render()


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("data/samples/*.png"))
    root = tk.Tk()
    Editor(root, paths)
    root.mainloop()
