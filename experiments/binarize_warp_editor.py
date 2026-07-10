"""Same warp editor, but the warp screen shows the binarized (GoodNotes-style)
mask instead of the raw grayscale. Press 'b' to toggle back to grayscale.

    python binarize_warp_editor.py [img1.png ...]
"""
import sys, glob
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import experiments.segment_pages as sp
from experiments.warp_editor import Editor, MAXW, MAXH
from experiments.binarize import binarize


class BinEditor(Editor):
    def __init__(self, root, paths):
        self.binmode = True
        root.bind("b", lambda e: (setattr(self, "binmode", not self.binmode), self.render()))
        super().__init__(root, paths)

    def show_warp(self, idx):
        quad = np.array(self.pts[idx], np.float32) / self.scales[idx]
        scan = sp.warp(self.imgs[idx], quad)
        gray = (scan @ [0.299, 0.587, 0.114]).astype(np.uint8)
        pim = Image.fromarray(binarize(gray) if self.binmode else gray)
        s = min(MAXW / pim.width, MAXH / pim.height, 1.0)
        pim = pim.resize((round(pim.width * s), round(pim.height * s)))
        self.photo = ImageTk.PhotoImage(pim)
        self.canvas.create_image(MAXW // 2, MAXH // 2, image=self.photo)


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("data/samples/*.png"))
    root = tk.Tk()
    BinEditor(root, paths)
    root.mainloop()
