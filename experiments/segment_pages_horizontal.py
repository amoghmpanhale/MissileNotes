"""Segment pages with the horizontal FairScan model.
The ArUco marker sits 90 deg clockwise from the vertical model's, and the
model expects the page in that horizontal frame. So: rotate upright (via the
marker), rotate 90 deg CW into the model's frame, infer, then rotate the mask
back to upright and show everything upright. One image at a time; space=next."""
import sys, glob, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import experiments.segment_pages as sp  # reuse upright(), mask_for(), load_model()

# ponytail: same code as segment_pages, only the weights differ. Reassign the
# module global that load_model() reads instead of forking the whole file.
sp.MODEL = "model_horizontal/fairscan-segmentation-model.pt"


def main(paths):
    t = time.perf_counter()
    model = sp.load_model()
    print(f"model loaded on {sp.DEVICE} in {time.perf_counter()-t:.2f}s")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    state = {"i": -1}

    def show_next():
        state["i"] += 1
        if state["i"] >= len(paths):
            plt.close(fig)
            return
        p = paths[state["i"]]
        img = sp.upright(Image.open(p).convert("RGB"))   # upright frame
        horiz = img.rotate(-90, expand=True)             # 90 deg CW -> model frame
        t = time.perf_counter()
        mask = sp.mask_for(model, horiz)                 # mask in horizontal frame
        mask = np.rot90(mask)                            # CCW 90 -> back to upright
        mask = sp.keep_marker_island(mask, img)          # marker island only, filled
        quad = sp.page_quad(mask, img)
        print(f"{p.split('/')[-1]}: inference {time.perf_counter()-t:.3f}s")

        # flash the mask + quad on the upright image before warping
        ax.clear()
        ax.axis("off")
        ax.imshow(img)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask] = [1, 0, 0, sp.OPACITY]
        ax.imshow(overlay)
        if quad is not None:
            loop = np.vstack([quad, quad[0]])
            ax.plot(loop[:, 0], loop[:, 1], "-", color="lime", linewidth=2)
            ax.plot(quad[:, 0], quad[:, 1], "o", color="yellow",
                    markersize=8, markeredgecolor="black")
        ax.set_title(f"{state['i']+1}/{len(paths)}  {p.split('/')[-1]}  (mask)")
        plt.pause(0.02)

        ax.clear()
        ax.axis("off")
        scan = sp.warp(img, quad) if quad is not None else np.asarray(img)
        gray = scan @ [0.299, 0.587, 0.114]  # luminance
        ax.imshow(gray, cmap="gray")  # flattened scan, grayscale
        ax.set_title(f"{state['i']+1}/{len(paths)}  {p.split('/')[-1]}  (spacebar=next)")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            show_next()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show_next()
    plt.show()


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("data/samples/*.png"))
    main(paths)
