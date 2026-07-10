"""Turn the blobby FairScan mask into a clean, straight page quadrilateral,
using the page's ruled lines to find its true angle and extent.

Reads cached probability maps from data/cache (run cache_masks.py first) --
the model is NEVER loaded here.

    python refine_page.py            # viewer over all samples (space=next)
    python refine_page.py --save     # write comparisons to data/refined/ and exit
"""
import sys, glob, os
import numpy as np
import cv2
from PIL import Image

CACHE = "data/cache"
THRESH = 5          # same threshold you liked on the prob map
ANGLE_TOL = 6       # degrees; segments within this of the dominant angle are "ruled lines"
PCTL = (1, 99)      # robust page extent in the deskewed frame


def rough_mask(prob):
    """prob>THRESH, largest connected component only."""
    m = (prob > THRESH).astype(np.uint8)
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return m.astype(bool)
    biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return lbl == biggest


def ruled_lines(gray, region):
    """HoughLinesP segments whose midpoint lands inside the rough page region."""
    edges = cv2.Canny(gray, 50, 150)
    edges[~region] = 0
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                           minLineLength=gray.shape[1] // 6, maxLineGap=20)
    return np.empty((0, 4)) if segs is None else segs.reshape(-1, 4)


def dominant_angle(segs):
    """Median angle (deg, -90..90) of the biggest parallel cluster of segments."""
    ang = np.degrees(np.arctan2(segs[:, 3] - segs[:, 1], segs[:, 2] - segs[:, 0]))
    ang = (ang + 90) % 180 - 90                      # fold to -90..90
    hist, edges = np.histogram(ang, bins=36, range=(-90, 90))
    center = edges[np.argmax(hist)] + 2.5            # dominant 5-deg bin
    keep = np.abs((ang - center + 90) % 180 - 90) < ANGLE_TOL
    return np.median(ang[keep]), keep


def rot(pts, deg, c):
    r = np.deg2rad(deg)
    R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
    return (pts - c) @ R.T + c


def refine(img, prob):
    """Return the clean page quad (4x2 float) or None if no ruled lines found."""
    region = rough_mask(prob)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    segs = ruled_lines(gray, region)
    if len(segs) == 0:
        return None
    theta, keep = dominant_angle(segs)
    if keep.sum() == 0:
        return None
    # angle from the ruled lines; extent from the fairscan blob (it covers the
    # page well, just jaggedly -- line endpoints don't reach the page edges).
    ys, xs = np.where(region)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    c = np.array(gray.shape[::-1]) / 2.0
    d = rot(pts, -theta, c)                            # deskew: ruled lines -> horizontal
    lo = np.percentile(d, PCTL[0], axis=0)
    hi = np.percentile(d, PCTL[1], axis=0)
    box = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    return rot(box, theta, c)                          # rotate the rectangle back


def draw(img, prob):
    """Side-by-side: fairscan blob (red) vs refined quad (green)."""
    out = img.copy()
    left = out.copy()
    left[rough_mask(prob)] = (0.5 * left[rough_mask(prob)] + [128, 0, 0]).astype(np.uint8)
    right = out.copy()
    quad = refine(img, prob)
    if quad is not None:
        cv2.polylines(right, [quad.astype(np.int32)], True, (0, 255, 0), 6)
        fill = right.copy()
        cv2.fillPoly(fill, [quad.astype(np.int32)], (0, 128, 0))
        right = (0.7 * right + 0.3 * fill).astype(np.uint8)
    return np.hstack([left, right])


def load(p):
    img = np.asarray(Image.open(p).convert("RGB"))
    cp = os.path.join(CACHE, os.path.splitext(os.path.basename(p))[0] + ".png")
    prob = np.asarray(Image.open(cp))
    return img, prob


def main(paths, save):
    if save:
        os.makedirs("data/refined", exist_ok=True)
        for p in paths:
            img, prob = load(p)
            o = os.path.join("data/refined", os.path.basename(p))
            Image.fromarray(draw(img, prob)).save(o)
            print(f"{os.path.basename(p)} -> {o}  quad={'yes' if refine(img,prob) is not None else 'NONE'}")
        return
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 7)); ax.axis("off")
    st = {"i": -1}

    def show():
        st["i"] = (st["i"] + 1) % len(paths)
        img, prob = load(paths[st["i"]])
        ax.clear(); ax.axis("off")
        ax.imshow(draw(img, prob))
        ax.set_title(f"{st['i']+1}/{len(paths)}  {os.path.basename(paths[st['i']])}"
                     "   left=fairscan  right=refined   (space=next)")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", lambda e: show() if e.key == " " else None)
    show(); plt.show()


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--save"]
    main(args or sorted(glob.glob("data/samples/*.png")), "--save" in sys.argv)
