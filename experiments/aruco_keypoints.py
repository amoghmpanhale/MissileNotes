"""Detect ArUco markers in data/samples and overlay their key points with matplotlib.

Usage: python aruco_keypoints.py [image_or_dir]   (default: data/samples)
"""
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

# ponytail: probe the common dicts instead of hardcoding one; the marker's
# origin dict is unknown. Drop to a single dict if you know it.
DICTS = [
    "DICT_4X4_50", "DICT_5X5_50", "DICT_6X6_50", "DICT_7X7_50",
    "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_36h11",
]


def detect(gray):
    """Return (corners, ids, dict_name) for the first dict that finds markers."""
    for name in DICTS:
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
        det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
        if ids is not None:
            return corners, ids, name
    return [], None, None


def draw(ax, path):
    ax.clear()
    ax.axis("off")
    img = cv2.imread(str(path))
    if img is None:
        ax.set_title(f"skip (unreadable): {path.name}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, dict_name = detect(gray)

    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{path.name} — {dict_name or 'no markers found'}  "
                 "(space=next, backspace=prev, q=quit)")

    # corner order from OpenCV: top-left, top-right, bottom-right, bottom-left
    labels = ["TL", "TR", "BR", "BL"]
    colors = ["red", "lime", "cyan", "magenta"]
    for c, mid in zip(corners, (ids.flatten() if ids is not None else [])):
        pts = c[0]  # shape (4, 2)
        for (x, y), lab, col in zip(pts, labels, colors):
            ax.plot(x, y, "o", color=col, markersize=8, markeredgecolor="black")
            ax.annotate(lab, (x, y), color=col, fontsize=9, weight="bold",
                        xytext=(6, 6), textcoords="offset points")
        cx, cy = pts.mean(axis=0)
        ax.plot(cx, cy, "x", color="yellow", markersize=10, markeredgewidth=2)
        ax.annotate(f"id {mid}", (cx, cy), color="yellow", fontsize=11,
                    weight="bold", ha="center", xytext=(0, 14),
                    textcoords="offset points")


def browse(imgs):
    i = [0]
    fig, ax = plt.subplots(figsize=(8, 10))

    def render():
        draw(ax, imgs[i[0]])
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key in (" ", "right", "down"):
            i[0] = (i[0] + 1) % len(imgs)
            render()
        elif event.key in ("backspace", "left", "up"):
            i[0] = (i[0] - 1) % len(imgs)
            render()
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    plt.show()


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/samples")
    imgs = sorted(target.glob("*.png")) if target.is_dir() else [target]
    if not imgs:
        sys.exit(f"no images at {target}")
    browse(imgs)
