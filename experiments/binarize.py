"""Turn a warped grayscale scan into a clean black-ink-on-white binary mask
(the GoodNotes / iPad look). Isolates handwriting + aruco markers, kills paper
texture and uneven lighting.

    python binarize.py scan.png                 # -> scan_bin.png
    python binarize.py scan.png out.png
    from binarize import binarize                # gray uint8 ndarray -> uint8 0/255
"""
import sys
import numpy as np
import cv2


def binarize(gray, block=41, C=15, min_area=6):
    """gray: HxW uint8. Returns HxW uint8 with 0=ink, 255=paper (like a scan)."""
    # local threshold beats a global one under uneven lighting / shadows
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, block | 1, C)  # 255 where ink
    # drop salt specks smaller than a pen stroke
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    ink = np.where(keep[lbl], 255, 0).astype(np.uint8)
    return 255 - ink  # ink black, paper white


def demo():
    g = np.full((60, 60), 200, np.uint8)          # paper
    g[20:40, 30:33] = 30                           # a thin ink stroke
    g[0, 0] = 20                                    # a 1px speck -> should die
    out = binarize(g, block=11, C=5, min_area=6)
    assert out[30, 31] == 0, "ink should be black"
    assert out[0, 0] == 255, "speck should be removed"
    assert out[5, 55] == 255, "paper should be white"
    print("ok")


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "demo":
        demo()
    else:
        src = sys.argv[1]
        dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + "_bin.png"
        gray = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(dst, binarize(gray))
        print(dst)
