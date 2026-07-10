"""Segment pages in data/samples with the FairScan model.
One image at a time; press spacebar to run the next inference and show it."""
import sys, glob, time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from PIL import Image
import segmentation_models_pytorch as smp

_ARUCO = cv2.aruco.ArucoDetector(
    cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50),
    cv2.aruco.DetectorParameters())


def upright(img):
    """Rotate a PIL RGB image so the page reads upright, using the 6x6 ArUco
    marker. The marker is printed upside-down relative to the upright page,
    so 'page up' is the marker's down direction. No marker -> assume upright."""
    corners, ids, _ = _ARUCO.detectMarkers(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY))
    if ids is None:
        return img
    c = corners[0][0]          # (4,2): TL,TR,BR,BL in the marker's own frame
    ux, uy = c[0] - c[3]       # marker-up = TL - BL, in y-down pixel coords
    # page-up = -marker-up; rotate CCW so page-up points visually up (90 deg)
    theta = round((90 - np.degrees(np.arctan2(uy, -ux))) / 90) * 90 % 360
    return img.rotate(theta, expand=True) if theta else img

MODEL = "model/fairscan-segmentation-model.pt"
SIZE = 512  # model input resolution
OPACITY = 0.5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)


def load_model():
    m = smp.DeepLabV3Plus(encoder_name="mobilenet_v2", encoder_weights=None,
                          in_channels=3, classes=1)
    m.load_state_dict(torch.load(MODEL, map_location="cpu", weights_only=False))
    return m.eval().to(DEVICE)


def mask_for(model, img):
    """Return HxW bool mask at the image's original resolution."""
    x = torch.from_numpy(np.asarray(img.resize((SIZE, SIZE)), np.float32) / 255.0).to(DEVICE)
    x = ((x.permute(2, 0, 1) - MEAN) / STD).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu()
    prob = Image.fromarray((prob.numpy() * 255).astype(np.uint8)).resize(img.size)
    mask = (np.asarray(prob) > 124).astype(np.uint8)
    # close fills holes inside the page, open trims overflow past its edges.
    # kernel ~1.5% of the short side; scale with image so it works at any size.
    k = max(3, int(0.015 * min(img.size)) | 1)  # odd, >=3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask.astype(bool)


def keep_marker_island(mask, img):
    """Keep only the mask island the ArUco marker sits inside, filled solid via
    its outer contour (drops interior holes). Other islands are discarded.
    No marker -> keep the largest island. mask and img must share a frame."""
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    corners, ids, _ = _ARUCO.detectMarkers(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY))
    chosen = None
    if ids is not None:
        cx, cy = corners[0][0].mean(axis=0)
        inside = [c for c in contours
                  if cv2.pointPolygonTest(c, (float(cx), float(cy)), False) >= 0]
        chosen = inside[0] if inside else None
    if chosen is None:
        chosen = max(contours, key=cv2.contourArea)
    filled = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(filled, [chosen], -1, 1, thickness=cv2.FILLED)
    return filled.astype(bool)


def page_quad(mask, img):
    """Reduce the page island to a 4-corner polygon. The marker's corners are
    added as guaranteed anchors (it sits on a page corner), convex-hulled with
    the mask outline, then approximated to a quad. Returns (4,2) float32 or None."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    pts = np.vstack([c.reshape(-1, 2) for c in cnts]).astype(np.float32)
    corners, ids, _ = _ARUCO.detectMarkers(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY))
    if ids is not None:
        pts = np.vstack([pts, corners[0][0]])  # anchor the marker's page corner
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    for f in np.linspace(0.01, 0.12, 30):  # loosen tolerance until it lands on 4
        a = cv2.approxPolyDP(hull, f * peri, True)
        if len(a) == 4:
            return a.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(hull))  # fallback: tightest enclosing rect


def quad_mask(quad, shape):
    """Fill a (4,2) quad into a solid bool mask of the given HxW shape."""
    m = np.zeros(shape, np.uint8)
    cv2.fillConvexPoly(m, quad.astype(np.int32), 1)
    return m.astype(bool)


def order_corners(pts):
    """Order 4 points as TL, TR, BR, BL via the sum/diff trick."""
    pts = np.asarray(pts, np.float32)
    s = pts.sum(1)
    d = (pts[:, 0] - pts[:, 1])  # x - y
    return np.array([pts[s.argmin()], pts[d.argmax()],
                     pts[s.argmax()], pts[d.argmin()]], np.float32)


def warp(img, quad):
    """Perspective-warp the quad region of img to a flat, front-on rectangle
    (the camscanner look). Output size matches the quad's own edge lengths."""
    q = order_corners(quad)
    tl, tr, br, bl = q
    W = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    H = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(np.asarray(img), M, (W, H))


def main(paths):
    t = time.perf_counter()
    model = load_model()  # loaded once, stays loaded
    print(f"model loaded on {DEVICE} in {time.perf_counter()-t:.2f}s")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("off")
    state = {"i": -1}

    def show_next():
        state["i"] += 1
        if state["i"] >= len(paths):
            plt.close(fig)
            return
        p = paths[state["i"]]
        img = upright(Image.open(p).convert("RGB"))
        t = time.perf_counter()
        mask = mask_for(model, img)  # inference happens here, one image
        mask = keep_marker_island(mask, img)
        quad = page_quad(mask, img)
        print(f"{p.split('/')[-1]}: inference {time.perf_counter()-t:.3f}s")
        ax.clear()
        ax.axis("off")
        ax.imshow(warp(img, quad) if quad is not None else img)  # flattened scan
        ax.set_title(f"{state['i']+1}/{len(paths)}  {p.split('/')[-1]}  (spacebar=next)")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == " ":
            show_next()

    fig.canvas.mpl_connect("key_press_event", on_key)
    show_next()  # first image
    plt.show()


if __name__ == "__main__":
    paths = sys.argv[1:] or sorted(glob.glob("data/samples/*.png"))
    main(paths)
