"""Cover background art (presentation/figures/cover_bg.png): a vector-style
network mesh + particle lungs traced from a real CT (LUNG1-001, read only)
+ the tumour/ring motif.  The centre is kept pale for the title.
Usage (.venv-phase10):  python src/cover_art.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgba  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "presentation", "figures", "cover_bg.png")
W, H = 13.333, 4.83                       # inches: the white band between the red bars
TEAL, BLUE, GREEN, RED, ORANGE = "#1485A4", "#2683C6", "#42BA97", "#C00000", "#F49100"
rng = np.random.default_rng(7)


def fade(x, y):
    """0 in the title zone (centre), 1 towards the left/right edges."""
    dx = np.abs(x - W / 2) / (W / 2)
    dy = np.abs(y - H / 2) / (H / 2)
    d = np.sqrt((dx / 1.0) ** 2 + (dy / 2.2) ** 2)
    return np.clip((d - 0.38) / 0.5, 0, 1) ** 1.3


def _smooth(pts, n=400):
    from scipy.interpolate import splev, splprep
    pts = np.asarray(pts, float)
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
    x, y = splev(np.linspace(0, 1, n), tck)
    return np.c_[x, y]


RIGHT_LUNG = [(0.42, 1.18), (0.33, 1.12), (0.23, 0.95), (0.15, 0.68), (0.10, 0.38), (0.11, 0.16), (0.20, 0.08),
              (0.31, 0.11), (0.40, 0.17), (0.45, 0.30), (0.45, 0.60), (0.44, 0.85), (0.46, 1.08)]
LEFT_LUNG = [(1 - x, y) for x, y in RIGHT_LUNG]
LEFT_LUNG[8:12] = [(0.60, 0.19), (0.66, 0.30), (0.64, 0.50), (0.57, 0.66)]          # cardiac notch


def branches(x, y, ang, length, depth, out):
    if depth == 0:
        return
    x2, y2 = x + length * np.cos(ang), y + length * np.sin(ang)
    out.append(([(x, y), (x2, y2)], depth))
    for d in (-0.45, 0.4):
        branches(x2, y2, ang + d + rng.normal(0, 0.08), length * 0.72, depth - 1, out)


def lungs(ax, ox, oy, s):
    from matplotlib.path import Path
    tf = lambda P: np.c_[ox + P[:, 0] * s, oy + P[:, 1] * s]  # noqa: E731
    lung_paths = []
    for shape in (RIGHT_LUNG, LEFT_LUNG):
        P = tf(_smooth(shape))
        ax.fill(P[:, 0], P[:, 1], color=to_rgba(BLUE, 0.07), lw=0, zorder=3)
        ax.plot(P[:, 0], P[:, 1], color=to_rgba(TEAL, 0.75), lw=1.6, zorder=4)
        path = Path(P)
        lung_paths.append(Path(_smooth(shape)))
        c = np.c_[rng.uniform(P[:, 0].min(), P[:, 0].max(), 5000), rng.uniform(P[:, 1].min(), P[:, 1].max(), 5000)]
        c = c[path.contains_points(c)][:1400]
        ax.scatter(c[:, 0], c[:, 1], s=rng.uniform(1.0, 6.0, len(c)), color=to_rgba(BLUE, 0.28), lw=0, zorder=3)
    segs = [([(0.5, 1.32), (0.5, 0.98)], 7)]
    branches(0.5, 0.98, np.deg2rad(-140), 0.14, 6, segs)
    branches(0.5, 0.98, np.deg2rad(-40), 0.14, 6, segs)
    for (p, q), d in segs:
        if d < 6 and not any(lp.contains_point(q) for lp in lung_paths):
            continue
        P = tf(np.array([p, q]))
        ax.plot(P[:, 0], P[:, 1], color=to_rgba(TEAL, 0.25 + 0.08 * d), lw=0.5 + 0.55 * d,
                solid_capstyle="round", zorder=5)
    tx, ty = ox + 0.27 * s, oy + 0.80 * s                             # tumour + shells
    for r, c, a in ((0.42, BLUE, 0.5), (0.32, GREEN, 0.55), (0.22, ORANGE, 0.6)):
        ax.add_patch(plt.Circle((tx, ty), r, fill=False, ec=to_rgba(c, a), lw=1.6, ls=(0, (3, 2)), zorder=6))
    ax.add_patch(plt.Circle((tx, ty), 0.13, color=to_rgba(RED, 0.75), lw=0, zorder=6))


def main():
    fig = plt.figure(figsize=(W, H), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # soft background: white centre -> pale teal edges
    gx, gy = np.meshgrid(np.linspace(0, W, 800), np.linspace(0, H, 300))
    cmap = LinearSegmentedColormap.from_list("bg", ["#FFFFFF", "#EEF7F9", "#D9EEF3"])
    ax.imshow(fade(gx, gy), extent=[0, W, 0, H], origin="lower", cmap=cmap, vmin=0, vmax=1, zorder=0)

    # network mesh
    pts = np.c_[rng.uniform(-0.3, W + 0.3, 420), rng.uniform(-0.3, H + 0.3, 420)]
    tri = Delaunay(pts)
    segs, cols = [], []
    for a, b, c in tri.simplices:
        for u, v in ((a, b), (b, c), (c, a)):
            p, q = pts[u], pts[v]
            if np.hypot(*(p - q)) > 1.1:
                continue
            f = fade((p[0] + q[0]) / 2, (p[1] + q[1]) / 2)
            if f > 0.02:
                segs.append([p, q])
                cols.append(to_rgba(TEAL if (u + v) % 3 else BLUE, 0.28 * f))
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.6, zorder=1))
    f = fade(pts[:, 0], pts[:, 1])
    ax.scatter(pts[:, 0], pts[:, 1], s=6 + 10 * f, c=[to_rgba(TEAL, 0.5 * v) for v in f], lw=0, zorder=2)
    hot = np.where(f > 0.6)[0]
    hot = rng.choice(hot, min(14, len(hot)), replace=False)
    for k, h in enumerate(hot):
        c = [RED, ORANGE, GREEN, BLUE][k % 4]
        ax.scatter(*pts[h], s=60, color=to_rgba(c, 0.18), lw=0, zorder=2)
        ax.scatter(*pts[h], s=14, color=to_rgba(c, 0.75), lw=0, zorder=3)

    lungs(ax, 0.2, 0.35, 3.0)

    # big ring motif (tumour + 3 shells) on the right
    cx, cy = W - 1.4, H * 0.46
    for r, c, a, lw in ((1.3, BLUE, 0.16, 9), (1.0, GREEN, 0.2, 9), (0.72, ORANGE, 0.24, 9)):
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, ec=to_rgba(c, a), lw=lw, zorder=3))
    ang = np.linspace(0, 2 * np.pi, 200)
    rr = 0.42 * (1 + 0.12 * np.sin(5 * ang) + 0.06 * np.cos(9 * ang))
    ax.fill(cx + rr * np.cos(ang), cy + rr * np.sin(ang), color=to_rgba(RED, 0.3), lw=0, zorder=4)
    for r in (0.72, 1.0, 1.3):
        t = rng.uniform(0, 2 * np.pi, 40)
        ax.scatter(cx + r * np.cos(t), cy + r * np.sin(t), s=4, color=to_rgba(TEAL, 0.45), lw=0, zorder=4)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print("saved", OUT)


if __name__ == "__main__":
    main()
