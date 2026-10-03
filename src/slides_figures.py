"""Figures for the final course presentation (presentation/figures/*.png).

Reads only existing results / masks; nothing is re-trained.
Usage (.venv-phase10):  python src/slides_figures.py
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import SimpleITK as sitk  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
OUT = os.path.join(ROOT, "presentation", "figures")
os.makedirs(OUT, exist_ok=True)

for f in ("GOTHIC.TTF", "GOTHICB.TTF"):
    p = os.path.join("C:/Windows/Fonts", f)
    if os.path.isfile(p):
        font_manager.fontManager.addfont(p)
plt.rcParams.update({"font.family": "Arial", "font.size": 13, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight"})

SITE_COL = {"LUNG1": "#C00000", "RADIOGENOMICS": "#2683C6", "LPCD": "#42BA97", "NLST": "#F49100"}
SITE_NAME = {"LUNG1": "LUNG1", "RADIOGENOMICS": "Radiogenomics", "LPCD": "Lung-PET-CT-Dx", "NLST": "NLST"}
TEAL, BLUE, RED, GREEN, GREY = "#1485A4", "#2683C6", "#C00000", "#42BA97", "#8C8C8C"
COHORT = pd.read_csv(os.path.join(ROOT, "cohort", "phase12_four_site_cohort.csv"))


def roc_per_site():
    z = []
    for b in ("shells", "loc_clin"):
        o = pd.read_csv(os.path.join(ROOT, "results", "phase13", "pooled", b + "__combat", "oof.csv"))
        o["z"] = o.groupby("seed").ensemble.transform(lambda s: (s - s.mean()) / s.std())
        z.append(o.set_index(["PatientKey", "seed"]).z)
    s = pd.concat(z, axis=1).mean(1).groupby(level=0).mean()
    d = COHORT.set_index("PatientKey").join(s.rename("score"))
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for site, g in d.groupby("site"):
        fpr, tpr, _ = roc_curve(g.label, g.score)
        ax.plot(fpr, tpr, color=SITE_COL[site], lw=2.6,
                label=f"{SITE_NAME[site]}  (AUC {roc_auc_score(g.label, g.score):.2f})")
    ax.plot([0, 1], [0, 1], "--", color=GREY, lw=1.2, label="Random guess (0.50)")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve inside each hospital", fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, frameon=False)
    ax.set_aspect("equal")
    fig.savefig(os.path.join(OUT, "roc_per_site.png"), transparent=True)
    plt.close(fig)


def pca_combat():
    import phase11_cv as C
    from phase10_data import load_cfg
    cfg = load_cfg()
    C.FEAT = os.path.join(ROOT, "global_features", "phase13")
    X, _ = C.block_matrix(COHORT[["PatientKey"]], "shells")
    y, site = COHORT.label.values, COHORT.site.values
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax, harm, ttl in zip(axs, ("none", "combat"), ("Before ComBat", "After ComBat")):
        Z = PCA(2, random_state=0).fit_transform(C.Prep(harm, cfg).fit(X, y, site).Xtr_)
        for s in ("LUNG1", "RADIOGENOMICS", "LPCD", "NLST"):
            m = site == s
            ax.scatter(Z[m, 0], Z[m, 1], s=11, alpha=0.55, color=SITE_COL[s], label=SITE_NAME[s])
        ax.set_title(ttl, fontweight="bold", fontsize=16)
        ax.set_xlabel("Principal component 1")
        ax.set_ylabel("Principal component 2")
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].legend(markerscale=2.5, frameon=False, fontsize=11, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pca_combat.png"), transparent=True)
    plt.close(fig)


def _slice_png(a, ax, title, vmin, vmax, aspect=1.0):
    ax.imshow(a, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)
    ax.set_title(title, fontsize=13)
    ax.axis("off")


def ct_before_after(pid="LUNG1-001"):
    root = "D:/User/Documents/L.U.N.G/Vuong5ROI/outputs/full_cohort"
    img = sitk.ReadImage(os.path.join(root, pid, "CT.nii.gz"), sitk.sitkFloat32)
    gtv = sitk.ReadImage(os.path.join(root, pid, "ROI_A_GTV.nii.gz"))
    a, g = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(gtv) > 0
    zc = int(np.argmax(g.sum((1, 2))))
    yc = int(np.argmax(g.sum((0, 2))))
    sp = img.GetSpacing()
    new_sp = (2.0, 2.0, 2.0)
    size = [int(round(n * s / t)) for n, s, t in zip(img.GetSize(), sp, new_sp)]
    r = sitk.Resample(img, size, sitk.Transform(), sitk.sitkLinear, img.GetOrigin(), new_sp,
                      img.GetDirection(), -1024.0)
    b = np.clip(sitk.GetArrayFromImage(r), -1024, 200)
    z2 = int(round(zc * sp[2] / 2.0))
    y2 = int(round(yc * sp[1] / 2.0))
    fig, axs = plt.subplots(2, 2, figsize=(9.5, 8.6))
    _slice_png(a[zc], axs[0, 0], f"BEFORE – axial\n{sp[0]:.2f}×{sp[1]:.2f} mm pixels, full HU range",
               a.min(), a.max())
    _slice_png(b[z2], axs[0, 1], "AFTER – axial\n2×2 mm pixels, HU clipped to [−1024, 200]", -1024, 200)
    _slice_png(a[::-1, yc, :], axs[1, 0], f"BEFORE – front view\n{sp[2]:.1f} mm between slices (squashed)",
               a.min(), a.max())
    _slice_png(b[::-1, y2, :], axs[1, 1], "AFTER – front view\n2 mm cubes in every direction", -1024, 200)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ct_before_after.png"), transparent=True)
    plt.close(fig)


def _crop(m, pad):
    ys, xs = np.nonzero(m)
    return slice(max(ys.min() - pad, 0), ys.max() + pad), slice(max(xs.min() - pad, 0), xs.max() + pad)


def seg_gallery(key=None):
    v = pd.read_csv(os.path.join(ROOT, "results", "phase13", "medsam2_validation_latest_crop0_mid_n343.csv"))
    v = v[v.site == "RADIOGENOMICS"].copy()
    v["gain"] = v.ms2_tight - v.v5_tight
    if key is None:
        cand = v[(v.ms2_tight.between(0.86, 0.92))].sort_values("gain", ascending=False)
        key = cand.PatientKey.iloc[0]
    row = v.set_index("PatientKey").loc[key]
    pid = key.split("::")[1]
    root = os.path.join(ROOT, "external", "nsclc_radiogenomics", "masks", pid)
    kk = key.replace("::", "__")
    ct = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, "CT.nii.gz"), sitk.sitkFloat32))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(root, "ROI_A_GTV.nii.gz"))) > 0
    box = sitk.GetArrayFromImage(sitk.ReadImage(f"C:/LUNG_phase12/auto_masks_v2/{kk}/BOX.nii.gz")) > 0
    v5 = sitk.GetArrayFromImage(sitk.ReadImage(f"C:/LUNG_phase12/auto_masks_v2/{kk}/AUTO_GTV.nii.gz")) > 0
    ms = sitk.GetArrayFromImage(sitk.ReadImage(f"C:/LUNG_phase13/medsam2_masks/{kk}/AUTO_GTV.nii.gz")) > 0
    z = int(np.argmax(gt.sum((1, 2))))
    sy, sx = _crop(box[z] | gt[z], 25)
    panels = [("1. CT + box prompt", None, "#F49100"),
              (f"2. Old method: threshold\n(Dice {row.v5_tight:.2f})", v5, BLUE),
              (f"3. MedSAM2 (ours)\n(Dice {row.ms2_tight:.2f})", ms, GREEN),
              ("4. Expert outline\n(reference)", gt, RED)]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.4))
    for ax, (ttl, m, col) in zip(axs, panels):
        ax.imshow(np.clip(ct[z][sy, sx], -1000, 200), cmap="gray")
        if m is None:
            bb = box[z][sy, sx]
            ys, xs = np.nonzero(bb)
            ax.add_patch(plt.Rectangle((xs.min(), ys.min()), xs.max() - xs.min(), ys.max() - ys.min(),
                                       fill=False, ec=col, lw=2.5))
        else:
            ax.contour(m[z][sy, sx], [0.5], colors=[col], linewidths=2.5)
        ax.set_title(ttl, fontsize=14)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "seg_gallery.png"), transparent=True)
    plt.close(fig)
    return key


def shells_on_ct(key):
    pid = key.split("::")[1]
    root = os.path.join(ROOT, "external", "nsclc_radiogenomics", "masks", pid)
    img = sitk.ReadImage(os.path.join(root, "CT.nii.gz"), sitk.sitkFloat32)
    ct = sitk.GetArrayFromImage(img)
    ms = sitk.GetArrayFromImage(sitk.ReadImage(
        f"C:/LUNG_phase13/medsam2_masks/{key.replace('::', '__')}/AUTO_GTV.nii.gz")) > 0
    sp = img.GetSpacing()[::-1]
    d = ndi.distance_transform_edt(~ms, sampling=sp)
    z = int(np.argmax(ms.sum((1, 2))))
    sy, sx = _crop(ms[z], 30)
    fig, ax = plt.subplots(figsize=(5.4, 5.4))
    ax.imshow(np.clip(ct[z][sy, sx], -1000, 200), cmap="gray")
    ov = np.zeros(ms[z][sy, sx].shape + (4,))
    cols = [(ms[z][sy, sx], "#C00000"), ((d[z] > 0) & (d[z] <= 4), "#F49100"),
            ((d[z] > 4) & (d[z] <= 8), "#42BA97"), ((d[z] > 8) & (d[z] <= 12), "#2683C6")]
    for m, c in cols:
        mm = m[sy, sx] if m.shape != ov.shape[:2] else m
        ov[mm] = matplotlib.colors.to_rgba(c, 0.55)
    ax.imshow(ov)
    ax.axis("off")
    fig.savefig(os.path.join(OUT, "shells_on_ct.png"), transparent=True)
    plt.close(fig)


def external_ci():
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    groups = [("Lung3  (n = 79)", 0.686, (0.557, 0.807), 18, 0.903, 61, 0.676),
              ("TCGA  (n = 84)", 0.585, (0.462, 0.707), 29, 0.726, 55, 0.511)]
    w = 0.27
    for i, (nm, a, ci, n_thin, thin, n_thick, thick) in enumerate(groups):
        ax.bar(i - w, a, w * 0.94, color=TEAL, label="All patients (with 95% CI)" if i == 0 else None, zorder=2)
        ax.errorbar(i - w, a, yerr=[[a - ci[0]], [ci[1] - a]], color="#333333", capsize=7, lw=1.6, zorder=3)
        ax.bar(i, thin, w * 0.94, color=GREEN, label="Thin slices (≤ 2.5 mm)" if i == 0 else None, zorder=2)
        ax.bar(i + w, thick, w * 0.94, color="#F49100", label="Thick slices (> 2.5 mm)" if i == 0 else None, zorder=2)
        ax.text(i - w, ci[1] + 0.02, f"{a:.3f}", va="bottom", ha="center", fontsize=14, fontweight="bold", color=TEAL, bbox=dict(fc="white", ec="none", pad=1.5))
        ax.text(i, thin + 0.015, f"{thin:.2f}", ha="center", fontsize=14, fontweight="bold", color="#2E8B6E", bbox=dict(fc="white", ec="none", pad=1.5))
        ax.text(i + w, thick + 0.015, f"{thick:.2f}", ha="center", fontsize=14, fontweight="bold", color="#C06F00", bbox=dict(fc="white", ec="none", pad=1.5))
        ax.text(i, 0.315, f"n = {n_thin}", ha="center", fontsize=10, color="white")
        ax.text(i + w, 0.315, f"n = {n_thick}", ha="center", fontsize=10, color="white")
    x_end = 1.45
    ax.axhline(0.5, ls="--", color=GREY, lw=1.3, zorder=1)
    ax.text(x_end + 0.03, 0.5, "random\n(0.50)", color=GREY, fontsize=11, ha="left", va="center")
    ax.axhline(0.714, ls=":", color=RED, lw=1.8, zorder=1)
    ax.text(x_end + 0.03, 0.714, "unseen-hospital\nCV (0.71)", color=RED, fontsize=11, ha="left", va="center")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([g[0] for g in groups], fontsize=14, fontweight="bold")
    ax.set_ylim(0.3, 1.0)
    ax.set_xlim(-0.5, x_end)
    ax.set_ylabel("ROC-AUC", fontsize=13)
    ax.legend(frameon=False, fontsize=12, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0),
              handlelength=1.2, columnspacing=1.2)
    fig.savefig(os.path.join(OUT, "external_ci.png"), transparent=True)
    plt.close(fig)


if __name__ == "__main__" and len(sys.argv) > 1:
    globals()[sys.argv[1]]()
elif __name__ == "__main__":
    roc_per_site()
    print("roc done", flush=True)
    external_ci()
    ct_before_after()
    print("ct done", flush=True)
    k = seg_gallery()
    shells_on_ct(k)
    print("seg done", k, flush=True)
    pca_combat()
    print("pca done", flush=True)
