"""Figures for the final project report (report/figures/*.png).

Diagrams (pipeline, architecture, CV scheme, test protocol, Gantt) are drawn
here; result plots are read from the frozen result files (read only).
Usage (.venv-phase10):  python src/report_figures.py [name ...]
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "mathtext.fontset": "stix"})
TEAL, BLUE, GREEN, RED, ORANGE, GREY, DARK = "#1485A4", "#2683C6", "#42BA97", "#C00000", "#F49100", "#8C8C8C", "#0E5A70"
SITES = ["LUNG1", "RADIOGENOMICS", "LPCD", "NLST"]
SNAME = {"LUNG1": "LUNG1", "RADIOGENOMICS": "Radiogenomics", "LPCD": "Lung-PET-CT-Dx", "NLST": "NLST"}
SCOL = {"LUNG1": RED, "RADIOGENOMICS": BLUE, "LPCD": GREEN, "NLST": ORANGE}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), facecolor="white")
    plt.close(fig)
    print("saved", name)


def sb_weights(site, y):
    key = pd.Series(site).astype(str) + "|" + pd.Series(y).astype(str)
    w = 1.0 / key.map(key.value_counts()).values
    return w * len(w) / w.sum()


def sb_auc(y, s, site):
    return float(roc_auc_score(y, s, sample_weight=sb_weights(site, y)))


# ------------------------------------------------------------------ drawing
def box(ax, x, y, w, h, text, fc, tc="white", fs=9.5, bold=True, ec=None, sub=None, r=0.08):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc,
                                ec=ec or fc, lw=1.2, zorder=2))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", color=tc, fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=3, linespacing=1.1)
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center", color=tc, fontsize=fs - 1.8,
                zorder=3, linespacing=1.1)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=3, linespacing=1.15)


def arrow(ax, p, q, color="#444444", lw=1.4, style="-|>", ms=11, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 linestyle=ls, zorder=1, shrinkA=0, shrinkB=0))


def canvas(w, h, W, H):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return fig, ax


# ----------------------------------------------------------------- diagrams
def pipeline():
    fig, ax = canvas(7.0, 3.0, 10, 4.29)
    w, h, gap = 1.6, 1.0, 0.39
    xs = [0.1 + i * (w + gap) for i in range(5)]
    top = [("Chest CT", "DICOM series\n100-600 slices", BLUE),
           ("Preprocessing", "NIfTI, 2 mm isotropic,\nHU clip [-1024, 200]", BLUE),
           ("Tumour detection", "expert box or\nTotalSegmentator", DARK),
           ("Segmentation", "MedSAM2 from\none box prompt", DARK),
           ("Peritumoral shells", "0-4, 4-8, 8-12 mm\n(distance transform)", DARK)]
    bot = [("Feature extraction", "texture (279) +\nposition/clinical (13)", "#2E9E7E"),
           ("ComBat", "remove hospital\nshift and scale", "#2E9E7E"),
           ("Feature selection", "rank, prune |r|>0.9,\ntop-k", "#2E9E7E"),
           ("Classifier ensemble", "4 models x 2 blocks", RED),
           ("ADC / SCC", "fused score\ns > 0 : ADC", RED)][::-1]
    yt, yb = 2.75, 0.75
    for x, (t, s, c) in zip(xs, top):
        box(ax, x, yt, w, h, t, c, sub=s)
    for x, (t, s, c) in zip(xs, bot):
        box(ax, x, yb, w, h, t, c, sub=s)
    for i in range(4):
        arrow(ax, (xs[i] + w, yt + h / 2), (xs[i + 1], yt + h / 2))
        arrow(ax, (xs[i + 1], yb + h / 2), (xs[i] + w, yb + h / 2))
    arrow(ax, (xs[4] + w / 2, yt), (xs[4] + w / 2, yb + h))
    # bracket labels
    for x0, x1, lab, c in ((xs[0], xs[1] + w, "Image preparation", BLUE), (xs[2], xs[4] + w, "Tumour and regions", DARK)):
        ax.plot([x0, x0, x1, x1], [yt + h + 0.12, yt + h + 0.22, yt + h + 0.22, yt + h + 0.12], color=c, lw=1)
        ax.text((x0 + x1) / 2, yt + h + 0.32, lab, ha="center", va="bottom", fontsize=9.5, color=c, style="italic")
    for x0, x1, lab, c in ((xs[2], xs[4] + w, "Features (fitted on training data only)", "#2E9E7E"),
                           (xs[0], xs[1] + w, "Model", RED)):
        ax.plot([x0, x0, x1, x1], [yb - 0.12, yb - 0.22, yb - 0.22, yb - 0.12], color=c, lw=1)
        ax.text((x0 + x1) / 2, yb - 0.32, lab, ha="center", va="top", fontsize=9.5, color=c, style="italic")
    save(fig, "fig_pipeline")


def architecture():
    fig, ax = canvas(7.0, 3.9, 10, 5.57)
    ys = [3.25, 0.55]
    for yb, (name, dim) in zip(ys, [("Shell features", "1041 x 279"), ("Position + clinical", "1041 x 13")]):
        box(ax, 0.05, yb + 0.45, 1.45, 0.95, name, TEAL, sub=dim, fs=9.5)
        steps = ["Median imputation", "ComBat", "z-score", "Rank (within-site AUC)", "Prune |r| > 0.9, top-k"]
        px, pw, ph = 1.85, 1.95, 0.32
        ax.add_patch(FancyBboxPatch((px - 0.08, yb - 0.02), pw + 0.16, 1.92, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="#EEF7F9", ec=TEAL, lw=1, ls="--"))
        ax.text(px + pw / 2, yb + 1.8, "Preparation (train folds only)", ha="center", va="center", fontsize=8.5,
                color=DARK, style="italic")
        for i, s in enumerate(steps):
            box(ax, px, yb + 1.38 - i * 0.355, pw, ph, s, "white", tc=DARK, ec=TEAL, fs=8.5, bold=False, r=0.05)
            if i:
                arrow(ax, (px + pw / 2, yb + 1.38 - (i - 1) * 0.355), (px + pw / 2, yb + 1.38 - i * 0.355 + ph), lw=0.9, ms=7)
        arrow(ax, (1.5, yb + 0.92), (px - 0.08, yb + 0.92))
        models = ["Logistic regression (L2)", "SVM (RBF kernel)", "LightGBM (boosted trees)", "Ridge LR (all features)"]
        mx, mw, mh = 4.45, 2.15, 0.36
        for i, m in enumerate(models):
            yy = yb + 1.45 - i * 0.47
            box(ax, mx, yy, mw, mh, m, "#FDECEC", tc=RED, ec=RED, fs=8.8, bold=False, r=0.05)
            arrow(ax, (px + pw + 0.08, yb + 0.92), (mx, yy + mh / 2), lw=0.9, ms=7)
            arrow(ax, (mx + mw, yy + mh / 2), (7.25, yb + 0.92), lw=0.9, ms=7)
        ax.add_patch(plt.Circle((7.55, yb + 0.92), 0.3, color=TEAL, zorder=2))
        ax.text(7.55, yb + 0.92, "mean\nof z", ha="center", va="center", color="white", fontsize=8.5, zorder=3,
                fontweight="bold")
        ax.text(7.55, yb + 0.47, "block score", ha="center", va="top", fontsize=8.5, color=DARK, style="italic")
    ax.add_patch(FancyBboxPatch((8.35, 1.75), 1.6, 2.0, boxstyle="round,pad=0,rounding_size=0.08", fc=RED, ec=RED))
    ax.text(9.15, 3.4, "Final score s", ha="center", va="center", color="white", fontsize=10.5, fontweight="bold")
    ax.text(9.15, 2.75, "mean of the two\nz-scored block\nscores", ha="center", va="center", color="white", fontsize=8.8)
    ax.text(9.15, 2.05, "s > 0 : ADC\ns < 0 : SCC", ha="center", va="center", color="white", fontsize=9.5,
            fontweight="bold")
    for yb in ys:
        arrow(ax, (7.85, yb + 0.92), (8.35, 2.75))
    save(fig, "fig_architecture")


def cv_scheme():
    fig, ax = canvas(7.0, 3.3, 10, 4.71)
    ax.text(0.05, 4.55, "(a) Repeated nested cross-validation (5 seeds x 5 outer folds)", fontsize=10.5, fontweight="bold", va="top")
    w, h = 0.98, 0.42
    for f in range(5):
        x = 0.25 + f * (w + 0.08)
        test = f == 2
        box(ax, x, 3.45, w, h, "Test" if test else "Train", ORANGE if test else TEAL, fs=9)
    ax.text(0.25 + 2.5 * (w + 0.08), 3.3, "outer fold k (repeated for k = 1...5 and seeds 42-46 = 25 runs)",
            ha="center", va="top", fontsize=8.5, style="italic")
    arrow(ax, (1.3, 3.0), (1.3, 2.55))
    for f in range(3):
        x = 0.25 + f * 1.5
        box(ax, x, 2.0, 1.42, 0.42, "Inner validation" if f == 1 else "Inner train", GREEN if f == 1 else "#7FB8C9", fs=8.5)
    ax.text(0.25 + 2.25, 1.85, "inner 3-fold CV on the training part only:\nchooses k, C, leaves by site-balanced AUC",
            ha="center", va="top", fontsize=8.5, style="italic")
    # LOSO
    ax.text(5.75, 4.55, "(b) Leave-one-hospital-out (LOSO)", fontsize=10.5, fontweight="bold", va="top")
    names = ["LUNG1", "Radiogen.", "LPCD", "NLST"]
    cw, ch = 0.85, 0.5
    for j, n in enumerate(names):
        ax.text(6.05 + j * (cw + 0.07) + cw / 2, 4.1, n, ha="center", va="center", fontsize=8.5)
    for r in range(4):
        yy = 3.45 - r * (ch + 0.12)
        ax.text(5.95, yy + ch / 2, f"run {r + 1}", ha="right", va="center", fontsize=8.5)
        for j in range(4):
            held = r == j
            box(ax, 6.05 + j * (cw + 0.07), yy, cw, ch, "test" if held else "train",
                GREY if held else list(SCOL.values())[j], fs=8.5, bold=held)
    ax.text(7.85, 0.95, "train on 3 hospitals, test on the 4th;\nthe test hospital is mapped by label-free ComBat",
            ha="center", va="top", fontsize=8.5, style="italic")
    save(fig, "fig_cv_scheme")


def protocol():
    fig, ax = canvas(7.0, 2.25, 10, 3.21)
    steps = [("1. Pre-register", "plan + primary\nmetric written", BLUE),
             ("2. Prepare data", "images only;\nlabels sealed", BLUE),
             ("3. Run frozen\npipeline", "TotalSegmentator\n+ MedSAM2 + model", DARK),
             ("4. Freeze", "save predictions\n+ SHA-256 hash", DARK),
             ("5. Unseal labels", "read exactly once", RED),
             ("6. Score once", "AUC + 95% CI;\nno re-runs", RED)]
    w, h, gap = 1.45, 1.35, 0.2
    for i, (t, s, c) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.0), w, h, boxstyle="round,pad=0,rounding_size=0.08", fc=c, ec=c))
        ax.text(x + w / 2, 1.0 + h * 0.7, t, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold",
                linespacing=1.05)
        ax.text(x + w / 2, 1.0 + h * 0.28, s, ha="center", va="center", color="white", fontsize=8.3, linespacing=1.1)
        if i:
            arrow(ax, (x - gap, 1.0 + h / 2), (x, 1.0 + h / 2))
    ax.plot([0.1 + 4 * (w + gap) - gap / 2] * 2, [0.65, 2.75], color=RED, lw=2.2, ls=(0, (4, 2)))
    ax.text(0.1 + 4 * (w + gap) - gap / 2, 2.85, "label wall", ha="center", va="bottom", color=RED, fontsize=9, style="italic")
    ax.text(5, 0.45, "No retraining, no tuning and no manual input at any step (applied to Lung3 and TCGA)",
            ha="center", va="center", fontsize=9, style="italic")
    save(fig, "fig_protocol")


def gantt():
    weeks = ["Mid-\nbreak", "W8", "W9", "W10", "W11", "W12", "W13", "W14"]
    tasks = [
        ("Literature review and topic selection", 0, 1, BLUE),
        ("LUNG1 cohort, preprocessing, patch extraction", 0, 1, BLUE),
        ("Attention-MIL experiments (Gradient, GTV+rim)", 1, 2, TEAL),
        ("Feature selection, CNN-MIL, global radiomics", 2, 3, TEAL),
        ("Second cohort (Radiogenomics) + progress talk", 3, 4, TEAL),
        ("Shortcut analysis, fair metric, ComBat, shells", 4, 5, GREEN),
        ("Lung-PET-CT-Dx and NLST added (1041 pts)", 5, 6, GREEN),
        ("MedSAM2 masks, foundation models, adversarial FT", 6, 7, GREEN),
        ("Locked external tests (TCGA, Lung3)", 6, 7, RED),
        ("Slides, final presentation and report", 7, 8, ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    for i, (t, a, b, c) in enumerate(tasks):
        ax.barh(i, b - a, left=a, color=c, height=0.62, edgecolor="white")
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([t[0] for t in tasks], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(len(weeks)) + 0.5)
    ax.set_xticklabels(weeks, fontsize=9.5)
    ax.set_xlim(0, len(weeks))
    ax.xaxis.tick_top()
    ax.grid(axis="x", color="#DDDDDD", lw=0.8)
    for x in range(1, len(weeks)):
        ax.axvline(x, color="#DDDDDD", lw=0.8, zorder=0)
    ax.axvline(3.5, color=RED, lw=1.4, ls="--")
    ax.text(3.5, len(tasks) - 0.35, "progress\npresentation", color=RED, fontsize=8.5, ha="center", va="top")
    ax.axvline(7.5, color=DARK, lw=1.4, ls="--")
    ax.text(7.5, len(tasks) - 0.35, "final\ndemo", color=DARK, fontsize=8.5, ha="center", va="top")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_gantt")


# ------------------------------------------------------------------- plots
def shortcut():
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    labs = ["Hospital name only\n(no image)", "Naive radiomics\nmodel", "Final model\n(this work)"]
    raw, fair = [0.710, 0.727, np.nan], [0.497, 0.642, 0.722]
    x = np.arange(3)
    b1 = ax.bar(x - 0.19, raw, 0.36, color=RED, label="Standard pooled AUC")
    b2 = ax.bar(x + 0.19, fair, 0.36, color=TEAL, label="Site-balanced AUC")
    for bars in (b1, b2):
        for r in bars:
            if np.isfinite(r.get_height()):
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.008, f"{r.get_height():.3f}",
                        ha="center", va="bottom", fontsize=9.5)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.text(-0.45, 0.505, "chance", color=GREY, fontsize=9, ha="left", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(0.4, 0.8)
    ax.set_ylabel("ROC-AUC")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", ncol=2)
    save(fig, "fig_shortcut")


def dataset():
    names = ["LUNG1", "Radio-\ngenomics", "Lung-PET-\nCT-Dx", "NLST", "Lung3\n(locked)", "TCGA\n(locked)"]
    adc = np.array([51, 112, 244, 290, 44, 48])
    scc = np.array([151, 29, 57, 107, 35, 36])
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x = np.arange(6)
    for i in range(6):
        hatch = "///" if i >= 4 else None
        ax.bar(x[i], adc[i] / (adc[i] + scc[i]) * 100, color=GREEN, hatch=hatch, edgecolor="white")
        ax.bar(x[i], scc[i] / (adc[i] + scc[i]) * 100, bottom=adc[i] / (adc[i] + scc[i]) * 100, color=ORANGE,
               hatch=hatch, edgecolor="white")
        pa = adc[i] / (adc[i] + scc[i]) * 100
        ax.text(x[i], pa / 2, f"{adc[i]}", ha="center", va="center", color="white", fontweight="bold")
        ax.text(x[i], pa + (100 - pa) / 2, f"{scc[i]}", ha="center", va="center", color="white", fontweight="bold")
    ax.axvline(3.5, color=GREY, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5)
    ax.set_ylabel("Share of patients (%)")
    ax.set_ylim(0, 100)
    ax.bar(0, 0, color=GREEN, label="ADC")
    ax.bar(0, 0, color=ORANGE, label="SCC")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=9.5)
    save(fig, "fig_dataset")


def dice():
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    groups = ["LUNG1\ntight box", "Radiogenomics\ntight box", "LUNG1\nloose box (+8 mm)", "Radiogenomics\nloose box (+8 mm)"]
    old = [0.812, 0.830, 0.572, 0.680]
    new = [0.843, 0.850, 0.738, 0.743]
    x = np.arange(4)
    b1 = ax.bar(x - 0.19, old, 0.36, color=BLUE, label="HU-threshold method (v5)")
    b2 = ax.bar(x + 0.19, new, 0.36, color=GREEN, label="MedSAM2 (final)")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.2f}", ha="center",
                    va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("Mean Dice vs expert GTV")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    save(fig, "fig_dice")


def siteleak():
    blocks = [("gtv", "Tumour texture"), ("ring_in", "Inner ring"), ("shells", "Shells"), ("loc_clin", "Position +\nclinical")]
    before, after = [], []
    for b, _ in blocks:
        before.append(json.load(open(os.path.join(ROOT, "results/phase11/pooled", f"{b}__none", "metrics.json")))["site_leak_auc"])
        after.append(json.load(open(os.path.join(ROOT, "results/phase11/pooled", f"{b}__combat", "metrics.json")))["site_leak_auc"])
    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    x = np.arange(len(blocks))
    b1 = ax.bar(x - 0.19, before, 0.36, color=RED, label="Before ComBat")
    b2 = ax.bar(x + 0.19, after, 0.36, color=TEAL, label="After ComBat")
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.01, f"{r.get_height():.2f}", ha="center",
                    va="bottom", fontsize=9)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([b[1] for b in blocks])
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Hospital-detector AUC")
    ax.legend(frameon=False, fontsize=9.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    save(fig, "fig_siteleak")
    return dict(zip([b for b, _ in blocks], zip(before, after)))


def combat_demo():
    """One shell feature, all 1041 patients: per-hospital density before/after ComBat (illustration)."""
    from scipy.stats import gaussian_kde
    from phase10_harmonise import Harmoniser
    coh = pd.read_csv(os.path.join(ROOT, "cohort", "phase12_four_site_cohort.csv"))
    rad = pd.read_csv(os.path.join(ROOT, "global_features/phase13/radiomics.csv"))
    rad = rad[rad.status == "ok"].set_index("PatientKey")
    cols = [c for c in rad.columns if c.startswith("rad__shell")]
    X = rad[cols].apply(pd.to_numeric, errors="coerce").reindex(coh.PatientKey).values
    X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
    keep = X.std(0) > 1e-8
    X, cols = X[:, keep], [c for c, k in zip(cols, keep) if k]
    site = coh.site.values
    name = "rad__shell1__original_glcm_JointEntropy" if "rad__shell1__original_glcm_JointEntropy" in cols else cols[0]
    j = cols.index(name)
    H = Harmoniser("combat", True).fit(X, site, coh.label.values)
    Xh = H.transform(X, site)
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    for ax, M, ttl in ((axs[0], X, "Before ComBat"), (axs[1], Xh, "After ComBat")):
        v = M[:, j]
        grid = np.linspace(np.percentile(X[:, j], 0.5), np.percentile(X[:, j], 99.5), 300)
        for s in SITES:
            k = gaussian_kde(v[site == s])
            ax.plot(grid, k(grid), color=SCOL[s], lw=1.8, label=SNAME[s])
            ax.fill_between(grid, k(grid), color=SCOL[s], alpha=0.08)
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel("10th-percentile HU in the 0-4 mm shell", fontsize=9.5)
    axs[0].set_ylabel("Density")
    axs[1].legend(frameon=False, fontsize=8.5)
    save(fig, "fig_combat_demo")
    return name


def load_oof(block):
    d = pd.read_csv(os.path.join(ROOT, "results/phase13/pooled", f"{block}__combat", "oof.csv"))
    return d


def models_vs_ensemble():
    rows = []
    fused = []
    for blk, lab in (("shells", "Shells block"), ("loc_clin", "Position + clinical block")):
        d = load_oof(blk)
        for m, ml in (("logreg", "Logistic reg."), ("svm", "SVM (RBF)"), ("lgbm", "LightGBM"), ("ridge_all", "Ridge (all)"),
                      ("ensemble", "Ensemble of 4")):
            v = np.mean([sb_auc(g.y.values, g[m].values, g.site.values) for _, g in d.groupby("seed")])
            rows.append((lab, ml, v))
        d = d.copy()
        d["z"] = d.groupby("seed")["ensemble"].transform(lambda v: (v - v.mean()) / v.std())
        fused.append(d.set_index(["seed", "idx"]).sort_index())
    z = (fused[0]["z"] + fused[1]["z"]) / 2
    base = fused[0]
    fv = np.mean([sb_auc(base.loc[s].y.values, g.values, base.loc[s].site.values) for s, g in z.groupby(level=0)])
    df = pd.DataFrame(rows, columns=["block", "model", "auc"])
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    mods = df.model.unique()
    x = np.arange(len(mods))
    for k, (blk, c) in enumerate((("Shells block", TEAL), ("Position + clinical block", ORANGE))):
        vals = df[df.block == blk].set_index("model").loc[mods, "auc"].values
        bars = ax.bar(x + (k - 0.5) * 0.36, vals, 0.36, color=c, label=blk)
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.004, f"{r.get_height():.3f}", ha="center",
                    va="bottom", fontsize=8.3)
    ax.bar(len(mods), fv, 0.5, color=RED, label="Fused (final)")
    ax.text(len(mods), fv + 0.004, f"{fv:.3f}", ha="center", va="bottom", fontsize=8.3)
    ax.set_xticks(list(x) + [len(mods)])
    ax.set_xticklabels(list(mods) + ["Two blocks\nfused"], fontsize=9.5)
    ax.set_ylim(0.6, 0.75)
    ax.set_ylabel("Site-balanced AUC")
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    save(fig, "fig_models")
    df.to_csv(os.path.join(OUT, "models_vs_ensemble.csv"), index=False)
    return df, fv, z, base


def hparams():
    order = {"logreg": ("Logistic reg.: C", "C", [0.01, 0.1, 1.0]),
             "svm": ("SVM: C", "C", [0.1, 1.0, 10.0]),
             "lgbm": ("LightGBM: leaves", "num_leaves", [4, 8]),
             "ridge_all": ("Ridge: C", "C", [0.0001, 0.0003, 0.001, 0.003, 0.01])}
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.4))
    for ax, (m, (ttl, key, vals)) in zip(axs, order.items()):
        for k, (blk, c) in enumerate((("shells", TEAL), ("loc_clin", ORANGE))):
            met = json.load(open(os.path.join(ROOT, "results/phase13/pooled", f"{blk}__combat", "metrics.json")))
            ch = [f["chosen"][m][key] for f in met["folds"]]
            cnt = [sum(np.isclose(v, c_) for c_ in ch) for v in vals]
            ax.bar(np.arange(len(vals)) + (k - 0.5) * 0.38, cnt, 0.38, color=c,
                   label={"shells": "Shells", "loc_clin": "Position + clinical"}[blk])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"{v:g}" for v in vals], fontsize=8, rotation=0 if len(vals) < 4 else 35)
        ax.set_title(ttl, fontsize=10)
        ax.set_ylim(0, 25)
    axs[0].set_ylabel("Times chosen (of 25)")
    axs[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_hparams")
    # top-k for the logistic regression
    out = {}
    for blk in ("shells", "loc_clin"):
        met = json.load(open(os.path.join(ROOT, "results/phase13/pooled", f"{blk}__combat", "metrics.json")))
        out[blk] = {m: pd.Series([str({kk: vv for kk, vv in f["chosen"][m].items() if kk != "inner"})
                                  for f in met["folds"]]).value_counts().head(2).to_dict() for m in order}
    return out


def loso():
    fu = pd.read_csv(os.path.join(ROOT, "results/phase13/fusion/fusion_all_combat.csv"))
    r = fu[fu.combo == "shells+loc_clin"].iloc[0]
    vals = [r[f"loso_{s}"] for s in SITES]
    within = [r[s] for s in SITES]
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    x = np.arange(5)
    b1 = ax.bar(x[:4] - 0.19, within, 0.36, color="#9FD3E0", label="Pooled CV (hospital seen in training)")
    b2 = ax.bar(x[:4] + 0.19, vals, 0.36, color=[SCOL[s] for s in SITES], label="Unseen hospital (LOSO)")
    b3 = ax.bar(4 + 0.19, np.mean(vals), 0.36, color=DARK)
    ax.bar(4 - 0.19, r["sb_auc"], 0.36, color="#9FD3E0")
    ax.text(4 - 0.19, r["sb_auc"] + 0.008, f"{r['sb_auc']:.3f}", ha="center", va="bottom", fontsize=8.5)
    for bars in (b1, b2, b3):
        for rr in bars:
            ax.text(rr.get_x() + rr.get_width() / 2, rr.get_height() + 0.008, f"{rr.get_height():.3f}", ha="center",
                    va="bottom", fontsize=8.5)
    ax.axhline(0.5, color=GREY, ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(["LUNG1", "Radio-\ngenomics", "Lung-PET-\nCT-Dx", "NLST", "Mean /\npooled"], fontsize=9.5)
    ax.set_ylim(0.4, 0.95)
    ax.set_ylabel("ROC-AUC")
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")
    save(fig, "fig_loso")
    return dict(zip(SITES, vals)), within, r["sb_auc"], np.mean(vals)


def ablation():
    items = [("CT-FM (foundation model)", 0.662, RED), ("FMCIB (deep CNN features)", 0.675, RED),
             ("Inner tumour ring", 0.698, TEAL), ("Shells (3 rings)", 0.700, TEAL),
             ("Tumour texture + shape (GTV)", 0.702, TEAL), ("Position + clinical", 0.709, TEAL),
             ("All 7 blocks fused", 0.720, BLUE), ("Shells + position (FINAL)", 0.722, DARK)]
    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    y = np.arange(len(items))
    ax.barh(y, [i[1] for i in items], color=[i[2] for i in items], height=0.62)
    for k, it in enumerate(items):
        ax.text(it[1] + 0.002, k, f"{it[1]:.3f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([i[0] for i in items], fontsize=9.5)
    ax.set_xlim(0.6, 0.74)
    ax.set_xlabel("Site-balanced AUC (5 x 5 CV, 1041 patients)")
    save(fig, "fig_ablation")


def journey():
    st = ["Attention-MIL\n(LUNG1)", "CNN +\nattention", "2 cohorts\n(pooled AUC)", "Fair metric\n+ ComBat", "3 hospitals",
          "4 hospitals", "+ MedSAM2\n(final)", "Lung3\nlocked test"]
    v = [0.604, 0.615, 0.675, 0.681, 0.718, 0.714, 0.722, 0.686]
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    ax.plot(range(len(v)), v, "-o", color=TEAL, lw=2, ms=6)
    for i, val in enumerate(v):
        ax.text(i, val + 0.007, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.axvline(2.5, color=RED, lw=1.4, ls="--")
    ax.text(2.45, 0.735, "progress\npresentation", color=RED, ha="right", va="top", fontsize=8.5)
    ax.set_xticks(range(len(v)))
    ax.set_xticklabels(st, fontsize=8.5)
    ax.set_ylim(0.57, 0.745)
    ax.set_ylabel("ROC-AUC")
    save(fig, "fig_journey")


def scores_and_confusion(z, base):
    avg = z.groupby(level=1).mean()
    b0 = base.loc[base.index.get_level_values(0)[0]].loc[avg.index]
    df = pd.DataFrame({"s": avg.values, "y": b0.y.values, "site": b0.site.values})
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    pos = []
    for i, s in enumerate(SITES):
        for k, (yy, c) in enumerate(((1, GREEN), (0, ORANGE))):
            v = df[(df.site == s) & (df.y == yy)].s.values
            p = i * 1.0 + (k - 0.5) * 0.36
            bp = ax.boxplot(v, positions=[p], widths=0.3, patch_artist=True, showfliers=False,
                            medianprops=dict(color="black"))
            bp["boxes"][0].set(facecolor=c, alpha=0.85)
            jit = np.random.default_rng(i * 2 + k).uniform(-0.09, 0.09, len(v))
            ax.scatter(p + jit, v, s=3, color="black", alpha=0.35, zorder=3)
            pos.append(p)
    ax.axhline(0, color=RED, ls="--", lw=1.2)
    ax.text(0.5, 0.05, "threshold s = 0", color=RED, fontsize=9, ha="center", va="bottom")
    ax.set_xticks(range(4))
    ax.set_xticklabels([SNAME[s] for s in SITES])
    ax.set_ylabel("Fused out-of-fold score s")
    ax.scatter([], [], marker="s", color=GREEN, label="ADC")
    ax.scatter([], [], marker="s", color=ORANGE, label="SCC")
    ax.legend(frameon=False, fontsize=9, loc="upper left", ncol=2)
    save(fig, "fig_scores")

    l3 = pd.read_csv(os.path.join(ROOT, "results/phase14b/lung3_scored.csv")).dropna(subset=["y"])
    tc = pd.read_csv(os.path.join(ROOT, "results/phase14/tcga_scored.csv")).dropna(subset=["y"])
    sets = [("Cross-validation\n(1041 patients)", df.y.values, df.s.values),
            ("Lung3 locked test\n(79 patients)", l3.y.values, l3.primary.values),
            ("TCGA locked test\n(84 patients)", tc.y.values, tc.primary.values)]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.45))
    stats = {}
    for ax, (ttl, y, s) in zip(axs, sets):
        p = (s > 0).astype(int)
        cm = np.array([[((y == 1) & (p == 1)).sum(), ((y == 1) & (p == 0)).sum()],
                       [((y == 0) & (p == 1)).sum(), ((y == 0) & (p == 0)).sum()]])
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for a in range(2):
            for b in range(2):
                frac = cm[a, b] / cm[a].sum()
                ax.text(b, a, f"{cm[a, b]}\n({frac:.0%})", ha="center", va="center", fontsize=9.5,
                        color="white" if frac > 0.55 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ADC", "SCC"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["ADC", "SCC"])
        ax.set_xlabel("Predicted", fontsize=9.5)
        ax.set_title(ttl, fontsize=10)
        for sp in ax.spines.values():
            sp.set_visible(False)
        stats[ttl.split("\n")[0]] = dict(sens=cm[0, 0] / cm[0].sum(), spec=cm[1, 1] / cm[1].sum(), cm=cm.tolist())
    axs[0].set_ylabel("True", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig_confusion")
    return stats


FIGS = {"pipeline": pipeline, "architecture": architecture, "cv": cv_scheme, "protocol": protocol, "gantt": gantt,
        "shortcut": shortcut, "dataset": dataset, "dice": dice, "siteleak": siteleak, "combat": combat_demo,
        "hparams": hparams, "loso": loso, "ablation": ablation, "journey": journey}

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS) + ["models"]
    for n in names:
        if n == "models":
            df, fv, z, base = models_vs_ensemble()
            print(df.round(3).to_string(), "\nfused", round(fv, 3))
            print(scores_and_confusion(z, base))
        else:
            r = FIGS[n]()
            if r is not None:
                print(n, r)
