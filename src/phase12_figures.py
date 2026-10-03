"""Phase 12 - four-site shortcut demo, headline tables and figures.

PRIMARY = "shells+loc_clin" (ComBat), frozen from Phase 10/11 before NLST
was seen.  Usage: python src/phase12_figures.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase12_cv as P12  # noqa: E402  (patches phase11_cv globals)
import phase11_cv as C  # noqa: E402
from phase10_data import load_cfg  # noqa: E402
from phase10_figures import AQUA, BLUE, GRAY, INK, INK2, ORANGE, hbars, title  # noqa: E402
from phase11_figures import shortcut_demo  # noqa: E402

warnings.filterwarnings("ignore")
PRIMARY = P12.PRIMARY
PURPLE = "#8e5bd0"
SITE_NAME = {"LUNG1": "LUNG1 (Maastricht)", "RADIOGENOMICS": "Radiogenomics (Stanford)",
             "LPCD": "Lung-PET-CT-Dx (Harbin)", "NLST": "NLST (US screening, 33 centres)"}
SITE_COL = {"LUNG1": BLUE, "RADIOGENOMICS": ORANGE, "LPCD": AQUA, "NLST": PURPLE}
SITES = C.SITES


def main():
    cfg = load_cfg()
    coh = P12.cohort()
    RES = C.RES
    fig_dir = os.path.join(RES, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fus = pd.read_csv(os.path.join(RES, "fusion", "fusion_all_combat.csv")).set_index("combo")
    prim = fus.loc[PRIMARY]
    sd = shortcut_demo(cfg, coh)
    print("shortcut", sd)

    # ---- fig 1: shortcut on 4 sites
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    hbars(ax, ["Hospital only (no image)\nstandard pooled AUC", "Naive radiomics model\nstandard pooled AUC",
               "Naive radiomics model\nhonest site-balanced AUC", "Primary model\nhonest site-balanced AUC"],
          [sd["site_only"]["raw"], sd["naive"]["raw"], sd["naive"]["sb"], prim.sb_auc],
          [GRAY, ORANGE, ORANGE, BLUE], xlim=(0.4, 0.85))
    ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xlabel("ROC-AUC")
    title(ax, "Four hospitals: the shortcut is still there, and still removed",
          f"{len(coh)} patients (LUNG1, Radiogenomics, Lung-PET-CT-Dx, NLST).")
    fig.savefig(os.path.join(fig_dir, "fig1_shortcut_4sites.png"))
    plt.close(fig)

    # ---- fig 2: blocks + key fusions, pooled and unseen hospital
    order = ["fmcib", "ring_in", "shells", "gtv", "loc_clin", "gtv+ring_in+shells+loc_clin+fmcib",
             "gtv+shells+loc_clin", PRIMARY]
    lab = ["FMCIB foundation model", "Inner rim radiomics", "Peritumoral shells", "Tumour radiomics",
           "Position + age/sex", "Fusion of all 5", "Best fusion (post hoc)", "PRIMARY: shells + position"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    hbars(axes[0], lab, fus.loc[order, "sb_auc"].values, [GRAY] * 5 + [BLUE] * 3,
          fus.loc[order, "ci_lo"].values, fus.loc[order, "ci_hi"].values, xlim=(0.45, 0.82))
    axes[0].axvline(0.5, color=INK2, lw=1, ls=":")
    axes[0].set_xlabel("Pooled site-balanced ROC-AUC (95% CI)")
    title(axes[0], "Pooled, 5x5 CV")
    y = np.arange(len(order))[::-1]
    for s, off in zip(SITES, (-0.27, -0.09, 0.09, 0.27)):
        axes[1].scatter(fus.loc[order, "loso_" + s].values, y + off, color=SITE_COL[s], s=26,
                        label=SITE_NAME[s] + " unseen", zorder=3)
    axes[1].scatter(fus.loc[order, "loso_mean"].values, y, marker="|", s=260, color=INK, label="mean", zorder=4)
    axes[1].set_xlim(0.45, 0.9)
    axes[1].axvline(0.5, color=INK2, lw=1, ls=":")
    axes[1].set_xlabel("ROC-AUC on the held-out hospital")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    axes[1].grid(axis="y", visible=False)
    title(axes[1], "Each hospital held out in turn")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_blocks_fusion.png"))
    plt.close(fig)

    # ---- fig 3: ROC per site, primary (seed-averaged pooled OOF)
    zs = [C.zload(os.path.join(RES, "pooled", f"{b}__combat", "oof.csv"), "ensemble") for b in PRIMARY.split("+")]
    z = sum(d["z"] for d in zs) / len(zs)
    avg = z.groupby(level=1).mean()
    base = zs[0].loc[zs[0].index.get_level_values(0)[0]].loc[avg.index]
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    thr_rows = {}
    for s in SITES:
        m = base.site.values == s
        fpr, tpr, _ = roc_curve(base.y.values[m], avg.values[m])
        ax.plot(fpr, tpr, color=SITE_COL[s], lw=2,
                label=f"{SITE_NAME[s]}  {roc_auc_score(base.y.values[m], avg.values[m]):.3f}")
        yh, yy = (avg.values[m] > 0).astype(int), base.y.values[m]
        se, sp = float((yh[yy == 1] == 1).mean()), float((yh[yy == 0] == 0).mean())
        thr_rows[s] = dict(sensitivity=se, specificity=sp, balanced_accuracy=(se + sp) / 2)
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls=":")
    ax.set_aspect("equal")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity (ADC)")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    title(ax, "Primary model, within each hospital")
    fig.savefig(os.path.join(fig_dir, "fig3_roc_per_site.png"))
    plt.close(fig)

    # ---- fig 4: progress across phases
    s11 = json.load(open(os.path.join(C.ROOT, "results", "phase11", "phase11_summary.json")))
    s10 = json.load(open(os.path.join(C.ROOT, "results", "phase10", "phase10_summary.json")))
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    labels = ["Phase 8B (2 hospitals)\nunseen-hospital mean",
              "Phase 10 (2 hospitals)\nunseen-hospital mean",
              "Phase 11 (3 hospitals)\nunseen-hospital mean",
              "Phase 12 (4 hospitals)\nunseen-hospital mean",
              "Phase 11 (3 hospitals)\npooled site-balanced",
              "Phase 12 (4 hospitals)\npooled site-balanced"]
    vals = [0.5692, np.mean(list(s10["transport_fused_auc"]["combat"].values())), s11["loso_mean"],
            prim.loso_mean, s11["pooled"]["sb_auc"], prim.sb_auc]
    hbars(ax, labels, vals, [GRAY, GRAY, ORANGE, BLUE, ORANGE, BLUE], xlim=(0.45, 0.78))
    ax.axvline(0.7, color=INK2, lw=1, ls="--")
    ax.set_xlabel("ROC-AUC")
    title(ax, "Adding NLST: transport to an unseen hospital improves most")
    fig.savefig(os.path.join(fig_dir, "fig4_phase_progress.png"))
    plt.close(fig)

    sub = os.path.join(RES, "subset_single_lesion", "fusion", "fusion_all_combat.csv")
    sub = pd.read_csv(sub).set_index("combo").loc[PRIMARY].to_dict() if os.path.isfile(sub) else None
    summ = dict(primary=PRIMARY, pooled=dict(prim[["sb_auc", "ci_lo", "ci_hi"] + list(SITES)]),
                loso={s: float(prim["loso_" + s]) for s in SITES}, loso_mean=float(prim.loso_mean),
                threshold_metrics=thr_rows, shortcut_demo=sd,
                best_fusion_post_hoc=dict(combo=fus.index[0], **fus.iloc[0][["sb_auc", "ci_lo", "ci_hi", "loso_mean"]]),
                all_fusions=dict(n=int(len(fus)), pooled_median=float(fus.sb_auc.median()),
                                 pooled_range=[float(fus.sb_auc.min()), float(fus.sb_auc.max())],
                                 loso_median=float(fus.loso_mean.median())),
                single_lesion_sensitivity=sub, n_patients=int(len(coh)),
                counts={f"{s}_{'ADC' if yy else 'SCC'}": int(n) for (s, yy), n in coh.groupby(["site", "y"]).size().items()})
    json.dump(summ, open(os.path.join(RES, "phase12_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
