"""Phase 11 - shortcut demonstration (3 sites), headline tables and figures.

PRIMARY model = pre-specified before any Phase-11 data existed: the Phase-10
winner (outer peritumoral tissue + tumour position + age/sex), i.e. block
fusion "shells+loc_clin", ComBat.

Usage: python src/phase11_figures.py
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_cv import Prep, sb_auc  # noqa: E402
from phase10_data import load_cfg  # noqa: E402
from phase10_figures import AQUA, BLUE, GRAY, INK, INK2, ORANGE, hbars, title  # noqa: E402
from phase11_cv import RES, SITES, block_matrix, cohort, zload  # noqa: E402

warnings.filterwarnings("ignore")
PRIMARY = "shells+loc_clin"
SITE_NAME = {"LUNG1": "LUNG1 (Maastricht)", "RADIOGENOMICS": "Radiogenomics (Stanford)",
             "LPCD": "Lung-PET-CT-Dx (Harbin)"}
SITE_COL = {"LUNG1": BLUE, "RADIOGENOMICS": ORANGE, "LPCD": AQUA}


def shortcut_demo(cfg, coh):
    y, site = coh.y.values, coh.site.values
    X, _ = block_matrix(coh, "shells")
    strata = np.array([f"{a}|{b}" for a, b in zip(site, y)])
    out = {"site_only": [], "naive": []}
    for seed in cfg["experiment"]["seeds"]:
        s1, s2 = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata):
            prev = {g: y[tr][site[tr] == g].mean() for g in np.unique(site)}
            s1[te] = [prev[g] for g in site[te]]
            pr = Prep("none", cfg).fit(X[tr], y[tr], site[tr])
            sel = pr.select(25)
            m = LogisticRegression(C=0.1, max_iter=5000).fit(pr.Xtr_[:, sel], y[tr])
            s2[te] = m.decision_function(pr.transform(X[te], site[te])[:, sel])
        for k, s in (("site_only", s1), ("naive", s2)):
            out[k].append(dict(raw=float(roc_auc_score(y, s)), sb=sb_auc(y, s, site)))
    return {k: {m: float(np.mean([r[m] for r in v])) for m in ("raw", "sb")} for k, v in out.items()}


def main():
    cfg = load_cfg()
    coh = cohort()
    fig_dir = os.path.join(RES, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    fus = pd.read_csv(os.path.join(RES, "fusion", "fusion_all_combat.csv")).set_index("combo")
    prim = fus.loc[PRIMARY]
    sd = shortcut_demo(cfg, coh)
    print("shortcut", sd)

    # ---- fig 1: shortcut on 3 sites
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    hbars(ax, ["Hospital only (no image)\nstandard pooled AUC", "Naive radiomics model\nstandard pooled AUC",
               "Naive radiomics model\nhonest site-balanced AUC", "Phase-11 model\nhonest site-balanced AUC"],
          [sd["site_only"]["raw"], sd["naive"]["raw"], sd["naive"]["sb"], prim.sb_auc],
          [GRAY, ORANGE, ORANGE, BLUE], xlim=(0.4, 0.85))
    ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xlabel("ROC-AUC")
    title(ax, "Three hospitals: the shortcut is still there, and still removed",
          "644 patients (LUNG1, Radiogenomics, Lung-PET-CT-Dx).")
    fig.savefig(os.path.join(fig_dir, "fig1_shortcut_3sites.png"))
    plt.close(fig)

    # ---- fig 2: blocks + fusions, pooled and unseen-hospital
    order = ["gtv", "fmcib", "mil", "fmcib_bag", "ring_in", "shells", "loc_clin",
             "ring_in+shells+loc_clin+gtv+fmcib+fmcib_bag+mil", PRIMARY]
    lab = ["Tumour radiomics", "FMCIB (1 crop)", "Attention-MIL (16 crops)", "FMCIB bag, mean",
           "Inner rim radiomics", "Peritumoral shells", "Position + age/sex",
           "Fusion of all 7", "PRIMARY: shells + position"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    hbars(axes[0], lab, fus.loc[order, "sb_auc"].values, [GRAY] * 7 + [BLUE, BLUE],
          fus.loc[order, "ci_lo"].values, fus.loc[order, "ci_hi"].values, xlim=(0.45, 0.82))
    axes[0].axvline(0.5, color=INK2, lw=1, ls=":")
    axes[0].set_xlabel("Pooled site-balanced ROC-AUC (95% CI)")
    title(axes[0], "Pooled, 5x5 CV")
    y = np.arange(len(order))[::-1]
    for s, off in zip(SITES, (-0.22, 0, 0.22)):
        axes[1].scatter(fus.loc[order, "loso_" + s].values, y + off, color=SITE_COL[s], s=28,
                        label=SITE_NAME[s] + " unseen", zorder=3)
    axes[1].scatter(fus.loc[order, "loso_mean"].values, y, marker="|", s=260, color=INK, label="mean", zorder=4)
    axes[1].set_xlim(0.45, 0.85)
    axes[1].axvline(0.5, color=INK2, lw=1, ls=":")
    axes[1].set_xlabel("ROC-AUC on the held-out hospital")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    axes[1].grid(axis="y", visible=False)
    title(axes[1], "Each hospital held out in turn")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig2_blocks_fusion.png"))
    plt.close(fig)

    # ---- fig 3: ROC per site, primary model (seed-averaged pooled OOF)
    zs = [zload(os.path.join(RES, "pooled", f"{b}__combat", "oof.csv"), "ensemble") for b in PRIMARY.split("+")]
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
        yh = (avg.values[m] > 0).astype(int)
        yy = base.y.values[m]
        se, sp = float((yh[yy == 1] == 1).mean()), float((yh[yy == 0] == 0).mean())
        thr_rows[s] = dict(sensitivity=se, specificity=sp, balanced_accuracy=(se + sp) / 2)
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls=":")
    ax.set_aspect("equal")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity (ADC)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    title(ax, "Primary model, within each hospital")
    fig.savefig(os.path.join(fig_dir, "fig3_roc_per_site.png"))
    plt.close(fig)

    # ---- fig 4: Phase 10 vs Phase 11
    p10 = json.load(open(os.path.join(RES.replace("phase11", "phase10"), "phase10_summary.json")))["final_model"]
    t10 = json.load(open(os.path.join(RES.replace("phase11", "phase10"), "phase10_summary.json")))["transport_fused_auc"]
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    labels = ["Phase 8B (2 hospitals)\nunseen-hospital mean", "Phase 10 (2 hospitals)\nunseen-hospital mean",
              "Phase 11 (3 hospitals)\nunseen-hospital mean",
              "Phase 10 (2 hospitals)\npooled site-balanced", "Phase 11 (3 hospitals)\npooled site-balanced"]
    vals = [0.5692, np.mean(list(t10["combat"].values())), prim.loso_mean, p10["sb_auc"], prim.sb_auc]
    hbars(ax, labels, vals, [GRAY, ORANGE, BLUE, ORANGE, BLUE], xlim=(0.45, 0.78))
    ax.axvline(0.7, color=INK2, lw=1, ls="--")
    ax.set_xlabel("ROC-AUC")
    title(ax, "More hospitals, more patients: the honest score crosses 0.70")
    fig.savefig(os.path.join(fig_dir, "fig4_phase_progress.png"))
    plt.close(fig)

    summ = dict(primary=PRIMARY, pooled=dict(prim[["sb_auc", "ci_lo", "ci_hi"] + list(SITES)]),
                loso={s: float(prim["loso_" + s]) for s in SITES}, loso_mean=float(prim.loso_mean),
                threshold_metrics=thr_rows, shortcut_demo=sd,
                all_fusions=dict(n=int(len(fus)), pooled_median=float(fus.sb_auc.median()),
                                 pooled_range=[float(fus.sb_auc.min()), float(fus.sb_auc.max())],
                                 loso_median=float(fus.loso_mean.median())),
                n_patients=int(len(coh)),
                counts={f"{s}_{'ADC' if yy else 'SCC'}": int(n)
                        for (s, yy), n in coh.groupby(["site", "y"]).size().items()})
    json.dump(summ, open(os.path.join(RES, "phase11_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
