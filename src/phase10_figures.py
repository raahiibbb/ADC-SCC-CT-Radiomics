"""Phase 10 - summary tables and presentation figures (reads results only).

Usage: python src/phase10_figures.py
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
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_data import load_cfg, p  # noqa: E402

BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FINAL_HARM, FINAL_COMBO = "combat", "rim+peri+loc_clin"
BLOCK_LABEL = {"gtv": "Tumour radiomics", "rim": "Rim radiomics", "peri": "Peritumoral shells",
               "loc_clin": "Position + age/sex", "fmcib": "FMCIB deep features"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight"})


def title(ax, t, sub=None):
    ax.set_title(t, loc="left", fontsize=12, color=INK, fontweight="bold", pad=18 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=INK2)


def hbars(ax, labels, vals, colors, lo=None, hi=None, xlim=(0.4, 0.85)):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, np.array(vals) - xlim[0], left=xlim[0], height=0.55, color=colors,
            edgecolor="white", linewidth=2)
    if lo is not None:
        ax.errorbar(vals, y, xerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none",
                    ecolor=INK2, elinewidth=1, capsize=3)
    for yi, v in zip(y, vals):
        ax.text((hi[list(vals).index(v)] if hi is not None else v) + 0.006, yi, f"{v:.3f}",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK)
    ax.set_xlim(*xlim)
    ax.grid(axis="y", visible=False)


def main():
    cfg = load_cfg()
    R = p(cfg, "results_dir")
    fig_dir = os.path.join(R, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    summary = {}

    # ---------------- 1. shortcut demonstration
    sd = json.load(open(os.path.join(R, "shortcut_demo", "shortcut_demo.json")))["summary"]
    fus = pd.read_csv(os.path.join(R, "fusion", "fusion_all_combinations.csv"))
    final = fus[(fus.harm == FINAL_HARM) & (fus.combo == FINAL_COMBO)].iloc[0]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    labels = ["Hospital only (no image)\nstandard pooled AUC",
              "Naive radiomics model\nstandard pooled AUC",
              "Naive radiomics model\nhonest site-balanced AUC",
              "Phase-10 model\nhonest site-balanced AUC"]
    vals = [sd["site_only"]["raw_pooled_auc"], sd["naive"]["raw_pooled_auc"],
            sd["naive"]["site_balanced_auc"], final.sb_auc]
    hbars(ax, labels, vals, [GRAY, ORANGE, ORANGE, BLUE], xlim=(0.4, 0.85))
    ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xlabel("ROC-AUC")
    title(ax, "The high pooled score is the hospital shortcut",
          "Knowing only the hospital already gives 0.76. Honest scoring exposes it.")
    fig.savefig(os.path.join(fig_dir, "fig1_shortcut.png"))
    plt.close(fig)
    summary["shortcut_demo"] = sd

    # ---------------- 2. site leakage before / after harmonisation
    rows = []
    for b in BLOCK_LABEL:
        for h in ("none", "combat", "balanced_combat"):
            m = json.load(open(os.path.join(R, "cv", f"{b}__{h}", "metrics.json")))
            rows.append(dict(block=b, harm=h, site_auc=m["site_classifier_auc_mean"],
                             sb_auc=m["models"]["ensemble"]["mean"]["sb_auc"],
                             lung1=m["models"]["ensemble"]["mean"]["auc_LUNG1"],
                             radiogenomics=m["models"]["ensemble"]["mean"]["auc_RADIOGENOMICS"]))
    blk = pd.DataFrame(rows)
    blk.to_csv(os.path.join(R, "block_summary.csv"), index=False)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(BLOCK_LABEL))
    for i, (h, c, lab) in enumerate([("none", ORANGE, "No harmonisation"),
                                     ("combat", BLUE, "ComBat"),
                                     ("balanced_combat", AQUA, "Balanced ComBat")]):
        v = blk[blk.harm == h].set_index("block").loc[list(BLOCK_LABEL), "site_auc"].values
        ax.bar(x + (i - 1) * 0.26, v, width=0.24, color=c, label=lab, edgecolor="white", linewidth=2)
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([BLOCK_LABEL[b].replace(" ", "\n", 1) for b in BLOCK_LABEL], fontsize=8.5, color=INK)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Hospital-classifier AUC")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.02))
    ax.grid(axis="x", visible=False)
    title(ax, "Harmonisation removes the hospital fingerprint",
          "How well a classifier can tell which hospital a patient came from (0.5 = impossible).")
    fig.savefig(os.path.join(fig_dir, "fig2_site_leakage.png"))
    plt.close(fig)

    # ---------------- 3. single blocks vs fusion (honest AUC + 95% CI)
    f = fus[fus.harm == FINAL_HARM].set_index("combo")
    order = ["gtv", "fmcib", "loc_clin", "peri", "rim", "rim+loc_clin", FINAL_COMBO]
    lab = [BLOCK_LABEL.get(o, o.replace("+", " + ").replace("loc_clin", "position")
                           .replace("peri", "shells")) for o in order]
    lab[-1] = "FINAL: rim + shells + position"
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    hbars(ax, lab, f.loc[order, "sb_auc"].values, [GRAY] * 5 + [BLUE, BLUE],
          f.loc[order, "ci_lo"].values, f.loc[order, "ci_hi"].values, xlim=(0.45, 0.80))
    ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.set_xlabel("Site-balanced ROC-AUC (5x5 CV, 95% bootstrap CI)")
    title(ax, "Complementary information sources add up",
          "Equal-weight late fusion of independently trained models (ComBat-harmonised).")
    fig.savefig(os.path.join(fig_dir, "fig3_fusion.png"))
    plt.close(fig)

    # ---------------- 4. ROC per site, final model (seed-averaged OOF)
    sc = pd.read_csv(os.path.join(R, "fusion", "fusion_seed_avg_scores.csv"))
    s = sc[(sc.harm == FINAL_HARM) & (sc.combo == FINAL_COMBO)]
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for site, c, nm in (("LUNG1", BLUE, "LUNG1 (Maastricht)"),
                        ("RADIOGENOMICS", ORANGE, "Radiogenomics (Stanford)")):
        g = s[s.site == site]
        fpr, tpr, _ = roc_curve(g.y, g.score)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{nm}  AUC {roc_auc_score(g.y, g.score):.3f}")
    ax.plot([0, 1], [0, 1], color=GRAY, lw=1, ls=":")
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity (ADC)")
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    title(ax, "Final model, within each hospital")
    fig.savefig(os.path.join(fig_dir, "fig4_roc_per_site.png"))
    plt.close(fig)

    # ---------------- 5. transport, fused
    tr_rows = []
    parts = {b: pd.read_csv(os.path.join(R, "transport", f"{b}_scores.csv"))
             for b in ("rim", "peri", "loc_clin")}
    for b, d in parts.items():
        d["z"] = d.groupby(["direction", "seed", "mapping"])["score"].transform(
            lambda v: (v - v.mean()) / v.std())
    key = ["direction", "seed", "mapping", "idx"]
    fused = pd.concat([d.set_index(key)["z"] for d in parts.values()], axis=1).mean(1).rename("z")
    fused = fused.reset_index().merge(parts["rim"][key + ["y"]], on=key)
    for name, d in list(parts.items()) + [("fused", fused)]:
        col = "z" if name == "fused" else "score"
        for (dr, sd_, mp), g in d.groupby(["direction", "seed", "mapping"]):
            tr_rows.append(dict(block=name, direction=dr, seed=sd_, mapping=mp,
                                auc=roc_auc_score(g.y, g[col])))
    tr = pd.DataFrame(tr_rows).groupby(["block", "direction", "mapping"])["auc"].mean().unstack()
    tr.to_csv(os.path.join(R, "transport_auc_summary.csv"))
    print("\nTRANSPORT AUC\n", tr.round(3).to_string())
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    ft = tr.loc["fused"]
    dirs = ["LUNG1->RADIOGENOMICS", "RADIOGENOMICS->LUNG1"]
    x = np.arange(3)
    for i, (mp, c, lb) in enumerate([("none", ORANGE, "No harmonisation"),
                                     ("combat", BLUE, "ComBat (label-free)"),
                                     ("prior_known", AQUA, "Prevalence-informed ComBat")]):
        v = [ft.loc[d_, mp] for d_ in dirs] + [ft.loc[dirs, mp].mean()]
        ax.bar(x + (i - 1) * 0.26, v, width=0.24, color=c, label=lb, edgecolor="white", linewidth=2)
        for xi, vi in zip(x + (i - 1) * 0.26, v):
            ax.text(xi, vi + 0.005, f"{vi:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["Train LUNG1\ntest Radiogenomics", "Train Radiogenomics\ntest LUNG1", "Mean"],
                       color=INK)
    ax.set_ylim(0.45, 0.78)
    ax.axhline(0.5, color=INK2, lw=1, ls=":")
    ax.set_ylabel("ROC-AUC on the unseen hospital")
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.02))
    ax.grid(axis="x", visible=False)
    title(ax, "Train on one hospital, test on the other",
          "Final fused model. No destination patient label is used (green uses only the site's ADC share).")
    fig.savefig(os.path.join(fig_dir, "fig5_transport.png"))
    plt.close(fig)

    # ---------------- 6. the clinical prior: tumour position by histology
    loc = pd.read_csv(os.path.join(p(cfg, "extra_features_dir"), "loc.csv"))
    coh = pd.read_csv(p(cfg, "cohort_csv"))
    m = coh.merge(loc, on="PatientKey")
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    for ax, feat, nm in ((axes[0], "loc__rel_lateral", "Lateral position in the lung\n0 = medial, 1 = chest wall"),
                         (axes[1], "loc__contact_medial_frac", "Share of tumour contact\nfacing the mediastinum")):
        data, pos, cols = [], [], []
        for i, site in enumerate(("LUNG1", "RADIOGENOMICS")):
            for j, (lbl, c) in enumerate(((0, ORANGE), (1, BLUE))):
                data.append(m[(m.Cohort == site) & (m.label == lbl)][feat].values)
                pos.append(i * 2.6 + j)
                cols.append(c)
        bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, showfliers=False,
                        medianprops=dict(color="white", lw=2))
        for patch, c in zip(bp["boxes"], cols):
            patch.set(facecolor=c, edgecolor=c)
        for w in bp["whiskers"] + bp["caps"]:
            w.set(color=INK2)
        ax.set_xticks([0.5, 3.1])
        ax.set_xticklabels(["LUNG1", "Radiogenomics"], color=INK)
        ax.set_title(nm, fontsize=9.5, color=INK, loc="left")
        ax.grid(axis="x", visible=False)
    axes[1].legend([bp["boxes"][0], bp["boxes"][1]], ["SCC", "ADC"], frameon=False, fontsize=8.5)
    fig.suptitle("Where the tumour sits: ADC more peripheral, SCC more central (both hospitals)",
                 x=0.02, ha="left", fontsize=11.5, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fig6_tumour_position.png"))
    plt.close(fig)

    # ---------------- threshold metrics of the final model (fold-free threshold
    # = 0 on the seed-averaged z-score, i.e. the training-balanced midpoint)
    yhat = (s.score > 0).astype(int)
    thr = {}
    for site, g in [("pooled", s)] + list(s.groupby("site")):
        yh = (g.score > 0).astype(int)
        sens = float((yh[g.y == 1] == 1).mean())
        spec = float((yh[g.y == 0] == 0).mean())
        thr[site] = dict(sensitivity=sens, specificity=spec, balanced_accuracy=(sens + spec) / 2)
    summary["final_model"] = dict(harm=FINAL_HARM, combo=FINAL_COMBO, **final.drop(["harm", "combo"]).to_dict(),
                                  threshold_metrics=thr)
    summary["transport_fused_auc"] = tr.loc["fused"].to_dict()
    json.dump(summary, open(os.path.join(R, "phase10_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(summary["final_model"], indent=1, default=float))


if __name__ == "__main__":
    main()
