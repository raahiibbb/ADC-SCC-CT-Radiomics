"""Extra numbers for the final report: dataset statistics, parameter counts,
computational cost and the full set of threshold metrics.

Reads only frozen feature / prediction files.  The final (primary) model is
refitted exactly as in phase14_test.predict (all 1041 patients, seed 42) and
checked against the frozen TCGA predictions before anything is counted.
Writes results/report_stats/stats.json and report/figures/fig_stats_*.png.
Usage (.venv-report):  python src/report_stats.py
"""
from __future__ import annotations

import collections
import json
import os
import pickle
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import phase11_cv as C  # noqa: E402
import report_figures as RF  # noqa: E402
from phase10_data import load_cfg  # noqa: E402
from phase10_harmonise import Harmoniser  # noqa: E402

plt = RF.plt
OUT = os.path.join(ROOT, "results", "report_stats")
DEV_FEAT = os.path.join(ROOT, "global_features", "phase13")
SITES = RF.SITES
R = {}


def rd(path):
    return pd.read_csv(os.path.join(ROOT, path))


# ------------------------------------------------------------------ cohorts
def tables():
    dev = rd("cohort/phase12_four_site_cohort.csv")
    dev = dev.merge(rd("global_features/phase13/clin.csv"), on="PatientKey").merge(
        rd("global_features/phase13/loc.csv"), on="PatientKey")
    dev["cohort"] = dev.site
    ext = []
    for name, sc, fd in (("LUNG3", "results/phase14b/lung3_scored.csv", "phase14b"),
                         ("TCGA", "results/phase14/tcga_scored.csv", "phase14")):
        s = rd(sc).dropna(subset=["y"])[["PatientKey", "y"]].rename(columns={"y": "label"})
        s = s.merge(rd(f"global_features/{fd}/clin.csv"), on="PatientKey").merge(
            rd(f"global_features/{fd}/loc.csv"), on="PatientKey", how="left")
        s["cohort"] = s["site"] = name
        ext.append(s)
    d = pd.concat([dev] + ext, ignore_index=True)
    d["label"] = d.label.astype(int)
    d["vol_ml"] = np.exp(d.loc__log_gtv_volume) / 1000.0
    return d


def describe(d):
    rows = []
    for c, g in d.groupby("cohort", sort=False):
        for lab, h in ((1, g[g.label == 1]), (0, g[g.label == 0])):
            q = h.vol_ml.quantile([0.25, 0.5, 0.75]).values
            rows.append(dict(cohort=c, cls="ADC" if lab else "SCC", n=len(h),
                             age_mean=h.clin__age.mean(), age_sd=h.clin__age.std(), age_missing=int(h.clin__age.isna().sum()),
                             male_pct=100 * h.clin__male.mean(), vol_med=q[1], vol_q1=q[0], vol_q3=q[2],
                             left_pct=100 * h.loc__left.mean(), contact_med=h.loc__boundary_contact_frac.median(),
                             relz_med=h.loc__rel_z.median()))
    return pd.DataFrame(rows)


def mw_auc(x, y):
    m = ~np.isnan(x)
    return roc_auc_score(y[m], x[m])


def clinical_tests(dev):
    """ADC vs SCC (pooled Mann-Whitney / chi-square), between hospitals (Kruskal-Wallis /
    chi-square) and the mean within-hospital AUC of the raw variable (sign: >0.5 = higher in ADC)."""
    y = dev.label.values
    out = {}
    for v, kind in (("clin__age", "c"), ("clin__male", "b"), ("vol_ml", "c"), ("loc__left", "b"),
                    ("loc__rel_z", "c"), ("loc__boundary_contact_frac", "c"), ("loc__centroid_depth_mm", "c")):
        x = dev[v].values.astype(float)
        if kind == "c":
            p_cls = stats.mannwhitneyu(x[(y == 1) & ~np.isnan(x)], x[(y == 0) & ~np.isnan(x)]).pvalue
            p_site = stats.kruskal(*[x[(dev.site == s).values & ~np.isnan(x)] for s in SITES]).pvalue
        else:
            p_cls = stats.chi2_contingency(pd.crosstab(dev[v], y)).pvalue
            p_site = stats.chi2_contingency(pd.crosstab(dev[v], dev.site)).pvalue
        within = np.mean([mw_auc(x[(dev.site == s).values], y[(dev.site == s).values]) for s in SITES])
        out[v] = dict(p_adc_vs_scc=float(p_cls), p_between_hospitals=float(p_site), within_hospital_auc=float(within),
                      adc=float(np.nanmean(x[y == 1])), scc=float(np.nanmean(x[y == 0])))
    return out


def feature_tests(dev):
    """279 shell features: hospital effect (Kruskal-Wallis eta^2, BH-FDR) before / after ComBat and
    histology effect within each hospital (Mann-Whitney, BH-FDR; mean |AUC-0.5|)."""
    C.FEAT = DEV_FEAT
    X, cols = C.block_matrix(dev[["PatientKey"]], "shells")
    keep = np.isnan(X).mean(0) <= 0.2
    X, cols = X[:, keep], [c for c, k in zip(cols, keep) if k]
    X = np.where(np.isnan(X), np.nanmedian(X, 0), X)
    X = X[:, X.std(0) > 1e-8]
    y, site = dev.label.values, dev.site.values
    Xc = Harmoniser("combat", True).fit(X, site, y).transform(X, site)
    n, k = len(y), len(SITES)

    def hosp(M):
        H, p = zip(*[stats.kruskal(*[M[site == s, j] for s in SITES]) for j in range(M.shape[1])])
        H, p = np.array(H), np.array(p)
        eta = (H - k + 1) / (n - k)
        sig = multipletests(p, 0.05, "fdr_bh")[0]
        return eta, sig

    eta0, sig0 = hosp(X)
    eta1, sig1 = hosp(Xc)
    def hist(M):
        """within-hospital ADC vs SCC: mean |AUC-0.5| and number of hospitals with a BH-significant difference."""
        rho, nsig = np.zeros(M.shape[1]), np.zeros(M.shape[1], int)
        for s in SITES:
            m = site == s
            a = np.array([roc_auc_score(y[m], M[m, j]) for j in range(M.shape[1])])
            p = np.array([stats.mannwhitneyu(M[m & (y == 1), j], M[m & (y == 0), j]).pvalue for j in range(M.shape[1])])
            rho += np.abs(a - 0.5) / k
            nsig += multipletests(p, 0.05, "fdr_bh")[0]
        return rho, nsig

    rho0, nsig0 = hist(X)
    rho, nsig = hist(Xc)
    Z = (Xc - Xc.mean(0)) / Xc.std(0)
    r = np.abs(np.corrcoef(Z.T))[np.triu_indices(Z.shape[1], 1)]
    res = dict(n_features=int(X.shape[1]), hosp_sig_before=int(sig0.sum()), hosp_sig_after=int(sig1.sum()),
               eta_median_before=float(np.median(eta0)), eta_median_after=float(np.median(eta1)),
               hist_sig_in_ge1=int((nsig >= 1).sum()), hist_sig_in_ge2=int((nsig >= 2).sum()),
               hist_sig_in_all=int((nsig == k).sum()), rho_median=float(np.median(rho)), rho_max=float(rho.max()),
               top_rho_feature=cols[int(np.argmax(rho))], frac_pairs_r_gt_0_9=float((r > 0.9).mean()),
               frac_pairs_r_gt_0_7=float((r > 0.7).mean()))
    # figure: hospital effect before/after ComBat, and histology vs hospital effect
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8))
    eta0, eta1 = np.clip(eta0, 0, None), np.clip(eta1, 0, None)   # negative eta^2 = no effect
    bins = np.linspace(0, max(eta0.max(), eta1.max()) + 1e-3, 30)
    axs[0].hist(eta0, bins, color=RF.RED, alpha=0.75, label="before ComBat")
    axs[0].hist(eta1, bins, color=RF.TEAL, alpha=0.75, label="after ComBat")
    axs[0].set_yscale("log")
    axs[0].set_xlabel(r"Hospital effect size $\eta^2_H$ (Kruskal-Wallis)")
    axs[0].set_ylabel("Number of features")
    axs[0].set_title("(a) Hospital differences", fontsize=10)
    res["hist_sig_before"] = [int((nsig0 >= 1).sum()), int((nsig0 >= 2).sum())]
    groups = ["Hospital\ndiffers", "ADC/SCC differ\nin ≥ 1 hospital", "ADC/SCC differ\nin ≥ 2 hospitals"]
    before = [sig0.sum(), (nsig0 >= 1).sum(), (nsig0 >= 2).sum()]
    after = [sig1.sum(), (nsig >= 1).sum(), (nsig >= 2).sum()]
    x = np.arange(3)
    for off, vals, col, lab in ((-0.19, before, RF.RED, "before ComBat"), (0.19, after, RF.TEAL, "after ComBat")):
        bars = axs[1].bar(x + off, vals, 0.36, color=col, label=lab)
        for bb, v in zip(bars, vals):
            axs[1].text(bb.get_x() + bb.get_width() / 2, v + 4, str(int(v)), ha="center", va="bottom", fontsize=8.5)
    axs[1].axhline(X.shape[1], color=RF.GREY, ls=":", lw=1)
    axs[1].text(2.62, X.shape[1] + 4, f"all {X.shape[1]} features", ha="right", va="bottom", fontsize=8.5,
                color=RF.GREY)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(groups, fontsize=8.5)
    axs[1].set_ylim(0, 315)
    axs[1].set_ylabel("Number of features (q < 0.05)")
    axs[1].set_title("(b) Significant differences", fontsize=10)
    h, lab = axs[1].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=9.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    RF.save(fig, "fig_stats_features")
    return res


def clinical_figure(d):
    order = SITES + ["LUNG3", "TCGA"]
    names = {**RF.SNAME, "LUNG3": "Lung3*", "TCGA": "TCGA*"}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, (v, lab, logy) in zip(axs, (("clin__age", "Age (years)", False), ("vol_ml", "Tumour volume (ml)", True),
                                        ("loc__boundary_contact_frac", "Lung-boundary contact fraction", False))):
        for i, c in enumerate(order):
            for kk, (yy, col) in enumerate(((1, RF.GREEN), (0, RF.ORANGE))):
                x = d[(d.cohort == c) & (d.label == yy)][v].dropna().values
                bp = ax.boxplot(x, positions=[i + (kk - 0.5) * 0.36], widths=0.3, patch_artist=True, showfliers=False,
                                medianprops=dict(color="black", lw=1))
                bp["boxes"][0].set(facecolor=col, alpha=0.85)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([names[c] for c in order], rotation=40, ha="right", fontsize=8.5)
        ax.set_ylabel(lab, fontsize=9.5)
        if logy:
            ax.set_yscale("log")
        ax.axvline(3.5, color=RF.GREY, ls=":", lw=1)
    axs[1].scatter([], [], marker="s", color=RF.GREEN, label="ADC")
    axs[1].scatter([], [], marker="s", color=RF.ORANGE, label="SCC")
    axs[1].legend(frameon=False, fontsize=9, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    RF.save(fig, "fig_stats_clinical")


# --------------------------------------------------------- parameter counts
def count_lgbm(m):
    thr = leaves = 0
    for t in m.booster_.dump_model()["tree_info"]:
        stack = [t["tree_structure"]]
        while stack:
            nd = stack.pop()
            if "split_index" in nd:
                thr += 1
                stack += [nd["left_child"], nd["right_child"]]
            else:
                leaves += 1
    return thr, leaves


def depth_lgbm(m):
    tot = 0.0
    for t in m.booster_.dump_model()["tree_info"]:
        def walk(nd, dpt):
            if "split_index" not in nd:
                return [(dpt, nd.get("leaf_count", 1))]
            return walk(nd["left_child"], dpt + 1) + walk(nd["right_child"], dpt + 1)
        lv = walk(t["tree_structure"], 0)
        tot += sum(a * b for a, b in lv) / max(sum(b for _, b in lv), 1)
    return tot


def params(pr, fitted, H):
    p = int(pr.keep2_.sum())
    non = dict(median=int(pr.keep_.sum()), combat=2 * p + 2 * H * p, zscore=2 * p, score_norm=2 * len(fitted))
    tr, flops, rows = {}, 7 * p, {}
    for k, (m, sel, mu, sd, hp) in fitted.items():
        ksel = len(sel)
        if k in ("logreg", "ridge_all"):
            tr[k] = int(m.coef_.size + m.intercept_.size)
            flops += 2 * ksel + 1
            rows[k] = dict(features=ksel, trainable=tr[k], stored=0)
        elif k == "svm":
            nsv = int(m.support_vectors_.shape[0])
            tr[k] = int(m.dual_coef_.size + m.intercept_.size)
            non["svm_support_vectors"] = int(m.support_vectors_.size)
            flops += nsv * (3 * ksel + 20 + 2)
            rows[k] = dict(features=ksel, trainable=tr[k], stored=int(m.support_vectors_.size), n_sv=nsv)
        else:
            thr, lv = count_lgbm(m)
            tr[k] = thr + lv
            non["lgbm_split_feature_ids"] = thr
            flops += int(round(depth_lgbm(m))) + 200
            rows[k] = dict(features=ksel, trainable=tr[k], thresholds=thr, leaves=lv)
        rows[k]["hp"] = {a: b for a, b in hp.items() if a != "inner"}
    return dict(n_input=p, trainable=tr, trainable_total=int(sum(tr.values())), non_trainable=non,
                non_trainable_total=int(sum(non.values())), flops_per_patient=int(flops), models=rows)


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        pass


def count_checkpoint(path, key=None):
    """Count tensor elements in a PyTorch zip checkpoint without importing torch."""
    z = zipfile.ZipFile(path)
    pk = [n for n in z.namelist() if n.endswith("data.pkl")][0]

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                return lambda st, off, size, *a: ("T", tuple(size))
            if name == "_rebuild_parameter":
                return lambda data, *a: data
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if mod == "builtins":
                return getattr(__import__("builtins"), name)
            return type(name, (_Stub,), {})

        def persistent_load(self, pid):
            return None

    obj = U(z.open(pk)).load()
    if key is not None and isinstance(obj, dict) and key in obj:
        obj = obj[key]
    tot, n = 0, 0

    def walk(o):
        nonlocal tot, n
        if isinstance(o, tuple) and len(o) == 2 and o[0] == "T":
            tot += int(np.prod(o[1])) if len(o[1]) else 1
            n += 1
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
    walk(obj)
    return dict(elements=tot, tensors=n, top_keys=list(obj.keys())[:8] if isinstance(obj, dict) else None)


# ------------------------------------------------------------- final model
def final_model(dev):
    cfg = load_cfg()
    y, site = dev.label.values.astype(int), dev.site.values
    tst = rd("cohort/phase14_tcga_cohort.csv")
    frozen = rd("results/phase14/tcga_predictions.csv").set_index("PatientKey").loc[tst.PatientKey]
    out, timing, pcount, z = {}, {}, {}, {}
    for b in ("shells", "loc_clin"):
        C.FEAT = DEV_FEAT
        Xd, _ = C.block_matrix(dev[["PatientKey"]], b)
        C.FEAT = os.path.join(ROOT, "global_features", "phase14")
        Xt, _ = C.block_matrix(tst[["PatientKey"]], b)
        t0 = time.perf_counter()
        pr, fitted = C.select_and_fit(cfg, Xd, y, site, 42, "combat")
        t_fit = time.perf_counter() - t0
        # one outer CV fold (80 % of patients) for the nested-CV cost estimate
        idx = np.random.RandomState(0).permutation(len(y))[: int(0.8 * len(y))]
        t0 = time.perf_counter()
        C.select_and_fit(cfg, Xd[idx], y[idx], site[idx], 42, "combat")
        t_fold = time.perf_counter() - t0
        Xr = Xt[:, pr.keep_]
        Xr = np.where(np.isnan(Xr), pr.med_, Xr)[:, pr.keep2_]
        pr.h_.fit_new_site(Xr, "TCGA", balanced=False)
        st = np.array(["TCGA"] * len(Xr))

        def infer():
            Z = (pr.h_.transform(Xr, st) - pr.mu_) / pr.sd_
            return C.ens(fitted, Z)
        s = infer()
        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):
            infer()
        t_inf = (time.perf_counter() - t0) / reps / len(Xr)
        t0 = time.perf_counter()
        for _ in range(reps):
            Z1 = (pr.h_.transform(Xr[:1], st[:1]) - pr.mu_) / pr.sd_
            C.ens(fitted, Z1)
        t_one = (time.perf_counter() - t0) / reps
        out[b] = float(np.corrcoef(s, frozen["score_" + b].values)[0, 1])
        z[b] = (s - s.mean()) / s.std()
        timing[b] = dict(fit_all_s=t_fit, fit_one_outer_fold_s=t_fold, infer_batch_ms_per_patient=1e3 * t_inf,
                         infer_single_patient_ms=1e3 * t_one,
                         model_bytes=len(pickle.dumps((pr, fitted))))
        pcount[b] = params(pr, fitted, len(SITES))
        print(b, "refit corr with frozen", round(out[b], 6), "| fit", round(t_fit, 1), "s", flush=True)
    prim = (z["shells"] + z["loc_clin"]) / 2
    out["primary_max_abs_diff"] = float(np.abs(prim - frozen["primary"].values).max())
    return out, timing, pcount


# ----------------------------------------------------------------- metrics
def metric_set(y, s, w=None):
    w = np.ones(len(y)) if w is None else w
    p = (s > 0).astype(int)
    tp, fn = w[(y == 1) & (p == 1)].sum(), w[(y == 1) & (p == 0)].sum()
    fp, tn = w[(y == 0) & (p == 1)].sum(), w[(y == 0) & (p == 0)].sum()
    se, sp = tp / (tp + fn), tn / (tn + fp)
    ppv, npv = tp / (tp + fp), tn / (tn + fn)
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(auc=float(roc_auc_score(y, s, sample_weight=w)), acc=float((tp + tn) / (tp + tn + fp + fn)),
                se=float(se), sp=float(sp), ba=float((se + sp) / 2), ppv=float(ppv), npv=float(npv),
                f1=float(2 * ppv * se / (ppv + se)), mcc=float(mcc))


def all_metrics():
    fused = []
    for blk in ("shells", "loc_clin"):
        d = RF.load_oof(blk).copy()
        d["z"] = d.groupby("seed")["ensemble"].transform(lambda v: (v - v.mean()) / v.std())
        fused.append(d.set_index(["seed", "idx"]).sort_index())
    z = ((fused[0]["z"] + fused[1]["z"]) / 2).groupby(level=1).mean()
    b0 = fused[0].loc[fused[0].index.get_level_values(0)[0]].loc[z.index]
    y, s, site = b0.y.values.astype(int), z.values, b0.site.values
    l3 = rd("results/phase14b/lung3_scored.csv").dropna(subset=["y"])
    tc = rd("results/phase14/tcga_scored.csv").dropna(subset=["y"])
    return {"cv_site_balanced": metric_set(y, s, RF.sb_weights(site, y)), "cv_plain": metric_set(y, s),
            "lung3": metric_set(l3.y.values.astype(int), l3.primary.values),
            "tcga": metric_set(tc.y.values.astype(int), tc.primary.values)}


def main():
    os.makedirs(OUT, exist_ok=True)
    d = tables()
    dev = d[d.cohort.isin(SITES)].reset_index(drop=True)
    R["describe"] = describe(d).round(3).to_dict("records")
    R["clinical_tests"] = clinical_tests(dev)
    clinical_figure(d)
    R["feature_tests"] = feature_tests(dev)
    print(json.dumps(R["clinical_tests"], indent=1), json.dumps(R["feature_tests"], indent=1), flush=True)
    R["metrics"] = all_metrics()
    print(json.dumps(R["metrics"], indent=1), flush=True)
    R["checkpoints"] = {
        "MedSAM2_latest": count_checkpoint(os.path.join(ROOT, "external/MedSAM2/checkpoints/MedSAM2_latest.pt"), "model"),
        "attention_mil_fold1": count_checkpoint(os.path.join(ROOT, "models/attention_mil/fold_1.pt")),
        "mean_mil_fold1": count_checkpoint(os.path.join(ROOT, "models/mean_mil/fold_1.pt")),
        "cnn_attention_fold1": count_checkpoint(os.path.join(ROOT, "models/phase6b_cnn_attention/fold_1.pt")),
    }
    print(json.dumps(R["checkpoints"], indent=1), flush=True)
    R["refit_check"], R["timing"], R["params"] = final_model(dev)
    json.dump(R, open(os.path.join(OUT, "stats.json"), "w"), indent=1, default=float)
    print(json.dumps({k: R[k] for k in ("refit_check", "timing", "params")}, indent=1, default=float))


if __name__ == "__main__":
    main()
