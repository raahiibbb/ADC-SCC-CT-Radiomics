"""Phase 10 - leave-one-site-out TRANSPORT with label-free destination
harmonisation.

Train on one site (all its labels), score the other site ONCE.  The
destination's labels are never used, except by the clearly marked `oracle`
reference, which is reported as an upper bound only and is not deployable.

Destination mapping, all in the raw feature space, before the source
z-score / selection / model:
  none      : destination used as is.
  combat    : destination location/scale = its plain mean/SD (label-free, but
              confounded by the destination's unknown class mix).
  balanced  : destination location/scale = class-balanced mean / within-class
              SD, with class membership estimated by EM from the source model's
              posteriors, jointly with the destination prior (Saerens EM).
  oracle    : as `balanced` but with the TRUE destination labels (upper bound).
The source reference is always the source's own class-balanced location and
within-class SD, so every mapping targets the same space.

Usage: python src/phase10_transport.py [--featureset rim loc_clin ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_cv import MODELS, Prep, fit_model, grid, sb_auc, sb_weights, score  # noqa: E402
from phase10_data import blocks, cohort, load_cfg, matrix, p  # noqa: E402
from phase10_harmonise import _site_stats, saerens_em_prior  # noqa: E402

MAPPINGS = ("none", "combat", "balanced_em", "prior_known", "prior_minus", "prior_plus", "oracle")


def fit_source(cfg, X, y, seed):
    """Inner-CV model selection on the source site, then refit (harm = none:
    one site).  Returns prep + one fitted model per kind + train-score stats."""
    site = np.array(["SRC"] * len(y))
    inner = StratifiedKFold(cfg["experiment"]["n_inner_folds"], shuffle=True, random_state=seed)
    sc = {k: {i: [] for i in range(len(grid(cfg, k)))} for k in MODELS}
    for a, b in inner.split(X, y):
        pr = Prep("none", cfg).fit(X[a], y[a], site[a])
        Xa, Xb = pr.Xtr_, pr.transform(X[b], site[b])
        w = sb_weights(site[a], y[a])
        for k in MODELS:
            for i, hp in enumerate(grid(cfg, k)):
                sel = pr.select(hp["k"])
                m = fit_model(cfg, k, hp, Xa[:, sel], y[a], w, seed)
                sc[k][i].append(roc_auc_score(y[b], score(m, Xb[:, sel])))
    pr = Prep("none", cfg).fit(X, y, site)
    w = sb_weights(site, y)
    fitted = {}
    for k in MODELS:
        best = max(sc[k], key=lambda i: np.mean(sc[k][i]))
        hp = grid(cfg, k)[best]
        sel = pr.select(hp["k"])
        m = fit_model(cfg, k, hp, pr.Xtr_[:, sel], y, w, seed)
        s = score(m, pr.Xtr_[:, sel])
        fitted[k] = (m, sel, s.mean(), s.std() + 1e-8, hp)
    return pr, fitted


def ens_score(pr, fitted, Xraw):
    Z = pr.transform(Xraw, np.array(["SRC"] * len(Xraw)))
    zs = [(score(m, Z[:, sel]) - mu) / sd for m, sel, mu, sd, _ in fitted.values()]
    lr, sel = fitted["logreg"][0], fitted["logreg"][1]
    return np.mean(zs, axis=0), lr.predict_proba(Z[:, sel])[:, 1]


def raw_clean(pr, X):
    X = X[:, pr.keep_]
    X = np.where(np.isnan(X), pr.med_, X)
    return X[:, pr.keep2_]


def map_dest(src_loc, src_sd, Xd, w1=None, balanced=False):
    loc, var = _site_stats(Xd, None, balanced, w1)
    return (Xd - loc) / np.sqrt(np.maximum(var, 1e-12)) * src_sd + src_loc


def map_dest_prior(src_loc, src_sd, src_delta, Xd, pi):
    """Prevalence-informed ComBat: the destination is described by ONE number,
    its ADC prevalence pi (a site's case mix, not any patient's label).
    Assuming the class effect transports up to the site's scale r:
        total var_d = r^2 (sd_s^2 + pi (1-pi) delta_s^2)
        balanced loc_d = mean_d - (pi - 0.5) r delta_s
    and the destination is mapped by (x - loc_d) / r + src_loc."""
    m_d, v_d = Xd.mean(0), Xd.var(0)
    r = np.sqrt(np.maximum(v_d, 1e-12) / np.maximum(src_sd ** 2 + pi * (1 - pi) * src_delta ** 2, 1e-12))
    loc_d = m_d - (pi - 0.5) * r * src_delta
    return (Xd - loc_d) / r + src_loc


def decision_metrics(y, p):
    yhat = (p >= 0.5).astype(int)
    sens = float((yhat[y == 1] == 1).mean())
    spec = float((yhat[y == 0] == 0).mean())
    return dict(bacc=0.5 * (sens + spec), sens=sens, spec=spec)


def run_direction(cfg, X, y, site, src, dst, seed):
    s, d = site == src, site == dst
    Xs, ys, Xd, yd = X[s], y[s], X[d], y[d]
    pr, fitted = fit_source(cfg, Xs, ys, seed)
    # source reference (class-balanced) in the cleaned raw space
    Cs = raw_clean(pr, Xs)
    src_loc, src_var = _site_stats(Cs, ys, True)
    src_sd = np.sqrt(np.maximum(src_var, 1e-12))
    Cd = raw_clean(pr, Xd)

    def full(C):  # cleaned-raw -> something ens_score accepts (re-expand)
        out = np.full((len(C), len(pr.keep_)), np.nan)
        idx = np.where(pr.keep_)[0][pr.keep2_]
        out[:, idx] = C
        return out

    m1 = Cs[ys == 1].mean(0)
    m0 = Cs[ys == 0].mean(0)
    src_delta = m1 - m0
    pi_true = float(yd.mean())
    res, prob, extra = {}, {}, {}
    res["none"], prob["none"] = ens_score(pr, fitted, Xd)
    res["combat"], prob["combat"] = ens_score(pr, fitted, full(map_dest(src_loc, src_sd, Cd)))
    # label-free EM: start from the plain mapping, iterate soft balanced mapping
    p = prob["combat"]
    pi_hist = []
    for _ in range(cfg["harmonisation"]["em_iterations"]):
        q, pi = saerens_em_prior(p, 0.5)
        pi_hist.append(float(pi))
        e, p_new = ens_score(pr, fitted, full(map_dest(src_loc, src_sd, Cd, q, True)))
        done = np.max(np.abs(p_new - p)) < cfg["harmonisation"]["em_tolerance"]
        p = p_new
        if done:
            break
    res["balanced_em"], prob["balanced_em"] = e, p
    # prevalence-informed: ONE number per destination site
    res["prior_known"], prob["prior_known"] = ens_score(
        pr, fitted, full(map_dest_prior(src_loc, src_sd, src_delta, Cd, pi_true)))
    mis = cfg["transport"]["prior_misspecification"]
    for tag, pi_m in (("prior_minus", pi_true - mis), ("prior_plus", pi_true + mis)):
        pi_m = float(np.clip(pi_m, 0.02, 0.98))
        res[tag], prob[tag] = ens_score(
            pr, fitted, full(map_dest_prior(src_loc, src_sd, src_delta, Cd, pi_m)))
    res["oracle"], prob["oracle"] = ens_score(
        pr, fitted, full(map_dest(src_loc, src_sd, Cd, yd.astype(float), True)))
    extra["em_prior_estimates"] = pi_hist
    extra["true_dest_prior"] = pi_true
    met = {}
    for m in MAPPINGS:
        met[m + "__auc"] = float(roc_auc_score(yd, res[m]))
        for k, v in decision_metrics(yd, prob[m]).items():
            met[m + "__" + k] = v
    return met, extra, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featureset", nargs="*", default=["rim", "gtv_rim", "loc_clin"])
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    cfg = load_cfg()
    coh, blk = cohort(cfg), blocks(cfg)
    y, site = coh["y"].values, coh["site"].values
    outdir = os.path.join(p(cfg, "results_dir"), "transport")
    os.makedirs(outdir, exist_ok=True)
    dirs = [("LUNG1", "RADIOGENOMICS"), ("RADIOGENOMICS", "LUNG1")]
    allres = {}
    for fs in a.featureset:
        X, _ = matrix(cfg, fs, coh, blk)
        jobs = [(src, dst, sd) for src, dst in dirs for sd in cfg["experiment"]["seeds"]]
        out = Parallel(n_jobs=a.jobs)(delayed(run_direction)(cfg, X, y, site, s_, d_, sd)
                                      for s_, d_, sd in jobs)
        rows, scores = [], []
        for (s_, d_, sd), (auc, extra, res) in zip(jobs, out):
            rows.append(dict(featureset=fs, direction=f"{s_}->{d_}", seed=sd, **auc,
                             em_prior_final=extra["em_prior_estimates"][-1] if extra["em_prior_estimates"] else None,
                             true_dest_prior=extra["true_dest_prior"]))
            idx = np.where(site == d_)[0]
            for m in MAPPINGS:
                scores.append(pd.DataFrame(dict(featureset=fs, direction=f"{s_}->{d_}", seed=sd,
                                                mapping=m, idx=idx, y=y[idx], score=res[m])))
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(outdir, f"{fs}_per_seed.csv"), index=False)
        pd.concat(scores).to_csv(os.path.join(outdir, f"{fs}_scores.csv"), index=False)
        summ = df.groupby("direction")[[c for c in df.columns if "__" in c]
                                       + ["em_prior_final", "true_dest_prior"]].mean()
        allres[fs] = summ.to_dict()
        print(f"\n== {fs}")
        for metric in ("auc", "bacc", "sens", "spec"):
            t = summ[[m + "__" + metric for m in MAPPINGS]].copy()
            t.columns = MAPPINGS
            t.loc["mean"] = t.mean()
            print(f"-- {metric}\n{t.round(3).to_string()}")
        print(summ[["em_prior_final", "true_dest_prior"]].round(3).to_string(), flush=True)
    json.dump(allres, open(os.path.join(outdir, "summary_%s.json" % "_".join(a.featureset)), "w"), indent=1)


if __name__ == "__main__":
    main()
