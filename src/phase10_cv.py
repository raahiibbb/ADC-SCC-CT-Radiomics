"""Phase 10 - pooled multi-centre repeated nested CV.

For every (feature set, harmonisation) pair:
  5 seeds x 5 outer folds, stratified on site x label.
  Inside every outer TRAINING partition only:
     harmonise -> median-impute -> z-score -> rank features by site-balanced
     univariate AUC -> correlation prune -> top-k -> model with site x label
     balanced sample weights.
  k and model hyper-parameters are chosen by a 3-fold inner CV on the
  site-balanced ROC-AUC.  Test patients are touched once, at scoring.

Also, per outer fold, a SITE classifier is trained on the harmonised training
features and scored on the test fold (label-balanced weights), to measure how
much acquisition information survives harmonisation.

Usage:
  python src/phase10_cv.py --featureset gtv_rim --harm balanced_combat
  python src/phase10_cv.py --all            # every set/method whose blocks exist
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_data import FEATURE_SETS, blocks, cohort, load_cfg, matrix, p  # noqa: E402
from phase10_harmonise import Harmoniser  # noqa: E402

warnings.filterwarnings("ignore")
MODELS = ("logreg", "svm", "lgbm", "ridge_all")


# --------------------------------------------------------------------- metrics
def sb_weights(site, y):
    """1 / n_{site,label}, mean 1: every site counts equally, each site 50/50."""
    key = pd.Series(site).astype(str) + "|" + pd.Series(y).astype(str)
    cnt = key.map(key.value_counts())
    w = 1.0 / cnt.values
    return w * len(w) / w.sum()


def sb_auc(y, s, site):
    return float(roc_auc_score(y, s, sample_weight=sb_weights(site, y)))


def site_aucs(y, s, site):
    out = {}
    for g in np.unique(site):
        m = site == g
        out[str(g)] = float(roc_auc_score(y[m], s[m]))
    return out


def univariate_score(X, y, site):
    """Mean over sites of the within-site AUC, folded around 0.5 (vectorised)."""
    sc = np.zeros(X.shape[1])
    sites = np.unique(site)
    for g in sites:
        m = site == g
        yy = y[m]
        n1, n0 = yy.sum(), (1 - yy).sum()
        r = rankdata(X[m], axis=0)
        auc = (r[yy == 1].sum(0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        sc += np.abs(auc - 0.5)
    return sc / len(sites)


# ---------------------------------------------------------------- preprocessing
class Prep:
    """Everything fitted on the training partition, then frozen."""

    def __init__(self, harm, cfg):
        self.harm, self.cfg = harm, cfg

    def fit(self, X, y, site):
        fc = self.cfg["features"]
        miss = np.isnan(X).mean(0)
        self.keep_ = miss <= 0.2
        X = X[:, self.keep_]
        self.med_ = np.nanmedian(X, axis=0)
        X = np.where(np.isnan(X), self.med_, X)
        sd = X.std(0)
        self.keep2_ = sd > fc["drop_near_constant_std"]
        X = X[:, self.keep2_]
        self.h_ = Harmoniser(self.harm, self.cfg["harmonisation"]["empirical_bayes"]).fit(X, site, y)
        X = self.h_.transform(X, site)
        self.mu_, self.sd_ = X.mean(0), X.std(0) + 1e-8
        X = (X - self.mu_) / self.sd_
        self.order_ = np.argsort(-univariate_score(X, y, site))
        self.Xtr_ = X
        return self

    def transform(self, X, site):
        X = X[:, self.keep_]
        X = np.where(np.isnan(X), self.med_, X)[:, self.keep2_]
        X = self.h_.transform(X, site)
        return (X - self.mu_) / self.sd_

    def select(self, k):
        if k == 0:  # every feature (dense ridge model)
            return list(range(self.Xtr_.shape[1]))
        thr = self.cfg["features"]["correlation_prune"]
        if not hasattr(self, "_sel_cache"):
            self._sel_cache = {}
        kmax = max(self.cfg["features"]["top_k_grid"])
        if "full" not in self._sel_cache:
            kept = []
            C = None
            for j in self.order_:
                if len(kept) >= kmax:
                    break
                if kept:
                    v = self.Xtr_[:, j]
                    r = np.abs(C.T @ v) / len(v)
                    if r.max() > thr:
                        continue
                kept.append(j)
                C = self.Xtr_[:, kept]
            self._sel_cache["full"] = kept
        return self._sel_cache["full"][:k]


# ---------------------------------------------------------------------- models
def grid(cfg, kind):
    ks = cfg["features"]["top_k_grid"]
    m = cfg["models"]
    if kind == "logreg":
        return [dict(k=k, C=c) for k in ks for c in m["logreg"]["C_grid"]]
    if kind == "ridge_all":
        return [dict(k=0, C=c) for c in m["ridge_all"]["C_grid"]]
    if kind == "svm":
        return [dict(k=k, C=c) for k in ks for c in m["svm"]["C_grid"]]
    return [dict(k=k, num_leaves=nl) for k in ks for nl in m["lgbm"]["num_leaves_grid"]]


def fit_model(cfg, kind, hp, X, y, w, seed):
    if kind in ("logreg", "ridge_all"):
        mdl = LogisticRegression(C=hp["C"], max_iter=5000)
        mdl.fit(X, y, sample_weight=w)
        return mdl
    if kind == "svm":
        mdl = SVC(C=hp["C"], kernel="rbf", gamma=cfg["models"]["svm"]["gamma"])
        mdl.fit(X, y, sample_weight=w)
        return mdl
    import lightgbm as lgb
    g = cfg["models"]["lgbm"]
    mdl = lgb.LGBMClassifier(n_estimators=g["n_estimators"], learning_rate=g["learning_rate"],
                             num_leaves=hp["num_leaves"], min_child_samples=g["min_child_samples"],
                             subsample=g["subsample"], subsample_freq=1,
                             colsample_bytree=g["colsample_bytree"], random_state=seed,
                             n_jobs=1, verbose=-1)
    mdl.fit(X, y, sample_weight=w)
    return mdl


def score(mdl, X):
    if hasattr(mdl, "decision_function"):
        return mdl.decision_function(X)
    return mdl.predict_proba(X)[:, 1]


# ------------------------------------------------------------------ one fold
def run_fold(cfg, X, y, site, tr, te, harm, seed, fold):
    t0 = time.time()
    strata = np.array([f"{a}|{b}" for a, b in zip(site, y)])
    # ---- inner CV (training partition only)
    inner = StratifiedKFold(cfg["experiment"]["n_inner_folds"], shuffle=True, random_state=seed + 1000 * fold)
    inner_scores = {kind: {i: [] for i in range(len(grid(cfg, kind)))} for kind in MODELS}
    for itr, iva in inner.split(tr, strata[tr]):
        a, b = tr[itr], tr[iva]
        pr = Prep(harm, cfg).fit(X[a], y[a], site[a])
        Xa, Xb = pr.Xtr_, pr.transform(X[b], site[b])
        w = sb_weights(site[a], y[a])
        for kind in MODELS:
            for i, hp in enumerate(grid(cfg, kind)):
                sel = pr.select(hp["k"])
                m = fit_model(cfg, kind, hp, Xa[:, sel], y[a], w, seed)
                inner_scores[kind][i].append(sb_auc(y[b], score(m, Xb[:, sel]), site[b]))
    # ---- refit on the whole outer training partition
    pr = Prep(harm, cfg).fit(X[tr], y[tr], site[tr])
    Xtr, Xte = pr.Xtr_, pr.transform(X[te], site[te])
    w = sb_weights(site[tr], y[tr])
    out = pd.DataFrame({"idx": te, "seed": seed, "fold": fold})
    chosen = {}
    zs = []
    for kind in MODELS:
        means = {i: np.mean(v) for i, v in inner_scores[kind].items()}
        best = max(means, key=means.get)
        hp = grid(cfg, kind)[best]
        chosen[kind] = dict(hp, inner_sb_auc=float(means[best]))
        sel = pr.select(hp["k"])
        m = fit_model(cfg, kind, hp, Xtr[:, sel], y[tr], w, seed)
        s_tr, s_te = score(m, Xtr[:, sel]), score(m, Xte[:, sel])
        out[kind] = s_te
        zs.append((s_te - s_tr.mean()) / (s_tr.std() + 1e-8))
    out["ensemble"] = np.mean(zs, axis=0)
    # ---- site-leakage diagnostic on the harmonised training features
    site_bin = (site == np.unique(site)[0]).astype(int)
    lab_w_tr = sb_weights(site[tr], y[tr])
    sc = LogisticRegression(C=0.1, max_iter=5000).fit(Xtr, site_bin[tr], sample_weight=lab_w_tr)
    s_site = sc.decision_function(Xte)
    site_auc = float(roc_auc_score(site_bin[te], s_site, sample_weight=sb_weights(site[te], y[te])))
    meta = dict(seed=seed, fold=fold, harm=harm, chosen=chosen, site_classifier_auc=site_auc,
                n_train=int(len(tr)), n_test=int(len(te)), n_features_after_filter=int(Xtr.shape[1]),
                seconds=round(time.time() - t0, 1))
    return out, meta


# ------------------------------------------------------------------ one config
def summarise(df, coh, cfg, models):
    y, site = coh["y"].values, coh["site"].values
    res = {}
    for kind in models:
        per_seed = []
        for sd, g in df.groupby("seed"):
            g = g.sort_values("idx")
            idx = g["idx"].values
            per_seed.append(dict(sb_auc=sb_auc(y[idx], g[kind].values, site[idx]),
                                 raw_auc=float(roc_auc_score(y[idx], g[kind].values)),
                                 **{"auc_" + k: v for k, v in site_aucs(y[idx], g[kind].values, site[idx]).items()}))
        ps = pd.DataFrame(per_seed)
        # seed-averaged OOF score per patient (each seed's scores z-scored first)
        z = df.copy()
        z[kind] = z.groupby("seed")[kind].transform(lambda v: (v - v.mean()) / (v.std() + 1e-8))
        avg = z.groupby("idx")[kind].mean().reindex(range(len(y))).values
        boot = bootstrap(y, avg, site, cfg)
        res[kind] = dict(mean={c: float(ps[c].mean()) for c in ps},
                         sd={c: float(ps[c].std(ddof=1)) for c in ps},
                         seed_avg=dict(sb_auc=sb_auc(y, avg, site),
                                       raw_auc=float(roc_auc_score(y, avg)),
                                       **{"auc_" + k: v for k, v in site_aucs(y, avg, site).items()}),
                         sb_auc_95ci=boot)
    return res


def bootstrap(y, s, site, cfg):
    rng = np.random.default_rng(cfg["evaluation"]["bootstrap_seed"])
    strata = pd.Series(site).astype(str) + "|" + pd.Series(y).astype(str)
    groups = [np.where(strata.values == k)[0] for k in strata.unique()]
    vals = []
    for _ in range(cfg["evaluation"]["bootstrap_resamples"]):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        vals.append(sb_auc(y[idx], s[idx], site[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def run_config(cfg, featureset, harm, coh, blk, n_jobs):
    outdir = os.path.join(p(cfg, "results_dir"), "cv", f"{featureset}__{harm}")
    if os.path.isfile(os.path.join(outdir, "metrics.json")):
        print(f"[skip] {featureset} / {harm} already complete")
        return json.load(open(os.path.join(outdir, "metrics.json")))
    os.makedirs(outdir, exist_ok=True)
    X, names = matrix(cfg, featureset, coh, blk)
    y, site = coh["y"].values, coh["site"].values
    strata = np.array([f"{a}|{b}" for a, b in zip(site, y)])
    jobs = []
    for seed in cfg["experiment"]["seeds"]:
        skf = StratifiedKFold(cfg["experiment"]["n_outer_folds"], shuffle=True, random_state=seed)
        for f, (tr, te) in enumerate(skf.split(X, strata)):
            jobs.append((seed, f, tr, te))
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs)(delayed(run_fold)(cfg, X, y, site, tr, te, harm, s, f)
                                  for s, f, tr, te in jobs)
    df = pd.concat([r[0] for r in res], ignore_index=True)
    meta = [r[1] for r in res]
    df.insert(1, "PatientKey", coh["PatientKey"].values[df["idx"].values])
    df.insert(2, "site", site[df["idx"].values])
    df.insert(3, "y", y[df["idx"].values])
    df.to_csv(os.path.join(outdir, "oof.csv"), index=False)
    metrics = dict(featureset=featureset, harm=harm, n_features_input=len(names),
                   n_patients=int(len(y)), models=summarise(df, coh, cfg, MODELS + ("ensemble",)),
                   site_classifier_auc_mean=float(np.mean([m["site_classifier_auc"] for m in meta])),
                   folds=meta, seconds=round(time.time() - t0, 1))
    json.dump(metrics, open(os.path.join(outdir, "metrics.json"), "w"), indent=1)
    return metrics


def line(m):
    e = m["models"]["ensemble"]
    best = max(m["models"], key=lambda k: m["models"][k]["mean"]["sb_auc"])
    b = m["models"][best]
    return (f"{m['featureset']:<18} {m['harm']:<16} ens sbAUC {e['mean']['sb_auc']:.3f}"
            f" (L1 {e['mean'].get('auc_LUNG1', float('nan')):.3f} RG {e['mean'].get('auc_RADIOGENOMICS', float('nan')):.3f})"
            f" | best {best} {b['mean']['sb_auc']:.3f} | siteAUC {m['site_classifier_auc_mean']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--featureset", nargs="*")
    ap.add_argument("--harm", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    cfg = load_cfg()
    coh, blk = cohort(cfg), blocks(cfg)
    sets = a.featureset or [k for k, v in FEATURE_SETS.items() if all(b in blk for b in v)]
    harms = a.harm or cfg["harmonisation"]["methods"]
    for fs in sets:
        for h in harms:
            m = run_config(cfg, fs, h, coh, blk, a.jobs)
            print(line(m), flush=True)


if __name__ == "__main__":
    main()
