"""Phase 11 - three-hospital evaluation (LUNG1, Radiogenomics, Lung-PET-CT-Dx).

  pooled : 5 seeds x 5-fold, strata = site x label, site-balanced AUC
           (Phase-10 machinery: harmonise -> select -> 4 models -> ensemble).
  loso   : INTERNAL-EXTERNAL CV.  Train on two hospitals (inner-CV model
           selection, ComBat fitted on those two), then score the third ONCE.
           The held-out hospital is mapped into the training reference with
           its own label-free location/scale; its labels are only used to
           compute the final AUC.
  fusion : equal-weight fusion of block scores; every combination reported.

Usage:
  python src/phase11_cv.py pooled --blocks ring_in shells loc_clin ...
  python src/phase11_cv.py loso   --blocks ...
  python src/phase11_cv.py fusion
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_cv import MODELS, Prep, bootstrap, fit_model, grid, sb_auc, sb_weights, score, site_aucs  # noqa: E402
from phase10_data import load_cfg  # noqa: E402
from phase10_harmonise import Harmoniser  # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FEAT = os.path.join(ROOT, "global_features", "phase11")
RES = os.path.join(ROOT, "results", "phase11")
HARM = "combat"
SITES = ("LUNG1", "RADIOGENOMICS", "LPCD")

# block name -> (csv, column prefix filter)
BLOCKS = {
    "gtv": ("radiomics", "rad__gtv__"),
    "ring_in": ("radiomics", "rad__ring_in__"),
    "shells": ("radiomics", "rad__shell"),
    "loc_clin": (("loc", "clin"), ("loc__", "clin__")),
    "fmcib": ("fmcib", "fmcib__"),
    "fmcib_bag": ("fmcib_bagmean", "fbmean__"),
}


def cohort():
    c = pd.read_csv(os.path.join(ROOT, "cohort", "phase11_three_site_cohort.csv"))
    c["y"] = c["label"].astype(int)
    return c


def block_matrix(coh, name):
    files, prefixes = BLOCKS[name]
    files = files if isinstance(files, tuple) else (files,)
    prefixes = prefixes if isinstance(prefixes, tuple) else (prefixes,)
    parts = []
    for f, pre in zip(files, prefixes):
        d = pd.read_csv(os.path.join(FEAT, f + ".csv"))
        d = d[d.status == "ok"].set_index("PatientKey")
        cols = [c for c in d.columns if c.startswith(pre) and "voxels" not in c]
        parts.append(d[cols].apply(pd.to_numeric, errors="coerce").reindex(coh.PatientKey))
    X = pd.concat(parts, axis=1)
    return X.values.astype(float), list(X.columns)


def site_leak_auc(Xtr, Xte, site_tr, site_te, y_tr, y_te):
    """mean one-vs-rest AUC of a hospital classifier (label-balanced weights)."""
    out = []
    for s in np.unique(site_tr):
        t_tr, t_te = (site_tr == s).astype(int), (site_te == s).astype(int)
        if t_te.min() == t_te.max():
            continue
        m = LogisticRegression(C=0.1, max_iter=5000).fit(Xtr, t_tr, sample_weight=sb_weights(site_tr, y_tr))
        out.append(roc_auc_score(t_te, m.decision_function(Xte), sample_weight=sb_weights(site_te, y_te)))
    return float(np.mean(out))


def select_and_fit(cfg, X, y, site, seed, harm):
    """Inner-CV hyper-parameter choice on (X, y, site), then refit on all of it."""
    strata = np.array([f"{a}|{b}" for a, b in zip(site, y)])
    inner = StratifiedKFold(cfg["experiment"]["n_inner_folds"], shuffle=True, random_state=seed)
    sc = {k: {i: [] for i in range(len(grid(cfg, k)))} for k in MODELS}
    for a, b in inner.split(X, strata):
        pr = Prep(harm, cfg).fit(X[a], y[a], site[a])
        Xa, Xb = pr.Xtr_, pr.transform(X[b], site[b])
        w = sb_weights(site[a], y[a])
        for k in MODELS:
            for i, hp in enumerate(grid(cfg, k)):
                sel = pr.select(hp["k"])
                m = fit_model(cfg, k, hp, Xa[:, sel], y[a], w, seed)
                sc[k][i].append(sb_auc(y[b], score(m, Xb[:, sel]), site[b]))
    pr = Prep(harm, cfg).fit(X, y, site)
    w = sb_weights(site, y)
    fitted = {}
    for k in MODELS:
        best = max(sc[k], key=lambda i: np.mean(sc[k][i]))
        hp = grid(cfg, k)[best]
        sel = pr.select(hp["k"])
        m = fit_model(cfg, k, hp, pr.Xtr_[:, sel], y, w, seed)
        s = score(m, pr.Xtr_[:, sel])
        fitted[k] = (m, sel, s.mean(), s.std() + 1e-8, dict(hp, inner=float(np.mean(sc[k][best]))))
    return pr, fitted


def ens(fitted, Z):
    return np.mean([(score(m, Z[:, sel]) - mu) / sd for m, sel, mu, sd, _ in fitted.values()], axis=0)


# ------------------------------------------------------------------ pooled
def pooled_fold(cfg, X, y, site, tr, te, seed, fold, harm):
    pr, fitted = select_and_fit(cfg, X[tr], y[tr], site[tr], seed + 1000 * fold, harm)
    Zte = pr.transform(X[te], site[te])
    out = pd.DataFrame({"idx": te, "seed": seed, "fold": fold, "ensemble": ens(fitted, Zte)})
    for k, (m, sel, mu, sd, hp) in fitted.items():
        out[k] = score(m, Zte[:, sel])
    leak = site_leak_auc(pr.Xtr_, Zte, site[tr], site[te], y[tr], y[te])
    return out, dict(seed=seed, fold=fold, site_leak_auc=leak,
                     chosen={k: v[4] for k, v in fitted.items()})


def run_pooled(cfg, coh, blocks, harms, jobs):
    y, site = coh.y.values, coh.site.values
    strata = np.array([f"{a}|{b}" for a, b in zip(site, y)])
    for b in blocks:
        X, _ = block_matrix(coh, b)
        for harm in harms:
            d = os.path.join(RES, "pooled", f"{b}__{harm}")
            if os.path.isfile(os.path.join(d, "metrics.json")):
                print("[skip]", b, harm)
                continue
            os.makedirs(d, exist_ok=True)
            jobs_ = [(s, f, tr, te) for s in cfg["experiment"]["seeds"]
                     for f, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=s).split(X, strata))]
            t0 = time.time()
            res = Parallel(n_jobs=jobs)(delayed(pooled_fold)(cfg, X, y, site, tr, te, s, f, harm)
                                        for s, f, tr, te in jobs_)
            df = pd.concat([r[0] for r in res], ignore_index=True)
            df.insert(1, "PatientKey", coh.PatientKey.values[df.idx])
            df.insert(2, "site", site[df.idx])
            df.insert(3, "y", y[df.idx])
            df.to_csv(os.path.join(d, "oof.csv"), index=False)
            per = []
            for s, g in df.groupby("seed"):
                per.append(dict(sb_auc=sb_auc(g.y.values, g.ensemble.values, g.site.values),
                                raw_auc=float(roc_auc_score(g.y, g.ensemble)),
                                **site_aucs(g.y.values, g.ensemble.values, g.site.values)))
            per = pd.DataFrame(per)
            met = dict(block=b, harm=harm, n=int(len(y)), ensemble_mean=per.mean().to_dict(),
                       ensemble_sd=per.std(ddof=1).to_dict(),
                       site_leak_auc=float(np.mean([r[1]["site_leak_auc"] for r in res])),
                       folds=[r[1] for r in res], seconds=round(time.time() - t0, 1))
            json.dump(met, open(os.path.join(d, "metrics.json"), "w"), indent=1)
            e = met["ensemble_mean"]
            print(f"{b:10s} {harm:16s} sbAUC {e['sb_auc']:.3f} | " +
                  " ".join(f"{s} {e.get(s, float('nan')):.3f}" for s in SITES) +
                  f" | siteLeak {met['site_leak_auc']:.3f}", flush=True)


# -------------------------------------------------------------------- loso
def loso_one(cfg, X, y, site, held, seed, harm):
    tr, te = site != held, site == held
    pr, fitted = select_and_fit(cfg, X[tr], y[tr], site[tr], seed, harm)
    # label-free mapping of the held-out hospital into the training reference
    Xte_raw = X[te][:, pr.keep_]
    Xte_raw = np.where(np.isnan(Xte_raw), pr.med_, Xte_raw)[:, pr.keep2_]
    if harm != "none":
        pr.h_.fit_new_site(Xte_raw, held, balanced=False)
    Zte = (pr.h_.transform(Xte_raw, np.array([held] * te.sum())) - pr.mu_) / pr.sd_
    s = ens(fitted, Zte)
    return pd.DataFrame({"idx": np.where(te)[0], "seed": seed, "held_out": held, "score": s})


def run_loso(cfg, coh, blocks, harms, jobs):
    y, site = coh.y.values, coh.site.values
    for b in blocks:
        X, _ = block_matrix(coh, b)
        for harm in harms:
            d = os.path.join(RES, "loso", f"{b}__{harm}")
            if os.path.isfile(os.path.join(d, "scores.csv")):
                print("[skip]", b, harm)
                continue
            os.makedirs(d, exist_ok=True)
            res = Parallel(n_jobs=jobs)(delayed(loso_one)(cfg, X, y, site, h, s, harm)
                                        for h in SITES for s in cfg["experiment"]["seeds"])
            df = pd.concat(res, ignore_index=True)
            df["y"] = y[df.idx]
            df["PatientKey"] = coh.PatientKey.values[df.idx]
            df.to_csv(os.path.join(d, "scores.csv"), index=False)
            aucs = df.groupby(["held_out", "seed"]).apply(lambda g: roc_auc_score(g.y, g.score)).groupby(level=0).mean()
            print(f"{b:10s} {harm:16s} LOSO " + " ".join(f"{k} {v:.3f}" for k, v in aucs.items())
                  + f" | mean {aucs.mean():.3f}", flush=True)


# ------------------------------------------------------------------ fusion
def zload(path, col):
    d = pd.read_csv(path)
    key = ["seed", "idx"] if "held_out" not in d else ["held_out", "seed", "idx"]
    grp = ["seed"] if "held_out" not in d else ["held_out", "seed"]
    d["z"] = d.groupby(grp)[col].transform(lambda v: (v - v.mean()) / v.std())
    return d.set_index(key).sort_index()


def run_fusion(cfg, blocks, harm):
    rows = []
    names = [b for b in blocks if os.path.isfile(os.path.join(RES, "pooled", f"{b}__{harm}", "oof.csv"))]
    P = {b: zload(os.path.join(RES, "pooled", f"{b}__{harm}", "oof.csv"), "ensemble") for b in names}
    L = {b: zload(os.path.join(RES, "loso", f"{b}__{harm}", "scores.csv"), "score")
         for b in names if os.path.isfile(os.path.join(RES, "loso", f"{b}__{harm}", "scores.csv"))}
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            z = sum(P[b]["z"] for b in combo) / r
            base = P[combo[0]]
            per = []
            for s, g in z.groupby(level=0):
                bb = base.loc[s]
                per.append(dict(sb_auc=sb_auc(bb.y.values, g.values, bb.site.values),
                                **site_aucs(bb.y.values, g.values, bb.site.values)))
            per = pd.DataFrame(per).mean()
            avg = z.groupby(level=1).mean()
            b0 = base.loc[base.index.get_level_values(0)[0]].loc[avg.index]
            ci = bootstrap(b0.y.values, avg.values, b0.site.values, cfg)
            row = dict(combo="+".join(combo), n_blocks=r, sb_auc=per.sb_auc, ci_lo=ci[0], ci_hi=ci[1],
                       **{s: per.get(s, np.nan) for s in SITES})
            if all(b in L for b in combo):
                zl = sum(L[b]["z"] for b in combo) / r
                yl = L[combo[0]]["y"]
                la = pd.DataFrame({"z": zl, "y": yl}).groupby(level=[0, 1]).apply(
                    lambda g: roc_auc_score(g.y, g.z)).groupby(level=0).mean()
                for s in SITES:
                    row["loso_" + s] = la.get(s, np.nan)
                row["loso_mean"] = la.mean()
            rows.append(row)
    df = pd.DataFrame(rows).sort_values("sb_auc", ascending=False)
    os.makedirs(os.path.join(RES, "fusion"), exist_ok=True)
    df.to_csv(os.path.join(RES, "fusion", f"fusion_all_{harm}.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["pooled", "loso", "fusion"])
    ap.add_argument("--blocks", nargs="*", default=list(BLOCKS))
    ap.add_argument("--harm", nargs="*", default=[HARM])
    ap.add_argument("--jobs", type=int, default=14)
    a = ap.parse_args()
    cfg = load_cfg()
    coh = cohort()
    if a.step == "pooled":
        run_pooled(cfg, coh, a.blocks, a.harm, a.jobs)
    elif a.step == "loso":
        run_loso(cfg, coh, a.blocks, a.harm, a.jobs)
    else:
        for h in a.harm:
            run_fusion(cfg, a.blocks, h)


if __name__ == "__main__":
    main()
