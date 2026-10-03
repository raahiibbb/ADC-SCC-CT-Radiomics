"""Phase 10 - shortcut demonstration: how a naive pooled analysis of these two
cohorts produces an inflated ROC-AUC.

A. SITE ONLY: score = the site's training ADC prevalence (no image at all).
B. NAIVE: rim radiomics, no harmonisation, no site-balanced weights, plain
   pooled ROC-AUC  (the "high score" setting).
C. The same NAIVE model scored with the site-balanced AUC.
All in the identical 5 x 5 stratified CV used by every Phase-10 result.

Usage: python src/phase10_shortcut_demo.py
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_cv import Prep, sb_auc, site_aucs  # noqa: E402
from phase10_data import blocks, cohort, load_cfg, matrix, p  # noqa: E402

warnings.filterwarnings("ignore")
K, C = 25, 0.1   # a fixed, ordinary configuration - deliberately not tuned


def main():
    cfg = load_cfg()
    coh, blk = cohort(cfg), blocks(cfg)
    y, site = coh["y"].values, coh["site"].values
    X, _ = matrix(cfg, "rim", coh, blk)
    strata = np.array([f"{a}|{b}" for a, b in zip(site, y)])
    out = {"site_only": [], "naive": []}
    for seed in cfg["experiment"]["seeds"]:
        s_site, s_naive = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strata):
            prev = {g: y[tr][site[tr] == g].mean() for g in np.unique(site)}
            s_site[te] = [prev[g] for g in site[te]]
            pr = Prep("none", cfg).fit(X[tr], y[tr], site[tr])
            sel = pr.select(K)
            m = LogisticRegression(C=C, max_iter=5000).fit(pr.Xtr_[:, sel], y[tr])   # no weights
            s_naive[te] = m.decision_function(pr.transform(X[te], site[te])[:, sel])
        for name, s in (("site_only", s_site), ("naive", s_naive)):
            out[name].append(dict(raw_pooled_auc=float(roc_auc_score(y, s)),
                                  site_balanced_auc=sb_auc(y, s, site),
                                  **site_aucs(y, s, site) if name == "naive" else {}))
    summ = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in out.items()}
    for k, v in summ.items():
        print(k, {m: round(x, 3) for m, x in v.items()})
    d = os.path.join(p(cfg, "results_dir"), "shortcut_demo")
    os.makedirs(d, exist_ok=True)
    json.dump(dict(summary=summ, per_seed=out, config=dict(features="rim", k=K, C=C)),
              open(os.path.join(d, "shortcut_demo.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
