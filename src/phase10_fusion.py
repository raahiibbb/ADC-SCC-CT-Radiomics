"""Phase 10 - equal-weight late fusion of per-block OOF predictions.

Each block model (gtv, rim, loc_clin, fmcib) was trained and scored inside the
SAME outer folds (same seeds -> identical StratifiedKFold splits), so fusing
their out-of-fold scores is itself out-of-fold.  Fusion = mean of per-seed
z-scored ensemble scores with EQUAL weights - nothing is fitted, so no
additional selection or leakage.  EVERY combination is reported (no hidden
cherry-picking); the number of combinations is stated with the results.

Usage: python src/phase10_fusion.py
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_cv import bootstrap, sb_auc, site_aucs  # noqa: E402
from phase10_data import load_cfg, p  # noqa: E402

BLOCKS = ("gtv", "rim", "peri", "loc_clin", "fmcib")
HARMS = ("none", "combat", "balanced_combat")


def load(cfg, fs, harm):
    d = pd.read_csv(os.path.join(p(cfg, "results_dir"), "cv", f"{fs}__{harm}", "oof.csv"))
    d["z"] = d.groupby("seed")["ensemble"].transform(lambda v: (v - v.mean()) / v.std())
    return d.set_index(["seed", "idx"]).sort_index()


def evaluate(cfg, ds):
    base = ds[0]
    z = sum(d["z"] for d in ds) / len(ds)
    per = []
    for s, g in z.groupby(level=0):
        b = base.loc[s]
        y, site = b["y"].values, b["site"].values
        per.append(dict(sb_auc=sb_auc(y, g.values, site), **site_aucs(y, g.values, site)))
    per = pd.DataFrame(per)
    avg = z.groupby(level=1).mean()
    b0 = base.loc[base.index.get_level_values(0)[0]].loc[avg.index]
    ci = bootstrap(b0["y"].values, avg.values, b0["site"].values, cfg)
    return dict(sb_auc=per["sb_auc"].mean(), sb_auc_sd=per["sb_auc"].std(ddof=1),
                LUNG1=per["LUNG1"].mean(), RADIOGENOMICS=per["RADIOGENOMICS"].mean(),
                ci_lo=ci[0], ci_hi=ci[1]), avg, b0


def main():
    cfg = load_cfg()
    out = os.path.join(p(cfg, "results_dir"), "fusion")
    os.makedirs(out, exist_ok=True)
    rows, keep = [], {}
    for harm in HARMS:
        data = {b: load(cfg, b, harm) for b in BLOCKS}
        for r in range(1, len(BLOCKS) + 1):
            for combo in itertools.combinations(BLOCKS, r):
                m, avg, b0 = evaluate(cfg, [data[b] for b in combo])
                rows.append(dict(harm=harm, combo="+".join(combo), n_blocks=r, **m))
                keep[(harm, "+".join(combo))] = (avg, b0)
    df = pd.DataFrame(rows).sort_values(["harm", "sb_auc"], ascending=[True, False])
    df.to_csv(os.path.join(out, "fusion_all_combinations.csv"), index=False)
    for harm in HARMS:
        print(f"\n== {harm}  ({len(df[df.harm == harm])} combinations, all shown)")
        print(df[df.harm == harm].drop(columns="harm").round(3).to_string(index=False))
    # seed-averaged OOF scores of every fusion, for figures / threshold metrics
    allsc = []
    for (harm, combo), (avg, b0) in keep.items():
        allsc.append(pd.DataFrame(dict(harm=harm, combo=combo, idx=avg.index, score=avg.values,
                                       y=b0["y"].values, site=b0["site"].values,
                                       PatientKey=b0["PatientKey"].values)))
    pd.concat(allsc).to_csv(os.path.join(out, "fusion_seed_avg_scores.csv"), index=False)


if __name__ == "__main__":
    main()
