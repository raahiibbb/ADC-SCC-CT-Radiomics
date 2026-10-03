"""Phase 8B - cohort assembly, the frozen multicenter splits and the freeze step.

This module runs BEFORE any Phase-8B label reaches an estimator.  It

  * assembles the 343-patient multicenter cohort with globally unique
    PatientKeys (LUNG1::... / RADIOGENOMICS::...),
  * proves the two cached global-radiomics banks are name-for-name and
    order-for-order compatible and carry the same extraction signature,
  * screens for duplicated patients across the two collections,
  * freezes the source-stratified 5-fold outer split, the inner-CV assignment
    and the cross-cohort assignment, and
  * writes reports/phase8b_run.json carrying the SHA-256 of
    docs/PHASE8B_MULTICENTER_PROTOCOL.md, the frozen gate and the frozen
    primary-metric convention.

Every modelling driver calls `verify_frozen(cfg)` at start-up and refuses to
run if the protocol hash, the config hash or the split hash has moved.

PyRadiomics is never invoked here.  Nothing is harmonised.

    ./.venv/Scripts/python.exe src/phase8b_data.py --config config/phase8b_multicenter.yaml --freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

META_COLUMNS = ("PatientID", "Histology", "label")


# ---------------------------------------------------------------------------
# config / paths / hashing
# ---------------------------------------------------------------------------

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_project_root"] = PROJECT_ROOT
    return cfg


def ppath(cfg, key_or_rel):
    rel = cfg["paths"].get(key_or_rel, key_or_rel)
    return os.path.join(cfg["_project_root"], *rel.split("/"))


def sha256(path):
    d = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            d.update(b)
    return d.hexdigest()


def log(msg, fh=None):
    print(msg, flush=True)
    if fh:
        fh.write(msg + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# cohort assembly
# ---------------------------------------------------------------------------

def _lung1_rows(cfg):
    c = cfg["cohorts"]["lung1"]
    df = pd.read_csv(ppath(cfg, "lung1_cohort"))
    keep = df[(df["eligible"]) & (~df["excluded_from_gtv_rim"])].copy()
    keep = keep.sort_values("PatientID").reset_index(drop=True)
    keep["label"] = keep["Histology"].map(c["histology_map"]).astype(int)
    _assert_counts("LUNG1", keep, c)
    bad = set(c["exclusions"]) & set(keep["PatientID"])
    if bad:
        raise RuntimeError("an excluded LUNG1 patient is present: %s" % sorted(bad))
    return pd.DataFrame({"Cohort": "LUNG1", "PatientID": keep["PatientID"].astype(str),
                         "Histology": keep["Histology"].astype(str),
                         "label": keep["label"].astype(int)})


def _radiogenomics_rows(cfg):
    c = cfg["cohorts"]["radiogenomics"]
    df = pd.read_csv(ppath(cfg, "external_cohort"))
    inc = df[df["included"].astype(str).str.lower() == "true"].copy()
    inc = inc.sort_values("PatientID").reset_index(drop=True)
    inc["label"] = inc["Histology"].map(c["histology_map"]).astype(int)
    _assert_counts("RADIOGENOMICS", inc, c)
    return pd.DataFrame({"Cohort": "RADIOGENOMICS", "PatientID": inc["PatientID"].astype(str),
                         "Histology": inc["Histology"].astype(str),
                         "label": inc["label"].astype(int)})


def _assert_counts(tag, df, c):
    n, adc, scc = len(df), int((df.label == 1).sum()), int((df.label == 0).sum())
    want = (c["expected_patients"], c["expected_adc"], c["expected_scc"])
    if (n, adc, scc) != want:
        raise RuntimeError("%s cohort is %d/%d ADC/%d SCC, expected %d/%d/%d"
                           % (tag, n, adc, scc, want[0], want[1], want[2]))


def build_cohort(cfg):
    """The 343-patient multicenter cohort, sorted by PatientKey."""
    a, b = _lung1_rows(cfg), _radiogenomics_rows(cfg)
    raw_overlap = sorted(set(a.PatientID) & set(b.PatientID))
    if cfg["cohorts"]["duplicate_screen"]["require_zero_raw_id_collision"] and raw_overlap:
        raise RuntimeError("raw PatientID collision across collections: %s" % raw_overlap[:10])

    df = pd.concat([a, b], ignore_index=True)
    prefix = {"LUNG1": cfg["cohorts"]["lung1"]["key_prefix"],
              "RADIOGENOMICS": cfg["cohorts"]["radiogenomics"]["key_prefix"]}
    df["PatientKey"] = [cfg["cohorts"]["patient_key_format"].format(
        prefix=prefix[c], PatientID=p) for c, p in zip(df.Cohort, df.PatientID)]
    df["cohort_histology"] = ["%s_%s" % (c, "ADC" if y == 1 else "SCC")
                              for c, y in zip(df.Cohort, df.label)]
    df = df.sort_values("PatientKey").reset_index(drop=True)
    df = df[["PatientKey", "Cohort", "PatientID", "Histology", "label", "cohort_histology"]]

    comb = cfg["cohorts"]["combined"]
    n, adc, scc = len(df), int((df.label == 1).sum()), int((df.label == 0).sum())
    if (n, adc, scc) != (comb["expected_patients"], comb["expected_adc"], comb["expected_scc"]):
        raise RuntimeError("combined cohort is %d/%d ADC/%d SCC" % (n, adc, scc))
    if df.PatientKey.duplicated().any():
        raise RuntimeError("duplicate PatientKey")
    got = df.cohort_histology.value_counts().to_dict()
    want = cfg["cohorts"]["strata"]["expected_counts"]
    if {k: int(v) for k, v in got.items()} != {k: int(v) for k, v in want.items()}:
        raise RuntimeError("stratum counts are %s, expected %s" % (got, want))
    df.attrs["raw_id_overlap"] = raw_overlap
    return df


# ---------------------------------------------------------------------------
# feature banks - cached, never re-extracted
# ---------------------------------------------------------------------------

def load_region(cfg, region, cohort):
    """The combined 343-row matrix for one region, aligned to `cohort` order.

    Refuses to return anything unless the two cached banks are name-for-name
    and order-for-order identical and carry the frozen extraction signature.
    """
    fk = cfg["features"]
    lp = ppath(cfg, "lung1_%s_csv" % region)
    ep = ppath(cfg, "external_%s_csv" % region)
    a, b = pd.read_csv(lp), pd.read_csv(ep)

    drop = tuple(fk["drop_column_prefixes"])
    names_a = [c for c in a.columns if c not in META_COLUMNS and not c.startswith(drop)]
    names_b = [c for c in b.columns if c not in META_COLUMNS and not c.startswith(drop)]
    if fk["require_identical_names_and_order"] and names_a != names_b:
        raise RuntimeError("%s feature names/order differ between cohorts" % region)
    expected_n = int(fk["expected_n_%s" % region])
    if len(names_a) != expected_n:
        raise RuntimeError("%s bank has %d features, expected %d" % (region, len(names_a), expected_n))
    if region == "rim" and fk["rim_must_have_no_shape"] and any("_shape_" in n for n in names_a):
        raise RuntimeError("rim bank contains shape features")

    sig_a = json.load(open(ppath(cfg, "lung1_feature_names_json"), encoding="utf-8"))
    sig_b = json.load(open(ppath(cfg, "external_feature_names_json"), encoding="utf-8"))
    for tag, s in (("lung1", sig_a), ("external", sig_b)):
        if s.get("extraction_signature") != fk["expected_extraction_signature"]:
            raise RuntimeError("%s carries extraction signature %s, expected %s"
                               % (tag, s.get("extraction_signature"),
                                  fk["expected_extraction_signature"]))
        if s[region] != names_a:
            raise RuntimeError("%s feature-name manifest disagrees with the %s matrix"
                               % (tag, region))

    a = a.copy(); a["Cohort"] = "LUNG1"
    b = b.copy(); b["Cohort"] = "RADIOGENOMICS"
    both = pd.concat([a, b], ignore_index=True)
    prefix = {"LUNG1": cfg["cohorts"]["lung1"]["key_prefix"],
              "RADIOGENOMICS": cfg["cohorts"]["radiogenomics"]["key_prefix"]}
    both["PatientKey"] = [cfg["cohorts"]["patient_key_format"].format(
        prefix=prefix[c], PatientID=p) for c, p in zip(both.Cohort, both.PatientID)]
    both = both.set_index("PatientKey").loc[list(cohort.PatientKey)].reset_index()

    if not np.array_equal(both.label.to_numpy(int), cohort.label.to_numpy(int)):
        raise RuntimeError("%s feature labels disagree with the cohort" % region)
    x = both[names_a].to_numpy(dtype=np.float64)
    n_nonfinite = int((~np.isfinite(x)).sum())
    if n_nonfinite:
        raise RuntimeError("%s matrix carries %d non-finite values" % (region, n_nonfinite))

    return {"region": region, "names": names_a, "x": x,
            "keys": list(both.PatientKey), "y": cohort.label.to_numpy(int),
            "index": {k: i for i, k in enumerate(both.PatientKey)},
            "lung1_sha256": sha256(lp), "external_sha256": sha256(ep),
            "lung1_path": os.path.relpath(lp, cfg["_project_root"]).replace(os.sep, "/"),
            "external_path": os.path.relpath(ep, cfg["_project_root"]).replace(os.sep, "/")}


# ---------------------------------------------------------------------------
# duplicate screen
# ---------------------------------------------------------------------------

def duplicate_screen(cfg, cohort, mats):
    """Exact-duplicate rows plus a cross-cohort nearest-neighbour distance scan."""
    ds = cfg["cohorts"]["duplicate_screen"]
    out = {"unique_patient_keys": int(cohort.PatientKey.nunique()),
           "n_patients": int(len(cohort)),
           "raw_patient_id_collisions": list(cohort.attrs.get("raw_id_overlap", [])),
           "regions": {}}
    is_l1 = (cohort.Cohort == "LUNG1").to_numpy()

    for region, m in mats.items():
        x = m["x"]
        # exact duplicates, byte-level on the rounded row
        seen, dups = {}, []
        for i, row in enumerate(x):
            key = hashlib.sha256(np.ascontiguousarray(row).tobytes()).hexdigest()
            if key in seen:
                dups.append([m["keys"][seen[key]], m["keys"][i]])
            else:
                seen[key] = i
        # standardised distance scan
        mu, sd = x.mean(axis=0), x.std(axis=0, ddof=0)
        sd = np.where(sd > 0, sd, 1.0)
        z = (x - mu) / sd
        d = _pairwise_distance(z)
        np.fill_diagonal(d, np.inf)
        within = np.where(is_l1[:, None] == is_l1[None, :], d, np.inf)
        cross = np.where(is_l1[:, None] != is_l1[None, :], d, np.inf)
        within_nn = within.min(axis=1)
        pct = float(np.percentile(within_nn, float(ds["nn_distance_percentile"])))
        cross_nn = cross.min(axis=1)
        flagged = [[m["keys"][i], m["keys"][int(cross[i].argmin())], float(cross_nn[i])]
                   for i in range(len(cross_nn)) if cross_nn[i] < pct]
        out["regions"][region] = {
            "exact_duplicate_rows": dups,
            "within_cohort_nn_distance_p%g" % ds["nn_distance_percentile"]: pct,
            "within_cohort_nn_distance_min": float(within_nn.min()),
            "cross_cohort_nn_distance_min": float(cross_nn.min()),
            "cross_cohort_pairs_below_threshold": flagged}
        if ds["require_zero_exact_duplicate_rows"] and dups:
            raise RuntimeError("%s: exact duplicate feature rows %s" % (region, dups[:5]))
    out["candidate_duplicate_pairs"] = sum(
        len(v["cross_cohort_pairs_below_threshold"]) for v in out["regions"].values())
    out["ct_hash_comparison_is_not_evidence"] = (
        "the two collections were converted from DICOM by different code paths, so a "
        "genuine duplicate would still hash differently; provenance rests on disjoint "
        "collections, disjoint identifiers and this feature-space screen")
    return out


def _pairwise_distance(z):
    g = z @ z.T
    sq = np.diag(g)
    d2 = sq[:, None] + sq[None, :] - 2.0 * g
    return np.sqrt(np.maximum(d2, 0.0))


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------

def build_outer_split(cfg, cohort):
    v = cfg["validation"]["outer"]
    skf = StratifiedKFold(n_splits=int(v["n_splits"]), shuffle=bool(v["shuffle"]),
                          random_state=int(v["random_state"]))
    strat = cohort[v["stratify_by"]].to_numpy()
    rows, folds = [], {}
    for k, (tr, te) in enumerate(skf.split(np.arange(len(cohort)), strat), start=1):
        folds[k] = {"train": [cohort.PatientKey[i] for i in tr],
                    "test": [cohort.PatientKey[i] for i in te]}
        for i in tr:
            rows.append({"PatientKey": cohort.PatientKey[i], "fold": k, "split": "train"})
        for i in te:
            rows.append({"PatientKey": cohort.PatientKey[i], "fold": k, "split": "test"})
    df = pd.DataFrame(rows).merge(
        cohort[["PatientKey", "Cohort", "PatientID", "label", "cohort_histology"]],
        on="PatientKey", how="left")
    df = df.sort_values(["fold", "split", "PatientKey"]).reset_index(drop=True)

    levels = list(cfg["cohorts"]["strata"]["levels"])
    per_fold = {}
    for k, d in folds.items():
        sub = cohort[cohort.PatientKey.isin(d["test"])]
        counts = {lv: int((sub.cohort_histology == lv).sum()) for lv in levels}
        per_fold[k] = {"n_test": len(d["test"]), "n_train": len(d["train"]),
                       "test_strata": counts,
                       "test_lung1": int((sub.Cohort == "LUNG1").sum()),
                       "test_radiogenomics": int((sub.Cohort == "RADIOGENOMICS").sum()),
                       "test_adc": int((sub.label == 1).sum()),
                       "test_scc": int((sub.label == 0).sum())}
        if v["require_all_strata_in_every_fold"] and any(c == 0 for c in counts.values()):
            raise RuntimeError("outer fold %d is missing a stratum: %s" % (k, counts))
        if set(d["train"]) & set(d["test"]) or \
           set(d["train"]) | set(d["test"]) != set(cohort.PatientKey):
            raise RuntimeError("outer fold %d is not a clean partition" % k)
    return df, folds, per_fold


def inner_split(cfg, keys, strata, seed):
    """Stratified 3-fold over `strata` (cohort_histology pooled, label in transfer)."""
    n = int(cfg["validation"]["inner"]["n_splits"])
    skf = StratifiedKFold(n_splits=n, shuffle=bool(cfg["validation"]["inner"]["shuffle"]),
                          random_state=int(seed))
    return [(tr, va) for tr, va in skf.split(np.arange(len(keys)), np.asarray(strata))]


# ---------------------------------------------------------------------------
# source-balanced weights
# ---------------------------------------------------------------------------

def source_weights(cfg, strata):
    """w_i = N / (S_present * n_s) - equal TOTAL weight per stratum, mean 1.0.

    `strata` are the strata of the TRAINING partition only.  Nothing about a
    held-out patient can reach this function: it receives training strata and
    nothing else.
    """
    s = np.asarray(strata)
    levels, counts = np.unique(s, return_counts=True)
    n, n_levels = float(len(s)), float(len(levels))
    per = {lv: n / (n_levels * c) for lv, c in zip(levels, counts)}
    w = np.array([per[v] for v in s], dtype=np.float64)
    rec = {"n_train": int(len(s)), "n_strata_present": int(len(levels)),
           "counts": {str(lv): int(c) for lv, c in zip(levels, counts)},
           "weight_per_patient": {str(lv): float(per[lv]) for lv in levels},
           "total_weight_per_stratum": {str(lv): float(per[lv] * c)
                                        for lv, c in zip(levels, counts)},
           "mean_weight": float(w.mean())}
    tot = list(rec["total_weight_per_stratum"].values())
    tol = float(cfg["source_weights"]["equal_total_weight_tolerance"])
    rec["equal_total_weight"] = bool(max(tot) - min(tot) <= tol * max(1.0, max(tot)))
    rec["mean_weight_is_one"] = bool(
        abs(rec["mean_weight"] - 1.0) <= float(cfg["source_weights"]["mean_weight_tolerance"]))
    if not (rec["equal_total_weight"] and rec["mean_weight_is_one"]):
        raise RuntimeError("source-balanced weights failed their own invariants: %s" % rec)
    return w, rec


# ---------------------------------------------------------------------------
# freeze / verify
# ---------------------------------------------------------------------------

def protocol_sha(cfg):
    return sha256(os.path.join(cfg["_project_root"], *cfg["experiment"]["protocol_doc"].split("/")))


def verify_frozen(cfg, require_split=True):
    """Every driver calls this first and refuses to run if anything has moved."""
    run_path = ppath(cfg, "run_json")
    if not os.path.isfile(run_path):
        raise RuntimeError("reports/phase8b_run.json is missing - run "
                           "src/phase8b_data.py --freeze BEFORE any modelling")
    rec = json.load(open(run_path, encoding="utf-8"))
    now = {"protocol_sha256": protocol_sha(cfg), "config_sha256": sha256(cfg["_config_path"])}
    for k, v in now.items():
        if rec["frozen_before_modelling"][k] != v:
            raise RuntimeError("%s changed after the freeze (%s -> %s)"
                               % (k, rec["frozen_before_modelling"][k][:16], v[:16]))
    if require_split:
        sp = sha256(ppath(cfg, "split_csv"))
        if rec["frozen_before_modelling"]["split_csv_sha256"] != sp:
            raise RuntimeError("the frozen multicenter split has changed")
    return rec


def freeze(cfg):
    t0 = time.time()
    os.makedirs(os.path.dirname(ppath(cfg, "training_log")), exist_ok=True)
    fh = open(os.path.join(cfg["_project_root"], "logs", "phase8b_freeze.log"),
              "w", encoding="utf-8")

    log("Phase 8B freeze - protocol %s" % cfg["experiment"]["protocol_doc"], fh)
    cohort = build_cohort(cfg)
    for key in ("cohort_csv", "split_csv", "cross_cohort_json"):
        os.makedirs(os.path.dirname(ppath(cfg, key)), exist_ok=True)
    os.makedirs(ppath(cfg, "inner_cv_dir"), exist_ok=True)
    cohort.to_csv(ppath(cfg, "cohort_csv"), index=False)
    log("cohort: %d patients (%d ADC / %d SCC); strata %s"
        % (len(cohort), int((cohort.label == 1).sum()), int((cohort.label == 0).sum()),
           cohort.cohort_histology.value_counts().to_dict()), fh)

    mats = {r: load_region(cfg, r, cohort) for r in cfg["features"]["regions"]}
    for r, m in mats.items():
        log("  %s bank: %d x %d  (lung1 %s / external %s)"
            % (r, m["x"].shape[0], m["x"].shape[1],
               m["lung1_sha256"][:12], m["external_sha256"][:12]), fh)

    dup = duplicate_screen(cfg, cohort, mats)
    log("duplicate screen: %d unique keys, %d raw-id collisions, %d candidate duplicate pairs"
        % (dup["unique_patient_keys"], len(dup["raw_patient_id_collisions"]),
           dup["candidate_duplicate_pairs"]), fh)

    split_df, folds, per_fold = build_outer_split(cfg, cohort)
    split_df.to_csv(ppath(cfg, "split_csv"), index=False)
    split_sha = sha256(ppath(cfg, "split_csv"))

    # inner assignment + the weights each training partition will use
    strat_map = dict(zip(cohort.PatientKey, cohort.cohort_histology))
    inner_records, weight_records = {}, {}
    for k in sorted(folds):
        tr = folds[k]["train"]
        inner = inner_split(cfg, tr, [strat_map[p] for p in tr],
                            int(cfg["validation"]["inner"]["base_seed"]) + k)
        assign = {p: -1 for p in tr}
        detail = []
        for j, (a, b) in enumerate(inner):
            for i in b:
                assign[tr[i]] = j
            va = [strat_map[tr[i]] for i in b]
            tra = [strat_map[tr[i]] for i in a]
            detail.append({"inner_fold": j, "n_train": len(a), "n_val": len(b),
                           "train_strata": {lv: int(tra.count(lv))
                                            for lv in cfg["cohorts"]["strata"]["levels"]},
                           "val_strata": {lv: int(va.count(lv))
                                          for lv in cfg["cohorts"]["strata"]["levels"]}})
            _, wr = source_weights(cfg, tra)
            detail[-1]["inner_train_weights"] = wr
        pd.DataFrame([{"PatientKey": p, "outer_fold": k,
                       "split": "train" if p in assign else "test",
                       "inner_fold": assign.get(p, -1),
                       "cohort_histology": strat_map[p]}
                      for p in tr + folds[k]["test"]]).to_csv(
            os.path.join(ppath(cfg, "inner_cv_dir"), "fold_%d.csv" % k), index=False)
        inner_records[k] = {"inner_seed": int(cfg["validation"]["inner"]["base_seed"]) + k,
                            "folds": detail}
        _, wr = source_weights(cfg, [strat_map[p] for p in tr])
        weight_records[k] = wr
        log("  fold %d: %d train / %d test, inner seed %d, outer-train weights %s"
            % (k, len(tr), len(folds[k]["test"]), inner_records[k]["inner_seed"],
               {a: round(b, 4) for a, b in wr["weight_per_patient"].items()}), fh)

    meta = {"phase": "8B", "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generator": "src/phase8b_data.py",
            "rule": "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) on cohort_histology",
            "stratify_by": cfg["validation"]["outer"]["stratify_by"],
            "n_splits": int(cfg["validation"]["outer"]["n_splits"]),
            "random_state": int(cfg["validation"]["outer"]["random_state"]),
            "sort_by": cfg["validation"]["outer"]["sort_by"],
            "split_unit": "patient (PatientKey)",
            "n_patients": int(len(cohort)),
            "strata_counts": {k: int(v) for k, v in
                              cohort.cohort_histology.value_counts().to_dict().items()},
            "per_fold": per_fold,
            "split_csv_sha256": split_sha,
            "does_not_reuse_the_lung1_202_split": True,
            "reason": "cohort membership changed and stratification is now four-level"}
    with open(ppath(cfg, "split_meta_json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    cc = {"phase": "8B", "directions": []}
    for d in cfg["validation"]["transfer"]["directions"]:
        tr_keys = sorted(cohort.PatientKey[cohort.Cohort == d["train"]])
        te_keys = sorted(cohort.PatientKey[cohort.Cohort == d["test"]])
        cc["directions"].append({
            "id": d["id"], "train_cohort": d["train"], "test_cohort": d["test"],
            "inner_seed": int(d["inner_seed"]),
            "n_train": len(tr_keys), "n_test": len(te_keys),
            "train_adc": int(cohort.label[cohort.Cohort == d["train"]].sum()),
            "test_adc": int(cohort.label[cohort.Cohort == d["test"]].sum()),
            "rule": ("every choice - selector, hyperparameters, preprocessing, stacker, "
                     "calibration, threshold - is made inside the training cohort; the "
                     "destination cohort's labels are touched exactly once, at evaluation"),
            "train_keys": tr_keys, "test_keys": te_keys})
    with open(ppath(cfg, "cross_cohort_json"), "w", encoding="utf-8") as f:
        json.dump(cc, f, indent=1)

    run = {
        "phase": "8B",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frozen_before_modelling": {
            "protocol_doc": cfg["experiment"]["protocol_doc"],
            "protocol_sha256": protocol_sha(cfg),
            "config": os.path.relpath(cfg["_config_path"], cfg["_project_root"]).replace(os.sep, "/"),
            "config_sha256": sha256(cfg["_config_path"]),
            "cohort_csv_sha256": sha256(ppath(cfg, "cohort_csv")),
            "split_csv_sha256": split_sha,
            "split_meta_sha256": sha256(ppath(cfg, "split_meta_json")),
            "cross_cohort_sha256": sha256(ppath(cfg, "cross_cohort_json")),
            "primary_probability": cfg["evaluation"]["primary_probability"],
            "primary_metric_note": ("PROTOCOL 9.1: Phase-8B primary discrimination is the "
                                    "RAW out-of-fold probability ROC-AUC. This differs "
                                    "deliberately from Phase 7A, whose report is NOT altered."),
            "gate_8b": cfg["gate_8b"],
            "external_status": cfg["experiment"]["external_status"]},
        "cohort": {
            "n": int(len(cohort)),
            "adc": int((cohort.label == 1).sum()), "scc": int((cohort.label == 0).sum()),
            "lung1": int((cohort.Cohort == "LUNG1").sum()),
            "radiogenomics": int((cohort.Cohort == "RADIOGENOMICS").sum()),
            "strata": {k: int(v) for k, v in
                       cohort.cohort_histology.value_counts().to_dict().items()}},
        "features": {r: {"n_features": len(m["names"]),
                         "lung1_csv": m["lung1_path"], "lung1_sha256": m["lung1_sha256"],
                         "external_csv": m["external_path"], "external_sha256": m["external_sha256"],
                         "names_identical_and_ordered": True,
                         "extraction_signature": cfg["features"]["expected_extraction_signature"]}
                     for r, m in mats.items()},
        "pyradiomics_rerun": False,
        "duplicate_screen": dup,
        "outer_split": meta,
        "inner_cv": inner_records,
        "outer_training_source_weights": weight_records,
        "cross_cohort": {d["id"]: {k: v for k, v in d.items()
                                   if k not in ("train_keys", "test_keys")}
                         for d in cc["directions"]},
        "environment": environment(),
        "freeze_seconds": round(time.time() - t0, 1)}
    with open(ppath(cfg, "run_json"), "w", encoding="utf-8") as f:
        json.dump(run, f, indent=1, default=float)
    log("FROZEN: protocol %s | split %s | run record -> %s (%.1fs)"
        % (run["frozen_before_modelling"]["protocol_sha256"][:16], split_sha[:16],
           ppath(cfg, "run_json"), run["freeze_seconds"]), fh)
    fh.close()
    return run


def environment():
    import sklearn
    import scipy
    return {"python": platform.python_version(), "platform": platform.platform(),
            "numpy": np.__version__, "pandas": pd.__version__,
            "sklearn": sklearn.__version__, "scipy": scipy.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/phase8b_multicenter.yaml")
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.freeze:
        freeze(cfg)
    else:
        rec = verify_frozen(cfg)
        print("Phase 8B is frozen: protocol %s, split %s, %d patients"
              % (rec["frozen_before_modelling"]["protocol_sha256"][:16],
                 rec["frozen_before_modelling"]["split_csv_sha256"][:16],
                 rec["cohort"]["n"]))


if __name__ == "__main__":
    main()
