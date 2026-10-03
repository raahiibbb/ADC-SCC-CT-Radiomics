"""Phase 8B - the modelling driver: pooled nested CV, cross-cohort transport,
and the two label-free / post-hoc diagnostics.

    VIEW 1  pooled source-stratified 5-fold nested CV over all 343 patients
    VIEW 3  LUNG1 -> Radiogenomics and Radiogenomics -> LUNG1, each scored ONCE
    diag    a label-free cohort-identity classifier (GTV and rim)

VIEW 2 needs no training at all: it is the same pooled OOF predictions subset by
cohort, and is computed in src/phase8b_evaluate.py.

Everything is fitted on training patients only, at whatever level it is used:
imputation, the variance screen, scaling, feature selection, the selector-family
choice, selector and classifier hyperparameters, the four-stratum weights, the
fusion stacker, Platt calibration and the operating threshold.  No outer-test
patient and no destination-cohort patient influences any choice.

The run is RESUMABLE.  Each (view, region-or-model, fold) unit is checkpointed
to results/phase8b/checkpoints/ and a valid checkpoint carrying the current
protocol hash is skipped.  Re-running the same command resumes.

    ./.venv/Scripts/python.exe src/phase8b_run.py --config config/phase8b_multicenter.yaml --all
    ./.venv/Scripts/python.exe src/phase8b_run.py --config config/phase8b_multicenter.yaml --all --resume
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from phase8b_data import (build_cohort, inner_split, load_cfg, load_region,   # noqa: E402
                          log, ppath, protocol_sha, sha256, source_weights,
                          verify_frozen)
from phase8b_models import (auc, calibrate_and_threshold, candidates,         # noqa: E402
                            choose_candidate, fit_candidate, fit_stacker,
                            inner_surface, make_preprocessor)
from phase7a_models import fit_logistic                                       # noqa: E402

MODELS = ("A1", "A2", "A3")
REGION_OF = {"A1": "gtv", "A2": "rim"}
PRED_FILE = {"pooled": {"A1": "pooled_A1_oof.csv", "A2": "pooled_A2_oof.csv",
                        "A3": "pooled_A3_oof.csv"},
             "lung1_to_radiogenomics": {"A1": "lung1_to_radiogenomics_A1.csv",
                                        "A2": "lung1_to_radiogenomics_A2.csv",
                                        "A3": "lung1_to_radiogenomics_A3.csv"},
             "radiogenomics_to_lung1": {"A1": "radiogenomics_to_lung1_A1.csv",
                                        "A2": "radiogenomics_to_lung1_A2.csv",
                                        "A3": "radiogenomics_to_lung1_A3.csv"}}


# ---------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------

def ck_path(cfg, unit):
    return os.path.join(ppath(cfg, "checkpoints_dir"), "%s.json" % unit)


def ck_load(cfg, unit, psha):
    p = ck_path(cfg, unit)
    if not os.path.isfile(p):
        return None
    try:
        d = json.load(open(p, encoding="utf-8"))
    except (ValueError, OSError):
        return None
    if d.get("protocol_sha256") != psha or not d.get("complete"):
        return None
    return d


def ck_save(cfg, unit, psha, payload):
    payload = dict(payload)
    payload["protocol_sha256"] = psha
    payload["complete"] = True
    payload["written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    os.makedirs(ppath(cfg, "checkpoints_dir"), exist_ok=True)
    with open(ck_path(cfg, unit), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=_jsonable)


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return float(o)


def progress_write(cfg, done, total, current, t0):
    os.makedirs(ppath(cfg, "results_dir"), exist_ok=True)
    with open(ppath(cfg, "progress_json"), "w", encoding="utf-8") as fh:
        json.dump({"phase": "8B", "units_done": done, "units_total": total,
                   "current_unit": current,
                   "elapsed_seconds": round(time.time() - t0, 1),
                   "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}, fh, indent=1)


# ---------------------------------------------------------------------------
# VIEW 1 - pooled source-stratified nested CV
# ---------------------------------------------------------------------------

def pooled_fold(cfg, region, mat, cohort, tr_keys, te_keys, inner, strat, fh, tag, fold):
    """One outer fold of A1 (gtv) or A2 (rim) under source-balanced weights."""
    t0 = time.time()
    idx = mat["index"]
    tr = np.asarray([idx[k] for k in tr_keys])
    te = np.asarray([idx[k] for k in te_keys])
    x, y = mat["x"], mat["y"]
    y_tr = y[tr]
    strat_tr = np.asarray([strat[k] for k in tr_keys])

    def weights_for(sub_idx):
        w, _ = source_weights(cfg, strat_tr[sub_idx])
        return w

    surface, oof_cache = inner_surface(cfg, x[tr], y_tr, weights_for, inner)
    chosen, tie = choose_candidate(cfg, surface)
    inner_oof = oof_cache[chosen["candidate_id"]]

    cand = next(c for c in candidates(cfg) if c["id"] == chosen["candidate_id"])
    w_all, w_rec = source_weights(cfg, strat_tr)
    pre, est = fit_candidate(cfg, x[tr], y_tr, w_all, cand)
    p_test = est.predict_proba(pre.transform(x[te]))

    kept = pre.kept_indices
    names = [mat["names"][int(kept[j])] for j in est.selected]
    log("  %s fold %d: %-26s inner AUC %.4f  kept %d/%d dims  %d feats  (%.0fs)"
        % (tag, fold, chosen["candidate_id"], chosen["mean_inner_roc_auc"],
           int(pre.keep.sum()), pre.n_input, len(names), time.time() - t0), fh)
    return {"surface": surface, "chosen": chosen, "tie_break": tie,
            "inner_oof": inner_oof, "p_test": p_test, "pre": pre, "est": est,
            "selected_feature_names": names,
            "selected_feature_indices": [int(kept[j]) for j in est.selected],
            "coefficients": [float(v) for v in est.coefficients],
            "intercept": float(est.intercept),
            "n_selected": len(names),
            "n_kept_dimensions": int(pre.keep.sum()),
            "n_input_dimensions": int(pre.n_input),
            "source_weights": w_rec,
            "seconds": round(time.time() - t0, 1)}


def run_pooled(cfg, cohort, mats, psha, fh, counter):
    split = pd.read_csv(ppath(cfg, "split_csv"))
    strat = dict(zip(cohort.PatientKey, cohort.cohort_histology))
    y_map = dict(zip(cohort.PatientKey, cohort.label))
    coh_map = dict(zip(cohort.PatientKey, cohort.Cohort))
    hist_map = dict(zip(cohort.PatientKey, cohort.Histology))
    pid_map = dict(zip(cohort.PatientKey, cohort.PatientID))

    rows = {m: [] for m in MODELS}
    fold_records = {}

    for k in sorted(split.fold.unique()):
        f = split[split.fold == k]
        tr_keys = sorted(f[f.split == "train"].PatientKey)
        te_keys = sorted(f[f.split == "test"].PatientKey)
        unit = "pooled_fold_%d" % k
        cached = ck_load(cfg, unit, psha)
        if cached is not None:
            log("fold %d: cached, skipped" % k, fh)
            fold_records[int(k)] = cached["fold_record"]
            for m in MODELS:
                rows[m].extend(cached["rows"][m])
            counter(unit)
            continue

        y_tr = np.asarray([y_map[p] for p in tr_keys], dtype=int)
        inner = inner_split(cfg, tr_keys, [strat[p] for p in tr_keys],
                            int(cfg["validation"]["inner"]["base_seed"]) + int(k))
        log("fold %d: %d train (%d ADC) / %d test (%d ADC)"
            % (k, len(tr_keys), int(y_tr.sum()), len(te_keys),
               sum(y_map[p] for p in te_keys)), fh)

        res = {"A1": pooled_fold(cfg, "gtv", mats["gtv"], cohort, tr_keys, te_keys,
                                 inner, strat, fh, "A1", int(k)),
               "A2": pooled_fold(cfg, "rim", mats["rim"], cohort, tr_keys, te_keys,
                                 inner, strat, fh, "A2", int(k))}

        # ---- A3: two inputs, inner-OOF only ---------------------------------
        z_tr = np.column_stack([res["A1"]["inner_oof"], res["A2"]["inner_oof"]])
        if not np.all(np.isfinite(z_tr)):
            raise RuntimeError("fold %d: a component inner-OOF prediction is missing" % k)
        strat_tr = np.asarray([strat[p] for p in tr_keys])
        a3_oof = np.full(len(y_tr), np.nan)
        for a, b in inner:
            w, _ = source_weights(cfg, strat_tr[a])
            st = fit_stacker(cfg, z_tr[a], y_tr[a], w)
            a3_oof[b] = st.predict_proba(z_tr[b])[:, 1]
        w_all, _ = source_weights(cfg, strat_tr)
        stacker = fit_stacker(cfg, z_tr, y_tr, w_all)
        z_te = np.column_stack([res["A1"]["p_test"], res["A2"]["p_test"]])
        res["A3"] = {"inner_oof": a3_oof, "p_test": stacker.predict_proba(z_te)[:, 1],
                     "coefficients": [float(v) for v in stacker.coef_[0]],
                     "intercept": float(stacker.intercept_[0]),
                     "fusion_inputs": ["P_ADC_GTV", "P_ADC_RIM"],
                     "n_fusion_training_rows": int(len(y_tr)),
                     "z_train": z_tr, "z_test": z_te}
        log("  A3 fold %d: stacker coef [%.4f, %.4f] intercept %.4f (%d inner-OOF rows)"
            % (k, stacker.coef_[0][0], stacker.coef_[0][1],
               stacker.intercept_[0], len(y_tr)), fh)

        fold_rec = {"outer_fold": int(k),
                    "inner_seed": int(cfg["validation"]["inner"]["base_seed"]) + int(k),
                    "n_train": len(tr_keys), "n_test": len(te_keys),
                    "train_adc": int(y_tr.sum()),
                    "test_adc": int(sum(y_map[p] for p in te_keys)),
                    "test_lung1": int(sum(coh_map[p] == "LUNG1" for p in te_keys)),
                    "test_radiogenomics": int(sum(coh_map[p] == "RADIOGENOMICS" for p in te_keys))}
        fold_rows = {m: [] for m in MODELS}

        for m in MODELS:
            r = res[m]
            ct = calibrate_and_threshold(cfg, r["inner_oof"], y_tr, r["p_test"])
            for p, praw, pcal in zip(te_keys, r["p_test"], ct["calibrated_test"]):
                fold_rows[m].append({
                    "PatientKey": p, "Cohort": coh_map[p], "PatientID": pid_map[p],
                    "Histology": hist_map[p], "label": int(y_map[p]), "fold": int(k),
                    "p_adc_raw": float(praw), "p_adc_calibrated": float(pcal),
                    "threshold": ct["threshold"],
                    "pred_at_threshold": int(pcal >= ct["threshold"]),
                    "pred_at_0.5": int(pcal >= 0.5)})
            fold_rec[m] = {
                "calibration": {"a": ct["platt_a"], "b": ct["platt_b"],
                                "n_inner_oof": ct["n_calibration_rows"],
                                "valid": ct["platt_valid"]},
                "threshold": ct["threshold"],
                "inner_balanced_accuracy": ct["inner_balanced_accuracy"],
                "inner_oof_roc_auc_raw": auc(y_tr, r["inner_oof"]),
                "inner_oof_roc_auc_calibrated": auc(y_tr, ct["calibrated_inner_oof"])}
            if m in ("A1", "A2"):
                fold_rec[m].update({
                    "selector_family": r["chosen"]["family"],
                    "candidate_id": r["chosen"]["candidate_id"],
                    "l1_ratio": r["chosen"]["l1_ratio"], "C": r["chosen"]["C"],
                    "k": r["chosen"]["k"],
                    "mean_inner_roc_auc": r["chosen"]["mean_inner_roc_auc"],
                    "tie_break": r["tie_break"],
                    "n_selected_features": r["n_selected"],
                    "selected_features": r["selected_feature_names"],
                    "coefficients": r["coefficients"], "intercept": r["intercept"],
                    "n_kept_dimensions": r["n_kept_dimensions"],
                    "n_input_dimensions": r["n_input_dimensions"],
                    "source_weights": r["source_weights"],
                    "seconds": r["seconds"]})
                np.savez_compressed(
                    os.path.join(ppath(cfg, "preprocessing_dir"),
                                 "pooled_%s_fold_%d.npz" % (m, k)), **r["pre"].arrays())
                pd.DataFrame(r["surface"]).to_csv(
                    os.path.join(ppath(cfg, "results_dir"), "hyperparameter_surfaces",
                                 "pooled_%s_fold_%d.csv" % (m, k)), index=False)
            else:
                fold_rec[m].update({
                    "coefficients": r["coefficients"], "intercept": r["intercept"],
                    "fusion_inputs": r["fusion_inputs"],
                    "n_fusion_training_rows": r["n_fusion_training_rows"]})
                pd.DataFrame({"PatientKey": tr_keys, "label": y_tr,
                              "cohort_histology": [strat[p] for p in tr_keys],
                              "P_ADC_GTV": r["z_train"][:, 0],
                              "P_ADC_RIM": r["z_train"][:, 1],
                              "source": "inner_oof"}).to_csv(
                    os.path.join(ppath(cfg, "models_dir"), "A3_fusion",
                                 "pooled_fold_%d_training_inputs.csv" % k), index=False)
            with open(os.path.join(ppath(cfg, "models_dir"), _mdir(m),
                                   "pooled_fold_%d.json" % k), "w", encoding="utf-8") as f:
                json.dump(fold_rec[m], f, indent=1, default=_jsonable)
            rows[m].extend(fold_rows[m])

        fold_records[int(k)] = fold_rec
        ck_save(cfg, unit, psha, {"fold_record": fold_rec, "rows": fold_rows})
        counter(unit)

    for m in MODELS:
        df = pd.DataFrame(rows[m]).sort_values("PatientKey").reset_index(drop=True)
        n = int(cfg["cohorts"]["combined"]["expected_patients"])
        if len(df) != n or df.PatientKey.duplicated().any():
            raise RuntimeError("pooled %s produced %d rows (%d duplicated)"
                               % (m, len(df), int(df.PatientKey.duplicated().sum())))
        df.to_csv(os.path.join(ppath(cfg, "predictions_dir"), PRED_FILE["pooled"][m]),
                  index=False)
    return fold_records


def _mdir(m):
    return {"A1": "A1_gtv", "A2": "A2_rim", "A3": "A3_fusion"}[m]


# ---------------------------------------------------------------------------
# VIEW 3 - true cross-cohort transport, each direction scored ONCE
# ---------------------------------------------------------------------------

def run_transfer(cfg, cohort, mats, psha, fh, counter):
    tcfg = cfg["validation"]["transfer"]
    cw = tcfg["class_balancing"]
    out = {}
    for d in tcfg["directions"]:
        unit = "transfer_%s" % d["id"]
        cached = ck_load(cfg, unit, psha)
        if cached is not None:
            log("transfer %s: cached, skipped" % d["id"], fh)
            out[d["id"]] = cached["record"]
            for m in MODELS:
                pd.DataFrame(cached["rows"][m]).to_csv(
                    os.path.join(ppath(cfg, "predictions_dir"), PRED_FILE[d["id"]][m]),
                    index=False)
            counter(unit)
            continue

        src = cohort[cohort.Cohort == d["train"]].reset_index(drop=True)
        dst = cohort[cohort.Cohort == d["test"]].reset_index(drop=True)
        src_keys, dst_keys = list(src.PatientKey), list(dst.PatientKey)
        y_src = src.label.to_numpy(int)
        y_dst = dst.label.to_numpy(int)
        log("transfer %s: train %d (%d ADC) -> test %d (%d ADC), inner seed %d"
            % (d["id"], len(src_keys), int(y_src.sum()), len(dst_keys),
               int(y_dst.sum()), d["inner_seed"]), fh)

        skf = StratifiedKFold(n_splits=int(tcfg["inner_n_splits"]), shuffle=True,
                              random_state=int(d["inner_seed"]))
        inner = [(a, b) for a, b in skf.split(np.arange(len(src_keys)), y_src)]

        rec, comp = {"direction": d["id"], "train_cohort": d["train"],
                     "test_cohort": d["test"], "inner_seed": int(d["inner_seed"]),
                     "n_train": len(src_keys), "n_test": len(dst_keys),
                     "class_balancing": cw,
                     "destination_labels_used": "once, at final evaluation only"}, {}
        rowsd = {m: [] for m in MODELS}

        for m in ("A1", "A2"):
            region = REGION_OF[m]
            mat = mats[region]
            si = np.asarray([mat["index"][k] for k in src_keys])
            di = np.asarray([mat["index"][k] for k in dst_keys])
            t0 = time.time()
            surface, oof_cache = inner_surface(cfg, mat["x"][si], y_src, None, inner,
                                               class_weight=cw)
            chosen, tie = choose_candidate(cfg, surface)
            cand = next(c for c in candidates(cfg) if c["id"] == chosen["candidate_id"])
            inner_oof = oof_cache[chosen["candidate_id"]]
            pre, est = fit_candidate(cfg, mat["x"][si], y_src, None, cand, class_weight=cw)
            p_dst = est.predict_proba(pre.transform(mat["x"][di]))
            kept = pre.kept_indices
            names = [mat["names"][int(kept[j])] for j in est.selected]
            comp[m] = {"inner_oof": inner_oof, "p_test": p_dst}
            rec[m] = {"selector_family": chosen["family"],
                      "candidate_id": chosen["candidate_id"],
                      "l1_ratio": chosen["l1_ratio"], "C": chosen["C"], "k": chosen["k"],
                      "mean_inner_roc_auc": chosen["mean_inner_roc_auc"],
                      "tie_break": tie, "n_selected_features": len(names),
                      "selected_features": names,
                      "coefficients": [float(v) for v in est.coefficients],
                      "intercept": float(est.intercept),
                      "n_kept_dimensions": int(pre.keep.sum()),
                      "n_input_dimensions": int(pre.n_input),
                      "seconds": round(time.time() - t0, 1)}
            np.savez_compressed(
                os.path.join(ppath(cfg, "preprocessing_dir"), "%s_%s.npz" % (d["id"], m)),
                **pre.arrays())
            pd.DataFrame(surface).to_csv(
                os.path.join(ppath(cfg, "results_dir"), "hyperparameter_surfaces",
                             "%s_%s.csv" % (d["id"], m)), index=False)
            log("  %s %s: %-26s inner AUC %.4f  %d feats  (%.0fs)"
                % (d["id"], m, chosen["candidate_id"], chosen["mean_inner_roc_auc"],
                   len(names), rec[m]["seconds"]), fh)

        # ---- A3: stacker fitted on TRAINING-COHORT inner-OOF only -----------
        z_src = np.column_stack([comp["A1"]["inner_oof"], comp["A2"]["inner_oof"]])
        if not np.all(np.isfinite(z_src)):
            raise RuntimeError("%s: a component inner-OOF prediction is missing" % d["id"])
        a3_oof = np.full(len(y_src), np.nan)
        for a, b in inner:
            st = fit_stacker(cfg, z_src[a], y_src[a], None, class_weight=cw)
            a3_oof[b] = st.predict_proba(z_src[b])[:, 1]
        stacker = fit_stacker(cfg, z_src, y_src, None, class_weight=cw)
        z_dst = np.column_stack([comp["A1"]["p_test"], comp["A2"]["p_test"]])
        comp["A3"] = {"inner_oof": a3_oof, "p_test": stacker.predict_proba(z_dst)[:, 1]}
        rec["A3"] = {"coefficients": [float(v) for v in stacker.coef_[0]],
                     "intercept": float(stacker.intercept_[0]),
                     "fusion_inputs": ["P_ADC_GTV", "P_ADC_RIM"],
                     "n_fusion_training_rows": int(len(y_src)),
                     "stacker_training_source": "%s internal inner-OOF only" % d["train"]}
        pd.DataFrame({"PatientKey": src_keys, "label": y_src,
                      "P_ADC_GTV": z_src[:, 0], "P_ADC_RIM": z_src[:, 1],
                      "source": "inner_oof"}).to_csv(
            os.path.join(ppath(cfg, "models_dir"), "A3_fusion",
                         "%s_training_inputs.csv" % d["id"]), index=False)

        for m in MODELS:
            ct = calibrate_and_threshold(cfg, comp[m]["inner_oof"], y_src, comp[m]["p_test"])
            rec[m] = dict(rec.get(m, {}))
            rec[m].update({"calibration": {"a": ct["platt_a"], "b": ct["platt_b"],
                                           "n_inner_oof": ct["n_calibration_rows"],
                                           "valid": ct["platt_valid"]},
                           "threshold": ct["threshold"],
                           "inner_balanced_accuracy": ct["inner_balanced_accuracy"],
                           "inner_oof_roc_auc_raw": auc(y_src, comp[m]["inner_oof"])})
            for key, praw, pcal, yy, ph, pid in zip(
                    dst_keys, comp[m]["p_test"], ct["calibrated_test"], y_dst,
                    list(dst.Histology), list(dst.PatientID)):
                rowsd[m].append({"PatientKey": key, "Cohort": d["test"], "PatientID": pid,
                                 "Histology": ph, "label": int(yy),
                                 "p_adc_raw": float(praw), "p_adc_calibrated": float(pcal),
                                 "threshold": ct["threshold"],
                                 "pred_at_threshold": int(pcal >= ct["threshold"]),
                                 "pred_at_0.5": int(pcal >= 0.5)})
            pd.DataFrame(rowsd[m]).sort_values("PatientKey").to_csv(
                os.path.join(ppath(cfg, "predictions_dir"), PRED_FILE[d["id"]][m]),
                index=False)
            with open(os.path.join(ppath(cfg, "models_dir"), _mdir(m),
                                   "%s.json" % d["id"]), "w", encoding="utf-8") as f:
                json.dump(rec[m], f, indent=1, default=_jsonable)

        out[d["id"]] = rec
        ck_save(cfg, unit, psha, {"record": rec, "rows": rowsd})
        counter(unit)
    return out


# ---------------------------------------------------------------------------
# label-free domain diagnostic
# ---------------------------------------------------------------------------

def run_domain(cfg, cohort, mats, psha, fh, counter):
    """Cohort identity ONLY.  The histology column is never in the design matrix
    and never in the target vector."""
    unit = "domain_diagnostic"
    cached = ck_load(cfg, unit, psha)
    if cached is not None:
        log("domain diagnostic: cached, skipped", fh)
        counter(unit)
        return cached["record"]

    dcfg = cfg["domain_diagnostic"]
    domain = (cohort.Cohort == "RADIOGENOMICS").astype(int).to_numpy()
    skf = StratifiedKFold(n_splits=int(dcfg["cv"]["n_splits"]),
                          shuffle=bool(dcfg["cv"]["shuffle"]),
                          random_state=int(dcfg["cv"]["random_state"]))
    rec = {"target": "cohort_identity_LUNG1_vs_RADIOGENOMICS",
           "uses_histology": False,
           "note": "diagnostic only - never compared with a disease AUC as if equivalent",
           "n_lung1": int((domain == 0).sum()), "n_radiogenomics": int((domain == 1).sum()),
           "regions": {}}

    for region, mat in mats.items():
        x = mat["x"]
        oof = np.full(len(domain), np.nan)
        chosen_C = []
        for tr, te in skf.split(np.arange(len(domain)), domain):
            best, best_auc = None, -np.inf
            inner = StratifiedKFold(n_splits=3, shuffle=True,
                                    random_state=int(dcfg["random_state"]))
            for C in dcfg["C_grid"]:
                a = []
                for ia, ib in inner.split(np.arange(len(tr)), domain[tr]):
                    pre = make_preprocessor(cfg).fit(x[tr][ia])
                    est = fit_logistic(pre.transform(x[tr][ia]), domain[tr][ia],
                                       dict(penalty=dcfg["penalty"], solver=dcfg["solver"],
                                            C=C, class_weight=dcfg["class_weight"],
                                            max_iter=dcfg["max_iter"], tol=1e-6,
                                            random_state=dcfg["random_state"]), C=C)
                    a.append(auc(domain[tr][ib],
                                 est.predict_proba(pre.transform(x[tr][ib]))[:, 1]))
                if float(np.mean(a)) > best_auc + 1e-12:
                    best_auc, best = float(np.mean(a)), float(C)
            pre = make_preprocessor(cfg).fit(x[tr])
            est = fit_logistic(pre.transform(x[tr]), domain[tr],
                               dict(penalty=dcfg["penalty"], solver=dcfg["solver"],
                                    C=best, class_weight=dcfg["class_weight"],
                                    max_iter=dcfg["max_iter"], tol=1e-6,
                                    random_state=dcfg["random_state"]), C=best)
            oof[te] = est.predict_proba(pre.transform(x[te]))[:, 1]
            chosen_C.append(best)
        rec["regions"][region] = {"pooled_oof_roc_auc": auc(domain, oof),
                                  "selected_C_per_fold": chosen_C}
        pd.DataFrame({"PatientKey": mat["keys"], "Cohort": list(cohort.Cohort),
                      "domain": domain, "p_radiogenomics": oof}).to_csv(
            os.path.join(ppath(cfg, "results_dir"), "domain_%s_oof.csv" % region), index=False)
        log("domain diagnostic %s: pooled OOF ROC-AUC %.4f"
            % (region, rec["regions"][region]["pooled_oof_roc_auc"]), fh)

    ck_save(cfg, unit, psha, {"record": rec})
    counter(unit)
    return rec


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/phase8b_multicenter.yaml")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--resume", action="store_true", help="alias for --all; the run "
                                                          "is idempotent and skips complete units")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if not (args.all or args.resume):
        ap.error("pass --all (or --resume, which is the same idempotent run)")

    t0 = time.time()
    frozen = verify_frozen(cfg)
    psha = protocol_sha(cfg)

    for key in ("models_dir", "preprocessing_dir", "predictions_dir", "results_dir",
                "checkpoints_dir"):
        os.makedirs(ppath(cfg, key), exist_ok=True)
    for sub in ("hyperparameter_surfaces", "calibration", "thresholds", "roc_curves",
                "pr_curves", "confusion_matrices"):
        os.makedirs(os.path.join(ppath(cfg, "results_dir"), sub), exist_ok=True)
    for m in ("A1_gtv", "A2_rim", "A3_fusion"):
        os.makedirs(os.path.join(ppath(cfg, "models_dir"), m), exist_ok=True)
    os.makedirs(os.path.join(cfg["_project_root"], "logs"), exist_ok=True)
    fh = open(ppath(cfg, "training_log"), "a", encoding="utf-8")
    log("=== Phase 8B run started %s (protocol %s) ==="
        % (time.strftime("%Y-%m-%dT%H:%M:%S"), psha[:16]), fh)

    cohort = build_cohort(cfg)
    if sha256(ppath(cfg, "cohort_csv")) != frozen["frozen_before_modelling"]["cohort_csv_sha256"]:
        raise RuntimeError("the frozen multicenter cohort file has changed")
    mats = {r: load_region(cfg, r, cohort) for r in cfg["features"]["regions"]}

    units = ["pooled_fold_%d" % k for k in range(1, int(cfg["validation"]["outer"]["n_splits"]) + 1)]
    units += ["transfer_%s" % d["id"] for d in cfg["validation"]["transfer"]["directions"]]
    units += ["domain_diagnostic"]
    done = {"n": 0}

    def counter(unit):
        done["n"] += 1
        progress_write(cfg, done["n"], len(units), unit, t0)

    progress_write(cfg, 0, len(units), "starting", t0)
    folds = run_pooled(cfg, cohort, mats, psha, fh, counter)
    transfer = run_transfer(cfg, cohort, mats, psha, fh, counter)
    domain = run_domain(cfg, cohort, mats, psha, fh, counter)

    rec = json.load(open(ppath(cfg, "run_json"), encoding="utf-8"))
    rec["modelling"] = {"completed": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "wall_seconds": round(time.time() - t0, 1),
                        "n_candidates_per_region_per_partition": len(candidates(cfg)),
                        "pooled_folds": folds, "cross_cohort": transfer,
                        "domain_diagnostic": domain}
    with open(ppath(cfg, "run_json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1, default=_jsonable)
    progress_write(cfg, len(units), len(units), "complete", t0)
    log("=== Phase 8B modelling complete in %.1f s ===" % (time.time() - t0), fh)
    fh.close()


if __name__ == "__main__":
    main()
