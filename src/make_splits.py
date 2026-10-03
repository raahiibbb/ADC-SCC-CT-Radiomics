"""Freeze the patient-level Stratified 5-fold cross-validation split.

The split unit is ALWAYS the patient.  Every patch of a patient stays in the
same fold; nothing here touches patch data at all, so the split cannot depend on
extracted features and can be frozen before any extraction or training.

Writes splits/stratified_5fold.csv (one row per patient per fold) and
splits/stratified_5fold_meta.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cohort import load_cohort                       # noqa: E402
from config_io import DEFAULT_CONFIG, load_config, project_path   # noqa: E402

FIELDS = ["fold", "PatientID", "Histology", "label", "split"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing frozen split file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    v = cfg["validation"]

    rows = sorted([r for r in load_cohort(cfg) if r["eligible"]],
                  key=lambda r: r["PatientID"])
    pids = np.array([r["PatientID"] for r in rows])
    y = np.array([r["label"] for r in rows], dtype=int)
    if len(pids) != cfg["cohort"]["expected_eligible"]:
        raise SystemExit("expected %d eligible patients, found %d"
                         % (cfg["cohort"]["expected_eligible"], len(pids)))
    if len(set(pids)) != len(pids):
        raise SystemExit("duplicate patient IDs in cohort")

    out_path = project_path(cfg, *cfg["paths"]["splits_file"].split("/"))
    if os.path.exists(out_path) and not args.force:
        raise SystemExit("%s already exists (frozen). Use --force to regenerate."
                         % out_path)

    skf = StratifiedKFold(n_splits=int(v["n_splits"]), shuffle=bool(v["shuffle"]),
                          random_state=int(v["random_state"]))
    by_hist = {r["PatientID"]: r["Histology"] for r in rows}

    out_rows, fold_meta = [], []
    seen_test = set()
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        assert not (set(tr) & set(te)), "train/test overlap in fold %d" % k
        seen_test |= set(te.tolist())
        for i in tr:
            out_rows.append({"fold": k, "PatientID": pids[i],
                             "Histology": by_hist[pids[i]], "label": int(y[i]),
                             "split": "train"})
        for i in te:
            out_rows.append({"fold": k, "PatientID": pids[i],
                             "Histology": by_hist[pids[i]], "label": int(y[i]),
                             "split": "test"})
        n_adc_tr = int((y[tr] == 1).sum())
        n_scc_tr = int((y[tr] == 0).sum())
        fold_meta.append({
            "fold": k,
            "n_train": len(tr), "n_test": len(te),
            "train_adc": n_adc_tr, "train_scc": n_scc_tr,
            "test_adc": int((y[te] == 1).sum()), "test_scc": int((y[te] == 0).sum()),
            "train_adc_fraction": round(n_adc_tr / len(tr), 4),
            "test_adc_fraction": round(float((y[te] == 1).mean()), 4),
            # pos_weight is recomputed per fold at training time; recorded here
            # for reference only (Phase 2 must recompute it from its own subset)
            "reference_pos_weight_train": round(n_scc_tr / max(1, n_adc_tr), 4),
        })

    assert seen_test == set(range(len(y))), "test folds do not partition the cohort"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    with open(out_path, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()

    meta = {
        "frozen": True,
        "note": ("Patient-level split. All patches of a patient share a fold. "
                 "Any inner train/validation subset for model selection is drawn "
                 "from the train patients at training time and is NOT fixed here."),
        "roi": cfg["experiment"]["roi"],
        "roi_filename": cfg["experiment"]["roi_filename"],
        "cohort_file": cfg["paths"]["cohort_file"],
        "n_patients": len(pids),
        "n_adc": int((y == 1).sum()), "n_scc": int((y == 0).sum()),
        "label_encoding": cfg["cohort"]["histology_map"],
        "sklearn_call": ("StratifiedKFold(n_splits=%d, shuffle=%s, random_state=%d)"
                         % (v["n_splits"], v["shuffle"], v["random_state"])),
        "random_state": int(v["random_state"]),
        "patient_order": "sorted by PatientID ascending before splitting",
        "sklearn_version": __import__("sklearn").__version__,
        "numpy_version": np.__version__,
        "folds": fold_meta,
        "splits_file": os.path.relpath(out_path, cfg["_project_root"]).replace("\\", "/"),
        "splits_file_sha256": sha,
    }
    mpath = out_path.replace(".csv", "_meta.json")
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print("frozen split: %s" % out_path)
    print("sha256: %s" % sha)
    for f in fold_meta:
        print("  fold %d  train %3d (%2d ADC / %3d SCC)   test %2d (%2d ADC / %2d SCC)"
              % (f["fold"], f["n_train"], f["train_adc"], f["train_scc"],
                 f["n_test"], f["test_adc"], f["test_scc"]))


if __name__ == "__main__":
    main()
