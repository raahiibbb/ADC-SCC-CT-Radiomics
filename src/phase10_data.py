"""Phase 10 - data assembly (read only on every Phase-1..9 artefact)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_cfg(path=os.path.join(ROOT, "config", "phase10.yaml")):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_root"] = ROOT
    return cfg


def p(cfg, key):
    v = cfg["paths"][key]
    return v if os.path.isabs(v) else os.path.join(ROOT, *v.split("/"))


def cohort(cfg):
    c = pd.read_csv(p(cfg, "cohort_csv"))
    c["site"] = c["Cohort"]
    c["y"] = c["label"].astype(int)
    return c[["PatientKey", "PatientID", "site", "y"]].reset_index(drop=True)


def _radiomics(cfg, region):
    a = pd.read_csv(p(cfg, "lung1_%s_csv" % region))
    b = pd.read_csv(p(cfg, "external_%s_csv" % region))
    a["PatientKey"] = "LUNG1::" + a["PatientID"]
    b["PatientKey"] = "RADIOGENOMICS::" + b["PatientID"]
    cols = [c for c in a.columns if c not in ("PatientID", "Histology", "label")]
    assert cols == [c for c in b.columns if c not in ("PatientID", "Histology", "label")]
    df = pd.concat([a[cols], b[cols]], ignore_index=True).set_index("PatientKey")
    feat = [c for c in df.columns if not c.startswith("diagnostics")]
    df = df[feat].apply(pd.to_numeric, errors="coerce")
    df.columns = ["%s__%s" % (region, c) for c in df.columns]
    return df


def _clinical(cfg):
    l1 = pd.read_csv(p(cfg, "lung1_clinical_csv"))
    l1 = pd.DataFrame({"PatientKey": "LUNG1::" + l1["PatientID"],
                       "clin__age": pd.to_numeric(l1["age"], errors="coerce"),
                       "clin__male": (l1["gender"].str.lower() == "male").astype(float)})
    rg = pd.read_csv(p(cfg, "external_clinical_csv"))
    rg = pd.DataFrame({"PatientKey": "RADIOGENOMICS::" + rg["Case ID"],
                       "clin__age": pd.to_numeric(rg["Age at Histological Diagnosis"], errors="coerce"),
                       "clin__male": (rg["Gender"].str.lower() == "male").astype(float)})
    return pd.concat([l1, rg], ignore_index=True).drop_duplicates("PatientKey").set_index("PatientKey")


def _extra(cfg, name):
    f = os.path.join(p(cfg, "extra_features_dir"), "%s.csv" % name)
    if not os.path.isfile(f):
        return None
    df = pd.read_csv(f).set_index("PatientKey")
    # only the block's own feature columns (never ids, status or seed points)
    return df[[c for c in df.columns if c.startswith(name + "__")]]


def blocks(cfg):
    """All available feature blocks, each indexed by PatientKey."""
    out = {"gtv": _radiomics(cfg, "gtv"), "rim": _radiomics(cfg, "rim"),
           "clin": _clinical(cfg)}
    for name in ("fmcib", "loc", "peri"):
        e = _extra(cfg, name)
        if e is not None:
            out[name] = e
    return out


FEATURE_SETS = {
    # radiomics
    "gtv": ["gtv"], "rim": ["rim"], "gtv_rim": ["gtv", "rim"],
    # foundation model
    "fmcib": ["fmcib"],
    "peri": ["peri"],
    # location + clinical ("where and who")
    "loc_clin": ["loc", "clin"],
    # combinations
    "gtv_rim_loc_clin": ["gtv", "rim", "loc", "clin"],
    "rim_loc_clin": ["rim", "loc", "clin"],
    "fmcib_loc_clin": ["fmcib", "loc", "clin"],
    "all": ["gtv", "rim", "fmcib", "loc", "clin"],
}


def matrix(cfg, featureset, coh=None, blk=None):
    coh = cohort(cfg) if coh is None else coh
    blk = blocks(cfg) if blk is None else blk
    parts = []
    for b in FEATURE_SETS[featureset]:
        if b not in blk:
            raise FileNotFoundError("feature block %r not built yet" % b)
        parts.append(blk[b].reindex(coh["PatientKey"]))
    X = pd.concat(parts, axis=1)
    # training-free imputation is not possible for missing values, so they are
    # carried as NaN and imputed inside each training fold (median).
    return X.values.astype(float), list(X.columns)
