"""Phase 7A - pilot extraction QA on ~6 representative patients.

Gate for the full 202-patient extraction.  Every check re-derives its answer
from the cached extraction records and from the masks on disk; nothing is taken
on trust from the extraction driver.

    python src/phase7a_pilot_qa.py --config config/phase7a.yaml
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase7a_extract import (cache_path, extraction_signature, load_cfg,       # noqa: E402
                             load_cohort, pilot_patients, ppath)
from phase7a_regions import build_regions                                       # noqa: E402

EXPECTED_IMAGE_TYPES = 12          # original + 3 LoG + 8 wavelet sub-bands
EXPECTED_TEXTURE_PER_IMAGE = 93    # firstorder 18 + glcm 24 + glrlm 16 + glszm 16 + gldm 14 + ngtdm 5
EXPECTED_SHAPE = 14


def _families(names):
    out = {}
    for n in names:
        p = n.split("_")
        out.setdefault(p[0], collections.Counter())[p[1]] += 1
    return {k: dict(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/phase7a.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    sig = extraction_signature(cfg)
    cohort = load_cohort(cfg)
    pilots = pilot_patients(cfg, cohort)

    checks, per_patient = [], []

    def chk(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    recs = {}
    missing = [p for p in pilots if not os.path.isfile(cache_path(cfg, p))]
    chk("pilot records exist", not missing,
        "%d pilot patients: %s%s" % (len(pilots), ", ".join(pilots),
                                     "" if not missing else "; MISSING %s" % missing))
    if missing:
        raise SystemExit("run: python src/phase7a_extract.py --pilot  first")
    for p in pilots:
        with open(cache_path(cfg, p), "r", encoding="utf-8") as fh:
            recs[p] = json.load(fh)

    chk("extraction signature current",
        all(r["extraction_signature"] == sig for r in recs.values()),
        "signature %s" % sig)

    # ---- 1/2/3  geometry, re-derived from the masks on disk ----------------
    geo_ok, geo_detail = True, []
    for p in pilots:
        reg = build_regions(cfg, p)                 # raises if any property fails
        g = sitk.GetArrayViewFromImage(reg.masks["gtv"]) > 0
        r = sitk.GetArrayViewFromImage(reg.masks["rim"]) > 0
        e = sitk.ReadImage(os.path.join(cfg["paths"]["vuong_root"], p,
                                        cfg["paths"]["gtv_rim_filename"]))
        inter = int((g & r).sum())
        cached = recs[p]["geometry"]
        same = (int(g.sum()) == cached["gtv_voxels"] and int(r.sum()) == cached["rim_voxels"])
        ok = (inter == 0 and cached["union_ok"] and cached["native_subset_ok"] and same)
        geo_ok &= ok
        geo_detail.append({"PatientID": p, "gtv_voxels": int(g.sum()),
                           "rim_voxels": int(r.sum()),
                           "intersection": inter,
                           "union_mismatch": cached["union_vs_gtv_rim_mismatch_voxels"],
                           "native_gtv_outside": cached["native_gtv_outside_gtv_rim_voxels"],
                           "gtv_volume_mm3": round(cached["gtv_volume_mm3"], 1),
                           "rim_volume_mm3": round(cached["rim_volume_mm3"], 1),
                           "rim_fraction": round(cached["rim_fraction_of_gtv_rim"], 4),
                           "reproduced_from_disk": bool(same)})
        del e
        per_patient.append(geo_detail[-1])
    chk("GTV geometry, rim geometry and exact GTV+rim reconstruction", geo_ok,
        geo_detail)

    # ---- 4  extraction succeeded -------------------------------------------
    chk("feature extraction succeeded for both regions",
        all(recs[p]["regions"][reg]["n_features"] > 0
            for p in pilots for reg in ("gtv", "rim")),
        {p: {reg: recs[p]["regions"][reg]["n_features"] for reg in ("gtv", "rim")}
         for p in pilots})

    # ---- 5  no unexpected NaN / Inf ----------------------------------------
    bad = {p: {reg: (recs[p]["regions"][reg]["n_nan"], recs[p]["regions"][reg]["n_inf"])
               for reg in ("gtv", "rim")} for p in pilots}
    chk("no unexpected NaN / Inf",
        all(v == (0, 0) for d in bad.values() for v in d.values()),
        {"(n_nan, n_inf) per patient/region": bad})

    # ---- 6  feature names identical across patients -------------------------
    name_ok, name_detail = True, {}
    for reg in ("gtv", "rim"):
        ref = recs[pilots[0]]["regions"][reg]["feature_names"]
        same = all(recs[p]["regions"][reg]["feature_names"] == ref for p in pilots)
        name_ok &= same
        name_detail[reg] = {"n": len(ref), "identical_across_patients": bool(same)}
    chk("feature names identical across patients", name_ok, name_detail)

    # ---- 7  family / image-type counts --------------------------------------
    fam = {reg: _families(recs[pilots[0]]["regions"][reg]["feature_names"])
           for reg in ("gtv", "rim")}
    gtv_ok = (len(fam["gtv"]) == EXPECTED_IMAGE_TYPES
              and sum(fam["gtv"]["original"].values()) == EXPECTED_TEXTURE_PER_IMAGE + EXPECTED_SHAPE
              and fam["gtv"]["original"].get("shape") == EXPECTED_SHAPE
              and all(sum(v.values()) == EXPECTED_TEXTURE_PER_IMAGE
                      for k, v in fam["gtv"].items() if k != "original")
              and not any("shape" in v for k, v in fam["gtv"].items() if k != "original"))
    rim_ok = (len(fam["rim"]) == EXPECTED_IMAGE_TYPES
              and all(sum(v.values()) == EXPECTED_TEXTURE_PER_IMAGE for v in fam["rim"].values())
              and not any("shape" in v for v in fam["rim"].values()))
    chk("original / LoG / wavelet feature-family counts", gtv_ok and rim_ok,
        {"gtv_total": len(recs[pilots[0]]["regions"]["gtv"]["feature_names"]),
         "rim_total": len(recs[pilots[0]]["regions"]["rim"]["feature_names"]),
         "gtv": fam["gtv"], "rim": fam["rim"]})

    # ---- 8  physically plausible values -------------------------------------
    plaus, plaus_detail = True, []
    for p in pilots:
        for reg in ("gtv", "rim"):
            nm = recs[p]["regions"][reg]["feature_names"]
            vv = recs[p]["regions"][reg]["values"]
            d = {k: v for k, v in zip(nm, vv)}
            mean = d["original_firstorder_Mean"]
            mn, mx = d["original_firstorder_Minimum"], d["original_firstorder_Maximum"]
            ent = d["original_firstorder_Entropy"]
            row = {"PatientID": p, "region": reg, "mean_HU": round(mean, 2),
                   "min_HU": mn, "max_HU": mx, "entropy": round(ent, 3)}
            ok = (-1024.0 <= mn <= mx <= 200.0 and mn <= mean <= mx and 0.0 <= ent <= 12.0)
            if reg == "gtv":
                vol = d["original_shape_MeshVolume"]
                row["mesh_volume_mm3"] = round(vol, 1)
                ok &= vol > 0
            plaus &= ok
            row["plausible"] = bool(ok)
            plaus_detail.append(row)
    chk("feature values physically plausible "
        "(HU inside the resegmentation range, min<=mean<=max, finite entropy, positive volume)",
        plaus, plaus_detail)

    out = {"config": os.path.relpath(cfg["_config_path"], cfg["_project_root"]).replace(os.sep, "/"),
           "extraction_signature": sig,
           "pilot_patients": pilots,
           "n_checks": len(checks),
           "n_pass": sum(1 for c in checks if c["pass"]),
           "all_pass": all(c["pass"] for c in checks),
           "checks": checks}
    os.makedirs(ppath(cfg, "qa_dir"), exist_ok=True)
    with open(ppath(cfg, "pilot_json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    for c in checks:
        print("%-4s %s" % ("PASS" if c["pass"] else "FAIL", c["check"]))
    print("\n%d/%d pilot QA checks pass -> %s"
          % (out["n_pass"], out["n_checks"], ppath(cfg, "pilot_json")))
    if not out["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
