"""Phase 8A - external whole-region (GLOBAL) GTV / rim radiomics.

Phase 8A does NOT re-implement the Phase-7A extractor.  It loads the FROZEN
`config/phase7a.yaml`, redirects only its input root and its output paths to
the external namespace, and then calls the FROZEN `src/phase7a_extract.py` and
`src/phase7a_regions.py` functions.  Consequently:

  * the region derivation is the same code
      GTV = ROI_A_GTV.nii.gz on the shared 2 mm grid (nearest),
      rim = (ROI_E_GTV_Rim.nii.gz MINUS ROI_A_GTV.nii.gz) on that same grid;
  * the PyRadiomics parameter file is the same file;
  * the extraction signature is a hash over the scientific blocks only, so it
    MUST come out equal to the frozen Phase-7A signature - and the run refuses
    to continue if it does not.

Feature extraction is label-independent, so the FULL external bank is
extracted.  NO feature selection, NO model and NO metric is computed here.

    python src/phase8a_global.py --config config/phase8a_external.yaml [--limit N]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import phase7a_extract as P7                                       # noqa: E402
from phase8a_tcia import load_cfg, ppath                           # noqa: E402


def derived_phase7a_cfg(cfg8):
    """The frozen Phase-7A config with ONLY paths redirected to the external set."""
    p7_path = os.path.join(cfg8["_project_root"], "config", "phase7a.yaml")
    cfg7 = P7.load_cfg(p7_path)

    scientific = {k: copy.deepcopy(cfg7[k]) for k in ("regions", "preprocessing", "radiomics")}

    cfg = copy.deepcopy(cfg7)
    cfg["paths"]["vuong_root"] = ppath(cfg8, "masks_dir")
    cfg["paths"]["cache_dir"] = cfg8["paths"]["global_cache_dir"]
    cfg["paths"]["features_dir"] = cfg8["paths"]["global_features_dir"]
    cfg["paths"]["gtv_csv"] = cfg8["paths"]["gtv_csv"]
    cfg["paths"]["rim_csv"] = cfg8["paths"]["rim_csv"]
    cfg["paths"]["geometry_csv"] = cfg8["paths"]["region_geometry_csv"]
    cfg["paths"]["feature_names_json"] = cfg8["paths"]["global_feature_names_json"]
    cfg["paths"]["manifest_csv"] = ("external/nsclc_radiogenomics/global_features/"
                                    "extraction_manifest.csv")

    # nothing scientific may differ from the frozen Phase-7A configuration
    for k in ("regions", "preprocessing", "radiomics"):
        if cfg[k] != scientific[k]:
            raise RuntimeError("derived config changed the frozen %s block" % k)

    sig = P7.extraction_signature(cfg)
    expected = str(cfg8["global_radiomics"]["expected_phase7a_signature"])
    if sig != expected:
        raise RuntimeError("external global-radiomics signature %s != the frozen "
                           "Phase-7A signature %s" % (sig, expected))
    return cfg, sig


def included_patients(cfg8):
    df = pd.read_csv(ppath(cfg8, "cohort_csv"))
    inc = df[df["included"].astype(str).str.lower() == "true"].copy()
    inc = inc.sort_values("PatientID").reset_index(drop=True)
    inc["label"] = inc["label"].astype(int)
    return inc[["PatientID", "Histology", "label"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg8 = load_cfg(args.config)
    cfg7, sig = derived_phase7a_cfg(cfg8)
    print("external global-radiomics extraction signature: %s  (== frozen Phase-7A)" % sig)

    cohort = included_patients(cfg8)
    pids = list(cohort["PatientID"])
    if args.limit:
        pids = pids[:int(args.limit)]

    os.makedirs(P7.ppath(cfg7, "cache_dir"), exist_ok=True)
    extractors = {r: P7.build_extractor(cfg7, r) for r in ("gtv", "rim")}

    done = skipped = 0
    failures = []
    t0 = time.time()
    for i, pid in enumerate(pids, 1):
        if not args.force and P7.cached_ok(cfg7, pid, sig):
            skipped += 1
            continue
        try:
            rec = P7.extract_patient(cfg7, pid, extractors, sig)
            with open(P7.cache_path(cfg7, pid), "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=1)
            done += 1
            print("[%3d/%3d] %s  gtv %d / rim %d features  %.1fs"
                  % (i, len(pids), pid, rec["regions"]["gtv"]["n_features"],
                     rec["regions"]["rim"]["n_features"], rec["timing"]["total_s"]),
                  flush=True)
        except Exception as exc:                                   # noqa: BLE001
            failures.append({"PatientID": pid, "error": str(exc)[:300]})
            print("[%3d/%3d] %s  FAILED: %s" % (i, len(pids), pid, str(exc)[:200]),
                  flush=True)

    print("extracted %d, skipped %d cached, %d failures, %.1f min"
          % (done, skipped, len(failures), (time.time() - t0) / 60.0))

    if not failures and (args.limit is None):
        names, families, manifest = P7.assemble(cfg7, cohort, sig)
        print("gtv features: %d   rim features: %d" % (len(names["gtv"]), len(names["rim"])))
        print(json.dumps(families, indent=1)[:1200])
    elif failures:
        print(json.dumps(failures, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
