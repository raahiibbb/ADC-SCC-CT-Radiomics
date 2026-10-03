"""Phase 8A - external GTV+Rim local patch sampling.

Phase 8A does NOT re-implement the Phase-4 sampler.  It loads the FROZEN
`config/gtv_rim_experiment.yaml`, redirects ONLY the input root and the output
directory to the external namespace, and calls the FROZEN
`src/extract_patient.py:extract_patient`.  The external instances are therefore
produced by literally the same code path as the 37 464 LUNG1 GTV+Rim
instances:

  * the same 2 mm isotropic grid (`src/resample.py:reference_grid`), derived
    from each patient's own CT, CT linear / mask nearest;
  * the same global lattice anchored at resampled voxel (0, 0, 0)
    (`src/patches.py:enumerate_patches`);
  * the same 5x5x5 voxel patch geometry and stride 5, non-overlapping,
    `require_full_in_bounds`;
  * the same retention rule - centre inside the ROI, at least 27 ROI voxels in
    the block, and a complete finite 74-feature PyRadiomics vector.

`src/config_io.py:extraction_signature` hashes exactly the scientific keys
(roi, roi_filename, preprocessing, patches, radiomics, the PyRadiomics
parameter file and the histology map) and none of the paths.  The external run
therefore MUST reproduce the frozen LUNG1 GTV+Rim signature `9825ed13538ee681`,
and refuses to start if it does not.  That equality is the machine proof that
the two samplers are the same algorithm with the same parameters.

    python src/phase8a_patches.py --config config/phase8a_external.yaml [--limit N]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_io import extraction_signature, load_config              # noqa: E402
from extract_patient import build_extractor, extract_patient         # noqa: E402
from phase8a_tcia import load_cfg, ppath                             # noqa: E402


def derived_gtv_rim_cfg(cfg8):
    """The frozen Phase-4 GTV+Rim config with ONLY paths redirected."""
    p4 = os.path.join(cfg8["_project_root"], "config", "gtv_rim_experiment.yaml")
    cfg4 = load_config(p4)
    scientific = {k: copy.deepcopy(cfg4[k]) for k in ("preprocessing", "patches", "radiomics")}
    roi_before = (cfg4["experiment"]["roi"], cfg4["experiment"]["roi_filename"])

    cfg = copy.deepcopy(cfg4)
    cfg["paths"]["vuong_root"] = ppath(cfg8, "masks_dir")
    cfg["paths"]["features_dir"] = cfg8["paths"]["patch_features_dir"]

    for k in ("preprocessing", "patches", "radiomics"):
        if cfg[k] != scientific[k]:
            raise RuntimeError("derived config changed the frozen %s block" % k)
    if (cfg["experiment"]["roi"], cfg["experiment"]["roi_filename"]) != roi_before:
        raise RuntimeError("derived config changed the ROI")

    sig = extraction_signature(cfg)
    expected = str(cfg8["patch_sampling"]["expected_extraction_signature"])
    if sig != expected:
        raise RuntimeError("external patch-sampling signature %s != the frozen LUNG1 "
                           "GTV+Rim signature %s" % (sig, expected))
    return cfg, sig


def included_patients(cfg8):
    df = pd.read_csv(ppath(cfg8, "cohort_csv"))
    inc = df[df["included"].astype(str).str.lower() == "true"].copy()
    return inc.sort_values("PatientID").reset_index(drop=True)


def valid_bag(path, pid, signature):
    if not os.path.isfile(path):
        return False
    try:
        with np.load(path, allow_pickle=True) as z:
            return (str(z["patient_id"]) == pid
                    and str(z["config_signature"]) == signature
                    and int(np.asarray(z["features"]).shape[0]) > 0)
    except Exception:                                              # noqa: BLE001
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg8 = load_cfg(args.config)
    cfg4, sig = derived_gtv_rim_cfg(cfg8)
    print("external patch-sampling signature: %s  (== frozen LUNG1 GTV+Rim)" % sig)

    out_dir = ppath(cfg8, "patch_features_dir")
    os.makedirs(out_dir, exist_ok=True)
    cohort = included_patients(cfg8)
    rows = cohort.to_dict("records")
    if args.limit:
        rows = rows[:int(args.limit)]

    extractor = build_extractor(cfg4)
    manifest, failures = [], []
    done = skipped = 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        pid = r["PatientID"]
        npz = os.path.join(out_dir, "%s.npz" % pid)
        if not args.force and valid_bag(npz, pid, sig):
            skipped += 1
            with open(os.path.join(out_dir, "%s.qa.json" % pid), "r", encoding="utf-8") as fh:
                manifest.append(json.load(fh))
            continue
        patient = {"PatientID": pid, "Histology": r["Histology"], "label": int(r["label"])}
        try:
            qa = extract_patient(cfg4, patient, extractor=extractor, out_dir=out_dir)
            manifest.append(qa)
            done += 1
            print("[%3d/%3d] %s  %d/%d patches kept  %.1fs"
                  % (i, len(rows), pid, qa["n_retained_patches"],
                     qa["n_candidate_patches"], qa["timing_seconds"]["total"]), flush=True)
        except Exception as exc:                                   # noqa: BLE001
            failures.append({"PatientID": pid, "error": str(exc)[:300]})
            print("[%3d/%3d] %s  FAILED: %s" % (i, len(rows), pid, str(exc)[:200]), flush=True)

    n_patches = sum(m["n_retained_patches"] for m in manifest)
    print("extracted %d, skipped %d cached, %d failures, %d instances, %.1f min"
          % (done, skipped, len(failures), n_patches, (time.time() - t0) / 60.0))

    summary = {
        "extraction_signature": sig,
        "n_patients": len(manifest),
        "n_instances": int(n_patches),
        "n_failures": len(failures),
        "failures": failures,
        "bag_sizes": {
            "min": int(min((m["n_retained_patches"] for m in manifest), default=0)),
            "median": float(np.median([m["n_retained_patches"] for m in manifest]))
                      if manifest else 0.0,
            "max": int(max((m["n_retained_patches"] for m in manifest), default=0)),
            "n_below_20": int(sum(1 for m in manifest if m["n_retained_patches"] < 20)),
        },
        "rejections": _sum_rejections(manifest),
        "n_features": sorted({m["n_features"] for m in manifest}),
    }
    path = os.path.join(ppath(cfg8, "metadata_dir"), "patch_extraction_summary.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}, indent=1))
    return 1 if failures else 0


def _sum_rejections(manifest):
    out = {}
    for m in manifest:
        for k, v in m.get("rejected_by_reason", {}).items():
            out[k] = out.get(k, 0) + int(v)
    return dict(sorted(out.items()))


if __name__ == "__main__":
    sys.exit(main())
