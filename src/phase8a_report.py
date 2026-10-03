"""Phase 8A - run record and per-patient manifest.

Collects the provenance of everything Phase 8A produced into

    reports/phase8a_run.json      environment, source hashes, every signature
    reports/phase8a_manifest.csv  one row per downloaded subject

and nothing else.  No metric, no model, no comparison.

    python src/phase8a_report.py --config config/phase8a_external.yaml
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase8a_tcia import load_cfg, ppath, sha256_file                  # noqa: E402
from phase8a_global import derived_phase7a_cfg                         # noqa: E402
from phase8a_patches import derived_gtv_rim_cfg                        # noqa: E402
from phase8a_images import derived_phase6a_cfg                         # noqa: E402


def environment():
    import SimpleITK
    import pydicom
    import radiomics
    import scipy
    import sklearn
    env = {"python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "pandas": pd.__version__,
           "SimpleITK": SimpleITK.Version_VersionString(),
           "pydicom": pydicom.__version__, "pyradiomics": radiomics.__version__,
           "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    try:
        emb = json.load(open(os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "external", "nsclc_radiogenomics", "metadata",
            "cnn_embedding_run.json"), encoding="utf-8"))
        env["torch"] = emb["encoder"]["torch"]
        env["torchvision"] = emb["encoder"]["torchvision"]
    except Exception:                                                  # noqa: BLE001
        pass
    return env


def npz_counts(path, key):
    with np.load(path, allow_pickle=True) as z:
        return int(np.asarray(z[key]).shape[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    root = cfg["_project_root"]

    cohort = pd.read_csv(ppath(cfg, "cohort_csv"))
    inc = cohort[cohort["included"].astype(str).str.lower() == "true"]
    pids = sorted(inc["PatientID"].astype(str))

    _, gsig = derived_phase7a_cfg(cfg)
    _, psig = derived_gtv_rim_cfg(cfg)
    _, csig, tsig = derived_phase6a_cfg(cfg)

    gtv = pd.read_csv(ppath(cfg, "gtv_csv"))
    rim = pd.read_csv(ppath(cfg, "rim_csv"))
    meta_cols = ["PatientID", "Histology", "label"]
    n_gtv_feat = len([c for c in gtv.columns if c not in meta_cols])
    n_rim_feat = len([c for c in rim.columns if c not in meta_cols])

    rows = []
    n_inst = 0
    for _, r in cohort.iterrows():
        pid = str(r["PatientID"])
        row = {"PatientID": pid, "tcia_collection": cfg["source"]["collection"],
               "Histology": r["Histology"], "label": r["label"],
               "included": bool(str(r["included"]).lower() == "true"),
               "exclusion_reason": r["exclusion_reason"],
               "ct_series_uid": r["ct_series_uid"], "seg_series_uid": r["seg_series_uid"],
               "study_uid": r["ct_study_uid"], "seg_zip_sha256": r["seg_zip_sha256"]}
        if row["included"]:
            rec = json.load(open(os.path.join(ppath(cfg, "masks_dir"), pid,
                                              "record.json"), encoding="utf-8"))
            h, g = rec["ct_header"], rec["regions"]
            pf = os.path.join(ppath(cfg, "patch_features_dir"), "%s.npz" % pid)
            ipn = os.path.join(ppath(cfg, "image_patches_dir"), "%s.npz" % pid)
            emb = os.path.join(ppath(cfg, "cnn_embeddings_dir"), "%s.npz" % pid)
            n_bag = npz_counts(pf, "features") if os.path.isfile(pf) else 0
            n_inst += n_bag
            row.update({
                "ct_n_slices": h["n_slices"],
                "ct_pixel_spacing_mm": h["pixel_spacing_mm"][0],
                "ct_slice_spacing_mm": h["slice_spacing_mm_median"],
                "ct_sha256": rec["files"]["ct"]["sha256"],
                "gtv_sha256": rec["files"]["gtv"]["sha256"],
                "rim_sha256": rec["files"]["rim"]["sha256"],
                "gtv_rim_sha256": rec["files"]["gtv_rim"]["sha256"],
                "gtv_voxels": g["native_gtv_voxels"],
                "rim_voxels": g["native_rim_voxels"],
                "gtv_rim_voxels": g["native_gtv_rim_voxels"],
                "gtv_volume_mm3": round(g["native_gtv_volume_mm3"], 2),
                "rim_volume_mm3": round(g["native_rim_volume_mm3"], 2),
                "rim_expansion_mm": g["expansion_mm"],
                "n_global_features_gtv": n_gtv_feat,
                "n_global_features_rim": n_rim_feat,
                "global_radiomics_signature": gsig,
                "n_instances": n_bag,
                "patch_signature": psig,
                "n_crops": npz_counts(ipn, "images") if os.path.isfile(ipn) else 0,
                "crop_signature": csig,
                "n_embeddings": npz_counts(emb, "embeddings") if os.path.isfile(emb) else 0,
                "embedding_signature": cfg["encoder"]["expected_state_fingerprint"],
                "transform_signature": tsig,
            })
        rows.append(row)

    man = pd.DataFrame(rows).sort_values("PatientID").reset_index(drop=True)
    man.to_csv(ppath(cfg, "manifest_csv"), index=False)

    seg_index = json.load(open(ppath(cfg, "seg_index_json"), encoding="utf-8"))
    scr = json.load(open(os.path.join(ppath(cfg, "metadata_dir"), "screening.json"),
                         encoding="utf-8"))
    patch_summary = json.load(open(os.path.join(ppath(cfg, "metadata_dir"),
                                                "patch_extraction_summary.json"),
                                   encoding="utf-8"))
    emb_run = json.load(open(os.path.join(ppath(cfg, "metadata_dir"),
                                          "cnn_embedding_run.json"), encoding="utf-8"))

    run = {
        "phase": "8A",
        "purpose": ("Build and validate the external NSCLC-Radiogenomics cohort and "
                    "prepare it for a LATER, LOCKED external test.  No classification "
                    "result of any kind is computed here."),
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_sha256": sha256_file(cfg["_config_path"]),
        "environment": environment(),
        "source_metadata": {
            "collection": cfg["source"]["collection"],
            "collection_doi": cfg["source"]["collection_doi"],
            "nbia_api": cfg["source"]["nbia_api"],
            "clinical_csv_url": cfg["source"]["clinical_csv_url"],
            "clinical_csv_sha256": sha256_file(ppath(cfg, "clinical_csv")),
            "tcia_patients_sha256": sha256_file(ppath(cfg, "tcia_patients_json")),
            "tcia_series_sha256": sha256_file(ppath(cfg, "tcia_series_json")),
            "n_subjects": scr["n_subjects"],
            "n_seg_series": seg_index["n_seg_series"],
        },
        "cohort": {
            "n_subjects": int(len(cohort)),
            "n_included": int(len(inc)),
            "n_excluded": int(len(cohort) - len(inc)),
            "n_adc": int((inc["label"] == 1).sum()),
            "n_scc": int((inc["label"] == 0).sum()),
            "exclusion_reason_counts": cohort[~cohort["included"].astype(str)
                                              .str.lower().eq("true")]
            ["exclusion_reason"].value_counts().to_dict(),
        },
        "signatures": {
            "global_radiomics": gsig,
            "global_radiomics_expected_phase7a": cfg["global_radiomics"]["expected_phase7a_signature"],
            "patch_sampling": psig,
            "patch_sampling_expected_lung1": cfg["patch_sampling"]["expected_extraction_signature"],
            "crop": csig,
            "crop_expected_phase6a": cfg["crop"]["expected_phase6a_crop_signature"],
            "transform": tsig,
            "transform_expected_phase6a": cfg["transform"]["expected_phase6a_transform_signature"],
            "encoder_state_fingerprint": emb_run["encoder"]["state_fingerprint"],
            "encoder_expected_phase6a": cfg["encoder"]["expected_state_fingerprint"],
            "cpu_probe_sha256": emb_run["cpu_probe_sha256"],
        },
        "products": {
            "n_global_feature_rows": {"gtv": int(len(gtv)), "rim": int(len(rim))},
            "n_global_features": {"gtv": n_gtv_feat, "rim": n_rim_feat},
            "n_instances": int(n_inst),
            "bag_sizes": patch_summary["bag_sizes"],
            "patch_rejections": patch_summary["rejections"],
            "n_embeddings": emb_run["n_embeddings"],
            "embedding_dim": 512,
        },
        "scope": {
            "classification_metrics_computed": 0,
            "models_trained": 0,
            "thresholds_selected": 0,
            "calibrations_fitted": 0,
            "feature_selections_performed": 0,
            "combat_fitted": False,
            "external_labels_used_for": ["eligibility", "recorded_ground_truth"],
            "phase8b_started": False,
        },
    }
    with open(ppath(cfg, "run_json"), "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=1, default=str)
    print(json.dumps({k: run[k] for k in ("cohort", "signatures", "products", "scope")},
                     indent=1, default=str))
    print("\nmanifest: %s (%d rows)" % (ppath(cfg, "manifest_csv"), len(man)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
