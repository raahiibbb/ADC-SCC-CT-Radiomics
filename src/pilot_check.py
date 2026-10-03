"""Phase-4 pilot verification for the GTV+Rim extraction.

Runs the automatic acceptance checks the pilot has to pass before the full
202-patient run is allowed to start:

  1  bags exist, are valid under the GTV+Rim signature, labels match metadata
  2  74 features, identical names in the identical frozen order, all finite
  3  patch geometry: 5x5x5 voxels, stride 5, non-overlapping, full in bounds
  4  the SAME global lattice as Gradient: every start index is a multiple of 5
     (anchor = resampled voxel (0,0,0)), and for patients shared with the
     Gradient pilot the resampled grid is bit-identical and the Gradient patch
     starts are a subset of the GTV+Rim ones (ROI_D is contained in ROI_E)
  5  world <-> index coordinate round-trip, re-derived from the CT, error 0
  6  physical patch size exactly 10x10x10 mm
  7  runtime model for the full cohort

    python src/pilot_check.py --config config/gtv_rim_experiment.yaml
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cohort import load_cohort                                   # noqa: E402
from config_io import (DEFAULT_CONFIG, cfg_path, extraction_signature,  # noqa: E402
                       load_config, project_path)
from extract_patient import validate_existing                    # noqa: E402
from resample import resample_ct_and_mask                        # noqa: E402

R = []


def check(name, ok, detail):
    R.append({"check": name, "pass": bool(ok), "detail": detail})
    print("[%s] %-34s %s" % ("PASS" if ok else "FAIL", name, detail))
    return bool(ok)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--reference-config", default="config/experiment.yaml",
                    help="frozen ROI to compare the lattice against")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ref = load_config(project_path(cfg, *args.reference_config.split("/")))
    sig = extraction_signature(cfg)
    feat_dir = project_path(cfg, *cfg["paths"]["features_dir"].split("/"))
    ref_dir = project_path(ref, *ref["paths"]["features_dir"].split("/"))

    ids = [l.split("#")[0].strip() for l in open(cfg_path(cfg, "pilot_file"),
                                                 encoding="utf-8")]
    ids = [i for i in ids if i]
    cohort = {r["PatientID"]: r for r in load_cohort(cfg)}

    fn_rel = cfg["paths"].get("feature_names_file", "config/feature_names.txt")
    with open(project_path(cfg, *fn_rel.split("/")), encoding="utf-8") as fh:
        expected_names = [l.strip() for l in fh if l.strip()]

    size = np.array(cfg["patches"]["size_voxels"], dtype=int)
    stride = np.array(cfg["patches"]["stride_voxels"], dtype=int)
    spacing = np.array(cfg["preprocessing"]["resample_spacing_mm"], dtype=float)
    min_roi = int(cfg["patches"]["min_roi_voxels"])

    print("roi              : %s (%s)" % (cfg["experiment"]["roi"],
                                          cfg["experiment"]["roi_filename"]))
    print("config signature : %s" % sig)
    print("pilot            : %s" % ", ".join(ids))
    print("-" * 96)

    invalid, per_patient = [], []
    bad_names, nonfinite, bad_size, bad_lattice = [], [], [], []
    bad_bounds, overlap, bad_center, bad_minroi, bad_boxmm = [], [], [], [], []
    roundtrip, grid_mismatch, not_superset = [], [], []
    classes = set()

    for pid in ids:
        npz = os.path.join(feat_dir, "%s.npz" % pid)
        rec = cohort[pid]
        ok, why = validate_existing(npz, pid, sig, rec["label"])
        if not ok:
            invalid.append({"patient_id": pid, "reason": why})
            continue
        classes.add(rec["Histology"])

        with np.load(npz, allow_pickle=True) as z:
            f = z["features"]
            names = [str(x) for x in z["feature_names"]]
            starts = z["coords_start_index"].astype(np.int64)
            centers = z["coords_index"].astype(np.int64)
            world = z["coords_world"]
            bmin, bmax = z["bbox_world_min"], z["bbox_world_max"]
            roi_vox = z["roi_voxels"]
            gsize = z["grid_size"].astype(np.int64)
            gorigin = z["grid_origin"]
            gspacing = z["grid_spacing"]
            gdirection = z["grid_direction"]
            psize = z["patch_size"].astype(int)
            pstride = z["patch_stride"].astype(int)
        n = int(f.shape[0])

        if names != expected_names:
            bad_names.append(pid)
        if not np.all(np.isfinite(f)):
            nonfinite.append(pid)
        if not (np.array_equal(psize, size) and np.array_equal(pstride, stride)):
            bad_size.append(pid)
        # lattice anchored at voxel (0,0,0)  ->  start % stride == 0
        if np.any(starts % stride[None, :] != 0):
            bad_lattice.append(pid)
        if not np.array_equal(centers, starts + (size // 2)[None, :]):
            bad_center.append(pid)
        # non-overlapping and fully inside the resampled volume
        if len({tuple(s) for s in starts.tolist()}) != n:
            overlap.append(pid)
        if np.any(starts < 0) or np.any(starts + size[None, :] > gsize[None, :]):
            bad_bounds.append(pid)
        if n and int(roi_vox.min()) < min_roi:
            bad_minroi.append(pid)
        # physical patch box exactly size * spacing mm
        if n and not np.allclose(np.abs(bmax - bmin).max(axis=0), size * spacing,
                                 atol=1e-6):
            bad_boxmm.append(pid)

        # independent world<->index round-trip from a fresh resampling
        pdir = os.path.join(cfg["paths"]["vuong_root"], pid)
        ct = sitk.ReadImage(os.path.join(pdir, cfg["paths"]["ct_filename"]))
        mk = sitk.ReadImage(os.path.join(pdir, cfg["experiment"]["roi_filename"]))
        ct_r, mask_r, grid = resample_ct_and_mask(ct, mk, cfg)
        mask_a = sitk.GetArrayFromImage(mask_r)
        back = np.array([ct_r.TransformPhysicalPointToIndex([float(x) for x in p])
                         for p in world], dtype=np.int64)
        err = int(np.abs(back - centers).max()) if n else 0
        centres_in = int(sum(1 for c in centers if mask_a[c[2], c[1], c[0]] > 0))
        if err != 0 or centres_in != n:
            roundtrip.append({"patient_id": pid, "err": err,
                              "centres_in_roi": centres_in, "n": n})
        grid_ok = (list(gsize) == list(grid["size"])
                   and np.allclose(gspacing, grid["spacing"])
                   and np.allclose(gorigin, grid["origin"])
                   and np.allclose(gdirection, grid["direction"]))
        if not grid_ok:
            grid_mismatch.append(pid)

        # shared lattice with the frozen reference ROI, where available
        ref_npz = os.path.join(ref_dir, "%s.npz" % pid)
        shared = None
        if os.path.isfile(ref_npz):
            with np.load(ref_npz, allow_pickle=True) as rz:
                rstarts = rz["coords_start_index"].astype(np.int64)
                rgrid = (list(rz["grid_size"].astype(np.int64)),
                         rz["grid_origin"], rz["grid_spacing"], rz["grid_direction"])
            same_grid = (rgrid[0] == list(gsize)
                         and np.array_equal(rgrid[1], gorigin)
                         and np.array_equal(rgrid[2], gspacing)
                         and np.array_equal(rgrid[3], gdirection))
            a = {tuple(s) for s in rstarts.tolist()}
            b = {tuple(s) for s in starts.tolist()}
            shared = {"reference_roi": ref["experiment"]["roi"],
                      "reference_patches": len(a), "shared_starts": len(a & b),
                      "reference_subset_of_this": bool(a <= b),
                      "identical_resampled_grid": bool(same_grid)}
            if not same_grid and pid not in grid_mismatch:
                grid_mismatch.append(pid)
            if not a <= b:
                not_superset.append({"patient_id": pid,
                                     "reference_only": len(a - b)})

        per_patient.append({
            "patient_id": pid, "histology": rec["Histology"],
            "label": int(rec["label"]), "n_patches": n,
            "grid_size": [int(v) for v in gsize],
            "grid_reproduced_from_ct": bool(grid_ok),
            "roi_voxels_min": int(roi_vox.min()) if n else None,
            "roi_voxels_max": int(roi_vox.max()) if n else None,
            "world_index_roundtrip_max_error_voxels": err,
            "centres_inside_roi": centres_in,
            "reference_lattice": shared,
        })

    # ------------------------------------------------------------------ checks
    check("1_bags_valid", not invalid and len(per_patient) == len(ids),
          "%d/%d valid bags under signature %s" % (len(per_patient), len(ids), sig))
    check("2_both_classes_and_sizes", len(classes) == 2,
          "%s; patches per bag %s" % (sorted(classes),
                                      [p["n_patches"] for p in per_patient]))
    check("3_feature_names_frozen_order", not bad_names,
          "%d name/order mismatches against %d frozen names"
          % (len(bad_names), len(expected_names)))
    check("4_all_features_finite", not nonfinite,
          "%d patients with non-finite values in %d patches"
          % (len(nonfinite), sum(p["n_patches"] for p in per_patient)))
    check("5_patch_size_and_stride", not bad_size and not bad_center,
          "size %s stride %s in every bag; centre == start + 2"
          % (list(size), list(stride)))
    check("6_global_lattice_anchor_0", not bad_lattice,
          "%d bags with a start index off the stride-%d lattice"
          % (len(bad_lattice), stride[0]))
    check("7_non_overlapping_in_bounds", not overlap and not bad_bounds,
          "%d duplicate starts, %d out-of-bounds blocks"
          % (len(overlap), len(bad_bounds)))
    check("8_min_roi_voxels_respected", not bad_minroi,
          "min roi_voxels over the pilot = %d (threshold %d)"
          % (min(p["roi_voxels_min"] for p in per_patient), min_roi))
    check("9_physical_patch_box_mm", not bad_boxmm,
          "every saved box is %g x %g x %g mm" % tuple(size * spacing))
    check("10_world_index_roundtrip", not roundtrip,
          "max error 0 voxels, every centre inside the ROI, re-derived from CT")
    n_ref = sum(1 for p in per_patient if p["reference_lattice"])
    check("11_same_lattice_as_reference", not grid_mismatch and not not_superset,
          "%d grid mismatches; %d/%d bags compared to %s, %d missing a %s patch"
          % (len(grid_mismatch), n_ref, len(per_patient), ref["experiment"]["roi"],
             len(not_superset), ref["experiment"]["roi"]))

    # ------------------------------------------------------------------ runtime
    summary_path = cfg_path(cfg, "extraction_summary_json", tag="pilot")
    runtime = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            s = json.load(fh)
        n_elig = int(s["n_eligible_cohort"])
        secs = s["seconds_per_patient_mean"]
        patches = s["patches_total_retained"]
        vol_field = cfg["cohort"].get("roi_volume_field", "gradient_volume_mm3")
        rows = load_cohort(cfg)
        vols = {r["PatientID"]: float(r[vol_field] or 0) for r in rows}
        pilot_vol = sum(vols[p] for p in ids)
        cohort_vol = sum(vols[r["PatientID"]] for r in rows if r["eligible"])
        runtime = {
            "pilot_wall_seconds": s["wall_seconds"],
            "pilot_patients": s["n_selected"],
            "pilot_patches": patches,
            "seconds_per_patient_mean_pilot": secs,
            "eligible_cohort": n_elig,
            "naive_projection_minutes": round(secs * n_elig / 60.0, 1),
            "pilot_roi_volume_mm3": round(pilot_vol, 1),
            "cohort_roi_volume_mm3": round(cohort_vol, 1),
            # the pilot is deliberately size-extreme, so the honest projection
            # scales by ROI volume rather than by patient count
            "volume_scaled_projection_minutes": round(
                s["wall_seconds"] * cohort_vol / pilot_vol / 60.0, 1),
            "projected_cohort_patches": int(round(
                patches * cohort_vol / pilot_vol)),
        }
        print("-" * 96)
        print("runtime: pilot %.1f s for %d patches; naive projection %.1f min, "
              "ROI-volume-scaled %.1f min (~%d patches)"
              % (s["wall_seconds"], patches, runtime["naive_projection_minutes"],
                 runtime["volume_scaled_projection_minutes"],
                 runtime["projected_cohort_patches"]))

    out = {
        "roi": cfg["experiment"]["roi"],
        "config_signature": sig,
        "pilot_patients": ids,
        "per_patient": per_patient,
        "problems": {"invalid": invalid, "name_mismatch": bad_names,
                     "nonfinite": nonfinite, "bad_patch_size": bad_size,
                     "off_lattice": bad_lattice, "duplicate_starts": overlap,
                     "out_of_bounds": bad_bounds, "below_min_roi": bad_minroi,
                     "bad_box_mm": bad_boxmm, "roundtrip": roundtrip,
                     "grid_mismatch": grid_mismatch,
                     "reference_not_subset": not_superset},
        "runtime": runtime,
        "checks": R,
        "n_checks": len(R),
        "n_passed": sum(1 for r in R if r["pass"]),
        "all_passed": all(r["pass"] for r in R),
    }
    path = project_path(cfg, "reports", "gtv_rim_pilot_check.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("-" * 96)
    print("%d/%d pilot checks passed  ->  %s"
          % (out["n_passed"], out["n_checks"], path))
    return 0 if out["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
