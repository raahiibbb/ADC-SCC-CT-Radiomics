"""Header-level geometry / integrity sweep over an ROI cohort (ROI from --config).

Checks per patient, without loading full voxel arrays where avoidable:
  * CT.nii.gz and the configured ROI mask exist,
  * identical size / spacing / origin / direction,
  * mask is binary and non-empty,
  * ROI fits inside the CT field of view,
  * the ROI bounding box is large enough to hold at least one 5x5x5 patch
    after 2 mm resampling.

Writes reports/geometry_check.csv and prints a summary.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cohort import load_cohort                      # noqa: E402
from config_io import (DEFAULT_CONFIG, cfg_path, load_config,    # noqa: E402
                       project_path)
from resample import reference_grid                 # noqa: E402

FIELDS = ["PatientID", "Histology", "label", "ct_exists", "roi_exists",
          "ct_size", "ct_spacing", "ct_pixel_type", "size_match", "spacing_match",
          "origin_match", "direction_match", "mask_binary", "mask_nonempty",
          "mask_voxels", "mask_volume_mm3", "roi_inside_fov", "resampled_size",
          "roi_bbox_extent_ijk_2mm", "bbox_fits_one_patch", "ct_hu_min", "ct_hu_max",
          "roi_hu_min", "roi_hu_median", "roi_hu_max", "frac_roi_hu_in_reseg_range",
          "all_ok"]


def check_patient(cfg, rec, deep=True):
    pid = rec["PatientID"]
    pdir = os.path.join(cfg["paths"]["vuong_root"], pid)
    ct_p = os.path.join(pdir, cfg["paths"]["ct_filename"])
    roi_p = os.path.join(pdir, cfg["experiment"]["roi_filename"])
    row = {k: "" for k in FIELDS}
    row.update({"PatientID": pid, "Histology": rec["Histology"], "label": rec["label"],
                "ct_exists": os.path.isfile(ct_p), "roi_exists": os.path.isfile(roi_p)})
    if not (row["ct_exists"] and row["roi_exists"]):
        row["all_ok"] = False
        return row

    ct = sitk.ReadImage(ct_p)
    mk = sitk.ReadImage(roi_p)
    row["ct_size"] = "x".join(str(v) for v in ct.GetSize())
    row["ct_spacing"] = "x".join("%.4f" % v for v in ct.GetSpacing())
    row["ct_pixel_type"] = ct.GetPixelIDTypeAsString()
    row["size_match"] = mk.GetSize() == ct.GetSize()
    row["spacing_match"] = bool(np.allclose(mk.GetSpacing(), ct.GetSpacing()))
    row["origin_match"] = bool(np.allclose(mk.GetOrigin(), ct.GetOrigin()))
    row["direction_match"] = bool(np.allclose(mk.GetDirection(), ct.GetDirection()))

    ma = sitk.GetArrayFromImage(mk)
    uniq = np.unique(ma)
    row["mask_binary"] = bool(set(uniq.tolist()) <= {0, 1})
    nvox = int((ma > 0).sum())
    row["mask_nonempty"] = nvox > 0
    row["mask_voxels"] = nvox
    row["mask_volume_mm3"] = round(nvox * float(np.prod(mk.GetSpacing())), 1)

    grid = reference_grid(ct, cfg["preprocessing"]["resample_spacing_mm"])
    row["resampled_size"] = "x".join(str(v) for v in grid["size"])

    if nvox:
        nz = np.argwhere(ma > 0)
        # extent in mm -> extent in 2 mm voxels
        ext_mm = (nz.max(0) - nz.min(0) + 1)[::-1] * np.array(ct.GetSpacing())
        ext_2mm = np.floor(ext_mm / np.array(cfg["preprocessing"]["resample_spacing_mm"]))
        row["roi_bbox_extent_ijk_2mm"] = "x".join(str(int(v)) for v in ext_2mm)
        row["bbox_fits_one_patch"] = bool(
            np.all(ext_2mm >= np.array(cfg["patches"]["size_voxels"])))
        # ROI must be inside the CT array (it shares the grid, so this is a bounds check)
        row["roi_inside_fov"] = bool(np.all(nz.min(0) >= 0) and np.all(
            nz.max(0) < np.array(ma.shape)))
    else:
        row["bbox_fits_one_patch"] = False
        row["roi_inside_fov"] = False

    if deep:
        a = sitk.GetArrayFromImage(ct)
        row["ct_hu_min"] = float(a.min())
        row["ct_hu_max"] = float(a.max())
        if nvox:
            hu = a[ma > 0]
            lo, hi = cfg["radiomics"]["resegment_range"]
            row["roi_hu_min"] = float(hu.min())
            row["roi_hu_median"] = float(np.median(hu))
            row["roi_hu_max"] = float(hu.max())
            row["frac_roi_hu_in_reseg_range"] = round(
                float(((hu >= lo) & (hu <= hi)).mean()), 4)

    row["all_ok"] = bool(row["size_match"] and row["spacing_match"]
                         and row["origin_match"] and row["direction_match"]
                         and row["mask_binary"] and row["mask_nonempty"]
                         and row["roi_inside_fov"] and row["bbox_fits_one_patch"])
    return row


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--shallow", action="store_true",
                    help="skip loading the CT array (no HU statistics)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows_in = [r for r in load_cohort(cfg) if r["eligible"]]
    if args.limit:
        rows_in = rows_in[:args.limit]

    out_rows = []
    for n, rec in enumerate(rows_in, 1):
        r = check_patient(cfg, rec, deep=not args.shallow)
        out_rows.append(r)
        if n % 20 == 0 or n == len(rows_in):
            print("  checked %d/%d" % (n, len(rows_in)))

    reports = project_path(cfg, "reports")
    os.makedirs(reports, exist_ok=True)
    path = cfg_path(cfg, "geometry_csv")
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out_rows)

    bad = [r for r in out_rows if not r["all_ok"]]
    fr = [r["frac_roi_hu_in_reseg_range"] for r in out_rows
          if r["frac_roi_hu_in_reseg_range"] != ""]
    summary = {
        "n_checked": len(out_rows),
        "n_all_ok": len(out_rows) - len(bad),
        "n_failed": len(bad),
        "failed": [r["PatientID"] for r in bad],
        "distinct_ct_spacings": sorted({r["ct_spacing"] for r in out_rows}),
        "n_distinct_ct_spacings": len({r["ct_spacing"] for r in out_rows}),
        "all_masks_binary": all(r["mask_binary"] for r in out_rows),
        "all_geometry_matched": all(r["size_match"] and r["spacing_match"]
                                    and r["origin_match"] and r["direction_match"]
                                    for r in out_rows),
        "roi_hu_in_reseg_range_min": (round(min(fr), 4) if fr else None),
        "roi_hu_in_reseg_range_mean": (round(sum(fr) / len(fr), 4) if fr else None),
    }
    with open(cfg_path(cfg, "geometry_json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print("written:", path)


if __name__ == "__main__":
    main()
