"""Phase 12 - DICOM -> NIfTI, Sybil boxes -> BOX mask -> AUTO_GTV (NLST);
DICOM -> NIfTI only for the locked TCGA test (no label read, no mask yet).

NLST lesion choice: the per-slice Sybil boxes are grouped into 3-D lesions
(connected components of the box mask); the LARGEST lesion is kept and the
number of lesions is recorded (sensitivity analysis: single-lesion patients).
AUTO_GTV = the frozen Phase-11 box-seeded segmentation (-750 HU).

Resumable (skips patients whose record.json says ok).
Usage: python src/phase12_build.py nlst|tcga [--jobs 8]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from joblib import Parallel, delayed
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase11_autoseg import segment  # noqa: E402
from phase12_select import cfg_load  # noqa: E402

CFG = cfg_load()
R = CFG["root"]


def read_series(folder):
    files = glob.glob(os.path.join(folder, "*"))
    ds = []
    for f in files:
        try:
            d = pydicom.dcmread(f, stop_before_pixels=True)
            if "ImagePositionPatient" not in d:
                continue
            ds.append((float(d.ImagePositionPatient[2]), f, str(d.SOPInstanceUID), d))
        except Exception:
            continue
    ds.sort(key=lambda t: t[0])
    seen, uniq = set(), []
    for t in ds:
        k = round(t[0], 3)
        if k not in seen:
            seen.add(k)
            uniq.append(t)
    if len(uniq) < 10:
        raise RuntimeError("only %d usable slices" % len(uniq))
    rd = sitk.ImageSeriesReader()
    rd.SetFileNames([t[1] for t in uniq])
    img = rd.Execute()
    d0 = uniq[len(uniq) // 2][3]
    zs = np.array([t[0] for t in uniq])
    info = dict(n_slices=len(uniq), n_files=len(files), spacing=list(img.GetSpacing()), size=list(img.GetSize()),
                slice_spacing_median=float(np.median(np.diff(zs))),
                **{k: str(getattr(d0, k, "")) for k in ("Manufacturer", "ManufacturerModelName", "ConvolutionKernel",
                                                         "SliceThickness", "KVP", "ContrastBolusAgent",
                                                         "PatientAge", "PatientSex")})
    return img, [t[2] for t in uniq], info


def nlst_one(row, boxes):
    pid = str(row.PatientID)
    dest = os.path.join(R, "nlst", "nifti", pid)
    rec_p = os.path.join(dest, "record.json")
    if os.path.isfile(rec_p) and json.load(open(rec_p)).get("status") == "ok":
        return json.load(open(rec_p))
    os.makedirs(dest, exist_ok=True)
    try:
        img, sops, info = read_series(os.path.join(R, "nlst", "dicom", row.CTSeriesInstanceUID))
        zi = {s: i for i, s in enumerate(sops)}
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        box = np.zeros(arr.shape, bool)
        nb = 0
        for b in boxes.itertuples():
            if b.SOPInstanceUID in zi:
                x0, y0 = int(np.floor(b.x0)), int(np.floor(b.y0))
                x1, y1 = int(np.ceil(b.x1)), int(np.ceil(b.y1))
                box[zi[b.SOPInstanceUID], max(y0, 0):y1, max(x0, 0):x1] = True
                nb += 1
        if nb == 0:
            raise RuntimeError("no box matched a slice")
        lab, n_les = ndi.label(box)
        vol = np.bincount(lab.ravel())[1:]
        box = lab == (np.argmax(vol) + 1)
        m = segment(arr, box, CFG["autoseg"]["threshold_hu"])
        if m.sum() == 0:
            raise RuntimeError("empty AUTO_GTV")
        sitk.WriteImage(img, os.path.join(dest, "CT.nii.gz"))
        for name, a in (("BOX", box), ("AUTO_GTV", m)):
            im = sitk.GetImageFromArray(a.astype(np.uint8))
            im.CopyInformation(img)
            sitk.WriteImage(im, os.path.join(dest, name + ".nii.gz"))
        vox = float(np.prod(img.GetSpacing())) / 1000.0
        rec = dict(status="ok", PatientID=pid, SeriesInstanceUID=row.CTSeriesInstanceUID, n_boxes_matched=nb,
                   n_boxes_sr=len(boxes), n_lesions=int(n_les), auto_volume_ml=float(m.sum() * vox),
                   box_volume_ml=float(box.sum() * vox), **info)
    except Exception as e:
        rec = dict(status="failed", PatientID=pid, error=str(e)[:300])
    json.dump(rec, open(rec_p, "w"), indent=1)
    return rec


def tcga_one(row):
    pid = str(row.PatientID)
    dest = os.path.join(R, "tcga", "nifti", pid)
    rec_p = os.path.join(dest, "record.json")
    if os.path.isfile(rec_p) and json.load(open(rec_p)).get("status") == "ok":
        return json.load(open(rec_p))
    os.makedirs(dest, exist_ok=True)
    try:
        img, _, info = read_series(os.path.join(R, "tcga", "dicom", row.SeriesInstanceUID))
        sitk.WriteImage(img, os.path.join(dest, "CT.nii.gz"))
        rec = dict(status="ok", PatientID=pid, SeriesInstanceUID=row.SeriesInstanceUID, **info)
    except Exception as e:
        rec = dict(status="failed", PatientID=pid, error=str(e)[:300])
    json.dump(rec, open(rec_p, "w"), indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site", choices=["nlst", "tcga"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if a.site == "nlst":
        sel = pd.read_csv(os.path.join(R, "nlst_selection.csv"), dtype={"PatientID": str})
        bx = pd.read_parquet(os.path.join(R, "nlst_boxes.parquet"))
        bx = {k: g for k, g in bx.groupby("CTSeriesInstanceUID")}
        recs = Parallel(n_jobs=a.jobs, verbose=2)(
            delayed(nlst_one)(r, bx[r.CTSeriesInstanceUID]) for r in sel.itertuples())
    else:
        sel = pd.read_csv(os.path.join(R, "tcga_selection.csv"))
        recs = Parallel(n_jobs=a.jobs, verbose=2)(delayed(tcga_one)(r) for r in sel.itertuples())
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(R, f"{a.site}_build_manifest.csv"), index=False)
    print(df.status.value_counts().to_dict())
    if (df.status != "ok").any():
        print(df[df.status != "ok"][["PatientID", "error"]].to_string())


if __name__ == "__main__":
    main()
