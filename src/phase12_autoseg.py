"""Phase 12 - cavity-fenced box-seeded segmentation (v5), for ALL sites.

ADOPTED: v5 = Phase-11 rule inside (box AND chest-cavity hull) on slices
where >= 50 % of the box lies in the hull (per-slice convex hull of the
air-based lung mask, grown 2 mm).  Validation on 343 expert GTVs (Dice,
mean LUNG1/RG): tight boxes v1 0.818/0.824 -> v5 0.812/0.830; boxes loosened
by 8 mm v1 0.553/0.594 -> v5 0.572/0.680.  Rejected: v2 (box AND 10 mm-closed
lung envelope; 0.393/0.722 tight) and v3 (envelope arbitration; 0.733/0.755)
- the air-based envelope misses large central tumours / collapse.

Original motivation (v2 text below):

Why: NLST Sybil boxes are loose and often include chest wall / mediastinum,
which the Phase-11 rule (HU > -750 inside the box) swallows.  v2 = the same
rule inside (box AND lung envelope), where the lung envelope is the Phase-10
lung mask closed with a ball of `closing_mm` (keeps juxtapleural tumours).

To keep the mask source identical across hospitals, v2 is applied to all
four sites (LUNG1, Radiogenomics, LPCD, NLST).  It is validated first on the
343 expert GTVs (LUNG1 + Radiogenomics; tight boxes from the GTV AND the
same boxes loosened by `loosen_mm` to mimic Sybil), no ADC/SCC label read.

Usage:
  python src/phase12_autoseg.py validate
  python src/phase12_autoseg.py run --jobs 8
Outputs: results/phase12/autoseg_v2_validation.csv,
         C:/LUNG_phase12/auto_masks_v2/<key>/{AUTO_GTV,BOX}.nii.gz + manifest
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_data import cohort as cohort10, load_cfg  # noqa: E402
from phase10_extract_loc import resample, segment_lungs  # noqa: E402
from phase11_autoseg import boxes_from_mask, dice, load_dev, segment  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THR = -750
CLOSING_MM = 10.0
LOOSEN_MM = 8.0
AUTO2 = "C:/LUNG_phase12/auto_masks_v2"


def ball(r):
    g = np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (g ** 2).sum(0) <= r * r


def lung_envelope(ct_img, closing_mm=CLOSING_MM):
    """native-grid boolean lung envelope."""
    c2 = resample(ct_img, sitk.sitkLinear, -1024.0)
    lung, _ = segment_lungs(sitk.GetArrayFromImage(c2))
    r = int(round(closing_mm / 2.0))
    lung = np.pad(lung, r)
    lung = ndi.binary_closing(lung, structure=ball(r))[r:-r, r:-r, r:-r]
    lung = np.stack([ndi.binary_fill_holes(s) for s in lung])
    li = sitk.GetImageFromArray(lung.astype(np.uint8))
    li.CopyInformation(c2)
    back = sitk.Resample(li, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(back) > 0


def cavity_hull(ct_img, grow_mm=2.0):
    """native-grid chest-cavity estimate: per axial slice, convex hull of the
    (air-based) lung mask, grown by grow_mm.  Keeps central/hilar tumours and
    collapse (between/inside the lungs), cuts lateral/posterior chest wall."""
    from skimage.morphology import convex_hull_image
    c2 = resample(ct_img, sitk.sitkLinear, -1024.0)
    lung, _ = segment_lungs(sitk.GetArrayFromImage(c2))
    hull = np.zeros_like(lung)
    for z in range(lung.shape[0]):
        if lung[z].sum() > 20:
            hull[z] = convex_hull_image(lung[z])
    r = int(round(grow_mm / 2.0))
    if r:
        hull = ndi.binary_dilation(hull, structure=ndi.generate_binary_structure(3, 1), iterations=r)
    li = sitk.GetImageFromArray(hull.astype(np.uint8))
    li.CopyInformation(c2)
    back = sitk.Resample(li, ct_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(back) > 0


def fence(box, hull, min_box_in=0.5):
    """per slice: restrict the box to the cavity hull only when most of the
    box lies inside it (else the tumour is in collapsed lung / outside the
    air-based hull and the box is kept unchanged)."""
    out = box.copy()
    for z in np.where(box.reshape(box.shape[0], -1).any(1))[0]:
        b = box[z]
        if (b & hull[z]).sum() >= min_box_in * b.sum():
            out[z] = b & hull[z]
    return out


def segment_v5(arr, box, hull):
    return segment(arr, fence(box, hull), THR)


def segment_v2(arr, box, env):
    return segment(arr, box & env, THR)


def segment_v3(arr, box, env, min_in=0.5):
    """Phase-11 rule, but the lung envelope ARBITRATES between per-slice
    components: prefer the largest component that lies mostly (>= min_in)
    inside the envelope and trim it to the envelope; if no component does
    (large central tumours / collapse, where the air-based envelope misses
    the tumour) fall back to the Phase-11 choice (largest component)."""
    out = np.zeros_like(box, dtype=bool)
    st = ndi.generate_binary_structure(2, 1)
    for z in np.where(box.reshape(box.shape[0], -1).any(1))[0]:
        cand = ndi.binary_opening((arr[z] > THR) & box[z], structure=st, iterations=1)
        lab, n = ndi.label(cand)
        if n == 0:
            continue
        vol = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        inn = np.bincount(lab.ravel(), weights=env[z].ravel(), minlength=n + 1)[1:]
        frac = inn / np.maximum(vol, 1)
        ok = np.where(frac >= min_in)[0]
        if len(ok):
            j = ok[np.argmax(inn[ok])]
            sel = (lab == j + 1) & env[z]
        else:
            sel = lab == (np.argmax(vol) + 1)
        out[z] = ndi.binary_fill_holes(sel)
    lab, n = ndi.label(out)
    if n > 1:
        vol = np.bincount(lab.ravel())[1:]
        out = np.isin(lab, np.where(vol >= 0.1 * vol.max())[0] + 1)
    return out


def loosen(box, spacing, mm):
    """grow each per-slice box in-plane by `mm` (mimic loose Sybil boxes)."""
    out = np.zeros_like(box)
    py, px = int(round(mm / spacing[1])), int(round(mm / spacing[0]))
    for z in np.where(box.reshape(box.shape[0], -1).any(1))[0]:
        ys, xs = np.nonzero(box[z])
        out[z, max(ys.min() - py, 0):ys.max() + py + 1, max(xs.min() - px, 0):xs.max() + px + 1] = True
    return out


def validate_one(cfg, key, pid):
    try:
        ct_img, ct, g = load_dev(cfg, key, pid)
        env = lung_envelope(ct_img)
        hull = cavity_hull(ct_img)
        tight = boxes_from_mask(g)
        loose = loosen(tight, ct_img.GetSpacing(), LOOSEN_MM)
        return {"PatientKey": key,
                "gtv_in_envelope": float((g & env).sum() / max(g.sum(), 1)),
                "gtv_in_hull": float((g & hull).sum() / max(g.sum(), 1)),
                "v5_tight": dice(segment_v5(ct, tight, hull), g), "v5_loose": dice(segment_v5(ct, loose, hull), g),
                "v4_tight": dice(segment(ct, tight & hull, THR), g), "v4_loose": dice(segment(ct, loose & hull, THR), g),
                "v1_tight": dice(segment(ct, tight, THR), g), 
                "v1_loose": dice(segment(ct, loose, THR), g), }
    except Exception as e:
        return {"PatientKey": key, "error": str(e)[:200]}


def run_one(key, ct_path, box_path):
    dest = os.path.join(AUTO2, key.replace("::", "__"))
    fn = os.path.join(dest, "AUTO_GTV.nii.gz")
    try:
        ct_img = sitk.ReadImage(ct_path, sitk.sitkFloat32)
        vox = float(np.prod(ct_img.GetSpacing())) / 1000.0
        if os.path.isfile(fn):
            m = sitk.GetArrayFromImage(sitk.ReadImage(fn)) > 0
            return {"PatientKey": key, "status": "ok", "ct_path": ct_path, "auto_volume_ml": m.sum() * vox}
        os.makedirs(dest, exist_ok=True)
        box = sitk.GetArrayFromImage(sitk.ReadImage(box_path)) > 0
        m = segment_v5(sitk.GetArrayFromImage(ct_img), box, cavity_hull(ct_img))
        if m.sum() == 0:
            return {"PatientKey": key, "status": "failed: empty mask"}
        for name, a in (("BOX", box), ("AUTO_GTV", m)):
            im = sitk.GetImageFromArray(a.astype(np.uint8))
            im.CopyInformation(ct_img)
            sitk.WriteImage(im, os.path.join(dest, name + ".nii.gz"))
        return {"PatientKey": key, "status": "ok", "ct_path": ct_path, "auto_volume_ml": m.sum() * vox,
                "box_volume_ml": box.sum() * vox}
    except Exception as e:
        return {"PatientKey": key, "status": "failed: %s" % str(e)[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["validate", "run"])
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", "phase12")
    os.makedirs(out, exist_ok=True)
    if a.step == "validate":
        cfg = load_cfg()
        c = cohort10(cfg)
        rows = Parallel(n_jobs=a.jobs, verbose=2)(delayed(validate_one)(cfg, k, p)
                                                   for k, p in zip(c.PatientKey, c.PatientID))
        df = pd.DataFrame(rows)
        df["site"] = df.PatientKey.str.split("::").str[0]
        df.to_csv(os.path.join(out, "autoseg_v2_validation.csv"), index=False)
        print(df.groupby("site")[[c for c in df.columns if c.startswith(("gtv_", "v1", "v4", "v5"))]]
              .agg(["mean", "median"]).round(3).T.to_string())
        return
    coh = pd.read_csv(os.path.join(ROOT, "cohort", "phase12_four_site_cohort.csv"))
    box_paths = []
    for k, pid in zip(coh.PatientKey, coh.PatientID):
        if k.startswith("NLST::"):
            box_paths.append(os.path.join("C:/LUNG_phase12/nlst/nifti", str(pid), "BOX.nii.gz"))
        else:
            box_paths.append(os.path.join("C:/LUNG_phase11/auto_masks", k.replace("::", "__"), "BOX.nii.gz"))
    rows = Parallel(n_jobs=a.jobs, verbose=2)(delayed(run_one)(k, c, b)
                                               for k, c, b in zip(coh.PatientKey, coh.ct_path, box_paths))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(AUTO2, "autoseg_manifest.csv"), index=False)
    print(df.status.value_counts().to_dict())


if __name__ == "__main__":
    main()
