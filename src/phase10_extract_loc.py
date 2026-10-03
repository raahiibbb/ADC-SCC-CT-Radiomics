"""Phase 10 - tumour LOCATION features ("where is the tumour in the lung").

Clinical prior: squamous cell carcinoma tends to arise centrally (main /
lobar bronchi, near the hilum); adenocarcinoma tends to be peripheral.

Lungs are segmented by ONE deterministic threshold method applied identically
to both cohorts (a model-based segmenter would add a second acquisition
dependency, and a cohort-specific mask source would be a site confound).

All geometry is computed on a 2 mm isotropic grid in LPS physical space
(+x = patient left, +y = posterior, +z = superior).  No label is read.
Resumable: patients already in the output CSV are skipped.

Usage:  python src/phase10_extract_loc.py [--jobs 8]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
from joblib import Parallel, delayed
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_data import cohort, load_cfg, p  # noqa: E402

SPACING = 2.0


def paths(cfg, key, pid):
    root = p(cfg, "lung1_image_root") if key.startswith("LUNG1") else p(cfg, "external_image_root")
    return os.path.join(root, pid, "CT.nii.gz"), os.path.join(root, pid, "ROI_A_GTV.nii.gz")


def resample(img, interp, default):
    sp, sz = img.GetSpacing(), img.GetSize()
    new_sz = [int(round(sz[i] * sp[i] / SPACING)) for i in range(3)]
    return sitk.Resample(img, new_sz, sitk.Transform(), interp, img.GetOrigin(),
                         (SPACING,) * 3, img.GetDirection(), default, img.GetPixelID())


def segment_lungs(hu):
    """hu: (z, y, x) array on the 2 mm grid -> boolean lung mask (holes filled)."""
    body = hu > -500
    body = ndi.binary_opening(body, iterations=1)
    lab, n = ndi.label(body)
    if n == 0:
        raise RuntimeError("no body")
    body = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    body = np.stack([ndi.binary_fill_holes(s) for s in body])
    air = (hu < -400) & body
    air = ndi.binary_opening(air, iterations=1)
    lab, n = ndi.label(air)
    if n == 0:
        raise RuntimeError("no lung air")
    vol = np.bincount(lab.ravel())[1:]
    order = np.argsort(-vol)
    keep = [order[0] + 1]
    if n > 1 and vol[order[1]] > 0.1 * vol[order[0]]:
        keep.append(order[1] + 1)
    lung = np.isin(lab, keep)
    lung = ndi.binary_closing(lung, structure=ndi.generate_binary_structure(3, 1), iterations=3)
    lung = np.stack([ndi.binary_fill_holes(s) for s in lung])
    return lung, body


def features(cfg, key, pid):
    ct_p, gtv_p = paths(cfg, key, pid)
    ct = sitk.ReadImage(ct_p, sitk.sitkFloat32)
    gtv = sitk.ReadImage(gtv_p, sitk.sitkUInt8)
    ct2 = resample(ct, sitk.sitkLinear, -1024.0)
    g2 = sitk.Resample(gtv, ct2, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    d = np.array(ct2.GetDirection()).reshape(3, 3)
    if not np.allclose(np.abs(d), np.eye(3), atol=1e-3):
        raise RuntimeError("non-axis-aligned direction")
    hu = sitk.GetArrayFromImage(ct2)          # (z, y, x)
    g = sitk.GetArrayFromImage(g2) > 0
    if g.sum() == 0:
        raise RuntimeError("empty GTV after resampling")
    lung, body = segment_lungs(hu)
    region = lung | g                           # tumour counts as lung parenchyma

    # voxel index -> physical LPS mm (axis-aligned, possibly flipped)
    org, sgn = np.array(ct2.GetOrigin()), np.sign(np.diag(d))

    def phys(idx_zyx):
        i = np.asarray(idx_zyx, float)[..., ::-1]           # -> (x, y, z)
        return org + sgn * i * SPACING

    gz, gy, gx = np.nonzero(g)
    c_idx = np.array([gz.mean(), gy.mean(), gx.mean()])
    c = phys(c_idx)
    bz, by, bx = np.nonzero(body)
    bxs = phys(np.stack([np.zeros_like(bx), np.zeros_like(bx), bx], 1))[:, 0]
    midline = 0.5 * (bxs.min() + bxs.max())
    half_w = 0.5 * (bxs.max() - bxs.min())
    left = c[0] > midline

    # tumour-side parenchyma
    lz, ly, lx = np.nonzero(region)
    lp = phys(np.stack([lz, ly, lx], 1))
    side = (lp[:, 0] > midline) if left else (lp[:, 0] <= midline)
    lp = lp[side]
    zlo, zhi = phys([[gz.min(), 0, 0]])[0, 2], phys([[gz.max(), 0, 0]])[0, 2]
    zlo, zhi = min(zlo, zhi), max(zlo, zhi)
    band = lp[(lp[:, 2] >= zlo) & (lp[:, 2] <= zhi)]
    xs = band[:, 0]
    x_med = xs.min() if left else xs.max()      # the edge nearest the midline
    x_lat = xs.max() if left else xs.min()
    rel_lat = np.clip(abs(c[0] - x_med) / max(abs(x_lat - x_med), 1e-3), 0, 1)
    ys = band[:, 1]
    rel_ap = np.clip((c[1] - ys.min()) / max(ys.max() - ys.min(), 1e-3), 0, 1)
    zs = lp[:, 2]
    rel_z = np.clip((c[2] - zs.min()) / max(zs.max() - zs.min(), 1e-3), 0, 1)

    # distance (mm) from each parenchyma voxel to the nearest non-parenchyma voxel
    edt = ndi.distance_transform_edt(region) * SPACING
    ci = np.round(c_idx).astype(int)
    centroid_depth = float(edt[tuple(ci)]) if region[tuple(ci)] else 0.0
    # depth normalised by the deepest point of that lung (0 = at a boundary)
    side_mask = np.zeros_like(region)
    xphys = phys(np.stack([np.zeros(hu.shape[2]), np.zeros(hu.shape[2]), np.arange(hu.shape[2])], 1))[:, 0]
    side_cols = (xphys > midline) if left else (xphys <= midline)
    side_mask[:, :, side_cols] = region[:, :, side_cols]
    max_depth = float(edt[side_mask].max()) if side_mask.any() else 1.0

    # contact of the tumour with the mediastinum vs the chest wall: tumour
    # boundary voxels whose outside neighbour lies medial vs lateral
    shell = ndi.binary_dilation(g, iterations=1) & ~region
    sz_, sy_, sx_ = np.nonzero(shell)
    if len(sx_):
        sp_ = phys(np.stack([sz_, sy_, sx_], 1))[:, 0]
        medial = (np.abs(sp_ - midline) < abs(c[0] - midline))
        frac_med = float(medial.mean())
        contact = float(len(sx_)) / max(float((ndi.binary_dilation(g) & ~g).sum()), 1.0)
    else:
        frac_med, contact = 0.0, 0.0

    return dict(PatientKey=key, status="ok",
                loc__left=float(left),
                loc__rel_lateral=float(rel_lat),
                loc__rel_ap=float(rel_ap),
                loc__rel_z=float(rel_z),
                loc__midline_dist_norm=float(abs(c[0] - midline) / max(half_w, 1e-3)),
                loc__centroid_depth_mm=centroid_depth,
                loc__centroid_depth_rel=centroid_depth / max(max_depth, 1e-3),
                loc__boundary_contact_frac=contact,
                loc__contact_medial_frac=frac_med,
                loc__gtv_outside_lung_frac=float((g & ~lung).sum() / g.sum()),
                loc__log_gtv_volume=float(np.log(g.sum() * SPACING ** 3)))


def safe(cfg, key, pid):
    try:
        return features(cfg, key, pid)
    except Exception as e:  # recorded, never silently dropped
        return dict(PatientKey=key, status="failed: %s" % e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = load_cfg()
    out = os.path.join(p(cfg, "extra_features_dir"), "loc.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    coh = cohort(cfg)
    done = set()
    if os.path.isfile(out):
        prev = pd.read_csv(out)
        done = set(prev.loc[prev["status"] == "ok", "PatientKey"])
    todo = coh[~coh["PatientKey"].isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    t0 = time.time()
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(safe)(cfg, k, i)
                                              for k, i in zip(todo["PatientKey"], todo["PatientID"]))
    new = pd.DataFrame(rows)
    if os.path.isfile(out):
        prev = pd.read_csv(out)
        new = pd.concat([prev[~prev["PatientKey"].isin(new["PatientKey"])], new], ignore_index=True)
    new.to_csv(out, index=False)
    print("done %d in %.0fs; failures: %d" % (len(rows), time.time() - t0,
                                             int((new["status"] != "ok").sum())))


if __name__ == "__main__":
    main()
