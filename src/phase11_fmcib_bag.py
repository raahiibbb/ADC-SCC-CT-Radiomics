"""Phase 11 - multi-crop FMCIB bags (local foundation-model MIL instances).

Per patient: seeds = AUTO_GTV centroid + (K-1) points on the AUTO_GTV boundary
chosen by farthest-point sampling (deterministic, spread over the whole
tumour surface).  Each seed -> 50 mm FMCIB crop -> frozen 4096-d embedding.
Output: one .npy bag per patient (K x 4096, float16) + seed coordinates.
Pooled summaries (mean, max) are written as feature blocks for the CV code.

No label is read.  Resumable.   Usage: python src/phase11_fmcib_bag.py [--k 16]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase10_extract_fmcib as F  # noqa: E402
from phase11_features import OUT, ROOT, mask_path  # noqa: E402

BAG_DIR = os.path.join(OUT, "fmcib_bags")


def seeds(mask_img, k):
    m = sitk.GetArrayFromImage(mask_img) > 0                 # (z, y, x)
    boundary = m & ~ndi.binary_erosion(m)
    idx = np.argwhere(boundary)
    phys = np.array([mask_img.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in i[::-1]))
                     for i in idx])
    st = sitk.LabelShapeStatisticsImageFilter()
    st.Execute(mask_img > 0)
    pts = [np.array(st.GetCentroid(1))]
    if len(phys):
        d = np.linalg.norm(phys - pts[0], axis=1)
        for _ in range(k - 1):
            j = int(np.argmax(d))
            pts.append(phys[j])
            d = np.minimum(d, np.linalg.norm(phys - phys[j], axis=1))
    while len(pts) < k:                                       # tiny tumours
        pts.append(pts[0])
    return np.array(pts)


def crop(ct, seed):
    ref = sitk.Image([F.SIZE] * 3, sitk.sitkFloat32)
    ref.SetSpacing((1.0, 1.0, 1.0))
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    ref.SetOrigin(tuple(seed - F.SIZE // 2))
    a = sitk.GetArrayFromImage(sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear, -1024.0, sitk.sitkFloat32))
    return (a + 1024.0) / 3072.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()
    os.makedirs(BAG_DIR, exist_ok=True)
    coh = pd.read_csv(os.path.join(ROOT, "cohort", "phase11_three_site_cohort.csv"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = F.load_model(dev)
    t0, fails = time.time(), []
    for i, (key, ctp) in enumerate(zip(coh.PatientKey, coh.ct_path)):
        fn = os.path.join(BAG_DIR, key.replace("::", "__") + ".npy")
        if os.path.isfile(fn):
            continue
        try:
            ct = sitk.ReadImage(ctp, sitk.sitkFloat32)
            sd = seeds(sitk.ReadImage(mask_path(key), sitk.sitkUInt8), a.k)
            x = np.stack([crop(ct, s) for s in sd])[:, None]
            with torch.no_grad():
                f = model(torch.from_numpy(x).float().to(dev)).cpu().numpy()
            np.save(fn, f.astype(np.float16))
            np.save(fn.replace(".npy", "_seeds.npy"), sd)
        except Exception as e:
            fails.append((key, str(e)[:200]))
        if (i + 1) % 50 == 0:
            print("%d/%d %.0fs" % (i + 1, len(coh), time.time() - t0), flush=True)
    # pooled summary blocks
    rows = []
    for key in coh.PatientKey:
        fn = os.path.join(BAG_DIR, key.replace("::", "__") + ".npy")
        if not os.path.isfile(fn):
            rows.append({"PatientKey": key, "status": "failed"})
            continue
        b = np.load(fn).astype(np.float32)
        r = {"PatientKey": key, "status": "ok"}
        r.update({"fbmean__f%04d" % j: float(v) for j, v in enumerate(b.mean(0))})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "fmcib_bagmean.csv"), index=False)
    print("done in %.0fs; failures %d %s" % (time.time() - t0, len(fails), fails[:3]))


if __name__ == "__main__":
    main()
