"""Phase 10 - PERITUMORAL multi-shell radiomics.

Three concentric shells OUTSIDE the GTV, by exact Euclidean distance on the
2 mm isotropic grid:  (0, 4], (4, 8], (8, 12] mm, restricted to the body.
PyRadiomics, ORIGINAL image only (least acquisition-sensitive), the project's
frozen intensity settings: resegment [-1024, 200] HU, binWidth 20 HU, no shape.

Runs in the PyRadiomics environment (.venv).  No label is read.  Resumable.

Usage:  .venv/Scripts/python.exe src/phase10_extract_shells.py --jobs 12
"""
from __future__ import annotations

import argparse
import logging
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
from phase10_extract_loc import SPACING, paths, resample, segment_lungs  # noqa: E402

SHELLS_MM = [(0.0, 4.0), (4.0, 8.0), (8.0, 12.0)]
PARAMS = {
    "imageType": {"Original": {}},
    "featureClass": {"firstorder": [], "glcm": [], "glrlm": [], "glszm": [], "gldm": [], "ngtdm": []},
    "setting": {"resegmentRange": [-1024, 200], "resegmentMode": "absolute", "binWidth": 20,
                "label": 1, "preCrop": True, "padDistance": 2, "geometryTolerance": 1e-4,
                "correctMask": False, "normalize": False, "force2D": False,
                "minimumROIDimensions": 2, "distances": [1], "symmetricalGLCM": True},
}


def extract(cfg, key, pid):
    from radiomics import featureextractor
    logging.getLogger("radiomics").setLevel(logging.ERROR)
    ext = featureextractor.RadiomicsFeatureExtractor(PARAMS)
    ct_p, gtv_p = paths(cfg, key, pid)
    ct = resample(sitk.ReadImage(ct_p, sitk.sitkFloat32), sitk.sitkLinear, -1024.0)
    g = sitk.Resample(sitk.ReadImage(gtv_p, sitk.sitkUInt8), ct, sitk.Transform(),
                      sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    ga = sitk.GetArrayFromImage(g) > 0
    _, body = segment_lungs(sitk.GetArrayFromImage(ct))
    dist = ndi.distance_transform_edt(~ga) * SPACING
    row = {"PatientKey": key, "status": "ok"}
    for lo, hi in SHELLS_MM:
        shell = (dist > lo) & (dist <= hi) & body
        m = sitk.GetImageFromArray(shell.astype(np.uint8))
        m.CopyInformation(ct)
        tag = "shell%02d_%02d" % (lo, hi)
        res = ext.execute(ct, m, label=1)
        for k, v in res.items():
            if not k.startswith("diagnostics"):
                row["peri__%s__%s" % (tag, k)] = float(v)
        row["peri__%s__voxels" % tag] = int(shell.sum())
    return row


def safe(cfg, key, pid):
    try:
        return extract(cfg, key, pid)
    except Exception as e:
        return {"PatientKey": key, "status": "failed: %s" % e}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = load_cfg()
    out = os.path.join(p(cfg, "extra_features_dir"), "peri.csv")
    coh = cohort(cfg)
    prev = pd.read_csv(out) if os.path.isfile(out) else None
    done = set(prev.loc[prev["status"] == "ok", "PatientKey"]) if prev is not None else set()
    todo = coh[~coh["PatientKey"].isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    t0 = time.time()
    rows = Parallel(n_jobs=a.jobs, verbose=5)(delayed(safe)(cfg, k, i)
                                              for k, i in zip(todo["PatientKey"], todo["PatientID"]))
    new = pd.DataFrame(rows)
    if prev is not None:
        new = pd.concat([prev[~prev["PatientKey"].isin(new["PatientKey"])], new], ignore_index=True)
    new.to_csv(out, index=False)
    print("done %d in %.0fs; failures %d" % (len(rows), time.time() - t0,
                                            int((new["status"] != "ok").sum())))


if __name__ == "__main__":
    main()
