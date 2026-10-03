"""Phase 13b - CT-FM foundation-model embeddings of the tumour region.

CT-FM (Pai et al. 2025, project-lighter/ct_fm_feature_extractor, Apache-2.0;
SegResEncoder, 77M params, intra-sample SimCLR on 148k IDC CTs).  Weights are
safetensors (no pickle).  The crop reproduces the PRETRAINING geometry
(experiments/fm/base.yaml + frameworks/intrasample_simclr.yaml):
  orientation SPL (== SimpleITK identity-direction array order z,y,x in LPS),
  spacing 3 x 1 x 1 mm (z, y, x), patch 64 x 128 x 128 mm -> 32 x 128 x 128 vox,
  HU clipped to [-1024, 2048] and scaled to [0, 1].
The crop is centred on the tumour-mask centroid (mask root given by
--mask-root; default = the Phase-13a MedSAM2 masks).  Output = global average
of the last encoder stage (512-d), plus a tumour-weighted pooling variant
(average over feature-map cells overlapping the downsampled tumour mask).

No label is read.  Resumable.   Usage (.venv-medsam2):
  python src/phase13_ctfm.py [--mask-root C:/LUNG_phase13/medsam2_masks]
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZE_XYZ = (128, 128, 32)
SPACING_XYZ = (1.0, 1.0, 3.0)
OUT = os.path.join(ROOT, "global_features", "phase13")


def crop(img, centre, interp=sitk.sitkLinear, default=-1024.0, pix=sitk.sitkFloat32):
    ref = sitk.Image(list(SIZE_XYZ), pix)
    ref.SetSpacing(SPACING_XYZ)
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    ref.SetOrigin(tuple(np.asarray(centre) - (np.asarray(SIZE_XYZ) - 1) * np.asarray(SPACING_XYZ) / 2.0))
    return sitk.GetArrayFromImage(sitk.Resample(img, ref, sitk.Transform(), interp, default, pix))  # (z, y, x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-root", default="C:/LUNG_phase13/medsam2_masks")
    ap.add_argument("--name", default="ctfm")
    a = ap.parse_args()
    from lighter_zoo import SegResEncoder
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor").to(dev).eval()
    coh = pd.read_csv(os.path.join(ROOT, "cohort", "phase12_four_site_cohort.csv"))
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, a.name + ".csv")
    prev = pd.read_csv(fn) if os.path.isfile(fn) else None
    done = set(prev.loc[prev.status == "ok", "PatientKey"]) if prev is not None else set()
    rows, t0 = [], time.time()
    for i, (k, ctp) in enumerate(zip(coh.PatientKey, coh.ct_path)):
        if k in done:
            continue
        try:
            img = sitk.ReadImage(ctp, sitk.sitkFloat32)
            m = sitk.ReadImage(os.path.join(a.mask_root, k.replace("::", "__"), "AUTO_GTV.nii.gz"), sitk.sitkUInt8)
            st = sitk.LabelShapeStatisticsImageFilter()
            st.Execute(m > 0)
            c = st.GetCentroid(1)
            x = crop(img, c)
            mk = crop(m, c, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8) > 0
            x = (np.clip(x, -1024.0, 2048.0) + 1024.0) / 3072.0
            with torch.no_grad():
                fm = model(torch.from_numpy(x[None, None]).float().to(dev))[-1]      # (1, 512, z', y', x')
                g = F.adaptive_avg_pool3d(fm, 1).flatten().cpu().numpy()
                mm = F.adaptive_max_pool3d(torch.from_numpy(mk[None, None].astype(np.float32)).to(dev),
                                           fm.shape[2:])                              # tumour cells
                w = mm / mm.sum().clamp(min=1.0)
                t = (fm * w).sum((2, 3, 4)).flatten().cpu().numpy() if mm.sum() > 0 else g
            r = {"PatientKey": k, "status": "ok"}
            r.update({f"{a.name}__g{j:03d}": float(v) for j, v in enumerate(g)})
            r.update({f"{a.name}t__t{j:03d}": float(v) for j, v in enumerate(t)})
        except Exception as e:
            r = {"PatientKey": k, "status": "failed: %s" % str(e)[:200]}
        rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(coh)} {time.time() - t0:.0f}s", flush=True)
    new = pd.DataFrame(rows)
    if prev is not None:
        new = pd.concat([prev[~prev.PatientKey.isin(new.PatientKey)], new], ignore_index=True)
    new.to_csv(fn, index=False)
    print(a.name, len(new), "rows, failures", int((new.status != "ok").sum()), "feature map", tuple(fm.shape))


if __name__ == "__main__":
    main()
