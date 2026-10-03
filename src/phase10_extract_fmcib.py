"""Phase 10 - FMCIB foundation-model features (Pai et al., Nat Mach Intell 2024).

Frozen, externally pretrained 3D ResNet50 (MONAI, widen_factor 2) with the
official weights (Zenodo record 10528450), loaded with torch.load(...,
weights_only=True) - tensors only, no pickled code.

Preprocessing reproduces the official FMCIB pipeline:
  seed = GTV centroid (physical LPS mm)
  CT resampled (linear) to 1 mm isotropic, LPS, 50x50x50 mm cube around seed
  intensity (HU + 1024) / 3072, array order (z, y, x)
Output: 4096-d feature vector per patient.  No label is read.  Resumable.

Usage:  python src/phase10_extract_fmcib.py
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from monai.networks.nets import resnet50

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase10_data import ROOT, cohort, load_cfg, p  # noqa: E402

WEIGHTS = os.path.join(ROOT, "models", "phase10", "fmcib_model_weights.torch")
WEIGHTS_SHA256 = "8cc92ec452d1099a6de6bbfdb9e124c79d750e513a86ce6be6d91d187aca4bc2"
SIZE = 50


def load_model(device):
    h = hashlib.sha256(open(WEIGHTS, "rb").read()).hexdigest()
    if h != WEIGHTS_SHA256:
        raise RuntimeError("FMCIB weight file hash mismatch: %s" % h)
    m = resnet50(pretrained=False, n_input_channels=1, widen_factor=2,
                 conv1_t_stride=2, feed_forward=False, bias_downsample=True)
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=True)["trunk_state_dict"]
    missing, unexpected = m.load_state_dict(sd, strict=True), None
    return m.to(device).eval()


def crop(cfg, key, pid):
    root = p(cfg, "lung1_image_root") if key.startswith("LUNG1") else p(cfg, "external_image_root")
    ct = sitk.ReadImage(os.path.join(root, pid, "CT.nii.gz"), sitk.sitkFloat32)
    gtv = sitk.ReadImage(os.path.join(root, pid, "ROI_A_GTV.nii.gz"), sitk.sitkUInt8)
    st = sitk.LabelShapeStatisticsImageFilter()
    st.Execute(gtv > 0)
    seed = np.array(st.GetCentroid(1))                       # physical LPS mm
    ref = sitk.Image([SIZE] * 3, sitk.sitkFloat32)
    ref.SetSpacing((1.0, 1.0, 1.0))
    ref.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))            # LPS
    ref.SetOrigin(tuple(seed - SIZE // 2))
    patch = sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear, -1024.0, sitk.sitkFloat32)
    arr = sitk.GetArrayFromImage(patch)                      # (z, y, x)
    return (arr + 1024.0) / 3072.0, seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = load_cfg()
    out = os.path.join(p(cfg, "extra_features_dir"), "fmcib.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    coh = cohort(cfg)
    prev = pd.read_csv(out) if os.path.isfile(out) else None
    done = set(prev.loc[prev["status"] == "ok", "PatientKey"]) if prev is not None else set()
    todo = coh[~coh["PatientKey"].isin(done)]
    if a.limit:
        todo = todo.head(a.limit)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(dev)
    rows, t0 = [], time.time()
    for i, (key, pid) in enumerate(zip(todo["PatientKey"], todo["PatientID"])):
        try:
            x, seed = crop(cfg, key, pid)
            with torch.no_grad():
                f = model(torch.from_numpy(x[None, None]).float().to(dev)).cpu().numpy()[0]
            row = {"PatientKey": key, "status": "ok",
                   "seed_x": seed[0], "seed_y": seed[1], "seed_z": seed[2]}
            row.update({"fmcib__f%04d" % j: float(v) for j, v in enumerate(f)})
        except Exception as e:
            row = {"PatientKey": key, "status": "failed: %s" % e}
        rows.append(row)
        if (i + 1) % 25 == 0:
            print("%d/%d  %.0fs" % (i + 1, len(todo), time.time() - t0), flush=True)
    new = pd.DataFrame(rows)
    if prev is not None:
        new = pd.concat([prev[~prev["PatientKey"].isin(new["PatientKey"])], new], ignore_index=True)
    new.to_csv(out, index=False)
    print("done %d in %.0fs on %s; failures %d" % (len(rows), time.time() - t0, dev,
                                                  int((new["status"] != "ok").sum())))


if __name__ == "__main__":
    main()
