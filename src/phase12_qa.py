"""Phase 12 - NLST auto-segmentation QA (no label read).

Gallery of 12 patients (v5 masks) spread over the AUTO_GTV / BOX volume-ratio range
(lowest ratios = ground-glass / sub-solid nodules where the frozen -750 HU
threshold keeps only the solid part).  Also records, per patient, the share
of box voxels in the ground-glass range (-750, -300] HU.

Usage: python src/phase12_qa.py
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import SimpleITK as sitk  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NIFTI = "C:/LUNG_phase12/nlst/nifti"
OUT = os.path.join(ROOT, "results", "phase12")


AUTO2 = os.environ.get("ADC_MASK_ROOT", "C:/LUNG_phase12/auto_masks_v2")
TAG = os.environ.get("ADC_QA_TAG", "")


def load(pid, mask="v5"):
    d = os.path.join(NIFTI, pid)
    ct, bx = [sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(d, f + ".nii.gz"))) for f in ("CT", "BOX")]
    mp = os.path.join(AUTO2, "NLST__" + pid, "AUTO_GTV.nii.gz") if mask == "v5" else os.path.join(d, "AUTO_GTV.nii.gz")
    return ct, bx, sitk.GetArrayFromImage(sitk.ReadImage(mp))


def main():
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    m = pd.read_csv("C:/LUNG_phase12/nlst_build_manifest.csv", dtype={"PatientID": str})
    m = m[m.status == "ok"].copy()
    v5 = pd.read_csv(os.path.join(AUTO2, "autoseg_manifest.csv"))
    v5 = v5[v5.PatientKey.str.startswith("NLST::")].assign(PatientID=lambda d: d.PatientKey.str[6:])
    m = m.drop(columns=["auto_volume_ml"]).merge(v5[["PatientID", "auto_volume_ml"]], on="PatientID")
    m["ratio"] = m.auto_volume_ml / m.box_volume_ml
    ggo = []
    for pid in m.PatientID:
        ct, bx, _ = load(pid)
        v = ct[bx > 0]
        ggo.append(float(((v > -750) & (v <= -300)).mean()))
    m["box_ggo_fraction"] = ggo
    m[["PatientID", "n_lesions", "auto_volume_ml", "box_volume_ml", "ratio", "box_ggo_fraction",
       "slice_spacing_median", "ConvolutionKernel", "Manufacturer"]].to_csv(
        os.path.join(OUT, f"nlst_autoseg_qc{TAG}.csv"), index=False)
    pick = m.sort_values("ratio").iloc[np.linspace(0, len(m) - 1, 12).astype(int)]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9.4))
    for ax, r in zip(axes.ravel(), pick.itertuples()):
        ct, bx, g = load(r.PatientID)
        z = int(np.argmax(g.reshape(g.shape[0], -1).sum(1)))
        ys, xs = np.nonzero(bx[z])
        pad = 25
        y0, y1 = max(ys.min() - pad, 0), ys.max() + pad
        x0, x1 = max(xs.min() - pad, 0), xs.max() + pad
        ax.imshow(ct[z, y0:y1, x0:x1], cmap="gray", vmin=-1200, vmax=300)
        ax.contour(bx[z, y0:y1, x0:x1], colors="#f5c518", linewidths=1)
        ax.contour(g[z, y0:y1, x0:x1], colors="#eb3434", linewidths=1.2)
        ax.set_title(f"ratio {r.ratio:.2f} | GGO {r.box_ggo_fraction:.2f} | {r.auto_volume_ml:.1f} ml", fontsize=8)
        ax.axis("off")
    fig.suptitle(f"NLST: expert Sybil box (yellow) -> auto-segmentation{TAG} (red), "
                 "sorted by AUTO/BOX volume ratio", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures", f"nlst_autoseg_gallery{TAG}.png"), dpi=110)
    print(m[["ratio", "box_ggo_fraction"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
