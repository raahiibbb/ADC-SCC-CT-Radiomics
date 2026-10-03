"""Phase 13a - box-prompted foundation-model tumour segmentation (MedSAM2).

MedSAM2 (Ma et al. 2025, Apache-2.0; SAM2.1-tiny backbone fine-tuned on
medical images/videos incl. CT lesions) is prompted with the SAME per-slice
boxes that seeded the Phase-11/12 threshold segmentation:
  LUNG1 / Radiogenomics : tight per-slice boxes of the expert GTV
  LPCD                  : PASCAL-VOC boxes
  NLST                  : Sybil boxes (loose)
The CT sub-volume spanning the box slices (+/- pad) is cropped in-plane around
the union box (+margin), lung-windowed (DeepLesion lung-lesion window
[-1350, 150] HU), scaled to 0-255 and fed as a 'video'; a box prompt is added
on every annotated slice, masks are propagated forward and backward and the
result is kept inside the (slightly dilated) box.  3-D post-processing is the
same as Phase 11 (keep components >= 10 % of the largest).

Checkpoints are loaded by MedSAM2's own loader with torch.load(weights_only=True).
No ADC/SCC label is read anywhere.

Usage (.venv-medsam2):
  python src/phase13_medsam2.py validate [--n 80] [--ckpt latest|ctlesion] [--crop 1|0]
  python src/phase13_medsam2.py run --ckpt ...          # all 1041 patients
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from PIL import Image
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.join(ROOT, "external", "MedSAM2")
sys.path.insert(0, REPO)

WINDOW = (-1350.0, 150.0)
Z_PAD = 2
MARGIN_FRAC, MARGIN_MIN_MM = 0.5, 15.0
BOX_DILATE_MM = 3.0
CKPTS = {"latest": "MedSAM2_latest.pt", "ctlesion": "MedSAM2_CTLesion.pt"}
OUT_ROOT = "C:/LUNG_phase13/medsam2_masks"
VAL_OUT = os.path.join(ROOT, "results", "phase13")


def build(ckpt):
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from sam2.build_sam import build_sam2_video_predictor_npz
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=os.path.join(REPO, "sam2", "configs"), version_base="1.2")
    return build_sam2_video_predictor_npz("sam2.1_hiera_t512.yaml",
                                          os.path.join(REPO, "checkpoints", CKPTS[ckpt]))


def to_video(sub):
    """(d, h, w) HU -> (d, 3, 512, 512) normalised tensor on GPU."""
    lo, hi = WINDOW
    u8 = (np.clip((sub - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    frames = np.stack([np.array(Image.fromarray(s).convert("RGB").resize((512, 512), Image.BILINEAR))
                       for s in u8]).transpose(0, 3, 1, 2).astype(np.float32) / 255.0
    t = torch.from_numpy(frames).cuda()
    mean = torch.tensor((0.485, 0.456, 0.406), device="cuda")[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device="cuda")[:, None, None]
    return (t - mean) / std


@torch.inference_mode()
def segment(pred, ct, box, spacing, crop=True, prompt="all"):
    """ct (z,y,x) HU, box (z,y,x) bool, spacing (sx,sy,sz) -> bool mask."""
    zs = np.where(box.reshape(box.shape[0], -1).any(1))[0]
    z0, z1 = max(zs.min() - Z_PAD, 0), min(zs.max() + Z_PAD, ct.shape[0] - 1)
    ys, xs = np.nonzero(box[zs].any(0))
    if crop:
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        side = int(max(h, w) * (1 + 2 * MARGIN_FRAC) + 2 * MARGIN_MIN_MM / spacing[0])
        side = min(side, max(ct.shape[1], ct.shape[2]))
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        y0 = int(np.clip(cy - side // 2, 0, max(ct.shape[1] - side, 0)))
        x0 = int(np.clip(cx - side // 2, 0, max(ct.shape[2] - side, 0)))
        y1, x1 = min(y0 + side, ct.shape[1]), min(x0 + side, ct.shape[2])
    else:
        y0, x0, y1, x1 = 0, 0, ct.shape[1], ct.shape[2]
    sub = ct[z0:z1 + 1, y0:y1, x0:x1]
    bsub = box[z0:z1 + 1, y0:y1, x0:x1]
    vid = to_video(sub)
    H, W = sub.shape[1:]
    out = np.zeros(sub.shape, bool)
    prompts = [int(z) - z0 for z in zs]
    if prompt == "mid":                      # official MedSAM2 style: one key-slice box
        areas = [bsub[f].sum() for f in prompts]
        prompts = [prompts[int(np.argmax(areas))]]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for reverse in (False, True):
            st = pred.init_state(vid, H, W)
            for f in prompts:
                yy, xx = np.nonzero(bsub[f])
                pred.add_new_points_or_box(inference_state=st, frame_idx=f, obj_id=1,
                                           box=np.array([xx.min(), yy.min(), xx.max(), yy.max()], np.float32))
            start = min(prompts) if not reverse else max(prompts)
            for fi, _, logits in pred.propagate_in_video(st, start_frame_idx=start, reverse=reverse):
                out[fi] |= (logits[0] > 0.0).cpu().numpy()[0]
            pred.reset_state(st)
    # keep inside the (slightly dilated) box, box slices only
    r = max(int(round(BOX_DILATE_MM / spacing[0])), 1)
    keep = np.zeros_like(bsub)
    for f in [int(z) - z0 for z in zs]:
        keep[f] = ndi.binary_dilation(bsub[f], iterations=r)
    out &= keep
    full = np.zeros(ct.shape, bool)
    full[z0:z1 + 1, y0:y1, x0:x1] = out
    lab, n = ndi.label(full)
    if n > 1:
        vol = np.bincount(lab.ravel())[1:]
        full = np.isin(lab, np.where(vol >= 0.1 * vol.max())[0] + 1)
    return full


def dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else 1.0


def boxes_from_mask(m):
    b = np.zeros_like(m, dtype=bool)
    for z in np.where(m.reshape(m.shape[0], -1).any(1))[0]:
        ys, xs = np.nonzero(m[z])
        b[z, ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    return b


def loosen(box, spacing, mm=8.0):
    out = np.zeros_like(box)
    py, px = int(round(mm / spacing[1])), int(round(mm / spacing[0]))
    for z in np.where(box.reshape(box.shape[0], -1).any(1))[0]:
        ys, xs = np.nonzero(box[z])
        out[z, max(ys.min() - py, 0):ys.max() + py + 1, max(xs.min() - px, 0):xs.max() + px + 1] = True
    return out


def dev_cases():
    c = pd.read_csv(os.path.join(ROOT, "cohort", "phase8b_multicenter_cohort.csv"))
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config", "phase10.yaml"), encoding="utf-8"))
    paths = cfg["paths"]
    rows = []
    for k, pid in zip(c.PatientKey, c.PatientID):
        root = paths["lung1_image_root"] if k.startswith("LUNG1") else paths["external_image_root"]
        root = root if os.path.isabs(root) else os.path.join(ROOT, *root.split("/"))
        rows.append((k, os.path.join(root, pid, "CT.nii.gz"), os.path.join(root, pid, "ROI_A_GTV.nii.gz")))
    return rows


def validate(a):
    pred = build(a.ckpt)
    cases = dev_cases()
    rng = np.random.RandomState(0)
    if a.n and a.n < len(cases):
        # stratified by site only (no label): half LUNG1, half Radiogenomics
        l1 = [c for c in cases if c[0].startswith("LUNG1")]
        rg = [c for c in cases if not c[0].startswith("LUNG1")]
        cases = [l1[i] for i in rng.choice(len(l1), a.n // 2, replace=False)] + \
                [rg[i] for i in rng.choice(len(rg), a.n - a.n // 2, replace=False)]
    v5 = pd.read_csv(os.path.join(ROOT, "results", "phase12", "autoseg_v2_validation.csv")).set_index("PatientKey")
    rows, t0 = [], time.time()
    for i, (k, ctp, gp) in enumerate(cases):
        try:
            img = sitk.ReadImage(ctp, sitk.sitkFloat32)
            g = sitk.Resample(sitk.ReadImage(gp, sitk.sitkUInt8), img, sitk.Transform(),
                              sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            ct, g = sitk.GetArrayFromImage(img), sitk.GetArrayFromImage(g) > 0
            tight = boxes_from_mask(g)
            loose = loosen(tight, img.GetSpacing())
            r = {"PatientKey": k,
                 "ms2_tight": dice(segment(pred, ct, tight, img.GetSpacing(), bool(a.crop), a.prompt), g),
                 "ms2_loose": dice(segment(pred, ct, loose, img.GetSpacing(), bool(a.crop), a.prompt), g)}
            if k in v5.index:
                r.update(v5_tight=v5.loc[k, "v5_tight"], v5_loose=v5.loc[k, "v5_loose"])
        except Exception as e:
            r = {"PatientKey": k, "error": str(e)[:300]}
        rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"{i + 1}/{len(cases)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["site"] = df.PatientKey.str.split("::").str[0]
    os.makedirs(VAL_OUT, exist_ok=True)
    tag = f"{a.ckpt}_crop{a.crop}_{a.prompt}_n{len(cases)}"
    df.to_csv(os.path.join(VAL_OUT, f"medsam2_validation_{tag}.csv"), index=False)
    cols = [c for c in ("v5_tight", "ms2_tight", "v5_loose", "ms2_loose") if c in df]
    print(tag, "\n", df.groupby("site")[cols].agg(["mean", "median"]).round(3).T.to_string())
    if "error" in df:
        print("errors:", df.error.notna().sum(), df.error.dropna().head(3).tolist())


def run(a):
    pred = build(a.ckpt)
    coh = pd.read_csv(os.path.join(ROOT, "cohort", "phase12_four_site_cohort.csv"))
    auto2 = "C:/LUNG_phase12/auto_masks_v2"
    rows, t0 = [], time.time()
    for i, (k, ctp) in enumerate(zip(coh.PatientKey, coh.ct_path)):
        dest = os.path.join(OUT_ROOT, k.replace("::", "__"))
        fn = os.path.join(dest, "AUTO_GTV.nii.gz")
        try:
            img = sitk.ReadImage(ctp, sitk.sitkFloat32)
            vox = float(np.prod(img.GetSpacing())) / 1000.0
            if not os.path.isfile(fn):
                box = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(auto2, k.replace("::", "__"), "BOX.nii.gz"))) > 0
                m = segment(pred, sitk.GetArrayFromImage(img), box, img.GetSpacing(), bool(a.crop), a.prompt)
                if m.sum() == 0:
                    raise RuntimeError("empty mask")
                os.makedirs(dest, exist_ok=True)
                mi = sitk.GetImageFromArray(m.astype(np.uint8))
                mi.CopyInformation(img)
                sitk.WriteImage(mi, fn)
            m = sitk.GetArrayFromImage(sitk.ReadImage(fn)) > 0
            rows.append({"PatientKey": k, "status": "ok", "ct_path": ctp, "auto_volume_ml": m.sum() * vox})
        except Exception as e:
            rows.append({"PatientKey": k, "status": "failed: %s" % str(e)[:200]})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(coh)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(OUT_ROOT, exist_ok=True)
    df.to_csv(os.path.join(OUT_ROOT, "autoseg_manifest.csv"), index=False)
    json.dump(dict(ckpt=CKPTS[a.ckpt], crop=a.crop, prompt=a.prompt, window=WINDOW, z_pad=Z_PAD, margin_frac=MARGIN_FRAC,
                   margin_min_mm=MARGIN_MIN_MM, box_dilate_mm=BOX_DILATE_MM),
              open(os.path.join(OUT_ROOT, "settings.json"), "w"), indent=1)
    print(df.status.value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["validate", "run"])
    ap.add_argument("--ckpt", default="latest", choices=list(CKPTS))
    ap.add_argument("--crop", type=int, default=1)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--prompt", default="all", choices=["all", "mid"])
    a = ap.parse_args()
    validate(a) if a.step == "validate" else run(a)


if __name__ == "__main__":
    main()
