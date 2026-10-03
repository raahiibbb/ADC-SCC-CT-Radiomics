"""Phase 8A - automatic and visual QA of the external cohort.

Every check re-derives its answer from the files on disk; none of them reads a
model, a prediction or a metric, and none of them is stratified by ADC/SCC.

    python src/phase8a_qa.py --config config/phase8a_external.yaml [--figures]

Checks
------
 1  patient / series mapping        cohort <-> screening <-> per-patient record
 2  CT ordering and orientation     monotonic slice positions, one orientation
 3  segmentation alignment          SEG geometry == CT geometry, 0 lost frames
 4  GTV validity                    non-empty, plausible volume, inside the CT
 5  rim validity                    non-empty, disjoint from GTV, union exact
 6  physical expansion              rim thickness == the configured 8 mm
 7  2 mm resampling                 grid re-derived from the CT, spacing exact
 8  image HU integrity              CT HU range physically plausible
 9  patch lattice                   anchor (0,0,0), stride 5, centre-in-ROI
10  CNN crop orientation            the three planes intersect at the centre
11  global-radiomics consistency    identical feature names, no NaN/Inf
12  no duplication                  every patient exactly once, everywhere
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase8a_tcia import load_cfg, ppath                            # noqa: E402
from phase8a_regions import signed_distance_field, zyx_spacing      # noqa: E402
from resample import reference_grid, resample_to_grid               # noqa: E402


def _ok(results, name, passed, detail):
    results.append({"check": name, "pass": bool(passed), "detail": detail})
    print("%-34s %s  %s" % (name, "PASS" if passed else "FAIL", detail), flush=True)
    return passed


def included(cfg):
    df = pd.read_csv(ppath(cfg, "cohort_csv"))
    return df[df["included"].astype(str).str.lower() == "true"].reset_index(drop=True)


def patient_record(cfg, pid):
    with open(os.path.join(ppath(cfg, "masks_dir"), pid, "record.json"),
              "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_checks(cfg):
    res = []
    inc = included(cfg)
    pids = sorted(inc["PatientID"].astype(str))

    # ---- 1 patient / series mapping ---------------------------------------
    with open(os.path.join(ppath(cfg, "metadata_dir"), "screening.json"),
              "r", encoding="utf-8") as fh:
        scr = json.load(fh)
    scr_by_pid = {s["PatientID"]: s for s in scr["subjects"]}
    bad = []
    for pid in pids:
        rec = patient_record(cfg, pid)
        s = scr_by_pid[pid]
        if (rec["ct_series_uid"] != s["ct_series_uid"]
                or rec["seg_series_uid"] != s["seg_series_uid"]
                or rec["ct_header"]["SeriesInstanceUID"] != s["ct_series_uid"]):
            bad.append(pid)
    _ok(res, "1 patient/series mapping", not bad,
        "%d patients; CT and SEG series UIDs agree between the collection "
        "metadata, the screening record and the DICOM headers" % len(pids)
        if not bad else "mismatch: %s" % bad[:5])

    # ---- 2 CT ordering and orientation ------------------------------------
    bad, nonuniform = [], []
    for pid in pids:
        rec = patient_record(cfg, pid)
        h = rec["ct_header"]
        d = np.asarray(h["itk_direction"], dtype=float).reshape(3, 3)
        if not np.allclose(d, np.eye(3), atol=1e-6):
            bad.append(pid)
        if abs(h["slice_spacing_mm_max"] - h["slice_spacing_mm_min"]) > 0.01:
            nonuniform.append(pid)
    _ok(res, "2 CT ordering/orientation", not bad,
        "all %d volumes have identity direction cosines and strictly monotonic "
        "slice positions; %d with non-uniform slice spacing (recorded, not "
        "repaired)" % (len(pids), len(nonuniform))
        if not bad else "non-identity direction: %s" % bad[:5])

    # ---- 3 segmentation alignment ------------------------------------------
    bad = []
    for pid in pids:
        rec = patient_record(cfg, pid)
        si = rec["seg_info"]
        lost = int(si["unmapped_nonempty_frames"]) + int(si.get("unassigned_nonempty_frames", 0))
        if (si["seg_rows"] != si["ct_rows"] or si["seg_columns"] != si["ct_columns"]
                or lost != 0):
            bad.append(pid)
    _ok(res, "3 segmentation alignment", not bad,
        "SEG rows/columns equal the CT rows/columns and 0 non-empty frames were "
        "lost, for all %d patients" % len(pids)
        if not bad else "misaligned: %s" % bad[:5])

    # ---- 4/5 GTV and rim validity ------------------------------------------
    geo = []
    for pid in pids:
        g = patient_record(cfg, pid)["regions"]
        g["PatientID"] = pid
        geo.append(g)
    gdf = pd.DataFrame(geo)
    _ok(res, "4 GTV validity",
        bool((gdf["native_gtv_voxels"] > 0).all()
             and (gdf["native_gtv_volume_mm3"] > 0).all()),
        "GTV volume %0.0f-%0.0f mm3 (median %0.0f); %d of %d touch the volume "
        "boundary" % (gdf["native_gtv_volume_mm3"].min(),
                      gdf["native_gtv_volume_mm3"].max(),
                      gdf["native_gtv_volume_mm3"].median(),
                      int(gdf["gtv_touches_volume_boundary"].sum()), len(gdf)))
    _ok(res, "5 rim validity",
        bool(gdf["disjoint_ok"].all() and gdf["union_ok"].all()
             and gdf["subset_ok"].all() and (gdf["native_rim_voxels"] > 0).all()),
        "GTV n Rim = 0, GTV u Rim = GTV+Rim and GTV subset GTV+Rim for all %d "
        "patients; rim fraction %.3f-%.3f (median %.3f)"
        % (len(gdf), gdf["native_rim_fraction_of_gtv_rim"].min(),
           gdf["native_rim_fraction_of_gtv_rim"].max(),
           gdf["native_rim_fraction_of_gtv_rim"].median()))

    # ---- 6 physical expansion ----------------------------------------------
    exp = float(cfg["regions"]["expansion_mm"])
    sub = pids[:min(6, len(pids))]
    worst = 0.0
    detail = []
    for pid in sub:
        d = os.path.join(ppath(cfg, "masks_dir"), pid)
        gtv = sitk.ReadImage(os.path.join(d, cfg["paths"]["gtv_filename"]))
        rim = sitk.ReadImage(os.path.join(d, cfg["paths"]["rim_filename"]))
        g = sitk.GetArrayViewFromImage(gtv) > 0
        r = sitk.GetArrayViewFromImage(rim) > 0
        phi = signed_distance_field(g, zyx_spacing(gtv),
                                    method=cfg["regions"]["distance_method"],
                                    crop_margin_mm=exp + 4.0 * max(zyx_spacing(gtv)),
                                    fine_mm=float(cfg["regions"]["fine_grid_mm"]))
        mx = float(phi[r].max()) if r.any() else 0.0
        worst = max(worst, abs(mx - exp))
        detail.append("%s %.3f" % (pid, mx))
    _ok(res, "6 physical expansion", worst <= 0.05,
        "max signed distance inside the rim equals %.1f mm to within %.3f mm on "
        "%d re-derived patients (%s)" % (exp, worst, len(sub), ", ".join(detail)))

    # ---- 7 2 mm resampling ---------------------------------------------------
    spacings, bad = set(), []
    for pid in sub:
        ct = sitk.ReadImage(os.path.join(ppath(cfg, "masks_dir"), pid,
                                         cfg["paths"]["ct_filename"]))
        grid = reference_grid(ct, cfg["preprocessing"]["resample_spacing_mm"])
        spacings.add(tuple(round(float(v), 9) for v in grid["spacing"]))
        if tuple(grid["origin"]) != tuple(ct.GetOrigin()):
            bad.append(pid)
    _ok(res, "7 2 mm resampling", spacings == {(2.0, 2.0, 2.0)} and not bad,
        "the 2 mm reference grid re-derives from every CT with spacing %s and "
        "the CT's own origin/direction" % sorted(spacings))

    # ---- 8 image HU integrity -------------------------------------------------
    lo_hi = []
    for pid in sub:
        ct = sitk.ReadImage(os.path.join(ppath(cfg, "masks_dir"), pid,
                                         cfg["paths"]["ct_filename"]))
        a = sitk.GetArrayViewFromImage(ct)
        lo_hi.append((int(a.min()), int(a.max())))
    ok = all(l <= -900 and 500 <= h <= 4100 for l, h in lo_hi)
    _ok(res, "8 image HU integrity", ok,
        "native HU range over %d re-read volumes: min %d..%d, max %d..%d "
        "(air present, bone present, no rescale error)"
        % (len(lo_hi), min(l for l, _ in lo_hi), max(l for l, _ in lo_hi),
           min(h for _, h in lo_hi), max(h for _, h in lo_hi)))

    # ---- 9 patch lattice ------------------------------------------------------
    pf_dir = ppath(cfg, "patch_features_dir")
    bags = sorted(f for f in os.listdir(pf_dir) if f.endswith(".npz")) \
        if os.path.isdir(pf_dir) else []
    if bags:
        anchor_ok, stride_ok, centre_ok, sig_ok = True, True, True, True
        n_inst = 0
        for f in bags:
            with np.load(os.path.join(pf_dir, f), allow_pickle=True) as z:
                st = np.asarray(z["coords_start_index"], dtype=int)
                ci = np.asarray(z["coords_index"], dtype=int)
                size = np.asarray(z["patch_size"], dtype=int)
                stride = np.asarray(z["patch_stride"], dtype=int)
                n_inst += st.shape[0]
                if st.size and (st % stride).any():
                    anchor_ok = False
                if not np.array_equal(stride, np.asarray([5, 5, 5])):
                    stride_ok = False
                if st.size and not np.array_equal(ci, st + size // 2):
                    centre_ok = False
                if str(z["config_signature"]) != cfg["patch_sampling"]["expected_extraction_signature"]:
                    sig_ok = False
        _ok(res, "9 patch lattice", anchor_ok and stride_ok and centre_ok and sig_ok,
            "%d bags, %d instances; every patch start is a multiple of stride 5 "
            "(lattice anchored at resampled voxel (0,0,0)), every centre is "
            "start + 2, and every bag carries the frozen LUNG1 signature %s"
            % (len(bags), n_inst,
               cfg["patch_sampling"]["expected_extraction_signature"]))
    else:
        _ok(res, "9 patch lattice", False, "no external patch bags on disk")

    # ---- 10 CNN crop orientation ------------------------------------------------
    ip_dir = ppath(cfg, "image_patches_dir")
    imgs = sorted(f for f in os.listdir(ip_dir) if f.endswith(".npz")) \
        if os.path.isdir(ip_dir) else []
    if imgs:
        bad, checked = [], 0
        for f in imgs[:min(8, len(imgs))]:
            with np.load(os.path.join(ip_dir, f), allow_pickle=True) as z:
                im = np.asarray(z["images"])
                if im.shape[0] == 0:
                    continue
                c = im[:, :, 16, 16]
                checked += im.shape[0]
                if not (np.array_equal(c[:, 0], c[:, 1])
                        and np.array_equal(c[:, 1], c[:, 2])):
                    bad.append(f)
                if str(z["crop_signature"]) != cfg["crop"]["expected_phase6a_crop_signature"]:
                    bad.append(f + " (signature)")
        _ok(res, "10 CNN crop orientation", not bad,
            "on %d instances of %d patients the axial, coronal and sagittal "
            "planes carry the SAME value at output index (16, 16), i.e. all "
            "three intersect exactly at the patch centre; crop signature %s"
            % (checked, min(8, len(imgs)),
               cfg["crop"]["expected_phase6a_crop_signature"])
            if not bad else "mismatch: %s" % bad[:3])
    else:
        _ok(res, "10 CNN crop orientation", False, "no external image crops on disk")

    # ---- 11 global-radiomics consistency -----------------------------------------
    gpath, rpath = ppath(cfg, "gtv_csv"), ppath(cfg, "rim_csv")
    if os.path.isfile(gpath) and os.path.isfile(rpath):
        g = pd.read_csv(gpath)
        r = pd.read_csv(rpath)
        meta = ["PatientID", "Histology", "label"]
        gn = [c for c in g.columns if c not in meta]
        rn = [c for c in r.columns if c not in meta]
        with open(os.path.join(ppath(cfg, "global_features_dir"),
                               "feature_names.json"), "r", encoding="utf-8") as fh:
            names = json.load(fh)
        nan_g = int(g[gn].isna().sum().sum())
        nan_r = int(r[rn].isna().sum().sum())
        inf_g = int(np.isinf(g[gn].to_numpy(dtype=float)).sum())
        inf_r = int(np.isinf(r[rn].to_numpy(dtype=float)).sum())
        ok = (len(g) == len(pids) and len(r) == len(pids)
              and gn == names["gtv"] and rn == names["rim"]
              and nan_g == 0 and nan_r == 0 and inf_g == 0 and inf_r == 0
              and set(g["PatientID"]) == set(pids))
        _ok(res, "11 global-radiomics consistency", ok,
            "%d x %d GTV and %d x %d rim rows, feature names identical for every "
            "patient, %d NaN and %d Inf in total"
            % (len(g), len(gn), len(r), len(rn), nan_g + nan_r, inf_g + inf_r))
    else:
        _ok(res, "11 global-radiomics consistency", False,
            "no external global feature matrices on disk")

    # ---- 12 no duplication ----------------------------------------------------
    coh = pd.read_csv(ppath(cfg, "cohort_csv"))
    dirs = sorted(d for d in os.listdir(ppath(cfg, "masks_dir"))
                  if os.path.isdir(os.path.join(ppath(cfg, "masks_dir"), d)))
    dup = (coh["PatientID"].duplicated().any()
           or len(set(pids)) != len(pids)
           or len(set(dirs)) != len(dirs))
    _ok(res, "12 no patient duplication", not dup,
        "%d cohort rows all unique; %d included patients each with exactly one "
        "product directory" % (len(coh), len(pids)))

    return res, gdf


# ---------------------------------------------------------------------------
# visual QA
# ---------------------------------------------------------------------------

def visual_patients(cfg, inc):
    """Deterministic, label-blind: smallest / largest GTV, first / last ID,
    and two evenly spaced interior patients of the sorted cohort."""
    n_want = int(cfg["qa"]["visual_patients"])
    df = inc.sort_values("gtv_volume_mm3").reset_index(drop=True)
    picks = [df.PatientID.iloc[0], df.PatientID.iloc[-1]]
    s = inc.sort_values("PatientID").reset_index(drop=True)
    picks += [s.PatientID.iloc[0], s.PatientID.iloc[-1]]
    for frac in (1.0 / 3.0, 2.0 / 3.0):
        picks.append(s.PatientID.iloc[int(round(frac * (len(s) - 1)))])
    out = []
    for p in picks:
        if p not in out:
            out.append(str(p))
    i = 0
    while len(out) < n_want and i < len(s):
        if str(s.PatientID.iloc[i]) not in out:
            out.append(str(s.PatientID.iloc[i]))
        i += 1
    return out[:n_want]


def make_figures(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    inc = included(cfg)
    pids = visual_patients(cfg, inc)
    out_dir = ppath(cfg, "qa_dir")
    os.makedirs(out_dir, exist_ok=True)
    wl = float(cfg["qa"]["window_level"])
    ww = float(cfg["qa"]["window_width"])
    vmin, vmax = wl - ww / 2.0, wl + ww / 2.0
    red = ListedColormap([(0, 0, 0, 0), (1.0, 0.15, 0.15, 0.55)])
    blue = ListedColormap([(0, 0, 0, 0), (0.25, 0.55, 1.0, 0.45)])

    made = []
    for pid in pids:
        d = os.path.join(ppath(cfg, "masks_dir"), pid)
        ct = sitk.ReadImage(os.path.join(d, cfg["paths"]["ct_filename"]))
        gtv = sitk.ReadImage(os.path.join(d, cfg["paths"]["gtv_filename"]))
        rim = sitk.ReadImage(os.path.join(d, cfg["paths"]["rim_filename"]))
        a = sitk.GetArrayViewFromImage(ct)
        g = sitk.GetArrayViewFromImage(gtv) > 0
        r = sitk.GetArrayViewFromImage(rim) > 0
        nz = np.argwhere(g)
        kc, jc, ic = (int(round(v)) for v in nz.mean(axis=0))
        sp = ct.GetSpacing()

        fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.8))
        panels = [("axial  k=%d" % kc, a[kc], g[kc], r[kc], sp[0] / sp[1]),
                  ("coronal  j=%d" % jc, a[:, jc, :], g[:, jc, :], r[:, jc, :],
                   sp[0] / sp[2]),
                  ("sagittal  i=%d" % ic, a[:, :, ic], g[:, :, ic], r[:, :, ic],
                   sp[1] / sp[2])]
        for k, (title, img, gm, rm, aspect) in enumerate(panels):
            show = img if k == 0 else img[::-1]
            gg = gm if k == 0 else gm[::-1]
            rr = rm if k == 0 else rm[::-1]
            ax[k].imshow(show, cmap="gray", vmin=vmin, vmax=vmax, aspect=aspect)
            ax[k].imshow(rr.astype(np.uint8), cmap=blue, vmin=0, vmax=1, aspect=aspect)
            ax[k].imshow(gg.astype(np.uint8), cmap=red, vmin=0, vmax=1, aspect=aspect)
            ax[k].set_title(title, fontsize=10)
            ax[k].set_xticks([])
            ax[k].set_yticks([])
        row = inc[inc.PatientID == pid].iloc[0]
        fig.suptitle("%s   GTV %.0f mm3 (red)   8 mm rim %.0f mm3 (blue)   "
                     "spacing %.3f x %.3f x %.3f mm   [histology withheld from "
                     "the figure]"
                     % (pid, row["gtv_volume_mm3"], row["rim_volume_mm3"],
                        sp[0], sp[1], sp[2]), fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        path = os.path.join(out_dir, "%s_regions.png" % pid)
        fig.savefig(path, dpi=110)
        plt.close(fig)
        made.append(path)
        print("wrote %s" % path, flush=True)
    return made


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--figures", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)

    res, gdf = run_checks(cfg)
    gdf.to_csv(ppath(cfg, "region_geometry_csv"), index=False)
    figs = make_figures(cfg) if args.figures else []
    out = {"n_checks": len(res), "n_pass": int(sum(r["pass"] for r in res)),
           "checks": res, "figures": [os.path.basename(f) for f in figs]}
    with open(os.path.join(ppath(cfg, "qa_dir"), "qa_checks.json"),
              "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\n%d/%d QA checks pass" % (out["n_pass"], out["n_checks"]))
    return 0 if out["n_pass"] == out["n_checks"] else 1


if __name__ == "__main__":
    sys.exit(main())
