"""Phase-4 sanity checks: the full 202-patient GTV+Rim extraction.

Run with:
    ./.venv/Scripts/python.exe tests/test_phase4.py

Covers the 18 required Phase-4 QA items plus the checks that keep the GTV+Rim
experiment comparable to the frozen Gradient one and keep every Gradient
artefact untouched. No classifier is trained, loaded or evaluated anywhere in
this file, and no radiomics is recomputed except in group W, which deliberately
re-derives the resampled mask from the CT for a seeded sample of patients.

Groups
  U  cohort, inventory and labels (items 1, 2, 3, 6, 7, 12)
  V  bag structure, features and coordinates, all 202 (items 4, 5, 8, 10, 11, 14)
  W  independent re-derivation from CT + mask (seeded sample) and geometry (13)
  X  provenance: signatures, rejections, splits (items 9, 15)
  Y  the frozen Gradient experiment and the read-only trees (items 16, 17, 18)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import statistics
import sys

import numpy as np
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cohort import load_cohort                                     # noqa: E402
from config_io import (cfg_path, extraction_signature,             # noqa: E402
                       load_config, project_path)
from extract_patient import validate_existing                      # noqa: E402
from patches import patch_roi_stats                                # noqa: E402
from resample import resample_ct_and_mask                          # noqa: E402
import readonly_guard                                              # noqa: E402

CFG = load_config(os.path.join(ROOT, "config", "gtv_rim_experiment.yaml"))
REF = load_config(os.path.join(ROOT, "config", "experiment.yaml"))
SIG = extraction_signature(CFG)
REF_SIG = extraction_signature(REF)
FEAT_DIR = project_path(CFG, *CFG["paths"]["features_dir"].split("/"))
REF_FEAT_DIR = project_path(REF, *REF["paths"]["features_dir"].split("/"))
N_EXPECTED = int(CFG["cohort"]["expected_eligible"])
SMALL_BAG = 20
SAMPLE_SIZE = 12
SAMPLE_SEED = 42
CHECKS = []


def check(group, name):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


def _cohort():
    return {r["PatientID"]: r for r in load_cohort(CFG)}


def _eligible_ids():
    return sorted(p for p, r in _cohort().items() if r["eligible"])


def _pilot_ids():
    with open(cfg_path(CFG, "pilot_file"), encoding="utf-8") as fh:
        ids = [l.split("#")[0].strip() for l in fh]
    return [i for i in ids if i]


def _bag(pid, d=None):
    return np.load(os.path.join(d or FEAT_DIR, "%s.npz" % pid), allow_pickle=True)


def _qa(pid):
    with open(os.path.join(FEAT_DIR, "%s.qa.json" % pid), encoding="utf-8") as fh:
        return json.load(fh)


def _feature_names():
    rel = CFG["paths"].get("feature_names_file", "config/feature_names.txt")
    with open(project_path(CFG, *rel.split("/")), encoding="utf-8") as fh:
        return [l.strip() for l in fh if l.strip()]


def _sample_ids():
    coh = _cohort()
    pool = [p for p in _eligible_ids() if p not in set(_pilot_ids())]
    adc = [p for p in pool if coh[p]["label"] == 1]
    scc = [p for p in pool if coh[p]["label"] == 0]
    rng = random.Random(SAMPLE_SEED)
    return sorted(rng.sample(adc, SAMPLE_SIZE // 2)
                  + rng.sample(scc, SAMPLE_SIZE - SAMPLE_SIZE // 2))


def _resampled(pid):
    pdir = os.path.join(CFG["paths"]["vuong_root"], pid)
    ct = sitk.ReadImage(os.path.join(pdir, CFG["paths"]["ct_filename"]))
    mk = sitk.ReadImage(os.path.join(pdir, CFG["experiment"]["roi_filename"]))
    ct_r, mk_r, grid = resample_ct_and_mask(ct, mk, CFG)
    return ct_r, sitk.GetArrayFromImage(mk_r), grid


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _split_rows(cfg):
    p = project_path(cfg, *cfg["paths"]["splits_file"].split("/"))
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh)), p


# =================================================== U: cohort and inventory

@check("U", "exactly 202 GTV+Rim bags, one per eligible patient, none extra")
def u1():
    want = set(_eligible_ids())
    have = {os.path.splitext(f)[0] for f in os.listdir(FEAT_DIR)
            if f.endswith(".npz")}
    if len(want) != N_EXPECTED:
        return False, "cohort says %d eligible, expected %d" % (len(want), N_EXPECTED)
    if want != have:
        return False, ("missing %d, unexpected %d"
                       % (len(want - have), len(have - want)))
    return True, "%d bags, exact 1:1 with the eligible cohort" % len(have)


@check("U", "ADC / SCC counts match the current read-only metadata")
def u2():
    coh = _cohort()
    elig = _eligible_ids()
    adc = [p for p in elig if coh[p]["label"] == 1]
    scc = [p for p in elig if coh[p]["label"] == 0]
    # re-derive independently from the two metadata sources
    with open(CFG["paths"]["cohort_summary"], encoding="utf-8-sig") as fh:
        summ = {r["PatientID"]: r for r in csv.DictReader(fh)}
    with open(CFG["paths"]["dataset_metadata"], encoding="utf-8-sig") as fh:
        mast = {r["PatientID"]: r for r in csv.DictReader(fh)}
    col = CFG["cohort"]["exclusion_column"]
    redo = [p for p in summ
            if str(summ[p][col]).strip().lower() not in ("true", "1", "yes")]
    r_adc = [p for p in redo if mast[p]["Histology"].strip() == "adenocarcinoma"]
    if sorted(redo) != elig:
        return False, "independent re-derivation gives %d patients" % len(redo)
    if len(adc) != 51 or len(scc) != 151 or len(r_adc) != 51:
        return False, "%d ADC / %d SCC (re-derived %d ADC)" % (len(adc), len(scc),
                                                              len(r_adc))
    return True, "51 ADC / 151 SCC, re-derived from both metadata sources"


@check("U", "LUNG1-011 is excluded and the eight Gradient-only exclusions are not")
def u3():
    coh = _cohort()
    elig = set(_eligible_ids())
    if "LUNG1-011" in elig or os.path.isfile(os.path.join(FEAT_DIR,
                                                          "LUNG1-011.npz")):
        return False, "LUNG1-011 is present"
    grad_only = set(REF["cohort"]["gradient_exclusions"]) - {"LUNG1-011"}
    back = sorted(grad_only & elig)
    if len(back) != 8:
        return False, "%d of the 8 Gradient-only exclusions are eligible" % len(back)
    return True, ("LUNG1-011 excluded; the 8 Gradient-only exclusions (%s) are "
                  "back in" % ", ".join(b.replace("LUNG1-", "") for b in back))


@check("U", "no duplicate and no unexpected patient anywhere")
def u6():
    ids = [os.path.splitext(f)[0] for f in os.listdir(FEAT_DIR)
           if f.endswith(".npz")]
    dupes = {i for i in ids if ids.count(i) > 1}
    saved = [str(_bag(p)["patient_id"]) for p in sorted(set(ids))]
    if dupes or len(set(saved)) != len(saved):
        return False, "%d duplicate ids" % len(dupes)
    if sorted(saved) != _eligible_ids():
        return False, "saved patient_id set differs from the eligible cohort"
    extra = [f for f in os.listdir(FEAT_DIR)
             if not (f.endswith(".npz") or f.endswith(".qa.json"))]
    if extra:
        return False, "%d unexpected files in the feature directory" % len(extra)
    return True, "%d unique ids, no stray files" % len(saved)


@check("U", "every saved label and histology agrees with the cohort metadata")
def u12():
    coh = _cohort()
    bad = []
    for pid in _eligible_ids():
        with _bag(pid) as z:
            if (int(z["label"]) != int(coh[pid]["label"])
                    or str(z["histology"]) != coh[pid]["Histology"]
                    or str(z["roi_name"]) != CFG["experiment"]["roi"]):
                bad.append(pid)
    return not bad, "%d disagreements over %d bags" % (len(bad), len(_eligible_ids()))


@check("U", "ADC is label 1 and SCC is label 0, as in the Gradient experiment")
def u12b():
    if CFG["cohort"]["histology_map"] != REF["cohort"]["histology_map"]:
        return False, "histology_map differs from the Gradient config"
    coh = _cohort()
    bad = [p for p in _eligible_ids()
           if (coh[p]["Histology"] == "adenocarcinoma") != (coh[p]["label"] == 1)]
    return not bad, "adenocarcinoma -> 1, squamous cell carcinoma -> 0 (%d bad)" % len(bad)


# =========================================== V: bag structure, all 202 bags

@check("V", "all 202 bags carry the same 74 feature names in the same order")
def v4():
    want = _feature_names()
    bad = []
    for pid in _eligible_ids():
        with _bag(pid) as z:
            if [str(x) for x in z["feature_names"]] != want:
                bad.append(pid)
            elif int(z["features"].shape[1]) != len(want):
                bad.append(pid)
    if bad:
        return False, "%d bags differ" % len(bad)
    ref_names = None
    with _bag(_eligible_ids()[0]) as z:
        ref_names = [str(x) for x in z["feature_names"]]
    with _bag(sorted(os.path.splitext(f)[0] for f in os.listdir(REF_FEAT_DIR)
                     if f.endswith(".npz"))[0], REF_FEAT_DIR) as z:
        grad_names = [str(x) for x in z["feature_names"]]
    if ref_names != grad_names:
        return False, "feature order differs from the Gradient bags"
    if any(n.startswith("original_shape") or not n.startswith("original_")
           for n in want):
        return False, "shape or filtered features present"
    return True, ("%d features, identical order in all %d bags AND identical to "
                  "the Gradient bags" % (len(want), N_EXPECTED))


@check("V", "no non-finite value in any retained feature, cohort-wide")
def v8():
    bad, total = [], 0
    for pid in _eligible_ids():
        with _bag(pid) as z:
            f = z["features"]
            total += int(f.size)
            if f.size and not np.all(np.isfinite(f)):
                bad.append(pid)
    return not bad, "%d patients bad over %d values" % (len(bad), total)


@check("V", "every bag validates under the GTV+Rim signature")
def v8b():
    coh = _cohort()
    bad = []
    for pid in _eligible_ids():
        ok, why = validate_existing(os.path.join(FEAT_DIR, "%s.npz" % pid),
                                    pid, SIG, coh[pid]["label"])
        if not ok:
            bad.append({"pid": pid, "why": why})
    return not bad, "%d invalid bags" % len(bad)


@check("V", "patch geometry: 5x5x5, stride 5, non-overlapping, in bounds")
def v10():
    size = np.array(CFG["patches"]["size_voxels"], int)
    stride = np.array(CFG["patches"]["stride_voxels"], int)
    bad = []
    for pid in _eligible_ids():
        with _bag(pid) as z:
            st = z["coords_start_index"].astype(np.int64)
            ce = z["coords_index"].astype(np.int64)
            gs = z["grid_size"].astype(np.int64)
            n = int(z["features"].shape[0])
            if not (np.array_equal(z["patch_size"].astype(int), size)
                    and np.array_equal(z["patch_stride"].astype(int), stride)):
                bad.append((pid, "size/stride")); continue
            if np.any(st % stride[None, :] != 0):
                bad.append((pid, "off-lattice")); continue
            if not np.array_equal(ce, st + (size // 2)[None, :]):
                bad.append((pid, "centre")); continue
            if len({tuple(s) for s in st.tolist()}) != n:
                bad.append((pid, "overlap")); continue
            if np.any(st < 0) or np.any(st + size[None, :] > gs[None, :]):
                bad.append((pid, "bounds"))
    return not bad, ("%d problems; the lattice is anchored at voxel (0,0,0) in "
                     "all %d bags" % (len(bad), N_EXPECTED))


@check("V", "retention rule respected: centre in ROI and >= 27 ROI voxels")
def v10b():
    mn = int(CFG["patches"]["min_roi_voxels"])
    npv = int(np.prod(CFG["patches"]["size_voxels"]))
    lows, fracs, mins = [], [], []
    for pid in _eligible_ids():
        with _bag(pid) as z:
            rv = z["roi_voxels"].astype(np.int64)
            rf = z["roi_fraction"].astype(np.float64)
        if not len(rv):
            continue
        mins.append(int(rv.min()))
        if int(rv.min()) < mn:
            lows.append(pid)
        if int(rv.max()) > npv:
            lows.append(pid)
        if not np.allclose(rf, rv / float(npv), atol=1e-6):
            fracs.append(pid)
    return (not lows and not fracs,
            "min roi_voxels cohort-wide = %d (threshold %d, max possible %d); "
            "%d roi_fraction mismatches" % (min(mins), mn, npv, len(fracs)))


@check("V", "patch-count distribution and small/large bags are reported")
def v10c():
    n = []
    for pid in _eligible_ids():
        with _bag(pid) as z:
            n.append(int(z["features"].shape[0]))
    if min(n) <= 0:
        return False, "%d empty bags" % sum(1 for v in n if v == 0)
    small = sum(1 for v in n if v < SMALL_BAG)
    return True, ("min %d, q1 %g, median %g, q3 %g, max %d; %d bags < %d patches"
                  % (min(n), statistics.quantiles(n, n=4)[0],
                     statistics.median(n), statistics.quantiles(n, n=4)[2],
                     max(n), small, SMALL_BAG))


@check("V", "world <-> index coordinates round-trip in all 202 bags")
def v14():
    worst, bad = 0, []
    for pid in _eligible_ids():
        with _bag(pid) as z:
            o = z["grid_origin"].astype(np.float64)
            sp = z["grid_spacing"].astype(np.float64)
            d = z["grid_direction"].astype(np.float64).reshape(3, 3)
            idx = z["coords_index"].astype(np.float64)
            w = z["coords_world"].astype(np.float64)
            n = idx.shape[0]
        if not n:
            continue
        # world = origin + direction @ diag(spacing) @ index
        fwd = o[None, :] + idx.dot((d * sp[None, :]).T)
        err = float(np.abs(fwd - w).max())
        worst = max(worst, err)
        if err > 1e-6:
            bad.append(pid)
        back = np.linalg.solve((d * sp[None, :]), (w - o[None, :]).T).T
        if np.abs(back - idx).max() > 1e-6:
            bad.append(pid)
    return not bad, ("max |world - origin - D*S*index| = %.3g mm over %d bags"
                     % (worst, N_EXPECTED))


@check("V", "physical patch box is exactly 10 x 10 x 10 mm everywhere")
def v14b():
    want = (np.array(CFG["patches"]["size_voxels"], float)
            * np.array(CFG["preprocessing"]["resample_spacing_mm"], float))
    bad = []
    for pid in _eligible_ids():
        with _bag(pid) as z:
            a, b = z["bbox_world_min"], z["bbox_world_max"]
            if len(a) and not np.allclose(np.abs(b - a).max(axis=0), want,
                                          atol=1e-6):
                bad.append(pid)
    return not bad, "%d bags with a box != %s mm" % (len(bad), list(want))


# ================================= W: independent re-derivation and geometry

@check("W", "patch centres and ROI voxel counts re-derived from a fresh resampling")
def w13a():
    sample = _sample_ids()
    size = CFG["patches"]["size_voxels"]
    n_checked, bad = 0, []
    for pid in sample:
        _, mask_a, grid = _resampled(pid)
        with _bag(pid) as z:
            st = z["coords_start_index"].astype(int)
            ce = z["coords_index"].astype(int)
            rv = z["roi_voxels"].astype(int)
            if list(z["grid_size"].astype(int)) != list(grid["size"]):
                bad.append((pid, "grid")); continue
        for s, c, v in zip(st, ce, rv):
            n_checked += 1
            if mask_a[c[2], c[1], c[0]] == 0:
                bad.append((pid, "centre outside ROI")); break
            if patch_roi_stats(mask_a, s, size)[0] != int(v):
                bad.append((pid, "roi_voxels")); break
    return not bad, ("%d patch centres and voxel counts recounted over %d "
                     "sampled patients (seed %d)" % (n_checked, len(sample),
                                                     SAMPLE_SEED))


@check("W", "saved world coordinates map back through the freshly built grid")
def w14():
    bad, worst = [], 0
    for pid in _sample_ids()[:6]:
        ct_r, _, _ = _resampled(pid)
        with _bag(pid) as z:
            w = z["coords_world"]
            idx = z["coords_index"].astype(np.int64)
        back = np.array([ct_r.TransformPhysicalPointToIndex([float(x) for x in p])
                         for p in w], dtype=np.int64)
        e = int(np.abs(back - idx).max()) if len(back) else 0
        worst = max(worst, e)
        if e:
            bad.append(pid)
    return not bad, "round-trip error %d voxels on 6 re-resampled patients" % worst


@check("W", "every CT / GTV+Rim mask geometry is valid (202-patient sweep)")
def w13b():
    p = cfg_path(CFG, "geometry_json")
    if not os.path.isfile(p):
        return False, "geometry summary missing -- run src/verify_geometry.py"
    with open(p, encoding="utf-8") as fh:
        g = json.load(fh)
    ok = (g["n_checked"] == N_EXPECTED and g["n_failed"] == 0
          and g["all_masks_binary"] and g["all_geometry_matched"])
    return ok, ("%d checked, %d failed, masks binary %s, CT/mask geometry "
                "matched %s, ROI HU in [-1024,200] mean %.4f"
                % (g["n_checked"], g["n_failed"], g["all_masks_binary"],
                   g["all_geometry_matched"], g["roi_hu_in_reseg_range_mean"]))


@check("W", "resegmentation to [-1024, 200] HU is in force in the saved features")
def w13c():
    names = _feature_names()
    i_min, i_max = (names.index("original_firstorder_Minimum"),
                    names.index("original_firstorder_Maximum"))
    lo, hi = CFG["radiomics"]["resegment_range"]
    gmin, gmax = np.inf, -np.inf
    for pid in _eligible_ids():
        with _bag(pid) as z:
            f = z["features"]
            if f.shape[0]:
                gmin = min(gmin, float(f[:, i_min].min()))
                gmax = max(gmax, float(f[:, i_max].max()))
    return (gmin >= lo - 1e-3 and gmax <= hi + 1e-3,
            "cohort firstorder HU range [%.1f, %.1f], configured [%d, %d]"
            % (gmin, gmax, lo, hi))


# ===================================== X: provenance, rejections and splits

@check("V", "the 194 shared patients sit on the identical physical lattice")
def v10d():
    ref_ids = {r["PatientID"] for r in load_cohort(REF) if r["eligible"]}
    shared = sorted(ref_ids & set(_eligible_ids()))
    bad_grid, not_subset, ratios = [], [], []
    for pid in shared:
        with _bag(pid) as z:
            b = {tuple(s) for s in z["coords_start_index"].astype(int).tolist()}
            g = (list(z["grid_size"].astype(int)), z["grid_origin"],
                 z["grid_spacing"], z["grid_direction"])
        with _bag(pid, REF_FEAT_DIR) as z:
            a = {tuple(s) for s in z["coords_start_index"].astype(int).tolist()}
            rg = (list(z["grid_size"].astype(int)), z["grid_origin"],
                  z["grid_spacing"], z["grid_direction"])
        if not (g[0] == rg[0] and np.array_equal(g[1], rg[1])
                and np.array_equal(g[2], rg[2]) and np.array_equal(g[3], rg[3])):
            bad_grid.append(pid)
        # ROI_D (gradient) is geometrically contained in ROI_E (GTV+Rim), so on a
        # shared lattice every Gradient patch must reappear as a GTV+Rim patch
        if not a <= b:
            not_subset.append(pid)
        if a:
            ratios.append(len(b) / float(len(a)))
    return (not bad_grid and not not_subset,
            "%d shared patients: %d grid mismatches, %d bags missing a Gradient "
            "patch; GTV+Rim/Gradient patch ratio median %.3f"
            % (len(shared), len(bad_grid), len(not_subset),
               statistics.median(ratios)))


@check("X", "one identical extraction signature across every bag and sidecar")
def x15():
    bad = []
    for pid in _eligible_ids():
        with _bag(pid) as z:
            if str(z["config_signature"]) != SIG:
                bad.append(pid)
        if _qa(pid).get("config_signature") != SIG:
            bad.append(pid)
    if SIG == REF_SIG:
        return False, "GTV+Rim signature collides with the Gradient one"
    return not bad, ("%d mismatches; all %d bags + sidecars == %s (Gradient is "
                     "%s)" % (len(bad), N_EXPECTED, SIG, REF_SIG))


@check("X", "the frozen scientific parameters are byte-identical to Gradient")
def x15b():
    diffs = [k for k in ("preprocessing", "patches") if CFG[k] != REF[k]]
    rad = {k: v for k, v in CFG["radiomics"].items()}
    ref_rad = {k: v for k, v in REF["radiomics"].items()}
    if rad != ref_rad:
        diffs.append("radiomics")
    if CFG["validation"] != REF["validation"]:
        diffs.append("validation")
    return not diffs, ("resampling, patch, radiomics and CV settings identical "
                       "to config/experiment.yaml; differing blocks: %s" % diffs)


@check("X", "patch rejection counts and causes are recorded and reconcile")
def x9():
    reasons = ["center_outside_roi", "roi_voxels_below_min", "pyradiomics_error",
               "nonfinite_features", "feature_count_mismatch"]
    tot = dict((k, 0) for k in reasons)
    cand = kept = 0
    bad = []
    for pid in _eligible_ids():
        q = _qa(pid)
        with _bag(pid) as z:
            n = int(z["features"].shape[0])
        if q["n_retained_patches"] != n:
            bad.append(pid)
        if q["n_candidate_patches"] != q["n_retained_patches"] + sum(
                q["rejected_by_reason"].values()):
            bad.append(pid)
        for k in reasons:
            tot[k] += q["rejected_by_reason"][k]
        cand += q["n_candidate_patches"]
        kept += n
    feat_rej = sum(tot[k] for k in reasons[2:])
    share = feat_rej / float(kept + feat_rej) if (kept + feat_rej) else 0.0
    return (not bad and share < 0.005,
            "%d candidates -> %d kept; center_outside %d, below_min %d, "
            "pyradiomics_error %d (%.4f%% of extractable), nonfinite %d, "
            "count_mismatch %d" % (cand, kept, tot[reasons[0]], tot[reasons[1]],
                                   tot["pyradiomics_error"], 100.0 * share,
                                   tot["nonfinite_features"],
                                   tot["feature_count_mismatch"]))


@check("X", "no patient-level extraction failure in the driver summary")
def x9b():
    p = cfg_path(CFG, "extraction_summary_json", tag="all")
    if not os.path.isfile(p):
        return False, "driver summary missing"
    with open(p, encoding="utf-8") as fh:
        s = json.load(fh)
    ok = (not s["failures"] and s["n_failed"] == 0
          and s["n_extracted_now"] + s["n_skipped_valid"] == N_EXPECTED
          and s["config_signature"] == SIG)
    return ok, ("%d selected, %d extracted, %d skipped valid, %d failed, %.1f s"
                % (s["n_selected"], s["n_extracted_now"], s["n_skipped_valid"],
                   s["n_failed"], s["wall_seconds"]))


@check("X", "the new GTV+Rim split is patient-level, stratified and complete")
def x_split():
    rows, path = _split_rows(CFG)
    folds = sorted({int(r["fold"]) for r in rows})
    if folds != [1, 2, 3, 4, 5]:
        return False, "folds %s" % folds
    seen = set()
    for k in folds:
        tr = {r["PatientID"] for r in rows if int(r["fold"]) == k
              and r["split"] == "train"}
        te = {r["PatientID"] for r in rows if int(r["fold"]) == k
              and r["split"] == "test"}
        if tr & te:
            return False, "train/test overlap in fold %d" % k
        if tr | te != set(_eligible_ids()):
            return False, "fold %d does not cover the cohort" % k
        if seen & te:
            return False, "test folds overlap at fold %d" % k
        adc = sum(1 for r in rows if int(r["fold"]) == k and r["split"] == "test"
                  and int(r["label"]) == 1)
        if not 9 <= adc <= 12:
            return False, "fold %d test ADC = %d, not stratified" % (k, adc)
        seen |= te
    if seen != set(_eligible_ids()):
        return False, "test folds do not partition the cohort"
    coh = _cohort()
    if any(int(r["label"]) != int(coh[r["PatientID"]]["label"]) for r in rows):
        return False, "split labels disagree with the cohort"
    with open(path.replace(".csv", "_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta["splits_file_sha256"] != _sha(path):
        return False, "split file sha256 does not match its meta record"
    return True, ("5 folds, disjoint, %d patients covered exactly once; "
                  "sha256 %s" % (len(seen), meta["splits_file_sha256"][:16]))


@check("X", "the cohort overlap report is consistent with both cohorts")
def x_overlap():
    p = project_path(CFG, "reports", "gradient_vs_gtv_rim_overlap.json")
    if not os.path.isfile(p):
        return False, "overlap report missing -- run src/roi_compare.py"
    with open(p, encoding="utf-8") as fh:
        o = json.load(fh)
    new_ids = set(_eligible_ids())
    ref_ids = {r["PatientID"] for r in load_cohort(REF) if r["eligible"]}
    shared = sorted(new_ids & ref_ids)
    ok = (o["shared"]["n"] == len(shared) == 194
          and set(o["shared_patient_ids"]) == set(shared)
          and o["new_cohort"]["n"] == N_EXPECTED
          and o["reference_cohort"]["n"] == 194
          and o["gtv_rim_only"]["n"] == 8
          and o["gradient_only"]["n"] == 0
          and not o["label_disagreements"])
    return ok, ("%d shared, %d GTV+Rim-only, %d Gradient-only, %d label "
                "conflicts" % (o["shared"]["n"], o["gtv_rim_only"]["n"],
                               o["gradient_only"]["n"],
                               len(o["label_disagreements"])))


# =================================== Y: the frozen Gradient run and read-only

@check("Y", "all 194 frozen Gradient bags are byte-identical")
def y16():
    p = project_path(CFG, "reports", "gradient_frozen_manifest.json")
    with open(p, encoding="utf-8") as fh:
        man = json.load(fh)["files"]
    bags = {k: v for k, v in man.items() if k.startswith("patch_features/gradient/")}
    changed = [k for k, v in bags.items()
               if not os.path.isfile(project_path(CFG, *k.split("/")))
               or _sha(project_path(CFG, *k.split("/"))) != v]
    n_npz = sum(1 for k in bags if k.endswith(".npz"))
    return not changed, ("%d Gradient files (%d .npz) unchanged, %d changed"
                         % (len(bags), n_npz, len(changed)))


@check("Y", "every other frozen Gradient artefact is byte-identical too")
def y16b():
    p = project_path(CFG, "reports", "gradient_frozen_manifest.json")
    with open(p, encoding="utf-8") as fh:
        man = json.load(fh)["files"]
    rest = {k: v for k, v in man.items()
            if not k.startswith("patch_features/gradient/")}
    changed = [k for k, v in rest.items()
               if not os.path.isfile(project_path(CFG, *k.split("/")))
               or _sha(project_path(CFG, *k.split("/"))) != v]
    return not changed, ("%d Phase-1/2/3/3B artefacts (splits, cohort, configs, "
                         "models, predictions, results, attention, "
                         "preprocessing) unchanged; %d changed"
                         % (len(rest), len(changed)))


@check("Y", "the Gradient outer split is unchanged and still matches its meta")
def y17():
    p = project_path(REF, *REF["paths"]["splits_file"].split("/"))
    with open(p.replace(".csv", "_meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    sha = _sha(p)
    if sha != meta["splits_file_sha256"]:
        return False, "sha256 changed"
    rows, _ = _split_rows(REF)
    ids = {r["PatientID"] for r in rows}
    if len(ids) != 194:
        return False, "%d patients in the Gradient split" % len(ids)
    new_p = project_path(CFG, *CFG["paths"]["splits_file"].split("/"))
    if os.path.abspath(new_p) == os.path.abspath(p):
        return False, "the GTV+Rim split would overwrite the Gradient one"
    return True, "194 patients, sha256 %s, separate file from the GTV+Rim split" % sha[:16]


@check("Y", "the external Dataset and Vuong5ROI trees are byte-for-byte untouched")
def y18():
    ok, g = readonly_guard.verify(CFG)
    return ok, "%d files: +%d -%d ~%d" % (g["n_now"], g["n_added"],
                                          g["n_removed"], g["n_changed"])


@check("Y", "every file Phase 4 wrote is inside the project root")
def y18b():
    root = os.path.abspath(CFG["_project_root"])
    outs = [FEAT_DIR, project_path(CFG, "reports"), project_path(CFG, "logs"),
            project_path(CFG, "splits"), project_path(CFG, "cohort"),
            cfg_path(CFG, "qa_patches_dir")]
    outside = [p for p in outs if not os.path.abspath(p).startswith(root)]
    ext = [os.path.abspath(CFG["paths"]["vuong_root"]),
           os.path.abspath(os.path.dirname(CFG["paths"]["dataset_metadata"]))]
    clash = [p for p in outs if any(os.path.abspath(p).startswith(e) for e in ext)]
    return not outside and not clash, ("outside project root: %s; inside a "
                                       "read-only tree: %s" % (outside, clash))


@check("Y", "no GTV+Rim output landed in a Gradient namespace")
def y18c():
    pairs = [("features_dir", FEAT_DIR, REF_FEAT_DIR),
             ("splits_file", project_path(CFG, *CFG["paths"]["splits_file"].split("/")),
              project_path(REF, *REF["paths"]["splits_file"].split("/"))),
             ("cohort_file", project_path(CFG, *CFG["paths"]["cohort_file"].split("/")),
              project_path(REF, *REF["paths"]["cohort_file"].split("/"))),
             ("qa_patches_dir", cfg_path(CFG, "qa_patches_dir"),
              cfg_path(REF, "qa_patches_dir"))]
    same = [k for k, a, b in pairs if os.path.abspath(a) == os.path.abspath(b)]
    stray = [f for f in os.listdir(REF_FEAT_DIR)
             if os.path.splitext(f)[0].split(".")[0] not in
             {r["PatientID"] for r in load_cohort(REF) if r["eligible"]}]
    return not same and not stray, ("%d colliding output paths, %d stray files "
                                    "in the Gradient feature directory"
                                    % (len(same), len(stray)))


def main():
    results = []
    n_fail = 0
    width = max(len(n) for _, n, _ in CHECKS)
    last = None
    for i, (group, name, fn) in enumerate(CHECKS, 1):
        if group != last:
            print("\n--- group %s ---" % group)
            last = group
        try:
            ok, msg = fn()
        except Exception as exc:                        # noqa: BLE001
            ok, msg = False, "EXCEPTION: %s" % str(exc)[:300]
        n_fail += 0 if ok else 1
        print("%-4s %s  %-*s  %s" % ("PASS" if ok else "FAIL", group, width,
                                     name, msg))
        results.append({"n": i, "group": group, "name": name,
                        "pass": bool(ok), "message": msg})

    out = {"roi": CFG["experiment"]["roi"],
           "n_checks": len(CHECKS), "n_passed": len(CHECKS) - n_fail,
           "n_failed": n_fail, "config_signature": SIG,
           "gradient_config_signature": REF_SIG,
           "sample_seed": SAMPLE_SEED, "sample_size": SAMPLE_SIZE,
           "sampled_patients": _sample_ids(), "checks": results}
    path = project_path(CFG, "reports", "phase4_tests.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n%d checks, %d passed, %d failed  ->  %s"
          % (len(CHECKS), len(CHECKS) - n_fail, n_fail, path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
