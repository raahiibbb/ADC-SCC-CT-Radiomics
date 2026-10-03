"""Phase-2 sanity checks: the full 194-patient Gradient extraction.

Run with:
    ./.venv/Scripts/python.exe tests/test_phase2.py

Phase-1's group E only inspected the 6 pilot bags. These checks apply the same
scrutiny to the whole cohort, and re-derive the resampled mask from scratch for
a seeded random sample of *newly extracted* (non-pilot) patients, so the saved
coordinates and ROI voxel counts are verified against the imaging data itself
rather than trusted.

Groups
  G  cohort-wide bag inventory (all 194)
  H  cohort-wide structure and coordinates (all 194, no re-resampling)
  I  independent re-derivation from CT + mask (seeded sample of 12 non-pilot)
  J  provenance, splits and read-only guarantees
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys

import numpy as np
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cohort import load_cohort                                     # noqa: E402
from config_io import extraction_signature, load_config, project_path  # noqa: E402
from extract_patient import validate_existing                      # noqa: E402
from patches import patch_roi_stats                                # noqa: E402
from resample import resample_ct_and_mask                          # noqa: E402
import readonly_guard                                              # noqa: E402

CFG = load_config()
SIG = extraction_signature(CFG)
FEAT_DIR = project_path(CFG, *CFG["paths"]["features_dir"].split("/"))
CHECKS = []

SAMPLE_SIZE = 12
SAMPLE_SEED = 42


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
    p = project_path(CFG, "config", "pilot_patients.txt")
    with open(p, encoding="utf-8") as fh:
        ids = [l.split("#")[0].strip() for l in fh]
    return [i for i in ids if i]


def _bag(pid):
    return np.load(os.path.join(FEAT_DIR, "%s.npz" % pid), allow_pickle=True)


def _qa(pid):
    with open(os.path.join(FEAT_DIR, "%s.qa.json" % pid), encoding="utf-8") as fh:
        return json.load(fh)


def _sample_ids():
    """Seeded, reproducible sample of non-pilot patients, both histologies."""
    coh = _cohort()
    pool = [p for p in _eligible_ids() if p not in set(_pilot_ids())]
    adc = [p for p in pool if coh[p]["label"] == 1]
    scc = [p for p in pool if coh[p]["label"] == 0]
    rng = random.Random(SAMPLE_SEED)
    return sorted(rng.sample(adc, SAMPLE_SIZE // 2)
                  + rng.sample(scc, SAMPLE_SIZE - SAMPLE_SIZE // 2))


def _resampled_mask(pid):
    pdir = os.path.join(CFG["paths"]["vuong_root"], pid)
    ct = sitk.ReadImage(os.path.join(pdir, CFG["paths"]["ct_filename"]))
    mk = sitk.ReadImage(os.path.join(pdir, CFG["experiment"]["roi_filename"]))
    _, mk_r, grid = resample_ct_and_mask(ct, mk, CFG)
    return sitk.GetArrayFromImage(mk_r), grid


# ------------------------------------------------- G: cohort-wide inventory

@check("G", "exactly 194 bags exist, one per eligible patient, none extra")
def g1():
    want = set(_eligible_ids())
    have = set(os.path.splitext(f)[0] for f in os.listdir(FEAT_DIR)
               if f.endswith(".npz"))
    if len(want) != 194:
        return False, "cohort says %d eligible, expected 194" % len(want)
    if want != have:
        return False, "missing %d, extra %d" % (len(want - have), len(have - want))
    return True, "194 bags, exact 1:1 with the eligible cohort"


@check("G", "every bag passes validate_existing under the frozen signature")
def g2():
    coh = _cohort()
    bad = []
    for pid in _eligible_ids():
        ok, why = validate_existing(os.path.join(FEAT_DIR, "%s.npz" % pid),
                                    pid, SIG, coh[pid]["label"])
        if not ok:
            bad.append("%s (%s)" % (pid, why))
    return not bad, "0 invalid" if not bad else "%d invalid: %s" % (len(bad), bad[:3])


@check("G", "every bag has a QA sidecar and retained counts agree with the matrix")
def g3():
    for pid in _eligible_ids():
        if not os.path.isfile(os.path.join(FEAT_DIR, "%s.qa.json" % pid)):
            return False, "%s: no QA sidecar" % pid
        if _qa(pid)["n_retained_patches"] != int(_bag(pid)["features"].shape[0]):
            return False, "%s: sidecar/matrix row count disagree" % pid
    return True, "194 sidecars, all counts agree"


@check("G", "class balance is exactly 49 ADC / 145 SCC")
def g4():
    labels = [int(_bag(p)["label"]) for p in _eligible_ids()]
    n_adc, n_scc = labels.count(1), labels.count(0)
    ok = n_adc == 49 and n_scc == 145
    return ok, "%d ADC / %d SCC" % (n_adc, n_scc)


@check("G", "saved patient_id, histology and label match the cohort metadata")
def g5():
    coh = _cohort()
    for pid in _eligible_ids():
        z = _bag(pid)
        if str(z["patient_id"]) != pid:
            return False, "%s: bag says %s" % (pid, z["patient_id"])
        if int(z["label"]) != int(coh[pid]["label"]):
            return False, "%s: label %d vs cohort %d" % (pid, int(z["label"]),
                                                         coh[pid]["label"])
        if str(z["histology"]) != coh[pid]["Histology"]:
            return False, "%s: histology mismatch" % pid
        if str(z["roi_name"]) != "gradient":
            return False, "%s: roi_name %s" % (pid, z["roi_name"])
    return True, "IDs, labels, histology and ROI name all agree"


@check("G", "no bag is empty")
def g6():
    empty = [p for p in _eligible_ids() if int(_bag(p)["features"].shape[0]) == 0]
    return not empty, "0 empty bags" if not empty else "empty: %s" % empty


# ------------------------------------------- H: cohort-wide structure (194)

@check("H", "all 194 bags carry the same 74 feature names in the same order")
def h1():
    with open(project_path(CFG, "config", "feature_names.txt"), encoding="utf-8") as fh:
        want = [l.strip() for l in fh if l.strip()]
    if len(want) != 74:
        return False, "config/feature_names.txt has %d names" % len(want)
    for pid in _eligible_ids():
        got = [str(x) for x in _bag(pid)["feature_names"]]
        if got != want:
            return False, "%s: feature names differ" % pid
    return True, "74 names, identical order in every bag"


@check("H", "no shape and no filtered features anywhere in the cohort")
def h2():
    names = [str(x) for x in _bag(_eligible_ids()[0])["feature_names"]]
    bad = [n for n in names if "shape" in n.lower() or not n.startswith("original_")]
    return not bad, "0 shape/filtered" if not bad else "found %s" % bad[:3]


@check("H", "every feature value in the cohort is finite")
def h3():
    total = 0
    for pid in _eligible_ids():
        f = _bag(pid)["features"]
        n = int(np.sum(~np.isfinite(f)))
        if n:
            return False, "%s: %d non-finite" % (pid, n)
        total += f.size
    return True, "%d feature values, all finite" % total


@check("H", "features are float32 and every array in a bag has N rows")
def h4():
    for pid in _eligible_ids():
        z = _bag(pid)
        f = z["features"]
        if f.dtype != np.float32:
            return False, "%s: features dtype %s" % (pid, f.dtype)
        n = f.shape[0]
        for k in ["coords_index", "coords_start_index", "coords_world",
                  "bbox_world_min", "bbox_world_max"]:
            if z[k].shape != (n, 3):
                return False, "%s: %s shape %s" % (pid, k, z[k].shape)
        if z["roi_voxels"].shape != (n,) or z["roi_fraction"].shape != (n,):
            return False, "%s: roi_voxels/roi_fraction shape" % pid
    return True, "row counts consistent in all 194 bags"


@check("H", "centre == start + 2, starts on the stride-5 lattice, never duplicated")
def h5():
    for pid in _eligible_ids():
        z = _bag(pid)
        s = z["coords_start_index"]
        if not np.array_equal(z["coords_index"], s + 2):
            return False, "%s: centre != start + 2" % pid
        if np.any(s % 5 != 0):
            return False, "%s: off-lattice start" % pid
        if len(set(tuple(r) for r in s.tolist())) != len(s):
            return False, "%s: duplicate start index" % pid
    return True, "lattice, centring and non-overlap hold cohort-wide"


@check("H", "coords_world round-trips to coords_index for all 194 bags")
def h6():
    worst = 0
    for pid in _eligible_ids():
        z = _bag(pid)
        ref = sitk.Image(*[int(v) for v in z["grid_size"]], sitk.sitkUInt8)
        ref.SetSpacing([float(v) for v in z["grid_spacing"]])
        ref.SetOrigin([float(v) for v in z["grid_origin"]])
        ref.SetDirection([float(v) for v in z["grid_direction"]])
        back = np.array([ref.TransformPhysicalPointToIndex([float(x) for x in p])
                         for p in z["coords_world"]])
        err = int(np.abs(back - z["coords_index"]).max()) if len(back) else 0
        worst = max(worst, err)
        if err:
            return False, "%s: round-trip error %d voxels" % (pid, err)
    return True, "max world<->index error %d voxels" % worst


@check("H", "world bounding boxes are exactly 10 mm per side cohort-wide")
def h7():
    for pid in _eligible_ids():
        z = _bag(pid)
        ext = z["bbox_world_max"] - z["bbox_world_min"]
        if not np.allclose(ext, 10.0, atol=1e-6):
            return False, "%s: extents %s" % (pid, np.unique(np.round(ext, 4))[:3])
    return True, "all patches are 10x10x10 mm"


@check("H", "roi_voxels in [27, 125] and roi_fraction == roi_voxels / 125")
def h8():
    lo = 10 ** 9
    for pid in _eligible_ids():
        z = _bag(pid)
        v = z["roi_voxels"]
        if int(v.min()) < CFG["patches"]["min_roi_voxels"]:
            return False, "%s: min roi_voxels %d" % (pid, v.min())
        if int(v.max()) > 125:
            return False, "%s: max roi_voxels %d" % (pid, v.max())
        if not np.allclose(z["roi_fraction"], v / 125.0, atol=1e-6):
            return False, "%s: roi_fraction inconsistent" % pid
        lo = min(lo, int(v.min()))
    return True, "cohort minimum roi_voxels %d (threshold %d)" % (
        lo, CFG["patches"]["min_roi_voxels"])


@check("H", "resegmentation to [-1024, 200] HU is in force cohort-wide")
def h9():
    names = [str(x) for x in _bag(_eligible_ids()[0])["feature_names"]]
    i_max = names.index("original_firstorder_Maximum")
    i_min = names.index("original_firstorder_Minimum")
    hi, lo = -10 ** 9, 10 ** 9
    for pid in _eligible_ids():
        f = _bag(pid)["features"]
        hi = max(hi, float(f[:, i_max].max()))
        lo = min(lo, float(f[:, i_min].min()))
    if hi > 200.0 or lo < -1024.0:
        return False, "observed HU range [%.1f, %.1f] escapes [-1024, 200]" % (lo, hi)
    return True, "cohort firstorder HU range [%.1f, %.1f]" % (lo, hi)


@check("H", "QA accounting balances: candidates == retained + rejections")
def h10():
    for pid in _eligible_ids():
        qa = _qa(pid)
        if qa["n_retained_patches"] + qa["n_rejected_patches"] != qa["n_candidate_patches"]:
            return False, "%s: accounting does not balance" % pid
        if sum(qa["rejected_by_reason"].values()) != qa["n_rejected_patches"]:
            return False, "%s: reason codes do not sum to rejections" % pid
    return True, "accounting balances for all 194 patients"


@check("H", "no feature column is constant across the whole cohort")
def h11():
    mats = [_bag(p)["features"] for p in _eligible_ids()]
    all_f = np.vstack(mats)
    const = [i for i in range(all_f.shape[1])
             if float(all_f[:, i].max() - all_f[:, i].min()) == 0.0]
    names = [str(x) for x in _bag(_eligible_ids()[0])["feature_names"]]
    return not const, ("0 constant columns over %d patches" % all_f.shape[0]
                       if not const else
                       "constant: %s" % [names[i] for i in const])


# --------------------------------- I: independent re-derivation from imaging

@check("I", "sampled bags: every patch centre is inside the re-derived ROI mask")
def i1():
    ids = _sample_ids()
    for pid in ids:
        a, _ = _resampled_mask(pid)
        c = _bag(pid)["coords_index"]
        inside = a[c[:, 2], c[:, 1], c[:, 0]] > 0
        if not np.all(inside):
            return False, "%s: %d centres outside" % (pid, int((~inside).sum()))
    return True, "%d patients re-derived, all centres inside the ROI" % len(ids)


@check("I", "sampled bags: roi_voxels recount from the re-derived mask matches")
def i2():
    ids = _sample_ids()
    n = 0
    for pid in ids:
        a, _ = _resampled_mask(pid)
        z = _bag(pid)
        for s, v in zip(z["coords_start_index"], z["roi_voxels"]):
            m, _ = patch_roi_stats(a, s, CFG["patches"]["size_voxels"])
            if m != int(v):
                return False, "%s @ %s: stored %d, recount %d" % (pid, s.tolist(), v, m)
            n += 1
    return True, "%d patches recounted against the mask, all match" % n


@check("I", "sampled bags: the saved grid matches a freshly derived grid")
def i3():
    for pid in _sample_ids():
        z = _bag(pid)
        _, grid = _resampled_mask(pid)
        if [int(v) for v in z["grid_size"]] != [int(v) for v in grid["size"]]:
            return False, "%s: grid size differs" % pid
        if not np.allclose(z["grid_spacing"], grid["spacing"], atol=1e-9):
            return False, "%s: grid spacing differs" % pid
        if not np.allclose(z["grid_origin"], grid["origin"], atol=1e-9):
            return False, "%s: grid origin differs" % pid
        if not np.allclose(z["grid_direction"], grid["direction"], atol=1e-9):
            return False, "%s: grid direction differs" % pid
        if not np.allclose(z["grid_spacing"], CFG["preprocessing"]["resample_spacing_mm"]):
            return False, "%s: spacing is not the configured 2 mm" % pid
    return True, "grids reproduce exactly for the sample"


@check("I", "sampled bags: no retained patch escapes the resampled volume")
def i4():
    for pid in _sample_ids():
        z = _bag(pid)
        size = np.array([int(v) for v in z["grid_size"]])
        s = z["coords_start_index"]
        if np.any(s < 0) or np.any(s + np.array(CFG["patches"]["size_voxels"]) > size):
            return False, "%s: a block leaves the volume" % pid
    return True, "all blocks fully inside the volume"


# --------------------------------------- J: provenance, split, read-only

@check("J", "config_signature is identical in every bag and every sidecar")
def j1():
    for pid in _eligible_ids():
        if str(_bag(pid)["config_signature"]) != SIG:
            return False, "%s: bag signature differs" % pid
        if _qa(pid).get("config_signature") != SIG:
            return False, "%s: sidecar signature differs" % pid
    return True, "all 194 == %s" % SIG


@check("J", "npz_version and patch geometry provenance are uniform")
def j2():
    for pid in _eligible_ids():
        z = _bag(pid)
        if int(z["npz_version"]) != 1:
            return False, "%s: npz_version %s" % (pid, z["npz_version"])
        if [int(v) for v in z["patch_size"]] != list(CFG["patches"]["size_voxels"]):
            return False, "%s: patch_size %s" % (pid, z["patch_size"])
        if ([int(v) for v in np.atleast_1d(z["patch_stride"])]
                != [int(v) for v in CFG["patches"]["stride_voxels"]]):
            return False, "%s: patch_stride %s" % (pid, z["patch_stride"])
    return True, "npz v1, 5x5x5 patches, stride 5 everywhere"


@check("J", "the frozen split still covers exactly the 194 extracted patients")
def j3():
    with open(project_path(CFG, "splits", "stratified_5fold.csv"),
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    split_ids = sorted(set(r["PatientID"] for r in rows))
    bags = sorted(os.path.splitext(f)[0] for f in os.listdir(FEAT_DIR)
                  if f.endswith(".npz"))
    if split_ids != bags:
        return False, "%d split ids vs %d bags" % (len(split_ids), len(bags))
    return True, "%d patients, exact match" % len(bags)


@check("J", "the split file is unchanged (sha256 matches its meta)")
def j4():
    import hashlib
    p = project_path(CFG, "splits", "stratified_5fold.csv")
    with open(p, "rb") as fh:
        got = hashlib.sha256(fh.read()).hexdigest()
    with open(project_path(CFG, "splits", "stratified_5fold_meta.json"),
              encoding="utf-8") as fh:
        want = json.load(fh)["splits_file_sha256"]
    return got == want, "%s" % got[:16]


@check("J", "split labels still agree with every bag's saved label")
def j5():
    with open(project_path(CFG, "splits", "stratified_5fold.csv"),
              encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if int(_bag(r["PatientID"])["label"]) != int(r["label"]):
                return False, "%s: split/bag label disagree" % r["PatientID"]
    return True, "all fold rows agree with the bags"


@check("J", "no train/test overlap in any fold, test folds partition the cohort")
def j6():
    with open(project_path(CFG, "splits", "stratified_5fold.csv"),
              encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    seen = []
    for fold in sorted(set(int(r["fold"]) for r in rows)):
        tr = set(r["PatientID"] for r in rows
                 if int(r["fold"]) == fold and r["split"] == "train")
        te = set(r["PatientID"] for r in rows
                 if int(r["fold"]) == fold and r["split"] == "test")
        if tr & te:
            return False, "fold %d: %d patients in both" % (fold, len(tr & te))
        seen += list(te)
    if len(seen) != len(set(seen)) or len(set(seen)) != 194:
        return False, "test folds cover %d unique of %d rows" % (len(set(seen)),
                                                                 len(seen))
    return True, "5 folds, disjoint, 194 patients covered exactly once"


@check("J", "the external Dataset and Vuong5ROI trees are unchanged")
def j7():
    ok, d = readonly_guard.verify(CFG)
    return ok, "%d files: +%d -%d ~%d" % (d["n_now"], d["n_added"], d["n_removed"],
                                          d["n_changed"])


@check("J", "every file written by Phase 2 is inside the project root")
def j8():
    root = os.path.abspath(project_path(CFG))
    for d in [FEAT_DIR, project_path(CFG, "reports"), project_path(CFG, "logs")]:
        if not os.path.abspath(d).startswith(root):
            return False, "%s escapes the project root" % d
    return True, "features, reports and logs all under %s" % root


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

    out = {"n_checks": len(CHECKS), "n_passed": len(CHECKS) - n_fail,
           "n_failed": n_fail, "config_signature": SIG,
           "sample_seed": SAMPLE_SEED, "sample_size": SAMPLE_SIZE,
           "sampled_patients": _sample_ids(), "checks": results}
    path = project_path(CFG, "reports", "phase2_tests.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n%d checks, %d passed, %d failed  ->  %s"
          % (len(CHECKS), len(CHECKS) - n_fail, n_fail, path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
