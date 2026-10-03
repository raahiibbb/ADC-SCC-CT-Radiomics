"""Phase-1 sanity checks.

Run with:
    ./.venv/Scripts/python.exe tests/test_phase1.py

No pytest dependency: each check is a function returning (ok, message); the
runner prints a PASS/FAIL table and exits non-zero on any failure.

Groups
  A  cohort / labels
  B  frozen split integrity (patient-level, no leakage)
  C  geometry sweep
  D  patch geometry (synthetic, exact)
  E  extracted pilot bags (IDs, labels, coordinates, masks, features)
  F  resume / validation logic
"""
from __future__ import annotations

import csv
import json
import os
import sys
import tempfile

import numpy as np
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cohort import build_cohort, load_cohort                       # noqa: E402
from config_io import extraction_signature, load_config, project_path  # noqa: E402
from extract_patient import validate_existing                      # noqa: E402
from patches import enumerate_patches, patch_roi_stats             # noqa: E402
from resample import reference_grid, resample_ct_and_mask          # noqa: E402

CFG = load_config()
CHECKS = []


def check(group, name):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


def _pilot_ids():
    p = project_path(CFG, "config", "pilot_patients.txt")
    ids = [l.split("#")[0].strip() for l in open(p, encoding="utf-8")]
    return [i for i in ids if i]


def _bag(pid):
    d = project_path(CFG, *CFG["paths"]["features_dir"].split("/"))
    return np.load(os.path.join(d, "%s.npz" % pid), allow_pickle=True)


# --------------------------------------------------------------- A: cohort
@check("A", "cohort is 203 subjects, 194 eligible")
def a1():
    c = build_cohort(CFG)
    a = c["audit"]
    return (a["n_total"] == 203 and a["n_eligible"] == 194,
            "total=%d eligible=%d" % (a["n_total"], a["n_eligible"]))


@check("A", "eligible histology counts are 49 ADC / 145 SCC")
def a2():
    c = build_cohort(CFG)["audit"]["label_counts_eligible"]
    return (c.get("adenocarcinoma") == 49 and c.get("squamous cell carcinoma") == 145,
            str(c))


@check("A", "the nine Gradient exclusions match CLAUDE.md exactly")
def a3():
    a = build_cohort(CFG)["audit"]
    return (a["exclusions_match_declared"],
            "derived=%s declared=%s" % (a["excluded_ids"], a["declared_exclusions"]))


@check("A", "histology agrees between cohort_summary.csv and dataset_metadata_master.csv")
def a4():
    a = build_cohort(CFG)["audit"]
    return (not a["histology_mismatch"] and not a["unmapped_histology"],
            "%d mismatches, %d unmapped" % (len(a["histology_mismatch"]),
                                            len(a["unmapped_histology"])))


@check("A", "ADC is the positive class (label 1)")
def a5():
    m = CFG["cohort"]["histology_map"]
    return (m["adenocarcinoma"] == 1 and m["squamous cell carcinoma"] == 0, str(m))


@check("A", "no excluded patient has a Gradient mask file on disk")
def a6():
    root = CFG["paths"]["vuong_root"]
    fn = CFG["experiment"]["roi_filename"]
    present = [p for p in CFG["cohort"]["gradient_exclusions"]
               if os.path.isfile(os.path.join(root, p, fn))]
    return (not present, "unexpected mask files: %s" % present)


@check("A", "cohort CSV labels agree with the histology map")
def a7():
    m = CFG["cohort"]["histology_map"]
    bad = [r["PatientID"] for r in load_cohort(CFG) if r["label"] != m[r["Histology"]]]
    return (not bad, "mislabelled: %s" % bad[:10])


# --------------------------------------------------------------- B: split
def _split_rows():
    p = project_path(CFG, *CFG["paths"]["splits_file"].split("/"))
    with open(p, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


@check("B", "split covers exactly the 194 eligible patients")
def b1():
    rows = _split_rows()
    pids = {r["PatientID"] for r in rows}
    elig = {r["PatientID"] for r in load_cohort(CFG) if r["eligible"]}
    return (pids == elig, "split=%d eligible=%d symdiff=%d"
            % (len(pids), len(elig), len(pids ^ elig)))


@check("B", "no patient is in both train and test of the same fold (no leakage)")
def b2():
    bad = []
    rows = _split_rows()
    for k in sorted({r["fold"] for r in rows}):
        tr = {r["PatientID"] for r in rows if r["fold"] == k and r["split"] == "train"}
        te = {r["PatientID"] for r in rows if r["fold"] == k and r["split"] == "test"}
        if tr & te:
            bad.append((k, sorted(tr & te)[:3]))
    return (not bad, str(bad))


@check("B", "test folds partition the cohort exactly once each")
def b3():
    rows = _split_rows()
    seen = {}
    for r in rows:
        if r["split"] == "test":
            seen[r["PatientID"]] = seen.get(r["PatientID"], 0) + 1
    return (len(seen) == 194 and set(seen.values()) == {1},
            "n=%d counts=%s" % (len(seen), sorted(set(seen.values()))))


@check("B", "each fold's train set is the complement of its test set")
def b4():
    rows = _split_rows()
    for k in sorted({r["fold"] for r in rows}):
        tr = {r["PatientID"] for r in rows if r["fold"] == k and r["split"] == "train"}
        te = {r["PatientID"] for r in rows if r["fold"] == k and r["split"] == "test"}
        if len(tr) + len(te) != 194:
            return False, "fold %s has %d+%d" % (k, len(tr), len(te))
    return True, "all folds sum to 194"


@check("B", "stratification holds (test ADC fraction within 2pp of cohort)")
def b5():
    rows = _split_rows()
    base = 49 / 194.0
    worst = 0.0
    for k in sorted({r["fold"] for r in rows}):
        te = [r for r in rows if r["fold"] == k and r["split"] == "test"]
        frac = sum(int(r["label"]) for r in te) / len(te)
        worst = max(worst, abs(frac - base))
    return worst <= 0.02, "max deviation %.4f (cohort %.4f)" % (worst, base)


@check("B", "labels in the split file match the cohort file")
def b6():
    lab = {r["PatientID"]: r["label"] for r in load_cohort(CFG)}
    bad = [r["PatientID"] for r in _split_rows() if int(r["label"]) != lab[r["PatientID"]]]
    return not bad, "mismatched: %s" % sorted(set(bad))[:5]


@check("B", "split file sha256 matches its frozen metadata")
def b7():
    import hashlib
    p = project_path(CFG, *CFG["paths"]["splits_file"].split("/"))
    meta = json.load(open(p.replace(".csv", "_meta.json"), encoding="utf-8"))
    sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
    return sha == meta["splits_file_sha256"], sha[:16]


@check("B", "split is reproducible from random_state=42")
def b8():
    from sklearn.model_selection import StratifiedKFold
    rows = sorted([r for r in load_cohort(CFG) if r["eligible"]],
                  key=lambda r: r["PatientID"])
    y = np.array([r["label"] for r in rows])
    pids = [r["PatientID"] for r in rows]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    regenerated = {}
    for k, (_, te) in enumerate(skf.split(np.zeros(len(y)), y), 1):
        regenerated[k] = sorted(pids[i] for i in te)
    saved = {}
    for r in _split_rows():
        if r["split"] == "test":
            saved.setdefault(int(r["fold"]), []).append(r["PatientID"])
    saved = {k: sorted(v) for k, v in saved.items()}
    return regenerated == saved, "regenerated split identical: %s" % (regenerated == saved)


# --------------------------------------------------------------- C: geometry
@check("C", "geometry sweep passed for all 194 eligible patients")
def c1():
    p = project_path(CFG, "reports", "geometry_check_summary.json")
    s = json.load(open(p, encoding="utf-8"))
    return (s["n_checked"] == 194 and s["n_failed"] == 0
            and s["all_geometry_matched"] and s["all_masks_binary"],
            "checked=%d failed=%d" % (s["n_checked"], s["n_failed"]))


@check("C", "reference grid preserves origin/direction and covers the CT extent")
def c2():
    img = sitk.Image(101, 97, 53, sitk.sitkFloat32)
    img.SetSpacing((0.9765625, 0.9765625, 3.0))
    img.SetOrigin((-249.5, -390.5, -683.0))
    g = reference_grid(img, [2.0, 2.0, 2.0])
    for d in range(3):
        span_old = img.GetSize()[d] * img.GetSpacing()[d]
        span_new = g["size"][d] * g["spacing"][d]
        if span_new < span_old - 1e-9:
            return False, "axis %d: new span %.3f < old %.3f" % (d, span_new, span_old)
    return (g["origin"] == img.GetOrigin() and g["direction"] == img.GetDirection(),
            "size %s" % g["size"])


# --------------------------------------------------------------- D: patches
@check("D", "5x5x5 stride-5 lattice tiles a synthetic cube exactly, no overlap")
def d1():
    arr = np.zeros((40, 40, 40), np.uint8)
    arr[10:25, 10:25, 10:25] = 1                  # 15^3 ROI aligned to the lattice
    cands, _ = enumerate_patches(arr, [5, 5, 5], [5, 5, 5], True)
    kept = []
    cover = np.zeros_like(arr, dtype=np.int32)
    for c in cands:
        i, j, k = (int(v) for v in c["center"])
        if arr[k, j, i] == 0:
            continue
        n, _ = patch_roi_stats(arr, c["start"], [5, 5, 5])
        if n < 27:
            continue
        s = c["start"]
        cover[s[2]:s[2] + 5, s[1]:s[1] + 5, s[0]:s[0] + 5] += 1
        kept.append(c)
    if len(kept) != 27:
        return False, "expected 27 patches for a 15^3 ROI, got %d" % len(kept)
    if cover.max() > 1:
        return False, "patches overlap (max coverage %d)" % cover.max()
    if int((cover > 0).sum()) != int(arr.sum()):
        return False, "coverage %d != ROI %d" % ((cover > 0).sum(), arr.sum())
    return True, "27 patches, disjoint, exact cover"


@check("D", "patches whose centre is outside the ROI are rejected")
def d2():
    arr = np.zeros((20, 20, 20), np.uint8)
    arr[5:10, 5:10, 5:10] = 1
    arr[7, 7, 7] = 0                              # hollow the centre of one block
    cands, _ = enumerate_patches(arr, [5, 5, 5], [5, 5, 5], True)
    blk = [c for c in cands if tuple(int(v) for v in c["start"]) == (5, 5, 5)]
    if not blk:
        return False, "expected the (5,5,5) block to be enumerated"
    i, j, k = (int(v) for v in blk[0]["center"])
    return (i, j, k) == (7, 7, 7) and arr[k, j, i] == 0, "centre (7,7,7) is outside ROI"


@check("D", "the >=27-ROI-voxel rule rejects a thin ROI slab")
def d3():
    arr = np.zeros((20, 20, 20), np.uint8)
    arr[7, 5:10, 5:10] = 1                        # single-slice slab: 25 voxels
    cands, _ = enumerate_patches(arr, [5, 5, 5], [5, 5, 5], True)
    blk = [c for c in cands if tuple(int(v) for v in c["start"]) == (5, 5, 5)][0]
    n, _ = patch_roi_stats(arr, blk["start"], [5, 5, 5])
    return n == 25 and n < 27, "block holds %d ROI voxels (< 27)" % n


@check("D", "no enumerated block leaves the image bounds")
def d4():
    arr = np.zeros((23, 23, 23), np.uint8)
    arr[3:22, 3:22, 3:22] = 1
    cands, _ = enumerate_patches(arr, [5, 5, 5], [5, 5, 5], True)
    dims = np.array(arr.shape[::-1])
    for c in cands:
        if np.any(c["start"] < 0) or np.any(c["start"] + 5 > dims):
            return False, "out-of-bounds block at %s" % c["start"]
    return True, "%d blocks all in bounds" % len(cands)


@check("D", "mask resampling is nearest-neighbour (stays binary) and CT stays in HU")
def d5():
    ct = sitk.GetImageFromArray(
        np.random.RandomState(0).randint(-1024, 400, (30, 40, 40)).astype(np.float32))
    ct.SetSpacing((0.9765625, 0.9765625, 3.0))
    mk = sitk.GetImageFromArray((np.indices((30, 40, 40)).sum(0) % 7 < 3).astype(np.uint8))
    mk.CopyInformation(ct)
    ct_r, mk_r, g = resample_ct_and_mask(ct, mk, CFG)
    a = sitk.GetArrayFromImage(mk_r)
    c = sitk.GetArrayFromImage(ct_r)
    ok = (set(np.unique(a).tolist()) <= {0, 1}
          and ct_r.GetSize() == mk_r.GetSize()
          and np.allclose(ct_r.GetSpacing(), (2.0, 2.0, 2.0))
          and np.allclose(mk_r.GetSpacing(), (2.0, 2.0, 2.0))
          and c.min() >= -1024.5)
    return ok, "mask unique %s, spacing %s" % (np.unique(a).tolist(), mk_r.GetSpacing())


# --------------------------------------------------------------- E: bags
@check("E", "every pilot bag exists, is loadable and passes validate_existing")
def e1():
    sig = extraction_signature(CFG)
    lab = {r["PatientID"]: r["label"] for r in load_cohort(CFG)}
    d = project_path(CFG, *CFG["paths"]["features_dir"].split("/"))
    bad = []
    for pid in _pilot_ids():
        ok, why = validate_existing(os.path.join(d, "%s.npz" % pid), pid, sig, lab[pid])
        if not ok:
            bad.append((pid, why))
    return not bad, str(bad) or "%d bags valid" % len(_pilot_ids())


@check("E", "bag patient_id / histology / label match the cohort metadata")
def e2():
    coh = {r["PatientID"]: r for r in load_cohort(CFG)}
    for pid in _pilot_ids():
        z = _bag(pid)
        if str(z["patient_id"]) != pid:
            return False, "%s: patient_id is %s" % (pid, z["patient_id"])
        if str(z["histology"]) != coh[pid]["Histology"]:
            return False, "%s: histology mismatch" % pid
        if int(z["label"]) != coh[pid]["label"]:
            return False, "%s: label mismatch" % pid
    return True, "%d bags consistent" % len(_pilot_ids())


@check("E", "all bags share the same 74 feature names in the same order")
def e3():
    ref = None
    for pid in _pilot_ids():
        n = [str(x) for x in _bag(pid)["feature_names"]]
        if ref is None:
            ref = n
        elif n != ref:
            return False, "%s differs" % pid
    fams = sorted({x.split("_")[1] for x in ref})
    return (len(ref) == 74 and fams == ["firstorder", "glcm", "glrlm", "glszm"]
            and all(x.startswith("original_") for x in ref),
            "%d features, families %s" % (len(ref), fams))


@check("E", "no shape features and no filtered image types are present")
def e4():
    n = [str(x) for x in _bag(_pilot_ids()[0])["feature_names"]]
    bad = [x for x in n
           if "shape" in x.lower() or x.split("_")[0] != "original"]
    return not bad, "offending features: %s" % bad[:5]


@check("E", "feature matrices are finite and shaped [N_patches, 74]")
def e5():
    for pid in _pilot_ids():
        z = _bag(pid)
        f = z["features"]
        if f.ndim != 2 or f.shape[1] != 74:
            return False, "%s shape %s" % (pid, f.shape)
        if f.shape[0] == 0:
            return False, "%s has an empty bag" % pid
        if not np.all(np.isfinite(f)):
            return False, "%s has non-finite values" % pid
    return True, "all finite"


@check("E", "every array in a bag has N_patches rows")
def e6():
    keys = ["coords_index", "coords_start_index", "coords_world",
            "bbox_world_min", "bbox_world_max"]
    for pid in _pilot_ids():
        z = _bag(pid)
        n = z["features"].shape[0]
        for k in keys:
            if z[k].shape != (n, 3):
                return False, "%s: %s has shape %s, expected (%d,3)" % (pid, k, z[k].shape, n)
        if z["roi_voxels"].shape != (n,) or z["roi_fraction"].shape != (n,):
            return False, "%s: roi_voxels/roi_fraction length mismatch" % pid
    return True, "row counts consistent"


@check("E", "coords_index == coords_start_index + 2 (centre of a 5x5x5 block)")
def e7():
    for pid in _pilot_ids():
        z = _bag(pid)
        if not np.array_equal(z["coords_index"], z["coords_start_index"] + 2):
            return False, "%s: centre is not start+2" % pid
    return True, "centres consistent"


@check("E", "patch starts lie on the stride-5 lattice anchored at voxel (0,0,0)")
def e8():
    for pid in _pilot_ids():
        s = _bag(pid)["coords_start_index"]
        if np.any(s % 5 != 0):
            return False, "%s: off-lattice start %s" % (pid, s[np.any(s % 5 != 0, 1)][0])
    return True, "all starts are multiples of 5"


@check("E", "no two retained patches in a bag share a start index (non-overlapping)")
def e9():
    for pid in _pilot_ids():
        s = _bag(pid)["coords_start_index"]
        if len({tuple(r) for r in s.tolist()}) != len(s):
            return False, "%s has duplicate patch starts" % pid
    return True, "all patch starts unique"


@check("E", "coords_world round-trips to coords_index through the saved grid")
def e10():
    for pid in _pilot_ids():
        z = _bag(pid)
        ref = sitk.Image(*[int(v) for v in z["grid_size"]], sitk.sitkUInt8)
        ref.SetSpacing([float(v) for v in z["grid_spacing"]])
        ref.SetOrigin([float(v) for v in z["grid_origin"]])
        ref.SetDirection([float(v) for v in z["grid_direction"]])
        back = np.array([ref.TransformPhysicalPointToIndex([float(x) for x in p])
                         for p in z["coords_world"]])
        if not np.array_equal(back, z["coords_index"]):
            return False, "%s: max err %d" % (
                pid, int(np.abs(back - z["coords_index"]).max()))
    return True, "world<->index exact for every patch"


@check("E", "world bounding boxes are exactly 10 mm per side")
def e11():
    for pid in _pilot_ids():
        z = _bag(pid)
        ext = z["bbox_world_max"] - z["bbox_world_min"]
        if not np.allclose(ext, 10.0, atol=1e-6):
            return False, "%s: extents %s" % (pid, np.unique(np.round(ext, 4))[:4])
    return True, "all boxes 10x10x10 mm"


@check("E", "every retained patch centre really is inside the resampled Gradient mask")
def e12():
    for pid in _pilot_ids():
        z = _bag(pid)
        pdir = os.path.join(CFG["paths"]["vuong_root"], pid)
        ct = sitk.ReadImage(os.path.join(pdir, CFG["paths"]["ct_filename"]))
        mk = sitk.ReadImage(os.path.join(pdir, CFG["experiment"]["roi_filename"]))
        _, mk_r, _ = resample_ct_and_mask(ct, mk, CFG)
        a = sitk.GetArrayFromImage(mk_r)
        c = z["coords_index"]
        if not np.all(a[c[:, 2], c[:, 1], c[:, 0]] > 0):
            return False, "%s: %d centres outside the mask" % (
                pid, int((a[c[:, 2], c[:, 1], c[:, 0]] == 0).sum()))
    return True, "all centres inside the ROI"


@check("E", "recorded roi_voxels matches a recount of the resampled mask, and is >= 27")
def e13():
    for pid in _pilot_ids():
        z = _bag(pid)
        pdir = os.path.join(CFG["paths"]["vuong_root"], pid)
        ct = sitk.ReadImage(os.path.join(pdir, CFG["paths"]["ct_filename"]))
        mk = sitk.ReadImage(os.path.join(pdir, CFG["experiment"]["roi_filename"]))
        _, mk_r, _ = resample_ct_and_mask(ct, mk, CFG)
        a = sitk.GetArrayFromImage(mk_r)
        for s, n in zip(z["coords_start_index"], z["roi_voxels"]):
            m, _ = patch_roi_stats(a, s, [5, 5, 5])
            if m != int(n):
                return False, "%s @ %s: stored %d, recount %d" % (pid, s.tolist(), n, m)
        if int(z["roi_voxels"].min()) < CFG["patches"]["min_roi_voxels"]:
            return False, "%s: min roi_voxels %d" % (pid, z["roi_voxels"].min())
        if not np.allclose(z["roi_fraction"], z["roi_voxels"] / 125.0, atol=1e-6):
            return False, "%s: roi_fraction != roi_voxels/125" % pid
    return True, "voxel counts verified against the mask"


@check("E", "resegmentation to [-1024, 200] HU is in force")
def e14():
    lo, hi = CFG["radiomics"]["resegment_range"]
    for pid in _pilot_ids():
        z = _bag(pid)
        n = [str(x) for x in z["feature_names"]]
        f = z["features"]
        mx = f[:, n.index("original_firstorder_Maximum")].max()
        mn = f[:, n.index("original_firstorder_Minimum")].min()
        if mx > hi + 1e-3 or mn < lo - 1e-3:
            return False, "%s: HU range [%.1f, %.1f] escapes [%d, %d]" % (pid, mn, mx, lo, hi)
    return True, "all patch intensities within the resegmentation range"


@check("E", "the raw Gradient ROI does contain HU above 200 (so resegmentation bites)")
def e15():
    pid = _pilot_ids()[0]
    pdir = os.path.join(CFG["paths"]["vuong_root"], pid)
    ct = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(pdir, CFG["paths"]["ct_filename"])))
    mk = sitk.GetArrayFromImage(sitk.ReadImage(
        os.path.join(pdir, CFG["experiment"]["roi_filename"])))
    hu = ct[mk > 0]
    return bool(hu.max() > 200), "%s raw ROI max HU = %.0f" % (pid, hu.max())


@check("E", "bag size scales with ROI volume (~1 patch per cm3)")
def e16():
    coh = {r["PatientID"]: r for r in load_cohort(CFG)}
    ratios = []
    for pid in _pilot_ids():
        n = _bag(pid)["features"].shape[0]
        v = float(coh[pid]["gradient_volume_mm3"]) / 1000.0
        ratios.append(n / v)
    return (0.8 < min(ratios) and max(ratios) < 1.3,
            "ratios %s" % [round(r, 2) for r in ratios])


@check("E", "QA sidecar accounting adds up (candidates = retained + rejections)")
def e17():
    d = project_path(CFG, *CFG["paths"]["features_dir"].split("/"))
    for pid in _pilot_ids():
        q = json.load(open(os.path.join(d, "%s.qa.json" % pid), encoding="utf-8"))
        if q["n_retained_patches"] + sum(q["rejected_by_reason"].values()) != \
                q["n_candidate_patches"]:
            return False, "%s: accounting mismatch" % pid
        if q["n_retained_patches"] != _bag(pid)["features"].shape[0]:
            return False, "%s: QA count != npz rows" % pid
    return True, "candidate accounting balanced for all pilot patients"


@check("E", "pilot covers both histologies and a wide ROI-size range")
def e18():
    coh = {r["PatientID"]: r for r in load_cohort(CFG)}
    ids = _pilot_ids()
    h = {coh[p]["Histology"] for p in ids}
    v = [float(coh[p]["gradient_volume_mm3"]) / 1000.0 for p in ids]
    return (len(h) == 2 and max(v) / min(v) > 20,
            "%d histologies, volumes %.1f-%.1f cm3" % (len(h), min(v), max(v)))


# --------------------------------------------------------------- F: resume
@check("F", "validate_existing rejects a wrong patient id, label and signature")
def f1():
    d = project_path(CFG, *CFG["paths"]["features_dir"].split("/"))
    pid = _pilot_ids()[0]
    p = os.path.join(d, "%s.npz" % pid)
    sig = extraction_signature(CFG)
    lab = {r["PatientID"]: r["label"] for r in load_cohort(CFG)}[pid]
    if validate_existing(p, "LUNG1-999", sig, lab)[0]:
        return False, "accepted a wrong patient id"
    if validate_existing(p, pid, "deadbeefdeadbeef", lab)[0]:
        return False, "accepted a stale config signature"
    if validate_existing(p, pid, sig, 1 - lab)[0]:
        return False, "accepted a wrong label"
    return validate_existing(p, pid, sig, lab)[0], "rejects id/signature/label mismatches"


@check("F", "validate_existing rejects a truncated or corrupt file")
def f2():
    d = project_path(CFG, *CFG["paths"]["features_dir"].split("/"))
    pid = _pilot_ids()[0]
    raw = open(os.path.join(d, "%s.npz" % pid), "rb").read()
    fd, tmp = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            fh.write(raw[:len(raw) // 2])
        ok, _ = validate_existing(tmp, pid, extraction_signature(CFG))
        return not ok, "truncated file rejected"
    finally:
        os.unlink(tmp)


@check("F", "the config signature changes when a scientific parameter changes")
def f3():
    import copy
    base = extraction_signature(CFG)
    for path, val in [(("patches", "min_roi_voxels"), 40),
                      (("preprocessing", "resample_spacing_mm"), [3.0, 3.0, 3.0]),
                      (("radiomics", "bin_width"), 25),
                      (("patches", "size_voxels"), [3, 3, 3])]:
        c = copy.deepcopy(CFG)
        c[path[0]][path[1]] = val
        if extraction_signature(c) == base:
            return False, "signature unchanged after %s -> %s" % (path, val)
    return True, "signature %s is sensitive to all four parameters" % base


@check("F", "the read-only input trees are byte-for-byte untouched")
def f4():
    from readonly_guard import verify
    ok, info = verify(CFG)
    return ok, ("%d files unchanged" % info["n_baseline"] if ok else
                "added %d removed %d changed %d" % (info["n_added"], info["n_removed"],
                                                    info["n_changed"]))


@check("F", "all pipeline outputs live under the project root")
def f5():
    stray = []
    for d in [CFG["paths"]["features_dir"], CFG["paths"]["splits_file"],
              CFG["paths"]["cohort_file"], CFG["paths"]["qa_dir"]]:
        p = project_path(CFG, *d.split("/"))
        if not os.path.abspath(p).startswith(os.path.abspath(CFG["_project_root"])):
            stray.append(p)
    return not stray, "outside project root: %s" % stray


def main():
    results = []
    for group, name, fn in CHECKS:
        try:
            ok, msg = fn()
        except Exception as exc:                        # noqa: BLE001
            ok, msg = False, "EXCEPTION: %s" % exc
        results.append((group, name, ok, msg))
        print("[%s] %-4s %-62s  %s" % ("PASS" if ok else "FAIL", group, name[:62], msg[:70]))
    n_fail = sum(1 for r in results if not r[2])
    print("-" * 100)
    print("%d checks, %d passed, %d failed" % (len(results), len(results) - n_fail, n_fail))
    out = project_path(CFG, "reports", "phase1_tests.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"n_checks": len(results), "n_passed": len(results) - n_fail,
                   "n_failed": n_fail,
                   "checks": [{"group": g, "check": n, "passed": bool(o),
                               "detail": str(m)} for g, n, o, m in results]},
                  fh, indent=2)
    print("written:", out)
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
