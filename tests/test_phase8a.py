"""Phase-8A audit: external cohort, regions, compatibility, scope, freeze.

Run with:
    ./.venv/Scripts/python.exe tests/test_phase8a.py

Phase 8A builds an UNTOUCHED external NSCLC-Radiogenomics cohort and prepares
it for a LATER, LOCKED external test.  It trains nothing and computes no
classification result, so most of these checks are about what must NOT exist.
Every check re-derives its answer from files on disk.

Groups
  A  cohort, eligibility and provenance                        (1-7)
  B  regions: GTV, the 8 mm rim and GTV+Rim                    (8-12)
  C  preprocessing compatibility with the frozen LUNG1 arms    (13-18)
  D  the patch sampler is the frozen LUNG1 sampler             (19-22)
  E  scope: no model, no metric, no tuning on external labels  (23-28)
  F  integrity and instance identity                           (29-32)
  G  frozen artefacts and the read-only external trees         (33-35)
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import SimpleITK as sitk

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from phase8a_tcia import load_cfg, ppath                                   # noqa: E402
from phase8a_regions import signed_distance_field, zyx_spacing             # noqa: E402
from phase8a_global import derived_phase7a_cfg                             # noqa: E402
from phase8a_patches import derived_gtv_rim_cfg                            # noqa: E402
from phase8a_images import derived_phase6a_cfg                             # noqa: E402
from resample import reference_grid                                        # noqa: E402
from patches import enumerate_patches, patch_roi_stats                     # noqa: E402
from resample import resample_to_grid                                      # noqa: E402

CFG = load_cfg(os.path.join(ROOT, "config", "phase8a_external.yaml"))
CHECKS = []

# Phase-8A source files whose CONTENT the scope scans inspect.  The audit file
# itself is deliberately excluded: a scanner carrying its own search lists would
# self-hit (the defect found in Phases 5C, 6A, 6B and 7A).
PHASE8A_PY = ("src/phase8a_tcia.py", "src/phase8a_dicom.py",
              "src/phase8a_regions.py", "src/phase8a_build.py",
              "src/phase8a_global.py", "src/phase8a_patches.py",
              "src/phase8a_images.py", "src/phase8a_embed.py",
              "src/phase8a_batch.py", "src/phase8a_qa.py",
              "src/phase8a_report.py")

AUDIT_OUTPUTS = tuple("reports/phase%s_tests.json" % p for p in
                      ("1", "2", "3", "3b", "4", "5", "5c", "6a", "6b", "6c", "7a"))
STATUS_DOC = "docs/PROJECT_STATUS.md"


def check(n, name, ok, detail):
    CHECKS.append({"n": n, "check": name, "pass": bool(ok), "detail": detail})
    print("%-4s %2d  %s" % ("PASS" if ok else "FAIL", n, name))
    if not ok:
        print("        %s" % (detail if isinstance(detail, str) else json.dumps(detail, default=str)[:900]))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def token_present(src, token):
    """Whole-identifier match, never substring - the Phase-6B fix, kept."""
    return re.search(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(token), src) is not None


def executable_source(path):
    """Source with docstrings stripped, so prose cannot trip a scan."""
    src = open(path, "r", encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


def code_source(path):
    """Source with docstrings AND every string literal removed.

    Scope scans look for CONSTRUCTS, not for prose or configuration keys.  A
    config key such as `cfg["batch_descriptors"]["combat"]` - which exists
    precisely to switch harmonisation OFF - must not be mistaken for a ComBat
    implementation.  This is the same self-hit family defeated in Phases 5C,
    6A, 6B and 7A, defeated here by construction rather than by an exception
    list.
    """
    tree = ast.parse(open(path, "r", encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.unparse(ast.fix_missing_locations(tree))


def phase8a_sources():
    return {p: code_source(os.path.join(ROOT, p))
            for p in PHASE8A_PY if os.path.isfile(os.path.join(ROOT, p))}


COHORT = pd.read_csv(ppath(CFG, "cohort_csv"))
INC = COHORT[COHORT["included"].astype(str).str.lower() == "true"].reset_index(drop=True)
PIDS = sorted(INC["PatientID"].astype(str))
RUN = json.load(open(ppath(CFG, "run_json"), encoding="utf-8")) \
    if os.path.isfile(ppath(CFG, "run_json")) else {}


def record(pid):
    with open(os.path.join(ppath(CFG, "masks_dir"), pid, "record.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


def sample_pids(n=6):
    return PIDS[:: max(1, len(PIDS) // n)][:n]


# =========================================================================
# A  cohort, eligibility and provenance
# =========================================================================

def a1_every_subject_accounted_for():
    patients = json.load(open(ppath(CFG, "tcia_patients_json"), encoding="utf-8"))
    coll = sorted({p["PatientId"] for p in patients})
    rows = sorted(COHORT["PatientID"].astype(str))
    ex = pd.read_csv(ppath(CFG, "exclusions_csv"))
    excluded = sorted(ex["PatientID"].astype(str))
    ok = (rows == coll
          and not COHORT["PatientID"].duplicated().any()
          and excluded == sorted(COHORT[~COHORT["included"].astype(str)
                                        .str.lower().eq("true")]["PatientID"].astype(str))
          and bool((COHORT["included"].astype(str).str.lower().eq("true")
                    | (COHORT["exclusion_reason"].astype(str).str.len() > 0)).all()))
    check(1, "every downloaded subject appears exactly once in cohort.csv with an "
             "explicit inclusion or exclusion reason", ok,
          {"subjects_in_collection": len(coll), "cohort_rows": len(COHORT),
           "included": int(len(INC)), "exclusions_csv_rows": len(ex)})


def a2_eligibility_rederived():
    """Re-derive the included set independently from the raw metadata."""
    clin = pd.read_csv(ppath(CFG, "clinical_csv"))
    hcol = CFG["eligibility"]["histology_column"]
    hmap = {str(k).strip().lower(): int(v)
            for k, v in CFG["eligibility"]["histology_map"].items()}
    seg = json.load(open(ppath(CFG, "seg_index_json"), encoding="utf-8"))
    with_seg = {}
    for r in seg["series"]:
        with_seg.setdefault(r["PatientID"], []).append(r)
    want = set()
    for _, row in clin.iterrows():
        pid = str(row[CFG["eligibility"]["case_id_column"]]).strip()
        h = "" if pd.isna(row[hcol]) else str(row[hcol]).strip().lower()
        if h in hmap and len(with_seg.get(pid, [])) == 1:
            want.add(pid)
    got = set(PIDS)
    lost = sorted(want - got)
    extra = sorted(got - want)
    # every lost subject must carry a recorded, image-level exclusion reason
    reasons = {p: str(COHORT.loc[COHORT.PatientID == p, "exclusion_reason"].iloc[0])
               for p in lost}
    ok = (not extra) and all(r for r in reasons.values())
    check(2, "the included set re-derives from the clinical CSV and the SEG index; "
             "every difference has a recorded image-level reason", ok,
          {"metadata_candidates": len(want), "included": len(got),
           "dropped_at_image_level": reasons, "unexpected_inclusions": extra})


def a3_reason_vocabulary():
    from phase8a_build import RULE_ORDER
    allowed = set(RULE_ORDER)
    inc_mask = COHORT["included"].astype(str).str.lower().eq("true")
    exc = COHORT[~inc_mask]["exclusion_reason"].fillna("").astype(str)
    inc = COHORT[inc_mask]["exclusion_reason"].fillna("").astype(str)
    unknown = sorted(set(exc) - allowed)
    # an excluded subject must carry exactly one frozen rule code; an included
    # subject must carry none at all
    ok = (not unknown) and "not_built" not in set(exc) and (inc == "").all()
    counts = exc.value_counts().to_dict()
    check(3, "every exclusion reason comes from the frozen deterministic rule list", ok,
          {"reasons": counts, "unknown": unknown,
           "included_rows_with_a_reason": int((inc != "").sum())})


def a4_label_mapping():
    hm = {str(k).strip().lower(): int(v)
          for k, v in CFG["eligibility"]["histology_map"].items()}
    bad = [p for p in PIDS
           if int(INC.loc[INC.PatientID == p, "label"].iloc[0])
           != hm[str(INC.loc[INC.PatientID == p, "Histology"].iloc[0]).strip().lower()]]
    adc = int((INC["label"] == 1).sum())
    scc = int((INC["label"] == 0).sum())
    ok = (not bad and hm.get("adenocarcinoma") == 1
          and hm.get("squamous cell carcinoma") == 0 and adc + scc == len(INC))
    check(4, "ADC = 1 / SCC = 0, the same coding as LUNG1, correct for every patient", ok,
          {"n_adc": adc, "n_scc": scc, "mismatches": bad[:5]})


def a5_provenance_complete():
    need = ("PatientID", "Histology", "label", "seg_series_uid", "ct_series_uid",
            "ct_study_uid", "seg_zip_sha256", "ct_sha256", "gtv_sha256",
            "rim_sha256", "gtv_rim_sha256")
    missing_cols = [c for c in need if c not in COHORT.columns]
    blank = [c for c in need if c not in missing_cols
             and INC[c].astype(str).str.strip().eq("").any()]
    hashes_ok, bad = True, []
    for pid in sample_pids(8):
        r = record(pid)
        for k, key in (("ct", "ct_sha256"), ("gtv", "gtv_sha256"),
                       ("rim", "rim_sha256"), ("gtv_rim", "gtv_rim_sha256")):
            path = os.path.join(ROOT, *r["files"][k]["path"].split("/"))
            if sha256(path) != str(INC.loc[INC.PatientID == pid, key].iloc[0]):
                hashes_ok = False
                bad.append("%s/%s" % (pid, k))
    ok = not missing_cols and not blank and hashes_ok
    check(5, "provenance is complete: TCIA subject, CT and SEG series UIDs, histology, "
             "mapped label, eligibility status and mask hashes", ok,
          {"missing_columns": missing_cols, "blank_columns": blank,
           "rehashed_patients": len(sample_pids(8)), "hash_mismatches": bad})


def a6_no_duplication():
    dirs = [d for d in os.listdir(ppath(CFG, "masks_dir"))
            if os.path.isdir(os.path.join(ppath(CFG, "masks_dir"), d))]
    sets = {"masks": sorted(dirs)}
    for tag, key in (("patch_features", "patch_features_dir"),
                     ("image_patches", "image_patches_dir"),
                     ("cnn_embeddings", "cnn_embeddings_dir")):
        d = ppath(CFG, key)
        sets[tag] = sorted(f[:-4] for f in os.listdir(d)
                           if f.endswith(".npz")) if os.path.isdir(d) else []
    g = pd.read_csv(ppath(CFG, "gtv_csv"))
    r = pd.read_csv(ppath(CFG, "rim_csv"))
    ok = (sets["masks"] == PIDS and sets["patch_features"] == PIDS
          and sets["image_patches"] == PIDS and sets["cnn_embeddings"] == PIDS
          and sorted(g["PatientID"].astype(str)) == PIDS
          and sorted(r["PatientID"].astype(str)) == PIDS
          and not g["PatientID"].duplicated().any()
          and not r["PatientID"].duplicated().any())
    check(6, "no patient duplication: every product set is exactly the included cohort, "
             "once each", ok, {k: len(v) for k, v in sets.items()} |
          {"gtv_rows": len(g), "rim_rows": len(r), "included": len(PIDS)})


def a7_source_metadata_hashed():
    rec = RUN.get("source_metadata", {})
    ok = (sha256(ppath(CFG, "clinical_csv")) == rec.get("clinical_csv_sha256")
          and sha256(ppath(CFG, "tcia_series_json")) == rec.get("tcia_series_sha256")
          and sha256(ppath(CFG, "tcia_patients_json")) == rec.get("tcia_patients_sha256")
          and rec.get("collection") == CFG["source"]["collection"])
    check(7, "the public source metadata is recorded and hashed (clinical CSV, TCIA "
             "patient and series listings)", ok,
          {"clinical_csv_sha256": sha256(ppath(CFG, "clinical_csv"))[:24],
           "recorded": str(rec.get("clinical_csv_sha256"))[:24],
           "collection": rec.get("collection")})


# =========================================================================
# B  regions
# =========================================================================

def b8_set_algebra_from_disk():
    bad, n = [], 0
    for pid in PIDS:
        d = os.path.join(ppath(CFG, "masks_dir"), pid)
        # GetArrayFromImage, not the view: a view of a temporary sitk.Image
        # dangles as soon as the call returns
        g = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(d, CFG["paths"]["gtv_filename"]))) > 0
        r = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(d, CFG["paths"]["rim_filename"]))) > 0
        e = sitk.GetArrayFromImage(sitk.ReadImage(
            os.path.join(d, CFG["paths"]["gtv_rim_filename"]))) > 0
        n += 1
        if int((g & r).sum()) or int(((g | r) ^ e).sum()) or int((g & ~e).sum()):
            bad.append(pid)
    check(8, "GTV n Rim = empty, GTV u Rim = GTV+Rim and GTV subset GTV+Rim, "
             "re-derived voxelwise from the mask files of every patient", not bad,
          {"patients_verified": n, "violations": bad[:10]})


def b9_physical_expansion():
    exp = float(CFG["regions"]["expansion_mm"])
    worst, detail = 0.0, {}
    for pid in sample_pids(6):
        d = os.path.join(ppath(CFG, "masks_dir"), pid)
        gtv = sitk.ReadImage(os.path.join(d, CFG["paths"]["gtv_filename"]))
        rim = sitk.ReadImage(os.path.join(d, CFG["paths"]["rim_filename"]))
        g = sitk.GetArrayViewFromImage(gtv) > 0
        r = sitk.GetArrayViewFromImage(rim) > 0
        samp = zyx_spacing(gtv)
        phi = signed_distance_field(g, samp, method=CFG["regions"]["distance_method"],
                                    crop_margin_mm=exp + 4.0 * max(samp),
                                    fine_mm=float(CFG["regions"]["fine_grid_mm"]))
        mx = float(phi[r].max())
        rebuilt = ((phi <= exp) | g) & ~g
        detail[pid] = {"max_distance_mm": round(mx, 4),
                       "voxels_differing_from_stored_rim": int((rebuilt ^ r).sum())}
        worst = max(worst, abs(mx - exp))
        if int((rebuilt ^ r).sum()):
            worst = 1e9
    check(9, "the rim is exactly the %g mm physical expansion outside the GTV, "
             "re-derived from the stored GTV with the validated signed-distance "
             "method" % exp, worst <= 0.05, detail)


def b10_masks_match_ct_geometry():
    bad = []
    for pid in PIDS:
        d = os.path.join(ppath(CFG, "masks_dir"), pid)
        ct = sitk.ReadImage(os.path.join(d, CFG["paths"]["ct_filename"]))
        for f in (CFG["paths"]["gtv_filename"], CFG["paths"]["rim_filename"],
                  CFG["paths"]["gtv_rim_filename"]):
            m = sitk.ReadImage(os.path.join(d, f))
            if (tuple(m.GetSize()) != tuple(ct.GetSize())
                    or not np.allclose(m.GetSpacing(), ct.GetSpacing())
                    or not np.allclose(m.GetOrigin(), ct.GetOrigin())
                    or not np.allclose(m.GetDirection(), ct.GetDirection())):
                bad.append("%s/%s" % (pid, f))
    check(10, "every mask shares the size, spacing, origin and direction of its own CT",
          not bad, {"patients": len(PIDS), "mismatches": bad[:10]})


def b11_regions_non_empty_and_plausible():
    geo = pd.read_csv(ppath(CFG, "region_geometry_csv"))
    ok = (len(geo) == len(PIDS)
          and bool((geo["native_gtv_voxels"] > 0).all())
          and bool((geo["native_rim_voxels"] > 0).all())
          and bool((geo["native_gtv_rim_voxels"] > 0).all())
          and bool((geo["native_rim_fraction_of_gtv_rim"] > 0).all())
          and bool((geo["native_rim_fraction_of_gtv_rim"] < 1).all()))
    check(11, "every GTV, rim and GTV+Rim region is non-empty for every patient", ok,
          {"patients": len(geo),
           "gtv_volume_mm3": [round(float(geo["native_gtv_volume_mm3"].min()), 1),
                              round(float(geo["native_gtv_volume_mm3"].median()), 1),
                              round(float(geo["native_gtv_volume_mm3"].max()), 1)],
           "rim_fraction": [round(float(geo["native_rim_fraction_of_gtv_rim"].min()), 3),
                            round(float(geo["native_rim_fraction_of_gtv_rim"].median()), 3),
                            round(float(geo["native_rim_fraction_of_gtv_rim"].max()), 3)]})


def b12_seg_decoding_lost_nothing():
    bad, by_pos, by_uid = [], 0, 0
    for pid in PIDS:
        si = record(pid)["seg_info"]
        by_pos += int(si["frames_mapped_by_position"])
        by_uid += int(si["frames_mapped_by_sop_uid"])
        lost = int(si["unmapped_nonempty_frames"]) + int(si.get("unassigned_nonempty_frames", 0))
        if lost or si["seg_rows"] != si["ct_rows"] or si["seg_columns"] != si["ct_columns"]:
            bad.append(pid)
    check(12, "every SEG object decodes onto its CT grid with 0 non-empty frames lost "
              "and identical in-plane size", not bad,
          {"patients": len(PIDS), "frames_mapped_by_sop_uid": by_uid,
           "frames_mapped_by_image_position": by_pos, "failures": bad[:10]})


# =========================================================================
# C  preprocessing compatibility
# =========================================================================

def c13_two_mm_grid():
    spac, origins_ok = set(), True
    for pid in sample_pids(8):
        ct = sitk.ReadImage(os.path.join(ppath(CFG, "masks_dir"), pid,
                                         CFG["paths"]["ct_filename"]))
        grid = reference_grid(ct, CFG["preprocessing"]["resample_spacing_mm"])
        spac.add(tuple(round(float(v), 9) for v in grid["spacing"]))
        origins_ok &= tuple(grid["origin"]) == tuple(ct.GetOrigin())
    ok = (spac == {(2.0, 2.0, 2.0)} and origins_ok
          and CFG["preprocessing"]["ct_interpolator"] == "linear"
          and CFG["preprocessing"]["mask_interpolator"] == "nearest"
          and float(CFG["preprocessing"]["ct_default_value"]) == -1024.0)
    check(13, "2 mm isotropic grid, CT linear / mask nearest, identical to the frozen "
              "LUNG1 preprocessing", ok,
          {"spacings": sorted(spac), "origin_is_the_ct_origin": origins_ok,
           "interpolators": [CFG["preprocessing"]["ct_interpolator"],
                             CFG["preprocessing"]["mask_interpolator"]]})


def c14_global_signature_is_phase7a():
    cfg7, sig = derived_phase7a_cfg(CFG)
    p7 = json.load(open(os.path.join(ROOT, "reports", "phase7a_run.json"), encoding="utf-8"))
    lung1_sig = p7.get("extraction_signature") or p7.get("signature")
    ok = (sig == CFG["global_radiomics"]["expected_phase7a_signature"]
          and (lung1_sig is None or sig == lung1_sig)
          and cfg7["paths"]["pyradiomics_params"] == "config/phase7a_pyradiomics.yaml")
    check(14, "the external global-radiomics extraction signature equals the frozen "
              "Phase-7A signature, from the same parameter file", ok,
          {"external_signature": sig, "phase7a_run_record": lung1_sig,
           "expected": CFG["global_radiomics"]["expected_phase7a_signature"]})


def c15_global_feature_names_identical_to_lung1():
    ext = json.load(open(ppath(CFG, "global_feature_names_json"), encoding="utf-8"))
    l1 = json.load(open(os.path.join(ROOT, "global_features", "phase7a",
                                     "feature_names.json"), encoding="utf-8"))
    ok = (ext["gtv"] == l1["gtv"] and ext["rim"] == l1["rim"]
          and ext["family_counts"] == l1["family_counts"])
    check(15, "the external global feature bank is name-for-name and order-for-order "
              "identical to the frozen LUNG1 Phase-7A bank", ok,
          {"n_gtv": len(ext["gtv"]), "n_rim": len(ext["rim"]),
           "lung1_n_gtv": len(l1["gtv"]), "lung1_n_rim": len(l1["rim"]),
           "family_counts_match": ext["family_counts"] == l1["family_counts"]})


def c16_patch_signature_is_lung1():
    _, sig = derived_gtv_rim_cfg(CFG)
    names = [l.strip() for l in open(os.path.join(ROOT, "config", "feature_names.txt"),
                                     encoding="utf-8") if l.strip()]
    bad = []
    for pid in sample_pids(6):
        with np.load(os.path.join(ppath(CFG, "patch_features_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            if list(np.asarray(z["feature_names"], dtype=object)) != names:
                bad.append(pid)
            if str(z["config_signature"]) != sig:
                bad.append(pid + " (signature)")
    lung1 = None
    with np.load(os.path.join(ROOT, "patch_features", "gtv_rim",
                              "LUNG1-001.npz"), allow_pickle=True) as z:
        lung1 = str(z["config_signature"])
    ok = (sig == CFG["patch_sampling"]["expected_extraction_signature"] == lung1
          and not bad and len(names) == 74)
    check(16, "the external patch-sampling signature equals the signature stored in the "
              "frozen LUNG1 GTV+Rim bags, and the 74 feature names match exactly", ok,
          {"external_signature": sig, "lung1_bag_signature": lung1,
           "n_features": len(names), "mismatches": bad})


def c17_crop_geometry():
    _, sig, tsig = derived_phase6a_cfg(CFG)
    lo, hi = (int(v) for v in CFG["crop"]["hu_clip"])
    bad, checked = [], 0
    for pid in sample_pids(6):
        with np.load(os.path.join(ppath(CFG, "image_patches_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            im = np.asarray(z["images"])
            checked += int(im.shape[0])
            if im.dtype != np.int16 or im.shape[1:] != (3, 32, 32):
                bad.append("%s shape %s %s" % (pid, im.shape, im.dtype))
            c = im[:, :, 16, 16]
            if not (np.array_equal(c[:, 0], c[:, 1]) and np.array_equal(c[:, 1], c[:, 2])):
                bad.append("%s planes do not intersect at the centre" % pid)
            if int(im.min()) < lo or int(im.max()) > hi:
                bad.append("%s HU outside [%d, %d]" % (pid, lo, hi))
            if str(z["crop_signature"]) != sig:
                bad.append("%s crop signature" % pid)
            if int(z["outer_fold"]) != -1:
                bad.append("%s outer_fold %s" % (pid, int(z["outer_fold"])))
    ok = (sig == CFG["crop"]["expected_phase6a_crop_signature"]
          and tsig == CFG["transform"]["expected_phase6a_transform_signature"]
          and not bad)
    check(17, "the external 2.5D crops carry the frozen Phase-6A crop and transform "
              "signatures, [3, 32, 32] int16, HU clipped to [-1024, 200], the three "
              "planes intersecting at the patch centre, and outer_fold = -1", ok,
          {"crop_signature": sig, "transform_signature": tsig,
           "instances_checked": checked, "problems": bad[:6]})


def c18_encoder_frozen():
    meta_path = os.path.join(ppath(CFG, "metadata_dir"), "cnn_embedding_run.json")
    meta = json.load(open(meta_path, encoding="utf-8"))
    enc = meta["encoder"]
    bad = []
    for pid in sample_pids(6):
        with np.load(os.path.join(ppath(CFG, "cnn_embeddings_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            if (str(z["encoder_state_fingerprint"]) != CFG["encoder"]["expected_state_fingerprint"]
                    or int(z["embedding_dim"]) != 512
                    or not bool(z["encoder_frozen"])
                    or str(z["encoder_head"]) != "fc -> Identity"):
                bad.append(pid)
    ok = (enc["state_fingerprint"] == CFG["encoder"]["expected_state_fingerprint"]
          and enc["weights_enum"] == CFG["encoder"]["weights_enum"]
          and int(enc["n_parameters_requiring_grad"]) == 0
          and enc["training_mode"] is False and int(enc["embedding_dim"]) == 512
          and not bad)
    check(18, "the embeddings come from the SAME frozen pretrained ResNet18 as Phase 6A: "
              "identical state fingerprint, fc -> Identity, 0 trainable parameters, "
              "eval mode, 512-D", ok,
          {"state_fingerprint": enc["state_fingerprint"][:24],
           "expected": CFG["encoder"]["expected_state_fingerprint"][:24],
           "weights": enc["weights_enum"], "trainable": enc["n_parameters_requiring_grad"],
           "training_mode": enc["training_mode"], "file_mismatches": bad})


# =========================================================================
# D  the patch sampler is the frozen LUNG1 sampler
# =========================================================================

def d19_sampler_is_the_frozen_module():
    """The driver must import the frozen sampler, and that sampler must be
    byte-identical to its pre-Phase-8A state."""
    snap = json.load(open(ppath(CFG, "pre_snapshot"), encoding="utf-8"))["files"]
    frozen = ("src/extract_patient.py", "src/patches.py", "src/resample.py",
              "src/config_io.py", "config/radiomics_params.yaml",
              "config/gtv_rim_experiment.yaml", "src/image_patches.py",
              "src/cnn_embed.py", "config/phase6a.yaml", "config/phase7a.yaml",
              "config/phase7a_pyradiomics.yaml", "src/phase7a_extract.py",
              "src/phase7a_regions.py")
    changed = [p for p in frozen
               if p in snap and sha256(os.path.join(ROOT, p)) != snap[p]]
    src = executable_source(os.path.join(ROOT, "src", "phase8a_patches.py"))
    imports_frozen = (token_present(src, "extract_patient")
                      and token_present(src, "extraction_signature")
                      and not token_present(src, "enumerate_patches"))
    img = executable_source(os.path.join(ROOT, "src", "phase8a_images.py"))
    imports_crop = token_present(img, "extract_patient_images") and token_present(img, "IP")
    ok = not changed and imports_frozen and imports_crop
    check(19, "the external sampler IS the frozen LUNG1 code: every frozen sampler / "
              "crop / extraction module is byte-identical and the Phase-8A drivers "
              "call them rather than re-implementing them", ok,
          {"frozen_modules_verified": len(frozen), "changed": changed,
           "phase8a_patches_calls_extract_patient": imports_frozen,
           "phase8a_images_calls_extract_patient_images": imports_crop})


def d20_lattice_anchor():
    bad, n_inst, n_bags = [], 0, 0
    for pid in PIDS:
        with np.load(os.path.join(ppath(CFG, "patch_features_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            st = np.asarray(z["coords_start_index"], dtype=int)
            ci = np.asarray(z["coords_index"], dtype=int)
            size = np.asarray(z["patch_size"], dtype=int)
            stride = np.asarray(z["patch_stride"], dtype=int)
            n_bags += 1
            n_inst += int(st.shape[0])
            if not np.array_equal(size, [5, 5, 5]) or not np.array_equal(stride, [5, 5, 5]):
                bad.append("%s geometry" % pid)
            if st.size and (st % stride).any():
                bad.append("%s anchor" % pid)
            if st.size and not np.array_equal(ci, st + size // 2):
                bad.append("%s centre" % pid)
    check(20, "the external lattice is anchored at resampled voxel (0, 0, 0) with "
              "5x5x5 patches and stride 5: every start index is a multiple of 5 and "
              "every centre is start + 2", not bad,
          {"bags": n_bags, "instances": n_inst, "violations": bad[:10]})


def d21_retention_rule_rederived():
    """Re-run the frozen retention arithmetic from the masks on disk."""
    cfg4, _ = derived_gtv_rim_cfg(CFG)
    pre = cfg4["preprocessing"]
    pc = cfg4["patches"]
    detail, ok = {}, True
    for pid in sample_pids(4):
        d = os.path.join(ppath(CFG, "masks_dir"), pid)
        ct = sitk.ReadImage(os.path.join(d, CFG["paths"]["ct_filename"]))
        mk = sitk.ReadImage(os.path.join(d, CFG["paths"]["gtv_rim_filename"]))
        grid = reference_grid(ct, pre["resample_spacing_mm"])
        mr = resample_to_grid(sitk.Cast(mk, sitk.sitkUInt8), grid,
                              pre["mask_interpolator"], 0)
        arr = sitk.GetArrayFromImage(sitk.Cast(mr > 0, sitk.sitkUInt8))
        cands, _ = enumerate_patches(arr, pc["size_voxels"], pc["stride_voxels"],
                                     bool(pc["require_full_in_bounds"]))
        keep = []
        for c in cands:
            i, j, k = (int(v) for v in c["center"])
            if not (0 <= k < arr.shape[0] and 0 <= j < arr.shape[1] and 0 <= i < arr.shape[2]):
                continue
            if arr[k, j, i] == 0:
                continue
            n_roi, _ = patch_roi_stats(arr, c["start"], pc["size_voxels"])
            if n_roi < int(pc["min_roi_voxels"]):
                continue
            keep.append([int(v) for v in c["start"]])
        with np.load(os.path.join(ppath(CFG, "patch_features_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            stored = [list(map(int, s)) for s in np.asarray(z["coords_start_index"])]
            roi_v = np.asarray(z["roi_voxels"], dtype=int)
        extra = [s for s in stored if s not in keep]
        dropped = len(keep) - len(stored)
        good = (not extra) and dropped >= 0 and int(roi_v.min()) >= int(pc["min_roi_voxels"])
        ok &= good
        detail[pid] = {"geometric_candidates": len(cands),
                       "pass_centre_and_27_voxel_rule": len(keep),
                       "stored_instances": len(stored),
                       "dropped_by_pyradiomics_validity": dropped,
                       "min_roi_voxels_stored": int(roi_v.min()) if roi_v.size else None,
                       "stored_but_not_re_derived": extra[:3]}
    check(21, "the retention rule re-derives from the masks on disk: patch centre inside "
              "the ROI, at least 27 ROI voxels in the block, and nothing stored that the "
              "rule rejects", ok, detail)


def d22_non_overlapping():
    bad = []
    for pid in sample_pids(8):
        with np.load(os.path.join(ppath(CFG, "patch_features_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            st = np.asarray(z["coords_start_index"], dtype=int)
        if st.size and len({tuple(s) for s in st}) != st.shape[0]:
            bad.append(pid)
    check(22, "patches are non-overlapping: a start index occurs at most once per bag "
              "and the stride equals the patch size", not bad,
          {"patients_checked": len(sample_pids(8)), "duplicates": bad})


# =========================================================================
# E  scope
# =========================================================================

FORBIDDEN_METRIC_TOKENS = (
    "roc_auc_score", "average_precision_score", "accuracy_score",
    "balanced_accuracy_score", "confusion_matrix", "precision_recall_curve",
    "roc_curve", "matthews_corrcoef", "f1_score", "brier_score_loss",
    "log_loss", "recall_score", "precision_score",
)
FORBIDDEN_MODEL_TOKENS = (
    "LogisticRegression", "StandardScaler", "SVC", "RandomForestClassifier",
    "GradientBoostingClassifier", "CalibratedClassifierCV", "StratifiedKFold",
    "train_test_split", "GridSearchCV", "cross_val_score",
    "MeanMIL", "GatedAttentionMIL", "ClamStyleGatedAttentionMIL",
    "backward", "optimizer", "Adam", "BCEWithLogitsLoss", "pos_weight",
    "mrmr", "combat", "neuroCombat", "SMOTE",
)


def e23_no_classification_metric():
    hits = []
    for p, src in phase8a_sources().items():
        for t in FORBIDDEN_METRIC_TOKENS:
            if token_present(src, t):
                hits.append("%s: %s" % (p, t))
    res = json.load(open(os.path.join(ppath(CFG, "qa_dir"), "qa_checks.json"),
                         encoding="utf-8"))
    metric_files = [f for f in os.listdir(ppath(CFG, "external_root"))
                    if f.lower().endswith(".json") and "metric" in f.lower()]
    check(23, "Phase 8A computes NO classification metric: no AUC, PR-AUC, accuracy, "
              "confusion matrix, ROC/PR curve or Brier construct exists in any "
              "Phase-8A module, and no metric artefact was written",
          not hits and not metric_files,
          {"modules_scanned": len(phase8a_sources()), "hits": hits,
           "qa_checks": "%d/%d" % (res["n_pass"], res["n_checks"]),
           "metric_files": metric_files})


def e24_no_model_trained():
    hits = []
    for p, src in phase8a_sources().items():
        for t in FORBIDDEN_MODEL_TOKENS:
            if token_present(src, t):
                hits.append("%s: %s" % (p, t))
    # no checkpoint, prediction or model artefact anywhere under the namespace
    stray = []
    for root, _, fs in os.walk(ppath(CFG, "external_root")):
        for f in fs:
            if f.endswith((".pt", ".pth", ".pkl", ".joblib")) or "prediction" in f.lower():
                stray.append(os.path.join(root, f))
    check(24, "no model is trained, fitted, selected or stored on the external cohort: "
              "no estimator, optimiser, loss or MIL construct in any Phase-8A module and "
              "no checkpoint or prediction artefact in the namespace",
          not hits and not stray, {"hits": hits, "stray_artefacts": stray[:10]})


def e25_no_harmonisation():
    hits = []
    for p, src in phase8a_sources().items():
        for t in ("ComBat", "combat", "harmonize", "harmonise", "site_classifier"):
            if token_present(src, t):
                hits.append("%s: %s" % (p, t))
    cfg_says_no = CFG["batch_descriptors"]["combat"] is False
    summ = json.load(open(ppath(CFG, "acquisition_summary_json"), encoding="utf-8"))
    ok = not hits and cfg_says_no and "DESCRIPTIVE ONLY" in summ["note"]
    check(25, "batch descriptors are descriptive only: ComBat is not fitted, no site "
              "classifier exists and the two cohorts are never combined", ok,
          {"config_combat": CFG["batch_descriptors"]["combat"], "hits": hits,
           "external_n": summ["external"]["n_patients"],
           "lung1_n": summ["lung1_gtv_rim_adc_scc"]["n_patients"]})


def e26_no_feature_selection():
    hits = []
    for p, src in phase8a_sources().items():
        for t in ("SelectKBest", "RFE", "Lasso", "ElasticNet", "f_classif",
                  "mutual_info_classif", "select_features"):
            if token_present(src, t):
                hits.append("%s: %s" % (p, t))
    g = pd.read_csv(ppath(CFG, "gtv_csv"), nrows=1)
    r = pd.read_csv(ppath(CFG, "rim_csv"), nrows=1)
    l1 = json.load(open(os.path.join(ROOT, "global_features", "phase7a",
                                     "feature_names.json"), encoding="utf-8"))
    full = (len([c for c in g.columns if c not in ("PatientID", "Histology", "label")])
            == len(l1["gtv"])
            and len([c for c in r.columns if c not in ("PatientID", "Histology", "label")])
            == len(l1["rim"]))
    check(26, "no feature selection is performed on the external patients: the FULL "
              "label-independent bank is extracted and stored", not hits and full,
          {"hits": hits, "gtv_columns": len(g.columns) - 3,
           "rim_columns": len(r.columns) - 3, "full_bank": full})


def e27_label_used_only_for_eligibility_and_record():
    """The label may reach the cohort file and the per-patient record.  It must
    not reach any geometry, feature, crop or embedding decision."""
    geometry_modules = ("src/phase8a_dicom.py", "src/phase8a_regions.py")
    hits = []
    for p in geometry_modules:
        src = executable_source(os.path.join(ROOT, p))
        # a bare "y" is not scanned for: in a geometry module it is the second
        # element of an (x, y, z) coordinate, never an outcome variable
        for t in ("histology", "Histology", "adenocarcinoma", "Adenocarcinoma",
                  "squamous", "ADC", "SCC", "y_true", "y_label", "outcome"):
            if token_present(src, t):
                hits.append("%s: %s" % (p, t))
    # the extraction drivers may carry the label as a stored attribute, but may
    # never branch or condition on it
    driver_calls = {}
    for p in ("src/phase8a_global.py", "src/phase8a_patches.py",
              "src/phase8a_images.py", "src/phase8a_embed.py"):
        src = code_source(os.path.join(ROOT, p))
        driver_calls[p] = {"branches_on_label": bool(
            re.search(r"if[^\n]*(?<![A-Za-z0-9_])(label|Histology|histology)"
                      r"(?![A-Za-z0-9_])[^\n]*(==|!=|>|<)", src))}
    branch = [p for p, v in driver_calls.items() if v["branches_on_label"]]
    check(27, "the external histology is used only to decide ADC/SCC eligibility and to "
              "record the ground truth: no geometry module reads it and no extraction "
              "driver branches on it", not hits and not branch,
          {"geometry_modules_scanned": list(geometry_modules), "hits": hits,
           "drivers_branching_on_label": branch})


def e28_no_lung1_split_touched():
    hits = []
    for p, src in phase8a_sources().items():
        for t in ("gtv_rim_stratified_5fold", "load_outer_folds", "splits_file"):
            if token_present(src, t):
                hits.append("%s: %s" % (p, t))
    folds = set()
    for pid in PIDS:
        with np.load(os.path.join(ppath(CFG, "image_patches_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            folds.add(int(z["outer_fold"]))
    check(28, "no LUNG1 split is read or written, and every external instance carries "
              "outer_fold = -1 so it can never join a LUNG1 fold",
          not hits and folds == {-1}, {"hits": hits, "outer_folds_seen": sorted(folds)})


# =========================================================================
# F  integrity and instance identity
# =========================================================================

def f29_no_nan_or_inf():
    g = pd.read_csv(ppath(CFG, "gtv_csv"))
    r = pd.read_csv(ppath(CFG, "rim_csv"))
    meta = ["PatientID", "Histology", "label"]
    gv = g[[c for c in g.columns if c not in meta]].to_numpy(dtype=float)
    rv = r[[c for c in r.columns if c not in meta]].to_numpy(dtype=float)
    counts = {"global_gtv_nan": int(np.isnan(gv).sum()),
              "global_gtv_inf": int(np.isinf(gv).sum()),
              "global_rim_nan": int(np.isnan(rv).sum()),
              "global_rim_inf": int(np.isinf(rv).sum()),
              "patch_nonfinite": 0, "embedding_nonfinite": 0, "crop_out_of_range": 0}
    lo, hi = (int(v) for v in CFG["crop"]["hu_clip"])
    for pid in PIDS:
        with np.load(os.path.join(ppath(CFG, "patch_features_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            f = np.asarray(z["features"], dtype=np.float64)
            counts["patch_nonfinite"] += int((~np.isfinite(f)).sum())
        with np.load(os.path.join(ppath(CFG, "cnn_embeddings_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            e = np.asarray(z["embeddings"])
            counts["embedding_nonfinite"] += int((~np.isfinite(e)).sum())
        with np.load(os.path.join(ppath(CFG, "image_patches_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            im = np.asarray(z["images"])
            if im.size:
                counts["crop_out_of_range"] += int((im < lo).sum() + (im > hi).sum())
    check(29, "no NaN or Inf anywhere: global features, local patch features, CNN "
              "embeddings, and every crop inside the HU window",
          all(v == 0 for v in counts.values()), counts)


def f30_instance_identity():
    bad, n = [], 0
    for pid in PIDS:
        with np.load(os.path.join(ppath(CFG, "patch_features_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            ci = np.asarray(z["coords_index"], dtype=np.int32)
            cw = np.asarray(z["coords_world"], dtype=np.float64)
            n_bag = int(np.asarray(z["features"]).shape[0])
        fp = hashlib.sha256()
        fp.update(np.ascontiguousarray(ci, dtype=np.int32).tobytes())
        fp.update(np.ascontiguousarray(cw, dtype=np.float64).tobytes())
        fp = fp.hexdigest()
        with np.load(os.path.join(ppath(CFG, "image_patches_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            n_img = int(np.asarray(z["images"]).shape[0])
            img_fp = str(z["source_coords_sha256"])
            img_ci = np.asarray(z["coords_index"], dtype=np.int32)
        with np.load(os.path.join(ppath(CFG, "cnn_embeddings_dir"), "%s.npz" % pid),
                     allow_pickle=True) as z:
            n_emb = int(np.asarray(z["embeddings"]).shape[0])
            emb_fp = str(z["source_coords_sha256"])
            idx = np.asarray(z["patch_index"], dtype=np.int32)
        n += n_bag
        if not (n_bag == n_img == n_emb and fp == img_fp == emb_fp
                and np.array_equal(ci, img_ci)
                and np.array_equal(idx, np.arange(n_bag, dtype=np.int32))):
            bad.append(pid)
    check(30, "bag row i = crop row i = embedding row i for every patient, proved by "
              "counts, coordinate arrays and a shared coordinate fingerprint", not bad,
          {"patients": len(PIDS), "instances": n, "mismatches": bad[:10]})


def f31_qa_ran():
    res = json.load(open(os.path.join(ppath(CFG, "qa_dir"), "qa_checks.json"),
                         encoding="utf-8"))
    figs = sorted(f for f in os.listdir(ppath(CFG, "qa_dir")) if f.endswith(".png"))
    ok = (res["n_pass"] == res["n_checks"]
          and len(figs) >= int(CFG["qa"]["visual_patients"]))
    check(31, "the automatic QA suite passes and representative visual QA images exist",
          ok, {"qa": "%d/%d" % (res["n_pass"], res["n_checks"]),
               "figures": figs,
               "failed": [c["check"] for c in res["checks"] if not c["pass"]]})


def f32_acquisition_descriptors():
    acq = pd.read_csv(ppath(CFG, "acquisition_csv"))
    summ = json.load(open(ppath(CFG, "acquisition_summary_json"), encoding="utf-8"))
    need = ("Manufacturer", "ManufacturerModelName", "SliceThickness",
            "PixelSpacing", "ConvolutionKernel", "KVP", "ContrastBolusAgent")
    ok = (len(acq) == len(PIDS) and all(c in acq.columns for c in need)
          and "headline_differences" in summ)
    check(32, "acquisition / domain-shift descriptors are collected for every external "
              "patient and summarised beside the LUNG1 distributions", ok,
          {"rows": len(acq), "missing_columns": [c for c in need if c not in acq.columns],
           "headline": summ.get("headline_differences")})


# =========================================================================
# G  freeze and read-only
# =========================================================================

def g33_phases_1_to_7a_unchanged():
    snap = json.load(open(ppath(CFG, "pre_snapshot"), encoding="utf-8"))["files"]
    changed, missing, semantic = [], [], {}
    for rel, want in snap.items():
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            missing.append(rel)
            continue
        if sha256(p) == want:
            continue
        if rel in AUDIT_OUTPUTS:
            d = json.load(open(p, encoding="utf-8"))
            # the suites are not uniform: some write n_pass, others n_passed
            n_pass = d.get("n_pass", d.get("n_passed"))
            n_checks, n_failed = d.get("n_checks"), d.get("n_failed", 0)
            semantic[rel] = {"n_pass": n_pass, "n_checks": n_checks,
                             "n_failed": n_failed}
            if n_pass != n_checks or n_failed:
                changed.append(rel)
            continue
        if rel == STATUS_DOC:
            txt = open(p, encoding="utf-8").read()
            semantic[rel] = {"records_phase_7a": "Phase 7A" in txt,
                             "records_phase_8a": "Phase 8A" in txt}
            if not (semantic[rel]["records_phase_7a"] and semantic[rel]["records_phase_8a"]):
                changed.append(rel)
            continue
        changed.append(rel)
    check(33, "every Phase-1 to Phase-7A artefact is byte-identical; the audit outputs "
              "and the required status document are verified semantically instead",
          not changed and not missing,
          {"files_in_snapshot": len(snap), "changed": changed[:20],
           "missing": missing[:20], "semantic": semantic})


def g34_external_trees_readonly():
    # re-run the project's own frozen guard rather than re-implementing it: it
    # compares every read-only file's size and mtime against the baseline
    import readonly_guard as RG
    from config_io import load_config
    ok, info = RG.verify(load_config(RG.DEFAULT_CONFIG))
    check(34, "Dataset/ and Vuong5ROI/ are untouched", ok,
          {"files": info["n_baseline"], "added": info["n_added"],
           "removed": info["n_removed"], "changed": info["n_changed"]})


def g35_namespace_clean():
    snap = json.load(open(ppath(CFG, "pre_snapshot"), encoding="utf-8"))["files"]
    allowed = ("external/nsclc_radiogenomics", "config/phase8a_external.yaml",
               "src/phase8a_", "tests/test_phase8a.py", "tools/phase8a_baseline.py",
               "reports/phase8a", "reports/PHASE8A", "logs/phase8a", "docs/PROJECT_STATUS.md")
    stray = []
    for d in ("predictions", "results", "models", "preprocessing", "splits", "qa",
              "global_features", "feature_selection", "attention", "patch_features",
              "image_patches", "cnn_embeddings", "cohort", "patient_summaries",
              "external", "config", "src", "tools", "tests", "docs", "reports"):
        root_d = os.path.join(ROOT, d)
        if not os.path.isdir(root_d):
            continue
        for root, _, fs in os.walk(root_d):
            for f in fs:
                rel = os.path.relpath(os.path.join(root, f), ROOT).replace(os.sep, "/")
                if rel in snap or "/__pycache__/" in rel or rel.endswith(".pyc"):
                    continue   # interpreter byte-code caches are not artefacts
                if not any(rel.startswith(a) for a in allowed):
                    stray.append(rel)
    # and no Phase-8B artefact of any kind
    b8 = [p for p in os.listdir(os.path.join(ROOT, "src")) if "phase8b" in p.lower()]
    b8 += [p for p in os.listdir(os.path.join(ROOT, "config")) if "phase8b" in p.lower()]
    check(35, "Phase 8A wrote nothing outside its own namespace and no Phase-8B "
              "artefact exists", not stray and not b8,
          {"files_outside_the_phase8a_namespace": stray[:20], "phase8b_files": b8})


def main():
    for fn in (a1_every_subject_accounted_for, a2_eligibility_rederived,
               a3_reason_vocabulary, a4_label_mapping, a5_provenance_complete,
               a6_no_duplication, a7_source_metadata_hashed,
               b8_set_algebra_from_disk, b9_physical_expansion,
               b10_masks_match_ct_geometry, b11_regions_non_empty_and_plausible,
               b12_seg_decoding_lost_nothing,
               c13_two_mm_grid, c14_global_signature_is_phase7a,
               c15_global_feature_names_identical_to_lung1,
               c16_patch_signature_is_lung1, c17_crop_geometry, c18_encoder_frozen,
               d19_sampler_is_the_frozen_module, d20_lattice_anchor,
               d21_retention_rule_rederived, d22_non_overlapping,
               e23_no_classification_metric, e24_no_model_trained,
               e25_no_harmonisation, e26_no_feature_selection,
               e27_label_used_only_for_eligibility_and_record,
               e28_no_lung1_split_touched,
               f29_no_nan_or_inf, f30_instance_identity, f31_qa_ran,
               f32_acquisition_descriptors,
               g33_phases_1_to_7a_unchanged, g34_external_trees_readonly,
               g35_namespace_clean):
        try:
            fn()
        except Exception as exc:                     # a raising check is a FAIL
            check(len(CHECKS) + 1, fn.__name__, False, "%s: %s" % (type(exc).__name__, exc))

    n_pass = sum(1 for c in CHECKS if c["pass"])
    out = {"phase": "8A", "n_checks": len(CHECKS), "n_pass": n_pass,
           "all_pass": n_pass == len(CHECKS), "checks": CHECKS}
    with open(ppath(CFG, "tests_json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("\n%d/%d Phase-8A checks pass -> %s" % (n_pass, len(CHECKS), ppath(CFG, "tests_json")))
    sys.exit(0 if n_pass == len(CHECKS) else 1)


if __name__ == "__main__":
    main()
