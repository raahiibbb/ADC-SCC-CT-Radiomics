"""Phase 8A - eligibility screening and per-subject product construction.

    python src/phase8a_build.py --config config/phase8a_external.yaml --screen
    python src/phase8a_build.py --config config/phase8a_external.yaml --build [--limit N]
    python src/phase8a_build.py --config config/phase8a_external.yaml --cohort

--screen  applies every eligibility rule that can be decided from metadata and
          the (already downloaded) SEG headers.  No image is downloaded and no
          image is looked at.
--build   downloads the referenced CT series of each screened candidate, reads
          it, decodes the SEG onto its grid, builds GTV / rim / GTV+Rim and
          writes CT.nii.gz + the three masks.  Resumable; the raw DICOM is
          deleted after conversion.
--cohort  assembles cohort.csv (every downloaded subject, with an explicit
          inclusion or exclusion reason) and exclusions.csv.

The histology label is used here for exactly one thing - deciding ADC/SCC
eligibility - and is carried through to the cohort file as the recorded ground
truth for a LATER, LOCKED external test.  It never touches a feature, a
preprocessing choice, a model or a threshold, and Phase 8A computes no
classification result of any kind.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase8a_tcia import (download_series_zip, load_cfg, ppath,       # noqa: E402
                          sha256_file)
from phase8a_dicom import (acquisition_tags, read_ct, read_seg,       # noqa: E402
                           referenced_series_of_zip_member, segment_summary)
from phase8a_regions import build_regions, same_geometry              # noqa: E402

# Exclusion reason codes, applied in this fixed order.  The FIRST failing rule
# is the recorded reason, so the outcome does not depend on evaluation order.
RULE_ORDER = [
    "no_clinical_record",
    "histology_missing",
    "histology_not_adc_or_scc",
    "no_segmentation",
    "multiple_segmentation_series",
    "ambiguous_tumour_segment",
    "segmentation_references_no_ct",
    "segmentation_references_multiple_ct",
    "referenced_ct_not_in_collection",
    "referenced_ct_patient_mismatch",
    "ct_download_failed",
    "ct_unreadable",
    "segmentation_misaligned",
    "segmentation_unmapped_frames",
    "segmentation_empty",
    "region_construction_failed",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _log(msg, path=None):
    print(msg, flush=True)
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("%s  %s\n" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg))


def load_metadata(cfg):
    with open(ppath(cfg, "tcia_patients_json"), "r", encoding="utf-8") as fh:
        patients = json.load(fh)
    with open(ppath(cfg, "tcia_series_json"), "r", encoding="utf-8") as fh:
        series = json.load(fh)
    with open(ppath(cfg, "seg_index_json"), "r", encoding="utf-8") as fh:
        seg_index = json.load(fh)
    clinical = pd.read_csv(ppath(cfg, "clinical_csv"))
    return patients, series, seg_index, clinical


def seg_dicom_path(cfg, series_uid):
    """Extract the SEG zip into the work directory and return its single file."""
    work = os.path.join(ppath(cfg, "work_dir"), "seg", series_uid)
    if not os.path.isdir(work) or not os.listdir(work):
        os.makedirs(work, exist_ok=True)
        zpath = os.path.join(ppath(cfg, "seg_zip_dir"), "%s.zip" % series_uid)
        with zipfile.ZipFile(zpath) as zf:
            zf.extractall(work)
    files = sorted(f for f in os.listdir(work) if f.lower().endswith(".dcm"))
    if len(files) != 1:
        raise RuntimeError("SEG series %s holds %d DICOM files" % (series_uid, len(files)))
    return os.path.join(work, files[0])


# ---------------------------------------------------------------------------
# stage 1: metadata-only screening
# ---------------------------------------------------------------------------

def screen(cfg):
    patients, series, seg_index, clinical = load_metadata(cfg)
    el = cfg["eligibility"]
    hist_col, case_col = el["histology_column"], el["case_id_column"]
    hmap = {str(k).strip().lower(): int(v) for k, v in el["histology_map"].items()}

    by_uid = {s["SeriesInstanceUID"]: s for s in series}
    segs_by_patient = {}
    for r in seg_index["series"]:
        segs_by_patient.setdefault(r["PatientID"], []).append(r)

    clin_by_case = {}
    for _, row in clinical.iterrows():
        clin_by_case.setdefault(str(row[case_col]).strip(), []).append(row)

    subjects = sorted({p["PatientId"] for p in patients})
    records = []
    for pid in subjects:
        rec = {"PatientID": pid, "included": False, "exclusion_reason": "",
               "histology": "", "label": None,
               "n_seg_series": len(segs_by_patient.get(pid, [])),
               "seg_series_uid": "", "seg_study_uid": "", "seg_zip_sha256": "",
               "seg_n_segments": None, "seg_segment_label": "",
               "seg_property_type": "", "seg_algorithm": "",
               "ct_series_uid": "", "ct_study_uid": "",
               "ct_expected_slices": None, "ct_filesize_bytes": None,
               "n_clinical_rows": len(clin_by_case.get(pid, []))}

        rows = clin_by_case.get(pid, [])
        if len(rows) != 1:
            rec["exclusion_reason"] = "no_clinical_record"
            records.append(rec)
            continue
        hist = rows[0][hist_col]
        hist = "" if pd.isna(hist) else str(hist).strip()
        rec["histology"] = hist
        if not hist:
            rec["exclusion_reason"] = "histology_missing"
            records.append(rec)
            continue
        if hist.lower() not in hmap:
            rec["exclusion_reason"] = "histology_not_adc_or_scc"
            records.append(rec)
            continue
        rec["label"] = hmap[hist.lower()]

        segs = segs_by_patient.get(pid, [])
        if not segs:
            rec["exclusion_reason"] = "no_segmentation"
            records.append(rec)
            continue
        if len(segs) > 1:
            rec["exclusion_reason"] = "multiple_segmentation_series"
            records.append(rec)
            continue
        s = segs[0]
        rec["seg_series_uid"] = s["SeriesInstanceUID"]
        rec["seg_study_uid"] = s["StudyInstanceUID"]
        rec["seg_zip_sha256"] = s["zip_sha256"]

        dcm = seg_dicom_path(cfg, s["SeriesInstanceUID"])
        summary = segment_summary(dcm)
        rec["seg_n_segments"] = len(summary)
        seg_no = _pick_tumour_segment(cfg, summary)
        if seg_no is None:
            rec["exclusion_reason"] = "ambiguous_tumour_segment"
            records.append(rec)
            continue
        rec["seg_segment_number"] = seg_no
        rec["seg_segment_label"] = summary[seg_no]["label"]
        rec["seg_property_type"] = summary[seg_no]["property_type"]
        rec["seg_algorithm"] = summary[seg_no]["algorithm"]

        refs = referenced_series_of_zip_member(dcm)
        if not refs:
            rec["exclusion_reason"] = "segmentation_references_no_ct"
            records.append(rec)
            continue
        if len(refs) > 1:
            rec["exclusion_reason"] = "segmentation_references_multiple_ct"
            records.append(rec)
            continue
        ct_uid = refs[0]
        ct = by_uid.get(ct_uid)
        if ct is None or str(ct.get("Modality")) != str(cfg["ct_selection"]["require_modality"]):
            rec["exclusion_reason"] = "referenced_ct_not_in_collection"
            records.append(rec)
            continue
        if str(ct["PatientID"]) != pid:
            rec["exclusion_reason"] = "referenced_ct_patient_mismatch"
            records.append(rec)
            continue
        rec["ct_series_uid"] = ct_uid
        rec["ct_study_uid"] = ct["StudyInstanceUID"]
        rec["ct_expected_slices"] = int(ct.get("ImageCount") or 0)
        rec["ct_filesize_bytes"] = int(ct.get("FileSize") or 0)
        rec["ct_series_description"] = str(ct.get("SeriesDescription", ""))
        rec["ct_manufacturer"] = str(ct.get("Manufacturer", ""))
        rec["ct_model"] = str(ct.get("ManufacturerModelName", ""))
        rec["included"] = True                       # provisionally; --build decides
        records.append(rec)

    out = {"collection": cfg["source"]["collection"],
           "n_subjects": len(records),
           "n_screened_in": int(sum(r["included"] for r in records)),
           "reason_counts": _counts(r["exclusion_reason"] for r in records if not r["included"]),
           "histology_counts": _counts(r["histology"] for r in records),
           "subjects": records}
    path = os.path.join(ppath(cfg, "metadata_dir"), "screening.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("screened %d subjects, %d candidates" % (out["n_subjects"], out["n_screened_in"]))
    print(json.dumps(out["reason_counts"], indent=1))
    return out


def _pick_tumour_segment(cfg, summary):
    ts = cfg["eligibility"]["tumour_segment"]
    if len(summary) == 1 and bool(ts["accept_single_segment"]):
        return sorted(summary)[0]
    want = [t.strip().lower() for t in ts.get("property_types", [])]
    cands = [n for n, s in sorted(summary.items())
             if str(s["property_type"]).strip().lower() in want]
    if len(cands) == 1:
        return cands[0]
    return None


def _counts(it):
    out = {}
    for v in it:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))


# ---------------------------------------------------------------------------
# stage 2: per-subject construction
# ---------------------------------------------------------------------------

def patient_dir(cfg, pid):
    return os.path.join(ppath(cfg, "masks_dir"), pid)


def product_paths(cfg, pid):
    d = patient_dir(cfg, pid)
    return {
        "dir": d,
        "ct": os.path.join(d, cfg["paths"]["ct_filename"]),
        "gtv": os.path.join(d, cfg["paths"]["gtv_filename"]),
        "rim": os.path.join(d, cfg["paths"]["rim_filename"]),
        "gtv_rim": os.path.join(d, cfg["paths"]["gtv_rim_filename"]),
        "record": os.path.join(d, "record.json"),
    }


def already_built(cfg, pid):
    p = product_paths(cfg, pid)
    if not all(os.path.isfile(p[k]) for k in ("ct", "gtv", "rim", "gtv_rim", "record")):
        return None
    try:
        with open(p["record"], "r", encoding="utf-8") as fh:
            rec = json.load(fh)
    except Exception:                                              # noqa: BLE001
        return None
    if rec.get("status") != "ok" or rec.get("schema") != 1:
        return None
    return rec


def ct_zip_path(cfg, pid):
    return os.path.join(ppath(cfg, "work_dir"), "ct", pid, "ct.zip")


def prefetch_ct(cfg, subject):
    """Download one CT series zip into the work directory if it is not there."""
    pid = subject["PatientID"]
    zpath = ct_zip_path(cfg, pid)
    if os.path.isfile(zpath):
        return zpath, "cached"
    os.makedirs(os.path.dirname(zpath), exist_ok=True)
    download_series_zip(cfg, subject["ct_series_uid"], zpath)
    return zpath, "downloaded"


def build_one(cfg, subject, logpath=None):
    """Download the CT, decode the SEG, build the three masks.  Returns a record."""
    pid = subject["PatientID"]
    p = product_paths(cfg, pid)
    os.makedirs(p["dir"], exist_ok=True)
    work = os.path.join(ppath(cfg, "work_dir"), "ct", pid)
    rec = {"schema": 1, "PatientID": pid, "status": "ok", "exclusion_reason": "",
           "histology": subject["histology"], "label": int(subject["label"]),
           "seg_series_uid": subject["seg_series_uid"],
           "seg_study_uid": subject["seg_study_uid"],
           "seg_zip_sha256": subject["seg_zip_sha256"],
           "seg_segment_number": subject.get("seg_segment_number"),
           "ct_series_uid": subject["ct_series_uid"],
           "ct_study_uid": subject["ct_study_uid"]}
    t0 = time.time()
    try:
        # ---- CT ------------------------------------------------------------
        os.makedirs(work, exist_ok=True)
        zpath = os.path.join(work, "ct.zip")
        if not os.path.isfile(zpath):
            n_bytes, digest = download_series_zip(cfg, subject["ct_series_uid"], zpath)
        else:
            n_bytes, digest = os.path.getsize(zpath), sha256_file(zpath)
        rec["ct_zip_bytes"] = int(n_bytes)
        rec["ct_zip_sha256"] = digest
        ddir = os.path.join(work, "dcm")
        if not os.path.isdir(ddir):
            os.makedirs(ddir, exist_ok=True)
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(ddir)
        files = sorted(os.path.join(r, f) for r, _, fs in os.walk(ddir)
                       for f in fs if f.lower().endswith(".dcm"))
        rec["ct_n_dicom_files"] = len(files)
        try:
            ct = read_ct(files,
                         spacing_tol=float(cfg["ct_selection"]["slice_spacing_tolerance_mm"]),
                         min_slices=int(cfg["ct_selection"]["min_slices"]))
        except Exception as exc:                                   # noqa: BLE001
            raise _Excluded("ct_unreadable", str(exc)[:300])
        rec["ct_header"] = ct.header
        rec["ct_warnings"] = ct.warnings
        rec["acquisition"] = acquisition_tags(ct.files[0],
                                              cfg["batch_descriptors"]["dicom_tags"])

        # ---- SEG -----------------------------------------------------------
        dcm = seg_dicom_path(cfg, subject["seg_series_uid"])
        try:
            segments, masks, info = read_seg(dcm, ct)
        except Exception as exc:                                   # noqa: BLE001
            raise _Excluded("segmentation_misaligned", str(exc)[:300])
        rec["seg_info"] = {k: v for k, v in info.items() if k != "warnings"}
        rec["seg_warnings"] = info["warnings"]
        n_lost = (int(info["unmapped_nonempty_frames"])
                  + int(info.get("unassigned_nonempty_frames", 0)))
        if n_lost > int(cfg["eligibility"]["max_unmapped_seg_frames"]):
            raise _Excluded("segmentation_unmapped_frames",
                            "%d non-empty SEG frames could not be placed on a CT slice"
                            % n_lost)
        seg_no = int(subject["seg_segment_number"])
        gtv = masks[seg_no]
        n_gtv = int(sitk.GetArrayViewFromImage(gtv).sum())
        rec["gtv_voxels_native"] = n_gtv
        if n_gtv < int(cfg["eligibility"]["min_gtv_voxels"]):
            raise _Excluded("segmentation_empty",
                            "GTV has %d voxels (< %d)"
                            % (n_gtv, cfg["eligibility"]["min_gtv_voxels"]))

        # ---- regions --------------------------------------------------------
        try:
            region_masks, rinfo = build_regions(gtv, cfg["regions"])
        except Exception as exc:                                   # noqa: BLE001
            raise _Excluded("region_construction_failed", str(exc)[:300])
        rec["regions"] = rinfo

        if bool(cfg["regions"]["verification"]["require_masks_match_ct_geometry"]):
            for k, m in region_masks.items():
                if not same_geometry(m, ct.image):
                    raise _Excluded("region_construction_failed",
                                    "%s mask geometry does not match the CT" % k)

        # ---- write ----------------------------------------------------------
        sitk.WriteImage(ct.image, p["ct"], True)
        sitk.WriteImage(region_masks["gtv"], p["gtv"], True)
        sitk.WriteImage(region_masks["rim"], p["rim"], True)
        sitk.WriteImage(region_masks["gtv_rim"], p["gtv_rim"], True)
        rec["files"] = {k: {"path": os.path.relpath(p[k], cfg["_project_root"]).replace(os.sep, "/"),
                            "sha256": sha256_file(p[k]), "bytes": os.path.getsize(p[k])}
                        for k in ("ct", "gtv", "rim", "gtv_rim")}
    except _Excluded as exc:
        rec["status"] = "excluded"
        rec["exclusion_reason"] = exc.reason
        rec["exclusion_detail"] = exc.detail
    except Exception as exc:                                       # noqa: BLE001
        rec["status"] = "excluded"
        rec["exclusion_reason"] = "ct_download_failed"
        rec["exclusion_detail"] = str(exc)[:300]
    finally:
        if not bool(cfg["source"]["download"]["keep_ct_dicom"]):
            shutil.rmtree(work, ignore_errors=True)

    rec["elapsed_s"] = round(time.time() - t0, 2)
    with open(p["record"], "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    _log("  %s  %s  %s  %.1fs" % (pid, rec["status"], rec["exclusion_reason"] or "-",
                                  rec["elapsed_s"]), logpath)
    return rec


class _Excluded(Exception):
    def __init__(self, reason, detail=""):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


def build(cfg, limit=None, force=False, lookahead=4):
    """Convert every screened candidate, downloading the next CTs in parallel.

    The TCIA getImage endpoint is the bottleneck (~0.5 MB/s, and measured to
    gain nothing from concurrent requests), so exactly ONE download runs at a
    time; it simply runs ahead of the conversion instead of blocking it.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor

    path = os.path.join(ppath(cfg, "metadata_dir"), "screening.json")
    with open(path, "r", encoding="utf-8") as fh:
        scr = json.load(fh)
    cands = [s for s in scr["subjects"] if s["included"]]
    if not force:
        todo = [s for s in cands if already_built(cfg, s["PatientID"]) is None]
    else:
        todo = list(cands)
    skipped = len(cands) - len(todo)
    if limit is not None:
        todo = todo[:int(limit)]

    logpath = ppath(cfg, "log")
    _log("building %d of %d candidates (%d already valid)"
         % (len(todo), len(cands), skipped), logpath)

    sem = threading.Semaphore(int(lookahead))
    futures = {}

    def _job(subject):
        try:
            return prefetch_ct(cfg, subject)
        except Exception as exc:                                   # noqa: BLE001
            return None, "failed: %s" % str(exc)[:200]

    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=1) as pool:
        def _submit(subject):
            sem.acquire()

            def _wrapped(s=subject):
                try:
                    return _job(s)
                finally:
                    pass
            futures[subject["PatientID"]] = pool.submit(_wrapped)

        for s in todo[:int(lookahead)]:
            _submit(s)

        for i, s in enumerate(todo, 1):
            pid = s["PatientID"]
            fut = futures.get(pid)
            if fut is not None:
                fut.result()
                sem.release()
            nxt = i - 1 + int(lookahead)
            if nxt < len(todo):
                _submit(todo[nxt])
            _log("[%3d/%3d] %s" % (i, len(todo), pid), logpath)
            build_one(cfg, s, logpath)
            done += 1
            el = time.time() - t0
            _log("        %d/%d done, %.1f min elapsed, ~%.1f min remaining"
                 % (done, len(todo), el / 60.0,
                    (el / done) * (len(todo) - done) / 60.0), logpath)

    _log("built %d, skipped %d already-valid" % (done, skipped), logpath)
    return done, skipped


# ---------------------------------------------------------------------------
# stage 3: the cohort files
# ---------------------------------------------------------------------------

def assemble_cohort(cfg):
    path = os.path.join(ppath(cfg, "metadata_dir"), "screening.json")
    with open(path, "r", encoding="utf-8") as fh:
        scr = json.load(fh)

    rows = []
    for s in scr["subjects"]:
        pid = s["PatientID"]
        row = {
            "PatientID": pid,
            "Histology": s["histology"],
            "label": s["label"] if s["label"] is not None else "",
            "included": False,
            "exclusion_reason": s["exclusion_reason"],
            "exclusion_detail": "",
            "n_clinical_rows": s["n_clinical_rows"],
            "n_seg_series": s["n_seg_series"],
            "seg_series_uid": s["seg_series_uid"],
            "seg_study_uid": s["seg_study_uid"],
            "seg_zip_sha256": s["seg_zip_sha256"],
            "seg_segment_label": s["seg_segment_label"],
            "seg_property_type": s["seg_property_type"],
            "ct_series_uid": s["ct_series_uid"],
            "ct_study_uid": s["ct_study_uid"],
        }
        if s["included"]:
            rec = already_built(cfg, pid)
            if rec is None:
                p = product_paths(cfg, pid)
                if os.path.isfile(p["record"]):
                    with open(p["record"], "r", encoding="utf-8") as fh:
                        rec = json.load(fh)
                else:
                    rec = {"status": "excluded", "exclusion_reason": "not_built",
                           "exclusion_detail": "no per-patient record on disk"}
            row["exclusion_reason"] = rec.get("exclusion_reason", "")
            row["exclusion_detail"] = rec.get("exclusion_detail", "")
            row["included"] = rec.get("status") == "ok"
            if row["included"]:
                g = rec["regions"]
                h = rec["ct_header"]
                row.update({
                    "ct_n_slices": h["n_slices"],
                    "ct_rows": h["rows"], "ct_columns": h["columns"],
                    "ct_pixel_spacing_mm": h["pixel_spacing_mm"][0],
                    "ct_slice_spacing_mm": h["slice_spacing_mm_median"],
                    "gtv_voxels_native": g["native_gtv_voxels"],
                    "rim_voxels_native": g["native_rim_voxels"],
                    "gtv_rim_voxels_native": g["native_gtv_rim_voxels"],
                    "gtv_volume_mm3": g["native_gtv_volume_mm3"],
                    "rim_volume_mm3": g["native_rim_volume_mm3"],
                    "gtv_rim_volume_mm3": g["native_gtv_rim_volume_mm3"],
                    "rim_fraction_of_gtv_rim": g["native_rim_fraction_of_gtv_rim"],
                    "gtv_touches_volume_boundary": g["gtv_touches_volume_boundary"],
                    "disjoint_ok": g["disjoint_ok"], "union_ok": g["union_ok"],
                    "subset_ok": g["subset_ok"],
                    "ct_sha256": rec["files"]["ct"]["sha256"],
                    "gtv_sha256": rec["files"]["gtv"]["sha256"],
                    "rim_sha256": rec["files"]["rim"]["sha256"],
                    "gtv_rim_sha256": rec["files"]["gtv_rim"]["sha256"],
                })
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("PatientID").reset_index(drop=True)
    order = {r: i for i, r in enumerate(RULE_ORDER)}
    df["_rule_rank"] = df["exclusion_reason"].map(lambda r: order.get(r, 999))
    df = df.drop(columns=["_rule_rank"])
    df.to_csv(ppath(cfg, "cohort_csv"), index=False)

    ex = df[~df["included"]][["PatientID", "Histology", "exclusion_reason",
                              "exclusion_detail", "n_clinical_rows", "n_seg_series",
                              "seg_series_uid", "ct_series_uid"]].copy()
    ex.to_csv(ppath(cfg, "exclusions_csv"), index=False)

    inc = df[df["included"]]
    summary = {
        "n_subjects_in_collection": int(len(df)),
        "n_included": int(len(inc)),
        "n_excluded": int(len(df) - len(inc)),
        "n_adc": int((inc["label"] == 1).sum()),
        "n_scc": int((inc["label"] == 0).sum()),
        "exclusion_reason_counts": _counts(df[~df["included"]]["exclusion_reason"]),
        "histology_counts_all": _counts(df["Histology"]),
        "histology_counts_included": _counts(inc["Histology"]),
    }
    print(json.dumps(summary, indent=1))
    return df, summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--cohort", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    if args.screen:
        screen(cfg)
    if args.build:
        build(cfg, limit=args.limit, force=args.force)
    if args.cohort:
        assemble_cohort(cfg)
    if not (args.screen or args.build or args.cohort):
        ap.error("choose --screen, --build and/or --cohort")
    return 0


if __name__ == "__main__":
    sys.exit(main())
