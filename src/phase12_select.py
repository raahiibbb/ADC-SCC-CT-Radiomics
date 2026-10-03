"""Phase 12 - parse NLST-Sybil box SRs, select one CT series per eligible
patient (NLST) and one per TCGA patient, write download manifests + size.

Series choice uses annotation / DICOM metadata only.  NLST histology is used
only for inclusion (ADC or SCC first primary).  TCGA labels are NOT read
(locked test; label = collection and is materialised only in Phase 14).

Usage: python src/phase12_select.py
Outputs: C:/LUNG_phase12/{nlst_boxes.parquet, nlst_selection.csv,
         tcga_selection.csv, download_*.txt}; cohort/phase12_nlst_candidates.csv
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import pydicom
import yaml


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def cfg_load():
    return yaml.safe_load(open(os.path.join(ROOT, "config", "phase12.yaml")))


def parse_sr(sr_dir):
    rows = []
    for f in glob.glob(os.path.join(sr_dir, "*", "*")):
        ds = pydicom.dcmread(f)
        ev = ds.CurrentRequestedProcedureEvidenceSequence[0]
        ct_series = ev.ReferencedSeriesSequence[0].SeriesInstanceUID
        tp = str(getattr(ds, "ClinicalTrialTimePointID", ""))
        for grp in ds.ContentSequence:
            if grp.ConceptNameCodeSequence[0].CodeMeaning != "Imaging Measurements":
                continue
            for mg in grp.ContentSequence:
                for it in mg.ContentSequence:
                    if it.ValueType == "SCOORD":
                        g = np.array(it.GraphicData, float).reshape(-1, 2)
                        rows.append(dict(PatientID=str(ds.PatientID), StudyInstanceUID=ds.StudyInstanceUID,
                                         CTSeriesInstanceUID=ct_series, timepoint=tp,
                                         SOPInstanceUID=it.ContentSequence[0].ReferencedSOPSequence[0].ReferencedSOPInstanceUID,
                                         x0=g[:, 0].min(), y0=g[:, 1].min(), x1=g[:, 0].max(), y1=g[:, 1].max()))
    return pd.DataFrame(rows)


def nlst_labels(c):
    d = pd.read_parquet(c["clinical_canc"])
    d["pid"] = d["pid"].astype(str)
    n_canc = d.groupby("pid").size()
    first = d[pd.to_numeric(d.lc_order, errors="coerce") == 1].drop_duplicates("pid").set_index("pid")
    m = pd.to_numeric(first.lc_morph, errors="coerce")
    lab = pd.Series(np.nan, index=first.index)
    lab[m.isin(c["adc_codes"])] = 1
    lab[m.isin(c["scc_codes"])] = 0
    out = pd.DataFrame(dict(y=lab, lc_morph=m, n_cancers=n_canc.reindex(first.index).values,
                            stage=first.de_stag_7thed, lc_topog=first.lc_topog, lesionsize=first.lesionsize))
    return out


def main():
    cfg = cfg_load()
    c = cfg["nlst"]
    root = cfg["root"]
    from idc_index import IDCClient
    idc = IDCClient()
    # ---------------- NLST
    bp = os.path.join(root, "nlst_boxes.parquet")
    boxes = pd.read_parquet(bp) if os.path.isfile(bp) else parse_sr(c["sr_dir"])
    boxes.to_parquet(bp)
    print("boxes", len(boxes), "series", boxes.CTSeriesInstanceUID.nunique(), "patients", boxes.PatientID.nunique())
    per = boxes.groupby(["PatientID", "CTSeriesInstanceUID", "timepoint"]).agg(
        n_boxes=("SOPInstanceUID", "size"), n_slices=("SOPInstanceUID", "nunique")).reset_index()
    meta = idc.sql_query("SELECT SeriesInstanceUID, series_size_MB, instanceCount, SeriesDescription, "
                         "series_aws_url, Manufacturer FROM index WHERE collection_id='nlst' AND Modality='CT'")
    per = per.merge(meta, left_on="CTSeriesInstanceUID", right_on="SeriesInstanceUID", how="left")
    lab = nlst_labels(c)
    per = per.join(lab, on="PatientID")
    per["tp_num"] = pd.to_numeric(per.timepoint.str.extract(r"(\d+)")[0], errors="coerce")
    elig = per[per.y.notna() & per.series_size_MB.notna() & (per.series_size_MB <= c["max_series_mb"])]
    if c["exclude_multiple_cancers"]:
        elig = elig[elig.n_cancers == 1]
    sel = (elig.sort_values(["PatientID", "tp_num", "n_slices"], ascending=[True, False, False])
               .drop_duplicates("PatientID"))
    sel.to_csv(os.path.join(root, "nlst_selection.csv"), index=False)
    print("NLST annotated patients:", per.PatientID.nunique(),
          "| with ADC/SCC label:", per[per.y.notna()].PatientID.nunique(),
          "| selected:", len(sel), "ADC", int(sel.y.sum()), "SCC", int((sel.y == 0).sum()),
          "| GB %.1f" % (sel.series_size_MB.sum() / 1024))
    excl = per.drop_duplicates("PatientID").assign(
        reason=lambda d: np.where(d.y.isna(), "not ADC/SCC or no first-cancer record",
                                  np.where(d.n_cancers > 1, "multiple primaries", "")))
    print(excl.reason.value_counts().to_dict())
    os.makedirs(os.path.join(ROOT, "cohort"), exist_ok=True)
    sel[["PatientID", "CTSeriesInstanceUID", "timepoint", "n_slices", "series_size_MB", "instanceCount",
         "SeriesDescription", "Manufacturer", "y", "lc_morph", "stage"]].to_csv(
        os.path.join(ROOT, "cohort", "phase12_nlst_candidates.csv"), index=False)
    # ---------------- TCGA (labels not read)
    t = cfg["tcga"]
    q = ("SELECT collection_id, PatientID, StudyInstanceUID, StudyDate, SeriesInstanceUID, SeriesDescription, "
         "instanceCount, series_size_MB, series_aws_url, Manufacturer FROM index WHERE collection_id IN ('%s') "
         "AND Modality='CT'" % "','".join(t["collections"]))
    tc = idc.sql_query(q)
    tc = tc[(tc.instanceCount >= t["min_images"]) & (tc.series_size_MB <= t["max_series_mb"])]
    desc = tc.SeriesDescription.fillna("").str.lower()
    tc = tc[~desc.str.contains("scout|topo|locali|mip|cor|sag|dose|report")]
    first_study = tc.sort_values("StudyDate").groupby("PatientID").StudyInstanceUID.first()
    tc = tc[tc.StudyInstanceUID.values == first_study.reindex(tc.PatientID).values]
    tsel = tc.sort_values(["PatientID", "instanceCount"], ascending=[True, False]).drop_duplicates("PatientID")
    tsel.drop(columns=["collection_id"]).to_csv(os.path.join(root, "tcga_selection.csv"), index=False)
    tsel[["PatientID", "collection_id"]].to_csv(os.path.join(root, "tcga_LOCKED_labels.csv"), index=False)
    print("TCGA patients selected:", len(tsel), "| GB %.1f" % (tsel.series_size_MB.sum() / 1024))
    # ---------------- manifests
    for name, df, key in (("nlst", sel, "CTSeriesInstanceUID"), ("tcga", tsel, "SeriesInstanceUID")):
        with open(os.path.join(root, f"download_{name}.txt"), "w") as f:
            for url, uid in zip(df.series_aws_url, df[key]):
                f.write(f"cp {url} {name}/dicom/{uid}/\n")


if __name__ == "__main__":
    sys.exit(main())
