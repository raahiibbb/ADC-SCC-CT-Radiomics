"""Phase 11 - build the Lung-PET-CT-Dx cohort (ADC 'A' / SCC 'G' patients).

  select   : for every CT series of every A/G patient, fetch its SOPInstanceUIDs
             from the NBIA API and count the PASCAL-VOC box annotations (files
             are named by SOPInstanceUID) that fall in it.  Keep the series
             with the most annotated slices (ties -> more images).
  download : fetch the selected series, convert to CT.nii.gz, and rasterise the
             2-D boxes of every annotated slice into a 3-D BOX.nii.gz on the
             same grid.  A record.json holds geometry + acquisition metadata.

No image content and no label is used to choose anything.  Resumable.

Usage:
  python src/phase11_lpcd_build.py select
  python src/phase11_lpcd_build.py download [--limit N]
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def cfg_load():
    c = yaml.safe_load(open(os.path.join(ROOT, "config", "phase11.yaml"), encoding="utf-8"))
    return c["lpcd"]


def http(url, c):
    last = None
    for _ in range(c["retries"] + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ADC-SCC-MIL/phase11"})
            with urllib.request.urlopen(req, timeout=c["timeout_s"]) as r:
                return r.read()
        except Exception as e:  # network hiccups are retried, then raised
            last = e
            time.sleep(c["retry_sleep_s"])
    raise RuntimeError("GET failed: %s (%s)" % (url[:120], last))


def api(c, endpoint, **params):
    return http("%s/%s?%s" % (c["nbia_api"], endpoint, urllib.parse.urlencode(params)), c)


# --------------------------------------------------------------------- select
def select(c):
    out_csv = os.path.join(c["data_root"], "series_selection.csv")
    cache_p = os.path.join(c["data_root"], "sop_cache.json")
    cache = json.load(open(cache_p)) if os.path.isfile(cache_p) else {}
    s = pd.DataFrame(json.load(open(os.path.join(ROOT, *c["series_listing"].split("/")))))
    s["letter"] = s["PatientID"].str.extract(r"Lung_Dx-([A-Z])")
    s = s[(s["Modality"] == "CT") & s["letter"].isin(list(c["histology_letters"]))]
    todo = [u for u in s["SeriesInstanceUID"] if u not in cache]
    print("series needing SOP lists:", len(todo), flush=True)

    def fetch(u):
        return u, [d["SOPInstanceUID"] for d in json.loads(api(c, "getSOPInstanceUIDs", SeriesInstanceUID=u))]

    with ThreadPoolExecutor(8) as ex:
        for i, f in enumerate(as_completed([ex.submit(fetch, u) for u in todo])):
            u, sops = f.result()
            cache[u] = sops
            if (i + 1) % 100 == 0:
                json.dump(cache, open(cache_p, "w"))
                print("  %d/%d" % (i + 1, len(todo)), flush=True)
    json.dump(cache, open(cache_p, "w"))

    rows = []
    for pid, g in s.groupby("PatientID"):
        short = pid.replace("Lung_Dx-", "")
        adir = os.path.join(c["annotations_dir"], short)
        ann = {f[:-4] for f in os.listdir(adir)} if os.path.isdir(adir) else set()
        best = None
        for _, r in g.iterrows():
            n_ann = len(ann & set(cache[r["SeriesInstanceUID"]]))
            key = (n_ann, int(r["ImageCount"]))
            if best is None or key > best[0]:
                best = (key, r)
        (n_ann, n_img), r = best
        rows.append(dict(PatientID=pid, short_id=short, letter=r["letter"],
                         histology=c["histology_letters"][r["letter"]],
                         label=int(r["letter"] == "A"), n_ct_series=len(g),
                         n_annotation_files=len(ann), n_annotated_in_series=n_ann,
                         SeriesInstanceUID=r["SeriesInstanceUID"], ImageCount=n_img,
                         SeriesDescription=r.get("SeriesDescription"),
                         Manufacturer=r.get("Manufacturer"), FileSize=r["FileSize"],
                         StudyDate=r.get("StudyDate"),
                         selectable=bool(n_ann > 0)))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    ok = df[df.selectable]
    print(df.groupby(["letter", "selectable"]).size())
    print("download size of selected series: %.1f GB" % (ok.FileSize.sum() / 1e9))


# ------------------------------------------------------------------- download
def boxes_for(c, short, sop):
    p = os.path.join(c["annotations_dir"], short, sop + ".xml")
    root = ET.parse(p).getroot()
    out = []
    for o in root.iter("object"):
        b = o.find("bndbox")
        out.append(tuple(int(float(b.find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")))
    return out


def build_patient(c, row):
    pid, short, uid = row["PatientID"], row["short_id"], row["SeriesInstanceUID"]
    dest = os.path.join(c["data_root"], "nifti", pid)
    rec_p = os.path.join(dest, "record.json")
    if os.path.isfile(rec_p) and json.load(open(rec_p)).get("status") == "ok":
        return pid, "skip"
    os.makedirs(dest, exist_ok=True)
    zdir = os.path.join(c["data_root"], "zips")
    os.makedirs(zdir, exist_ok=True)
    zp = os.path.join(zdir, pid + ".zip")
    if not (os.path.isfile(zp) and zipfile.is_zipfile(zp)):
        raw = api(c, "getImage", SeriesInstanceUID=uid)
        with open(zp + ".part", "wb") as f:
            f.write(raw)
        os.replace(zp + ".part", zp)
    tmp = tempfile.mkdtemp(prefix="lpcd_")
    try:
        with zipfile.ZipFile(zp) as z:
            z.extractall(tmp)
        files = [os.path.join(dp, f) for dp, _, fs in os.walk(tmp) for f in fs if f.lower().endswith(".dcm")]
        ds = []
        for f in files:
            d = pydicom.dcmread(f, stop_before_pixels=True)
            ds.append((float(d.ImagePositionPatient[2]), f, str(d.SOPInstanceUID), d))
        ds.sort(key=lambda t: t[0])
        # guard against duplicated positions (multi-phase) - keep the first
        seen, uniq = set(), []
        for t in ds:
            k = round(t[0], 3)
            if k not in seen:
                seen.add(k)
                uniq.append(t)
        rd = sitk.ImageSeriesReader()
        rd.SetFileNames([t[1] for t in uniq])
        img = rd.Execute()
        arr_shape = sitk.GetArrayFromImage(img).shape          # (z, y, x)
        box = np.zeros(arr_shape, np.uint8)
        n_boxes = 0
        for zi, (_, _, sop, _) in enumerate(uniq):
            ap = os.path.join(c["annotations_dir"], short, sop + ".xml")
            if os.path.isfile(ap):
                for (x0, y0, x1, y1) in boxes_for(c, short, sop):
                    box[zi, max(y0, 0):y1 + 1, max(x0, 0):x1 + 1] = 1
                    n_boxes += 1
        if n_boxes == 0:
            raise RuntimeError("no annotated slice matched after conversion")
        bimg = sitk.GetImageFromArray(box)
        bimg.CopyInformation(img)
        sitk.WriteImage(img, os.path.join(dest, "CT.nii.gz"))
        sitk.WriteImage(bimg, os.path.join(dest, "BOX.nii.gz"))
        d0 = uniq[len(uniq) // 2][3]
        zs = np.array([t[0] for t in uniq])
        rec = dict(status="ok", PatientID=pid, SeriesInstanceUID=uid, n_slices=len(uniq),
                   n_files=len(files), n_boxes=int(n_boxes),
                   n_annotated_slices=int((box.reshape(box.shape[0], -1).max(1) > 0).sum()),
                   spacing=list(img.GetSpacing()), size=list(img.GetSize()),
                   slice_spacing_median=float(np.median(np.diff(zs))) if len(zs) > 1 else None,
                   Manufacturer=str(getattr(d0, "Manufacturer", "")),
                   ManufacturerModelName=str(getattr(d0, "ManufacturerModelName", "")),
                   ConvolutionKernel=str(getattr(d0, "ConvolutionKernel", "")),
                   SliceThickness=str(getattr(d0, "SliceThickness", "")),
                   KVP=str(getattr(d0, "KVP", "")),
                   ContrastBolusAgent=str(getattr(d0, "ContrastBolusAgent", "")))
        json.dump(rec, open(rec_p, "w"), indent=1)
        if not c["keep_zips"]:
            os.remove(zp)
        return pid, "ok"
    except Exception as e:
        json.dump(dict(status="failed", PatientID=pid, error=str(e)[:500]), open(rec_p, "w"), indent=1)
        return pid, "failed: %s" % str(e)[:200]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def download(c, limit):
    sel = pd.read_csv(os.path.join(c["data_root"], "series_selection.csv"))
    sel = sel[sel.selectable]
    if limit:
        sel = sel.head(limit)
    t0 = time.time()
    res = []
    with ThreadPoolExecutor(c["download_threads"]) as ex:
        futs = [ex.submit(build_patient, c, r) for _, r in sel.iterrows()]
        for i, f in enumerate(as_completed(futs)):
            pid, st = f.result()
            res.append((pid, st))
            print("[%d/%d %.0fs] %s %s" % (i + 1, len(futs), time.time() - t0, pid, st), flush=True)
    bad = [r for r in res if r[1].startswith("failed")]
    print("done: %d ok/skip, %d failed" % (len(res) - len(bad), len(bad)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "download"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    c = cfg_load()
    os.makedirs(c["data_root"], exist_ok=True)
    select(c) if a.step == "select" else download(c, a.limit)


if __name__ == "__main__":
    main()
