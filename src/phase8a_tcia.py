"""Phase 8A - TCIA / NBIA access for the external NSCLC-Radiogenomics cohort.

Read-only with respect to everything the project already owns.  This module
only downloads public data and writes it into the Phase-8A namespace.

    python src/phase8a_tcia.py --config config/phase8a_external.yaml --metadata
    python src/phase8a_tcia.py --config config/phase8a_external.yaml --seg

`--metadata` fetches the collection's patient and series listings plus the
public clinical CSV.  `--seg` downloads every DICOM SEG series of the
collection (they are small) and records a SHA-256 for each.  Both steps are
resumable: an existing artefact with a matching hash is not re-fetched.

No label is read here.  No image is inspected here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import zipfile

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

def load_cfg(path):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = os.path.abspath(path)
    cfg["_project_root"] = PROJECT_ROOT
    return cfg


def ppath(cfg, key_or_rel):
    rel = cfg["paths"].get(key_or_rel, key_or_rel)
    if os.path.isabs(rel):
        return rel
    return os.path.join(cfg["_project_root"], *rel.split("/"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _get(url, timeout, retries, sleep_s):
    last = None
    for attempt in range(int(retries) + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "ADC-SCC-MIL/phase8a"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 200:
                    raise RuntimeError("HTTP %s" % resp.status)
                return resp.read()
        except Exception as exc:                                   # noqa: BLE001
            last = exc
            if attempt < int(retries):
                time.sleep(float(sleep_s) * (attempt + 1))
    raise RuntimeError("GET failed after %d attempts: %s (%s)"
                       % (int(retries) + 1, url, str(last)[:200]))


def api_get(cfg, endpoint, params=None):
    dl = cfg["source"]["download"]
    url = "%s/%s" % (cfg["source"]["nbia_api"].rstrip("/"), endpoint)
    if params:
        url = "%s?%s" % (url, urllib.parse.urlencode(params))
    raw = _get(url, dl["timeout_s"], dl["retries"], dl["retry_sleep_s"])
    return json.loads(raw.decode("utf-8"))


def download_series_zip(cfg, series_uid, dest_zip):
    """Download one DICOM series as a zip.  Returns (bytes_written, sha256)."""
    dl = cfg["source"]["download"]
    url = "%s/getImage?%s" % (cfg["source"]["nbia_api"].rstrip("/"),
                              urllib.parse.urlencode({"SeriesInstanceUID": series_uid}))
    raw = _get(url, dl["timeout_s"], dl["retries"], dl["retry_sleep_s"])
    os.makedirs(os.path.dirname(dest_zip), exist_ok=True)
    tmp = dest_zip + ".part"
    with open(tmp, "wb") as fh:
        fh.write(raw)
    os.replace(tmp, dest_zip)
    return len(raw), sha256_bytes(raw)


def extract_zip(zip_path, dest_dir):
    """Extract a TCIA series zip; returns the sorted list of .dcm paths."""
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    out = []
    for root, _, files in os.walk(dest_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                out.append(os.path.join(root, f))
    return sorted(out)


# ---------------------------------------------------------------------------
# steps
# ---------------------------------------------------------------------------

def fetch_metadata(cfg):
    """Collection patient list, series list and the public clinical CSV."""
    coll = cfg["source"]["collection"]
    os.makedirs(ppath(cfg, "metadata_dir"), exist_ok=True)

    patients = api_get(cfg, "getPatient", {"Collection": coll})
    series = api_get(cfg, "getSeries", {"Collection": coll})

    with open(ppath(cfg, "tcia_patients_json"), "w", encoding="utf-8") as fh:
        json.dump(patients, fh, indent=1)
    with open(ppath(cfg, "tcia_series_json"), "w", encoding="utf-8") as fh:
        json.dump(series, fh, indent=1)

    dl = cfg["source"]["download"]
    raw = _get(cfg["source"]["clinical_csv_url"],
               dl["timeout_s"], dl["retries"], dl["retry_sleep_s"])
    clin_path = ppath(cfg, "clinical_csv")
    with open(clin_path, "wb") as fh:
        fh.write(raw)

    info = {
        "collection": coll,
        "n_patients": len(patients),
        "n_series": len(series),
        "modalities": _counts(s["Modality"] for s in series),
        "clinical_csv_url": cfg["source"]["clinical_csv_url"],
        "clinical_csv_sha256": sha256_bytes(raw),
        "clinical_csv_bytes": len(raw),
        "tcia_patients_sha256": sha256_file(ppath(cfg, "tcia_patients_json")),
        "tcia_series_sha256": sha256_file(ppath(cfg, "tcia_series_json")),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(info, indent=1))
    return info


def _counts(it):
    out = {}
    for v in it:
        out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def fetch_segmentations(cfg):
    """Download every SEG series zip of the collection.  Resumable."""
    with open(ppath(cfg, "tcia_series_json"), "r", encoding="utf-8") as fh:
        series = json.load(fh)
    segs = [s for s in series if s["Modality"] == "SEG"]
    segs.sort(key=lambda s: (s["PatientID"], s["SeriesInstanceUID"]))

    out_dir = ppath(cfg, "seg_zip_dir")
    os.makedirs(out_dir, exist_ok=True)
    recs, n_new = [], 0
    for i, s in enumerate(segs, 1):
        uid = s["SeriesInstanceUID"]
        dest = os.path.join(out_dir, "%s.zip" % uid)
        if os.path.isfile(dest):
            n_bytes, digest = os.path.getsize(dest), sha256_file(dest)
        else:
            n_bytes, digest = download_series_zip(cfg, uid, dest)
            n_new += 1
            print("  [%3d/%3d] %s  %s  %.1f kB"
                  % (i, len(segs), s["PatientID"], uid[-12:], n_bytes / 1024.0),
                  flush=True)
        recs.append({"PatientID": s["PatientID"], "SeriesInstanceUID": uid,
                     "StudyInstanceUID": s["StudyInstanceUID"],
                     "SeriesDescription": s.get("SeriesDescription", ""),
                     "Manufacturer": s.get("Manufacturer", ""),
                     "zip_bytes": int(n_bytes), "zip_sha256": digest})

    with open(ppath(cfg, "seg_index_json"), "w", encoding="utf-8") as fh:
        json.dump({"collection": cfg["source"]["collection"],
                   "n_seg_series": len(recs),
                   "n_subjects_with_seg": len({r["PatientID"] for r in recs}),
                   "n_downloaded_this_run": n_new,
                   "series": recs}, fh, indent=1)
    print("SEG series: %d (%d subjects), %d newly downloaded"
          % (len(recs), len({r["PatientID"] for r in recs}), n_new))
    return recs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--metadata", action="store_true")
    ap.add_argument("--seg", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_cfg(args.config)
    if not (args.metadata or args.seg):
        ap.error("choose --metadata and/or --seg")
    if args.metadata:
        fetch_metadata(cfg)
    if args.seg:
        fetch_segmentations(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
