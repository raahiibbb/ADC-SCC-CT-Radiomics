"""Phase 6A - resumable driver for the 2.5D CT image-crop extraction.

    ./.venv/Scripts/python.exe src/run_image_extraction.py --pilot --tag pilot
    ./.venv/Scripts/python.exe src/run_image_extraction.py --all   --tag all

A patient is skipped when an image file already on disk passes
``validate_existing`` against the current crop signature AND has exactly the
frozen bag's instance count AND the same coordinate fingerprint.  Nothing under
``patch_features/``, ``splits/`` or the read-only external trees is written.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from image_patches import (coords_fingerprint, crop_signature,  # noqa: E402
                           eligible_patients, extract_patient_images,
                           load_frozen_bag, load_outer_folds, ppath,
                           validate_existing)
from image_patches import load_config  # noqa: E402

DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), "config", "phase6a.yaml")


def pilot_patients(cfg: dict):
    return [str(p) for p in cfg["pilot"]["patients"]]


def log(fh, msg: str) -> None:
    print(msg, flush=True)
    if fh is not None:
        fh.write(msg + "\n")
        fh.flush()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6A 2.5D CT crop extraction")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--tag", default="all")
    ap.add_argument("--force", action="store_true",
                    help="re-extract even if a valid file exists (do NOT use casually)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    sig = crop_signature(cfg)
    out_dir = ppath(cfg, cfg["outputs"]["image_dir"])
    os.makedirs(out_dir, exist_ok=True)
    log_path = ppath(cfg, cfg["outputs"]["image_log"])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    all_pids = eligible_patients(cfg)
    if args.patients:
        pids = [p for p in args.patients]
    elif args.pilot:
        pids = pilot_patients(cfg)
    elif args.all:
        pids = all_pids
    else:
        ap.error("choose one of --pilot / --all / --patients")

    folds = load_outer_folds(cfg)
    fh = open(log_path, "a", encoding="utf-8")
    log(fh, "=" * 72)
    log(fh, "Phase 6A image extraction  tag=%s  crop_signature=%s  n=%d  %s"
        % (args.tag, sig, len(pids), time.strftime("%Y-%m-%d %H:%M:%S")))

    rows, n_ok, n_skip, n_fail = [], 0, 0, 0
    t0 = time.time()
    for idx, pid in enumerate(pids, 1):
        path = os.path.join(out_dir, "%s.npz" % pid)
        try:
            bag = load_frozen_bag(cfg, pid)
            fp = coords_fingerprint(bag["coords_index"], bag["coords_world"])
            if not args.force and os.path.isfile(path):
                ok, why = validate_existing(path, pid, sig, bag["n_patches"], fp)
                if ok:
                    with np.load(path, allow_pickle=True) as z:
                        npad = np.asarray(z["n_padded_pixels"])
                        rows.append({
                            "PatientID": pid, "status": "skipped",
                            "histology": bag["histology"], "label": bag["label"],
                            "outer_fold": folds[pid],
                            "n_images": bag["n_patches"],
                            "n_frozen_patches": bag["n_patches"],
                            "n_padded_instances": int((npad > 0).sum()),
                            "n_padded_pixels": int(npad.sum()),
                            "hu_min": int(z["images"].min()) if bag["n_patches"] else None,
                            "hu_max": int(z["images"].max()) if bag["n_patches"] else None,
                            "source_coords_sha256": fp,
                            "source_bag_sha256": str(z["source_bag_sha256"]),
                            "bytes": os.path.getsize(path), "seconds": 0.0,
                        })
                    n_skip += 1
                    log(fh, "[%3d/%3d] %-12s SKIP (valid, %d instances)"
                        % (idx, len(pids), pid, bag["n_patches"]))
                    continue
                log(fh, "[%3d/%3d] %-12s re-extract: %s" % (idx, len(pids), pid, why))
            rec = extract_patient_images(cfg, pid, out_dir, folds[pid])
            rec_row = {"PatientID": pid, "status": "extracted"}
            rec_row.update({k: v for k, v in rec.items() if not k.startswith("_")})
            rec_row.pop("patient_id", None)
            rows.append(rec_row)
            n_ok += 1
            log(fh, "[%3d/%3d] %-12s ok  %4d instances  %3d padded  %5.1f s  %.1f MB"
                % (idx, len(pids), pid, rec["n_images"], rec["n_padded_instances"],
                   rec["seconds"], rec["bytes"] / 1e6))
        except Exception as exc:                                  # noqa: BLE001
            n_fail += 1
            rows.append({"PatientID": pid, "status": "failed",
                         "error": str(exc)[:400]})
            log(fh, "[%3d/%3d] %-12s FAILED: %s" % (idx, len(pids), pid, str(exc)[:300]))

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    man_path = ppath(cfg, cfg["outputs"]["image_manifest"])
    os.makedirs(os.path.dirname(man_path), exist_ok=True)
    if args.tag == "all" and args.all and not args.patients:
        df.to_csv(man_path, index=False)
    else:
        df.to_csv(man_path.replace(".csv", "_%s.csv" % args.tag), index=False)

    summary = {
        "tag": args.tag,
        "crop_signature": sig,
        "n_requested": len(pids),
        "n_extracted": n_ok,
        "n_skipped": n_skip,
        "n_failed": n_fail,
        "n_images_total": int(df.get("n_images", pd.Series(dtype=float)).sum())
        if "n_images" in df else 0,
        "n_padded_instances_total": int(df.get("n_padded_instances",
                                               pd.Series(dtype=float)).sum())
        if "n_padded_instances" in df else 0,
        "bytes_total": int(df.get("bytes", pd.Series(dtype=float)).sum())
        if "bytes" in df else 0,
        "seconds": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": os.path.relpath(cfg["_config_path"], cfg["_project_root"]).replace(os.sep, "/"),
    }
    run_path = ppath(cfg, cfg["outputs"]["image_run_json"])
    if not (args.tag == "all" and args.all and not args.patients):
        run_path = run_path.replace(".json", "_%s.json" % args.tag)
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log(fh, "extracted %d, skipped %d, failed %d, %d instances, %.1f s -> %s"
        % (n_ok, n_skip, n_fail, summary["n_images_total"], elapsed, run_path))
    fh.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
