"""Phase 8A - external 2.5D CT crops on the external GTV+Rim patch centres.

Phase 8A does NOT re-implement the Phase-6A crop.  It loads the FROZEN
`config/phase6a.yaml`, redirects ONLY the input root, the frozen-bag directory
and the output directory to the external namespace, and calls the FROZEN
`src/image_patches.py:extract_patient_images`.

`crop_signature` hashes the ROI, the ROI filename, the preprocessing block, the
crop block, the expected source extraction signature and the CT filename - and
none of the paths - so the external run MUST reproduce the frozen Phase-6A crop
signature `49756d388a80414f`, and refuses to start if it does not.

Each external patient carries `outer_fold = -1`: the external cohort has no
LUNG1 fold and takes no part in any LUNG1 split.

No CNN runs here and no classifier exists.

    python src/phase8a_images.py --config config/phase8a_external.yaml [--limit N]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import image_patches as IP                                          # noqa: E402
from phase8a_tcia import load_cfg, ppath                            # noqa: E402

EXTERNAL_FOLD = -1


def derived_phase6a_cfg(cfg8):
    """The frozen Phase-6A config with ONLY paths redirected."""
    p6 = os.path.join(cfg8["_project_root"], "config", "phase6a.yaml")
    cfg6 = IP.load_config(p6)
    scientific = {k: copy.deepcopy(cfg6[k])
                  for k in ("preprocessing", "crop", "transform", "encoder", "experiment")}

    cfg = copy.deepcopy(cfg6)
    cfg["paths"]["vuong_root"] = ppath(cfg8, "masks_dir")
    cfg["source"]["features_dir"] = cfg8["paths"]["patch_features_dir"]
    cfg["outputs"]["image_dir"] = cfg8["paths"]["image_patches_dir"]
    cfg["outputs"]["embedding_dir"] = cfg8["paths"]["cnn_embeddings_dir"]

    for k in ("preprocessing", "crop", "transform", "encoder", "experiment"):
        if cfg[k] != scientific[k]:
            raise RuntimeError("derived config changed the frozen %s block" % k)

    sig = IP.crop_signature(cfg)
    expected = str(cfg8["crop"]["expected_phase6a_crop_signature"])
    if sig != expected:
        raise RuntimeError("external crop signature %s != the frozen Phase-6A "
                           "signature %s" % (sig, expected))
    tsig = IP.transform_signature(cfg)
    texp = str(cfg8["transform"]["expected_phase6a_transform_signature"])
    if tsig != texp:
        raise RuntimeError("external transform signature %s != the frozen Phase-6A "
                           "signature %s" % (tsig, texp))
    return cfg, sig, tsig


def included_patients(cfg8):
    df = pd.read_csv(ppath(cfg8, "cohort_csv"))
    inc = df[df["included"].astype(str).str.lower() == "true"]
    return sorted(inc["PatientID"].astype(str))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    cfg8 = load_cfg(args.config)
    cfg6, sig, tsig = derived_phase6a_cfg(cfg8)
    print("external crop signature      : %s  (== frozen Phase-6A)" % sig)
    print("external transform signature : %s  (== frozen Phase-6A)" % tsig)

    out_dir = ppath(cfg8, "image_patches_dir")
    os.makedirs(out_dir, exist_ok=True)
    pids = included_patients(cfg8)
    if args.limit:
        pids = pids[:int(args.limit)]

    rows, failures = [], []
    done = skipped = 0
    t0 = time.time()
    for i, pid in enumerate(pids, 1):
        path = os.path.join(out_dir, "%s.npz" % pid)
        if not args.force:
            ok, _ = IP.validate_existing(path, pid, sig)
            if ok:
                skipped += 1
                with np.load(path, allow_pickle=True) as z:
                    rows.append({"PatientID": pid, "n_images": int(z["images"].shape[0]),
                                 "n_padded_instances": int((z["n_padded_pixels"] > 0).sum()),
                                 "n_padded_pixels": int(z["n_padded_pixels"].sum()),
                                 "crop_signature": str(z["crop_signature"]),
                                 "source_coords_sha256": str(z["source_coords_sha256"]),
                                 "cached": True})
                continue
        try:
            rec = IP.extract_patient_images(cfg6, pid, out_dir=out_dir,
                                            outer_fold=EXTERNAL_FOLD)
            rows.append({"PatientID": pid, "n_images": rec["n_images"],
                         "n_padded_instances": rec["n_padded_instances"],
                         "n_padded_pixels": rec["n_padded_pixels"],
                         "crop_signature": rec["crop_signature"],
                         "source_coords_sha256": rec["source_coords_sha256"],
                         "hu_min": rec["hu_min"], "hu_max": rec["hu_max"],
                         "bytes": rec["bytes"], "seconds": rec["seconds"],
                         "cached": False})
            done += 1
            print("[%3d/%3d] %s  %d crops  %d padded  %.1fs"
                  % (i, len(pids), pid, rec["n_images"], rec["n_padded_instances"],
                     rec["seconds"]), flush=True)
        except Exception as exc:                                   # noqa: BLE001
            failures.append({"PatientID": pid, "error": str(exc)[:300]})
            print("[%3d/%3d] %s  FAILED: %s" % (i, len(pids), pid, str(exc)[:200]), flush=True)

    df = pd.DataFrame(rows)
    man = os.path.join(ppath(cfg8, "metadata_dir"), "image_patch_manifest.csv")
    df.to_csv(man, index=False)
    print("cropped %d, skipped %d cached, %d failures, %d instances, %.1f min"
          % (done, skipped, len(failures), int(df["n_images"].sum()) if len(df) else 0,
             (time.time() - t0) / 60.0))
    if failures:
        print(json.dumps(failures, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
