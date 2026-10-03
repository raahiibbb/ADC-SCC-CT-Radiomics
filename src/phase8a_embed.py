"""Phase 8A - external frozen-ResNet18 embeddings of the external 2.5D crops.

Phase 8A does NOT re-implement the Phase-6A encoder.  It loads the FROZEN
`config/phase6a.yaml`, redirects ONLY the crop and embedding directories to the
external namespace, and calls the FROZEN `src/cnn_embed.py` builder and
`embed_patient`.  The encoder is torchvision ResNet18 with
`ResNet18_Weights.IMAGENET1K_V1`, `fc -> Identity`, every parameter
`requires_grad=False`, `model.eval()`, producing the 512-D penultimate
embedding.  The run refuses to continue unless the loaded weights reproduce the
frozen Phase-6A state fingerprint.

NO CLASSIFIER IS RUN.  Phase 8A produces the embeddings and stops; it computes
no probability, no metric and no comparison.

Requires the torchvision environment:

    ./.venv-phase6/Scripts/python.exe src/phase8a_embed.py --config config/phase8a_external.yaml
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

import cnn_embed as CE                                              # noqa: E402
import image_patches as IP                                          # noqa: E402


def _load_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def derived_phase6a_cfg(cfg8, project_root):
    cfg6 = IP.load_config(os.path.join(project_root, "config", "phase6a.yaml"))
    scientific = {k: copy.deepcopy(cfg6[k])
                  for k in ("preprocessing", "crop", "transform", "encoder", "experiment")}
    cfg = copy.deepcopy(cfg6)
    cfg["paths"]["vuong_root"] = os.path.join(project_root,
                                              *cfg8["paths"]["masks_dir"].split("/"))
    cfg["source"]["features_dir"] = cfg8["paths"]["patch_features_dir"]
    cfg["outputs"]["image_dir"] = cfg8["paths"]["image_patches_dir"]
    cfg["outputs"]["embedding_dir"] = cfg8["paths"]["cnn_embeddings_dir"]
    for k in ("preprocessing", "crop", "transform", "encoder", "experiment"):
        if cfg[k] != scientific[k]:
            raise RuntimeError("derived config changed the frozen %s block" % k)
    sig = IP.crop_signature(cfg)
    if sig != str(cfg8["crop"]["expected_phase6a_crop_signature"]):
        raise RuntimeError("crop signature %s != frozen Phase-6A" % sig)
    tsig = IP.transform_signature(cfg)
    if tsig != str(cfg8["transform"]["expected_phase6a_transform_signature"]):
        raise RuntimeError("transform signature %s != frozen Phase-6A" % tsig)
    return cfg, sig, tsig


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    cfg8 = _load_yaml(args.config)
    cfg6, sig, tsig = derived_phase6a_cfg(cfg8, project_root)
    print("external crop signature      : %s  (== frozen Phase-6A)" % sig)
    print("external transform signature : %s  (== frozen Phase-6A)" % tsig)

    import torch
    torch.manual_seed(int(cfg6["runtime"]["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = CE.resolve_device(cfg6)
    model, enc_meta = CE.build_encoder(cfg6, device=device, verbose=True)
    expected_fp = str(cfg8["encoder"]["expected_state_fingerprint"])
    if enc_meta["state_fingerprint"] != expected_fp:
        raise RuntimeError("encoder state fingerprint %s != the frozen Phase-6A %s"
                           % (enc_meta["state_fingerprint"], expected_fp))
    probe, probe_hash = CE.probe_vector(cfg6, model=None)
    print("cpu probe hash   : %s" % probe_hash[:32])
    model = model.to(device)

    cohort = pd.read_csv(os.path.join(project_root,
                                      *cfg8["paths"]["cohort_csv"].split("/")))
    pids = sorted(cohort[cohort["included"].astype(str).str.lower() == "true"]
                  ["PatientID"].astype(str))
    if args.limit:
        pids = pids[:int(args.limit)]

    img_dir = os.path.join(project_root, *cfg8["paths"]["image_patches_dir"].split("/"))
    out_dir = os.path.join(project_root, *cfg8["paths"]["cnn_embeddings_dir"].split("/"))
    os.makedirs(out_dir, exist_ok=True)

    rows, failures = [], []
    done = skipped = 0
    bs = int(cfg6["runtime"]["batch_size"])
    t0 = time.time()
    for i, pid in enumerate(pids, 1):
        path = os.path.join(out_dir, "%s.npz" % pid)
        with np.load(os.path.join(img_dir, "%s.npz" % pid), allow_pickle=True) as z:
            n_expected = int(z["images"].shape[0])
            coords_fp = str(z["source_coords_sha256"])
        if not args.force:
            ok, _ = CE.validate_existing_embedding(path, pid, enc_meta, cfg6,
                                                   n_expected=n_expected,
                                                   coords_fp=coords_fp)
            if ok:
                skipped += 1
                with np.load(path, allow_pickle=True) as z:
                    rows.append({"PatientID": pid, "n_embeddings": n_expected,
                                 "embedding_dim": int(z["embedding_dim"]),
                                 "source_coords_sha256": coords_fp, "cached": True})
                continue
        try:
            rec, bs = CE.embed_patient(cfg6, pid, model, enc_meta, device, bs,
                                       out_dir=out_dir)
            rows.append({"PatientID": pid, "n_embeddings": rec["n_embeddings"],
                         "embedding_dim": rec["embedding_dim"],
                         "emb_min": rec["emb_min"], "emb_max": rec["emb_max"],
                         "emb_mean": rec["emb_mean"],
                         "source_coords_sha256": rec["source_coords_sha256"],
                         "device": rec["device"], "batch_size": rec["batch_size"],
                         "seconds": rec["seconds"], "cached": False})
            done += 1
            print("[%3d/%3d] %s  %d x %d  %.1fs"
                  % (i, len(pids), pid, rec["n_embeddings"], rec["embedding_dim"],
                     rec["seconds"]), flush=True)
        except Exception as exc:                                   # noqa: BLE001
            failures.append({"PatientID": pid, "error": str(exc)[:300]})
            print("[%3d/%3d] %s  FAILED: %s" % (i, len(pids), pid, str(exc)[:200]),
                  flush=True)

    df = pd.DataFrame(rows)
    man = os.path.join(project_root, *cfg8["paths"]["metadata_dir"].split("/"),
                       "cnn_embedding_manifest.csv")
    df.to_csv(man, index=False)
    meta = {"encoder": enc_meta, "cpu_probe_sha256": probe_hash,
            "device": str(device), "crop_signature": sig,
            "transform_signature": tsig,
            "n_patients": len(rows),
            "n_embeddings": int(df["n_embeddings"].sum()) if len(df) else 0,
            "n_failures": len(failures), "failures": failures,
            "elapsed_min": round((time.time() - t0) / 60.0, 2)}
    with open(os.path.join(project_root, *cfg8["paths"]["metadata_dir"].split("/"),
                           "cnn_embedding_run.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    print("embedded %d, skipped %d cached, %d failures, %d instances, %.1f min"
          % (done, skipped, len(failures), meta["n_embeddings"], meta["elapsed_min"]))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
