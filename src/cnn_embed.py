"""Phase 6A - frozen ImageNet ResNet18 embeddings for the 2.5D CT crops.

    ./.venv-phase6/Scripts/python.exe src/cnn_embed.py --all --tag all

The encoder is pretrained externally, completely frozen, and never sees an
ADC/SCC label, so precomputing embeddings for all 202 patients before
cross-validation cannot leak anything.  Nothing here trains, fine-tunes, or
builds a patch-level classifier.

Guarantees enforced in code, not merely documented:

  * the exact torchvision weights enum named in the config is used;
  * a random initialisation is NEVER silently substituted - if the pretrained
    weights cannot be obtained the run FAILS;
  * every parameter has ``requires_grad = False`` and the module is in ``eval()``;
  * the classification head is replaced by ``Identity``, leaving the standard
    512-D penultimate embedding;
  * embedding row i corresponds to image row i corresponds to frozen radiomics
    bag row i - carried by ``patch_index`` and by the coordinate fingerprint.

This module imports only torch / torchvision / numpy / pandas / yaml.  It never
imports radiomics, SimpleITK, nibabel or pydicom, and never reads a CT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from image_patches import (crop_signature, load_config, ppath,  # noqa: E402
                           sha256_file, transform_signature)

DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), "config", "phase6a.yaml")
NPZ_VERSION = 1

EMB_REQUIRED_KEYS = [
    "embeddings", "patch_index", "coords_index", "coords_world", "patient_id",
    "histology", "label", "outer_fold", "encoder_architecture",
    "encoder_weights_enum", "encoder_weights_url", "encoder_weights_sha256",
    "encoder_state_fingerprint", "embedding_dim", "transform_id",
    "transform_signature", "image_crop_signature", "source_image_sha256",
    "source_coords_sha256", "source_extraction_signature", "torch_version",
    "torchvision_version", "device", "npz_version",
]


# --------------------------------------------------------------------- device
def describe_environment():
    import torch
    info = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": None,
        "gpu_total_memory_gb": None,
        "gpu_capability": None,
    }
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:                                             # noqa: BLE001
        info["torchvision"] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu_name"] = p.name
        info["gpu_total_memory_gb"] = round(p.total_memory / 1e9, 3)
        info["gpu_capability"] = "%d.%d" % (p.major, p.minor)
    return info


def print_environment(info=None):
    info = info or describe_environment()
    print("torch            : %s" % info["torch"])
    print("torchvision      : %s" % info["torchvision"])
    print("CUDA available   : %s (torch CUDA %s)"
          % (info["cuda_available"], info["torch_cuda_version"]))
    print("GPU              : %s" % (info["gpu_name"] or "none visible to torch"))
    print("VRAM             : %s GB" % (info["gpu_total_memory_gb"]
                                        if info["gpu_total_memory_gb"] else "n/a"))
    print("compute capability: %s" % (info["gpu_capability"] or "n/a"))
    return info


def resolve_device(cfg):
    import torch
    want = str(cfg["runtime"]["device"]).lower()
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if want == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config asks for CUDA but torch reports no CUDA device")
    return torch.device(want)


# -------------------------------------------------------------------- encoder
def _state_fingerprint(state) -> str:
    h = hashlib.sha256()
    for k in sorted(state):
        v = state[k]
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(v.detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def build_encoder(cfg, device=None, verbose=True):
    """A frozen, pretrained torchvision ResNet18 with the fc head removed.

    Raises rather than falling back to random initialisation.
    """
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import models as tvm

    ecfg = cfg["encoder"]
    if str(ecfg["framework"]) != "torchvision":
        raise RuntimeError("only the torchvision encoder path is implemented")
    if str(ecfg["architecture"]) != "resnet18":
        raise RuntimeError("config architecture is %s, expected resnet18"
                           % ecfg["architecture"])

    enum_name = str(ecfg["weights_enum"])                # "ResNet18_Weights.IMAGENET1K_V1"
    cls_name, member = enum_name.split(".")
    weights = getattr(getattr(tvm, cls_name), member)
    if weights.url != str(ecfg["weights_url"]):
        raise RuntimeError("torchvision weights URL %s does not match the configured %s"
                           % (weights.url, ecfg["weights_url"]))
    acc1 = float(weights.meta["_metrics"]["ImageNet-1K"]["acc@1"])
    if abs(acc1 - float(ecfg["expected_acc1"])) > 1e-6:
        raise RuntimeError("weights acc@1 %.3f != configured %.3f"
                           % (acc1, ecfg["expected_acc1"]))

    local = ecfg.get("local_weights_path")
    state = None
    weights_path = None
    if local:
        local = local if os.path.isabs(local) else ppath(cfg, local)
        if not os.path.isfile(local):
            raise RuntimeError("configured local_weights_path does not exist: %s" % local)
        state = torch.load(local, map_location="cpu", weights_only=True)
        weights_path = local
        model = tvm.resnet18(weights=None)
        missing, unexpected = model.load_state_dict(state, strict=True), None
    else:
        try:
            model = tvm.resnet18(weights=weights)
        except Exception as exc:                                  # noqa: BLE001
            raise RuntimeError(
                "pretrained %s could not be obtained (%s). Phase 6A refuses to "
                "continue with randomly initialised weights; supply the checkpoint "
                "via encoder.local_weights_path instead." % (enum_name, str(exc)[:200]))
        weights_path = os.path.join(torch.hub.get_dir(), "checkpoints",
                                    os.path.basename(weights.url))

    if bool(ecfg.get("allow_random_init", False)):
        raise RuntimeError("encoder.allow_random_init must stay false in Phase 6A")

    # PROOF that the weights are not random: an independently seeded random
    # initialisation must differ, and the checkpoint fingerprint must be stable.
    trained_fp = _state_fingerprint(model.state_dict())
    torch.manual_seed(0)
    rnd_fp = _state_fingerprint(tvm.resnet18(weights=None).state_dict())
    if trained_fp == rnd_fp:
        raise RuntimeError("encoder weights are indistinguishable from a random init")

    # remove the classification head, keep the standard penultimate embedding
    in_features = int(model.fc.in_features)
    if in_features != int(ecfg["embedding_dim"]):
        raise RuntimeError("resnet18 penultimate width is %d, config says %d"
                           % (in_features, ecfg["embedding_dim"]))
    model.fc = nn.Identity()

    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    n_grad = sum(1 for p in model.parameters() if p.requires_grad)
    if n_grad:
        raise RuntimeError("%d parameters still require grad" % n_grad)

    if device is not None:
        model = model.to(device)

    meta = {
        "architecture": "resnet18",
        "weights_enum": enum_name,
        "weights_url": weights.url,
        "weights_file": os.path.basename(weights.url),
        "weights_path": weights_path,
        "weights_sha256": (sha256_file(weights_path)
                           if weights_path and os.path.isfile(weights_path) else ""),
        "imagenet_acc1": acc1,
        "num_classes_meta": int(weights.meta.get("num_params", 0)),
        "state_fingerprint": trained_fp,
        "random_init_fingerprint": rnd_fp,
        "embedding_dim": in_features,
        "n_parameters": int(n_params),
        "n_parameters_requiring_grad": 0,
        "training_mode": bool(model.training),
        "head": "fc -> Identity",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    if verbose:
        print("encoder          : torchvision resnet18, %s" % enum_name)
        print("weights file     : %s" % meta["weights_file"])
        print("weights sha256   : %s" % (meta["weights_sha256"][:32] or "n/a"))
        print("ImageNet acc@1   : %.3f" % acc1)
        print("state fingerprint: %s" % trained_fp[:32])
        print("parameters       : %d, requiring grad: 0, training=%s"
              % (n_params, model.training))
        print("embedding dim    : %d" % in_features)
    return model, meta


# ------------------------------------------------------------------ transform
def input_transform(images_int16, cfg, device=None):
    """[B,3,32,32] int16 HU -> [B,3,224,224] float32, ImageNet-normalised."""
    import torch
    import torch.nn.functional as F

    tcfg = cfg["transform"]
    lo, hi = (float(v) for v in tcfg["hu_to_unit_range"])
    x = torch.as_tensor(np.ascontiguousarray(images_int16), dtype=torch.float32)
    if device is not None:
        x = x.to(device, non_blocking=True)
    x = (x - lo) / (hi - lo)                       # deterministic map to [0, 1]
    x = F.interpolate(x, size=tuple(int(v) for v in tcfg["resize_to"]),
                      mode=str(tcfg["resize_mode"]),
                      align_corners=bool(tcfg["align_corners"]),
                      antialias=bool(tcfg["antialias"]))
    mean = torch.as_tensor(tcfg["normalize_mean"], dtype=torch.float32,
                           device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(tcfg["normalize_std"], dtype=torch.float32,
                          device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def probe_vector(cfg, model=None):
    """Deterministic CPU probe: proves which weights + transform produced a run."""
    import torch
    if model is None:
        model, _ = build_encoder(cfg, device=None, verbose=False)
    else:
        model = model.to("cpu")
    lo, hi = (int(v) for v in cfg["crop"]["hu_clip"])
    ramp = np.linspace(lo, hi, 3 * 32 * 32).reshape(1, 3, 32, 32)
    x = np.rint(ramp).astype(np.int16)
    with torch.no_grad():
        v = model(input_transform(x, cfg, device=torch.device("cpu")))
    v = v.detach().cpu().numpy().astype(np.float32).reshape(-1)
    return v, hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()


# ------------------------------------------------------------------ embedding
def embed_array(model, images, cfg, device, batch_size):
    """Frozen forward pass over [N,3,32,32] int16 -> [N,512] float32."""
    import torch
    n = int(images.shape[0])
    out = np.empty((n, int(cfg["encoder"]["embedding_dim"])), dtype=np.float32)
    bs = int(batch_size)
    lo_bs = int(cfg["runtime"]["min_batch_size"])
    i = 0
    with torch.no_grad():
        while i < n:
            try:
                chunk = images[i:i + bs]
                x = input_transform(chunk, cfg, device)
                y = model(x)
                out[i:i + chunk.shape[0]] = y.detach().cpu().numpy().astype(np.float32)
                i += chunk.shape[0]
            except torch.cuda.OutOfMemoryError:                   # noqa: PERF203
                torch.cuda.empty_cache()
                if bs <= lo_bs:
                    raise
                bs = max(lo_bs, bs // 2)
                print("  CUDA OOM -> batch_size %d" % bs, flush=True)
    return out, bs


def embed_patient(cfg, pid, model, enc_meta, device, batch_size, out_dir=None):
    import torch
    t0 = time.time()
    out_dir = out_dir or ppath(cfg, cfg["outputs"]["embedding_dir"])
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(ppath(cfg, cfg["outputs"]["image_dir"]), "%s.npz" % pid)

    with np.load(img_path, allow_pickle=True) as z:
        images = np.asarray(z["images"])
        rec = {k: z[k] for k in ("patch_index", "coords_index", "coords_world",
                                 "coords_start_index", "n_padded_pixels", "padded")}
        pid_in = str(z["patient_id"])
        histology = str(z["histology"])
        label = int(z["label"])
        fold = int(z["outer_fold"])
        img_crop_sig = str(z["crop_signature"])
        src_sig = str(z["source_extraction_signature"])
        coords_fp = str(z["source_coords_sha256"])
    if pid_in != pid:
        raise RuntimeError("%s: image file says %s" % (pid, pid_in))
    if images.dtype != np.int16 or images.shape[1:] != (3, 32, 32):
        raise RuntimeError("%s: unexpected image array %s %s"
                           % (pid, images.shape, images.dtype))
    if img_crop_sig != crop_signature(cfg):
        raise RuntimeError("%s: image crop signature %s != current %s"
                           % (pid, img_crop_sig, crop_signature(cfg)))

    emb, used_bs = embed_array(model, images, cfg, device, batch_size)
    if not np.all(np.isfinite(emb)):
        raise RuntimeError("%s: non-finite embedding produced" % pid)

    npz_path = os.path.join(out_dir, "%s.npz" % pid)
    np.savez_compressed(
        npz_path,
        embeddings=emb,
        patch_index=rec["patch_index"],
        coords_index=rec["coords_index"],
        coords_world=rec["coords_world"],
        coords_start_index=rec["coords_start_index"],
        n_padded_pixels=rec["n_padded_pixels"],
        padded=rec["padded"],
        patient_id=np.array(pid),
        histology=np.array(histology),
        label=np.array(label, dtype=np.int8),
        outer_fold=np.array(fold, dtype=np.int8),
        encoder_architecture=np.array(enc_meta["architecture"]),
        encoder_weights_enum=np.array(enc_meta["weights_enum"]),
        encoder_weights_url=np.array(enc_meta["weights_url"]),
        encoder_weights_file=np.array(enc_meta["weights_file"]),
        encoder_weights_sha256=np.array(enc_meta["weights_sha256"]),
        encoder_state_fingerprint=np.array(enc_meta["state_fingerprint"]),
        encoder_imagenet_acc1=np.array(enc_meta["imagenet_acc1"], dtype=np.float64),
        encoder_frozen=np.array(True),
        encoder_head=np.array(enc_meta["head"]),
        embedding_dim=np.array(int(emb.shape[1]), dtype=np.int32),
        transform_id=np.array(str(cfg["transform"]["id"])),
        transform_signature=np.array(transform_signature(cfg)),
        image_crop_signature=np.array(img_crop_sig),
        source_image_sha256=np.array(sha256_file(img_path)),
        source_coords_sha256=np.array(coords_fp),
        source_extraction_signature=np.array(src_sig),
        torch_version=np.array(enc_meta["torch"]),
        torchvision_version=np.array(enc_meta["torchvision"]),
        numpy_version=np.array(np.__version__),
        device=np.array(str(device)),
        batch_size=np.array(int(used_bs), dtype=np.int32),
        npz_version=np.array(NPZ_VERSION, dtype=np.int32),
    )
    return {
        "patient_id": pid, "histology": histology, "label": label,
        "outer_fold": fold, "n_embeddings": int(emb.shape[0]),
        "embedding_dim": int(emb.shape[1]),
        "n_nonfinite": 0,
        "emb_min": float(emb.min()) if emb.size else None,
        "emb_max": float(emb.max()) if emb.size else None,
        "emb_mean": float(emb.mean()) if emb.size else None,
        "source_image_sha256": sha256_file(img_path),
        "source_coords_sha256": coords_fp,
        "device": str(device), "batch_size": int(used_bs),
        "bytes": os.path.getsize(npz_path),
        "seconds": round(time.time() - t0, 3),
        "_npz_path": npz_path,
    }, used_bs


def validate_existing_embedding(path, pid, enc_meta, cfg, n_expected=None,
                                coords_fp: Optional[str] = None):
    try:
        with np.load(path, allow_pickle=True) as z:
            for k in EMB_REQUIRED_KEYS:
                if k not in z:
                    return False, "missing key %s" % k
            if str(z["patient_id"]) != pid:
                return False, "patient_id mismatch"
            if str(z["encoder_state_fingerprint"]) != enc_meta["state_fingerprint"]:
                return False, "encoder weights changed"
            if str(z["transform_signature"]) != transform_signature(cfg):
                return False, "input transform changed"
            if str(z["image_crop_signature"]) != crop_signature(cfg):
                return False, "image crop signature changed"
            e = z["embeddings"]
            if e.dtype != np.float32 or e.ndim != 2:
                return False, "embeddings %s %s" % (e.shape, e.dtype)
            if e.shape[1] != int(cfg["encoder"]["embedding_dim"]):
                return False, "embedding dim %d" % e.shape[1]
            n = int(e.shape[0])
            if n_expected is not None and n != n_expected:
                return False, "n %d != expected %d" % (n, n_expected)
            if coords_fp is not None and str(z["source_coords_sha256"]) != coords_fp:
                return False, "coords fingerprint mismatch"
            if not np.array_equal(z["patch_index"], np.arange(n, dtype=np.int32)):
                return False, "patch_index is not 0..N-1"
            if n and not np.all(np.isfinite(e)):
                return False, "non-finite embeddings"
    except Exception as exc:                                      # noqa: BLE001
        return False, "unreadable: %s" % str(exc)[:200]
    return True, "ok"


# ---------------------------------------------------------------------- driver
def main(argv=None) -> int:
    import pandas as pd
    import torch

    ap = argparse.ArgumentParser(description="Phase 6A frozen ResNet18 embeddings")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--tag", default="all")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--env-only", action="store_true",
                    help="print the torch/CUDA/GPU report and exit")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    env = print_environment()
    if args.env_only:
        return 0
    if args.device:
        cfg["runtime"]["device"] = args.device
    device = resolve_device(cfg)
    batch_size = int(args.batch_size or cfg["runtime"]["batch_size"])
    torch.manual_seed(int(cfg["runtime"]["seed"]))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("device           : %s   batch_size %d" % (device, batch_size))

    model, enc_meta = build_encoder(cfg, device=device)
    pv, probe_hash = probe_vector(cfg)
    model = model.to(device)
    print("cpu probe sha256 : %s" % probe_hash[:32])

    img_dir = ppath(cfg, cfg["outputs"]["image_dir"])
    out_dir = ppath(cfg, cfg["outputs"]["embedding_dir"])
    os.makedirs(out_dir, exist_ok=True)
    all_pids = sorted(f[:-4] for f in os.listdir(img_dir) if f.endswith(".npz"))
    if args.patients:
        pids = list(args.patients)
    elif args.pilot:
        pids = [str(p) for p in cfg["pilot"]["patients"]]
    elif args.all:
        pids = all_pids
    else:
        ap.error("choose one of --all / --pilot / --patients")

    log_path = ppath(cfg, cfg["outputs"]["embedding_log"])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    fh = open(log_path, "a", encoding="utf-8")
    fh.write("=" * 72 + "\nPhase 6A embeddings tag=%s device=%s n=%d %s\n"
             % (args.tag, device, len(pids), time.strftime("%Y-%m-%d %H:%M:%S")))

    rows, n_ok, n_skip, n_fail = [], 0, 0, 0
    t0 = time.time()
    for idx, pid in enumerate(pids, 1):
        path = os.path.join(out_dir, "%s.npz" % pid)
        try:
            with np.load(os.path.join(img_dir, "%s.npz" % pid), allow_pickle=True) as z:
                n_img = int(z["images"].shape[0])
                cfp = str(z["source_coords_sha256"])
            if not args.force and os.path.isfile(path):
                ok, why = validate_existing_embedding(path, pid, enc_meta, cfg, n_img, cfp)
                if ok:
                    with np.load(path, allow_pickle=True) as z:
                        rows.append({"PatientID": pid, "status": "skipped",
                                     "histology": str(z["histology"]),
                                     "label": int(z["label"]),
                                     "outer_fold": int(z["outer_fold"]),
                                     "n_embeddings": int(z["embeddings"].shape[0]),
                                     "embedding_dim": int(z["embeddings"].shape[1]),
                                     "n_nonfinite": 0,
                                     "source_coords_sha256": cfp,
                                     "device": str(z["device"]),
                                     "batch_size": int(z["batch_size"]),
                                     "bytes": os.path.getsize(path), "seconds": 0.0})
                    n_skip += 1
                    msg = "[%3d/%3d] %-12s SKIP (valid, %d embeddings)" % (idx, len(pids), pid, n_img)
                    print(msg, flush=True); fh.write(msg + "\n")
                    continue
                msg = "[%3d/%3d] %-12s recompute: %s" % (idx, len(pids), pid, why)
                print(msg, flush=True); fh.write(msg + "\n")
            rec, batch_size = embed_patient(cfg, pid, model, enc_meta, device,
                                            batch_size, out_dir)
            row = {"PatientID": pid, "status": "computed"}
            row.update({k: v for k, v in rec.items() if not k.startswith("_")})
            row.pop("patient_id", None)
            rows.append(row)
            n_ok += 1
            msg = ("[%3d/%3d] %-12s ok  %4d x %d  %5.2f s"
                   % (idx, len(pids), pid, rec["n_embeddings"], rec["embedding_dim"],
                      rec["seconds"]))
            print(msg, flush=True); fh.write(msg + "\n")
        except Exception as exc:                                  # noqa: BLE001
            n_fail += 1
            rows.append({"PatientID": pid, "status": "failed", "error": str(exc)[:400]})
            msg = "[%3d/%3d] %-12s FAILED: %s" % (idx, len(pids), pid, str(exc)[:300])
            print(msg, flush=True); fh.write(msg + "\n")

    elapsed = time.time() - t0
    df = pd.DataFrame(rows)
    man = ppath(cfg, cfg["outputs"]["embedding_manifest"])
    run = ppath(cfg, cfg["outputs"]["embedding_run_json"])
    full_run = args.all and not args.patients
    if not full_run:
        man = man.replace(".csv", "_%s.csv" % args.tag)
        run = run.replace(".json", "_%s.json" % args.tag)
    os.makedirs(os.path.dirname(man), exist_ok=True)
    df.to_csv(man, index=False)

    summary = {
        "tag": args.tag,
        "environment": env,
        "device": str(device),
        "batch_size_final": int(batch_size),
        "batch_size_requested": int(args.batch_size or cfg["runtime"]["batch_size"]),
        "encoder": {k: v for k, v in enc_meta.items() if k != "weights_path"},
        "encoder_weights_path": enc_meta["weights_path"],
        "cpu_probe_sha256": probe_hash,
        "cpu_probe_head": [float(v) for v in pv[:8]],
        "transform": cfg["transform"],
        "transform_signature": transform_signature(cfg),
        "image_crop_signature": crop_signature(cfg),
        "n_requested": len(pids), "n_computed": n_ok, "n_skipped": n_skip,
        "n_failed": n_fail,
        "n_embeddings_total": int(df.get("n_embeddings", __import__("pandas").Series(dtype=float)).sum())
        if "n_embeddings" in df else 0,
        "bytes_total": int(df.get("bytes", __import__("pandas").Series(dtype=float)).sum())
        if "bytes" in df else 0,
        "seconds": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(run, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    msg = ("computed %d, skipped %d, failed %d, %d embeddings, %.1f s -> %s"
           % (n_ok, n_skip, n_fail, summary["n_embeddings_total"], elapsed, run))
    print(msg); fh.write(msg + "\n"); fh.close()
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
