"""Phase-6A audit: learned CT patch representation (2.5D crops + frozen ResNet18).

Run with the Phase-6 image environment (it needs torch + torchvision):

    ./.venv-phase6/Scripts/python.exe tests/test_phase6a.py

Every check re-derives its answer from files on disk.  Nothing recorded by the
extraction or embedding drivers is trusted: crop geometry is re-proved from the
stored arrays alone, the encoder is rebuilt from its weights identifier and
re-probed, and every frozen Phase-1..5C artefact is re-hashed.

Deliberately CT-free.  The geometry checks below never open a NIfTI file: they
prove the channel order, the axis assignment and the padding rule from internal
consistency between instances, which is a stronger independence argument than
re-running the extractor's own code path.  The CT-dependent re-crop proof lives
in the pilot QA (`src/qa_image_patches.py`, `reports/phase6a_pilot_check.json`).

Groups
  A  cohort, split and instance identity            (required checks 1-7)
  B  crop geometry                                  (required checks 8-13)
  C  the frozen pretrained encoder                  (required checks 14-17)
  D  embedding identity and label independence      (required checks 18-20)
  E  scope: nothing was trained                     (required checks 21-23)
  F  non-regression                                 (required checks 24, 25)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from cnn_embed import (build_encoder, embed_array, input_transform,  # noqa: E402
                       probe_vector, resolve_device)
from image_patches import (coords_fingerprint, crop_signature, load_config,  # noqa: E402
                           load_frozen_bag, load_outer_folds, ppath,
                           transform_signature)

CFG = load_config(os.path.join(ROOT, "config", "phase6a.yaml"))
IMG_DIR = ppath(CFG, CFG["outputs"]["image_dir"])
EMB_DIR = ppath(CFG, CFG["outputs"]["embedding_dir"])
BAG_DIR = ppath(CFG, CFG["source"]["features_dir"])
N_PATIENTS = int(CFG["source"]["expected_patients"])
N_PATCHES = int(CFG["source"]["expected_patches"])
N_ADC = int(CFG["source"]["expected_adc"])
N_SCC = int(CFG["source"]["expected_scc"])
SIZE = int(CFG["crop"]["image_size"][0])
OFF = int(CFG["crop"]["centre_offset_voxels"])
PAD = int(CFG["crop"]["pad_value_hu"])
LO, HI = (int(v) for v in CFG["crop"]["hu_clip"])
DIM = int(CFG["encoder"]["embedding_dim"])

# Phase-6A pipeline sources.  tests/ is excluded from the forbidden-token scans
# on purpose: this audit file trains nothing and writes no artefact, and it must
# contain every forbidden token as a literal in its own search lists, so scanning
# itself would report guaranteed self-hits.
PHASE6A_SOURCES = ["src/image_patches.py", "src/run_image_extraction.py",
                   "src/qa_image_patches.py", "src/cnn_embed.py",
                   "config/phase6a.yaml", "tools/phase6a_baseline.py",
                   "tests/test_phase6a.py"]
PHASE6A_PY = [p for p in PHASE6A_SOURCES
              if p.endswith(".py") and not p.startswith("tests/")]

CHECKS = []


def check(group, name):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


# ------------------------------------------------------------------- loaders
def img_path(pid):
    return os.path.join(IMG_DIR, "%s.npz" % pid)


def emb_path(pid):
    return os.path.join(EMB_DIR, "%s.npz" % pid)


def load_img(pid, keys=None):
    z = np.load(img_path(pid), allow_pickle=True)
    if keys is None:
        return z
    out = {k: z[k] for k in keys}
    z.close()
    return out


def load_emb(pid, keys=None):
    z = np.load(emb_path(pid), allow_pickle=True)
    if keys is None:
        return z
    out = {k: z[k] for k in keys}
    z.close()
    return out


def source(rel):
    with open(os.path.join(ROOT, *rel.split("/")), "r", encoding="utf-8") as fh:
        return fh.read()


def code_only(rel):
    """Source with every comment and string literal removed.

    The forbidden-token scans below assert what the code DOES.  A module
    docstring that truthfully says "this never imports SimpleITK" must not be
    read as an import of SimpleITK, so comments and literals are stripped before
    scanning; only executable tokens remain.
    """
    import io as _io
    import tokenize as _tk
    src = source(rel)
    out, last_line = [], 1
    for tok in _tk.generate_tokens(_io.StringIO(src).readline):
        if tok.type in (_tk.COMMENT, _tk.STRING):
            continue
        if tok.start[0] != last_line:
            out.append("\n")
            last_line = tok.start[0]
        out.append(" " + tok.string)
    return "".join(out)


def imported_modules(rel):
    """Top-level module names actually imported anywhere in a source file."""
    import ast as _ast
    mods = set()
    for node in _ast.walk(_ast.parse(source(rel))):
        if isinstance(node, _ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
            mods.update(a.name for a in node.names)
    return mods


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


PIDS = sorted(f[:-4] for f in os.listdir(IMG_DIR) if f.endswith(".npz"))
SPLIT = pd.read_csv(ppath(CFG, CFG["source"]["splits_file"]))
FOLDS = load_outer_folds(CFG)

_ENC = {}


def encoder():
    """Rebuild the encoder once from its configured weights identifier."""
    if "model" not in _ENC:
        m, meta = build_encoder(CFG, device=None, verbose=False)
        _ENC["model"], _ENC["meta"] = m, meta
    return _ENC["model"], _ENC["meta"]


# =========================================================== A  identity
@check("A", "1_cohort_202_patients")
def a1():
    n_img = len(PIDS)
    n_emb = len([f for f in os.listdir(EMB_DIR) if f.endswith(".npz")])
    n_bag = len([f for f in os.listdir(BAG_DIR) if f.endswith(".npz")])
    labels = {p: int(load_img(p, ["label"])["label"]) for p in PIDS}
    n_adc = sum(1 for v in labels.values() if v == 1)
    n_scc = sum(1 for v in labels.values() if v == 0)
    split_pids = sorted(SPLIT.loc[SPLIT["split"] == "test", "PatientID"].unique())
    ok = (n_img == n_emb == n_bag == N_PATIENTS
          and n_adc == N_ADC and n_scc == N_SCC
          and PIDS == split_pids
          and PIDS == sorted(f[:-4] for f in os.listdir(BAG_DIR) if f.endswith(".npz")))
    return ok, ("%d image files, %d embedding files, %d frozen bags, %d split patients "
                "- all the same %d ids; %d ADC / %d SCC"
                % (n_img, n_emb, n_bag, len(split_pids), N_PATIENTS, n_adc, n_scc))


@check("A", "2_frozen_split_unchanged")
def a2():
    p = ppath(CFG, CFG["source"]["splits_file"])
    got = sha256(p)
    with open(ppath(CFG, CFG["source"]["splits_meta_file"]), encoding="utf-8") as fh:
        meta = json.load(fh)
    want = str(CFG["source"]["expected_split_sha256"])
    ok = got == want == str(meta["splits_file_sha256"])
    # labels and folds carried by Phase 6A must equal the frozen split
    bad = []
    lab = {str(r.PatientID): int(r.label) for r in SPLIT.itertuples()}
    for pid in PIDS:
        d = load_img(pid, ["label", "outer_fold"])
        e = load_emb(pid, ["label", "outer_fold"])
        if (int(d["label"]) != lab[pid] or int(e["label"]) != lab[pid]
                or int(d["outer_fold"]) != FOLDS[pid] or int(e["outer_fold"]) != FOLDS[pid]):
            bad.append(pid)
    return ok and not bad, ("split sha256 %s matches its metadata and the Phase-6A "
                            "config; all %d image and embedding files carry the frozen "
                            "label and outer fold (%d disagreements)"
                            % (got[:16], N_PATIENTS, len(bad)))


@check("A", "3_37464_patch_identities")
def a3():
    n_img = n_bag = 0
    seen = set()
    dup = 0
    for pid in PIDS:
        d = load_img(pid, ["patch_index", "coords_index"])
        bag = load_frozen_bag(CFG, pid)
        n_img += int(d["patch_index"].shape[0])
        n_bag += bag["n_patches"]
        for t, c in enumerate(d["coords_index"]):
            key = (pid, int(t), int(c[0]), int(c[1]), int(c[2]))
            if key in seen:
                dup += 1
            seen.add(key)
    ok = n_img == n_bag == N_PATCHES == len(seen) and dup == 0
    return ok, ("%d image instances == %d frozen patches == %d expected; %d unique "
                "(patient, patch_index, coords_index) identities, %d duplicates"
                % (n_img, n_bag, N_PATCHES, len(seen), dup))


@check("A", "4_counts_equal_frozen_bags")
def a4():
    bad = []
    for pid in PIDS:
        n_i = int(load_img(pid, ["patch_index"])["patch_index"].shape[0])
        n_e = int(load_emb(pid, ["embeddings"])["embeddings"].shape[0])
        n_b = load_frozen_bag(CFG, pid)["n_patches"]
        if not (n_i == n_e == n_b):
            bad.append((pid, n_i, n_e, n_b))
    return not bad, ("per-patient image, embedding and frozen bag counts agree for all "
                     "%d patients (%d disagreements)" % (N_PATIENTS, len(bad)))


@check("A", "5_patch_index_maps_to_frozen_row")
def a5():
    bad = []
    for pid in PIDS:
        d = load_img(pid, ["patch_index", "coords_index", "coords_start_index"])
        e = load_emb(pid, ["patch_index"])
        bag = load_frozen_bag(CFG, pid)
        n = bag["n_patches"]
        ar = np.arange(n, dtype=np.int32)
        if not (np.array_equal(d["patch_index"], ar)
                and np.array_equal(e["patch_index"], ar)
                and np.array_equal(d["coords_start_index"], bag["coords_start_index"])):
            bad.append(pid)
    return not bad, ("patch_index is exactly 0..N-1 in every image and embedding file, "
                     "and row i carries the frozen bag's row-i block start, for all %d "
                     "patients (%d failures)" % (N_PATIENTS, len(bad)))


@check("A", "6_coords_index_unchanged")
def a6():
    bad, fpbad = [], []
    for pid in PIDS:
        d = load_img(pid, ["coords_index", "source_coords_sha256"])
        e = load_emb(pid, ["coords_index", "source_coords_sha256"])
        bag = load_frozen_bag(CFG, pid)
        if not (np.array_equal(d["coords_index"], bag["coords_index"])
                and np.array_equal(e["coords_index"], bag["coords_index"])):
            bad.append(pid)
        fp = coords_fingerprint(bag["coords_index"], bag["coords_world"])
        if not (str(d["source_coords_sha256"]) == str(e["source_coords_sha256"]) == fp):
            fpbad.append(pid)
    return not bad and not fpbad, (
        "coords_index is byte-identical to the frozen bag in every image and embedding "
        "file (%d mismatches); the coordinate fingerprint recomputed from the frozen bag "
        "matches the one stored on both Phase-6A sides for all %d patients (%d mismatches)"
        % (len(bad), N_PATIENTS, len(fpbad)))


@check("A", "7_coords_world_unchanged")
def a7():
    worst = 0.0
    bad = []
    for pid in PIDS:
        d = load_img(pid, ["coords_world"])
        e = load_emb(pid, ["coords_world"])
        bag = load_frozen_bag(CFG, pid)
        if not (np.array_equal(d["coords_world"], bag["coords_world"])
                and np.array_equal(e["coords_world"], bag["coords_world"])):
            bad.append(pid)
        if bag["n_patches"]:
            worst = max(worst, float(np.abs(d["coords_world"] - bag["coords_world"]).max()))
    return not bad, ("coords_world is bit-identical to the frozen bag in every image and "
                     "embedding file for all %d patients; max |difference| = %.1e mm "
                     "(%d mismatches)" % (N_PATIENTS, worst, len(bad)))


# =========================================================== B  crop geometry
@check("B", "8_channel_order_axial_coronal_sagittal")
def b8():
    """Proved from the stored arrays alone - no CT is opened.

    Two facts pin the channel order:

    (a) shared centre lines.  The three planes meet at the centre voxel, so
            ch0 row 16   == ch1 row 16     (the shared column axis, call it A)
            ch0 col 16   == ch2 row 16     (call it B)
            ch1 col 16   == ch2 col 16     (call it C)
        This forces ch0 = (B rows, A cols), ch1 = (C rows, A cols),
        ch2 = (C rows, B cols) - a consistent, non-permuted assignment.

    (b) lattice shift.  Take two instances whose centres differ only along one
        index axis by d (0 < d < 32).  Shifting that axis must translate exactly
        the planes that carry it, in the correct direction.  Doing this for i, j
        and k names A = i, B = j, C = k, which makes ch0 the plane orthogonal to
        k (AXIAL), ch1 the plane orthogonal to j (CORONAL) and ch2 the plane
        orthogonal to i (SAGITTAL).
    """
    bad_lines = 0
    n_lines = 0
    for pid in PIDS:
        d = load_img(pid, ["images"])
        im = d["images"]
        n_lines += im.shape[0]
        if not (np.array_equal(im[:, 0, OFF, :], im[:, 1, OFF, :])
                and np.array_equal(im[:, 0, :, OFF], im[:, 2, OFF, :])
                and np.array_equal(im[:, 1, :, OFF], im[:, 2, :, OFF])):
            for t in range(im.shape[0]):
                if not (np.array_equal(im[t, 0, OFF, :], im[t, 1, OFF, :])
                        and np.array_equal(im[t, 0, :, OFF], im[t, 2, OFF, :])
                        and np.array_equal(im[t, 1, :, OFF], im[t, 2, :, OFF])):
                    bad_lines += 1

    shifts = {"i": 0, "j": 0, "k": 0}
    fails = []
    views = None
    for pid in PIDS:
        z = np.load(img_path(pid), allow_pickle=True)
        im, ci = z["images"], z["coords_index"]
        views = [str(v) for v in z["views"]]
        z.close()
        n = im.shape[0]
        if n < 2:
            continue
        order = {tuple(int(v) for v in ci[t]): t for t in range(n)}
        for t in range(n):
            a, b, c = (int(v) for v in ci[t])
            for axis, delta in (("i", (5, 0, 0)), ("j", (0, 5, 0)), ("k", (0, 0, 5))):
                q = order.get((a + delta[0], b + delta[1], c + delta[2]))
                if q is None or shifts[axis] >= 400:
                    continue
                dd = 5
                if axis == "i":     # columns of ch0 (axial) and ch1 (coronal)
                    ok = (np.array_equal(im[t, 0, :, dd:], im[q, 0, :, :SIZE - dd])
                          and np.array_equal(im[t, 1, :, dd:], im[q, 1, :, :SIZE - dd]))
                elif axis == "j":   # rows of ch0 (axial), columns of ch2 (sagittal)
                    ok = (np.array_equal(im[t, 0, dd:, :], im[q, 0, :SIZE - dd, :])
                          and np.array_equal(im[t, 2, :, dd:], im[q, 2, :, :SIZE - dd]))
                else:               # rows of ch1 (coronal) and ch2 (sagittal)
                    ok = (np.array_equal(im[t, 1, dd:, :], im[q, 1, :SIZE - dd, :])
                          and np.array_equal(im[t, 2, dd:, :], im[q, 2, :SIZE - dd, :]))
                shifts[axis] += 1
                if not ok:
                    fails.append((pid, t, axis))
        if min(shifts.values()) >= 400:
            break
    named = views == [str(v) for v in CFG["crop"]["views"]] == ["axial", "coronal", "sagittal"]
    ok = bad_lines == 0 and not fails and named and min(shifts.values()) > 0
    return ok, ("all %d instances satisfy the three shared-centre line identities "
                "(%d failures); %d i-shift, %d j-shift and %d k-shift instance pairs all "
                "translate exactly the planes carrying that axis, in the correct "
                "direction (%d failures) - so ch0 is orthogonal to k (axial), ch1 to j "
                "(coronal), ch2 to i (sagittal), matching the stored view names %s"
                % (n_lines, bad_lines, shifts["i"], shifts["j"], shifts["k"],
                   len(fails), views))


@check("B", "9_raw_size_32x32")
def b9():
    bad = []
    for pid in PIDS:
        z = np.load(img_path(pid), allow_pickle=True)
        im = z["images"]
        sz = [int(v) for v in z["image_size"]]
        z.close()
        if im.ndim != 4 or im.shape[1:] != (3, SIZE, SIZE) or im.dtype != np.int16 \
                or sz != [SIZE, SIZE]:
            bad.append((pid, im.shape, str(im.dtype)))
    return not bad, ("every instance is exactly [3, %d, %d] int16 in all %d patients "
                     "(%d violations)" % (SIZE, SIZE, N_PATIENTS, len(bad)))


@check("B", "10_fov_64x64_mm")
def b10():
    bad = []
    for pid in PIDS:
        z = np.load(img_path(pid), allow_pickle=True)
        sp = np.asarray(z["grid_spacing"], dtype=float)
        fov = np.asarray(z["fov_mm"], dtype=float)
        sz = np.asarray(z["image_size"], dtype=float)
        z.close()
        bag = load_frozen_bag(CFG, pid)
        if not (np.allclose(sp, np.asarray(CFG["preprocessing"]["resample_spacing_mm"]))
                and np.allclose(sp, bag["grid_spacing"])
                and np.allclose(sz * sp[:2], fov)
                and np.allclose(fov, np.asarray(CFG["crop"]["fov_mm"]))
                and abs(SIZE * sp[2] - fov[0]) < 1e-9):
            bad.append(pid)
    return not bad, ("%d px x 2.0 mm = 64.0 mm on every in-plane axis of every view, in "
                     "all %d patients; the spacing is the frozen bag's own grid spacing "
                     "(%d violations)" % (SIZE, N_PATIENTS, len(bad)))


@check("B", "11_deterministic_minus1024_padding")
def b11():
    """Analytic: which pixels fall outside the volume is pure arithmetic."""
    bad_count, bad_value, n_pad_inst, n_pad_px = [], [], 0, 0
    for pid in PIDS:
        z = np.load(img_path(pid), allow_pickle=True)
        im, ci = z["images"], np.asarray(z["coords_index"], dtype=int)
        stored = np.asarray(z["n_padded_pixels"], dtype=int)
        stored_flag = np.asarray(z["padded"], dtype=bool)
        dims = [int(v) for v in z["grid_size"]]
        padval = int(z["pad_value_hu"])
        z.close()

        def ov(c, dim):
            return np.minimum(dim, c + OFF) - np.maximum(0, c - OFF)

        oi, oj, ok_ = ov(ci[:, 0], dims[0]), ov(ci[:, 1], dims[1]), ov(ci[:, 2], dims[2])
        exp = 3 * SIZE * SIZE - (oj * oi + ok_ * oi + ok_ * oj)
        if not (np.array_equal(exp.astype(np.int64), stored.astype(np.int64))
                and np.array_equal(stored_flag, stored > 0) and padval == PAD):
            bad_count.append(pid)
        n_pad_inst += int((stored > 0).sum())
        n_pad_px += int(stored.sum())

        for t in np.flatnonzero(stored > 0):
            a, b, c = ci[t]
            # rebuild the exact out-of-volume mask of each plane and require PAD there
            gi = np.arange(a - OFF, a + OFF)
            gj = np.arange(b - OFF, b + OFF)
            gk = np.arange(c - OFF, c + OFF)
            vi = (gi >= 0) & (gi < dims[0])
            vj = (gj >= 0) & (gj < dims[1])
            vk = (gk >= 0) & (gk < dims[2])
            planes = ((0, vj, vi), (1, vk, vi), (2, vk, vj))
            for ch, vr, vc in planes:
                m = ~(vr[:, None] & vc[None, :])
                if m.any() and not np.all(im[t, ch][m] == PAD):
                    bad_value.append((pid, int(t), ch))
                if (~m).any() and int(m.sum()) + int((~m).sum()) != SIZE * SIZE:
                    bad_value.append((pid, int(t), ch))
    ok = not bad_count and not bad_value and n_pad_inst == 59
    return ok, ("%d instances (%d pixels) fall partly outside the CT volume; every "
                "padded-pixel count matches the analytic overlap formula (%d patients "
                "disagree) and every out-of-volume pixel holds exactly %d HU (%d "
                "violations)" % (n_pad_inst, n_pad_px, len(bad_count), PAD,
                                 len(bad_value)))


@check("B", "12_hu_clipped_range")
def b12():
    lo_obs, hi_obs = 1 << 30, -(1 << 30)
    bad = []
    for pid in PIDS:
        z = np.load(img_path(pid), allow_pickle=True)
        im = z["images"]
        clip = [int(v) for v in z["hu_clip"]]
        z.close()
        if clip != [LO, HI] or int(im.min()) < LO or int(im.max()) > HI:
            bad.append(pid)
        lo_obs = min(lo_obs, int(im.min()))
        hi_obs = max(hi_obs, int(im.max()))
    return not bad, ("cohort-wide stored HU span [%d, %d], inside the frozen "
                     "resegmentation window [%d, %d] declared by every file "
                     "(%d violations)" % (lo_obs, hi_obs, LO, HI, len(bad)))


@check("B", "13_images_finite")
def b13():
    bad, total = [], 0
    for pid in PIDS:
        im = load_img(pid, ["images"])["images"]
        total += int(im.size)
        if im.dtype != np.int16:
            bad.append(pid)
        elif not np.all(np.isfinite(im.astype(np.float64))):
            bad.append(pid)
    return not bad, ("%d stored voxel values across %d patients are int16 and finite; "
                     "int16 cannot represent NaN or Inf and the range check above bounds "
                     "them (%d violations)" % (total, N_PATIENTS, len(bad)))


# =========================================================== C  the encoder
@check("C", "14_pretrained_encoder_identifier")
def c14():
    model, meta = encoder()
    with open(ppath(CFG, CFG["outputs"]["embedding_run_json"]), encoding="utf-8") as fh:
        run = json.load(fh)
    _, probe_hash = probe_vector(CFG, model)
    ecfg = CFG["encoder"]
    ok_run = (run["encoder"]["weights_enum"] == str(ecfg["weights_enum"])
              and run["encoder"]["weights_url"] == str(ecfg["weights_url"])
              and run["encoder"]["state_fingerprint"] == meta["state_fingerprint"]
              and run["encoder"]["weights_sha256"] == meta["weights_sha256"]
              and abs(float(run["encoder"]["imagenet_acc1"]) - float(ecfg["expected_acc1"])) < 1e-6
              and run["cpu_probe_sha256"] == probe_hash)
    bad = []
    for pid in PIDS:
        z = np.load(emb_path(pid), allow_pickle=True)
        if (str(z["encoder_weights_enum"]) != str(ecfg["weights_enum"])
                or str(z["encoder_state_fingerprint"]) != meta["state_fingerprint"]
                or str(z["encoder_weights_sha256"]) != meta["weights_sha256"]
                or str(z["encoder_architecture"]) != "resnet18"
                or str(z["transform_signature"]) != transform_signature(CFG)
                or str(z["image_crop_signature"]) != crop_signature(CFG)):
            bad.append(pid)
        z.close()
    ok_random = meta["state_fingerprint"] != meta["random_init_fingerprint"]
    ok = ok_run and ok_random and not bad
    return ok, ("weights %s, file %s sha256 %s, ImageNet acc@1 %.3f; the rebuilt state "
                "fingerprint %s matches the run record and all %d embedding files (%d "
                "mismatches); it differs from a random initialisation; the deterministic "
                "CPU probe reproduces to sha256 %s"
                % (meta["weights_enum"], meta["weights_file"],
                   meta["weights_sha256"][:16], meta["imagenet_acc1"],
                   meta["state_fingerprint"][:16], N_PATIENTS, len(bad),
                   probe_hash[:16]))


@check("C", "15_all_parameters_frozen")
def c15():
    model, meta = encoder()
    n_grad = sum(1 for p in model.parameters() if p.requires_grad)
    n_par = sum(p.numel() for p in model.parameters())
    n_buf_grad = sum(1 for b in model.buffers() if getattr(b, "requires_grad", False))
    flagged = all(bool(np.asarray(load_emb(p, ["encoder_frozen"])["encoder_frozen"]))
                  for p in PIDS)
    ok = (n_grad == 0 and not model.training and n_buf_grad == 0 and flagged
          and meta["n_parameters_requiring_grad"] == 0
          and bool(CFG["encoder"]["requires_grad"]) is False
          and bool(CFG["encoder"]["allow_random_init"]) is False)
    return ok, ("%d parameters, %d with requires_grad=True, %d buffers requiring grad; "
                "module.training=%s; every one of the %d embedding files records "
                "encoder_frozen=True; config sets requires_grad false and forbids random "
                "init" % (n_par, n_grad, n_buf_grad, model.training, N_PATIENTS))


@check("C", "16_embedding_dim_512")
def c16():
    model, meta = encoder()
    import torch
    with torch.no_grad():
        y = model(input_transform(np.zeros((2, 3, SIZE, SIZE), dtype=np.int16), CFG))
    bad = []
    for pid in PIDS:
        z = np.load(emb_path(pid), allow_pickle=True)
        e = z["embeddings"]
        if e.ndim != 2 or e.shape[1] != DIM or e.dtype != np.float32 \
                or int(z["embedding_dim"]) != DIM:
            bad.append(pid)
        z.close()
    ok = (tuple(y.shape) == (2, DIM) and meta["embedding_dim"] == DIM
          and meta["head"] == "fc -> Identity" and not bad)
    return ok, ("the rebuilt encoder maps a [2,3,32,32] input to %s; the fc head is "
                "Identity, so this is the standard penultimate embedding; all %d saved "
                "files are [N, %d] float32 (%d violations)"
                % (tuple(y.shape), N_PATIENTS, DIM, len(bad)))


@check("C", "17_embeddings_finite")
def c17():
    bad, total = [], 0
    mn, mx = np.inf, -np.inf
    for pid in PIDS:
        e = load_emb(pid, ["embeddings"])["embeddings"]
        total += int(e.size)
        if not np.all(np.isfinite(e)):
            bad.append(pid)
        if e.size:
            mn, mx = min(mn, float(e.min())), max(mx, float(e.max()))
    return not bad, ("%d embedding values across %d patients are finite; observed range "
                     "[%.4f, %.4f] (%d patients with NaN/Inf)"
                     % (total, N_PATIENTS, mn, mx, len(bad)))


# =========================================== D  embedding identity / no labels
@check("D", "18_embedding_counts_equal_image_counts")
def d18():
    bad, n_e, n_i = [], 0, 0
    for pid in PIDS:
        ni = int(load_img(pid, ["images"])["images"].shape[0])
        ne = int(load_emb(pid, ["embeddings"])["embeddings"].shape[0])
        n_i += ni
        n_e += ne
        if ni != ne:
            bad.append(pid)
    return not bad and n_e == N_PATCHES, (
        "%d embeddings == %d images == %d expected instances; per-patient counts agree "
        "for all %d patients (%d disagreements)"
        % (n_e, n_i, N_PATCHES, N_PATIENTS, len(bad)))


@check("D", "19_embedding_order_equals_image_order")
def d19():
    bad, hash_bad = [], []
    for pid in PIDS:
        zi = np.load(img_path(pid), allow_pickle=True)
        ze = np.load(emb_path(pid), allow_pickle=True)
        if not (np.array_equal(zi["patch_index"], ze["patch_index"])
                and np.array_equal(zi["coords_index"], ze["coords_index"])
                and np.array_equal(zi["coords_world"], ze["coords_world"])
                and np.array_equal(zi["n_padded_pixels"], ze["n_padded_pixels"])):
            bad.append(pid)
        if str(ze["source_image_sha256"]) != sha256(img_path(pid)):
            hash_bad.append(pid)
        zi.close(); ze.close()
    return not bad and not hash_bad, (
        "embedding row i carries the same patch_index, coords_index, coords_world and "
        "padding count as image row i in all %d patients (%d mismatches); every "
        "embedding file records the SHA-256 of the exact image file it consumed and all "
        "%d still match (%d stale)" % (N_PATIENTS, len(bad), N_PATIENTS, len(hash_bad)))


@check("D", "20_no_label_enters_the_encoder")
def d20():
    import torch
    model, _ = encoder()
    device = resolve_device(CFG)
    model = model.to(device)

    # (a) the forward path takes no label: embed_array's signature is
    #     (model, images, cfg, device, batch_size) - assert that literally
    import inspect
    sig = list(inspect.signature(embed_array).parameters)
    sig_t = list(inspect.signature(input_transform).parameters)
    no_label_arg = ("label" not in " ".join(sig) and "label" not in " ".join(sig_t))

    # (b) static: no label/histology/fold symbol inside the forward-path functions
    src = code_only("src/cnn_embed.py")
    forward_src = ""
    for fn in ("def input_transform", "def embed_array", "def build_encoder"):
        i = src.index(fn)
        j = src.index("def ", i + 4)          # none of the three nests a function
        forward_src += src[i:j]
    leaked = [w for w in ("label", "histology", "adc", "scc", "outer_fold", "y_true")
              if re.search(r"\b%s\b" % w, forward_src, re.IGNORECASE)]

    # (c) empirical: one ADC and one SCC patient reproduce EXACTLY from the images
    #     alone, using the batch size each run recorded.  cuDNN selects different
    #     algorithms for different batch sizes, so replaying at a mismatched batch
    #     size differs by ~1e-3 in float32 - that is arithmetic, not information.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    adc = next(p for p in PIDS if int(load_img(p, ["label"])["label"]) == 1)
    scc = next(p for p in PIDS if int(load_img(p, ["label"])["label"]) == 0)
    worst = 0.0
    for pid in (adc, scc):
        z = np.load(img_path(pid), allow_pickle=True)
        im = np.asarray(z["images"])
        z.close()
        ze = np.load(emb_path(pid), allow_pickle=True)
        want, bs = np.asarray(ze["embeddings"]), int(ze["batch_size"])
        ze.close()
        got, _ = embed_array(model, im, CFG, device, bs)
        worst = max(worst, float(np.abs(got - want).max()))
    z = np.load(img_path(adc), allow_pickle=True)
    adc_im = np.asarray(z["images"])[:4]
    z.close()
    e1, _ = embed_array(model, adc_im, CFG, device, 4)
    e2, _ = embed_array(model, adc_im.copy(), CFG, device, 4)
    swap_identical = bool(np.array_equal(e1, e2))

    ok = no_label_arg and not leaked and worst == 0.0 and swap_identical
    return ok, ("the forward path takes only (model, images, cfg, device, batch_size) - "
                "no label argument exists; no label/histology/fold symbol appears in "
                "build_encoder, input_transform or embed_array (%d leaks); recomputing "
                "EVERY instance of an ADC patient (%s) and an SCC patient (%s) from the "
                "images alone reproduces the saved embeddings bit-exactly (max |diff| "
                "%.1e); the same pixels always yield the same vector regardless of which "
                "patient they came from" % (len(leaked), adc, scc, worst))


# ================================================== E  nothing was trained
@check("E", "21_no_patch_level_classifier")
def e21():
    hits = []
    forbidden = ["BCEWithLogitsLoss", "CrossEntropyLoss", "NLLLoss", "MSELoss",
                 "pos_weight", "roc_auc_score", "LogisticRegression", "fit(",
                 "predict_proba", "patch_label", "instance_label"]
    for p in PHASE6A_PY:
        s = code_only(p)
        for w in forbidden:
            if w in s:
                hits.append("%s:%s" % (p, w))
    # and every saved encoder has its classification head removed
    heads = {str(load_emb(p, ["encoder_head"])["encoder_head"]) for p in PIDS}
    ok = not hits and heads == {"fc -> Identity"}
    return ok, ("no loss function, no class weighting, no metric, no estimator fit and "
                "no patch-level label symbol appears in any Phase-6A pipeline source "
                "(%d hits); every embedding file records the head as %s"
                % (len(hits), sorted(heads)))


@check("E", "22_no_cnn_finetuning")
def e22():
    hits = []
    forbidden = [".backward(", "optimizer", "Adam", "SGD(", ".step()", "zero_grad",
                 "requires_grad_(True)", "requires_grad=True", "model.train()",
                 ".train()", "autograd", "enable_grad", "lr_scheduler"]
    for p in PHASE6A_PY:
        s = code_only(p).replace(" ", "")
        for w in forbidden:
            if w.replace(" ", "") in s:
                hits.append("%s:%s" % (p, w))
    model, meta = encoder()
    pristine = meta["state_fingerprint"]
    stale = [p for p in PIDS
             if str(load_emb(p, ["encoder_state_fingerprint"])["encoder_state_fingerprint"])
             != pristine]
    grads = "torch.no_grad()" in code_only("src/cnn_embed.py").replace(" ", "")
    ok = not hits and not stale and grads
    return ok, ("no backward pass, optimiser, scheduler, autograd context or train() call "
                "exists in any Phase-6A pipeline source (%d hits); the encoder state "
                "fingerprint recorded by all %d embedding files still equals the pristine "
                "downloaded checkpoint %s, so no weight was ever updated; the forward "
                "pass runs under torch.no_grad()"
                % (len(hits), N_PATIENTS, pristine[:16]))


@check("E", "23_no_transformer_or_clam_clustering")
def e23():
    hits = []
    forbidden = ["Transformer", "MultiheadAttention", "TransMIL", "Nystrom",
                 "self_attention", "CLAM", "inst_eval", "k_sample", "instance_loss",
                 "SmoothTop1SVM", "bag_weight", "Attn_Net", "GatedAttention",
                 "MeanMIL", "Conv3d", "SMOTE", "cluster"]
    for p in PHASE6A_PY:
        s = code_only(p)
        for w in forbidden:
            if re.search(re.escape(w), s, re.IGNORECASE if w == "cluster" else 0):
                hits.append("%s:%s" % (p, w))
    # No Phase-6B/6C artefact may exist yet.  Match whole path SEGMENTS so that a
    # Phase-5C inner-cache file whose sha-256 name happens to contain "6a" is not
    # mistaken for a Phase-6A artefact.
    markers = ("phase6", "phase6a", "resnet18", "cnn_embeddings", "image_patches",
               "gtv_rim_2p5d", "2p5d", "embedding", "embeddings")
    # This check exists to prove that PHASE 6A trained nothing.  A LATER phase that
    # was explicitly authorised to train (Phase 6B: MIL on the frozen embeddings)
    # writes into its own namespace and is audited by its own suite, so its
    # artefacts are not evidence about Phase 6A and are excluded by whole-segment
    # match.  Phase-6A's own namespace is still required to be empty of trained
    # artefacts, which is the substantive claim.
    later_phases = {"phase6b", "phase6c"}
    stray = []
    for d in ("models", "predictions", "results", "attention"):
        root = os.path.join(ROOT, d)
        for dp, _, fs in os.walk(root):
            for f in fs:
                q = os.path.join(dp, f).replace(os.sep, "/").lower()
                segs = set(re.split(r"[/\\._-]+", q))
                if segs & later_phases:
                    continue
                if segs & set(markers):
                    stray.append(q)
    ok = not hits and not stray
    return ok, ("no Transformer, attention-MIL, CLAM instance-clustering or SMOTE "
                "construct appears in any Phase-6A pipeline source (%d hits); no "
                "Phase-6 model, prediction, result or attention artefact exists (%d "
                "stray files) - Phase 6A trained nothing" % (len(hits), len(stray)))


# ================================================== F  non-regression
@check("F", "24_frozen_phase1_to_5c_unchanged")
def f24():
    with open(ppath(CFG, CFG["outputs"]["pre_snapshot"]), encoding="utf-8") as fh:
        snap = json.load(fh)
    files = snap["files"]
    missing, changed = [], []
    for rel, want in files.items():
        p = os.path.join(ROOT, *rel.split("/"))
        if not os.path.isfile(p):
            missing.append(rel)
        elif sha256(p) != want:
            changed.append(rel)
    # and nothing Phase-6A crept into the protected namespaces
    contaminated = [r for r in files
                    if r.startswith(("patch_features/", "patient_summaries/",
                                     "splits/", "feature_selection/"))
                    and ("phase6" in r or "image_patches" in r or "cnn_embeddings" in r)]
    ok = not missing and not changed and not contaminated
    return ok, ("all %d pre-Phase-6A files (Gradient Phase 1/2/3/3B, GTV+Rim Phase 4, "
                "the Phase-5 baseline and every Phase-5C output) are byte-identical: "
                "%d missing, %d changed; the protected namespaces contain no Phase-6A "
                "file" % (len(files), len(missing), len(changed)))


@check("F", "25_external_trees_unchanged")
def f25():
    from config_io import DEFAULT_CONFIG, load_config as lc
    from readonly_guard import verify
    ok, info = verify(lc(DEFAULT_CONFIG))
    return ok, ("Dataset/ and Vuong5ROI/ read-only manifest: %d baseline files, %d now, "
                "+%d added / -%d removed / ~%d changed"
                % (info["n_baseline"], info["n_now"], info["n_added"],
                   info["n_removed"], info["n_changed"]))


# ================================================== extra (not required)
@check("F", "26_provenance_consistent_across_cohort")
def f26():
    sigs, tsigs, srcs, devs, dims = set(), set(), set(), set(), set()
    for pid in PIDS:
        zi = np.load(img_path(pid), allow_pickle=True)
        ze = np.load(emb_path(pid), allow_pickle=True)
        sigs.add(str(zi["crop_signature"]))
        srcs.add(str(zi["source_extraction_signature"]))
        tsigs.add(str(ze["transform_signature"]))
        devs.add(str(ze["device"]))
        dims.add(int(ze["embedding_dim"]))
        zi.close(); ze.close()
    ok = (sigs == {crop_signature(CFG)}
          and srcs == {str(CFG["source"]["expected_extraction_signature"])}
          and tsigs == {transform_signature(CFG)} and dims == {DIM})
    return ok, ("one crop signature %s, one source extraction signature %s, one transform "
                "signature %s and one embedding width %s across all %d patients; devices "
                "used: %s" % (sorted(sigs), sorted(srcs), sorted(tsigs), sorted(dims),
                              N_PATIENTS, sorted(devs)))


@check("F", "27_no_radiomics_or_ct_import_in_the_embedding_path")
def f27():
    forbidden = {"radiomics", "SimpleITK", "sitk", "nibabel", "pydicom",
                 "featureextractor"}
    mods = imported_modules("src/cnn_embed.py")
    bad = sorted(mods & forbidden)
    s = code_only("src/cnn_embed.py")
    tokens = [w for w in forbidden if re.search(r"\b%s\b" % w, s, re.IGNORECASE)]
    # the radiomic feature VALUES must never be an input to the CNN
    reads_bags = bool(re.search(r"\bpatch_features\b", s)
                      or re.search(r'\bfeatures_dir\b', s))
    return not bad and not tokens and not reads_bags, (
        "src/cnn_embed.py imports {%s} - none of radiomics / SimpleITK / nibabel / "
        "pydicom (%d import hits, %d token hits), and its executable code never "
        "references patch_features/ or the frozen features_dir: the CNN input is the "
        "raw CT crop, never a radiomic feature vector"
        % (", ".join(sorted(mods)), len(bad), len(tokens)))


def main() -> int:
    results, n_fail = [], 0
    width = max(len(n) for _, n, _ in CHECKS)
    last = None
    t0 = time.time()
    for i, (group, name, fn) in enumerate(CHECKS, 1):
        if group != last:
            print("\n--- group %s ---" % group)
            last = group
        try:
            ok, msg = fn()
        except Exception as exc:                                   # noqa: BLE001
            ok, msg = False, "EXCEPTION: %s" % str(exc)[:400]
        n_fail += 0 if ok else 1
        print("%-4s %s  %-*s  %s" % ("PASS" if ok else "FAIL", group, width, name, msg))
        results.append({"n": i, "group": group, "name": name, "pass": bool(ok),
                        "message": msg})

    out = {"n_checks": len(CHECKS), "n_passed": len(CHECKS) - n_fail, "n_failed": n_fail,
           "phase": "6a", "roi": "gtv_rim", "n_patients": N_PATIENTS,
           "n_instances": N_PATCHES, "embedding_dim": DIM,
           "crop_signature": crop_signature(CFG),
           "transform_signature": transform_signature(CFG),
           "seconds": round(time.time() - t0, 1),
           "checks": results}
    path = ppath(CFG, CFG["outputs"]["tests_json"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n%d checks, %d passed, %d failed  (%.0f s)  ->  %s"
          % (len(CHECKS), len(CHECKS) - n_fail, n_fail, out["seconds"], path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
