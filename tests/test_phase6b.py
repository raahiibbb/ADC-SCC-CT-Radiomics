"""Phase-6B leakage, scope and non-regression checks.

Run with:
    ./.venv/Scripts/python.exe tests/test_phase6b.py

Phase 6B trains the two Phase-5 MIL models on the frozen Phase-6A ResNet18
embeddings under the frozen Phase-5 nested-CV procedure.  These checks re-derive
the whole chain from files on disk rather than trusting any recorded value:
the frozen split and embeddings, the inner 3-fold CV, both scaler levels, the
epoch-selection curves, the refit predictions, the attention outputs, the paired
patient-level bootstrap, and the untouched Phase-1..6A artefacts.

Groups and the 32 required verifications
  A  cohort, split and embedding identity                     (1-6)
  B  what is NOT in the trainable model or the training path   (7, 8, 22-27)
  C  leakage: splits, variance screen, scalers, weighting, epoch choice (9-15)
  D  fairness, labels and numerical sanity                     (16-18)
  E  attention outputs                                         (19-21)
  F  frozen artefacts and the read-only external trees         (28-31)
  G  the paired bootstrap                                      (32)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from mil_data import (fit_fold_scaler, load_cohort, load_outer_split,  # noqa: E402
                      load_phase3_config, model_tag, pos_weight_from_patients,
                      ppath, sha256_file)
from mil_metrics import patient_metrics  # noqa: E402
from mil_models import GatedAttentionMIL, MeanMIL, build_model  # noqa: E402
from repr_paired_compare import paired_bootstrap  # noqa: E402

CFG = load_phase3_config(os.path.join(ROOT, "config", "phase6b.yaml"))
CFG5 = load_phase3_config(os.path.join(ROOT, "config", "phase5.yaml"))
MODELS = ("mean_mil", "attention_mil")
FOLDS = (1, 2, 3, 4, 5)
N_INNER = int(CFG["inner_cv"]["n_splits"])
THR = float(CFG["evaluation"]["threshold"])
N_PATIENTS = int(CFG["phase3"]["expected_patients"])
N_INSTANCES = int(CFG["phase3"]["expected_instances"])
N_DIMS = int(CFG["phase3"]["expected_features"])
EMB_DIR = os.path.join(ROOT, *CFG["phase3"]["features_dir"].split("/"))
CHECKS = []

COHORT = load_cohort(CFG)
SPLIT = load_outer_split(CFG)
RUN = json.load(open(ppath(CFG, CFG["outputs"]["run_json"]), "r", encoding="utf-8"))

# Every Phase-6B source file that trains, evaluates or diagnoses.  The audit file
# itself is deliberately EXCLUDED: the forbidden-token scans below assert what the
# PIPELINE does, and a scanner that carries its own search lists would self-hit
# (the same defect found in Phase 5C and again in Phase 6A).
PHASE6B_PY = ("src/train_mil_cv.py", "src/train_mil.py", "src/mil_data.py",
              "src/mil_models.py", "src/mil_metrics.py", "src/evaluate_mil.py",
              "src/attention_stats.py", "src/repr_paired_compare.py",
              "src/phase6b_diagnostics.py")


def check(group, name):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


# ------------------------------------------------------------------- helpers
def source(rel):
    with open(os.path.join(ROOT, *rel.split("/")), "r", encoding="utf-8") as fh:
        return fh.read()


def code_only(rel):
    """Source with every comment and string literal removed.

    The scans assert what the code DOES.  A docstring that truthfully says
    "this never fine-tunes the CNN" must not count as a fine-tuning construct.
    """
    import io as _io
    import tokenize as _tk
    out, last = [], 1
    for tok in _tk.generate_tokens(_io.StringIO(source(rel)).readline):
        if tok.type in (_tk.COMMENT, _tk.STRING):
            continue
        if tok.start[0] != last:
            out.append("\n")
            last = tok.start[0]
        out.append(" " + tok.string)
    return "".join(out)


def imported_modules(rel):
    import ast as _ast
    mods = set()
    for node in _ast.walk(_ast.parse(source(rel))):
        if isinstance(node, _ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, _ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
            mods.update(a.name for a in node.names)
    return mods


def scan(tokens):
    """(file, token) pairs where a forbidden token appears in executable code.

    Matching is at identifier boundaries, not by substring: the honest metric
    `confusion_matrix` must not count as a "fusion" construct, and `torch.stack`
    must not count as `torch.cat`.  This is the Phase-5C / Phase-6A self-hit
    failure mode, avoided by construction rather than by exception list.
    """
    import re as _re
    hits = []
    for rel in PHASE6B_PY:
        src = code_only(rel).lower()
        for t in tokens:
            pat = r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % _re.escape(t.lower())
            if _re.search(pat, src):
                hits.append((rel, t))
    return hits


def inner_df(fold):
    return pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["inner_splits_dir"]),
                                    "fold_%d.csv" % fold))


def preds(model, cfg=CFG):
    return pd.read_csv(os.path.join(ppath(cfg, cfg["outputs"]["predictions_dir"]),
                                    "%s_oof.csv" % model_tag(cfg, model)))


def prep_dir():
    return ppath(CFG, CFG["outputs"]["preprocessing_dir"])


def scaler_npz(fold, inner=None):
    name = ("fold_%d_final_scaler.npz" % fold if inner is None
            else "fold_%d_inner_%d_scaler.npz" % (fold, inner))
    return np.load(os.path.join(prep_dir(), name), allow_pickle=True)


def scaler_meta(fold):
    return json.load(open(os.path.join(prep_dir(), "fold_%d_scalers.json" % fold),
                          "r", encoding="utf-8"))


def curve(model, fold):
    return pd.read_csv(os.path.join(ppath(CFG, CFG["outputs"]["results_dir"]),
                                    "epoch_selection", "%s_fold_%d.csv" % (model, fold)))


def checkpoint(model, fold):
    return torch.load(os.path.join(ppath(CFG, CFG["outputs"]["models_dir"]),
                                   model_tag(CFG, model), "fold_%d.pt" % fold),
                      map_location="cpu", weights_only=False)


def outer_sets(fold):
    sub = SPLIT[SPLIT["fold"] == fold]
    return (set(sub.loc[sub["split"] == "train", "PatientID"]),
            set(sub.loc[sub["split"] == "test", "PatientID"]))


def snapshot():
    return json.load(open(ppath(CFG, "reports/phase6b_pre_snapshot.json"),
                          "r", encoding="utf-8"))["files"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def att_dir():
    return ppath(CFG, CFG["outputs"]["attention_dir"])


# The ONE documented exception to byte-identity, stated openly rather than
# quietly excluded.  `reports/phase6a_tests.json` is an audit OUTPUT, not a data
# artefact, and it is rewritten whenever the Phase-6A suite is re-run for
# non-regression.  Two things make it differ from the pre-Phase-6B snapshot:
#   (a) it records its own wall-clock runtime (`seconds`), which is never equal
#       across runs;
#   (b) Phase-6A check 23 asserted a PROJECT-WIDE absence of Phase-6 model /
#       prediction / result / attention artefacts as its way of proving "Phase 6A
#       trained nothing".  Phase 6B was explicitly authorised to train and writes
#       into its own `phase6b` namespace, so that clause had to be narrowed to
#       Phase-6A's own namespace.  Its substantive claim is unchanged and still
#       verified: after the fix the check reports 0 stray files again.
# Instead of a byte comparison this file is verified SEMANTICALLY below.
AUDIT_OUTPUT_EXCEPTION = "reports/phase6a_tests.json"


def phase6a_audit_ok():
    with open(os.path.join(ROOT, "reports", "phase6a_tests.json"),
              "r", encoding="utf-8") as fh:
        d = json.load(fh)
    if not (int(d["n_checks"]) == 27 and int(d["n_passed"]) == 27
            and int(d["n_failed"]) == 0):
        return False, "Phase-6A audit is %s/%s" % (d["n_passed"], d["n_checks"])
    c23 = next((c for c in d["checks"] if c["name"].startswith("23_")), None)
    if c23 is None or not c23["pass"] or "(0 stray files)" not in c23["message"]:
        return False, "Phase-6A check 23 no longer reports 0 stray files"
    return True, "Phase-6A audit still 27/27 with check 23 reporting 0 stray files"


# ============================================ A  cohort / split / embeddings
@check("A", "1 exactly 202 patients, 51 ADC / 151 SCC, everywhere")
def a1():
    files = sorted(f[:-4] for f in os.listdir(EMB_DIR) if f.endswith(".npz"))
    lab = np.array([COHORT.bags[p].label for p in COHORT.patient_ids])
    counts = {"embedding files": len(files), "loaded bags": len(COHORT.bags),
              "split patients": SPLIT["PatientID"].nunique()}
    for m in MODELS:
        counts["%s predictions" % m] = len(preds(m))
    counts["attention files"] = len([f for f in os.listdir(att_dir())
                                     if f.endswith(".npz")])
    bad = {k: v for k, v in counts.items() if v != N_PATIENTS}
    if bad:
        return False, "not %d: %s" % (N_PATIENTS, bad)
    if files != sorted(COHORT.patient_ids):
        return False, "embedding file names disagree with the loaded cohort"
    if int((lab == 1).sum()) != 51 or int((lab == 0).sum()) != 151:
        return False, "%d ADC / %d SCC" % (int((lab == 1).sum()), int((lab == 0).sum()))
    return True, ("202 in all of: embedding files, loaded bags, split, both prediction "
                  "files and the attention directory; 51 ADC / 151 SCC")


@check("A", "2 exactly one out-of-fold prediction per patient per model")
def a2():
    for m in MODELS:
        df = preds(m)
        if df["PatientID"].duplicated().any():
            return False, "%s: duplicate predictions" % m
        if sorted(df["PatientID"]) != sorted(COHORT.patient_ids):
            return False, "%s: prediction patients differ from the cohort" % m
        for _, r in df.iterrows():
            if r["PatientID"] not in outer_sets(int(r["outer_fold"]))[1]:
                return False, ("%s: %s scored by a fold it was not held out of"
                               % (m, r["PatientID"]))
    return True, ("202 unique predictions per model; every patient scored exactly once, "
                  "by the model of its own held-out fold")


@check("A", "3 the frozen outer split is unchanged")
def a3():
    meta = json.load(open(ppath(CFG, CFG["phase3"]["splits_meta_file"]),
                          "r", encoding="utf-8"))
    now = sha256_file(ppath(CFG, CFG["phase3"]["splits_file"]))
    if now != meta["splits_file_sha256"]:
        return False, "split sha256 %s != its metadata" % now[:16]
    if now != RUN["input_manifest"]["splits_file_sha256"]:
        return False, "the split changed during the Phase-6B run"
    if now != RUN["preflight_verification"]["splits_file_sha256"]:
        return False, "the pre-flight verification saw a different split"
    if now != snapshot()["splits/gtv_rim_stratified_5fold.csv"]:
        return False, "the split differs from the pre-Phase-6B snapshot"
    for f in FOLDS:
        tr, te = outer_sets(f)
        d = inner_df(f)
        if set(d.loc[d["assignment"] == "outer_test", "PatientID"]) != te or \
           set(d.loc[d["assignment"] == "outer_train", "PatientID"]) != tr:
            return False, "fold %d membership changed" % f
    union = sorted(p for f in FOLDS for p in outer_sets(f)[1])
    if union != sorted(COHORT.patient_ids):
        return False, "the 5 test folds do not partition the 202 patients"
    return True, ("sha256 %s… matches its Phase-4 metadata, the run manifest, the "
                  "pre-flight record and the snapshot; 5 test folds partition all 202"
                  % now[:16])


@check("A", "4 the CNN embedding files are unchanged")
def a4():
    snap = snapshot()
    changed, missing = [], []
    for pid in COHORT.patient_ids:
        rel = "%s/%s.npz" % (CFG["phase3"]["features_dir"], pid)
        if rel not in snap:
            missing.append(rel)
        elif sha256(os.path.join(EMB_DIR, "%s.npz" % pid)) != snap[rel]:
            changed.append(rel)
    manifest = RUN["input_manifest"]["bag_sha256"]
    drift = [p for p in COHORT.patient_ids
             if manifest.get("%s.npz" % p) != sha256(os.path.join(EMB_DIR, "%s.npz" % p))]
    ok = not changed and not missing and not drift and RUN["frozen_inputs_unchanged"]
    return ok, ("202 embedding files byte-identical to the pre-Phase-6B snapshot "
                "(%d changed, %d missing) and to the run's own before/after manifest "
                "(%d drifted)" % (len(changed), len(missing), len(drift)))


@check("A", "5 exactly 512 raw embedding dimensions in every file")
def a5():
    widths, dims = set(), set()
    for pid in COHORT.patient_ids:
        with np.load(os.path.join(EMB_DIR, "%s.npz" % pid), allow_pickle=True) as z:
            e = z["embeddings"]
            widths.add(int(e.shape[1]))
            dims.add(int(z["embedding_dim"]))
            if e.dtype != np.float32:
                return False, "%s: embeddings are %s, not float32" % (pid, e.dtype)
    if widths != {N_DIMS} or dims != {N_DIMS}:
        return False, "widths %s, declared dims %s" % (sorted(widths), sorted(dims))
    if len(COHORT.feature_names) != N_DIMS:
        return False, "%d declared dimension names" % len(COHORT.feature_names)
    if int(RUN["preflight_verification"]["n_features"]) != N_DIMS:
        return False, "the pre-flight record disagrees"
    return True, ("all 202 files are [N, 512] float32, declare embedding_dim 512, and "
                  "512 dimension names are declared in %s"
                  % CFG["phase3"]["feature_names_file"])


@check("A", "6 patient bag counts equal the Phase-6A manifests")
def a6():
    man = pd.read_csv(ppath(CFG, "reports/phase6a_embedding_manifest.csv"))
    col = next((c for c in man.columns
                if c.lower() in ("n_embeddings", "n_instances", "n_patches", "n")), None)
    pid_col = next((c for c in man.columns
                    if c.lower() in ("patientid", "patient_id")), None)
    if col is None or pid_col is None:
        return False, "cannot locate count/id columns in the Phase-6A manifest: %s" \
                      % list(man.columns)
    man = man.set_index(pid_col)
    bad = [p for p in COHORT.patient_ids
           if int(man.loc[p, col]) != COHORT.bags[p].n_patches]
    total = int(sum(b.n_patches for b in COHORT.bags.values()))
    # and against the frozen radiomics bags the embeddings were built on
    bag_dir = os.path.join(ROOT, "patch_features", "gtv_rim")
    bad_bag = []
    for p in COHORT.patient_ids:
        with np.load(os.path.join(bag_dir, "%s.npz" % p), allow_pickle=True) as z:
            if int(z["features"].shape[0]) != COHORT.bags[p].n_patches:
                bad_bag.append(p)
            elif not np.array_equal(np.asarray(z["coords_index"], dtype=np.int32),
                                    COHORT.bags[p].coords_index):
                bad_bag.append(p)
    ok = not bad and not bad_bag and total == N_INSTANCES
    return ok, ("%d instances total (expected %d); per-patient counts agree with the "
                "Phase-6A embedding manifest (%d disagreements) and with the frozen "
                "GTV+Rim bags including coords_index (%d disagreements)"
                % (total, N_INSTANCES, len(bad), len(bad_bag)))


# =========================== B  what is NOT in the model or the training path
@check("B", "7 no image crop or ResNet inference occurs during MIL training")
def b7():
    forbidden_mods = {"torchvision", "SimpleITK", "sitk", "nibabel", "pydicom",
                      "radiomics", "PIL", "cv2", "skimage"}
    hits = []
    for rel in PHASE6B_PY:
        bad = imported_modules(rel) & forbidden_mods
        if bad:
            hits.append((rel, sorted(bad)))
    tok = scan(("resnet", "image_patches", "crop_signature", "input_transform",
                "build_encoder", "embed_array", "imagenet", "interpolate",
                "conv2d", "conv3d", ".nii", "dicom"))
    if RUN.get("radiomics_imported_during_training", True):
        return False, "the run record says radiomics/SimpleITK was imported"
    for mod in ("torchvision", "SimpleITK", "radiomics", "nibabel", "pydicom"):
        if mod in sys.modules:
            return False, "%s is importable-loaded in this process" % mod
    ok = not hits and not tok
    return ok, ("no image/CNN module imported by any Phase-6B module (%d), no crop or "
                "encoder construct in executable code (%d); the MIL input is the "
                "PRE-COMPUTED embedding array read from disk" % (len(hits), len(tok)))


@check("B", "8 no CNN parameter exists in the trainable MIL model")
def b8():
    for m in MODELS:
        for f in FOLDS:
            ck = checkpoint(m, f)
            sd = ck["model_state_dict"]
            cls = MeanMIL if m == "mean_mil" else GatedAttentionMIL
            model = cls(input_dim=int(ck["input_dim"]),
                        hidden=list(CFG["model"]["encoder_hidden"]),
                        dropout=float(CFG["model"]["dropout"]),
                        attention_hidden=int(CFG["model"]["attention_hidden"]))
            model.load_state_dict(sd)
            for name, mod in model.named_modules():
                if isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d,
                                    torch.nn.BatchNorm2d, torch.nn.MultiheadAttention,
                                    torch.nn.TransformerEncoderLayer)):
                    return False, "%s fold %d contains %s" % (m, f, type(mod).__name__)
            n_par = sum(p.numel() for p in model.parameters())
            expect = {"mean_mil": 512 * 64 + 64 + 64 * 32 + 32 + 32 + 1,
                      "attention_mil": 512 * 64 + 64 + 64 * 32 + 32 + 32 + 1
                                       + 2 * (32 * 32 + 32) + (32 + 1)}[m]
            if n_par != expect:
                return False, "%s fold %d has %d parameters, expected %d" \
                              % (m, f, n_par, expect)
            if any(k.startswith(("layer", "conv", "bn", "downsample", "fc.")) for k in sd):
                return False, "%s fold %d carries ResNet-shaped tensors" % (m, f)
    n_mean = 512 * 64 + 64 + 64 * 32 + 32 + 32 + 1
    n_att = n_mean + 2 * (32 * 32 + 32) + 33
    return True, ("all 10 checkpoints are pure MIL heads: %d (mean) / %d (attention) "
                  "parameters, only Linear layers, no convolution, normalisation, "
                  "attention-module or ResNet tensor anywhere" % (n_mean, n_att))


@check("B", "22 no radiomics extraction occurs anywhere in Phase 6B")
def b22():
    hits = []
    for rel in PHASE6B_PY:
        bad = imported_modules(rel) & {"radiomics", "SimpleITK", "featureextractor"}
        if bad:
            hits.append((rel, sorted(bad)))
    tok = scan(("featureextractor", "radiomicsfeatureextractor", "binwidth",
                "resegmentrange", "glcm", "glrlm", "glszm", "firstorder"))
    ok = not hits and not tok
    return ok, ("no radiomics import (%d) and no radiomics construct in executable code "
                "(%d) in any of the %d Phase-6B modules"
                % (len(hits), len(tok), len(PHASE6B_PY)))


@check("B", "23 no CNN fine-tuning: the encoder identity is unchanged in every file")
def b23():
    want = str(CFG["phase3"]["expected_encoder"]["state_fingerprint"])
    for pid in COHORT.patient_ids:
        with np.load(os.path.join(EMB_DIR, "%s.npz" % pid), allow_pickle=True) as z:
            if str(z["encoder_state_fingerprint"]) != want:
                return False, "%s: encoder fingerprint changed" % pid
            if not bool(z["encoder_frozen"]):
                return False, "%s: encoder_frozen is false" % pid
            if str(z["encoder_head"]) != "fc -> Identity":
                return False, "%s: encoder head changed" % pid
    tok = scan(("requires_grad_(true", "unfreeze", "finetune", "fine_tune",
                "load_state_dict(torch.hub", "resnet18(", "resnet50("))
    if tok:
        return False, "fine-tuning construct found: %s" % tok
    if RUN["preflight_verification"]["encoder"]["encoder_state_fingerprint"] != want:
        return False, "the pre-flight record saw a different encoder"
    return True, ("all 202 embedding files still carry encoder fingerprint %s… with "
                  "encoder_frozen=True and head `fc -> Identity`; no Phase-6B module "
                  "constructs, loads or updates a CNN" % want[:16])


@check("B", "24 no patch-level label is ever used")
def b24():
    tok = scan(("np.repeat", "repeat(label", "patch_label", "instance_label",
                "pseudo_label", "tile_label"))
    if tok:
        return False, "label-replication construct found: %s" % tok
    # every recorded class count is a PATIENT count
    for f in FOLDS:
        meta = scaler_meta(f)
        n_patch = sum(COHORT.bags[p].n_patches for p in outer_sets(f)[0])
        if int(meta["n_outer_train_patients"]) != len(outer_sets(f)[0]):
            return False, "fold %d: outer-train patient count is wrong" % f
        if int(meta["n_outer_train_patients"]) == n_patch:
            return False, "fold %d: a patch count is being reported as a patient count" % f
        for j, im in meta["inner"].items():
            if int(im["n_inner_train_adc"]) + int(im["n_inner_train_scc"]) \
                    != int(im["n_inner_train_patients"]):
                return False, "fold %d inner %s: class counts are not patient counts" % (f, j)
    # the loss is evaluated once per BAG: one patient -> one forward pass -> one
    # logit -> one target.  Checked structurally on the parsed source.
    import ast as _ast
    tree = _ast.parse(source("src/train_mil.py"))
    fn = next((n for n in _ast.walk(tree)
               if isinstance(n, _ast.FunctionDef) and n.name == "train_one_epoch"), None)
    if fn is None:
        return False, "train_one_epoch no longer exists"
    body = _ast.dump(fn)
    if "targets" not in body or "logits" not in body:
        return False, "the training loop no longer accumulates one logit per patient"
    n_targets_append = sum(
        1 for n in _ast.walk(fn)
        if isinstance(n, _ast.Call) and isinstance(n.func, _ast.Attribute)
        and n.func.attr == "append" and isinstance(n.func.value, _ast.Name)
        and n.func.value.id == "targets")
    if n_targets_append != 1:
        return False, "the training loop appends targets %d times, expected once per bag" \
                      % n_targets_append
    src_bag = _ast.dump(_ast.parse(source("src/mil_models.py")))
    if "n_patches" in src_bag:
        return False, "a model references a per-patch count"
    return True, ("no label-replication primitive in any Phase-6B module; every recorded "
                  "class count is a patient count; the loss consumes one logit per bag")


@check("B", "25 no CLAM instance clustering")
def b25():
    tok = scan(("clam", "clam_sb", "clam_mb", "inst_eval", "instance_eval",
                "smoothtop1svm", "topk_instance", "instance_loss", "cluster"))
    if tok:
        return False, "CLAM/clustering construct found: %s" % tok
    for m in MODELS:
        for f in FOLDS:
            ck = checkpoint(m, f)
            extra = [k for k in ck["model_state_dict"]
                     if k.split(".")[0] not in ("encoder", "classifier",
                                                "att_V", "att_U", "att_w")]
            if extra:
                return False, "%s fold %d has unexpected tensors %s" % (m, f, extra)
    # Phase 6C was explicitly authorised to implement CLAM-inspired instance
    # clustering and writes its own `src/clam_mil.py` / `src/train_mil_clam.py`.
    # This check is about PHASE 6B, so - exactly as Phase-6B narrowed Phase-6A's
    # check 23 - the clause asks whether any CLAM module reaches the Phase-6B
    # pipeline, not whether one exists anywhere in the project.  The substantive
    # claim is unchanged and still verified above: no CLAM construct in any
    # Phase-6B module and no clustering tensor in any Phase-6B checkpoint.
    clam_modules = {p[:-3] for p in os.listdir(os.path.join(ROOT, "src"))
                    if p.endswith(".py") and ("clam" in p.lower() or "cluster" in p.lower())}
    reached = sorted(m for rel in PHASE6B_PY for m in (imported_modules(rel) & clam_modules))
    return (not reached), ("no CLAM or instance-clustering construct in executable code; "
                           "all 10 checkpoints hold only encoder/classifier(/attention) "
                           "tensors; %d of the %d CLAM/clustering modules now in src/ are "
                           "imported by the Phase-6B pipeline"
                           % (len(reached), len(clam_modules)))


@check("B", "26 no Transformer MIL")
def b26():
    tok = scan(("transformer", "transmil", "nystrom", "multiheadattention",
                "self_attention", "positional_encoding", "vit_"))
    if tok:
        return False, "Transformer construct found: %s" % tok
    for m in MODELS:
        for f in FOLDS:
            ck = checkpoint(m, f)
            if ck["model_name"] != m:
                return False, "%s fold %d checkpoint claims model %s" % (m, f, ck["model_name"])
    stray = [p for p in os.listdir(os.path.join(ROOT, "src"))
             if "transformer" in p.lower() or "transmil" in p.lower()]
    return (not stray), ("no Transformer / self-attention / TransMIL construct in "
                         "executable code and no such source file; both models are the "
                         "unchanged Phase-5 MeanMIL / GatedAttentionMIL")


@check("B", "27 no radiomics + CNN fusion")
def b27():
    tok = scan(("fusion", "concatenate_features", "np.hstack", "torch.cat"))
    if tok:
        return False, "fusion-shaped construct found: %s" % tok
    # every trained model saw exactly 512-wide input, and the features_dir is the
    # embedding directory alone
    if CFG["phase3"]["features_dir"] != "cnn_embeddings/gtv_rim_resnet18":
        return False, "features_dir is not the embedding directory"
    for m in MODELS:
        for f in FOLDS:
            ck = checkpoint(m, f)
            if int(ck["input_dim"]) > N_DIMS:
                return False, "%s fold %d input_dim %d exceeds 512" % (m, f, ck["input_dim"])
            first = ck["model_state_dict"]["encoder.net.0.weight"]
            if int(first.shape[1]) != int(ck["input_dim"]):
                return False, "%s fold %d first-layer width disagrees" % (m, f)
    names = set(COHORT.feature_names)
    if any(n.startswith("original_") for n in names):
        return False, "a handcrafted radiomic feature name reached the Phase-6B input"
    return True, ("the only input directory is cnn_embeddings/gtv_rim_resnet18; every "
                  "checkpoint's first layer is exactly its fold's retained-embedding "
                  "width; no handcrafted feature name appears in the input space")


# =========================================================== C  leakage chain
@check("C", "9 inner train / inner validation / outer test never overlap")
def c9():
    for f in FOLDS:
        tr, te = outer_sets(f)
        d = inner_df(f)
        assign = d[d["assignment"] == "outer_train"].set_index("PatientID")["inner_fold"]
        if set(assign.index) != tr:
            return False, "fold %d: inner assignment does not cover the outer-train set" % f
        if sorted(assign.unique()) != list(range(1, N_INNER + 1)):
            return False, "fold %d: unexpected inner fold ids" % f
        for j in range(1, N_INNER + 1):
            itr = set(assign.index[assign != j])
            iva = set(assign.index[assign == j])
            if itr & iva:
                return False, "fold %d inner %d: train/val overlap" % (f, j)
            if (itr | iva) & te:
                return False, "fold %d inner %d: an OUTER TEST patient is inside" % (f, j)
            if len(np.unique(COHORT.labels(sorted(iva)))) < 2:
                return False, "fold %d inner %d: single-class validation" % (f, j)
    return True, ("5 x 3 inner splits are disjoint, exhaustive over the outer-training "
                  "patients, both classes present, and no outer-test patient appears "
                  "anywhere in them")


@check("C", "10 the variance screen sees inner-training patients only")
def c10():
    total = 0
    for f in FOLDS:
        d = inner_df(f)
        assign = d[d["assignment"] == "outer_train"].set_index("PatientID")["inner_fold"]
        for j in range(1, N_INNER + 1):
            itr = sorted(assign.index[assign != j])
            refit = fit_fold_scaler(CFG, COHORT, itr)
            got = scaler_npz(f, j)
            if not np.array_equal(np.asarray(got["keep_idx"], dtype=int), refit.keep_idx):
                return False, "fold %d inner %d: retained dimensions differ from a refit" % (f, j)
            # a screen that had seen the inner-validation patients would generally differ
            iva = sorted(assign.index[assign == j])
            contam = fit_fold_scaler(CFG, COHORT, itr + iva)
            if (contam.keep_idx.size == refit.keep_idx.size
                    and np.array_equal(contam.mean, refit.mean)):
                return False, "fold %d inner %d: screen is indistinguishable from a contaminated one" % (f, j)
            total += 1
    kept = sorted({int(scaler_npz(f, j)["keep_idx"].size)
                   for f in FOLDS for j in range(1, N_INNER + 1)})
    return True, ("all %d inner variance screens reproduce exactly from their "
                  "inner-training patients alone and differ from a contaminated screen; "
                  "retained dimensions %s of %d" % (total, kept, N_DIMS))


@check("C", "11 the inner scaler sees inner-training PATCHES only")
def c11():
    for f in FOLDS:
        d = inner_df(f)
        assign = d[d["assignment"] == "outer_train"].set_index("PatientID")["inner_fold"]
        for j in range(1, N_INNER + 1):
            itr = sorted(assign.index[assign != j])
            refit = fit_fold_scaler(CFG, COHORT, itr)
            got = scaler_npz(f, j)
            if not (np.allclose(np.asarray(got["mean"], float), refit.mean, atol=1e-12)
                    and np.allclose(np.asarray(got["std"], float), refit.std, atol=1e-12)):
                return False, "fold %d inner %d: statistics differ from an inner-train refit" % (f, j)
            iva = sorted(assign.index[assign == j])
            _, te = outer_sets(f)
            for name, extra in (("inner-val", iva), ("outer-test", sorted(te))):
                bad = fit_fold_scaler(CFG, COHORT, itr + extra)
                if np.allclose(bad.mean, refit.mean, atol=1e-12):
                    return False, ("fold %d inner %d: scaler is indistinguishable from one "
                                   "that also saw the %s patients" % (f, j, name))
    return True, ("all 15 inner scalers refit exactly from inner-training patches and are "
                  "provably different from scalers that also saw the inner-validation or "
                  "outer-test patches")


@check("C", "12 the final scaler sees ALL outer-training patients and no test patient")
def c12():
    for f in FOLDS:
        tr, te = outer_sets(f)
        refit = fit_fold_scaler(CFG, COHORT, sorted(tr))
        got = scaler_npz(f)
        if not (np.allclose(np.asarray(got["mean"], float), refit.mean, atol=1e-12)
                and np.allclose(np.asarray(got["std"], float), refit.std, atol=1e-12)
                and np.array_equal(np.asarray(got["keep_idx"], int), refit.keep_idx)):
            return False, "fold %d: final scaler differs from an all-outer-train refit" % f
        bad = fit_fold_scaler(CFG, COHORT, sorted(tr) + sorted(te))
        if np.allclose(bad.mean, refit.mean, atol=1e-12):
            return False, "fold %d: final scaler is indistinguishable from a contaminated one" % f
        for j in range(1, N_INNER + 1):
            inner = scaler_npz(f, j)
            if np.allclose(np.asarray(inner["mean"], float), refit.mean, atol=1e-12):
                return False, "fold %d: the final scaler equals inner scaler %d" % (f, j)
        if scaler_meta(f)["final_scaling_source"] != "all_outer_train_patches_only":
            return False, "fold %d: recorded scaling source is wrong" % f
        ck = checkpoint("mean_mil", f)
        if not np.allclose(np.asarray(ck["scaler_mean"], float), refit.mean, atol=1e-12):
            return False, "fold %d: the checkpoint carries a different scaler" % f
    return True, ("all 5 final scalers refit exactly from ALL outer-training patches, "
                  "differ from every inner scaler and from a test-contaminated variant, "
                  "and match the scaler stored in the checkpoints")


@check("C", "13 pos_weight comes from PATIENT counts, never patch counts")
def c13():
    vals = []
    for f in FOLDS:
        tr, _ = outer_sets(f)
        want = pos_weight_from_patients(COHORT.labels(sorted(tr)))
        got = float(scaler_npz(f)["pos_weight"])
        if abs(got - want) > 1e-12:
            return False, "fold %d: final pos_weight %.6f != patient ratio %.6f" % (f, got, want)
        n_adc_patch = sum(COHORT.bags[p].n_patches for p in tr if COHORT.bags[p].label == 1)
        n_scc_patch = sum(COHORT.bags[p].n_patches for p in tr if COHORT.bags[p].label == 0)
        if abs(got - n_scc_patch / n_adc_patch) < 1e-9:
            return False, "fold %d: pos_weight equals the PATCH ratio" % f
        vals.append(got)
        d = inner_df(f)
        assign = d[d["assignment"] == "outer_train"].set_index("PatientID")["inner_fold"]
        for j in range(1, N_INNER + 1):
            itr = sorted(assign.index[assign != j])
            w = pos_weight_from_patients(COHORT.labels(itr))
            if abs(float(scaler_npz(f, j)["pos_weight"]) - w) > 1e-12:
                return False, "fold %d inner %d: pos_weight is not the patient ratio" % (f, j)
        for m in MODELS:
            if abs(float(checkpoint(m, f)["pos_weight"]) - want) > 1e-12:
                return False, "fold %d %s: checkpoint pos_weight differs" % (f, m)
    return True, ("5 final + 15 inner pos_weights equal their PATIENT ratios exactly "
                  "(final %s); none equals the corresponding patch ratio"
                  % ", ".join("%.4f" % v for v in vals))


@check("C", "14 the epoch is chosen only from the mean inner-CV ROC-AUC curve")
def c14():
    for m in MODELS:
        for f in FOLDS:
            c = curve(m, f)
            if len(c) != int(CFG["training"]["max_epochs"]):
                return False, "%s fold %d: %d epochs recorded" % (m, f, len(c))
            cols = ["inner_%d_val_roc_auc" % j for j in range(1, N_INNER + 1)]
            mean = c[cols].mean(axis=1).values
            if not np.allclose(mean, c["mean_val_roc_auc"].values, atol=1e-12):
                return False, "%s fold %d: the stored mean curve is not the mean" % (m, f)
            want = int(np.argmax(mean)) + 1          # first maximum -> lowest epoch
            got = int(RUN["selected_epochs"][m][str(f)])
            if want != got:
                return False, "%s fold %d: selected %d, argmax of the mean curve is %d" \
                              % (m, f, got, want)
            if int(checkpoint(m, f)["selected_epoch"]) != want:
                return False, "%s fold %d: checkpoint disagrees" % (m, f)
            if int(preds(m).query("outer_fold == @f")["selected_epoch"].unique()[0]) != want:
                return False, "%s fold %d: prediction rows disagree" % (m, f)
            # the selection value must be an inner-CV number, never the test AUC
            sel_auc = float(mean[want - 1])
            test_auc = patient_metrics(
                preds(m).query("outer_fold == @f")["true_label"].values,
                preds(m).query("outer_fold == @f")["predicted_probability_ADC"].values,
                THR)["roc_auc"]
            if abs(sel_auc - test_auc) < 1e-12:
                return False, "%s fold %d: the selection value equals the outer-test AUC" % (m, f)
    eps = {m: [int(RUN["selected_epochs"][m][str(f)]) for f in FOLDS] for m in MODELS}
    return True, ("all 10 selections are the argmax of the 3-fold mean curve over the "
                  "full 100-epoch budget, agree with checkpoints and prediction rows, "
                  "and never equal the outer-test AUC; epochs %s" % eps)


@check("C", "15 the final refit uses exactly the selected epoch, and reproduces")
def c15():
    from train_mil_cv import make_split, run_training
    from train_mil import predict
    n_ok = 0
    for m in MODELS:
        for f in FOLDS:
            tr, te = outer_sets(f)
            ck = checkpoint(m, f)
            sel = int(ck["selected_epoch"])
            split = make_split(CFG, COHORT, sorted(tr), sorted(te))
            offset = 0 if m == "mean_mil" else 1
            seed = int(CFG["seeds"]["final_base_seed"]) + 10 * f + offset
            if int(ck["seed"]) != seed:
                return False, "%s fold %d: refit seed %d != %d" % (m, f, ck["seed"], seed)
            model, _, src = run_training(CFG, COHORT, split, m, seed, sel,
                                         eval_every_epoch=False)
            prob, _ = predict(model, src, split.evaluate,
                              torch.device(str(CFG["training"]["device"])))
            stored = preds(m).set_index("PatientID").loc[sorted(te),
                                                         "predicted_probability_ADC"].values
            if np.max(np.abs(prob - stored)) > 1e-9:
                return False, ("%s fold %d: retraining for exactly %d epochs does not "
                               "reproduce the stored predictions (max |diff| %.2e)"
                               % (m, f, sel, float(np.max(np.abs(prob - stored)))))
            n_ok += len(te)
    return True, ("all %d out-of-fold predictions reproduce to < 1e-9 by retraining from "
                  "scratch for exactly the selected epoch with the recorded seed" % n_ok)


# ==================================== D  fairness, labels, numerical sanity
@check("D", "16 Mean and Attention share folds, preprocessing and training settings")
def d16():
    for f in FOLDS:
        a, b = checkpoint("mean_mil", f), checkpoint("attention_mil", f)
        for k in ("fold", "input_dim", "pos_weight", "inner_cv_seed"):
            if a[k] != b[k]:
                return False, "fold %d: %s differs between the two models" % (f, k)
        if not (np.allclose(a["scaler_mean"], b["scaler_mean"], atol=1e-12)
                and np.allclose(a["scaler_std"], b["scaler_std"], atol=1e-12)):
            return False, "fold %d: the two models used different scalers" % f
        if a["kept_features"] != b["kept_features"]:
            return False, "fold %d: the two models used different dimensions" % f
        if json.dumps(a["config"], sort_keys=True, default=str) != \
           json.dumps(b["config"], sort_keys=True, default=str):
            return False, "fold %d: the two models were trained under different settings" % f
        wa = a["model_state_dict"]["encoder.net.0.weight"].shape
        wb = b["model_state_dict"]["encoder.net.0.weight"].shape
        if wa != wb:
            return False, "fold %d: encoder shapes differ" % f
    pm, pa = preds("mean_mil"), preds("attention_mil")
    if pm["PatientID"].tolist() != pa["PatientID"].tolist() or \
       pm["outer_fold"].tolist() != pa["outer_fold"].tolist():
        return False, "the two models were evaluated on different fold assignments"
    return True, ("per fold the two models share the scaler, retained dimensions, "
                  "pos_weight, inner-CV seed, model/training/inner_cv/seeds config block "
                  "and the identical 512-wide encoder shape; identical fold assignment")


@check("D", "17 the 202 out-of-fold labels match the cohort metadata")
def d17():
    cohort_csv = pd.read_csv(ppath(CFG, CFG["phase3"]["cohort_file"]))
    idc = next(c for c in cohort_csv.columns if c.lower() in ("patientid", "patient_id"))
    lc_ = next((c for c in cohort_csv.columns if c.lower() == "label"), None)
    meta = cohort_csv.set_index(idc)
    split_lab = SPLIT.drop_duplicates("PatientID").set_index("PatientID")["label"]
    for m in MODELS:
        df = preds(m).set_index("PatientID")
        for pid in COHORT.patient_ids:
            y = int(COHORT.bags[pid].label)
            if int(df.loc[pid, "true_label"]) != y:
                return False, "%s: %s label differs from the embedding file" % (m, pid)
            if int(split_lab[pid]) != y:
                return False, "%s: %s label differs from the frozen split" % (m, pid)
            if lc_ is not None and int(meta.loc[pid, lc_]) != y:
                return False, "%s: %s label differs from %s" % (m, pid, CFG["phase3"]["cohort_file"])
    lab = np.array([COHORT.bags[p].label for p in COHORT.patient_ids])
    return True, ("embedding files, %s, the frozen split and both prediction files agree "
                  "on all 202 labels (%d ADC / %d SCC)"
                  % (CFG["phase3"]["cohort_file"], int((lab == 1).sum()), int((lab == 0).sum())))


@check("D", "18 no NaN or Inf in any prediction, curve, scaler or metric")
def d18():
    n = 0
    for m in MODELS:
        p = preds(m)["predicted_probability_ADC"].values
        if not np.all(np.isfinite(p)):
            return False, "%s: non-finite probabilities" % m
        if p.min() < 0.0 or p.max() > 1.0:
            return False, "%s: probabilities outside [0, 1]" % m
        n += p.size
        for f in FOLDS:
            c = curve(m, f)
            if not np.all(np.isfinite(c.values.astype(float))):
                return False, "%s fold %d: non-finite epoch curve" % (m, f)
    for f in FOLDS:
        for s in [scaler_npz(f)] + [scaler_npz(f, j) for j in range(1, N_INNER + 1)]:
            if not (np.all(np.isfinite(np.asarray(s["mean"], float)))
                    and np.all(np.isfinite(np.asarray(s["std"], float)))):
                return False, "fold %d: non-finite scaler statistics" % f
            if float(np.asarray(s["std"], float).min()) <= 0:
                return False, "fold %d: a retained dimension has non-positive sd" % f
    metrics = json.load(open(os.path.join(ppath(CFG, CFG["outputs"]["results_dir"]),
                                          "oof_metrics.json"), "r", encoding="utf-8"))
    # pooled metrics must recompute from the prediction files
    for m in MODELS:
        want = patient_metrics(preds(m)["true_label"].values,
                               preds(m)["predicted_probability_ADC"].values, THR)
        got = metrics["models"][m]["pooled_oof"]
        for k in ("roc_auc", "pr_auc", "balanced_accuracy", "sensitivity_adc",
                  "specificity", "precision", "f1", "mcc", "accuracy"):
            if not np.isfinite(got[k]) or abs(float(got[k]) - want[k]) > 1e-12:
                return False, "%s: pooled %s does not recompute" % (m, k)
    return True, ("%d probabilities, 10 epoch curves, 20 scalers and all pooled metrics "
                  "are finite; every pooled metric recomputes from the prediction files "
                  "to 1e-12" % n)


# =============================================================== E  attention
@check("E", "19 attention length equals the patient's instance count")
def e19():
    files = sorted(f for f in os.listdir(att_dir()) if f.endswith(".npz"))
    if len(files) != N_PATIENTS:
        return False, "%d attention files" % len(files)
    total = 0
    for fn in files:
        pid = fn[:-4]
        with np.load(os.path.join(att_dir(), fn), allow_pickle=True) as z:
            a = np.asarray(z["attention"], dtype=np.float64)
            if a.shape[0] != COHORT.bags[pid].n_patches:
                return False, "%s: %d weights for %d instances" \
                              % (pid, a.shape[0], COHORT.bags[pid].n_patches)
            if not np.array_equal(np.asarray(z["patch_index"], dtype=np.int32),
                                  np.arange(a.shape[0], dtype=np.int32)):
                return False, "%s: patch_index is not 0..N-1" % pid
            if not np.array_equal(np.asarray(z["coords_index"], dtype=np.int32),
                                  COHORT.bags[pid].coords_index):
                return False, "%s: coords_index differs from the embedding file" % pid
            if not np.array_equal(np.asarray(z["coords_world"], dtype=np.float64),
                                  COHORT.bags[pid].coords_world):
                return False, "%s: coords_world differs from the embedding file" % pid
            if "attention_weights" not in z.files:
                return False, "%s: the required attention_weights key is missing" % pid
            total += a.shape[0]
    if total != N_INSTANCES:
        return False, "%d attention weights in total, expected %d" % (total, N_INSTANCES)
    return True, ("202 files, %d attention weights = the frozen instance count; "
                  "patch_index 0..N-1 and both coordinate arrays are identical to the "
                  "embedding files" % total)


@check("E", "20 every attention weight is finite and strictly positive")
def e20():
    lo, hi = np.inf, -np.inf
    for fn in sorted(f for f in os.listdir(att_dir()) if f.endswith(".npz")):
        with np.load(os.path.join(att_dir(), fn), allow_pickle=True) as z:
            a = np.asarray(z["attention"], dtype=np.float64)
            if not np.all(np.isfinite(a)):
                return False, "%s: non-finite attention" % fn[:-4]
            if a.min() <= 0.0:
                return False, "%s: a non-positive softmax weight" % fn[:-4]
            lo, hi = min(lo, float(a.min())), max(hi, float(a.max()))
    return True, ("all %d weights finite and > 0; observed range [%.3e, %.6f]"
                  % (N_INSTANCES, lo, hi))


@check("E", "21 every bag's attention sums to 1")
def e21():
    worst, worst_pid = 0.0, None
    for fn in sorted(f for f in os.listdir(att_dir()) if f.endswith(".npz")):
        with np.load(os.path.join(att_dir(), fn), allow_pickle=True) as z:
            a = np.asarray(z["attention"], dtype=np.float64)
            d = abs(float(a.sum()) - 1.0)
            if d > worst:
                worst, worst_pid = d, fn[:-4]
            if abs(float(z["attention_sum"]) - float(a.sum())) > 1e-12:
                return False, "%s: recorded attention_sum disagrees" % fn[:-4]
    if worst > 1e-5:
        return False, "worst |sum - 1| = %.3e (%s)" % (worst, worst_pid)
    return True, "max |sum - 1| = %.2e over 202 bags (%s)" % (worst, worst_pid)


# ==================================== F  frozen artefacts and external trees
@check("F", "28 the Phase-5 predictions are byte-identical")
def f28():
    snap = snapshot()
    rels = ["predictions/gtv_rim_mean_mil_oof.csv",
            "predictions/gtv_rim_attention_mil_oof.csv",
            "results/phase5/oof_metrics.json"]
    for rel in rels:
        if sha256(os.path.join(ROOT, *rel.split("/"))) != snap[rel]:
            return False, "%s CHANGED" % rel
    got = {}
    for m in MODELS:
        df = pd.read_csv(os.path.join(ROOT, "predictions", "gtv_rim_%s_oof.csv" % m))
        got[m] = patient_metrics(df["true_label"].values,
                                 df["predicted_probability_ADC"].values, THR)["roc_auc"]
    if abs(got["mean_mil"] - 0.5702) > 5e-4 or abs(got["attention_mil"] - 0.5767) > 5e-4:
        return False, "the Phase-5 baseline no longer scores 0.5702 / 0.5767: %s" % got
    return True, ("Phase-5 prediction files byte-identical and the baseline still scores "
                  "pooled OOF ROC-AUC %.4f (mean) / %.4f (attention)"
                  % (got["mean_mil"], got["attention_mil"]))


@check("F", "29 the Phase-5C predictions are byte-identical")
def f29():
    snap = snapshot()
    rels = sorted(r for r in snap if r.startswith("predictions/phase5c_"))
    if len(rels) != 4:
        return False, "%d Phase-5C prediction files in the snapshot" % len(rels)
    for rel in rels:
        if sha256(os.path.join(ROOT, *rel.split("/"))) != snap[rel]:
            return False, "%s CHANGED" % rel
    df = pd.read_csv(os.path.join(ROOT, "predictions", "phase5c_mrmr_attention_oof.csv"))
    auc = patient_metrics(df["true_label"].values,
                          df["predicted_probability_ADC"].values, THR)["roc_auc"]
    if abs(auc - 0.5929) > 5e-4:
        return False, "the best handcrafted pipeline no longer scores 0.5929 (%.4f)" % auc
    fs = sorted(r for r in snap if r.startswith("feature_selection/phase5c/"))
    changed = [r for r in fs if sha256(os.path.join(ROOT, *r.split("/"))) != snap[r]]
    return (not changed), ("4 Phase-5C prediction files and %d feature-selection records "
                           "byte-identical (%d changed); mRMR+attention still scores %.4f"
                           % (len(fs), len(changed), auc))


@check("F", "30 the Phase-6A embeddings and crops are byte-identical")
def f30():
    snap = snapshot()
    rels = [r for r in snap if r.startswith(("cnn_embeddings/", "image_patches/"))]
    changed = [r for r in rels if sha256(os.path.join(ROOT, *r.split("/"))) != snap[r]]
    prov = [r for r in snap
            if r.startswith("reports/phase6a_") and r != AUDIT_OUTPUT_EXCEPTION]
    changed += [r for r in prov
                if os.path.isfile(os.path.join(ROOT, *r.split("/")))
                and sha256(os.path.join(ROOT, *r.split("/"))) != snap[r]]
    ok_audit, msg_audit = phase6a_audit_ok()
    return (not changed and ok_audit), (
        "%d Phase-6A crop/embedding files and %d Phase-6A provenance files "
        "byte-identical; %d changed. The one audit OUTPUT excluded from byte "
        "comparison (%s) is verified semantically instead: %s"
        % (len(rels), len(prov), len(changed), AUDIT_OUTPUT_EXCEPTION, msg_audit))


@check("F", "31 Dataset/ and Vuong5ROI/ remain read-only, and nothing else moved")
def f31():
    from config_io import DEFAULT_CONFIG, load_config as lc
    from readonly_guard import verify
    ok, info = verify(lc(DEFAULT_CONFIG))
    snap = snapshot()
    changed = [r for r in snap
               if r != AUDIT_OUTPUT_EXCEPTION
               and (not os.path.isfile(os.path.join(ROOT, *r.split("/")))
                    or sha256(os.path.join(ROOT, *r.split("/"))) != snap[r])]
    # Phase 6B must not have written into any protected namespace
    protected = ("patch_features/", "patient_summaries/", "feature_selection/",
                 "image_patches/", "cnn_embeddings/", "attention/gtv_rim/",
                 "attention/gradient", "models/gtv_rim_", "models/phase5c_",
                 "results/phase5", "predictions/gtv_rim_", "predictions/phase5c_",
                 "preprocessing/gtv_rim", "preprocessing/phase5c",
                 "splits/gtv_rim_inner_cv", "splits/phase5c_inner_cv")
    intruders = []
    for pre in protected:
        d = os.path.join(ROOT, *pre.rstrip("/").split("/"))
        base = d if os.path.isdir(d) else os.path.dirname(d)
        if not os.path.isdir(base):
            continue
        for root_, _, fs in os.walk(base):
            for f in fs:
                rel = os.path.relpath(os.path.join(root_, f), ROOT).replace(os.sep, "/")
                if rel.startswith(pre) and rel not in snap:
                    intruders.append(rel)
    ok_audit, _ = phase6a_audit_ok()
    ok = ok and not changed and not intruders and ok_audit
    return ok, ("Dataset/ + Vuong5ROI/: %d baseline files, %d now, +%d -%d ~%d; %d of %d "
                "pre-Phase-6B project files byte-identical (%d changed); %d new files "
                "inside a protected namespace; the single documented exception (%s, an "
                "audit output) is verified semantically"
                % (info["n_baseline"], info["n_now"], info["n_added"], info["n_removed"],
                   info["n_changed"], len(snap) - 1, len(snap), len(changed),
                   len(set(intruders)), AUDIT_OUTPUT_EXCEPTION))


# =============================================================== G  bootstrap
@check("G", "32 the paired bootstrap resamples PATIENTS only, and reproduces")
def g32():
    rc = CFG["representation_comparison"]
    boot = json.load(open(ppath(CFG, rc["outputs"]["bootstrap_json"]),
                          "r", encoding="utf-8"))
    n_patch = int(sum(b.n_patches for b in COHORT.bags.values()))
    n_done = 0
    for group, entries in (("primary_same_architecture", rc["primary"]),
                           ("secondary_benchmark", rc["secondary"])):
        for spec in entries:
            c = boot[group][str(spec["key"])]
            bs = c["bootstrap"]
            if bs["resampling_unit"] != "patient" or not bs["paired"]:
                return False, "%s: not a paired patient-level resample" % spec["key"]
            if int(bs["n_patients"]) != N_PATIENTS or int(bs["n_patients"]) == n_patch:
                return False, "%s: the unit looks like patches, not patients" % spec["key"]
            if int(bs["n_resamples_requested"]) != int(rc["bootstrap"]["n_resamples"]):
                return False, "%s: %d resamples" % (spec["key"], bs["n_resamples_requested"])
            new = preds(str(c["phase6b_model"])).set_index("PatientID").sort_index()
            ref = pd.read_csv(os.path.join(ROOT, *str(spec["reference_predictions"]).split("/"))
                              ).set_index("PatientID").sort_index()
            pids = sorted(set(new.index) & set(ref.index))
            if len(pids) != N_PATIENTS:
                return False, "%s: %d paired patients" % (spec["key"], len(pids))
            y = new.loc[pids, "true_label"].values.astype(int)
            if not np.array_equal(y, ref.loc[pids, "true_label"].values.astype(int)):
                return False, "%s: labels disagree across the pair" % spec["key"]
            redo = paired_bootstrap(
                y,
                new.loc[pids, "predicted_probability_ADC"].values.astype(float),
                ref.loc[pids, "predicted_probability_ADC"].values.astype(float),
                pids, int(rc["bootstrap"]["n_resamples"]), int(spec["seed"]),
                float(rc["bootstrap"]["ci"]))
            for k in ("roc_auc", "pr_auc"):
                for fld in ("difference_mean", "ci_low", "ci_high"):
                    if abs(redo["metrics"][k][fld] - float(bs["metrics"][k][fld])) > 1e-12:
                        return False, "%s: %s %s not reproducible from seed %d" \
                                      % (spec["key"], k, fld, spec["seed"])
                if bool(redo["metrics"][k]["ci_includes_zero"]) != \
                        bool(bs["metrics"][k]["ci_includes_zero"]):
                    return False, "%s: %s CI verdict differs" % (spec["key"], k)
            n_done += 1
    return True, ("all %d paired bootstraps reproduce exactly from their recorded seeds; "
                  "the unit is the patient (%d), never the patch (%d)"
                  % (n_done, N_PATIENTS, n_patch))


def main() -> int:
    results, n_fail = [], 0
    width = max(len(n) for _, n, _ in CHECKS)
    last = None
    for i, (group, name, fn) in enumerate(CHECKS, 1):
        if group != last:
            print("\n--- group %s ---" % group)
            last = group
        try:
            ok, msg = fn()
        except Exception as exc:                        # noqa: BLE001
            ok, msg = False, "EXCEPTION: %s" % str(exc)[:400]
        n_fail += 0 if ok else 1
        print("%-4s %s  %-*s  %s" % ("PASS" if ok else "FAIL", group, width, name, msg))
        results.append({"n": i, "group": group, "name": name, "pass": bool(ok),
                        "message": msg})

    out = {"n_checks": len(CHECKS), "n_passed": len(CHECKS) - n_fail, "n_failed": n_fail,
           "phase": "6b", "roi": "gtv_rim", "representation": "resnet18_imagenet_512d",
           "models": list(MODELS), "folds": list(FOLDS), "n_inner_splits": N_INNER,
           "threshold": THR, "n_patients": N_PATIENTS, "n_instances": N_INSTANCES,
           "n_embedding_dimensions": N_DIMS, "checks": results}
    path = ppath(CFG, "reports/phase6b_tests.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n%d checks, %d passed, %d failed  ->  %s"
          % (len(CHECKS), len(CHECKS) - n_fail, n_fail, path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
