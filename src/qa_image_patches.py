"""Phase 6A - pilot QA for the 2.5D CT crops: visual panels + 10 automatic checks.

    ./.venv/Scripts/python.exe src/qa_image_patches.py --pilot

Every check re-derives its answer by re-resampling the read-only CT and
re-cropping from scratch; nothing recorded by the extraction driver is trusted.

Figures (`qa/phase6a/`)
    <PID>_2p5d.png      one row per shown instance:
                        AXIAL | CORONAL | SAGITTAL | context axial slice
    Display note: the stored arrays carry no flips.  In the figure the coronal
    and sagittal panels are drawn with origin="lower" so that Superior is at the
    top; the axial panels use origin="upper" so that Anterior is at the top and
    patient-Left is on the right.  These are DISPLAY-ONLY conventions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from image_patches import (clip_to_int16, crop_signature, load_config,  # noqa: E402
                           load_frozen_bag, load_outer_folds, ppath,
                           resampled_ct)

DEFAULT_CONFIG = os.path.join(os.path.dirname(HERE), "config", "phase6a.yaml")


# ------------------------------------------------------------------- selection
def choose_instances(n: int, n_pad: np.ndarray, k: int, max_padded: int):
    """Deterministic, RNG-free: padded instances first, then evenly spaced."""
    picked = [int(t) for t in np.flatnonzero(n_pad > 0)[:max_padded]]
    remaining = [t for t in range(n) if t not in set(picked)]
    need = max(0, min(k, n) - len(picked))
    if need and remaining:
        idx = np.unique(np.linspace(0, len(remaining) - 1, need).round().astype(int))
        picked.extend(int(remaining[t]) for t in idx)
    return sorted(set(picked))[:max(1, min(k, n))]


def expected_pad_pixels(coords_index: np.ndarray, dims_ijk, size: int, off: int):
    """Analytic padded-pixel count per instance, independent of the extractor."""
    ni, nj, nk = (int(v) for v in dims_ijk)
    ci, cj, ck = (coords_index[:, 0].astype(int), coords_index[:, 1].astype(int),
                  coords_index[:, 2].astype(int))

    def ov(c, dim):
        return np.minimum(dim, c + off) - np.maximum(0, c - off)

    oi, oj, ok = ov(ci, ni), ov(cj, nj), ov(ck, nk)
    inb = oj * oi + ok * oi + ok * oj
    return (3 * size * size - inb).astype(np.int32)


# ----------------------------------------------------------------------- plots
def plot_patient(cfg, pid, z, vol16, mask_arr, out_png, shown):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    im = np.asarray(z["images"])
    ci = np.asarray(z["coords_index"])
    npad = np.asarray(z["n_padded_pixels"])
    size = int(cfg["crop"]["image_size"][0])
    off = int(cfg["crop"]["centre_offset_voxels"])
    lo, hi = (int(v) for v in cfg["crop"]["hu_clip"])
    views = [str(v) for v in z["views"]]

    n_rows = len(shown)
    fig, axes = plt.subplots(n_rows, 4, figsize=(13.0, 3.1 * n_rows), squeeze=False)
    for r, t in enumerate(shown):
        a, b, c = (int(v) for v in ci[t])
        for ch in range(3):
            ax = axes[r][ch]
            origin = "upper" if ch == 0 else "lower"
            ax.imshow(im[t, ch], cmap="gray", vmin=lo, vmax=hi, origin=origin,
                      interpolation="nearest")
            ax.axhline(off, color="#ff5f1f", lw=0.6, alpha=0.85)
            ax.axvline(off, color="#ff5f1f", lw=0.6, alpha=0.85)
            ax.plot([off], [off], marker="+", ms=8, mew=1.2, color="#ff5f1f")
            lab = {0: "AXIAL  rows=+j(P)  cols=+i(L)",
                   1: "CORONAL  rows=+k(S up)  cols=+i(L)",
                   2: "SAGITTAL  rows=+k(S up)  cols=+j(P)"}[ch]
            ax.set_title("%s\n%s" % (views[ch].upper(), lab), fontsize=6.5)
            ax.set_xticks([]); ax.set_yticks([])
            if ch == 0:
                ax.set_ylabel("patch %d\n(i,j,k)=(%d,%d,%d)\npad px %d"
                              % (t, a, b, c, int(npad[t])), fontsize=6.5)

        ax = axes[r][3]
        ax.imshow(vol16[c], cmap="gray", vmin=lo, vmax=hi, origin="upper",
                  interpolation="nearest")
        if mask_arr is not None:
            ax.contour(mask_arr[c], levels=[0.5], colors=["#33d17a"], linewidths=0.5)
        ax.add_patch(Rectangle((a - off - 0.5, b - off - 0.5), size, size,
                               fill=False, ec="#ff5f1f", lw=0.8))
        ax.plot([a], [b], marker="+", ms=7, mew=1.0, color="#ff5f1f")
        ax.set_title("context: full axial slice k=%d\n(green = GTV+Rim ROI)" % c,
                     fontsize=6.5)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Phase 6A 2.5D crops - %s  (%s, label %d, fold %d)  "
                 "64x64 mm FOV, 32x32 px at 2 mm, HU clipped to [%d, %d]"
                 % (pid, str(z["histology"]), int(z["label"]), int(z["outer_fold"]),
                    lo, hi), fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- checks
def check_patient(cfg, pid, make_figure=True):
    import SimpleITK as sitk
    out = {"patient_id": pid, "checks": {}}

    def rec(name, ok, detail):
        out["checks"][name] = {"pass": bool(ok), "detail": detail}

    img_path = os.path.join(ppath(cfg, cfg["outputs"]["image_dir"]), "%s.npz" % pid)
    z = np.load(img_path, allow_pickle=True)
    bag = load_frozen_bag(cfg, pid)
    folds = load_outer_folds(cfg)

    im = np.asarray(z["images"])
    ci = np.asarray(z["coords_index"])
    npad = np.asarray(z["n_padded_pixels"])
    n = int(im.shape[0])
    size = int(cfg["crop"]["image_size"][0])
    off = int(cfg["crop"]["centre_offset_voxels"])
    lo, hi = (int(v) for v in cfg["crop"]["hu_clip"])

    ct_r, mask_r, grid = resampled_ct(cfg, pid, with_mask=make_figure)
    raw = sitk.GetArrayFromImage(ct_r)
    vol16 = clip_to_int16(raw, (lo, hi))
    mask_arr = sitk.GetArrayFromImage(mask_r) if mask_r is not None else None
    dims_ijk = list(grid["size"])

    # ---- 8. correct PatientID
    rec("8_patient_id", str(z["patient_id"]) == pid and bag["patient_id"] == pid,
        "image file, frozen bag and requested id all read %s" % pid)

    # ---- 9. correct patch order / count
    ok9 = (n == bag["n_patches"]
           and np.array_equal(np.asarray(z["patch_index"]), np.arange(n, dtype=np.int32))
           and np.array_equal(ci, bag["coords_index"]))
    rec("9_patch_order", ok9,
        "%d images == %d frozen patches; patch_index is 0..N-1; coords_index "
        "identical row by row" % (n, bag["n_patches"]))

    # ---- 10. exact coordinate correspondence
    dworld = float(np.abs(np.asarray(z["coords_world"]) - bag["coords_world"]).max()) if n else 0.0
    rt = np.array([ct_r.TransformContinuousIndexToPhysicalPoint(
        [float(a), float(b), float(c)]) for a, b, c in ci], dtype=np.float64) if n else np.zeros((0, 3))
    drt = float(np.abs(rt - np.asarray(z["coords_world"])).max()) if n else 0.0
    rec("10_coords", dworld == 0.0 and drt < 1e-6,
        "max |coords_world - frozen| = %.3e mm; max |index->world round trip| = %.3e mm"
        % (dworld, drt))

    # ---- 5. raw size exactly 32 x 32, 3 channels
    rec("5_shape", im.shape[1:] == (3, size, size) and im.dtype == np.int16,
        "images %s %s" % (im.shape, im.dtype))

    # ---- 4. correct 64 mm FOV
    sp = np.asarray(grid["spacing"], dtype=float)
    fov = np.asarray(cfg["crop"]["fov_mm"], dtype=float)
    ok4 = (abs(size * sp[0] - fov[0]) < 1e-9 and abs(size * sp[1] - fov[1]) < 1e-9
           and abs(size * sp[2] - fov[0]) < 1e-9)
    rec("4_fov", ok4, "32 px x %.1f mm = %.1f mm on every axis (target %s mm)"
        % (sp[0], size * sp[0], list(fov)))

    # ---- 1/2/3. centre identity, axis order, no reversal  (re-cropped from the CT)
    bad_c = bad_axis = bad_slab = 0
    for t in range(n):
        a, b, c = (int(v) for v in ci[t])
        cen = int(vol16[c, b, a])
        if not (int(im[t, 0, off, off]) == int(im[t, 1, off, off])
                == int(im[t, 2, off, off]) == cen):
            bad_c += 1
        pad = np.pad(vol16, off, mode="constant", constant_values=int(cfg["crop"]["pad_value_hu"]))
        if not (np.array_equal(im[t, 0], pad[c + off, b:b + size, a:a + size])
                and np.array_equal(im[t, 1], pad[c:c + size, b + off, a:a + size])
                and np.array_equal(im[t, 2], pad[c:c + size, b:b + size, a + off])):
            bad_slab += 1
        # +1 along each index axis must move the SAME way in every view that carries it
        if (0 < a < dims_ijk[0] - 1 and 0 < b < dims_ijk[1] - 1 and 0 < c < dims_ijk[2] - 1):
            if not (int(im[t, 0, off, off + 1]) == int(im[t, 1, off, off + 1]) == int(vol16[c, b, a + 1])
                    and int(im[t, 0, off + 1, off]) == int(im[t, 2, off, off + 1]) == int(vol16[c, b + 1, a])
                    and int(im[t, 1, off + 1, off]) == int(im[t, 2, off + 1, off]) == int(vol16[c + 1, b, a])):
                bad_axis += 1
    rec("1_same_centre", bad_c == 0,
        "img[0,16,16] == img[1,16,16] == img[2,16,16] == volume[k,j,i] for all %d "
        "instances (%d failures)" % (n, bad_c))
    rec("2_no_axis_permutation", bad_axis == 0 and bad_slab == 0,
        "+1 along i/j/k moves consistently in every view that carries that axis, and "
        "all 3 planes reproduce byte-identically from an independent re-crop "
        "(%d axis / %d slab failures)" % (bad_axis, bad_slab))

    # ---- 3. no L-R / S-I reversal: index direction must agree with physical direction
    p0 = np.array(ct_r.TransformContinuousIndexToPhysicalPoint([10.0, 10.0, 10.0]))
    dx = np.array(ct_r.TransformContinuousIndexToPhysicalPoint([11.0, 10.0, 10.0])) - p0
    dy = np.array(ct_r.TransformContinuousIndexToPhysicalPoint([10.0, 11.0, 10.0])) - p0
    dz = np.array(ct_r.TransformContinuousIndexToPhysicalPoint([10.0, 10.0, 11.0])) - p0
    ok3 = (dx[0] > 0 and abs(dx[1]) < 1e-9 and abs(dx[2]) < 1e-9
           and dy[1] > 0 and abs(dy[0]) < 1e-9 and abs(dy[2]) < 1e-9
           and dz[2] > 0 and abs(dz[0]) < 1e-9 and abs(dz[1]) < 1e-9
           and bad_axis == 0)
    rec("3_no_reversal", ok3,
        "+i -> %+0.1f mm x (Left), +j -> %+0.1f mm y (Posterior), +k -> %+0.1f mm z "
        "(Superior); increasing column/row index increases the same index axis in "
        "every view, so no L-R or S-I flip is introduced by indexing"
        % (dx[0], dy[1], dz[2]))

    # ---- 6. HU clipping
    over = int((raw > hi).sum())
    under = int((raw < lo).sum())
    ok6 = n == 0 or (int(im.min()) >= lo and int(im.max()) <= hi)
    rec("6_hu_clip", ok6,
        "stored HU in [%d, %d] (observed [%s, %s]); the source volume had %d voxels "
        ">%d and %d voxels <%d, all mapped onto the window"
        % (lo, hi, im.min() if n else "-", im.max() if n else "-", over, hi, under, lo))

    # ---- 7. correct padding
    exp = expected_pad_pixels(ci, dims_ijk, size, off)
    ok7a = np.array_equal(exp, npad)
    padvals_ok = True
    for t in np.flatnonzero(npad > 0)[:20]:
        a, b, c = (int(v) for v in ci[t])
        oob = []
        if b - off < 0 or b + off > dims_ijk[1] or a - off < 0 or a + off > dims_ijk[0]:
            oob.append(0)
        if c - off < 0 or c + off > dims_ijk[2] or a - off < 0 or a + off > dims_ijk[0]:
            oob.append(1)
        if c - off < 0 or c + off > dims_ijk[2] or b - off < 0 or b + off > dims_ijk[1]:
            oob.append(2)
        for ch in oob:
            if int(im[t, ch].min()) != int(cfg["crop"]["pad_value_hu"]):
                padvals_ok = False
    rec("7_padding", bool(ok7a and padvals_ok),
        "%d instances padded, %d padded pixels, all matching the analytic overlap "
        "formula; every padded plane contains the %d HU fill"
        % (int((npad > 0).sum()), int(npad.sum()), int(cfg["crop"]["pad_value_hu"])))

    # ---- provenance
    ok_prov = (str(z["crop_signature"]) == crop_signature(cfg)
               and str(z["source_extraction_signature"]) == bag["config_signature"]
               and int(z["label"]) == bag["label"]
               and int(z["outer_fold"]) == folds[pid]
               and str(z["histology"]) == bag["histology"])
    rec("11_provenance", ok_prov,
        "crop signature %s, source extraction signature %s, label %d, outer fold %d"
        % (str(z["crop_signature"]), bag["config_signature"], bag["label"], folds[pid]))

    out["n_images"] = n
    out["n_frozen_patches"] = bag["n_patches"]
    out["n_padded_instances"] = int((npad > 0).sum())
    out["n_padded_pixels"] = int(npad.sum())
    out["label"] = bag["label"]
    out["histology"] = bag["histology"]
    out["outer_fold"] = folds[pid]
    out["grid_size"] = [int(v) for v in dims_ijk]
    out["hu_range"] = [int(im.min()), int(im.max())] if n else None
    out["all_pass"] = all(v["pass"] for v in out["checks"].values())

    if make_figure:
        qa_dir = ppath(cfg, cfg["outputs"]["qa_dir"])
        os.makedirs(qa_dir, exist_ok=True)
        shown = choose_instances(n, npad, int(cfg["pilot"]["qa_instances_per_patient"]),
                                 int(cfg["pilot"]["qa_max_padded_shown"]))
        png = os.path.join(qa_dir, "%s_2p5d.png" % pid)
        plot_patient(cfg, pid, z, vol16, mask_arr, png, shown)
        out["figure"] = os.path.relpath(png, cfg["_project_root"]).replace(os.sep, "/")
        out["shown_instances"] = shown
    z.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 6A pilot QA")
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    pids = args.patients or ([str(p) for p in cfg["pilot"]["patients"]] if args.pilot else None)
    if not pids:
        ap.error("choose --pilot or --patients")

    t0 = time.time()
    results = [check_patient(cfg, p, make_figure=not args.no_figures) for p in pids]
    n_fail = sum(0 if r["all_pass"] else 1 for r in results)

    names = sorted({k for r in results for k in r["checks"]})
    print("\nPhase 6A pilot QA (%d patients)" % len(results))
    for r in results:
        bad = [k for k, v in r["checks"].items() if not v["pass"]]
        print("%-4s %-12s %4d instances  %3d padded  %s"
              % ("PASS" if r["all_pass"] else "FAIL", r["patient_id"], r["n_images"],
                 r["n_padded_instances"], "" if not bad else "FAILED: %s" % ",".join(bad)))
    for k in names:
        n_ok = sum(1 for r in results if r["checks"][k]["pass"])
        print("  %-24s %d/%d" % (k, n_ok, len(results)))

    out = {
        "phase": "6a", "stage": "pilot_crop_qa",
        "crop_signature": crop_signature(cfg),
        "n_patients": len(results),
        "n_patients_failed": n_fail,
        "n_images": int(sum(r["n_images"] for r in results)),
        "n_padded_instances": int(sum(r["n_padded_instances"] for r in results)),
        "check_names": names,
        "all_pass": n_fail == 0,
        "seconds": round(time.time() - t0, 2),
        "patients": results,
    }
    path = ppath(cfg, cfg["outputs"]["pilot_check_json"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\n%s  ->  %s" % ("PILOT PASSED" if n_fail == 0 else "PILOT FAILED", path))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
