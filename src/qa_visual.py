"""Visual QA: retained patch footprints overlaid on the resampled CT + ROI.

Independence note: the overlay does NOT reuse the coordinates as they were
computed during extraction.  It re-derives the resampled grid from the CT and
maps `coords_world` from the saved bag back through
TransformPhysicalPointToIndex.  If the saved world coordinates and the saved
index coordinates disagree, the figure is wrong and the printed round-trip
error is non-zero -- so the figure doubles as a coordinate-correctness test.
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import SimpleITK as sitk                              # noqa: E402
from matplotlib.patches import Rectangle              # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cohort import load_cohort                        # noqa: E402
from config_io import (DEFAULT_CONFIG, cfg_path, load_config,   # noqa: E402
                       project_path)
from resample import resample_ct_and_mask             # noqa: E402

WL, WW = -600.0, 1600.0        # lung window


def make_overlay(cfg, pid, out_dir, n_slices=3):
    feat_dir = project_path(cfg, *cfg["paths"]["features_dir"].split("/"))
    with np.load(os.path.join(feat_dir, "%s.npz" % pid), allow_pickle=True) as z:
        coords_idx = z["coords_index"]
        coords_start = z["coords_start_index"]
        coords_world = z["coords_world"]
        roi_frac = z["roi_fraction"]
        size = z["patch_size"]
        label = int(z["label"])
        hist = str(z["histology"])
        grid_origin = z["grid_origin"]
        grid_spacing = z["grid_spacing"]

    pdir = os.path.join(cfg["paths"]["vuong_root"], pid)
    ct = sitk.ReadImage(os.path.join(pdir, cfg["paths"]["ct_filename"]))
    mk = sitk.ReadImage(os.path.join(pdir, cfg["experiment"]["roi_filename"]))
    ct_r, mask_r, grid = resample_ct_and_mask(ct, mk, cfg)
    ct_a = sitk.GetArrayFromImage(ct_r)
    mk_a = sitk.GetArrayFromImage(mask_r)

    # --- independent coordinate round-trip -------------------------------
    grid_ok = (np.allclose(grid["origin"], grid_origin)
               and np.allclose(grid["spacing"], grid_spacing))
    back = np.array([ct_r.TransformPhysicalPointToIndex([float(x) for x in p])
                     for p in coords_world], dtype=np.int64)
    roundtrip_err = int(np.abs(back - coords_idx.astype(np.int64)).max()) if len(back) else 0
    centers_in_roi = int(sum(1 for c in coords_idx if mk_a[c[2], c[1], c[0]] > 0))

    # --- pick the axial slices carrying the most retained patch centres ---
    if len(coords_idx) == 0:
        return None
    zs, counts = np.unique(coords_idx[:, 2], return_counts=True)
    top = zs[np.argsort(-counts)][:n_slices]
    top = sorted(top.tolist())

    lo, hi = WL - WW / 2.0, WL + WW / 2.0
    fig, axes = plt.subplots(1, len(top), figsize=(5.2 * len(top), 5.6))
    axes = np.atleast_1d(axes)
    for ax, zc in zip(axes, top):
        sel = coords_idx[:, 2] == zc
        ii = coords_idx[sel][:, 0]
        jj = coords_idx[sel][:, 1]
        pad = 28
        i0, i1 = max(0, ii.min() - pad), min(ct_a.shape[2], ii.max() + pad)
        j0, j1 = max(0, jj.min() - pad), min(ct_a.shape[1], jj.max() + pad)
        ax.imshow(ct_a[zc, j0:j1, i0:i1], cmap="gray", vmin=lo, vmax=hi,
                  origin="upper", interpolation="nearest")
        roi = np.ma.masked_where(mk_a[zc, j0:j1, i0:i1] == 0,
                                 mk_a[zc, j0:j1, i0:i1])
        ax.imshow(roi, cmap="autumn", alpha=0.28, origin="upper",
                  interpolation="nearest")
        ax.contour(mk_a[zc, j0:j1, i0:i1], levels=[0.5], colors="orange",
                   linewidths=0.8)
        for s, c, f in zip(coords_start[sel], coords_idx[sel], roi_frac[sel]):
            ax.add_patch(Rectangle((s[0] - 0.5 - i0, s[1] - 0.5 - j0),
                                   size[0], size[1], fill=False,
                                   edgecolor="#2ee6ff", linewidth=0.9,
                                   alpha=0.35 + 0.65 * float(f)))
            ax.plot(c[0] - i0, c[1] - j0, marker=".", color="#2ee6ff", markersize=2.5)
        ax.set_title("k = %d   (%d patch centres)" % (zc, int(sel.sum())), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("%s  |  %s (label %d)  |  ROI %s  |  %d retained %dx%dx%d patches "
                 "@ %g mm  |  world->index round-trip max err = %d voxel"
                 % (pid, hist, label, cfg["experiment"]["roi"], len(coords_idx),
                    size[0], size[1], size[2],
                    cfg["preprocessing"]["resample_spacing_mm"][0], roundtrip_err),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "%s_patches.png" % pid)
    fig.savefig(path, dpi=130)
    plt.close(fig)

    return {"patient_id": pid, "roi": cfg["experiment"]["roi"],
            "figure": path, "n_patches": int(len(coords_idx)),
            "grid_reproduced": bool(grid_ok),
            "world_to_index_roundtrip_max_error_voxels": roundtrip_err,
            "centers_inside_roi": centers_in_roi,
            "all_centers_inside_roi": centers_in_roi == len(coords_idx)}


def main():
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--patients", nargs="*", default=None)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.pilot or not args.patients:
        pf = cfg_path(cfg, "pilot_file")
        ids = [l.split("#")[0].strip() for l in open(pf, encoding="utf-8")]
        ids = [i for i in ids if i]
    else:
        ids = args.patients

    out_dir = cfg_path(cfg, "qa_patches_dir")
    results = []
    for pid in ids:
        r = make_overlay(cfg, pid, out_dir)
        results.append(r)
        print(pid, "->", r["figure"], "| roundtrip err",
              r["world_to_index_roundtrip_max_error_voxels"],
              "| all centres in ROI:", r["all_centers_inside_roi"])
    with open(os.path.join(out_dir, "visual_qa_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
