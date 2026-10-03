"""Non-overlapping 5x5x5 patch enumeration on the resampled grid.

The grid is anchored at resampled voxel index (0, 0, 0), not at the ROI bounding
box, so that the patch lattice of a given patient is identical for every ROI.
That keeps the later five-ROI comparison spatially aligned.

All index tuples in this module are (i, j, k) in SimpleITK order, i.e. the
numpy array is indexed [k, j, i].
"""
from __future__ import annotations

import numpy as np


def enumerate_patches(mask_arr: np.ndarray, size, stride, require_full_in_bounds=True):
    """Yield candidate blocks whose extent intersects the ROI bounding box.

    `mask_arr` is the numpy array (k, j, i) of the resampled mask.
    Returns a list of dicts with `start` and `center` in (i, j, k) order.
    """
    size = np.asarray(size, dtype=int)          # (i, j, k)
    stride = np.asarray(stride, dtype=int)
    dims_ijk = np.asarray(mask_arr.shape[::-1], dtype=int)

    nz = np.argwhere(mask_arr > 0)
    if nz.size == 0:
        return [], None
    bb_min_kji, bb_max_kji = nz.min(0), nz.max(0)
    bb_min = bb_min_kji[::-1]                   # (i, j, k)
    bb_max = bb_max_kji[::-1]

    # first lattice start <= bb_min, and last lattice start <= bb_max
    first = (bb_min // stride) * stride
    ranges = []
    for d in range(3):
        limit = dims_ijk[d] - size[d] if require_full_in_bounds else dims_ijk[d] - 1
        stops = np.arange(first[d], min(bb_max[d], limit) + 1, stride[d])
        ranges.append(stops[stops >= 0])

    out = []
    half = size // 2
    for i0 in ranges[0]:
        for j0 in ranges[1]:
            for k0 in ranges[2]:
                start = np.array([i0, j0, k0], dtype=int)
                out.append({"start": start, "center": start + half})
    bbox = {"min_ijk": bb_min.astype(int), "max_ijk": bb_max.astype(int)}
    return out, bbox


def patch_roi_stats(mask_arr: np.ndarray, start, size):
    """(n_roi_voxels, n_total_voxels) inside the block."""
    i0, j0, k0 = (int(v) for v in start)
    si, sj, sk = (int(v) for v in size)
    sub = mask_arr[k0:k0 + sk, j0:j0 + sj, i0:i0 + si]
    return int((sub > 0).sum()), int(sub.size)


def crop_block(image, start, size):
    """SimpleITK region crop, (i, j, k) order."""
    import SimpleITK as sitk
    return sitk.RegionOfInterest(image, [int(v) for v in size],
                                 [int(v) for v in start])
