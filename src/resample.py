"""2 mm isotropic resampling of CT (linear) and mask (nearest neighbour).

CT and mask are resampled onto one shared reference grid derived from the CT, so
the two resampled volumes are voxel-for-voxel aligned by construction.
"""
from __future__ import annotations

import math

import SimpleITK as sitk

_INTERP = {
    "linear": sitk.sitkLinear,
    "nearest": sitk.sitkNearestNeighbor,
    "bspline": sitk.sitkBSpline,
}


def reference_grid(image: sitk.Image, spacing_mm):
    """Grid covering the same physical extent as `image` at the target spacing."""
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    new_spacing = [float(s) for s in spacing_mm]
    new_size = [
        int(math.ceil(old_size[i] * old_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]
    return {
        "size": new_size,
        "spacing": new_spacing,
        "origin": image.GetOrigin(),
        "direction": image.GetDirection(),
    }


def resample_to_grid(image: sitk.Image, grid: dict, interpolator: str,
                     default_value: float) -> sitk.Image:
    rs = sitk.ResampleImageFilter()
    rs.SetSize([int(v) for v in grid["size"]])
    rs.SetOutputSpacing([float(v) for v in grid["spacing"]])
    rs.SetOutputOrigin(grid["origin"])
    rs.SetOutputDirection(grid["direction"])
    rs.SetTransform(sitk.Transform())
    rs.SetInterpolator(_INTERP[interpolator])
    rs.SetDefaultPixelValue(float(default_value))
    return rs.Execute(image)


def resample_ct_and_mask(ct: sitk.Image, mask: sitk.Image, cfg: dict):
    pre = cfg["preprocessing"]
    grid = reference_grid(ct, pre["resample_spacing_mm"])
    ct_r = resample_to_grid(sitk.Cast(ct, sitk.sitkFloat32), grid,
                            pre["ct_interpolator"], pre["ct_default_value"])
    mask_r = resample_to_grid(sitk.Cast(mask, sitk.sitkUInt8), grid,
                              pre["mask_interpolator"], 0)
    mask_r = sitk.Cast(mask_r > 0, sitk.sitkUInt8)
    return ct_r, mask_r, grid
