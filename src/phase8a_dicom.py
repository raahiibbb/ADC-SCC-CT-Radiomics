"""Phase 8A - DICOM CT series and DICOM SEG reading for the external cohort.

The algorithms here are the ones already validated on LUNG1 by the Vuong5ROI
project (`Vuong5ROI/src/vuong5roi/ct_io.py` and `seg_io.py`), reproduced so
that the external CT volumes and tumour masks are built the same way the LUNG1
CT.nii.gz / ROI_A_GTV.nii.gz products were built.  Nothing under `Vuong5ROI/`
is imported, executed or modified - it is a read-only reference.

CT
--
Slices are ordered by ImagePositionPatient projected on the true slice normal
(cross product of the two ImageOrientationPatient vectors), never by filename
or InstanceNumber.  A series with more than one orientation, more than one
in-plane size, more than one PixelSpacing, fewer than two slices or a
non-monotonic slice order is rejected rather than repaired.

SEG
---
The NSCLC-Radiogenomics segmentations are DICOM SEG objects already rasterised
on the referenced CT grid, so no resampling and no interpolation is performed.
Frames are mapped to CT slices primarily by the referenced source-image SOP
Instance UID and otherwise by ImagePositionPatient.  A frame that carries any
set pixel and cannot be mapped is an error, never a silent drop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pydicom
import SimpleITK as sitk


# ---------------------------------------------------------------------------
# CT
# ---------------------------------------------------------------------------

@dataclass
class CTSeries:
    image: sitk.Image
    files: List[str]
    sop_uids: List[str]
    z_positions: List[float]
    slice_spacings: List[float]
    header: dict
    warnings: List[str] = field(default_factory=list)


def read_ct(files, spacing_tol: float = 0.01, min_slices: int = 2) -> CTSeries:
    """Read one CT series into a sitk image, sorted along the slice normal."""
    warnings = []
    hdrs = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        hdrs.append((f, ds))
    if len(hdrs) < int(min_slices):
        raise ValueError("CT series has %d slices (< %d)" % (len(hdrs), min_slices))

    mods = {str(getattr(ds, "Modality", "")) for _, ds in hdrs}
    if mods != {"CT"}:
        raise ValueError("series modality is %s, expected CT" % sorted(mods))

    orients = {tuple(round(float(v), 6) for v in ds.ImageOrientationPatient)
               for _, ds in hdrs}
    if len(orients) != 1:
        raise ValueError("CT series has %d distinct ImageOrientationPatient values"
                         % len(orients))
    iop = np.asarray(next(iter(orients)), dtype=float)
    normal = np.cross(iop[:3], iop[3:])

    shapes = {(int(ds.Rows), int(ds.Columns)) for _, ds in hdrs}
    if len(shapes) != 1:
        raise ValueError("CT series has inconsistent in-plane sizes: %s" % sorted(shapes))

    px = {tuple(round(float(v), 6) for v in ds.PixelSpacing) for _, ds in hdrs}
    if len(px) != 1:
        raise ValueError("CT series has inconsistent PixelSpacing: %s" % sorted(px))

    keyed = []
    for f, ds in hdrs:
        ipp = np.asarray([float(v) for v in ds.ImagePositionPatient])
        keyed.append((float(np.dot(ipp, normal)), f, str(ds.SOPInstanceUID), float(ipp[2])))
    keyed.sort(key=lambda t: t[0])

    proj = np.asarray([k[0] for k in keyed])
    d = np.diff(proj)
    if np.any(d <= 0):
        raise ValueError("duplicate or non-monotonic CT slice positions")
    if float(d.max() - d.min()) > spacing_tol:
        warnings.append("non-uniform CT slice spacing: min=%.4f max=%.4f mm"
                        % (d.min(), d.max()))

    ordered = [k[1] for k in keyed]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(ordered)
    img = sitk.Cast(reader.Execute(), sitk.sitkInt16)

    itk_dz = float(img.GetSpacing()[2])
    hdr_dz = float(np.median(d))
    if abs(itk_dz - hdr_dz) > spacing_tol:
        warnings.append("ITK slice spacing %.4f mm differs from header-derived %.4f mm"
                        % (itk_dz, hdr_dz))

    ref = hdrs[0][1]
    header = {
        "SeriesInstanceUID": str(getattr(ref, "SeriesInstanceUID", "")),
        "StudyInstanceUID": str(getattr(ref, "StudyInstanceUID", "")),
        "n_slices": len(ordered),
        "rows": int(ref.Rows), "columns": int(ref.Columns),
        "pixel_spacing_mm": [float(v) for v in ref.PixelSpacing],
        "slice_spacing_mm_median": hdr_dz,
        "slice_spacing_mm_min": float(d.min()),
        "slice_spacing_mm_max": float(d.max()),
        "itk_spacing_mm": [float(v) for v in img.GetSpacing()],
        "itk_size": [int(v) for v in img.GetSize()],
        "itk_origin_mm": [float(v) for v in img.GetOrigin()],
        "itk_direction": [float(v) for v in img.GetDirection()],
    }
    return CTSeries(image=img, files=ordered, sop_uids=[k[2] for k in keyed],
                    z_positions=[k[3] for k in keyed],
                    slice_spacings=[float(v) for v in d],
                    header=header, warnings=warnings)


def acquisition_tags(dicom_path, tag_names):
    """Read the configured descriptive acquisition tags from one DICOM file."""
    ds = pydicom.dcmread(dicom_path, stop_before_pixels=True)
    out = {}
    for name in tag_names:
        v = getattr(ds, name, None)
        if v is None:
            out[name] = ""
        elif isinstance(v, (pydicom.multival.MultiValue, list, tuple)):
            out[name] = "\\".join(str(x) for x in v)
        else:
            out[name] = str(v)
    return out


# ---------------------------------------------------------------------------
# SEG
# ---------------------------------------------------------------------------

@dataclass
class SegmentInfo:
    number: int
    label: str
    description: str
    property_category: str
    property_type: str
    algorithm: str


def _code_meaning(ds, name):
    seq = getattr(ds, name, None)
    if seq:
        return str(getattr(seq[0], "CodeMeaning", ""))
    return ""


def read_seg(seg_path, ct: CTSeries):
    """Decode a DICOM SEG onto the CT grid.

    Returns (segments, masks, info) where masks[segment_number] is a uint8
    sitk image sharing the CT geometry exactly.
    """
    ds = pydicom.dcmread(seg_path)
    info = {"warnings": [], "unmapped_frames": 0, "unmapped_nonempty_frames": 0}

    segments = {}
    for s in ds.SegmentSequence:
        n = int(s.SegmentNumber)
        segments[n] = SegmentInfo(
            number=n,
            label=str(getattr(s, "SegmentLabel", "")),
            description=str(getattr(s, "SegmentDescription", "")),
            property_category=_code_meaning(s, "SegmentedPropertyCategoryCodeSequence"),
            property_type=_code_meaning(s, "SegmentedPropertyTypeCodeSequence"),
            algorithm=str(getattr(s, "SegmentAlgorithmType", "")),
        )

    ref = ct.image
    size = ref.GetSize()                      # (x, y, z)
    nz = len(ct.sop_uids)
    ny, nx = int(ds.Rows), int(ds.Columns)
    info["seg_rows"] = ny
    info["seg_columns"] = nx
    info["ct_rows"] = int(size[1])
    info["ct_columns"] = int(size[0])
    if (nx, ny) != (int(size[0]), int(size[1])):
        raise ValueError("SEG in-plane size (%d, %d) != CT (%d, %d)"
                         % (nx, ny, size[0], size[1]))

    frames = ds.pixel_array
    if frames.ndim == 2:
        frames = frames[None, ...]
    n_frames = int(getattr(ds, "NumberOfFrames", frames.shape[0]))
    if frames.shape[0] != n_frames:
        info["warnings"].append("pixel_array frames %d != NumberOfFrames %d"
                                % (frames.shape[0], n_frames))
    info["n_frames"] = int(frames.shape[0])

    uid_to_z = {u: i for i, u in enumerate(ct.sop_uids)}
    z_to_index = {round(z, 3): i for i, z in enumerate(ct.z_positions)}

    vols = {n: np.zeros((nz, ny, nx), dtype=np.uint8) for n in segments}
    mapped_by = {"sop_uid": 0, "position": 0}
    info["frames_without_segment_identification"] = 0
    info["unassigned_frames"] = 0
    info["unassigned_nonempty_frames"] = 0

    pf = ds.PerFrameFunctionalGroupsSequence
    for fi in range(min(len(pf), frames.shape[0])):
        g = pf[fi]
        try:
            seg_no = int(g.SegmentIdentificationSequence[0].ReferencedSegmentNumber)
        except Exception:                                          # noqa: BLE001
            # SegmentIdentificationSequence is only strictly required when a SEG
            # carries more than one segment.  The NSCLC-Radiogenomics objects
            # omit it and hold exactly one segment, so the frame is that
            # segment's by construction.  With two or more segments an
            # unidentified frame is recorded, never guessed at.
            info["frames_without_segment_identification"] += 1
            seg_no = sorted(segments)[0] if len(segments) == 1 else None
        if seg_no not in vols:
            info["unassigned_frames"] += 1
            if frames[fi].any():
                info["unassigned_nonempty_frames"] += 1
            continue

        zi, how = None, None
        try:
            src = g.DerivationImageSequence[0].SourceImageSequence[0]
            zi = uid_to_z.get(str(src.ReferencedSOPInstanceUID))
            how = "sop_uid"
        except Exception:                                          # noqa: BLE001
            zi = None
        if zi is None:
            try:
                ipp = g.PlanePositionSequence[0].ImagePositionPatient
                zi = z_to_index.get(round(float(ipp[2]), 3))
                how = "position"
            except Exception:                                      # noqa: BLE001
                zi = None

        f = frames[fi]
        if zi is None:
            info["unmapped_frames"] += 1
            if f.any():
                info["unmapped_nonempty_frames"] += 1
            continue
        mapped_by[how] += 1
        if f.any():
            vols[seg_no][zi] |= (f > 0).astype(np.uint8)

    info["frames_mapped_by_sop_uid"] = mapped_by["sop_uid"]
    info["frames_mapped_by_position"] = mapped_by["position"]

    masks = {}
    for n, arr in vols.items():
        m = sitk.GetImageFromArray(arr)
        m.CopyInformation(ref)
        masks[n] = sitk.Cast(m, sitk.sitkUInt8)

    info["referenced_series"] = referenced_ct_series(ds)
    info["seg_sop_instance_uid"] = str(getattr(ds, "SOPInstanceUID", ""))
    info["seg_series_instance_uid"] = str(getattr(ds, "SeriesInstanceUID", ""))
    info["seg_manufacturer"] = str(getattr(ds, "Manufacturer", ""))
    info["seg_software"] = str(getattr(ds, "SoftwareVersions", ""))
    info["seg_series_description"] = str(getattr(ds, "SeriesDescription", ""))
    return segments, masks, info


def referenced_ct_series(ds) -> List[str]:
    """Every SeriesInstanceUID the SEG declares as its source image series."""
    out = []
    for r in getattr(ds, "ReferencedSeriesSequence", []) or []:
        uid = str(getattr(r, "SeriesInstanceUID", "")).strip()
        if uid:
            out.append(uid)
    return sorted(set(out))


def referenced_series_of_zip_member(seg_dcm_path) -> List[str]:
    """Read only the referenced-series UIDs (no pixel data) from a SEG file."""
    ds = pydicom.dcmread(seg_dcm_path, stop_before_pixels=True)
    return referenced_ct_series(ds)


def segment_summary(seg_dcm_path) -> Dict[int, dict]:
    """Segment identification without decoding any pixel data."""
    ds = pydicom.dcmread(seg_dcm_path, stop_before_pixels=True)
    out = {}
    for s in getattr(ds, "SegmentSequence", []) or []:
        n = int(s.SegmentNumber)
        out[n] = {
            "label": str(getattr(s, "SegmentLabel", "")),
            "description": str(getattr(s, "SegmentDescription", "")),
            "property_category": _code_meaning(s, "SegmentedPropertyCategoryCodeSequence"),
            "property_type": _code_meaning(s, "SegmentedPropertyTypeCodeSequence"),
            "algorithm": str(getattr(s, "SegmentAlgorithmType", "")),
        }
    return out
