# Phase 12 report: fourth training site (NLST) and a locked TCGA test set

**Date:** 2026-09-26. User-authorised ("go with 55gb, start phase 12").
**Plan:** `docs/ROADMAP_PHASE12_14.md`. **Config:** `config/phase12.yaml`.

## 1. Summary

| | Phase 11 (3 sites, 644) | **Phase 12 (4 sites, 1041)** |
|---|---|---|
| Primary: pooled site-balanced AUC [95% CI] | 0.718 [0.669, 0.767] | **0.714 [0.677, 0.756]** |
| Primary: leave-one-site-out mean | 0.678 | **0.704** |
| Held out: LPCD | 0.735 | **0.839** |
| Held out: Radiogenomics | 0.674 | 0.665 |
| Held out: LUNG1 | 0.625 | 0.626 |
| Held out: NLST | n/a | 0.688 |
| Best fusion (post hoc) | 0.722 | 0.719 (gtv+shells+loc_clin) |

The primary model (shells + loc_clin, ComBat) was frozen in Phases 10–11, before any NLST data existed.

- **Transport improved.** Adding a fourth site raised the held-out-hospital mean from 0.678 to 0.704. It raised the held-out LPCD score by +0.10.
- **The pooled score did not move** (0.714 against 0.718). Its confidence interval narrowed, from 0.098 wide to 0.079.
- **The 0.75 target was not reached.** NLST is not an "easy" site: its within-site AUC is 0.69, compared with 0.86 for LPCD.

## 2. Data

| Site | ADC | SCC | Source of the tumour location |
|---|---|---|---|
| LUNG1 | 51 | 151 | tight boxes from the expert GTV |
| Radiogenomics | 112 | 29 | tight boxes from the expert GTV |
| LPCD | 244 | 57 | PASCAL-VOC boxes |
| **NLST (new)** | **290** | **107** | **expert NLST-Sybil boxes** (Zenodo 10.5281/zenodo.15643335, IDC v23) |
| TCGA-LUAD/LUSC (**LOCKED**) | – | – | none yet; 85 CT patients converted, labels not read |

### NLST selection (`src/phase12_select.py`)

- 581 patients have Sybil boxes. Of these:
  - 155 were excluded because the first primary is not ADC or SCC (ICD-O-3, `lc_morph`);
  - 29 were excluded because they have more than one lung primary.
- **397 patients were kept.**
- For each patient, one series was used: the latest screening time point that has boxes. Ties were broken by the number of annotated slices.
- Download: 36.1 GB from IDC (s5cmd), with 0 errors.
- Scanners: GE 213, Siemens 131, Toshiba 31, Philips 22. The median slice spacing is 2.0 mm.
- 378 of 397 patients have a single annotated lesion. The largest lesion is used.

### TCGA (`src/phase12_select.py`, `src/phase12_build.py tcga`)

- Selection: the first study, the axial CT with the most images, at least 60 images.
- **85 of 104 patients were kept.** The other 19 have only 5–10 mm or PET-attenuation CT.
- Download: 7.9 GB.
- All 85 converted to NIfTI. The median slice spacing is 5.0 mm (range 0.6–5.0).
- The labels (= collection) are in `C:/LUNG_phase12/tcga_LOCKED_labels.csv` and have not been read by any analysis.
- **There is no tumour location yet.** This must be solved before the one-time Phase-14 test.

The total download was 44 GB, to `C:/LUNG_phase12`, as approved.

## 3. Segmentation change (v5, applied to all sites)

The QA gallery showed that Sybil boxes are loose. The AUTO/BOX volume ratio has a median of 0.35, compared with 0.63–0.73 at the other sites. The frozen Phase-11 rule (HU > −750 inside the box) therefore absorbed chest wall and mediastinum.

Variants were validated on the 343 expert GTVs (LUNG1 + Radiogenomics). The "loose" setting uses the tight boxes grown by 8 mm, to mimic Sybil. No ADC/SCC label was used. The table shows mean Dice for LUNG1 / Radiogenomics.

| Variant | Tight boxes | Loose boxes |
|---|---|---|
| v1 (Phase 11) | 0.818 / 0.824 | 0.553 / 0.594 |
| v2: box ∩ lung envelope (air mask closed 10 mm) | 0.393 / 0.722 | rejected |
| v3: envelope decides between per-slice components | 0.733 / 0.755 | 0.421 / 0.539 |
| v4: box ∩ chest-cavity hull | 0.773 / 0.830 | 0.550 / 0.680 |
| **v5: v4 only on slices where ≥ 50 % of the box is inside the hull** | **0.812 / 0.830** | **0.572 / 0.680** |

- **Why v2 and v3 fail:** an air-based lung mask misses large central tumours and collapsed lung. In LUNG1 only a median 26 % of the GTV lies inside it.
- **v5 was adopted and applied to all 1041 patients** (`C:/LUNG_phase12/auto_masks_v2`, 0 failures). All features were re-extracted from v5, so the mask source remains identical at every site.
- **Residual limitation:** some NLST juxtapleural nodules still include chest wall; see `results/phase12/figures/nlst_autoseg_gallery.png`. A box-prompted segmentation foundation model is the proper fix and is proposed for Phase 13.

## 4. Features

Extractors are unchanged from Phase 11: PyRadiomics (gtv, ring_in, shells; original image only), loc, FMCIB centroid embedding, and age/sex. For NLST, age at the scan is age at randomisation plus the screening year.

All 1041 patients were extracted with 0 failures:
- radiomics: 20 min;
- FMCIB: 20 min;
- loc: 13 min.

## 5. Results (ComBat; 5 seeds × 5-fold; site-balanced AUC)

| Block | Pooled | LUNG1 | RG | LPCD | NLST | LOSO mean | Site-leak AUC |
|---|---|---|---|---|---|---|---|
| gtv | 0.693 | 0.560 | 0.643 | 0.860 | 0.683 | 0.682 | 0.574 |
| ring_in | 0.683 | 0.581 | 0.638 | 0.839 | 0.667 | 0.674 | 0.587 |
| shells | 0.689 | 0.641 | 0.635 | 0.833 | 0.650 | 0.671 | 0.602 |
| loc_clin | 0.707 | 0.585 | 0.699 | 0.844 | 0.700 | 0.702 | 0.545 |
| fmcib | 0.671 | 0.573 | 0.641 | 0.788 | 0.665 | 0.656 | 0.266 |
| **PRIMARY shells+loc_clin** | **0.714** | 0.625 | 0.679 | 0.863 | 0.693 | **0.704** | – |

- **All 31 fusions:** pooled 0.671–0.719, median 0.705; LOSO median 0.696.
- **Threshold metrics (primary, threshold at score 0), sensitivity / specificity:**
  - LUNG1: 0.549 / 0.623;
  - Radiogenomics: 0.688 / 0.655;
  - LPCD: 0.574 / 0.877;
  - NLST: 0.617 / 0.598.
- **Shortcut demonstration on 4 sites:**
  - hospital-only predictor: raw pooled AUC 0.710, site-balanced 0.497;
  - naive radiomics model: raw 0.727, honest 0.642.
- **Acquisition shortcut within NLST** (checked after the report was first written). Each value is the AUC of that acquisition variable alone for predicting the label:
  - kernel (13 levels): ≤ 0.597, an in-sample, optimistic upper bound;
  - manufacturer: ≤ 0.555;
  - slice spacing: 0.537;
  - screening year: 0.543.

  All of these are weak. Tumour volume reaches 0.640, which reflects biology: SCC tumours are larger.
- **Ablation: Phase-11 sites with v5 masks** (`--subset three_site`, run after the report was first written). This separates the effect of the mask change from the effect of adding NLST.

  | Configuration | Primary pooled | LOSO mean | Held-out LPCD |
  |---|---|---|---|
  | v1 masks, 3 sites (Phase 11) | 0.718 | 0.678 | 0.735 |
  | v5 masks, 3 sites | 0.716 | 0.682 | 0.747 |
  | v5 masks, + NLST | 0.714 | 0.704 | 0.839 |

  The mask change is neutral. **The transport gain comes from adding NLST.**
- **Sensitivity check, NLST single-lesion only (n = 1022):** primary pooled 0.716 [0.674, 0.758], LOSO mean 0.707. The result is unchanged.

## 6. Interpretation

1. **More sites mainly improve transport.** The largest gain is on the held-out LPCD (+0.10).
2. **NLST sits in the middle of the other sites** (about 0.69). Its tumours are small (median 4.3 ml, against 57.5 ml for LUNG1) and often sub-solid. This is not another "easy" cohort, so the pooled score is still about 0.71.
3. **Tumour position is still the single most transportable block** (LOSO 0.702). FMCIB is still below handcrafted features.
4. **LUNG1 remains around 0.62.** This is the main ceiling.

## 7. Files

- **Code:** `src/phase12_select.py`, `src/phase12_build.py`, `src/phase12_autoseg.py`, `src/phase12_features.py`, `src/phase12_cv.py`, `src/phase12_qa.py`, `src/phase12_figures.py`.
- **Data:**
  - `C:/LUNG_phase12/` holds the DICOM, NIfTI, `auto_masks_v2`, the manifests and the logs;
  - `cohort/phase12_four_site_cohort.csv`;
  - `global_features/phase12/`.
- **Results:** `results/phase12/`:
  - `pooled/`, `loso/`, `fusion/`;
  - `subset_single_lesion/`;
  - `autoseg_v2_validation.csv`, `nlst_autoseg_qc.csv`;
  - `phase12_summary.json`;
  - `figures/fig1–4`, `figures/nlst_autoseg_gallery.png`.
- **Environment:** `reports/phase12_environment.txt`.

## 8. Commands

```
.venv-phase10/Scripts/python src/phase12_select.py
s5cmd --no-sign-request --numworkers 32 run C:/LUNG_phase12/download_all.txt   (cwd C:/LUNG_phase12)
.venv-phase10/Scripts/python src/phase12_build.py nlst|tcga
.venv-phase10/Scripts/python src/phase12_features.py cohort
.venv-phase10/Scripts/python src/phase12_autoseg.py validate|run
.venv/Scripts/python src/phase12_features.py radiomics ; .venv-phase10/Scripts/python src/phase12_features.py loc|fmcib
.venv-phase10/Scripts/python src/phase12_cv.py pooled|loso|fusion [--subset single_lesion]
.venv-phase10/Scripts/python src/phase12_qa.py ; .venv-phase10/Scripts/python src/phase12_figures.py
```

## 9. Warnings

- **s5cmd on Windows:** a destination given as an absolute path on another drive fails to rename. Run it with cwd `C:/LUNG_phase12` and relative destinations.
- **Segmentation changed:** the v5 masks replace the Phase-11 masks for the Phase-12 analyses. Phase-11 results remain valid for Phase 11 and were not overwritten.
- **TCGA slices:** some TCGA series have non-uniform slice spacing (ITK warning). Check this before Phase 14.
- **TCGA labels:** they equal the collection. At test time, report per tissue-source-site where both classes exist.

## 10. Next recommended action (not started)

Phase 13, in this order:
1. **Box-prompted foundation-model segmentation** (for example MedSAM2) for all sites, validated against the expert GTVs.
2. **A bank of foundation-model embeddings** (CT-FM and others) on identical crops.
3. **Site-adversarial fine-tuning** of the best backbone.

Before Phase 14, TCGA needs tumour localisation.
