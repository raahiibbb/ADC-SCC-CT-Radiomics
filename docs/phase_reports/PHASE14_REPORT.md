# Phase 14 report: locked external test on TCGA-LUAD/LUSC

**Date:** 2026-09-27. User-authorised ("go for B": automatic tumour localisation).

**Pre-registration:** `docs/PHASE14_PREREGISTRATION.md`, written before any TCGA mask, prediction or label existed.

**Prediction freeze:** `results/phase14/tcga_predictions.csv`, sha256 `97feb8c67274a0c9844e217c45ae5627cc9346c8b9e33e13a794dc44c94f50e6`. It was written and hashed before `tcga_LOCKED_labels.csv` was first opened. The labels were read exactly once, by `phase14_test.py score`, which refuses to rescore.

## 1. Headline (pre-registered, reported unconditionally)

| Model | TCGA ROC-AUC [95% CI] | Sensitivity | Specificity | Balanced accuracy |
|---|---|---|---|---|
| **PRIMARY shells + loc_clin** | **0.585 [0.462, 0.707]** | 0.562 | 0.583 | 0.573 |
| Secondary gtv + shells + loc_clin | 0.568 [0.448, 0.689] | 0.562 | 0.583 | 0.573 |

- **Patients:** n = 84 (48 ADC / 36 SCC). 85 were selected; 1 was excluded because no nodule was detected, which is the pre-registered rule.
- **Missing features:** 3 patients had a feature failure and were imputed with the training medians:
  - 1 had a mask too small for radiomics;
  - 2 had non-axis-aligned CT directions, so the loc features could not be computed.
- **The site-stratified AUC could not be computed.** None of the 9 TCGA tissue-source sites contributed both classes. The collection, and hence the label, is **perfectly confounded with the contributing institution**. This is the same shortcut structure the project was built to expose. The pooled TCGA AUC therefore partly measures institution differences, and the harmonisation deliberately suppresses those.

## 2. Pipeline (fully automatic, frozen)

1. **Detection:** TotalSegmentator `lung_nodules` (v2.18.0, Apache-2.0, BLUEMIND AI), in `.venv-totalseg` (torch 2.5.1+cu124). The largest 3-D component is taken, and its per-slice boxes are used. It took 3.2 h on an RTX 3050.
2. **Segmentation:** the frozen Phase-13 MedSAM2 pipeline, with identical settings.
3. **Features:** the same extractors. Age/sex come from the DICOM headers (age missing for 4 patients, imputed).
4. **Model:** fit once on all 1041 development patients (ComBat, inner-CV hyper-parameters, seed 42). TCGA was mapped as one new site with label-free location/scale.

**Detector QA** on 60 development patients with expert GTVs (no histology used):

| | LUNG1 | Radiogenomics |
|---|---|---|
| Nodule detected | 26/30 | 29/30 |
| Largest component hits the GTV | 19/26 (73%) | 25/29 (86%) |
| Final MedSAM2 mask Dice vs GTV (median / mean) | 0.82 / 0.59 | 0.71 / 0.58 |

About 1 in 5 localisations is expected to target the wrong lesion.

## 3. Post-hoc diagnostics (NOT pre-registered; explanatory only)

| Subset / block | n | AUC |
|---|---|---|
| Slice spacing ≤ 2.5 mm | 29 (13 ADC) | **0.726** |
| Slice spacing > 2.5 mm (mostly 5 mm) | 55 (35 ADC) | 0.511 |
| Detector found a single component | 30 | 0.545 |
| Block: loc_clin | 84 | 0.633 |
| Block: gtv | 84 | 0.549 |
| Block: shells | 84 | 0.513 |
| Tumour volume alone | 84 | 0.635 (median ADC 7.8 ml vs SCC 32.1 ml) |

**Interpretation:**
1. **Thick slices are the dominant failure mode.** Two-thirds of TCGA is 5 mm, whereas the development sites are mostly ≤ 3 mm. On thin-slice TCGA scans the model scores 0.73, in line with the development LOSO results (0.63–0.84). On 5 mm scans the texture features (shells, gtv) carry no signal, while position, size, age and sex partially survive.
2. **Automatic localisation errors** (about 20% expected) add noise.
3. **The small n** gives a CI width of ±0.12, and the complete TSS–label confound further limits interpretation.
4. **Caution:** the thin/thick split is itself partly aligned with the TSS, so these subsets are not clean experiments.

## 4. Conclusion for the thesis

The pre-registered locked test did **not** replicate the cross-validated performance: AUC 0.585 against a development LOSO mean of 0.714. The diagnostics point to an **acquisition-transport limit**. Handcrafted texture learned on ≤ 3 mm CT does not transfer to 5 mm CT. The model is therefore reliable on thin-slice CT (TCGA ≤ 2.5 mm: 0.73) and must not be used on thick-slice scans.

This is a genuine, reportable external-validation finding. It also names the concrete next steps:
- a slice-thickness-robust model, trained with thick-slice simulation or multi-resolution features;
- a larger multi-thickness external cohort.

## 5. Files

- **Code:** `src/phase14_detect.py`, `src/phase14_segment.py`, `src/phase14_test.py`. `src/phase12_features.py` gained the `ADC_COHORT` env override.
- **Data:**
  - `C:/LUNG_phase14/detect/` (TotalSegmentator outputs, BOX, `detect_manifest.csv`);
  - `C:/LUNG_phase14/medsam2_masks/`;
  - `cohort/phase14_tcga_cohort.csv` (no labels);
  - `global_features/phase14/`.
- **Results:** `results/phase14/`:
  - `tcga_predictions.csv` (frozen, hashed);
  - `tcga_predictions_frozen.json`;
  - `tcga_result.json`;
  - `tcga_scored.csv`.
- **Environment:** `.venv-totalseg`.

---

# Phase 14B: second locked external test, Lung3 (NSCLC-Radiomics-Genomics)

**Pre-registration:** `docs/PHASE14B_LUNG3_PREREGISTRATION.md`, written before the images were downloaded. The images were downloaded by the user with the NBIA Data Retriever to `C:/LUNG_phase15/lung3`: 89 patients, one CT series each, 6.7 GB.

**Code:** `src/phase14b_lung3.py`, which reuses every frozen component unchanged.

**Prediction freeze:** `results/phase14b/tcga_predictions.csv`. The file name is inherited from the reused code; it contains the Lung3 predictions. sha256 `8a41a6fd7cf9d6b7096cd85535f4ca35009a17536cd4cd2a9b93272fcf76956d`. Labels were joined once, afterwards.

## Pipeline

- **Conversion:** 89/89. Slice spacing: 5 mm in 41, 4 mm in 28, ≤ 2.5 mm in 18, 3 mm in 2. Most of Lung3 is thick-slice.
- **Tumour localisation:** TotalSegmentator found a nodule in 89/89, and MedSAM2 produced 89/89 masks.
- **Features:** 0 failures. Age was missing for 4 patients and was imputed.
- **Labels:** 10 were excluded by the pre-registered rule (NSCLC NOS, pleomorphic, LCNEC, SCC with adeno features). 79 were scored (44 ADC / 35 SCC).

## Results (pre-registered, reported unconditionally)

| Model | AUC [95% CI] | Sensitivity | Specificity | Balanced accuracy | ≤ 2.5 mm (n = 18) | > 2.5 mm (n = 61) |
|---|---|---|---|---|---|---|
| **PRIMARY shells + loc_clin** | **0.686 [0.557, 0.807]** | 0.636 | 0.686 | 0.661 | 0.903 | 0.676 |
| Secondary gtv + shells + loc_clin | 0.704 [0.577, 0.825] | 0.636 | 0.657 | 0.647 | 0.903 | 0.702 |

## Interpretation

1. **Lung3 replicates the development transport performance.** The primary scores 0.686 against a LOSO mean of 0.714 and a within-RG surgical cohort score of 0.70.
2. **TCGA (0.585) is the outlier.** Its perfect institution–label confound and its automatic-localisation noise are the likely reasons.
3. **Thin-slice CT gives the highest external score** (0.90 in Lung3, 0.73 in TCGA), but both subsets are small (n = 18 and n = 29).
4. **Thick slices behaved differently in the two tests.** Thick-slice Lung3 still reached 0.68, whereas thick-slice TCGA was at chance (0.51). Slice thickness alone therefore does not explain the TCGA failure; the TCGA institution confound and its heterogeneous multi-centre acquisition remain candidate explanations.

## Combined external evidence (two locked tests, 163 patients)

| Test | Primary AUC [95% CI] |
|---|---|
| TCGA | 0.585 [0.462, 0.707] |
| Lung3 | 0.686 [0.557, 0.807] |

- The sample-size-weighted mean is about 0.635.
- On the pooled thin-slice subsets (n = 47) the model is highest: 0.73 in TCGA and 0.90 in Lung3.

**Limitation:** Lung3 comes from the same region (Maastricht) as the LUNG1 training cohort, with different patients and a surgical rather than radiotherapy pathway.
