# Phase 14B pre-registration: second locked external test (Lung3)

Written 2026-09-27, BEFORE any Lung3 image is downloaded (the user downloads it with the NBIA Data Retriever to `C:/LUNG_phase15/lung3`) and before any Lung3 prediction exists.

The Lung3 histology column was read ONLY to count classes for this plan. No per-patient label has been linked to any image, feature or prediction.

## Cohort

- **Source:** TCIA NSCLC-Radiomics-Genomics (Lung3), 89 surgically treated NSCLC patients (CC BY-NC 3.0, DOI 10.7937/K9/TCIA.2015.L4FRET6Z).
- **Labels:** `Lung3.metadata.xls`, column `characteristics.tag.histology`.
  - **ADC = 1:** every value containing "Adenocarcinoma", or equal to "Solid Type And Acinar".
  - **SCC = 0:** values starting with "Squamous Cell Carcinoma", EXCEPT "Squamous Cell Carcinoma, Other (Specify) with adeno features", which is excluded as mixed.
  - **Excluded:** "Non-Small Cell" (NOS), "Non-Small Cell, Pleomorphic Type", and "Carcinoma, Large Cell, Neuroendocrine".
- **CT series:** one per patient, the axial CT with the most images. Series with fewer than 40 images are excluded.

**Independence.** There is no PatientID or StudyInstanceUID overlap with LUNG1 (verified in Phase 9A). The institution is in the same Maastricht region as LUNG1, and this is stated as a limitation.

## Pipeline

Identical to Phase 14 and fully frozen:
1. TotalSegmentator `lung_nodules`, taking the largest component and its per-slice boxes.
2. MedSAM2 with the frozen Phase-13 settings.
3. The same extractors (age/sex from the DICOM header).
4. The PRIMARY model (shells + loc_clin), fitted ONCE on the 1041 development patients (seed 42, ComBat, inner-CV). Lung3 is mapped as one new site with label-free location/scale.

**Exclusions:** no detected nodule, an empty MedSAM2 mask, or a CT that cannot be read. All are counted and reported.

**Integrity:** the predictions are written and sha256-hashed BEFORE the labels are joined, and are scored once.

## Metrics

- **PRIMARY:** ROC-AUC on all included patients, with a 2000-sample stratified bootstrap 95% CI.
- **SECONDARY (pre-declared from the Phase-14 thickness finding):**
  - AUC restricted to slice spacing ≤ 2.5 mm;
  - AUC for spacing > 2.5 mm.
- **Also:** sensitivity, specificity and balanced accuracy at score 0. There is a single institution, so no site stratification is needed.

**Reporting rule:** the result is reported whatever it is, alongside TCGA (0.585). No re-tuning after the labels are read.
