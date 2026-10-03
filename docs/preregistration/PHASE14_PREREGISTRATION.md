# Phase 14 pre-registration: locked TCGA external test

Written 2026-09-26, BEFORE any TCGA label is read, any TCGA mask exists, or any TCGA prediction is made.

## Test cohort

- **Patients:** TCGA-LUAD / TCGA-LUSC CT, **85 patients**, selected in Phase 12 by metadata only (`C:/LUNG_phase12/tcga_selection.csv`).
- **Label:** the collection (LUAD = ADC = 1, LUSC = SCC = 0). The labels live in `C:/LUNG_phase12/tcga_LOCKED_labels.csv` and are read only by the final scoring step.

## Tumour localisation (fully automatic; user choice B)

1. **Detector:** TotalSegmentator task `lung_nodules` (BLUEMIND AI; Apache-2.0), with default settings.
2. **Lesion choice:** the largest 3-D connected component of `lung_nodules` is taken as the primary tumour.
3. **Boxes:** its per-slice bounding boxes become the box prompts.
4. **Mask:** the frozen Phase-13 MedSAM2 pipeline, with the same settings (`C:/LUNG_phase13/medsam2_masks/settings.json`).
5. **Failure handling:** if no nodule is detected, or MedSAM2 returns an empty mask, the patient is EXCLUDED. The number excluded is reported. Exclusion does not depend on the label.
6. **Detector QA before TCGA, with no histology used:** 60 development patients with expert GTVs (30 LUNG1 and 30 Radiogenomics, random seed 0). The metrics are:
   - the hit rate, meaning the largest component overlaps the GTV;
   - the Dice of the final MedSAM2 mask against the GTV.

## Frozen model

- **Model:** the PRIMARY, shells + loc_clin (MedSAM2 masks, ComBat), pre-specified since Phase 10.
- **Training:** fit ONCE on all 1041 development patients, using the Phase-12/13 `select_and_fit` machinery with inner-CV hyper-parameter choice and seed 42.
- **Harmonising TCGA:** TCGA is mapped into the training reference as ONE new site, with label-free location/scale (`fit_new_site`, balanced = False). This is identical to the LOSO procedure.
- **Features:** the same extractors; age/sex are taken from the DICOM header.
- **Secondary (pre-declared):** the post-hoc Phase-13 best, gtv + shells + loc_clin. It is reported as secondary only.

## Metrics (computed once)

- **Primary metric:** ROC-AUC on TCGA, with a 2000-sample stratified bootstrap 95% CI.
- **Also reported:**
  - balanced accuracy at score 0;
  - a tissue-source-site-stratified AUC (TSS = the 2-character code in the barcode), averaged over the TSS that contain both classes, with patient counts. This controls the collection = label confound.
- **Reporting rule:** the result is reported whatever it is. No re-tuning after the labels are read.
