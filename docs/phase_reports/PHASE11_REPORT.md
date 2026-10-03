# Phase 11 — Three-hospital scale-up: ADC vs SCC on 644 patients

**Status:** complete (2026-09-26).

**Authorised by** the user's explicit decisions of 2026-09-26:
- a 1-week scope;
- new TCIA data stored on `C:`;
- CT only, no PET;
- local GPU.

**Configs:** `config/phase11.yaml` (data, segmentation, features); `config/phase10.yaml` (models, CV, evaluation — unchanged).

**Environment:** `.venv-phase10` (+ openpyxl, pydicom 2.4.4); PyRadiomics in `.venv`.

> **Claim scope.** Leave-one-hospital-out is *internal–external* cross-validation. Each hospital is scored by a model that never saw any of its patients, and its labels are used only to compute the final AUC. Lung-PET-CT-Dx is new to this project, but it was not reserved in advance as a one-shot test set. Model-building choices (blocks, fusion rule) were fixed in Phase 10, before its data existed.

---

## 1. Summary

**Cohort.** TCIA Lung-PET-CT-Dx (Harbin Medical University; 301 patients, 244 ADC / 57 SCC) was added to LUNG1 (202) and NSCLC-Radiogenomics (141).
- Total: **644 patients, 407 ADC / 237 SCC, three hospitals on two continents.**
- Tumour masks for **all three** hospitals come from **one box-seeded automatic method**, so the mask source cannot act as a site fingerprint.
- The expert outlines of the two original cohorts are used only to measure that method: mean Dice 0.82 in each cohort.

**Primary model.** Pre-specified before any Phase-11 data existed: the Phase-10 winner, i.e. peritumoral tissue + tumour position + age/sex, ComBat-harmonised, equal-weight fusion.

| Evaluation | ROC-AUC |
|---|---|
| **Pooled, site-balanced (5×5 CV)** | **0.718** (95 % CI 0.669–0.767) |
| Within Lung-PET-CT-Dx / Radiogenomics / LUNG1 | 0.845 / 0.687 / 0.628 |
| **Held-out hospital: Lung-PET-CT-Dx** | **0.735** |
| Held-out hospital: Radiogenomics | 0.674 |
| Held-out hospital: LUNG1 | 0.625 |
| Held-out mean | 0.678 |

**Robustness to the choice of model.**
- All **127** equal-weight fusions of the 7 blocks fall between 0.668 and 0.722 pooled, with a **median of 0.710**.
- The fusion of all 7 blocks, which involves no selection at all, gives **0.717**.

**Shortcut on three hospitals.**
- A hospital-only predictor scores a standard pooled AUC of **0.756** (site-balanced 0.495).
- A naive radiomics model scores 0.752 pooled but only **0.649** site-balanced.

---

## 2. What is new in Phase 11

1. **A third, independent hospital.** It comes from a different continent and a different clinical setting (diagnostic PET/CT work-up), with heterogeneous acquisition:
   - scanners: Siemens 151 / GE 123 / Philips 27;
   - slice spacing from 0.6 mm to 10 mm, mostly 5 mm.
2. **Box → mask segmentation, validated against experts.** On each boxed slice:
   - keep voxels above −750 HU inside the box;
   - apply a 2-D opening and keep the largest component;
   - fill holes;
   - in 3-D, keep components ≥ 10 % of the largest.

   For the two expert-segmented cohorts, the boxes are the per-slice bounding boxes of the expert GTV. The threshold was chosen on Dice against the expert GTVs only, never on labels:

   | Threshold (HU) | LUNG1 mean Dice | Radiogenomics mean Dice |
   |---|---|---|
   | −700 | 0.827 | 0.802 |
   | **−750** | **0.818** | **0.824** |
   | −800 | 0.808 | 0.836 |

   −750 maximises the smaller of the two cohort means.
3. **Consistent features for everyone**, all on the automatic mask and a 2 mm grid, from the original image only (thick-slice cohort):

   | Block | Content |
   |---|---|
   | gtv | radiomics with shape |
   | ring_in | inner 4 mm rim |
   | shells | outer shells at 0–4, 4–8 and 8–12 mm |
   | loc_clin | tumour position + age/sex |
   | fmcib | FMCIB at the centroid |
   | fmcib_bag | mean of **16-crop FMCIB bags** (centroid + 15 farthest-point boundary seeds) |
   | mil | **gated attention-MIL** over the same bags |
4. **Internal–external cross-validation** over three hospitals.

---

## 3. Data acquisition (Lung-PET-CT-Dx)

| Item | Value |
|---|---|
| Source | TCIA Lung-PET-CT-Dx, DOI 10.7937/TCIA.2020.NNC2-0461, CC BY 4.0; annotations `rev12222020` (31 562 PASCAL-VOC XML files, one per annotated slice, named by SOPInstanceUID); clinical table `statistics-clinical-20201221.xlsx` |
| Eligible | patient-ID letter A (ADC, 251) or G (SCC, 61) with a CT series |
| Series choice | per patient, the CT series holding the most annotated slices (ties → more images). Chosen from metadata and annotation UIDs only; 1 048 series checked through the NBIA API |
| Selectable | 309 (3 ADC patients had no annotation in any CT series) |
| Downloaded | 36.9 GB to `C:/LUNG_phase11/lung_pet_ct_dx/zips` (kept); NIfTI + box masks in `.../nifti/<PatientID>/` |
| Excluded | **8** (A0170, A0204, A0206, A0265, G0041, G0044, G0048, G0058). Their annotated series is a *secondary-capture fused MPR* (SOP class 1.2.840.10008.5.1.4.1.1.7) with no 3-D geometry |
| Final | **301** (244 ADC / 57 SCC) |
| Visual QA | `qa/phase11/lpcd_box_check.png`, `qa/phase11/lpcd_autoseg_gallery.png` |

**Within-hospital shortcut check** (Lung-PET-CT-Dx). ROC-AUC of each acquisition variable for predicting ADC:

| Variable | AUC |
|---|---|
| slice spacing | 0.501 |
| pixel size | 0.487 |
| number of slices | 0.484 |
| study year | 0.458 |
| contrast | 0.491 |
| number of annotated slices (tumour extent) | 0.389 |

Scanner make and reconstruction kernel are also balanced across the classes. **No acquisition shortcut exists inside this hospital.** The only associated variable is tumour extent: SCCs are larger, which is biology, not acquisition.

---

## 4. Results

### 4.1 Single blocks (ComBat; `results/phase11/pooled`, `results/phase11/loso`)

| Block | Pooled site-balanced | LUNG1 | Radiogenomics | Lung-PET-CT-Dx | Held-out mean | Hospital-classifier AUC after ComBat |
|---|---|---|---|---|---|---|
| gtv | 0.685 | 0.540 | 0.645 | 0.844 | 0.657 | 0.58 |
| ring_in | 0.684 | 0.580 | 0.634 | 0.830 | 0.641 | 0.61 |
| shells | 0.693 | 0.641 | 0.640 | 0.796 | 0.639 | 0.64 |
| **loc_clin** | **0.707** | 0.592 | 0.699 | 0.833 | **0.684** | 0.56 |
| fmcib | 0.668 | 0.577 | 0.638 | 0.779 | 0.625 | 0.27 |
| fmcib_bag | 0.686 | 0.609 | 0.658 | 0.787 | 0.629 | 0.57 |
| mil (gated attention) | 0.669 | 0.586 | 0.666 | 0.759 | 0.634 | n/a |

Before ComBat, the hospital-classifier AUC is 0.80–0.91 depending on the block. ComBat reduces it for every radiomics and position block. For FMCIB it overshoots to 0.27, the same over-correction seen in Phase 10.

### 4.2 Fusion (`results/phase11/fusion/fusion_all_combat.csv`, all 127 combinations)

| Combination | Pooled (95 % CI) | Held-out LUNG1 | Held-out Radiogenomics | Held-out Lung-PET-CT-Dx | Held-out mean |
|---|---|---|---|---|---|
| **shells + loc_clin (PRIMARY, pre-specified)** | **0.718 (0.669–0.767)** | 0.625 | 0.674 | 0.735 | 0.678 |
| all 7 blocks | 0.717 (0.669–0.766) | 0.611 | 0.648 | 0.770 | 0.676 |
| best pooled (shells + loc_clin + gtv + fmcib_bag) | 0.720 | 0.613 | 0.662 | 0.770 | 0.681 |
| best held-out (loc_clin + gtv) | 0.714 | 0.587 | 0.672 | 0.813 | 0.691 |
| median of all 127 | 0.710 | n/a | n/a | n/a | 0.668 |

Primary model at the training-balanced midpoint (threshold 0):

| Hospital | Balanced accuracy | Sensitivity | Specificity |
|---|---|---|---|
| Lung-PET-CT-Dx | 0.732 | 0.586 | 0.877 |
| Radiogenomics | 0.689 | 0.688 | 0.690 |
| LUNG1 | 0.589 | 0.569 | 0.609 |

### 4.3 Harmonisation

- ComBat vs none changes the pooled AUC by −0.004 to +0.020 per block.
- For the held-out hospital, "none" is equal or slightly better for tumour-interior blocks. With three hospitals and site-balanced training weights, the model already has little incentive to use the site.
- ComBat's main measurable effect remains removing the site fingerprint (§4.1).

---

## 5. Interpretation, including the honest caveats

1. **The honest pooled score crosses 0.70, and the result is robust.** The pre-specified model gives 0.718; the median of 127 fusions is 0.710; the all-blocks fusion is 0.717.
2. **Most of the gain comes from case mix, not from better modelling of the old cohorts.**
   - Within LUNG1 and Radiogenomics the Phase-11 model is *not* better than Phase 10: 0.628 / 0.687 now vs 0.645 / 0.704 then. Phase 10 used expert masks and richer filtered features.
   - Lung-PET-CT-Dx is a diagnostic cohort in which SCCs are larger, central tumours and ADCs are smaller and more peripheral. That makes discrimination easier: 0.85 within the hospital and 0.74–0.81 when it is held out.
   - LUNG1 (locally advanced tumours treated with radiotherapy) stays hardest at about 0.6 in every configuration of every phase.

   **Performance is population-dependent.** That is a finding worth reporting, not a defect.
3. **Tumour position is the single most transportable signal.** It has the highest held-out mean of any block (0.684). It is anatomical, interpretable and clinically established: SCC tends to be central, ADC peripheral.
4. **Deep features do not beat handcrafted ones at this scale.** FMCIB 0.668, 16-crop bag 0.686, attention-MIL 0.669, radiomics 0.685–0.707. Attention-MIL again fails to beat mean pooling, consistent with Phases 3–6C.

---

## 6. Files

**Code**

| File | Purpose |
|---|---|
| `src/phase11_lpcd_build.py` | series selection and download/convert |
| `src/phase11_autoseg.py` | box-seeded segmentation, validation, run |
| `src/phase11_features.py` | cohort, radiomics, location, FMCIB |
| `src/phase11_fmcib_bag.py` | 16-crop bags |
| `src/phase11_cv.py` | pooled, leave-one-hospital-out, fusion |
| `src/phase11_mil.py` | gated attention-MIL |
| `src/phase11_figures.py` | figures and summary |

**Data**
- `cohort/phase11_three_site_cohort.csv`
- `global_features/phase11/{radiomics,loc,clin,fmcib,fmcib_bagmean}.csv`
- `global_features/phase11/fmcib_bags/*.npy`
- `C:/LUNG_phase11/` (images, boxes, auto masks, manifests)

**Results**
- `results/phase11/autoseg_validation*.csv`
- `results/phase11/pooled/`, `results/phase11/loso/`, `results/phase11/fusion/`
- `results/phase11/figures/fig1–fig4`
- `results/phase11/phase11_summary.json`

**Logs:** `logs/phase11_*.log`.

---

## 7. Commands

```bash
.venv-phase10/Scripts/python.exe src/phase11_lpcd_build.py select
.venv-phase10/Scripts/python.exe src/phase11_lpcd_build.py download          # ~70 min, resumable
.venv-phase10/Scripts/python.exe src/phase11_autoseg.py validate --thresholds -700 -750 -800
.venv-phase10/Scripts/python.exe src/phase11_autoseg.py run --threshold -750
.venv-phase10/Scripts/python.exe src/phase11_features.py cohort
.venv/Scripts/python.exe         src/phase11_features.py radiomics --jobs 9  # ~17 min
.venv-phase10/Scripts/python.exe src/phase11_features.py loc --jobs 4        # ~17 min
.venv-phase10/Scripts/python.exe src/phase11_features.py fmcib               # ~12 min GPU
.venv-phase10/Scripts/python.exe src/phase11_fmcib_bag.py --k 16             # ~15 min GPU
.venv-phase10/Scripts/python.exe src/phase11_cv.py pooled --blocks ring_in shells loc_clin gtv fmcib fmcib_bag --harm combat none
.venv-phase10/Scripts/python.exe src/phase11_cv.py loso   --blocks ring_in shells loc_clin gtv fmcib fmcib_bag --harm combat none
.venv-phase10/Scripts/python.exe src/phase11_mil.py pooled
.venv-phase10/Scripts/python.exe src/phase11_mil.py loso
.venv-phase10/Scripts/python.exe src/phase11_cv.py fusion --blocks ring_in shells loc_clin gtv fmcib fmcib_bag mil --harm combat
.venv-phase10/Scripts/python.exe src/phase11_figures.py
```

---

## 8. Limitations

- **Automatic masks.** Dice is 0.82. Tumours abutting the chest wall or mediastinum include some adjacent soft tissue inside the box; this is the same for all hospitals.
- **Box style.** Lung-PET-CT-Dx boxes were drawn by that collection's annotators; the development-cohort boxes are tight expert-derived boxes.
- **Clinical variables.** Age and sex only. Staging is not comparable across the cohorts (clinical vs pathological).
- **Held-out hospital scoring.** It uses label-free location/scale mapping, but hyper-parameters and the fusion rule were developed on these cohorts (the Phase-10 cohorts in particular). A reserved, never-touched cohort is still the gold standard.
- **Confidence intervals** are bootstrap intervals over seed-averaged out-of-fold scores; they do not include model-selection uncertainty.
