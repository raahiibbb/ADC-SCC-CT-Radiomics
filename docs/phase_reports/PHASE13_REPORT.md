# Phase 13 report: 13a segmentation, 13b foundation-model bank, 13c site-adversarial fine-tuning

**Date:** 2026-09-26. User-authorised ("quick check and phase 13").

**Pre-registration:** `docs/PHASE13_PREREGISTRATION.md`, written before any Phase-13 ADC/SCC result.

**Status:** Phase 13 is complete. 13c has been added (section 7), and its pre-registered result is negative.

## 0. Quick checks carried over from Phase 12

- **Ablation.** The v5 masks on the Phase-11 three sites give primary 0.716 and LOSO mean 0.682, against 0.718 and 0.678 with the v1 masks. The mask change was therefore neutral. The Phase-12 transport gain (LOSO 0.704, LPCD 0.839) comes from adding NLST.
- **Correction** to a statement made in chat: a −750 HU threshold already keeps most ground-glass tissue. Only very faint GGO below −750 is lost. The low AUTO/BOX ratio in NLST reflects loose boxes, not the loss of GGO.

## 1. 13a: box-prompted foundation-model segmentation (MedSAM2)

**Model:** MedSAM2 (bowang-lab, Apache-2.0).
- Commit `332f30d`.
- Checkpoint `MedSAM2_latest.pt`, sha256 `c92743b9…1060`.
- Loaded by the repo's own `torch.load(weights_only=True)`.
- The SAM2 CUDA post-processing extension is not built on Windows, so hole-fill post-processing is skipped. This is documented as harmless.

**Setup:** `.venv-medsam2` (Python 3.12, torch 2.5.1+cu124). Settings are recorded in `C:/LUNG_phase13/medsam2_masks/settings.json`:
- lung window [−1350, 150] HU, the DeepLesion lung-lesion window;
- full 512² slices;
- the same boxes as Phase 12;
- one box prompt on the slice with the largest box;
- bidirectional propagation, with the result kept inside the box dilated by 3 mm.

**Variant choice** used 40 expert GTVs and no labels. The rule, fixed before the last grid rows were read, was to maximise the worst mean Dice over {LUNG1, RG} × {tight, loose boxes}.

| Variant | Worst-case Dice |
|---|---|
| **latest / full / key-slice** | **0.701** |
| ctlesion / full / key-slice | 0.669 |
| latest / full / all-slice prompts | 0.647 (best on tight boxes, 0.888/0.877) |
| cropped variants | ≤ 0.52 |
| v5 | 0.530 |

**Confirmation on all 343 expert GTVs** (mean Dice, LUNG1 / RG):

| | Tight boxes | Loose boxes (+8 mm) |
|---|---|---|
| v5 (Phase 12) | 0.812 / 0.830 | 0.572 / 0.680 |
| **MedSAM2** | **0.843 / 0.850** | **0.738 / 0.743** |

**Run on all 1041 patients:** 0 failures, 38 min on an RTX 3050 4 GB. Median volume (ml): LUNG1 47.8, RG 6.8, LPCD 26.1, NLST 5.0. The QA gallery is `results/phase12/figures/nlst_autoseg_gallery_medsam2.png`. It shows the nodules captured including their GGO parts, with rare pleural inclusions remaining.

## 2. 13b: foundation-model embedding bank

- **CT-FM** (project-lighter `ct_fm_feature_extractor`, safetensors, Apache-2.0).
  - The crop copies the pretraining geometry: SPL orientation, 3 × 1 × 1 mm spacing, 32 × 128 × 128 voxels, HU [−1024, 2048] → [0, 1].
  - It is centred on the MedSAM2 centroid.
  - Two blocks: `ctfm`, the global average of the 512-d last stage (2 × 8 × 8 map), and `ctfmt`, the average over tumour-covered cells.
  - 1041 patients, 0 failures, 18 min.
- **FMCIB and radiomics** (gtv, ring_in, shells) and **loc** were re-extracted from the MedSAM2 masks. All 1041 patients succeeded with 0 failures.

## 3. Results (4 sites, 1041 patients; ComBat; 5 × 5 CV; site-balanced AUC)

| Block | Pooled | LUNG1 | RG | LPCD | NLST | LOSO mean | Site-leak |
|---|---|---|---|---|---|---|---|
| gtv | 0.702 | 0.587 | 0.676 | 0.833 | 0.695 | 0.695 | 0.570 |
| ring_in | 0.698 | 0.609 | 0.680 | 0.823 | 0.661 | 0.686 | 0.566 |
| shells | 0.700 | 0.667 | 0.667 | 0.822 | 0.650 | 0.691 | 0.606 |
| loc_clin | 0.709 | 0.578 | 0.707 | 0.839 | 0.711 | 0.704 | 0.542 |
| fmcib | 0.675 | 0.596 | 0.631 | 0.782 | 0.677 | 0.660 | 0.263 |
| ctfm | 0.662 | 0.592 | 0.637 | 0.765 | 0.660 | 0.652 | **0.765** |
| ctfmt | 0.662 | 0.606 | 0.623 | 0.778 | 0.650 | 0.652 | 0.747 |

| Model | Pooled [95% CI] | LOSO: LUNG1 / RG / LPCD / NLST | LOSO mean |
|---|---|---|---|
| **PRIMARY shells + loc_clin** (pre-specified) | **0.722 [0.687, 0.762]** | 0.625 / 0.704 / 0.835 / 0.692 | **0.714** |
| **SECONDARY fusion of all 7** (pre-declared) | 0.720 [0.686, 0.760] | 0.625 / 0.691 / 0.839 / 0.698 | 0.713 |
| Best fusion, post hoc: gtv + shells + loc_clin | 0.730 [0.695, 0.769] | 0.621 / 0.717 / 0.857 / 0.700 | 0.724 |
| All 127 fusions | median 0.715 | – | median 0.705 |

The Phase-12 primary on the v5 masks was pooled 0.714 and LOSO 0.704.

## 4. Interpretation

1. **Better masks give a small, consistent gain.** Every radiomics block improved by +0.009 to +0.015 pooled. The primary improved by +0.008 pooled and +0.010 LOSO. Shells on LUNG1 reached 0.667, the best LUNG1 value so far.
2. **CT-FM does not help.** It is weaker than handcrafted features (0.662) and **retains hospital identity after ComBat** (site-leak 0.77 against about 0.57 for radiomics). Its generic whole-CT contrastive pretraining encodes the scanner and protocol. This is a publishable negative finding.
3. **0.75 is still not reached.** The best honest figures are 0.722 (pre-specified) and 0.730 (post hoc). LUNG1 (about 0.63) remains the ceiling.

## 5. Files

- **Code:** `src/phase13_medsam2.py` and `src/phase13_ctfm.py`. `src/phase12_features.py`, `src/phase12_cv.py` and `src/phase12_qa.py` now take env-var overrides: `ADC_MASK_ROOT`, `ADC_FEAT_OUT`, `ADC_RES_OUT`, `ADC_QA_TAG`.
- **Data:**
  - `C:/LUNG_phase13/medsam2_masks/` (52 MB);
  - `global_features/phase13/`;
  - `external/MedSAM2/` (repo and checkpoints).
- **Results:**
  - `results/phase13/{pooled,loso,fusion}`;
  - `results/phase13/medsam2_validation_*.csv`.
- **Environment:** `reports/phase13_environment_medsam2.txt`.

## 6. Warnings

- **Environment incident.** Installing `lighter_zoo` with its dependencies replaced torch in `.venv-medsam2` with torch 2.14 CPU while a job was running. The job was stopped, torch 2.5.1+cu124 was force-reinstalled, the validation was rerun, and `lighter_zoo` and `monai` were reinstalled with `--no-deps`. **Always pass `--no-deps` in that environment.**
- **Site information.** CT-FM embeddings are site-informative even after ComBat, so do not fuse them without harmonisation.

## 7. 13c: site-adversarial fine-tuning (pre-registered; `src/phase13_advft.py`)

**Setup** (see the 13c addendum in the pre-registration):
- FMCIB, frozen up to layer4[1];
- the last bottleneck block (35M parameters) fine-tuned from the pretrained weights;
- 8 cached views per patient, with shift, flips, rot90, HU shift/noise and blur (cache 0.52 GB, built in 10 min);
- a label-conditional gradient-reversal site adversary (λmax 0.3) against λ = 0;
- no ComBat;
- the same 5 × 5 outer folds and LOSO as Phase 12.

Each variant (pooled plus LOSO) took about 8 min on the RTX 3050.

| Model | Pooled | LUNG1 | RG | LPCD | NLST | LOSO mean | Site-leak |
|---|---|---|---|---|---|---|---|
| ft (λ = 0) | 0.615 | 0.561 | 0.571 | 0.735 | 0.616 | 0.644 | 0.814 |
| advft (λ = 0.3) | 0.606 | 0.549 | 0.565 | 0.712 | 0.616 | 0.651 | 0.815 |
| *Reference: frozen FMCIB + ComBat (13b)* | *0.675* | | | | | *0.660* | *0.263* |
| **Pre-declared: shells + loc_clin + advft** | 0.708 | 0.616 | 0.691 | 0.841 | 0.692 | 0.710 | – |
| *Primary: shells + loc_clin* | *0.722* | | | | | *0.714* | |

**Verdict: negative on every pre-registered test.**
1. Fine-tuning is worse than the frozen features with ComBat, by −0.06 pooled.
2. The adversary slightly improves the LOSO mean (+0.007 over ft). It does **not** reduce site information: the site-leak stays at about 0.815. The gradient-reversal head is too weak to overcome acquisition signals that a simple probe finds easily.
3. Adding advft to the primary lowers it, from 0.722 to 0.708.
4. The best epochs are early (median 7, range 0–16). With about 660 training patients per fold, the 35M-parameter block fits site and noise faster than histology.

**Lesson.** At n ≈ 1000, explicit statistical harmonisation (ComBat) on frozen features beats end-to-end adversarial invariance. This is consistent with Phases 3–6 and 11: the deep, learned components never beat the handcrafted features plus harmonisation under an honest multi-site protocol.

The fusion table is in `results/phase13/fusion13c/fusion/fusion_all_combat.csv`.

## 8. Final Phase-13 headline

- **Pre-specified primary:** 0.722 [0.687, 0.762], LOSO mean 0.714.
- **Best post hoc:** 0.730, from gtv + shells + loc_clin on the MedSAM2 masks.
- **The 0.75 target was not reached** under the honest protocol.

## 9. Next (not started)

Phase 14, the locked TCGA test. TCGA needs tumour boxes first: either manual boxes (85 patients) or a detector. Then MedSAM2 → the frozen primary → a single scoring.
