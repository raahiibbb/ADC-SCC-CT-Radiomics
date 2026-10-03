# Phase 13 pre-registration (written 2026-09-26 BEFORE any Phase-13 ADC/SCC result)

The ADC/SCC evaluation is on the 4-site cohort (1041), with the same CV machinery as Phase 12: ComBat, 5 seeds × 5 folds, site-balanced AUC, and LOSO.

## Segmentation

MedSAM2 is chosen as follows:
- checkpoint `MedSAM2_latest.pt`, full slices;
- one key-slice box prompt, then bidirectional propagation.

It was chosen by a rule fixed before looking at the last grid rows: maximise the worst mean Dice over {LUNG1, RG} × {tight, loose boxes} on 40 expert GTVs. It was confirmed on all 343 expert GTVs: tight 0.843/0.850, loose 0.738/0.743 (against v5: 0.812/0.830 and 0.572/0.680). No label was used.

## Analyses

- **PRIMARY:** shells + loc_clin (frozen since Phase 10), with features re-extracted from the MedSAM2 masks.
- **SECONDARY (pre-declared):** equal-weight fusion of ALL blocks: gtv, ring_in, shells, loc_clin, fmcib, ctfm, ctfmt.
- **Everything else**, such as the best single fusion, is reported as POST HOC.
- **CT-FM crop:** the pretraining geometry (32 × 128 × 128 vox at 3 × 1 × 1 mm), centred on the mask centroid.
  - `ctfm` = global average;
  - `ctfmt` = average over the tumour-covered feature cells.

## Locked data

The TCGA labels stay unread.

## 13c addendum (written 2026-09-26 BEFORE any 13c result)

### Backbone and training
- **Backbone:** FMCIB, the 3D ResNet-50 already validated in this project.
  - The conv1 → layer4 block 2 stages are frozen.
  - The final bottleneck block (layer4[2]) is fine-tuned from the pretrained weights, with BatchNorm statistics frozen.
- **Input:** a 50 mm cube at 1 mm around the MedSAM2 centroid, with 8 cached views per patient. View 0 is un-augmented. Views 1–7 are augmented with:
  - a ±6 mm shift;
  - x/y flips;
  - in-plane rot90;
  - an HU shift of ±30 and Gaussian noise of 0–20 HU;
  - Gaussian blur with σ ≤ 1 voxel, to simulate the reconstruction kernel.
- **Heads:** a label head, Dropout 0.3 → Linear. The adversary is a gradient-reversal layer → [feature ⊕ label one-hot] → MLP(64) → site.
  - It is **label-conditional**, so it removes site information that the label does not explain. This matters because prevalence differs by site.
  - λ follows the DANN ramp to λmax = 0.3.
- **Losses:**
  - label loss: BCE with site × label balanced weights;
  - site loss: CE with site × label balanced weights.
- **Optimiser:** AdamW with lr 1e-4 for the block and 1e-3 for the heads, wd 1e-2, batch 32, at most 25 epochs.
- **Early stopping:** inner validation with 20% of the training patients (stratified by site × label), on site-balanced AUC averaged over the views, patience 6.
- **Prediction:** the mean logit over the 8 views.
- **Protocol:** the same outer folds and seeds as Phase 12 (5 × 5 pooled; LOSO with 5 seeds). ComBat is not used, because the adversary replaces it.

### Pre-declared comparisons
1. **advft** (λmax 0.3) against **ft** (λ = 0, identical otherwise). The hypothesis is that adversarial training improves the LOSO mean and lowers the site-leak.
2. The 13c headline candidate is the fusion **shells + loc_clin + advft**, compared with the primary shells + loc_clin (0.722 / LOSO 0.714).

Any other fusion is post hoc.
