# Phase 10: Confound-aware multi-centre CT classification of ADC vs SCC

**Status:** complete (2026-09-26).
**Authorised by:** the user's explicit decision of 2026-09-25/26. This decision lifts, for this phase only, the Phase 8B/9A bans on ComBat and harmonisation, SVM, LightGBM, clinical covariates and foundation-model features. Phases 1–9A are untouched.
**Config:** `config/phase10.yaml`.
**Environment:** `.venv-phase10` (Python 3.9, torch 2.6.0+cu124, scikit-learn 1.5.2, LightGBM 4.6.0, MONAI 1.3.2); PyRadiomics extraction ran in `.venv`. Full pin list: `reports/phase10_environment.txt`.

> **Claim scope.** This is multi-centre development on LUNG1 (202 patients) + NSCLC-Radiogenomics (141 patients). Train-on-one-hospital, test-on-the-other ("leave-one-site-out transport") is the strongest claim allowed. **Nothing here is independent external validation.** A third, untouched cohort is still required for that.

---

## 1. One-paragraph summary

The two public cohorts have **reversed class mixes**: LUNG1 is 25 % ADC and Radiogenomics is 79 % ADC. Their scanners also differ enough that a classifier identifies the hospital at ROC-AUC 0.998. A model can therefore reach a high pooled AUC by recognising the hospital, without learning anything about histology.

We show this directly:
- A "model" that knows **only the hospital** scores a standard pooled AUC of **0.758**.
- A naive radiomics model scores **0.787** pooled, but only **0.566** once the shortcut is neutralised.

We then:
1. adopt a **site-balanced AUC** as the primary metric; a site-only predictor scores exactly 0.5 on it;
2. remove the acquisition fingerprint with ComBat, fitted on training folds only. Hospital-classifier AUC falls from 0.95–1.00 to about 0.52;
3. add three new information sources: **tumour position** (central vs peripheral), **peritumoral multi-shell radiomics**, and **FMCIB foundation-model features**;
4. fuse independently trained models with fixed equal weights.

**The final model (rim radiomics + peritumoral shells + tumour position) reaches a site-balanced ROC-AUC of 0.681 (95 % CI 0.617–0.746).** That is +0.115 over the naive model on the same honest metric.
- Within each hospital: LUNG1 0.645, Radiogenomics 0.704.
- Transport, trained on one hospital and tested on the other: **0.704** (LUNG1 → Radiogenomics) and **0.636** (Radiogenomics → LUNG1).
- Phase 8B's transport under the same kind of test was 0.514–0.624.

---

## 2. Methodological contributions

1. **Shortcut quantification and a site-balanced metric.** Sample weight `1/(4·n_site,label)` makes every site count equally and each site 50/50, so the hospital-only predictor scores 0.5 by construction. All model selection uses this metric.
2. **Prevalence-balanced ComBat** (`src/phase10_harmonise.py`). Standard ComBat estimates a site's location as its plain mean. Under reversed prevalence that mean is partly a disease signal. The balanced variant instead uses:
   - the mean of the two class-conditional means for location,
   - the pooled within-class spread for scale.

   Applying it to a patient needs only the patient's site, never their label. On synthetic data with a 1-SD class effect, standard ComBat mis-aligns the ADC class across sites (means 3.13 vs 2.40), while the balanced variant aligns them (2.97 vs 3.09).
3. **Prevalence-informed ComBat for an unlabelled site** (`src/phase10_transport.py`). The destination site is described by one number, its ADC share, which is its case mix and not any patient label. A label-free EM variant (Saerens prior-shift EM) is also implemented.
4. **Anatomically motivated position features** (`src/phase10_extract_loc.py`). These come from one deterministic lung segmentation applied identically to both cohorts, so the mask source is not itself a site confound.
5. **Transparent fusion.** All 31 block combinations × 3 harmonisation settings are reported, with no hidden selection.

---

## 3. Protocol

| Item | Setting |
|---|---|
| Cohort | 343 patients: LUNG1 51 ADC / 151 SCC; Radiogenomics 112 ADC / 29 SCC (Phase-8B cohort, unchanged) |
| CV | 5 seeds (42–46) × stratified 5-fold, strata = site × label; identical splits for every block |
| Inside each training fold only | median imputation → harmonisation → z-score → ranking by mean within-site AUC → correlation prune 0.90 → top-k |
| Models | L2 logistic regression (k ∈ {10, 25, 50}), RBF SVM, LightGBM, dense ridge on all features. Hyper-parameters chosen by 3-fold inner CV on the site-balanced AUC. Block score = mean of the training-standardised model scores |
| Class and site imbalance | site × label balanced sample weights (no SMOTE) |
| Fusion | equal-weight mean of per-seed z-scored out-of-fold block scores; nothing fitted |
| Uncertainty | 2 000 stratified bootstrap resamples of the seed-averaged out-of-fold scores |
| Site-leakage diagnostic | per outer fold, logistic regression predicting the hospital from the harmonised training features, scored on the test fold with label-balanced weights |
| Transport | train on one site with inner-CV selection, score the other once; destination labels are used only by the clearly marked `oracle` upper bound |

### Feature blocks

| Block | Source | Size |
|---|---|---|
| gtv | Phase-7A/8A whole-GTV PyRadiomics (original + LoG + wavelet, with shape) | 1130 |
| rim | Phase-7A/8A rim (GTV+Rim minus GTV) | 1116 |
| peri | **new**: shells outside the GTV at (0, 4], (4, 8] and (8, 12] mm; original image only; resegment [-1024, 200] HU; binWidth 20 | 282 |
| loc_clin | **new**: 11 tumour-position features + age + sex | 13 |
| fmcib | **new**: FMCIB 3D ResNet50, frozen official weights (Zenodo 10528450, loaded `weights_only=True`, SHA-256 checked), 50 mm cube at the GTV centroid | 4096 |

Extraction failures: **0 of 343** for loc, peri and fmcib.

---

## 4. Results

### 4.1 The shortcut (`results/phase10/shortcut_demo/`, fig 1)

| Predictor | Standard pooled AUC | Site-balanced AUC |
|---|---|---|
| Hospital only, no image | **0.758** | 0.493 |
| Naive rim radiomics (no harmonisation, no weights) | **0.787** | 0.566 |
| Phase-10 final model | n/a | **0.681** |

### 4.2 Single blocks (ensemble, 5×5 CV; `results/phase10/block_summary.csv`, fig 2)

| Block | Harmonisation | Site-balanced AUC | LUNG1 | Radiogenomics | Hospital-classifier AUC |
|---|---|---|---|---|---|
| rim | none | 0.670 | 0.664 | 0.664 | 0.998 |
| rim | ComBat | **0.670** | 0.673 | 0.653 | 0.546 |
| rim | balanced ComBat | 0.658 | 0.668 | 0.651 | 0.521 |
| peri | ComBat | 0.655 | 0.630 | 0.675 | 0.542 |
| loc_clin | ComBat | 0.640 | 0.576 | 0.698 | 0.550 |
| fmcib | ComBat | 0.607 | 0.549 | 0.657 | 0.330 |
| gtv | ComBat | 0.590 | 0.571 | 0.601 | 0.536 |

Notes:
- **Harmonisation removes the site fingerprint in every block.** It barely changes pooled discrimination, because the site-balanced training weights already stop the model from exploiting the site.
- **FMCIB** after ComBat gives a hospital-classifier AUC of 0.33, i.e. over-correction. FMCIB is informative in Radiogenomics (0.66–0.68) but close to chance in LUNG1 (0.55). This is plausibly because LUNG1 tumours are large and a 50 mm crop sees only their interior.

### 4.3 Fusion (`results/phase10/fusion/fusion_all_combinations.csv`, fig 3)

Top combinations under ComBat, out of 31 (every combination is in the CSV):

| Combination | Site-balanced AUC ± SD over seeds | 95 % CI | LUNG1 | Radiogenomics |
|---|---|---|---|---|
| **rim + peri + loc_clin (FINAL)** | **0.681 ± 0.006** | 0.617–0.746 | 0.645 | 0.704 |
| rim + loc_clin | 0.678 ± 0.004 | 0.613–0.744 | 0.643 | 0.705 |
| rim + peri + loc_clin + fmcib | 0.673 ± 0.008 | 0.607–0.739 | 0.627 | 0.709 |
| gtv + rim + peri + loc_clin | 0.673 ± 0.008 | 0.607–0.739 | 0.640 | 0.694 |
| rim + peri | 0.672 ± 0.007 | 0.610–0.735 | 0.658 | 0.672 |

**Selection caveat.** The final combination was chosen after seeing these cross-validation results, out of 93 evaluated (31 combinations × 3 harmonisation settings). The top ten differ by less than 0.015, far inside the confidence interval, so the resulting optimism is small but not zero.

At a threshold of 0 on the seed-averaged z-score (the training-balanced midpoint), the final model's pooled results are:
- balanced accuracy 0.624
- sensitivity 0.626
- specificity 0.622

Radiogenomics alone: balanced accuracy 0.687.

### 4.4 Transport (`results/phase10/transport_auc_summary.csv`, fig 5)

ROC-AUC on the unseen hospital, averaged over 5 seeds, for the fused rim + peri + loc_clin model:

| Direction | None | ComBat | Balanced EM | Prevalence-informed | Oracle |
|---|---|---|---|---|---|
| LUNG1 → Radiogenomics | 0.695 | **0.704** | 0.710 | 0.699 | 0.698 |
| Radiogenomics → LUNG1 | 0.612 | **0.636** | 0.631 | 0.626 | 0.636 |
| Mean | 0.654 | **0.670** | 0.671 | 0.663 | 0.667 |

For comparison, Phase 8B's A3 transport was 0.624 / 0.514 (mean 0.569). Rim alone moves from 0.641 / 0.621 with no harmonisation to 0.684 / 0.653 with ComBat.

On decision metrics (rim block), harmonisation restores a usable operating point:
- mean balanced accuracy: 0.54 with none, 0.645 with ComBat;
- with no harmonisation, LUNG1 → Radiogenomics sensitivity collapses to 0.17 while specificity is 0.87.

### 4.5 Honest negative findings

1. **On these real data, balanced and prevalence-informed ComBat do not beat plain ComBat.** Even the oracle version, which uses the true destination labels, does not beat it. The class effect in the features is weak: the best single features reach a within-site AUC of about 0.65, so the prevalence bias in the site mean is small next to the acquisition shift. The balanced variant's advantage is real on synthetic data with a strong class effect, and it becomes material only there.
2. **Label-free EM cannot recover the destination prevalence.** It estimates 0.22–0.56 against a true 0.79, and 0.14–0.93 against a true 0.25. With a weak classifier, a site shift and a prevalence shift are not identifiable from unlabelled data. That is why the prevalence-informed variant, which needs one aggregate number, exists.
3. **FMCIB features do not help once radiomics and position are present.**
4. **The ceiling on these two public cohorts sits at about 0.68 site-balanced AUC.** Thirty-one fusions across three harmonisation settings cluster at 0.64–0.68. This agrees with the published cross-dataset range for these collections (Yang et al. 2021: 0.54–0.64).

---

## 5. Files

**Code**

| File | Purpose |
|---|---|
| `src/phase10_harmonise.py` | ComBat, balanced ComBat, Saerens prior EM |
| `src/phase10_data.py` | cohort and feature-block assembly (read-only) |
| `src/phase10_cv.py` | pooled repeated nested CV and site-leakage diagnostic |
| `src/phase10_fusion.py` | all-combinations equal-weight fusion |
| `src/phase10_transport.py` | leave-one-site-out transport |
| `src/phase10_shortcut_demo.py` | naive vs honest demonstration |
| `src/phase10_extract_loc.py` | tumour-position features |
| `src/phase10_extract_shells.py` | peritumoral shells (runs in `.venv`) |
| `src/phase10_extract_fmcib.py` | FMCIB features |
| `src/phase10_figures.py` | figures and summary tables |

**Features:** `global_features/phase10/{loc,peri,fmcib}.csv`. Each has a `status` column and one row per patient, with no failures.

**Weights:** `models/phase10/fmcib_model_weights.torch` (SHA-256 `8cc92ec4…a4bc2`).

**Results** (all under `results/phase10/`)

| Path | Contents |
|---|---|
| `cv/<block>__<harm>/{oof.csv,metrics.json}` | per-fold hyper-parameters, site-leakage AUC, timings |
| `cv_v1/` | the first run, before the dense-ridge model was added; kept as a record |
| `fusion/` | all combinations, seed-averaged scores |
| `transport/` | per seed, per mapping |
| `transport_v1/` | the first transport run; kept as a record |
| `shortcut_demo/` | the naive vs honest demonstration |
| `figures/fig1–fig6` | presentation figures |
| `phase10_summary.json` | headline numbers |

**Logs:** `logs/phase10_*.log`.

---

## 6. Commands

```bash
# features (resumable; each skips patients already done)
.venv-phase10/Scripts/python.exe src/phase10_extract_loc.py --jobs 10      # ~4 min
.venv/Scripts/python.exe         src/phase10_extract_shells.py --jobs 12   # ~5 min
.venv-phase10/Scripts/python.exe src/phase10_extract_fmcib.py             # ~3 min, GPU
# models (each config skipped if its metrics.json exists)
.venv-phase10/Scripts/python.exe src/phase10_cv.py --featureset rim gtv peri loc_clin fmcib gtv_rim --jobs 14
.venv-phase10/Scripts/python.exe src/phase10_fusion.py
.venv-phase10/Scripts/python.exe src/phase10_transport.py --featureset rim peri loc_clin --jobs 10
.venv-phase10/Scripts/python.exe src/phase10_shortcut_demo.py
.venv-phase10/Scripts/python.exe src/phase10_figures.py
```

---

## 7. Deviations and decisions made during the phase (recorded, not hidden)

- **The dense ridge model was added after the first FMCIB run.** That run showed that top-k univariate selection discards distributed deep-feature signal. All blocks were then re-run with the four-model pipeline. The first run is kept in `cv_v1/`.
- **Peritumoral shells were added after the first fusion.** Rim was the strongest block, which motivated them (following Tang et al. 2022).
- **Lung segmentation is threshold-based, not `lungmask`.** Loading the external `lungmask` weights was not pursued. A single deterministic method on both cohorts also avoids a mask-source site confound.
- **The final model was chosen among cross-validated fusions** (see the caveat in §4.3).
