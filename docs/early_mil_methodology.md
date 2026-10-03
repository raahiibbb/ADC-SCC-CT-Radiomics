# NSCLC ADC vs SCC — Project Handoff and Proposed Methodology

**Purpose of this file:** This is the working methodology/handoff document for continuing the project in another ChatGPT conversation. Treat the decisions in the **Locked decisions** section as the current plan unless experimental evidence forces a change.

**Working title (provisional):**  
**Weakly Supervised Attention-Based Local Radiomics Across Tumoral and Peritumoral ROIs for CT-Based ADC vs SCC Classification**

---

## 1. Project objective

The target task is **binary histological subtype classification of non-small-cell lung cancer (NSCLC):**

- **Adenocarcinoma (ADC / LUAD)**
- **Squamous cell carcinoma (SCC / LUSC)**

The main idea is **not** to treat an entire tumor or peritumoral region as one homogeneous object. Instead, each ROI is divided into small **3D local patches**, radiomic features are extracted from each patch, and the patient is treated as a **bag of local instances**. An attention-based Multiple Instance Learning (MIL) model learns which local patches contribute most strongly to the patient-level ADC/SCC decision.

This combines:

1. **Vuong et al.** — five tumoral/peritumoral ROIs, local radiomics, and the observation that peritumoral/rim information can be useful for ADC/SCC classification [1].
2. **Lu et al. / CLAM** — weak supervision, patch-level representations, attention pooling, and attention maps without requiring patch-level labels [2].
3. **Tang et al.** — evidence that intratumoral and peritumoral radiomics are complementary and that tissue closer to the tumor boundary contains useful ADC/SCC information [3].
4. **Song et al., Yang et al., Wu et al., Pasini et al.** — classical radiomics evidence, feature-family importance, and warnings about overfitting/generalization [4–7].
5. **Chen et al. (web literature)** — precedent for combining radiomic features with attention-based MIL in lung cancer, but with **different nodules as instances**, not local spatial patches within a single tumor/peritumoral ROI [8].

The proposed novelty is therefore **not** "radiomics + attention MIL" by itself. The more defensible novelty is:

> **Representing each Vuong-defined tumor/peritumoral ROI as a bag of local 3D radiomic patches and learning patient-level ADC/SCC classification through weakly supervised attention, while mapping learned attention back to the CT to determine where subtype-discriminative information is concentrated.**

Do **not** claim "first-ever" without a dedicated PubMed/Scopus/Web of Science novelty search immediately before writing the paper.

---

# 2. Current project state

The user has already generated masks corresponding to the **five ROIs defined by Vuong et al.** for the full working cohort. **Do not regenerate the ROIs unless an error is found.**

### The five Vuong ROIs

Vuong et al. defined [1]:

1. **GTV** — gross tumor volume.
2. **lung_exterior** — 0.8 cm expansion from the GTV into lung tissue only.
3. **iso_exterior** — 0.8 cm expansion from the GTV into lung and soft tissue.
4. **gradient** — 0.4 cm contraction inside the GTV plus 0.8 cm expansion outside the GTV.
5. **GTV+Rim** — union of the GTV and its 0.8 cm isotropic exterior region.

Vuong's local analysis used **non-overlapping 3 × 3 × 3 voxel patches** after resampling to **3.75 mm isotropic voxels**, and they reported a significant ADC-vs-SCC difference in the rim activation ratio but not in the GTV for one important texture feature [1].

### Important cohort-count discrepancy to resolve before training

The current user report mentions **204 patients**, while the project dataset documentation states a refined ADC/SCC cohort of:

- **152 SCC**
- **51 ADC**
- **203 total patients**

Before any cross-validation split is generated, the next conversation must verify the real number of usable patients from the actual mask/metadata files and explain any extra/missing case. Do **not** silently assume 203 or 204.

---

# 3. Practical constraints that must shape the methodology

The methodology is intentionally simplified because this project has real resource limitations.

### Compute constraints

- Local GPU: **RTX 3050 Laptop GPU, 4 GB VRAM**.
- Local RAM: approximately **16 GB**.
- Kaggle GPU access is available, but runtime/availability is limited.
- PyRadiomics feature extraction is mainly CPU-bound; a large GPU is not required for the first model.

### Data/expert constraints

- Only a modest number of independent patients (~203/204).
- ADC is the minority class.
- No routine access to radiologists/pathologists for manual re-segmentation or visual validation.
- No requirement to perform multi-reader segmentation robustness experiments.
- External datasets may be considered later, but the **first priority is to test whether the proposed idea works on the current cohort**.

### Therefore, the first version should NOT require

- manual ROI correction by clinicians,
- inter-observer ICC experiments,
- thousands of filtered features per patch,
- full CLAM instance clustering,
- a large 3D CNN,
- massive hyperparameter optimization,
- multiple external validation datasets,
- ComBat harmonization,
- patch-level ANOVA treating patches as independent patients.

---

# 4. Key conceptual point: what MIL means here

**MIL = Multiple Instance Learning.**

One **patient** is one **bag**.

The many local 3D patches inside the chosen ROI are the **instances** in that bag.

For patient \(j\):

\[
B_j = \{x_{j1}, x_{j2}, \ldots, x_{jN_j}\}
\]

where each \(x_{ji}\) is the radiomic feature vector of one local patch.

Only the patient label is known:

\[
y_j \in \{\mathrm{ADC}, \mathrm{SCC}\}
\]

We do **not** assume:

\[
x_{j1}=\mathrm{ADC},\quad x_{j2}=\mathrm{ADC},\quad \ldots
\]

even if the patient is ADC.

That distinction is crucial. Some patches may be highly informative, while others may contain weakly discriminative tumor, lung, vessel, soft tissue, necrotic, or boundary patterns.

Lu et al. used the same general weak-supervision philosophy for pathology slides: slide-level labels were known while individual image patches were not manually labeled [2].

---

# 5. Core experimental strategy

The study should be developed in **two stages**:

## Stage A — proof of concept (do this first)

Use **only the Gradient ROI**.

Pipeline:

\[
\text{CT + Gradient Mask}
\rightarrow
\text{3D local patches}
\rightarrow
\text{compact radiomics}
\rightarrow
\text{MIL}
\rightarrow
\text{ADC/SCC}
\]

Compare:

1. **Mean-pooling MIL**
2. **Attention-pooling MIL**

The purpose of Stage A is to answer one question:

> **Does learned attention over local gradient radiomics outperform treating all local patches equally?**

Do not spend weeks implementing every ROI or CNN before answering this.

---

## Stage B — full study

If Stage A is promising, repeat the same fixed pipeline independently for all five Vuong ROIs:

- GTV
- lung_exterior
- iso_exterior
- gradient
- GTV+Rim

This gives a clean table:

| ROI | Mean MIL | Attention MIL |
|---|---:|---:|
| GTV | AUC | AUC |
| lung_exterior | AUC | AUC |
| iso_exterior | AUC | AUC |
| gradient | AUC | AUC |
| GTV+Rim | AUC | AUC |

The architecture and preprocessing should remain the same across ROIs so the comparison remains meaningful.

---

# 6. Preprocessing for the first experiment

## 6.1 Input

For each patient:

- pretreatment CT volume,
- selected ROI mask,
- patient-level histology label (ADC or SCC).

Use the already-generated masks. No manual correction is required for the initial study.

---

## 6.2 Resampling

For the first implementation, resample CT and mask to:

\[
\boxed{2 \times 2 \times 2\ \text{mm}^3}
\]

Recommended interpolation:

- CT: linear interpolation,
- mask: nearest-neighbor interpolation.

### Rationale

Vuong et al. used 3.75 mm isotropic voxels [1]. Song et al. used 1 mm resampling in a much larger multi-dataset study [4]. A 2 mm grid is a practical compromise for this project: it reduces computation while preserving more local detail than 3.75 mm.

This is a **project design choice**, not a parameter copied directly from Vuong.

---

## 6.3 HU handling

Vuong restricted radiomic analysis to approximately:

\[
-1024 \le HU \le 200
\]

to reduce the effect of dense structures that could not be manually excluded [1].

Because the current project cannot manually inspect and clean every ROI, use one reproducible intensity policy for all patients. A reasonable first implementation is:

- preserve CT in HU,
- use a PyRadiomics resegmentation range of approximately **[-1024, 200] HU** for feature calculation,
- document that the automatically generated ROIs were not manually cleaned for every bronchus/bone interface.

Do not claim this is identical to Vuong: Vuong also manually excluded disturbing structures such as large air cavities and dense structures [1].

---

# 7. Patch extraction

## 7.1 Initial patch size

After 2 mm isotropic resampling, use:

\[
\boxed{5\times5\times5\ \text{voxels}}
\]

which corresponds to approximately:

\[
\boxed{10\times10\times10\ \text{mm}^3}
\]

This is physically similar to Vuong's local patch neighborhood:

\[
3 \times 3.75\ \text{mm} \approx 11.25\ \text{mm}
\]

per dimension [1].

---

## 7.2 Initial patch stride

Start with **non-overlapping patches**:

\[
\text{stride}=5\ \text{voxels}
\]

Reasons:

- substantially fewer patches,
- lower feature-extraction time,
- lower storage requirements,
- lower redundancy between neighboring patches,
- closer to Vuong's original local-radiomics experiment [1].

**Do not use 50–95% overlap for the first experiment.**

If attention MIL works, overlap can be tested later as an ablation or used only for smoother final attention-map visualization. Lu et al. used overlapping patches for fine-grained attention heatmaps, but overlap is not required to establish the basic MIL result [2].

---

## 7.3 Patch validity rule

Do not keep patches with almost no useful ROI content.

Initial automatic rule:

- patch center must fall inside the selected ROI,
- intersection between the patch and ROI must contain at least **27 valid ROI voxels**,
- skip the patch if PyRadiomics cannot compute the required texture features or returns invalid values.

Store the following for every retained patch:

- patient ID,
- ROI name,
- patch index,
- patch center \(x,y,z\) in image/world coordinates,
- physical bounding box,
- number/fraction of ROI voxels,
- radiomic feature vector.

The saved coordinates are essential for attention maps later.

---

# 8. Local radiomic feature extraction

## 8.1 Do not start with 1000+ features per patch

Song extracted 1409 global tumor features and found intensity, GLCM, GLRLM and GLSZM particularly important for ADC/SCC discrimination [4]. For hundreds of local patches per cohort, thousands of features per patch would be unnecessarily expensive and increase overfitting risk.

For the proof-of-concept model use only **original-image** features:

1. **First-order**
2. **GLCM**
3. **GLRLM**
4. **GLSZM**

With PyRadiomics this is roughly **70–80 features per patch** depending on software version/settings.

### Initially exclude

- shape features — every extracted patch has an artificial predefined cube shape, so patch shape is not a biological tumor-shape descriptor,
- wavelet features,
- LoG features,
- square/square-root/logarithm/exponential filters,
- LBP,
- large filtered feature banks.

Tang and Song show that filtered features can be useful [3,4], but they should be an **optional later experiment**, not part of the first proof of concept.

---

## 8.2 Discretization

Use one fixed discretization rule across the entire cohort.

Recommended starting choice:

\[
\boxed{\text{binWidth}=20\ HU}
\]

Vuong used a fixed 20 HU bin size for texture calculations [1].

Record the PyRadiomics version and all parameters in a YAML file so the experiment is reproducible. IBSI emphasizes standardized definitions/reporting of radiomic feature computation [9].

---

# 9. Feature cleaning and normalization

The first model should avoid complicated feature-selection pipelines.

## Required

1. Remove feature columns that are invalid for all/most patches.
2. Replace/handle isolated nonfinite values using a rule learned only from the training fold.
3. Z-score features:

\[
x' = \frac{x-\mu_{\text{train}}}{\sigma_{\text{train}}}
\]

where the mean and standard deviation are calculated from **training patients only**.

The test fold must never participate in normalization.

## Optional later

If severe instability or overfitting occurs, add a cheap correlation filter (for example \(|\rho|>0.95\)) inside the training fold.

Do **not** make inter-observer ICC robustness filtering mandatory for this project.

---

# 10. Why patch-level ANOVA is not recommended as the main method

A patient can produce tens or hundreds of patches, but those patches are **not independent patients**.

For example:

- ADC patient A → 80 patches
- ADC patient B → 65 patches

This does **not** create 145 independent ADC subjects.

Running a simple ADC-vs-SCC ANOVA over all patches can produce pseudo-replication and misleading p-values because patches from the same patient are correlated.

Therefore:

> **Do not assign the patient's ADC/SCC label to every patch and perform ordinary patch-level ANOVA as if all patches were independent samples.**

If statistical feature analysis is desired later, use patient-level summaries or a mixed-effects approach. It is not needed for the first attention-MIL test.

---

# 11. Proposed lightweight attention-MIL model

This model is intentionally small. It operates on radiomic vectors, not raw CT images.

Suppose each patch has \(F\approx74\) features.

## 11.1 Patch encoder

Example:

\[
F \rightarrow 64 \rightarrow 32
\]

Implementation:

- Linear(F, 64)
- ReLU
- Dropout(0.2)
- Linear(64, 32)
- ReLU

This produces one embedding \(h_i \in \mathbb{R}^{32}\) for each patch.

---

## 11.2 Mean-pooling baseline

For each patient:

\[
z_{\text{mean}}=\frac{1}{N}\sum_{i=1}^{N}h_i
\]

Then:

\[
z_{\text{mean}}\rightarrow \text{linear classifier}\rightarrow \text{ADC/SCC}
\]

This baseline asks:

> Are local radiomics useful if every patch is treated equally?

---

## 11.3 Attention pooling — proposed model

Use a small gated attention mechanism inspired by attention-based MIL/CLAM [2]:

\[
s_i = w^T[\tanh(Vh_i)\odot \sigma(Uh_i)]
\]

Normalize across all patches of the patient:

\[
\alpha_i=\frac{e^{s_i}}{\sum_j e^{s_j}}
\]

Then:

\[
z_{\text{att}}=\sum_i\alpha_i h_i
\]

Finally:

\[
z_{\text{att}}\rightarrow \text{linear classifier}\rightarrow \text{ADC/SCC}
\]

### Interpretation

\(\alpha_i\) is the relative contribution/attention assigned to patch \(i\).

A high attention score means:

> the model relied more strongly on this patch when forming the patient-level representation.

It does **not** mean the patch is pathologically proven to contain ADC/SCC morphology.

---

## 11.4 Do not implement full CLAM initially

Lu's full CLAM includes class-specific attention branches and an instance-level clustering loss with pseudo-labels [2].

For this project, initially borrow only the central ideas:

- patient-level weak supervision,
- local instances,
- trainable attention pooling,
- spatial attention visualization.

Only add CLAM-style instance clustering if the simple attention model works and there is a clear reason to test it.

---

# 12. Class imbalance

ADC is the minority class.

For the first model:

- do **not** use patch-level SMOTE,
- do **not** synthesize local patches.

Use:

- patient-level stratified folds,
- **weighted binary cross-entropy**.

If ADC is encoded as positive class:

\[
\text{pos\_weight}=
\frac{N_{\text{SCC, train}}}
{N_{\text{ADC, train}}}
\]

Calculate this separately inside each training fold.

Liu et al. showed that SMOTE can materially change classification performance [10], but synthetic oversampling is not necessary for the first MIL experiment and can be evaluated later if needed.

---

# 13. Validation design

## 13.1 The split unit is always the patient

This is non-negotiable.

All patches from one patient must remain in the same split.

Wrong:

- Patient X patches in training
- Patient X other patches in testing

Correct:

- Entire Patient X is either train/validation/test for that fold.

Patch leakage would produce falsely high performance.

---

## 13.2 Initial evaluation

Use:

\[
\boxed{\text{Stratified 5-fold cross-validation}}
\]

at the patient level.

Suggested first implementation:

- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`
- within each training fold, reserve ~10–15% of training patients as a validation subset for early stopping,
- fit normalization and any preprocessing only on the true training subset.

The first goal is **proof of concept**, not an expensive nested-CV study.

If results are promising and training is cheap, the final study can repeat 5-fold CV over 3 random seeds to estimate variability.

---

# 14. Training settings for the lightweight MIL model

Suggested starting values:

- optimizer: Adam
- learning rate: \(1\times10^{-3}\)
- weight decay: \(1\times10^{-4}\)
- dropout: 0.2
- epochs: maximum 100
- early stopping patience: 10–15 epochs
- loss: weighted `BCEWithLogitsLoss`
- batch unit: one patient bag or a very small number of bags
- save the best validation checkpoint per fold

Do not perform a huge hyperparameter search.

If tuning is required, change only one or two parameters, such as:

- learning rate \(10^{-3}\) vs \(10^{-4}\),
- hidden dimension 32 vs 64,
- dropout 0.2 vs 0.4.

---

# 15. Metrics

Report patient-level metrics, not patch-level accuracy.

Minimum:

- ROC-AUC
- balanced accuracy
- sensitivity/recall
- specificity
- precision
- F1 score
- MCC
- confusion matrix

Because the dataset is imbalanced, also report:

- PR-AUC if straightforward to compute.

For cross-validation, report:

\[
\text{mean}\pm\text{standard deviation}
\]

across folds.

Yang et al. demonstrated that generalization can deteriorate strongly across different datasets even when internal models appear useful [5]. Song et al. similarly emphasized multi-dataset generalization [4]. Therefore, do not interpret one unusually good fold as the final result.

---

# 16. Attention maps

Once the attention model is trained:

For each patient, save:

- patch coordinates,
- attention score \(\alpha_i\),
- predicted probability,
- true class.

Map the patch attention values back into the original/resampled CT coordinate system.

Initial visualization:

- assign each patch's attention score uniformly to its patch region,
- average values where patches overlap if overlap is introduced later,
- overlay the resulting attention map on CT.

Lu et al. demonstrated that attention scores can be spatially visualized and that overlapping patches can create fine-grained heatmaps [2].

### Important wording

Use:

> "model attention / model importance"

not:

> "pathologically confirmed discriminative tissue."

No expert annotation is required to generate the map; expert validation can be listed as future work.

---

# 17. Five-ROI experiment after Gradient proof of concept

After the Gradient model is working, run the **same pipeline** separately for:

1. GTV
2. lung_exterior
3. iso_exterior
4. gradient
5. GTV+Rim

Keep fixed:

- resampling,
- patch physical size,
- feature families,
- model architecture,
- loss,
- cross-validation procedure.

This tests whether the location of local information matters.

Vuong reported that iso_exterior, gradient and GTV+Rim showed useful ADC/SCC performance while GTV and lung_exterior were weaker in validation [1]. The proposed study asks whether **local learned attention** changes or strengthens that result.

---

# 18. High-value analysis: where does attention concentrate?

For the Gradient ROI, use the known GTV boundary to automatically characterize where high-attention patches are located.

Possible bands:

- **inner boundary:** -4 to 0 mm
- **immediate outer rim:** 0 to +4 mm
- **outer rim:** +4 to +8 mm

For every patient, calculate quantities such as:

\[
A_r=\sum_{i\in r}\alpha_i
\]

for each radial region \(r\).

Then compare ADC vs SCC at the **patient level**.

This provides an interpretable spatial result without manual medical annotation.

Tang et al. found that the 0–5 mm peritumoral ring produced more significant radiomic differences than the 5–10 mm ring [3]. Vuong also found a local rim-associated difference between ADC and SCC [1]. The proposed learned-attention analysis is a direct extension of that idea.

---

# 19. Conventional baselines — add after the first proposed model works

The priority is to test the local attention model first. Once it runs reliably, add simple baselines.

## Baseline 1 — global GTV radiomics

\[
\text{whole GTV radiomics}\rightarrow\text{Logistic Regression}\rightarrow\text{ADC/SCC}
\]

## Baseline 2 — global Gradient radiomics

\[
\text{whole Gradient radiomics}\rightarrow\text{Logistic Regression}\rightarrow\text{ADC/SCC}
\]

## Baseline 3 — mean-pooling local MIL

Already part of Stage A.

These provide a clear experimental story:

1. whole tumor,
2. whole boundary region,
3. local boundary representation without learned weighting,
4. local boundary representation with learned attention.

---

# 20. CNN: why it is not the first model, and how to add it later

A CNN is also a neural network. The first proposed model uses an **MLP** because the input is a vector of handcrafted radiomic features. A CNN is most appropriate when the input is the raw image patch.

## Why not start with a CNN?

The main limitation is not only GPU memory. There are only ~203/204 independent patients. Thousands of patches do not create thousands of independent patient labels. A raw-image CNN has far more capacity to overfit than the compact radiomics-MIL model.

The radiomics-MIL proof of concept is therefore useful because it asks whether the **local spatial hypothesis itself** has signal before adding a more complex feature extractor.

---

## Optional Phase 2 — CNN-MIL

If radiomics attention MIL is promising:

\[
\text{raw CT patch}
\rightarrow
\text{small CNN}
\rightarrow
\text{patch embedding}
\rightarrow
\text{attention MIL}
\rightarrow
\text{ADC/SCC}
\]

### Compute-friendly strategy

Do **not** keep an entire large bag of 3D patches inside GPU memory during end-to-end training at first.

Instead:

1. extract raw patches,
2. run the CNN in ordinary mini-batches,
3. save each patch's embedding,
4. train the attention-MIL model on saved embeddings.

This separates expensive image feature extraction from cheap MIL training.

### Kaggle role

Kaggle GPU is most useful for this CNN embedding stage.

The radiomics-MIL model itself is tiny and should fit on the local GPU or even CPU.

---

## Optional Phase 3 — radiomics + CNN fusion

For each patch:

\[
r_i=\text{radiomic vector}
\]

\[
c_i=\text{CNN embedding}
\]

Then:

\[
h_i=[r_i;c_i]
\]

followed by attention MIL.

Compare:

1. radiomics-attention MIL,
2. CNN-attention MIL,
3. radiomics + CNN fusion attention MIL.

This is a later extension, not part of the first proof of concept.

---

# 21. What NOT to do in the first implementation

Do not:

- regenerate the five ROI masks without evidence of a problem,
- manually clean all 203/204 patients,
- require a radiologist for the first model,
- treat patches as independent patients,
- perform ordinary patch-level ANOVA for ADC/SCC,
- extract 1000–2000 features from every patch,
- start with wavelet + LoG + LBP + every PyRadiomics filter,
- implement full CLAM before simple attention works,
- train a large 3D CNN first,
- split patches randomly across train/test,
- use the test fold to normalize features,
- optimize hundreds of hyperparameter combinations,
- claim attention maps are pathological ground truth,
- claim novelty as simply "radiomics + attention MIL."

---

# 22. Success criteria for the proof of concept

The first experiment is successful if:

1. the pipeline can extract valid local features from most/all usable patients,
2. attention MIL trains stably without leakage,
3. attention MIL performs at least comparably to mean pooling across folds,
4. attention maps show nonuniform model weighting rather than random/uniform scores,
5. results are not driven by one fold only.

A plausible strong outcome would be a consistent improvement such as:

\[
AUC_{\text{mean MIL}} < AUC_{\text{attention MIL}}
\]

across most folds.

Do not demand an unrealistically high AUC before considering the method useful. A modest but consistent gain plus spatial interpretability may be scientifically more credible than a suspiciously high internal AUC.

---

# 23. Recommended implementation order

## Phase 0 — preflight

1. Verify the exact usable patient count (203 vs 204).
2. Verify ADC/SCC label mapping.
3. Verify all Gradient masks align with CT volumes.
4. Record PyRadiomics version and preprocessing parameters.
5. Freeze one patient-level 5-fold split file and save it.

## Phase 1 — Gradient local-radiomics dataset

1. Resample CT/mask to 2 mm isotropic.
2. Extract non-overlapping 5×5×5 voxel patches.
3. Apply patch validity rule.
4. Extract first-order + GLCM + GLRLM + GLSZM.
5. Save one compact feature file per patient.

Suggested structure:

```text
features_gradient/
    LUNG1-001.npz
    LUNG1-002.npz
    ...
```

Each file should contain approximately:

```python
features   # [N_patches, N_features]
coords     # [N_patches, 3]
roi_voxels # [N_patches]
label      # scalar
patient_id
feature_names
```

## Phase 2 — models

1. Mean-pooling MIL.
2. Attention-pooling MIL.
3. 5-fold patient-level evaluation.
4. Export fold metrics and predictions.

## Phase 3 — interpretation

1. Save attention scores.
2. Generate CT attention overlays.
3. Analyze attention by distance from GTV boundary.

## Phase 4 — all five ROIs

Repeat the fixed pipeline for all five ROIs.

## Phase 5 — classical baselines

Add global GTV and global Gradient radiomics + logistic regression.

## Phase 6 — optional CNN

Only after local radiomics MIL has established whether the hypothesis is promising.

---

# 24. Files/results that should be saved

Reproducibility will be much easier if the pipeline saves:

```text
project/
├── configs/
│   ├── radiomics.yaml
│   └── experiment.yaml
├── splits/
│   └── stratified_5fold.csv
├── patch_features/
│   ├── gradient/
│   ├── gtv/
│   ├── lung_exterior/
│   ├── iso_exterior/
│   └── gtv_rim/
├── models/
│   ├── mean_mil/
│   └── attention_mil/
├── predictions/
│   ├── fold_1.csv
│   └── ...
├── attention/
│   ├── attention_scores/
│   └── overlays/
└── results/
    ├── cv_summary.csv
    ├── roc_curves/
    └── confusion_matrices/
```

Do not repeatedly recalculate radiomics during model training. Extract once, save, and reuse.

---

# 25. Novelty position

The components individually already exist:

- **Vuong et al.**: peritumoral ROIs + local radiomics + threshold-based activation maps [1].
- **Lu et al.**: weakly supervised patch attention + attention heatmaps, but in pathology WSIs using CNN-derived patch representations [2].
- **Tang et al.**: intratumoral + peritumoral CT radiomics for LUAD/LUSC [3].
- **Chen et al.**: radiomics + attention MIL for lung cancer diagnosis, but with separate nodules as instances rather than microregions within one tumor-boundary ROI [8].

Therefore the candidate novelty should be framed narrowly:

> **A patient-level weakly supervised MIL formulation in which local 3D radiomic subregions sampled from predefined tumoral/peritumoral CT ROIs are the instances, allowing the model to learn spatially varying ADC/SCC evidence and generate local attention maps without patch-level histology labels.**

Possible additional contribution:

> **Systematic comparison of the same local attention-MIL framework across all five Vuong ROIs to determine which anatomical context contains the strongest subtype-discriminative local information.**

Possible interpretability contribution:

> **Quantification of learned attention as a function of distance from the GTV boundary.**

These are candidate novelty claims, not yet formal claims of priority.

---

# 26. How the major references support this methodology

### Vuong et al. [1]

Supports:

- the five ROI definitions,
- the Gradient concept,
- local 3D radiomics,
- use of fixed 20 HU discretization,
- peritumoral/rim relevance,
- spatial activation-map motivation.

Vuong extracted local features using non-overlapping 3×3×3 patches and found a significant ADC/SCC difference in the rim activation ratio for a texture feature, but not in the GTV.

### Lu et al. [2]

Supports:

- weak supervision,
- patient/slide as a bag,
- patches as instances,
- learned attention weights,
- attention pooling,
- spatial attention maps,
- avoiding the need for patch-level annotation.

Our project does **not** copy Lu's pathology CNN or full CLAM clustering in the first version.

### Tang et al. [3]

Supports:

- combining intra- and peritumoral information,
- the importance of tissue near the tumor boundary,
- local spatial heterogeneity as a plausible source of subtype information.

They reported more significant radiomic features in the closer 0–5 mm peritumoral ring than the 5–10 mm ring.

### Song et al. [4]

Supports:

- ADC/SCC radiomics feasibility,
- importance of intensity, GLCM, GLRLM and GLSZM feature families,
- need to care about generalization.

Their best multi-dataset model reached AUCs around 0.819, 0.823, and 0.804 on three test cohorts.

### Yang et al. [5]

Supports:

- the danger of poor cross-center generalization,
- the fact that single-dataset models can perform much worse on other datasets,
- caution against believing one favorable internal result.

### Pasini et al. [6]

Supports:

- sensitivity of radiomics results to dataset composition and preprocessing,
- difficulty identifying one universally stable feature subset,
- importance of transparent parameter reporting.

### Wu et al. [7]

Supports:

- classical evidence that radiomic features are associated with ADC/SCC histology,
- the historical baseline for CT-radiomics subtype classification.

### Chen et al. [8]

Important novelty-control reference.

Supports:

- radiomic features can serve as MIL instance representations,
- deep attention MIL can estimate instance importance in lung cancer.

But Chen's MIL instances were pulmonary nodules in a patient-level diagnosis problem; our proposed instances are local 3D spatial subregions inside a predefined tumoral/peritumoral ROI for ADC/SCC subtype classification.

### IBSI [9]

Supports:

- standardized radiomic feature definitions,
- detailed reporting of preprocessing, discretization, and feature extraction.

---

# 27. References

## Primary project/source papers

[1] Vuong, D., Tanadini-Lang, S., Wu, Z., et al. (2020). **Radiomics Feature Activation Maps as a New Tool for Signature Interpretability.** *Frontiers in Oncology, 10*, 578895.  
https://doi.org/10.3389/fonc.2020.578895

[2] Lu, M. Y., Williamson, D. F. K., Chen, T. Y., Chen, R. J., Barbieri, M., & Mahmood, F. (2021). **Data-efficient and weakly supervised computational pathology on whole-slide images.** *Nature Biomedical Engineering, 5*, 555–570.  
https://doi.org/10.1038/s41551-020-00682-w

[3] Tang, X., Huang, H., Du, P., Wang, L., Yin, H., & Xu, X. (2022). **Intratumoral and peritumoral CT-based radiomics strategy reveals distinct subtypes of non-small-cell lung cancer.** *Journal of Cancer Research and Clinical Oncology, 148*, 2247–2260.  
https://doi.org/10.1007/s00432-022-04015-z

[4] Song, F., Song, X., Feng, Y., et al. (2023). **Radiomics feature analysis and model research for predicting histopathological subtypes of non-small cell lung cancer on CT images: A multi-dataset study.** *Medical Physics, 50*, 4351–4365.  
https://doi.org/10.1002/mp.16233

[5] Yang, F., Chen, W., Wei, H., Zhang, X., Yuan, S., Qiao, X., & Chen, Y.-W. (2021). **Machine Learning for Histologic Subtype Classification of Non-Small Cell Lung Cancer: A Retrospective Multicenter Radiomics Study.** *Frontiers in Oncology, 10*, 608598.  
https://doi.org/10.3389/fonc.2020.608598

[6] Pasini, G., Stefano, A., Russo, G., Comelli, A., Marinozzi, F., & Bini, F. (2023). **Phenotyping the Histopathological Subtypes of Non-Small-Cell Lung Carcinoma: How Beneficial Is Radiomics?** *Diagnostics, 13*(6), 1167.  
https://doi.org/10.3390/diagnostics13061167

[7] Wu, W., Parmar, C., Grossmann, P., et al. (2016). **Exploratory Study to Identify Radiomics Classifiers for Lung Cancer Histology.** *Frontiers in Oncology, 6*, 71.  
https://doi.org/10.3389/fonc.2016.00071

[10] Liu, J., Cui, J., Liu, F., Yuan, Y., Guo, F., & Zhang, G. (2019). **Multi-subtype classification model for non-small cell lung cancer based on radiomics: SLS model.** *Medical Physics, 46*(7), 3091–3100.  
https://doi.org/10.1002/mp.13551

## Additional web-verified methodological references

[8] Chen, J., Zeng, H., Zhang, C., Shi, Z., Dekker, A., Wee, L., & Bermejo, I. (2022). **Lung cancer diagnosis using deep attention-based multiple instance learning and radiomics.** *Medical Physics, 49*(5), 3134–3143.  
https://doi.org/10.1002/mp.15539

[9] Zwanenburg, A., Vallières, M., Abdalah, M. A., et al. (2020). **The Image Biomarker Standardization Initiative: Standardized Quantitative Radiomics for High-Throughput Image-based Phenotyping.** *Radiology, 295*(2), 328–338.  
https://doi.org/10.1148/radiol.2020191145

---

# 28. Immediate instruction for the next ChatGPT conversation

Use the following prompt after attaching/providing this Markdown file:

> **Continue my NSCLC ADC-vs-SCC project using the attached methodology handoff as the current source of truth. Do not redesign the project from scratch unless you identify a concrete technical flaw. I already generated the five Vuong ROI masks for the full cohort. First, help me perform Phase 0 preflight: verify patient count/labels/mask availability, then design and implement the Gradient-only local-radiomics dataset generation exactly as specified. Keep patient-level leakage prevention as a hard requirement. Before writing code, summarize what files/paths you need from me.**

---

# 29. First decision gate

Do not proceed directly to CNNs or all five ROIs.

The next goal is:

\[
\boxed{
\text{Gradient ROI}
\rightarrow
\text{local radiomics}
\rightarrow
\text{Mean MIL vs Attention MIL}
\rightarrow
\text{5-fold patient-level CV}
}
\]

After obtaining that result, decide whether to:

- proceed to all five ROIs,
- adjust patch size/overlap,
- add filtered features,
- try CNN-MIL,
- try radiomics + CNN fusion.

This prevents unnecessary compute and keeps the research question testable.

---

## End of handoff
