"""Text, equations, tables and figure placement of the final report.
Markup in strings: **bold**, *italic*, $latex$ (inline equation),
{fig:x} {tab:x} {eq:x} cross-references, {cite:key1,key2} citations.
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------- source code
def _func(path, name, indent=0):
    """text of `def name` (or `class name`) up to the next block at the same indent."""
    lines = open(os.path.join(HERE, path), encoding="utf8").read().split("\n")
    pad = " " * indent
    start = next(i for i, l in enumerate(lines) if re.match(rf"{pad}(def|class) {name}\b", l))
    out = [lines[start]]
    for l in lines[start + 1:]:
        if l.strip() and not l.startswith(pad + " ") and not l.startswith(pad + "\t"):
            break
        out.append(l)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def _strip_doc(code):
    """drop triple-quoted docstrings to save space."""
    return re.sub(r'\n\s*"""[\s\S]*?"""', "", code)


def code_blocks():
    a = "\n\n".join([_func("phase10_cv.py", "sb_weights"), _func("phase10_cv.py", "sb_auc"),
                     _func("phase10_cv.py", "univariate_score"), _func("phase11_cv.py", "site_leak_auc")])
    b = "\n\n".join([_strip_doc(_func("phase10_cv.py", "select", 4)).replace("\n    ", "\n"),
                     _func("phase10_cv.py", "fit_model"), _func("phase10_cv.py", "score")])
    c = _strip_doc("\n\n".join([_func("phase10_harmonise.py", "_eb_prior"), _func("phase10_harmonise.py", "_eb_shrink")]))
    h = _strip_doc(_func("phase10_harmonise.py", "Harmoniser"))
    h = h.replace("    # ------------------------------------------------------------------ fit\n", "")
    h = h.replace("    # ------------------------------------------------------------ transform\n", "")
    h = h.replace("    # ---------------------------------------------- unseen, unlabelled site\n", "")
    d = "\n\n".join([_func("phase11_cv.py", "select_and_fit"), _func("phase11_cv.py", "ens")])
    e = _func("phase11_cv.py", "loso_one")
    return (a, b), (c, h), (d, e)


# ------------------------------------------------------------------ references
REFS = {
    "sung2021": "H. Sung *et al.*, \"Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries,\" *CA: A Cancer Journal for Clinicians*, vol. 71, no. 3, pp. 209-249, 2021.",
    "travis2015": "W. D. Travis *et al.*, \"The 2015 World Health Organization classification of lung tumors,\" *Journal of Thoracic Oncology*, vol. 10, no. 9, pp. 1243-1260, 2015.",
    "zhu2018": "X. Zhu *et al.*, \"Radiomic signature as a diagnostic factor for histologic subtype classification of non-small cell lung cancer,\" *European Radiology*, vol. 28, no. 7, pp. 2772-2778, 2018.",
    "yang2021": "F. Yang *et al.*, \"Machine learning for histologic subtype classification of non-small cell lung cancer: a retrospective multicenter radiomics study,\" *Frontiers in Oncology*, vol. 10, art. 608598, 2021.",
    "pasini2023": "G. Pasini, A. Stefano, G. Russo, A. Comelli, F. Marinozzi and F. Bini, \"Phenotyping the histopathological subtypes of non-small-cell lung carcinoma: how beneficial is radiomics?,\" *Diagnostics*, vol. 13, no. 6, art. 1167, 2023.",
    "song2023": "F. Song *et al.*, \"Radiomics feature analysis and model research for predicting histopathological subtypes of non-small cell lung cancer on CT images: a multi-dataset study,\" *Medical Physics*, vol. 50, no. 7, pp. 4351-4365, 2023.",
    "chaunzwa2021": "T. L. Chaunzwa *et al.*, \"Deep learning classification of lung cancer histology using CT images,\" *Scientific Reports*, vol. 11, art. 5471, 2021.",
    "chen2023": "K. Chen, M. Wang and Z. Song, \"Multi-task learning-based histologic subtype classification of non-small cell lung cancer,\" *La Radiologia Medica*, vol. 128, 2023, doi: 10.1007/s11547-023-01621-w.",
    "tang2022": "X. Tang, H. Huang, P. Du, L. Wang, H. Yin and X. Xu, \"Intratumoral and peritumoral CT-based radiomics strategy reveals distinct subtypes of non-small-cell lung cancer,\" *Journal of Cancer Research and Clinical Oncology*, vol. 148, pp. 2247-2260, 2022.",
    "vuong2020": "D. Vuong *et al.*, \"Radiomics feature activation maps as a new tool for signature interpretability,\" *Frontiers in Oncology*, vol. 10, art. 578895, 2020.",
    "ilse2018": "M. Ilse, J. Tomczak and M. Welling, \"Attention-based deep multiple instance learning,\" in *Proc. 35th International Conference on Machine Learning (ICML)*, PMLR 80, pp. 2127-2136, 2018.",
    "lu2021": "M. Y. Lu *et al.*, \"Data-efficient and weakly supervised computational pathology on whole-slide images,\" *Nature Biomedical Engineering*, vol. 5, pp. 555-570, 2021.",
    "zech2018": "J. R. Zech *et al.*, \"Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional study,\" *PLoS Medicine*, vol. 15, no. 11, art. e1002683, 2018.",
    "geirhos2020": "R. Geirhos *et al.*, \"Shortcut learning in deep neural networks,\" *Nature Machine Intelligence*, vol. 2, pp. 665-673, 2020.",
    "johnson2007": "W. E. Johnson, C. Li and A. Rabinovic, \"Adjusting batch effects in microarray expression data using empirical Bayes methods,\" *Biostatistics*, vol. 8, no. 1, pp. 118-127, 2007.",
    "orlhac2018": "F. Orlhac *et al.*, \"A postreconstruction harmonization method for multicenter radiomic studies in PET,\" *Journal of Nuclear Medicine*, vol. 59, no. 8, pp. 1321-1328, 2018.",
    "pyradiomics": "J. J. M. van Griethuysen *et al.*, \"Computational radiomics system to decode the radiographic phenotype,\" *Cancer Research*, vol. 77, no. 21, pp. e104-e107, 2017.",
    "ibsi": "A. Zwanenburg *et al.*, \"The Image Biomarker Standardization Initiative: standardized quantitative radiomics for high-throughput image-based phenotyping,\" *Radiology*, vol. 295, no. 2, pp. 328-338, 2020.",
    "medsam2": "J. Ma *et al.*, \"MedSAM2: Segment anything in 3D medical images and videos,\" arXiv:2504.03600, 2025.",
    "totalseg": "J. Wasserthal *et al.*, \"TotalSegmentator: robust segmentation of 104 anatomic structures in CT images,\" *Radiology: Artificial Intelligence*, vol. 5, no. 5, art. e230024, 2023.",
    "fmcib": "S. Pai *et al.*, \"Foundation model for cancer imaging biomarkers,\" *Nature Machine Intelligence*, vol. 6, pp. 354-367, 2024.",
    "ctfm": "S. Pai *et al.*, \"Vision foundation models for computed tomography,\" arXiv:2501.09001, 2025.",
    "ganin2016": "Y. Ganin *et al.*, \"Domain-adversarial training of neural networks,\" *Journal of Machine Learning Research*, vol. 17, no. 59, pp. 1-35, 2016.",
    "he2016": "K. He, X. Zhang, S. Ren and J. Sun, \"Deep residual learning for image recognition,\" in *Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 770-778, 2016.",
    "ke2017": "G. Ke *et al.*, \"LightGBM: a highly efficient gradient boosting decision tree,\" in *Advances in Neural Information Processing Systems 30 (NIPS)*, pp. 3146-3154, 2017.",
    "cortes1995": "C. Cortes and V. Vapnik, \"Support-vector networks,\" *Machine Learning*, vol. 20, pp. 273-297, 1995.",
    "sklearn": "F. Pedregosa *et al.*, \"Scikit-learn: machine learning in Python,\" *Journal of Machine Learning Research*, vol. 12, pp. 2825-2830, 2011.",
    "efron1993": "B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York: Chapman & Hall, 1993.",
    "varma2006": "S. Varma and R. Simon, \"Bias in error estimation when using cross-validation for model selection,\" *BMC Bioinformatics*, vol. 7, art. 91, 2006.",
    "mrmr": "C. Ding and H. Peng, \"Minimum redundancy feature selection from microarray gene expression data,\" *Journal of Bioinformatics and Computational Biology*, vol. 3, no. 2, pp. 185-205, 2005.",
    "lasso": "R. Tibshirani, \"Regression shrinkage and selection via the lasso,\" *Journal of the Royal Statistical Society: Series B*, vol. 58, no. 1, pp. 267-288, 1996.",
    "clark2013": "K. Clark *et al.*, \"The Cancer Imaging Archive (TCIA): maintaining and operating a public information repository,\" *Journal of Digital Imaging*, vol. 26, no. 6, pp. 1045-1057, 2013.",
    "idc": "A. Fedorov *et al.*, \"National Cancer Institute Imaging Data Commons: toward transparency, reproducibility, and scalability in imaging artificial intelligence,\" *RadioGraphics*, vol. 43, no. 12, art. e230180, 2023.",
    "aerts2014": "H. J. W. L. Aerts *et al.*, \"Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach,\" *Nature Communications*, vol. 5, art. 4006, 2014.",
    "bakr2018": "S. Bakr *et al.*, \"A radiogenomic dataset of non-small cell lung cancer,\" *Scientific Data*, vol. 5, art. 180202, 2018.",
    "lpcd": "P. Li *et al.*, \"A large-scale CT and PET/CT dataset for lung cancer diagnosis (Lung-PET-CT-Dx),\" The Cancer Imaging Archive, 2020, doi: 10.7937/TCIA.2020.NNC2-0461.",
    "nlst": "National Lung Screening Trial Research Team, \"Reduced lung-cancer mortality with low-dose computed tomographic screening,\" *New England Journal of Medicine*, vol. 365, no. 5, pp. 395-409, 2011.",
    "sybil": "P. G. Mikhael *et al.*, \"Sybil: a validated deep learning model to predict future lung cancer risk from a single low-dose chest computed tomography,\" *Journal of Clinical Oncology*, vol. 41, no. 12, pp. 2191-2200, 2023.",
    "lung3": "H. J. W. L. Aerts *et al.*, \"Data from NSCLC-Radiomics-Genomics,\" The Cancer Imaging Archive, 2015, doi: 10.7937/K9/TCIA.2015.L4FRET6Z.",
    "tcga": "B. Albertina *et al.* and S. Kirk *et al.*, \"The Cancer Genome Atlas Lung Adenocarcinoma (TCGA-LUAD) and Lung Squamous Cell Carcinoma (TCGA-LUSC) collections,\" The Cancer Imaging Archive, 2016, doi: 10.7937/K9/TCIA.2016.JGNIHEP5 and 10.7937/K9/TCIA.2016.TYGKKFMQ.",
}


# ===================================================================== content
def CONTENT(b):
    # ------------------------------------------------------------- abstract
    b.h1("Abstract")
    b.abstract([
        "Adenocarcinoma (ADC) and squamous cell carcinoma (SCC) are the two main types of non-small cell lung "
        "cancer, and they are treated differently. Today the type is confirmed by biopsy, which is invasive, while "
        "almost every patient already has a chest CT scan. In this project we built a fully automatic CT pipeline "
        "that predicts ADC versus SCC, and we designed its evaluation so that the result holds at hospitals the "
        "model has never seen.",
        "We first showed that pooling CT data from several hospitals creates a shortcut: hospitals differ both in "
        "scanner style and in their ADC/SCC mix, so a predictor that only knows the hospital name reaches a pooled "
        "ROC-AUC of 0.71 without looking at any image. To remove this effect we used a site-balanced AUC, ComBat "
        "harmonisation fitted inside every training fold, and leave-one-hospital-out testing. The tumour is located "
        "by an expert box or TotalSegmentator and outlined by MedSAM2. Texture features are extracted from three "
        "peritumoral shells (0-4, 4-8 and 8-12 mm) and combined with tumour position, size, age and sex. Two "
        "feature blocks are each modelled by an ensemble of logistic regression, SVM, LightGBM and ridge "
        "regression, and the block scores are fused.",
        "On 1041 patients from four public cohorts (697 ADC, 344 SCC) the model reached a site-balanced AUC of "
        "0.722 (95% CI 0.687-0.762) in repeated nested cross-validation and 0.714 on average for an unseen "
        "hospital. In two locked, pre-registered external tests it scored 0.686 on Lung3 (79 patients) and 0.585 on "
        "TCGA (84 patients). Thin-slice CT gave the best external results, while 5 mm CT was the main failure "
        "mode. Larger deep models (attention-MIL, CNN features, CT foundation models and adversarial fine-tuning) "
        "did not beat the simpler harmonised radiomics model at this data size.",
        "**Keywords:** lung cancer, histologic subtype, radiomics, ComBat harmonisation, multi-centre validation, "
        "site-balanced AUC, MedSAM2.",
    ])

    # ---------------------------------------------------------- introduction
    b.h1("Introduction")
    b.p("Lung cancer is the leading cause of cancer death worldwide {cite:sung2021}. About 85% of cases are "
        "non-small cell lung cancer (NSCLC), and its two most common histologic types are adenocarcinoma (ADC) "
        "and squamous cell carcinoma (SCC) {cite:travis2015}. Knowing the type matters because treatment depends "
        "on it. Some chemotherapy drugs are used for one type and not the other, and most targeted therapies "
        "(for example for EGFR and ALK changes) are relevant mainly to ADC. At present the type is confirmed by "
        "a biopsy: a needle through the chest wall, a bronchoscope, or surgery. A biopsy is invasive, carries "
        "risks such as collapsed lung and bleeding, and sometimes the sample is too small to decide.")
    b.p("Almost every lung cancer patient has a chest CT scan before any treatment. On CT, ADC and SCC show "
        "different tendencies. ADC is more often peripheral and may have fuzzy or ground-glass edges, while SCC "
        "is more often central, solid and larger, and is more common in men and heavy smokers. These are "
        "tendencies and not rules, so the two types often look alike. Radiomics tries to capture such "
        "differences by computing many quantitative features (intensity, texture and shape) from the tumour "
        "region and feeding them to a machine-learning model {cite:aerts2014,pyradiomics}. Our project question "
        "was therefore: can a computer tell ADC from SCC using only the CT scan, and does the answer still hold "
        "at hospitals that were not used for training?")
    b.p("The second half of this question became the centre of the project. Public CT collections come from "
        "different hospitals, and each hospital has its own scanners and its own mix of patients. In our data "
        "one hospital is 75% SCC while the other three are 73% to 81% ADC. A model trained on the pooled data can "
        "learn to recognise the hospital from scanner texture and then guess the majority class of that hospital. "
        "This is a well-known form of shortcut learning {cite:zech2018,geirhos2020}. It gives a good pooled "
        "score but fails at a new hospital. A large part of this work was spent on measuring this shortcut and "
        "on building an evaluation that it cannot fool.")
    b.sub("Objectives")
    b.p("The objectives of the project were:")
    b.bullets([
        "to build a fully automatic CT pipeline (tumour detection, outlining, feature extraction and "
        "classification) for ADC versus SCC;",
        "to measure the hospital shortcut and use a metric and validation design that are not affected by it;",
        "to reduce scanner differences between hospitals by harmonisation, fitted on training data only;",
        "to compare simple handcrafted models with deep-learning alternatives under the same honest protocol;",
        "to test the frozen model once on two external datasets that were locked until the end.",
    ], numbered=True)
    b.sub("Why this is a Complex Engineering Problem")
    b.p("{tab:cep} maps the project to the attributes of a "
        "complex engineering problem. The main difficulty is that there is no obvious solution: a higher score "
        "on pooled data can be a worse model, so success itself had to be defined carefully.")
    b.table(["Attribute", "How it appears in this project"], [
        ["P1: Depth of knowledge", "Needs CT physics (Hounsfield units, slice thickness, reconstruction kernels), "
         "image processing (resampling, segmentation), statistics (empirical Bayes, bootstrap) and machine learning."],
        ["P2: Conflicting requirements", "Higher pooled accuracy conflicts with fairness across hospitals; removing "
         "scanner effects can also remove real disease signal; model capacity conflicts with small data."],
        ["P3: Depth of analysis", "No closed-form answer. Many design choices (regions, features, harmonisation, "
         "models, fusion) had to be tested in nested cross-validation without leaking test information."],
        ["P4: Familiarity of issues", "Hidden confounding between hospital and label is not visible in standard "
         "metrics and is rarely checked in the literature."],
        ["P5: Extent of applicable codes", "Radiomics standards (IBSI), DICOM, data licences (CC BY, CC BY-NC) and "
         "good practice for medical AI evaluation (pre-registration, locked test sets)."],
        ["P6: Stakeholders", "Patients, radiologists, pathologists, oncologists and hospitals with different scanners."],
        ["P7: Interdependence", "Mask quality affects features, features affect harmonisation, harmonisation "
         "affects model selection; a change in one stage changes all later stages."],
    ], "Mapping of the project to the attributes of a complex engineering problem", "cep", widths=[1.6, 4.6])
    b.sub("Possible alternative solutions")
    b.p("Several other routes could address the same clinical need. "
        "{tab:alt} compares them with the approach we chose.")
    b.table(["Alternative", "Advantage", "Disadvantage"], [
        ["Biopsy only (current practice)", "Gold standard; also gives tissue for gene tests", "Invasive, risky, "
         "sometimes not diagnostic, slow"],
        ["Visual reading by a radiologist", "No new tools needed", "Subjective; ADC and SCC often look alike"],
        ["End-to-end deep CNN on CT", "Learns features automatically", "Needs thousands of patients; tends to learn "
         "scanner style (shown in Section 5.3)"],
        ["Single-hospital radiomics", "Simple, high internal scores", "Rarely transfers to other hospitals"],
        ["PET/CT based model", "Uses metabolic information", "PET is not available for every patient"],
        ["Multi-hospital harmonised radiomics (chosen)", "Works with routine CT, small models, honest "
         "multi-centre evaluation", "Moderate accuracy; sensitive to thick CT slices"],
    ], "Alternative solutions considered", "alt", widths=[1.8, 2.1, 2.3])

    # --------------------------------------------------------------- design
    b.h1("Design")
    b.h2("Problem Formulation (PO(b))")
    b.h3("Identification of Scope")
    b.p("The scope of the project was fixed after the progress presentation (Week 10). In scope:")
    b.bullets([
        "binary classification of ADC versus SCC from a pre-treatment chest CT scan and the patient's age and sex;",
        "public, de-identified CT collections from The Cancer Imaging Archive (TCIA) {cite:clark2013} and the NCI "
        "Imaging Data Commons (IDC) {cite:idc};",
        "fully automatic tumour localisation and outlining, so that the same method works at a new hospital;",
        "evaluation inside the development hospitals (cross-validation), on unseen development hospitals "
        "(leave-one-hospital-out) and on two locked external datasets.",
    ])
    b.p("Out of scope: other NSCLC types (large cell, not otherwise specified), PET images, gene mutation "
        "prediction, survival prediction, and any clinical deployment. The model is a research decision-support "
        "tool, not a replacement for biopsy.")

    b.h3("Literature Review")
    b.sub("Radiomics for histologic subtype")
    b.p("Zhu *et al.* {cite:zhu2018} used five LASSO-selected radiomic "
        "features on 129 patients from one hospital and reported an AUC of 0.91, but without external testing. "
        "Yang *et al.* {cite:yang2021} studied 645 patients from three centres. When centres were mixed randomly "
        "the AUC was about 0.78, but when the model was trained on one centre and tested on another it fell to "
        "0.54-0.64. Pasini *et al.* {cite:pasini2023} used LUNG1 and NSCLC-Radiogenomics (two of our cohorts) and "
        "found that accuracy fell from 0.77 to 0.59 after ComBat harmonisation, which suggests that part of the "
        "pooled performance came from dataset differences. Song *et al.* {cite:song2023} collected 868 patients from "
        "eight TCIA databases and reported AUCs of 0.82 internally and 0.82 and 0.80 on TCGA and Lung3, choosing "
        "the best of 130 radiomics models and using expert tumour masks.")
    b.sub("Deep learning")
    b.p("Chaunzwa *et al.* {cite:chaunzwa2021} trained a CNN on 311 patients from one hospital "
        "(internal AUC 0.71) and obtained 0.60 on an external Lung3 subset. Chen *et al.* {cite:chen2023} used a "
        "multi-task CNN on 402 TCIA patients (internal 0.84, external 0.73). Foundation models pre-trained on large "
        "CT collections, such as FMCIB {cite:fmcib} and CT-FM {cite:ctfm}, give general-purpose image features that "
        "can be reused without training a network from scratch.")
    b.sub("Peritumoral and local information")
    b.p("Tang *et al.* {cite:tang2022} showed that features from the region "
        "around the tumour add information to those from inside it. Vuong *et al.* {cite:vuong2020} used local "
        "radiomic activation maps to see where the signal lies inside and around the tumour. Weakly supervised "
        "attention-based multiple instance learning (MIL) {cite:ilse2018,lu2021} learns from many local patches "
        "when only a patient-level label exists; this was the starting idea of our project.")
    b.sub("Multi-centre effects")
    b.p("ComBat {cite:johnson2007} was designed to remove batch effects in gene "
        "expression data and has been adapted to radiomics {cite:orlhac2018}. Shortcut learning of hospital "
        "identity was documented in chest radiographs {cite:zech2018} and in deep networks in general "
        "{cite:geirhos2020}. Radiomic feature definitions are standardised by the IBSI {cite:ibsi}, but "
        "standard definitions do not remove scanner differences.")
    b.sub("Gap")
    b.p("Most published ADC/SCC models report high scores from one hospital or from random splits of "
        "pooled data. Very few measure how much of the score is explained by the hospital itself, and very few "
        "test a model that was frozen before the external labels were opened. Our project addresses this gap.")

    b.h3("Formulation of Problem")
    b.p("We have $N$ patients from $H$ hospitals. Patient $i$ has a CT volume $I_i$, a hospital label $h_i "
        r"\in \{1,\dots,H\}$ and a histology label $y_i \in \{0,1\}$, where $y_i=1$ means ADC (the positive "
        "class) and $y_i=0$ means SCC. We want a function $f$ that maps a CT scan to a real-valued score "
        r"$s_i=f(I_i)$, with $\hat{y}_i=\mathbb{1}[s_i>0]$. Ranking quality is measured by the ROC-AUC, which is "
        "the probability that a random ADC patient receives a higher score than a random SCC patient:")
    b.eq(r"\mathrm{AUC}=\frac{1}{n_1 n_0}\sum_{i:y_i=1}\ \sum_{j:y_j=0}\left[\mathbb{1}(s_i>s_j)+\frac{1}{2}\mathbb{1}(s_i=s_j)\right]", "auc")
    b.p("where $n_1$ and $n_0$ are the numbers of ADC and SCC patients. The real goal is not the AUC on pooled "
        "data but the AUC at a hospital $h^{*}$ that was not used for training:")
    b.eq(r"f^{*}=\underset{f}{\mathrm{arg\,max}}\ \mathbb{E}_{h^{*}\notin\mathcal{H}_{\mathrm{train}}}\left[\mathrm{AUC}_{h^{*}}(f)\right]", "goal")
    b.p("From this goal we derived the following design requirements:")
    b.bullets([
        "R1. The metric used for model selection must give 0.5 to any predictor that only knows the hospital.",
        "R2. Every fitted step (imputation, harmonisation, scaling, feature selection, hyperparameters) must "
        "use training patients only.",
        "R3. Tumour masks must be produced by the same automatic method at every hospital, so that the mask "
        "source cannot become a new hospital fingerprint.",
        "R4. A new hospital must be processed without its labels.",
        "R5. The model must be small enough to train on a laptop GPU (4 GB) and to avoid overfitting about "
        "1000 patients.",
        "R6. External tests must be pre-registered, frozen with a hash and scored once.",
    ])

    b.h3("Analysis")
    b.p("{fig:dataset} shows the ADC/SCC mix of every cohort. LUNG1 is 25% ADC, whereas Radiogenomics, "
        "Lung-PET-CT-Dx and NLST are 79%, 81% and 73% ADC. Consider a predictor that never looks at the image and "
        "outputs only the ADC rate of the patient's hospital:")
    b.fig("fig_dataset.png", "Class mix of the four development cohorts and the two locked external test sets. "
          "Numbers inside the bars are patient counts. LUNG1 is reversed compared with the other hospitals.",
          "dataset", width=5.6)
    b.eq(r"s_i=\pi_{h_i},\qquad \pi_h=\frac{n_{h,1}}{n_{h,1}+n_{h,0}}", "siteonly")
    b.p("Every ADC patient from a high-ADC hospital is ranked above every SCC patient from LUNG1, so this predictor "
        "obtains a pooled AUC of 0.710 on our 1041 patients ({fig:shortcut}). A naive radiomics model scores "
        "0.727, which is only slightly better than knowing the hospital. A pooled AUC around 0.7 therefore says "
        "very little about whether a model has learned anything about the tumour.")
    b.p("To satisfy requirement R1 we weight each patient by the inverse size of its (hospital, class) group:")
    b.eq(r"w_i=\frac{1}{H\cdot n_{h_i,y_i}}", "weights")
    b.p("and define the site-balanced AUC as the weighted version of {eq:auc}:")
    b.eq(r"\mathrm{AUC}_{\mathrm{SB}}=\frac{\sum_{i:y_i=1}\sum_{j:y_j=0}w_iw_j\left[\mathbb{1}(s_i>s_j)+\frac{1}{2}\mathbb{1}(s_i=s_j)\right]}{\sum_{i:y_i=1}w_i\ \sum_{j:y_j=0}w_j}", "sbauc")
    b.p("After weighting, each of the $2H$ (hospital, class) groups has the same total weight $1/H$. For the "
        "hospital-only predictor of {eq:siteonly}, pairs from the same hospital are ties and contribute 1/2. For "
        r"two different hospitals $h$ and $h'$, the ordered pairs (ADC from $h$, SCC from $h'$) and (ADC from $h'$, "
        r"SCC from $h$) together contribute exactly 1, whichever of $\pi_h$ and $\pi_{h'}$ is larger. Hence")
    b.eq(r"\mathrm{AUC}_{\mathrm{SB}}(\pi)=\frac{1}{H^{2}}\left[\sum_{h=1}^{H}\frac{1}{2}+\sum_{h<h'}1\right]=\frac{1}{H^{2}}\left[\frac{H}{2}+\frac{H(H-1)}{2}\right]=\frac{1}{2}", "proof")
    b.p("so the shortcut is worth exactly nothing under this metric, as required. On real data the hospital-only "
        "predictor scores 0.497 and the naive radiomics model 0.642 ({fig:shortcut}). The same weights $w_i$ are "
        "used as sample weights during training, so the classifiers are not rewarded for predicting the hospital "
        "either. The second part of the analysis concerns scanner style: a logistic regression trained to "
        "recognise the hospital from texture features reached an AUC of 0.81-0.91 (Section 5.3.3). This motivated "
        "harmonisation (Section 3.2.4).")
    b.fig("fig_shortcut.png", "The hospital shortcut. A predictor that only knows the hospital reaches a pooled "
          "AUC of 0.710 but exactly chance under the site-balanced AUC. The final model reaches 0.722 under the fair "
          "metric.", "shortcut", width=4.8)

    # ------------------------------------------------------- design method
    b.h2("Design Method (PO(a))")
    b.p("This section gives the mathematics behind every stage of the pipeline. The overall flow is shown later "
        "in {fig:pipeline}.")
    b.h3("Image Preprocessing")
    b.p("Each DICOM series is stacked into one 3-D volume and converted to NIfTI. Scans have pixel sizes of "
        "0.6-1 mm and slice gaps of 0.6-10 mm, so all volumes are resampled to an isotropic grid of "
        r"$2\times2\times2$ mm. The CT is interpolated linearly: the value at a new voxel position $\mathbf{x}$ is "
        "a weighted average of the 8 surrounding original voxels,")
    b.eq(r"I'(\mathbf{x})=\sum_{k=1}^{8}w_k(\mathbf{x})\,I(\mathbf{x}_k),\qquad \sum_{k=1}^{8}w_k(\mathbf{x})=1", "interp")
    b.p("where the weights are products of the fractional distances along each axis. Masks are resampled by "
        "nearest neighbour so that they stay binary. Intensities are then clipped to the range from air to soft "
        "tissue, and grey levels are grouped into bins of width $\\Delta=20$ HU before texture is computed:")
    b.eq(r"\tilde{I}(\mathbf{x})=\min\left(\max\left(I'(\mathbf{x}),-1024\right),200\right),\qquad b(\mathbf{x})=\left\lfloor\frac{\tilde{I}(\mathbf{x})+1024}{\Delta}\right\rfloor+1", "bins")
    b.p("Clipping removes the effect of bone and metal, and binning makes texture less sensitive to small noise. "
        "2 mm was chosen as a compromise: many scans have 3-5 mm slices, and a finer grid would only invent detail.")

    b.h3("Tumour Segmentation and Peritumoral Shells")
    b.p("For development cohorts the tumour location is an expert box (LUNG1 and Radiogenomics: the bounding box "
        "of the expert outline; Lung-PET-CT-Dx and NLST: expert boxes). For external data the box comes from "
        "the TotalSegmentator lung-nodule model {cite:totalseg}, taking the largest detected nodule. The box is "
        "given as a prompt to MedSAM2 {cite:medsam2}, a medical version of the Segment Anything 2 model. MedSAM2 "
        "treats the axial slices as frames of a video: it outlines the tumour on the key slice and propagates the "
        "outline up and down. Segmentation quality against the expert outline $B$ is measured by the Dice score,")
    b.eq(r"\mathrm{DSC}(A,B)=\frac{2\left|A\cap B\right|}{\left|A\right|+\left|B\right|}", "dice")
    b.p("which is 1 for identical masks and 0 for no overlap. From the tumour mask $G$ we compute the Euclidean "
        "distance of every voxel to the tumour and build three shells around it, with radii $(r_0,r_1,r_2,r_3)=(0,4,8,12)$ mm:")
    b.eq(r"d(\mathbf{x})=\min_{\mathbf{g}\in G}\left\|\mathbf{x}-\mathbf{g}\right\|_2,\qquad S_k=\left\{\mathbf{x}:\ r_{k-1}<d(\mathbf{x})\le r_k\right\}", "shells")
    b.p("The shells describe the tumour border and its surroundings, where ADC and SCC are expected to differ "
        "{cite:tang2022}.")

    b.h3("Radiomic and Position Features")
    b.p("Features are computed with PyRadiomics {cite:pyradiomics} on the original image, following IBSI "
        "definitions {cite:ibsi}. Each region gives 93 texture features from six families: first-order "
        "statistics, GLCM, GLRLM, GLSZM, GLDM and NGTDM. Three examples show the idea. If $p(i)$ is the fraction "
        "of voxels in bin $i$, $p(i,j)$ the normalised grey-level co-occurrence matrix and $P(i,j)$ the number of "
        "runs of grey level $i$ with length $j$, then")
    b.eq(r"H=-\sum_{i=1}^{N_g}p(i)\log_2\left(p(i)+\epsilon\right),\qquad \mathrm{Contrast}=\sum_{i=1}^{N_g}\sum_{j=1}^{N_g}(i-j)^2\,p(i,j)", "texture")
    b.eq(r"\mathrm{SRE}=\frac{\sum_{i}\sum_{j}P(i,j)/j^{2}}{\sum_{i}\sum_{j}P(i,j)}", "sre")
    b.p("Entropy $H$ measures randomness of intensity, GLCM contrast measures local intensity jumps, and short "
        "run emphasis (SRE) is large when the texture is fine. The final model uses two feature blocks:")
    b.bullets([
        "Shells block (279 features): 93 texture features in each of the three shells.",
        "Position + clinical block (13 features): left or right lung; relative lateral, front-back and "
        "up-down position of the tumour centre; normalised distance from the midline; depth of the centre (in mm "
        "and relative); fraction of the tumour surface touching the lung boundary and the medial (mediastinal) "
        "part of it; fraction of the tumour outside the lung; log tumour volume; age; sex.",
    ])
    b.p("Other blocks were tested as alternatives: whole-tumour texture plus 14 shape features (GTV, 107 "
        "features), an inner 4 mm ring inside the tumour, FMCIB deep features {cite:fmcib} and CT-FM features "
        "{cite:ctfm}.")

    b.h3("Harmonisation with ComBat")
    b.p("ComBat {cite:johnson2007} assumes that feature $g$ of patient $i$ in hospital $h$ is shifted and "
        "scaled by the hospital:")
    b.eq(r"x_{ihg}=\alpha_g+\sigma_g\gamma_{hg}+\sigma_g\sqrt{\delta_{hg}}\,\varepsilon_{ihg},\qquad \varepsilon_{ihg}\sim\mathcal{N}(0,1)", "combat")
    b.p(r"where $\alpha_g$ is the overall mean, $\sigma_g$ the pooled standard deviation, $\gamma_{hg}$ the "
        r"hospital's location shift and $\delta_{hg}$ its scale factor. After standardising with the pooled "
        "estimates, the raw shift and scale of each hospital are")
    b.eq(r"z_{ihg}=\frac{x_{ihg}-\hat{\alpha}_g}{\hat{\sigma}_g},\qquad \hat{\gamma}_{hg}=\frac{1}{n_h}\sum_{i\in h}z_{ihg},\qquad \hat{\delta}_{hg}=\frac{1}{n_h}\sum_{i\in h}\left(z_{ihg}-\hat{\gamma}_{hg}\right)^2", "combat_raw")
    b.p(r"Small hospitals give noisy estimates, so ComBat uses empirical Bayes: across all features of hospital "
        r"$h$ it assumes $\gamma_{hg}\sim\mathcal{N}(\bar{\gamma}_h,\bar{\tau}_h^2)$ and $\delta_{hg}\sim"
        r"\mathrm{InvGamma}(\lambda_h,\theta_h)$, estimates these priors from the data, and iterates the posterior "
        "means")
    b.eq(r"\gamma^{*}_{hg}=\frac{n_h\bar{\tau}^2_h\hat{\gamma}_{hg}+\delta^{*}_{hg}\bar{\gamma}_h}{n_h\bar{\tau}^2_h+\delta^{*}_{hg}},\qquad \delta^{*}_{hg}=\frac{\frac{1}{2}\sum_{i\in h}\left(z_{ihg}-\gamma^{*}_{hg}\right)^2+\theta_h}{\frac{n_h}{2}+\lambda_h-1}", "combat_eb")
    b.p("until they converge. The harmonised feature is")
    b.eq(r"x^{*}_{ihg}=\hat{\sigma}_g\,\frac{z_{ihg}-\gamma^{*}_{hg}}{\sqrt{\delta^{*}_{hg}}}+\hat{\alpha}_g", "combat_adj")
    b.p("ComBat is fitted on the training patients of each fold only. A hospital that was not in the training "
        r"data (an unseen or external hospital) gets its own $\hat{\gamma}$ and $\hat{\delta}$ from its unlabelled "
        "features with {eq:combat_raw}, so no test label is used. {fig:combat_demo} shows the effect on one real "
        "feature: before ComBat the LUNG1 curve is clearly shifted, after ComBat the four hospitals overlap.")
    b.fig("fig_combat_demo.png", "Effect of ComBat on one shell feature (10th-percentile HU in the 0-4 mm shell) "
          "for all 1041 patients. Left: each hospital has its own location and spread. Right: after ComBat the "
          "distributions are aligned. (Fitted on all patients for illustration only.)", "combat_demo", width=6.0)

    b.h3("Feature Scaling, Ranking and Selection")
    b.p("Inside each training fold, features with more than 20% missing values or zero variance are dropped and "
        "missing values are replaced by the training median. After ComBat every feature is z-scored with the "
        "training mean and standard deviation, "
        r"$\tilde{x}_{ig}=(x^{*}_{ig}-\mu^{\mathrm{tr}}_g)/s^{\mathrm{tr}}_g$. Features are then ranked by how "
        "well they separate ADC from SCC within each hospital:")
    b.eq(r"\rho_g=\frac{1}{H}\sum_{h=1}^{H}\left|\mathrm{AUC}_h(\tilde{x}_g)-0.5\right|", "rank")
    b.p(r"where $\mathrm{AUC}_h$ is computed with {eq:auc} on hospital $h$ only. Ranking within hospitals avoids "
        "choosing features that mainly separate hospitals. Going down the ranked list, a feature is skipped if its "
        r"absolute correlation with an already kept feature is above 0.9, $|r_{gk}|=|\frac{1}{n}\sum_i\tilde{x}_{ig}"
        r"\tilde{x}_{ik}|>0.9$. The first $k\in\{10,25,50\}$ kept features are used, and $k$ is a hyperparameter.")

    b.h3("Classifiers")
    b.p("Four classic models are trained on each block, all with the sample weights $w_i$ of {eq:weights} "
        "(implemented with scikit-learn {cite:sklearn} and LightGBM {cite:ke2017}).")
    b.p("(1) L2 logistic regression models the ADC probability with the sigmoid function and minimises the "
        "weighted log-loss plus an L2 penalty:")
    b.eq(r"p(\mathbf{x})=\frac{1}{1+e^{-(\boldsymbol{\beta}^{\top}\mathbf{x}+\beta_0)}}", "sigmoid")
    b.eq(r"\min_{\boldsymbol{\beta},\beta_0}\ \frac{1}{2}\|\boldsymbol{\beta}\|_2^2+C\sum_{i=1}^{n}w_i\left[-y_i\log p(\mathbf{x}_i)-(1-y_i)\log\left(1-p(\mathbf{x}_i)\right)\right]", "logreg")
    b.p("A small $C$ means a strong penalty, so the weights stay small and the model stays simple.")
    b.p(r"(2) Support vector machine {cite:cortes1995} with labels $t_i=2y_i-1\in\{-1,+1\}$ finds the "
        "boundary with the widest margin, allowing some violations $\\xi_i$:")
    b.eq(r"\min_{\mathbf{v},b,\boldsymbol{\xi}}\ \frac{1}{2}\|\mathbf{v}\|^2+C\sum_{i}w_i\xi_i\quad \mathrm{s.t.}\quad t_i\left(\mathbf{v}^{\top}\phi(\mathbf{x}_i)+b\right)\ge1-\xi_i,\ \ \xi_i\ge0", "svm")
    b.p("Its decision function uses the radial basis function (RBF) kernel, which allows a curved boundary:")
    b.eq(r"f(\mathbf{x})=\sum_{i}\alpha_i t_i K(\mathbf{x}_i,\mathbf{x})+b,\qquad K(\mathbf{x},\mathbf{x}')=\exp\left(-\gamma_K\|\mathbf{x}-\mathbf{x}'\|^2\right),\quad \gamma_K=\frac{1}{p\,\mathrm{Var}(X)}", "rbf")
    b.p("(3) LightGBM {cite:ke2017} builds 200 small decision trees one after another. Each new tree "
        "$h_m$ is fitted to the errors (negative gradients) of the current model and added with a learning "
        "rate $\\eta=0.03$, for $m=1,\\dots,200$:")
    b.eq(r"F_m(\mathbf{x})=F_{m-1}(\mathbf{x})+\eta\,h_m(\mathbf{x}),\qquad r_{im}=y_i-\sigma\left(F_{m-1}(\mathbf{x}_i)\right)", "boost")
    b.p("The trees have 4 or 8 leaves, at least 10 patients per leaf, and use 80% of patients and 50% of "
        "features per tree, which limits overfitting.")
    b.p("(4) Ridge logistic regression on all features is {eq:logreg} with every feature of the block "
        "(no top-k) and a very small $C\\in[10^{-4},10^{-2}]$. It captures weak signal that is spread thinly over "
        "many correlated features.")

    b.h3("Ensemble and Score Fusion")
    b.p("Each model gives a real-valued score $f_m(\\mathbf{x})$ (the decision function, or the probability for "
        "LightGBM). Because the four scores have different scales, each is standardised with the mean $\\mu_m$ "
        "and standard deviation $\\sigma_m$ of its training scores and the four are averaged into a block score:")
    b.eq(r"s_b(\mathbf{x})=\frac{1}{4}\sum_{m=1}^{4}\frac{f_m(\mathbf{x})-\mu_m}{\sigma_m}", "block")
    b.p("The two block scores are z-scored and averaged with equal, fixed weights, so nothing is fitted at this "
        "stage:")
    b.eq(r"s(\mathbf{x})=\frac{1}{2}\left[z\left(s_{\mathrm{shell}}(\mathbf{x})\right)+z\left(s_{\mathrm{pos}}(\mathbf{x})\right)\right],\qquad \hat{y}=\mathbb{1}\left[s(\mathbf{x})>0\right]", "fusion")
    b.p("A positive score means ADC. Since the training weights balance the classes, 0 is the natural threshold.")

    b.h3("Evaluation Metrics")
    b.p("The primary metric is the site-balanced AUC of {eq:sbauc}. For the unseen-hospital and external tests "
        "the ordinary AUC is used, since only one hospital is scored at a time. At the threshold of "
        "{eq:fusion} we also report")
    b.eq(r"\mathrm{Se}=\frac{TP}{TP+FN},\qquad \mathrm{Sp}=\frac{TN}{TN+FP},\qquad \mathrm{BA}=\frac{\mathrm{Se}+\mathrm{Sp}}{2}", "metrics")
    b.p("where ADC is the positive class. Uncertainty is given by a stratified bootstrap {cite:efron1993}: "
        "patients are resampled with replacement inside each (hospital, class) group $B=2000$ times, the metric "
        r"is recomputed each time, and the 95% confidence interval is $[\hat{\theta}^{*}_{(2.5)},\ "
        r"\hat{\theta}^{*}_{(97.5)}]$. The amount of hospital information left in the features is measured by a "
        "hospital detector: a one-versus-rest logistic regression trained to predict the hospital, scored on "
        "the test fold,")
    b.eq(r"\mathrm{Leak}=\frac{1}{H}\sum_{h=1}^{H}\mathrm{AUC}\left(\mathbb{1}[h_i=h],\ g_h(\tilde{\mathbf{x}}_i)\right)", "leak")
    b.p("A value near 0.5 means the hospital can no longer be recognised. For visualisation we also use PCA, "
        r"which projects the features onto the two eigenvectors $\mathbf{u}_1,\mathbf{u}_2$ of the covariance "
        r"matrix $\Sigma=\frac{1}{n}\tilde{X}^{\top}\tilde{X}$ with the largest eigenvalues.")

    # -------------------------------------------- block diagram (replaces circuit)
    b.h2("System Block Diagram")
    b.p("{fig:pipeline} shows the complete pipeline from a raw CT scan to the ADC/SCC decision. Every step is "
        "automatic, and the same frozen pipeline was applied to both external test sets. The four colours group "
        "the steps into image preparation, tumour and regions, features, and model.")
    b.fig("fig_pipeline.png", "Block diagram of the complete pipeline. Steps in the lower row are fitted on "
          "training data only and then frozen.", "pipeline", width=6.2)
    b.bullets([
        "Chest CT and preprocessing: DICOM to NIfTI, 2 mm isotropic resampling, HU clipping ({eq:interp}, {eq:bins}).",
        "Tumour detection and segmentation: expert box or TotalSegmentator box, then MedSAM2 outline.",
        "Peritumoral shells: three distance bands around the tumour ({eq:shells}).",
        "Feature extraction: 279 shell texture features and 13 position and clinical features.",
        "ComBat, selection and classifiers: {eq:combat} to {eq:boost}, inside each training fold.",
        "Decision: fused score $s$ of {eq:fusion}; $s>0$ means ADC.",
    ])

    # ---------------------------------------- model architecture (replaces simulation model)
    b.h2("Model Architecture")
    b.p("{fig:architecture} shows the classifier part in detail. It has two parallel branches, one for each "
        "feature block. Each branch has its own preparation chain and its own four-model ensemble. The design "
        "is intentionally small: each model sees at most 50 features (or all 279 for the ridge model), so the "
        "whole classifier has a few hundred parameters plus the trees, compared with millions in a CNN. "
        "{tab:arch} lists the settings.")
    b.fig("fig_architecture.png", "Architecture of the final classifier: two feature blocks, each with a "
          "preparation chain and a four-model ensemble, fused by averaging z-scored block scores.", "architecture",
          width=6.2)
    b.table(["Component", "Setting"], [
        ["Input per patient", "279 shell features + 13 position/clinical features"],
        ["Missing values", "drop feature if > 20% missing; else training median"],
        ["Harmonisation", "ComBat with empirical Bayes, per fold; label-free mapping of new hospitals"],
        ["Selection", "within-hospital AUC ranking, correlation pruning at 0.9, top-k with k in {10, 25, 50}"],
        ["Logistic regression", "L2, C in {0.01, 0.1, 1}"],
        ["SVM", "RBF kernel, gamma = scale, C in {0.1, 1, 10}"],
        ["LightGBM", "200 trees, learning rate 0.03, leaves in {4, 8}, min. 10 per leaf, subsample 0.8, feature fraction 0.5"],
        ["Ridge (all features)", "L2 logistic regression, C in {0.0001, 0.0003, 0.001, 0.003, 0.01}"],
        ["Sample weights", "1 / (H x n of the hospital-class group), {eq:weights}"],
        ["Block score", "mean of the four training-standardised scores, {eq:block}"],
        ["Final score", "mean of the two z-scored block scores; ADC if s > 0, {eq:fusion}"],
    ], "Settings of the final model", "arch", widths=[1.7, 4.5])

    # ---------------------------------------- training/validation (replaces CAD)
    b.h2("Training and Validation Design")
    b.p("Two complementary validation schemes were used ({fig:cv_scheme}).")
    b.sub("Repeated nested cross-validation")
    b.p("Patients are split into 5 folds, stratified by hospital and class "
        "so that every fold has all eight (hospital, class) groups. Four folds train the model and the fifth "
        "tests it. Inside the training part, an inner 3-fold cross-validation chooses the hyperparameters "
        "($k$, $C$, number of leaves) by the site-balanced AUC; the model is then refitted on the whole training "
        "part. The outer loop is repeated with 5 random seeds (42-46), giving 25 test runs. Nested "
        "cross-validation avoids the optimistic bias that appears when hyperparameters are tuned on the test "
        "fold {cite:varma2006}.")
    b.sub("Leave-one-hospital-out (LOSO)")
    b.p("The model is trained (with the same inner tuning) on three hospitals "
        "and tested on the fourth, which is mapped into the training reference by label-free ComBat. This is "
        "repeated for all four hospitals and is the closest simulation of using the model at a new hospital.")
    b.fig("fig_cv_scheme.png", "Validation design: (a) repeated nested cross-validation with an inner 3-fold "
          "loop for hyperparameter tuning; (b) leave-one-hospital-out testing.", "cv_scheme", width=6.2)
    b.table(["Model", "Hyperparameters searched", "Most often chosen (25 folds)"], [
        ["Logistic regression", "C in {0.01, 0.1, 1}; k in {10, 25, 50}", "C = 0.01 (strongest penalty)"],
        ["SVM (RBF)", "C in {0.1, 1, 10}; k in {10, 25, 50}", "C = 0.1"],
        ["LightGBM", "leaves in {4, 8}; k in {10, 25, 50}", "4 leaves"],
        ["Ridge (all)", "C in {0.0001, 0.0003, 0.001, 0.003, 0.01}", "C = 0.003 (shells), 0.01 (position)"],
    ], "Hyperparameter grid; the selection metric is the site-balanced AUC of the inner folds", "grid",
        widths=[1.4, 2.6, 2.2])

    # ---------------------------------------- external protocol (replaces PCB)
    b.h2("Locked External Test Protocol")
    b.p("A model that is tested, adjusted and tested again slowly turns its test set into a training set. To "
        "avoid this, the two external datasets were handled with the protocol of {fig:protocol}. Before any "
        "external image was processed, a pre-registration document fixed the model (shells + position, frozen "
        "after Week 13), the exclusion rules and the primary metric. The images were processed by the frozen "
        "automatic pipeline. The model was fitted once on all 1041 development patients and the new dataset was "
        "mapped as one new site with label-free ComBat. The prediction file was saved together with its SHA-256 "
        "hash, a 64-character fingerprint that changes completely if even one byte of the file changes. Only then "
        "were the labels opened, once, by a scoring script that refuses to run a second time.")
    b.fig("fig_protocol.png", "Protocol of the locked external tests. Labels are opened only after the "
          "predictions are frozen and hashed.", "protocol", width=6.2)

    # ---------------------------------------- source code (replaces firmware)
    b.h2("Core Source Code")
    b.p("The project was written in Python, and the complete code is available in the GitHub repository given "
        "in Section 7.3. The most important parts are listed below: the site-balanced metric and feature ranking "
        "({tab:code1}), the ComBat harmoniser ({tab:code2}), and nested model selection, ensembling and "
        "leave-one-hospital-out testing ({tab:code3}).")
    (a1, a2), (c1, c2), (d1, d2) = code_blocks()
    b.code(a1, a2, "Source code: site-balanced weights and AUC, within-hospital feature ranking, hospital "
           "detector (left); correlation-pruned top-k selection and the four classifiers (right)", "code1")
    b.code(c1, c2, "Source code: empirical-Bayes priors and shrinkage (left); ComBat harmoniser with label-free "
           "mapping of a new hospital (right)", "code2")
    b.code(d1, d2, "Source code: inner-CV hyperparameter selection and ensemble (left); leave-one-hospital-out "
           "test with label-free harmonisation (right)", "code3")

    # ------------------------------------------------------------ implementation
    b.h1("Implementation")
    b.h2("Description")
    b.sub("Environment")
    b.p("All work ran on one laptop. {tab:env} lists the hardware and main software. Three "
        "separate Python environments were needed because MedSAM2, TotalSegmentator and the main pipeline "
        "require different PyTorch versions.")
    b.table(["Item", "Details"], [
        ["Hardware", "Laptop, NVIDIA RTX 3050 GPU (4 GB), 16 GB RAM, Windows 11"],
        ["Language", "Python 3.9 (main pipeline), 3.12 (MedSAM2)"],
        ["Image I/O and resampling", "SimpleITK, pydicom, dcm2niix-style conversion, NumPy, SciPy"],
        ["Features", "PyRadiomics 3 (texture), own code for position features and shells"],
        ["Machine learning", "scikit-learn 1.5.2, LightGBM 4.6.0, PyTorch 2.6 (deep baselines), MONAI 1.3.2"],
        ["Segmentation", "MedSAM2 (box-prompted), TotalSegmentator 2.18 (lung nodules)"],
        ["Data download", "NBIA Data Retriever (TCIA), s5cmd and idc-index (IDC)"],
        ["Reporting", "Matplotlib, python-pptx, python-docx"],
    ], "Implementation environment", "env", widths=[1.8, 4.4])
    b.sub("Data preparation")
    b.p("The DICOM series were downloaded (about 90 GB in total), checked for geometry "
        "(spacing, orientation, missing slices) and converted to NIfTI. {fig:ct_before} shows one LUNG1 scan "
        "before and after preprocessing. In the original grid the coronal view looks squashed because the slices "
        "are 3 mm apart while pixels are about 1 mm; after resampling the anatomy has the correct proportions and "
        "the HU window makes lung and soft tissue easy to see.")
    b.fig("ct_before_after.png", "Preprocessing of patient LUNG1-001: original axial and coronal views (left) "
          "and the same views after 2 mm isotropic resampling and HU clipping (right).", "ct_before", width=4.6)
    b.sub("Segmentation")
    b.p("Our first segmentation method kept voxels above -750 HU inside the box and cleaned "
        "the result with morphology. It worked with tight boxes but leaked into the chest wall when the box was "
        "loose, which was the case for the NLST boxes. MedSAM2 was therefore adopted for all 1041 patients "
        "(38 minutes on the laptop GPU, no failures). The settings were chosen on 40 expert outlines using the "
        "worst-case Dice, without looking at any ADC/SCC label: lung window [-1350, 150] HU, full 512 x 512 "
        "slices, one box prompt on the slice with the largest box, propagation in both directions, and the result "
        "limited to the box enlarged by 3 mm. {fig:seg} shows an example and {fig:dice} the validation on all 343 "
        "expert outlines.")
    b.fig("seg_gallery.png", "Segmentation of one Radiogenomics patient: box prompt, old threshold method "
          "(Dice 0.67), MedSAM2 (Dice 0.88) and the expert outline.", "seg", width=6.2)
    b.fig("fig_dice.png", "Mean Dice against expert outlines for tight boxes and loose boxes (+8 mm on each side, "
          "which imitates an automatic detector). MedSAM2 is better in all four cases and much better with "
          "loose boxes.", "dice", width=4.8)
    b.sub("Feature extraction")
    b.p("Shells were built with a Euclidean distance transform ({eq:shells}); "
        "{fig:shells} shows them on a real scan. PyRadiomics extraction for all regions took about 20 minutes "
        "and position features about 13 minutes using parallel jobs, with no failures in the development data. "
        "Extraction was run once and the features were saved, so model training never recomputed them.")
    b.fig("shells_on_ct.png", "Tumour (GTV) and the three peritumoral shells at 0-4, 4-8 and 8-12 mm on an axial "
          "CT slice.", "shells", width=3.6)
    b.sub("Training")
    b.p("One full 5 x 5 nested cross-validation of one feature block takes about 10 seconds to a "
        "few minutes with 14 parallel jobs, because the models are small. All splits, random seeds, chosen "
        "hyperparameters, out-of-fold predictions, metrics and package versions were saved for reproducibility.")
    b.h2("Implementation Challenges")
    b.bullets([
        "Different DICOM types. Eight Lung-PET-CT-Dx patients had annotations only on a secondary-capture "
        "image without 3-D geometry; they were excluded by rule.",
        "Loose boxes in NLST. The NLST boxes were about three times larger than the tumour, so the threshold "
        "segmentation absorbed chest wall. This was solved by MedSAM2.",
        "Software conflicts. Installing one foundation-model package replaced the GPU version of PyTorch "
        "with a CPU version during a run. The environment was repaired, and later packages were installed without changing the existing ones.",
        "Large downloads on Windows. The s5cmd tool failed to rename files across drives, so downloads were "
        "run from the destination drive with relative paths.",
        "Limited GPU memory (4 GB). Fine-tuning used a frozen backbone with only the last block trained, and "
        "cached augmented crops.",
    ])

    # ------------------------------------------------------- analysis & evaluation
    b.h1("Design Analysis and Evaluation")
    b.h2("Novelty")
    b.p("We did not invent ComBat, MedSAM2 or weighted AUC individually. The novelty of the project lies in how "
        "they are combined and in how the model is evaluated:")
    b.bullets([
        "Explicit measurement of the hospital shortcut for ADC/SCC. We show that the hospital name alone "
        "gives a pooled AUC of 0.71 and use a site-balanced AUC that provably gives 0.5 to any hospital-only "
        "predictor ({eq:proof}). The honest score improved from 0.642 (naive model) to 0.722.",
        "Fully automatic, uniform tumour outlining. TotalSegmentator plus a single MedSAM2 box prompt gives "
        "masks at Dice 0.85 against experts, with the same method at every hospital so that the mask source is "
        "not a hidden fingerprint.",
        "Four hospitals, 1041 patients, three countries. To our knowledge this is one of the largest public "
        "multi-hospital sets used for this task, and it allowed leave-one-hospital-out testing (0.714).",
        "Locked, pre-registered external tests. The plan was written first, predictions were hashed before the "
        "labels were opened, and each test was scored once (Lung3 0.686, TCGA 0.585).",
        "A fair comparison of simple and deep models. Attention-MIL, CNN features, two foundation models and "
        "adversarial fine-tuning were all tested under the same protocol and all scored lower.",
    ])

    b.h2("Design Considerations (PO(c))")
    b.h3("Considerations to public health and safety")
    b.p("The model is designed as decision support. Its output is a score, not a diagnosis, and a biopsy is still "
        "needed for treatment decisions and gene testing. With a sensitivity of 0.61 and a specificity of 0.68 at "
        "the default threshold (Section 5.3.3), a wrong prediction is common enough that the score must never be "
        "used alone. Several safety choices follow from this: the threshold is fixed in advance, the model reports "
        "results per hospital, and we state clearly that it should not be used on 5 mm CT, where external "
        "performance was at chance. Because the pipeline uses only the CT that the patient already has, it adds "
        "no radiation dose and no procedural risk.")
    b.h3("Considerations to environment")
    b.p("We chose small classical models instead of large networks. Training the final model takes minutes on a "
        "laptop GPU, and the whole project used roughly 10-15 kWh of electricity (Section 8). Public data was "
        "reused instead of collecting new scans, which avoids extra scanning and its energy and radiation cost. "
        "Downloaded data was kept on one disk and processed once, and extracted features were stored so that "
        "experiments did not repeat expensive computation.")
    b.h3("Considerations to cultural and societal needs")
    b.p("In Bangladesh and similar countries, biopsy facilities and pathologists are concentrated in large "
        "hospitals, while CT scanners are more widely available. A CT-based tool could help prioritise patients "
        "for biopsy or guide the choice of sample. Smoking rates are high among men, which raises the share of SCC. "
        "Our data come from the Netherlands, the USA and China, so performance in a South Asian population is "
        "unknown and must be tested locally before any use. The site-balanced evaluation is also a societal "
        "consideration: it makes sure that a hospital with fewer patients is not ignored by the metric.")

    b.h2("Investigations (PO(d))")
    b.h3("Design of Experiment")
    b.p("Each experiment changed one part of the system while the folds, seeds and metric stayed the same, so "
        "that differences could be attributed to that part. {tab:doe} lists the experiments.")
    b.table(["#", "Question", "What was varied", "Metric"], [
        ["E1", "How strong is the hospital shortcut?", "hospital-only predictor vs naive vs final model", "pooled and site-balanced AUC"],
        ["E2", "Does attention over local patches help?", "mean vs gated-attention MIL, patch features, ROI", "AUC (LUNG1)"],
        ["E3", "Which feature block carries signal?", "7 blocks alone and 127 fusions", "site-balanced AUC, LOSO"],
        ["E4", "Does harmonisation remove the fingerprint?", "ComBat on/off", "hospital-detector AUC, AUC"],
        ["E5", "Do better masks help?", "threshold masks vs MedSAM2", "Dice, AUC"],
        ["E6", "Do more hospitals help?", "2, 3 and 4 training hospitals", "LOSO AUC"],
        ["E7", "Do deep features beat radiomics?", "FMCIB, CT-FM, fine-tuned CNN with adversary", "AUC, hospital-detector AUC"],
        ["E8", "Does the frozen model transfer?", "locked Lung3 and TCGA", "AUC with 95% CI"],
    ], "Design of experiments", "doe", widths=[0.35, 2.0, 2.25, 1.6], size=9)

    b.h3("Data Collection")
    b.p("All data are public and de-identified. {tab:data} summarises the six cohorts. The development set has "
        "1041 patients (697 ADC, 344 SCC) from four sources {cite:aerts2014,bakr2018,lpcd,nlst}. NLST tumour boxes "
        "come from the Sybil project {cite:sybil}. Lung3 {cite:lung3} and TCGA {cite:tcga} were reserved as "
        "external test sets and were not opened until the end.")
    b.table(["Cohort", "Country / setting", "Patients (ADC/SCC)", "Typical slice", "Tumour location", "Role"], [
        ["LUNG1 (NSCLC-Radiomics)", "Netherlands, radiotherapy, mostly stage III", "202 (51/151)", "3 mm", "expert outline", "development"],
        ["NSCLC-Radiogenomics", "USA, surgical", "141 (112/29)", "1.25 mm", "expert outline", "development"],
        ["Lung-PET-CT-Dx", "China, diagnostic", "301 (244/57)", "mostly 5 mm", "expert boxes", "development"],
        ["NLST", "USA, screening trial", "397 (290/107)", "2 mm", "expert boxes (Sybil)", "development"],
        ["Lung3 (Radiomics-Genomics)", "Netherlands, surgical", "79 (44/35)", "4-5 mm", "automatic", "locked test"],
        ["TCGA-LUAD / LUSC", "USA, 9 hospitals", "84 (48/36)", "5 mm (2/3 of scans)", "automatic", "locked test"],
    ], "Datasets used in this project", "data", widths=[1.3, 1.35, 0.9, 0.8, 0.9, 0.95], size=8.5)
    b.sub("Inclusion rules")
    b.p("Patients needed a pre-treatment CT series and a pathology label of ADC or SCC. "
        "Mixed or unclear types were excluded: in NLST 155 patients whose first cancer was not ADC or SCC and 29 "
        "with more than one lung primary; in Lung-PET-CT-Dx 8 patients with secondary-capture images; in Lung3 "
        "10 patients with NSCLC not otherwise specified, large cell neuroendocrine or mixed histology; in TCGA 19 "
        "patients without a usable CT and 1 with no detected nodule. The selection used only metadata, never "
        "image-based predictions.")

    b.h3("Results and Analysis")
    b.sub("Early approach: attention-MIL on local patches")
    b.p("The project began with a weakly supervised "
        "attention-MIL model on LUNG1 {cite:ilse2018,lu2021}, where each patient was a bag of 1 cm radiomic "
        "patches. {tab:mil} summarises these trials, which were shown at the progress presentation. Attention did "
        "not reliably beat simple mean pooling, and the best result after adding a second cohort (0.675) was later "
        "found to be partly inflated by the hospital shortcut, since a hospital detector reached AUC 0.998 on "
        "those two cohorts. This led to the redesign described in this report.")
    b.table(["Trial", "Setup", "Result (AUC)", "Finding"], [
        ["1", "Gradient ROI, 29,184 patches, 74 features", "0.604 / 0.568", "attention unstable, lost to mean pooling"],
        ["2", "GTV + 8 mm rim, 37,464 patches", "0.570 mean, 0.577 attention", "wider region did not clearly help"],
        ["3", "mRMR {cite:mrmr} / LASSO {cite:lasso} selection inside CV", "0.593", "small gain, CI included zero"],
        ["4", "Frozen ResNet-18 {cite:he2016} patch embeddings", "0.549 mean, 0.615 attention", "first clear attention gain"],
        ["5", "CLAM-style instance loss", "0.615 to 0.568", "sharper attention, lower AUC"],
        ["6", "Global GTV and rim radiomics, late fusion", "0.567 / 0.537", "global models also weak on LUNG1"],
        ["7", "LUNG1 + Radiogenomics merged", "0.675 (pooled)", "partly hospital shortcut"],
    ], "Summary of the early attention-MIL phase (before the progress presentation)", "mil",
        widths=[0.45, 2.3, 1.35, 2.1], size=9)
    b.sub("The hospital shortcut (E1)")
    b.p("As shown in {fig:shortcut}, the hospital name alone gives a pooled "
        "AUC of 0.710 and the naive radiomics model 0.727. Under the site-balanced AUC these become 0.497 and "
        "0.642. All later results use the site-balanced metric.")
    b.sub("Main cross-validation result")
    b.p("The final model reached a site-balanced AUC of 0.722 (95% CI "
        "0.687-0.762) over 5 seeds x 5 folds on 1041 patients, with a standard deviation over seeds below 0.01. "
        "{fig:roc} shows the ROC curve inside each hospital, built from out-of-fold predictions averaged over the "
        "seeds. Every curve is above the diagonal. Lung-PET-CT-Dx is the easiest hospital (0.85) and LUNG1 the "
        "hardest (0.63). In LUNG1, 78% of patients are stage III, and at that stage ADC and SCC tumours have "
        "similar sizes (median about 44 versus 48 ml), so size and position carry little information.")
    b.fig("roc_per_site.png", "ROC curves of the final model inside each hospital (out-of-fold predictions, "
          "5 x 5 cross-validation). The dashed line is random guessing.", "roc", width=4.4)
    b.p("{fig:scores} shows the distribution of the fused score by hospital and class. In every hospital the ADC "
        "box lies above the SCC box, but the overlap is large, which matches an AUC around 0.7. {tab:perhosp} and "
        "{fig:confusion} give the results at the fixed threshold $s=0$.")
    b.fig("fig_scores.png", "Fused out-of-fold score by hospital and class. Points are patients; the red line is "
          "the decision threshold.", "scores", width=5.6)
    b.table(["Hospital", "Patients (ADC/SCC)", "AUC", "Sensitivity", "Specificity", "Balanced acc."], [
        ["LUNG1", "202 (51/151)", "0.631", "0.490", "0.623", "0.556"],
        ["Radiogenomics", "141 (112/29)", "0.710", "0.616", "0.724", "0.670"],
        ["Lung-PET-CT-Dx", "301 (244/57)", "0.852", "0.561", "0.877", "0.719"],
        ["NLST", "397 (290/107)", "0.694", "0.662", "0.645", "0.653"],
        ["All (site-balanced)", "1041 (697/344)", "0.722", "0.607", "0.680", "0.644"],
    ], "Cross-validation results per hospital at the threshold s = 0", "perhosp",
        widths=[1.4, 1.2, 0.7, 0.9, 0.9, 1.0], align="lccccc")
    b.fig("fig_confusion.png", "Confusion matrices at the threshold s = 0 for cross-validation and the two "
          "locked external tests. Percentages are per true class.", "confusion", width=6.2)
    b.sub("Unseen hospitals (LOSO, E6)")
    b.p("{fig:loso} compares each hospital's score when it was part of the "
        "training data (pooled CV) with its score when it was left out completely. The drop is very small "
        "(at most 0.017), and the LOSO mean is 0.714, almost equal to the pooled 0.722. This suggests that the "
        "model does not rely on memorised hospital style. More hospitals helped: with three training hospitals "
        "the LOSO mean was 0.678, and adding NLST raised it to 0.704 with the earlier masks, mostly by improving "
        "the held-out Lung-PET-CT-Dx score from 0.735 to 0.839.")
    b.fig("fig_loso.png", "AUC of each hospital in pooled cross-validation (light bars) and when the hospital "
          "was left out of training (coloured bars).", "loso", width=5.0)
    b.sub("Harmonisation (E4)")
    b.p("{fig:pca} shows the shell features projected by PCA. Before ComBat, LUNG1 "
        "forms its own cloud; after ComBat the hospitals overlap. {fig:siteleak} measures this with the hospital "
        "detector of {eq:leak}: it fell from 0.81-0.91 to 0.56-0.64 for the radiomics and position blocks. The "
        "site-balanced AUC changed by only -0.004 to +0.020 per block. ComBat therefore does not raise the score; "
        "it makes the score more trustworthy, because the remaining performance cannot come from scanner style.")
    b.fig("pca_combat.png", "PCA of the 279 shell features before and after ComBat; each point is a patient, "
          "coloured by hospital.", "pca", width=5.8)
    b.fig("fig_siteleak.png", "Hospital-detector AUC before and after ComBat (three-hospital stage). 0.5 means "
          "the hospital cannot be recognised.", "siteleak", width=4.6)
    b.sub("Ablation (E3, E5)")
    b.p("{fig:ablation} shows the site-balanced AUC of each feature block and of the "
        "fusions, all with MedSAM2 masks and ComBat. The handcrafted blocks are close to each other (0.698-0.709), "
        "the two deep-feature blocks are clearly lower, and fusing shells with position gives the best "
        "pre-specified result. {tab:ablation} lists the effect of each design change.")
    b.fig("fig_ablation.png", "Ablation: site-balanced AUC of single feature blocks and fusions (5 x 5 CV, 1041 "
          "patients). Red bars are deep features. Note that the x-axis starts at 0.6.", "ablation", width=5.2)
    b.table(["Change", "Before", "After", "Conclusion"], [
        ["Threshold masks to MedSAM2 masks", "0.714", "0.722", "better masks give better features"],
        ["3 to 4 training hospitals (LOSO mean)", "0.682", "0.704", "more diverse data helps transfer"],
        ["Best single block to shells + position", "0.709", "0.722", "the two blocks are complementary"],
        ["Shells + position to all 7 blocks", "0.722", "0.720", "more features add noise, not signal"],
        ["ComBat off to on (hospital detector)", "0.81-0.91", "0.56-0.64", "fingerprint largely removed"],
        ["Single model to 4-model ensemble (shells)", "0.677-0.697", "0.700", "ensemble is more stable"],
    ], "Effect of individual design changes (site-balanced AUC unless stated)", "ablation",
        widths=[2.3, 0.9, 0.9, 2.1], align="lccl")
    b.sub("Ensemble and tuning")
    b.p("{fig:models} compares each classifier with the ensemble. No single model is "
        "best in both blocks (ridge is best for shells, SVM for position), but the four-model ensemble is better "
        "than every single model in both blocks, and fusing the two blocks adds a further 0.013. {fig:hparams} "
        "shows which hyperparameters the inner cross-validation chose in the 25 outer folds. The simplest option "
        "won almost every time: C = 0.01 for logistic regression, C = 0.1 for the SVM and 4-leaf trees. This is a "
        "sign that the data are noisy and small, and that more flexible models would mainly fit noise.")
    b.fig("fig_models.png", "Site-balanced AUC of each classifier, of the four-model ensemble per block, and of "
          "the fused final model.", "models", width=5.6)
    b.fig("fig_hparams.png", "How often each hyperparameter value was chosen by the inner 3-fold "
          "cross-validation in the 25 outer folds.", "hparams", width=6.2)
    b.sub("Deep-learning alternatives (E7)")
    b.p("{tab:deep} lists the larger models that were tried under the "
        "same protocol. Frozen FMCIB features {cite:fmcib} reached 0.675. CT-FM {cite:ctfm} reached 0.662 and kept "
        "hospital information even after ComBat (detector 0.765), probably because its whole-CT pre-training "
        "encodes scanner style. Fine-tuning the last block of FMCIB (35 million parameters) with a gradient-reversal "
        "hospital adversary {cite:ganin2016} overfitted within a few epochs and scored 0.606, with a detector AUC "
        "of 0.815. With about 660 training patients per fold, the networks learned hospital and noise faster than "
        "histology.")
    b.table(["Approach", "Site-balanced AUC", "LOSO mean", "Hospital detector", "Finding"], [
        ["Attention-MIL on FMCIB crop bags", "0.669", "0.634", "n/a", "did not beat mean pooling"],
        ["FMCIB, frozen (3-D ResNet-50)", "0.675", "0.660", "0.263", "weaker than handcrafted features"],
        ["CT-FM, frozen", "0.662", "0.652", "0.765", "kept hospital identity"],
        ["Fine-tuned CNN", "0.615", "0.644", "0.814", "overfitted"],
        ["Fine-tuned CNN + adversary", "0.606", "0.651", "0.815", "adversary did not remove hospital"],
        ["Final model (shells + position)", "0.722", "0.714", "0.54-0.61", "best and simplest"],
    ], "Deep-learning alternatives compared with the final model", "deep",
        widths=[1.85, 0.95, 0.75, 0.9, 1.75], align="lcccl", size=9)
    b.sub("Project journey")
    b.p("{fig:journey} shows the main AUC at each stage of the project. The points use "
        "different data and metrics, so the plot tells the story rather than giving a strict comparison. The "
        "largest improvements after the progress presentation came from more hospitals, a fair metric and better "
        "masks, not from bigger models.")
    b.fig("fig_journey.png", "AUC at each stage of the project. The red line marks the progress presentation.",
          "journey", width=6.0)
    b.sub("Locked external tests (E8)")
    b.p("{fig:external} and {tab:external} show the two pre-registered "
        "results. On Lung3 the model reached 0.686 (95% CI 0.557-0.807), close to the expected LOSO level "
        "of 0.714. On TCGA it reached 0.585 (0.462-0.707). Slice thickness was important in both tests: on "
        "thin slices (at most 2.5 mm) the AUC was 0.90 on Lung3 and 0.73 on TCGA, while on thick slices it was "
        "0.68 and 0.51. These subgroups are small (18 and 29 thin-slice patients), so they are hints rather than "
        "proof. TCGA has a further problem: each of its 9 contributing hospitals sent only ADC or only SCC "
        "patients, so in TCGA the hospital is the label. A model designed to ignore hospital style cannot use "
        "this, and the effective sample size is closer to 9 hospitals than to 84 patients. TCGA tumours were also "
        "smaller (median 16 versus 29 ml) and were scanned with more reconstruction kernels (19 versus 10).")
    b.fig("external_ci.png", "Locked external tests: AUC on all patients with 95% bootstrap CI, and on thin "
          "(at most 2.5 mm) and thick slices. The red dotted line is the LOSO level (0.71).", "external", width=5.2)
    b.table(["Test set", "n (ADC/SCC)", "AUC [95% CI]", "Se", "Sp", "BA", "Thin slices", "Thick slices"], [
        ["Lung3", "79 (44/35)", "0.686 [0.557, 0.807]", "0.636", "0.686", "0.661", "0.903 (n=18)", "0.676 (n=61)"],
        ["TCGA", "84 (48/36)", "0.585 [0.462, 0.707]", "0.562", "0.583", "0.573", "0.726 (n=29)", "0.511 (n=55)"],
    ], "Locked external test results of the pre-registered model. The thin/thick split was pre-declared for "
       "Lung3 and post hoc for TCGA.", "external", widths=[0.65, 0.8, 1.3, 0.5, 0.5, 0.5, 0.95, 0.95],
        align="lcccccccc"[:8], size=8.5)
    b.sub("Comparison with existing work")
    b.p("{tab:compare} places our results next to published studies. "
        "Scores above 0.8 mostly come from one hospital or from random splits of pooled data. On truly "
        "independent hospitals the published range is about 0.54-0.73. On the same external dataset (Lung3) our "
        "locked result of 0.686 is higher than the 0.60 of Chaunzwa *et al., while Song et al.* report 0.80 on "
        "Lung3 and 0.82 on TCGA. Their numbers were obtained with expert tumour masks, by choosing the best of 130 "
        "models on the test sets, and with hospitals mixed randomly in training, so the two results were produced "
        "under different rules.")
    b.table(["Study", "Data", "How it was tested", "AUC", "Remarks"], [
        ["Zhu 2018 {cite:zhu2018}", "129 pts, 1 hospital", "split inside one hospital", "0.91", "small, no external test"],
        ["Pasini 2023 {cite:pasini2023}", "466 pts incl. LUNG1 + Radiogenomics", "random 80/20 split", "Acc 0.77 to 0.59", "drop after ComBat"],
        ["Yang 2021 {cite:yang2021}", "645 pts, 3 centres", "train one centre, test others", "0.54-0.64", "0.78 with random mixing"],
        ["Chaunzwa 2021 {cite:chaunzwa2021}", "311 pts, 1 hospital", "external Lung3 (49 pts)", "0.71 to 0.60", "deep CNN"],
        ["Chen 2023 {cite:chen2023}", "402 pts, TCIA", "external test (78 pts)", "0.84 to 0.73", "multi-task CNN"],
        ["Song 2023 {cite:song2023}", "868 pts, 8 TCIA sets", "internal + TCGA (97) + Lung3 (71)", "0.82 / 0.82 / 0.80", "best of 130 models, expert masks"],
        ["This work", "1041 pts, 4 hospitals", "site-balanced CV, LOSO, 2 locked tests", "0.72 / 0.71 / 0.69 / 0.59",
         "fair metric, automatic masks, pre-registered"],
    ], "Comparison with existing ADC/SCC studies (internal to external where two numbers are given)", "compare",
        widths=[1.25, 1.25, 1.35, 1.0, 1.35], size=8.5)

    b.h3("Interpretation and Conclusions on Data")
    b.bullets([
        "Pooled AUC can be misleading. In multi-hospital ADC/SCC data, the hospital alone explains a pooled AUC "
        "of about 0.71. Any multi-centre result should be checked with a site-balanced metric or per-hospital AUCs.",
        "There is real CT signal. Under the fair metric the model reaches 0.722, keeps 0.714 on unseen "
        "hospitals and 0.686 on the locked Lung3 test. The signal comes mainly from tumour position, size and the "
        "texture of the tumour border, which agrees with known clinical tendencies.",
        "Harmonisation makes results trustworthy rather than higher. ComBat removed most of the scanner "
        "fingerprint while leaving the AUC almost unchanged.",
        "Data diversity mattered more than model complexity. Adding hospitals and improving masks helped; "
        "attention, CNNs, foundation models and adversarial training did not.",
        "Slice thickness is the main limit. Texture learned mostly on thin CT does not transfer to 5 mm CT, "
        "which explains most of the external drop. The model should be used only on thin-slice CT.",
    ])

    b.h2("Limitations of Tools (PO(e))")
    b.p("{tab:limits} lists the main technical limitations of the tools we used and their effect on the results.")
    b.table(["Tool", "Limitation", "Effect / mitigation"], [
        ["PyRadiomics", "texture depends on voxel size, bin width and slice thickness", "fixed 2 mm grid and 20 HU bins; still unreliable on 5 mm CT"],
        ["Resampling (SimpleITK)", "cannot recover detail lost between thick slices", "5 mm scans become blurred along z; reported as main failure mode"],
        ["TotalSegmentator", "picks the largest nodule; can miss or choose the wrong lesion", "nodule found in 55/60 checks, correct lesion in about 80%; about 1 in 5 external masks may be wrong"],
        ["MedSAM2", "depends on the box; Dice 0.74 with loose boxes; hole filling not available on Windows", "box enlarged by only 3 mm; validated on 343 expert outlines"],
        ["ComBat", "assumes a location-scale shift; may remove real signal when class mixes differ; needs enough patients per site", "fitted per fold; new sites mapped label-free; balanced variants tested"],
        ["Classifiers (LR, SVM, LightGBM)", "small data, noisy labels", "strong regularisation, ensembles, nested tuning"],
        ["Bootstrap CI", "wide intervals at n of about 80", "external CIs are about plus or minus 0.12; reported in full"],
        ["Hardware (4 GB GPU)", "limits network size and batch size", "frozen backbones; deep models restricted to last-block fine-tuning"],
        ["Public labels", "histology from records (NLST codes, TCGA collection)", "strict inclusion rules; mixed types excluded"],
    ], "Limitations of the tools used", "limits", widths=[1.35, 2.35, 2.5], size=9)

    b.h2("Impact Assessment (PO(f))")
    b.h3("Assessment of Societal and Cultural Issues")
    b.p("A reliable CT-based estimate of cancer type could reduce delays and repeat biopsies, which matters most "
        "where pathology services are limited. On the other hand, there is a risk of over-trust: a confident-looking "
        "score could be taken as a diagnosis. Communication to doctors must therefore stress that the tool supports "
        "and does not replace biopsy. Since our data contain no South Asian patients, local validation is needed to "
        "avoid a tool that works worse for the population it is meant to help.")
    b.h3("Assessment of Health and Safety Issues")
    b.p("The pipeline is non-invasive and uses existing scans, so it creates no direct physical risk. The indirect "
        "risk is a wrong prediction. A false ADC call could delay the correct workup of an SCC patient and vice "
        "versa. With the current accuracy the model is not safe for clinical decisions. Before any clinical use it "
        "would need prospective testing, a radiologist check of the detected nodule, and rules for when the model "
        "must not be used (for example 5 mm CT).")
    b.h3("Assessment of Legal Issues")
    b.p("All datasets are public and de-identified, and were used under their licences: TCIA collections under "
        "CC BY 3.0/4.0 or CC BY-NC 3.0 (non-commercial, for example Lung3) and NLST data through the NCI Imaging "
        "Data Commons. The software tools are open source (Apache 2.0, BSD or MIT). A clinical version would be "
        "software as a medical device and would need regulatory approval (for example from the Directorate General "
        "of Drug Administration in Bangladesh, the FDA in the USA or CE marking in Europe), as well as compliance with "
        "patient data protection rules when used on hospital data.")

    b.h2("Sustainability Evaluation (PO(g))")
    b.sub("Technical sustainability")
    b.p("All configurations, random seeds, splits, feature names and package versions are saved, extraction is "
        "resumable, and every experiment can be rerun from saved features. The model is small and can be retrained "
        "in minutes when a new hospital joins.")
    b.sub("Economic sustainability")
    b.p("The pipeline uses free, open-source software and runs on a mid-range laptop, so a hospital would need no "
        "special hardware (Section 8).")
    b.sub("Environmental sustainability")
    b.p("Energy use is low (about 10-15 kWh for the whole project) because we avoided training large networks and "
        "reused public data.")
    b.sub("Long-term use")
    b.p("New hospitals can be added without their labels through label-free ComBat mapping, and the locked-test "
        "protocol can be repeated each time the model is updated.")

    b.h2("Ethical Issues (PO(h))")
    b.p("We applied the following ethical principles:")
    b.bullets([
        "Privacy: only de-identified public data were used; no attempt was made to identify any patient, and "
        "data were stored locally and not redistributed.",
        "Honest evaluation: every fitted step was kept inside the training folds, the external tests were "
        "pre-registered, frozen with a SHA-256 hash and scored once, and post hoc analyses are clearly labelled.",
        "Reporting negative results: failed approaches (attention-MIL, foundation models, adversarial "
        "fine-tuning) and the weak TCGA result are reported in full rather than hidden.",
        "Fairness: the site-balanced metric gives every hospital equal weight, and per-hospital results are "
        "shown so that weak performance in one group is visible.",
        "No overclaiming: the model is presented as research decision support, not as a diagnostic device.",
        "Credit: all datasets, tools and published methods are cited.",
    ])
    b.p("One ethical challenge was the temptation to adjust the model after seeing the first external result "
        "(TCGA). We did not change the model; instead the second test (Lung3) was pre-registered with the same "
        "frozen model, and the thin/thick analysis was declared in advance for it.")

    # ------------------------------------------------------------ reflection
    b.h1("Reflection on Individual and Team work (PO(i))")
    b.p("This was an individual project: Group 13 has one member. The reflection below therefore describes how "
        "the work was organised and how feedback from the instructors was used.")
    b.h2("Individual Contribution of Each Member")
    b.table(["Member", "Contribution"], [
        ["Rahib Mahasin (2106119)", "Topic selection and literature review; data download and preprocessing; "
         "segmentation and feature extraction; design of the fair metric and validation; all model experiments; "
         "locked external tests; figures, presentation slides and this report."],
    ], "Individual contribution", "contrib", widths=[1.8, 4.4])
    b.h2("Mode of TeamWork")
    b.p("Work was planned weekly with a short list of goals and a written log of every experiment (phase reports "
        "with commands, results and warnings). Feedback from the instructors at the progress presentation in Week 10 "
        "(more data, external validation, clearer evaluation) directly shaped the second half of the project. "
        "Decisions that could bias the results, such as the final model and the external test rules, were written "
        "down before the corresponding data were analysed.")
    b.h2("Diversity Statement of Team")
    b.p("As a single-member team, diversity came from the sources used rather than from team members: data from "
        "three countries and four clinical settings (radiotherapy, surgery, diagnosis and screening), methods from "
        "medical imaging, statistics and deep learning, and published work from many groups. The site-balanced "
        "evaluation was chosen so that smaller hospitals and minority classes are represented fairly in the result.")
    b.h2("Log Book of Project Implementation")
    b.p("The project started in the mid-term break (after Week 7) and ran until the final demonstration in Week 14. "
        "{tab:log} summarises the weekly log.")
    b.logbook([
        ["Mid-break", "Topic chosen; literature review; LUNG1 data and ROI products checked; method plan locked; "
         "29,184 local patches extracted", "Literature, data preparation", "Sole member", "planning and setup"],
        ["Week 8", "Mean vs attention MIL on Gradient ROI (0.604); GTV + rim bags, 37,464 patches (0.577)",
         "Model training", "Sole member", "attention unstable"],
        ["Week 9", "mRMR/LASSO selection (0.593); ResNet-18 patch embeddings (0.615); CLAM loss; global radiomics",
         "Experiments", "Sole member", "deep features first helped attention"],
        ["Week 10", "Radiogenomics added (343 pts, 0.675); hospital detector 0.998; progress presentation",
         "Data, presentation", "Sole member", "feedback: more data, external test"],
        ["Week 11", "Shortcut measured; site-balanced AUC; ComBat; shells and position features (0.681)",
         "Method redesign", "Sole member", "first honest result"],
        ["Week 12", "Lung-PET-CT-Dx (0.718) and NLST added: 1041 pts, LOSO 0.704; TCGA downloaded and locked",
         "Data scale-up", "Sole member", "about 80 GB downloaded"],
        ["Week 13", "MedSAM2 masks (0.722); CT-FM and adversarial tests; locked TCGA (0.585) and Lung3 (0.686)",
         "Segmentation, testing", "Sole member", "pre-registered, scored once"],
        ["Week 14", "Slides, final demonstration and report", "Communication", "Sole member", "project finished"],
    ], "Log book of project implementation", "log")

    # ------------------------------------------------------------ communication
    b.h1("Communication to External Stakeholders (PO(j))")
    b.h2("Executive Summary")
    b.p("Can a CT scan tell which type of lung cancer a patient has? A BUET student project has built a "
        "computer program that reads a routine chest CT scan and estimates whether a lung tumour is "
        "adenocarcinoma or squamous cell carcinoma, the two main types, which are treated differently. The program "
        "finds and outlines the tumour by itself, measures its position and the texture around it, and removes "
        "differences caused by different scanners. Tested on 1041 patients from hospitals in three countries, and "
        "then on two new datasets kept hidden until the end, it was right in about seven out of ten comparisons. "
        "It is a research tool that may one day support, not replace, biopsy.")
    b.h2("User Manual")
    b.p("The steps below run the frozen model on a new CT scan. Commands are run from the project folder with the "
        "Python environments described in Section 4.")
    b.bullets([
        "Install. Create the three environments (main, MedSAM2, TotalSegmentator) from the saved requirement "
        "files, and download the MedSAM2 checkpoint.",
        "Prepare the scan. Put each patient's CT DICOM series in its own folder. The scan should be a "
        "pre-treatment chest CT, preferably with slices of 2.5 mm or thinner.",
        "Convert and resample. Run the build script to convert DICOM to NIfTI and resample to 2 mm.",
        "Find the tumour. Run the detection script (TotalSegmentator lung nodules). Check the detected box on "
        "the saved preview image; if the wrong nodule was chosen, replace the box by hand.",
        "Outline the tumour. Run the MedSAM2 script with the saved settings; it writes one mask per patient.",
        "Extract features. Run the feature script for radiomics (shells) and position; age and sex are read "
        "from the DICOM header.",
        "Predict. Run the prediction script. It fits the frozen model on the 1041 development patients, maps "
        "the new scans as a new site with label-free ComBat, and writes one score per patient.",
        "Read the result. A score above 0 suggests ADC and below 0 suggests SCC; values near 0 are uncertain. "
        "The result must be interpreted by a doctor together with other clinical information.",
    ], numbered=True)
    b.h2("Github Link")
    b.p("The complete source code, configuration files, documentation, this report and the presentation slides "
        "are available in the GitHub repository below. The core parts of the code are also listed in Section 3.7.")
    b.p("https://github.com/raahiibbb/ADC-SCC-CT-Radiomics", align="left", human=False)

    # ------------------------------------------------------------ management
    b.h1("Project Management and Cost Analysis (PO(k))")
    b.p("This project is software only, so its costs are computing, data transfer and time rather than "
        "hardware components. The figures below are estimates in Bangladeshi Taka (BDT).")
    b.h2("Bill of Materials")
    b.table(["Item", "Quantity / use", "Cost (BDT)"], [
        ["Laptop with RTX 3050 GPU (already owned)", "share of value used over 2 months (about 90,000 BDT over 4 years)", "3,750"],
        ["Storage for data (part of a 1 TB disk)", "about 150 GB", "900"],
        ["Internet data", "about 90 GB downloaded over 2 months", "2,500"],
        ["Electricity", "about 15 kWh at about 12 BDT/kWh", "180"],
        ["Software (Python, PyRadiomics, scikit-learn, LightGBM, MedSAM2, TotalSegmentator)", "open source", "0"],
        ["Datasets (TCIA, IDC)", "public", "0"],
        ["**Total**", "", "**about 7,330**"],
    ], "Bill of materials (estimated)", "bom", widths=[3.0, 2.2, 1.0], align="llr")
    b.h2("Calculation of Per Unit Cost of Prototype")
    b.p("The prototype is the trained pipeline. Its development cost is the total of {tab:bom}, about 7,330 BDT, "
        "plus the student's time (about 8 weeks). The compute part is small: development runs used roughly "
        "100-150 GPU-hours of laptop time at about 0.1 kW, i.e. 10-15 kWh.")
    b.h2("Calculation of Per Unit Cost of Mass-Produced Unit")
    b.p("For a deployed tool the relevant unit is one patient analysed. Measured times on the laptop were about "
        "2.3 minutes per patient for nodule detection (3.2 hours for 85 TCGA scans), about 2 seconds for MedSAM2 "
        "(38 minutes for 1041 patients), about 2 seconds for feature extraction and less than a second for "
        "prediction, so about 2.5 minutes in total. With a power draw of 0.1 kW and electricity at 12 BDT/kWh, the cost per patient is")
    b.eq(r"C_{\mathrm{patient}}=\underbrace{0.1\times\frac{2.5}{60}\times12}_{\mathrm{energy}}+\underbrace{\frac{150000}{5\times20000}}_{\mathrm{workstation}}\approx0.05+1.5\approx1.6\ \mathrm{BDT}", "cost")
    b.p("assuming a dedicated 150,000 BDT workstation used for 20,000 scans per year for five years. Even with "
        "maintenance and staff time added, the cost per patient is a very small fraction of the cost of a biopsy.")
    b.h2("Timeline of Project Implementation")
    b.p("{fig:gantt} shows the timeline as a Gantt chart. The work was done in eight weeks: the mid-term break "
        "and Weeks 8-14. The progress presentation was in Week 10 and the final demonstration in Week 14.")
    b.fig("fig_gantt.png", "Gantt chart of the project.", "gantt", width=5.6)

    # ------------------------------------------------------------ future work
    b.h1("Future Work (PO(l))")
    b.p("The project is finished, but the results point to clear next steps:")
    b.bullets([
        "Robustness to thick slices. Train with thin scans artificially blurred to 5 mm, or use features that "
        "are less sensitive to slice thickness, since 5 mm CT was the main failure mode.",
        "Larger external test. Test on a multi-hospital dataset in which every hospital has both ADC and SCC, "
        "so that per-hospital AUCs can be computed, ideally including Bangladeshi hospitals.",
        "Human check of the detected nodule. A short radiologist confirmation of the automatic box would "
        "remove most of the wrong-lesion errors (about 1 in 5).",
        "PET and clinical data. Adding PET uptake and smoking history, where available, may add information "
        "that CT texture does not contain.",
        "Better use of deep models. Foundation models could be revisited with harmonisation built into "
        "training, or with much larger multi-hospital data.",
        "Prospective evaluation. Before any clinical use, the frozen model should be tested prospectively on "
        "new patients with the same locked protocol.",
    ])

    b.h1("References")
    b.references()
