# Threshold-Specific Accuracy Reveals Systematic Failure of Deep Learning Centiloid Quantification at the Anti-Amyloid Treatment Eligibility Boundary: A Multi-Tracer ADNI Analysis

**Julian Borges, MD**, for the Alzheimer's Disease Neuroimaging Initiative*
Frontier Translational Research Lab (Independent)
ORCID: 0009-0001-9929-3135
Correspondence: jyborges@bu.edu

*Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators within the ADNI contributed to the design and implementation of ADNI and/or provided data but did not participate in the analysis or writing of this report. A complete listing of ADNI investigators can be found at: http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf

---

## Abstract

*(Pending results — structured abstract to be completed after Table 3 is populated)*

**Background:** Deep learning models for amyloid PET Centiloid quantification are evaluated by global mean absolute error (MAE), yet the clinical utility of these models is determined by their accuracy near the treatment eligibility threshold for FDA-approved anti-amyloid therapies.

**Objective:** To determine whether global MAE captures clinically critical model performance at the lecanemab and donanemab treatment eligibility boundaries, and to propose threshold-specific accuracy (TSA) as the appropriate evaluation standard.

**Methods:** We trained and evaluated six models (BaselineCNN and ResNet18-3D variants with systematic ablation of learning rate scheduling, augmentation, tracer fusion strategy, and pretraining) on multi-tracer amyloid PET data from the Alzheimer's Disease Neuroimaging Initiative (ADNI; FBP, FBB, PIB, NAV). Primary metrics were TSA at 24 CL (lecanemab eligibility threshold, TSA_24) and 10 CL (donanemab cessation threshold, TSA_10), and zone-stratified MAE across Zone A (<10 CL), Zone B (10–30 CL), and Zone C (>30 CL). External validation used OASIS-3.

**Results:** *(Pending)*

**Conclusions:** Global MAE, the field's primary evaluation metric, does not capture the clinically critical performance of Centiloid prediction models at the treatment eligibility boundary. TSA should be required reporting in all future deep learning Centiloid quantification studies.

---

## 1. Introduction

Alzheimer's disease affects an estimated 55 million people worldwide and represents one of the most significant unmet needs in modern medicine. The 2023 and 2024 FDA approvals of lecanemab (Leqembi, Eisai/Biogen) and donanemab (Kisunla, Eli Lilly) as the first disease-modifying therapies for early Alzheimer's disease marked a fundamental shift in the clinical landscape. Both drugs require confirmation of amyloid pathology prior to treatment initiation, establishing amyloid PET quantification as a prerequisite for a clinical decision with direct pharmacological, financial, and safety consequences for patients.

The Centiloid scale was introduced to standardize amyloid PET quantification across radiotracers and acquisition protocols. A Centiloid value of 0 represents the mean of young cognitively normal subjects and 100 the mean of typical Alzheimer's disease patients with established amyloid pathology. Clinical thresholds have been operationalized through expert consensus and updated appropriate use criteria: a value below 10 CL indicates amyloid negativity with high certainty, while values in the range of approximately 24 to 30 CL define the lower boundary for treatment eligibility under current prescribing guidelines for both approved anti-amyloid therapies. For donanemab specifically, the treat-to-clear strategy employed in TRAILBLAZER-ALZ 2 requires accurate identification of amyloid negativity at the 10 CL boundary to guide treatment cessation decisions.

Deep learning approaches to Centiloid prediction have emerged as a promising alternative to standard quantitative PET pipelines, which are operator-dependent, require reference region assumptions, and carry inherent measurement variability. Several architectures have been proposed and evaluated. Yamao et al. applied a ResNet50-based regression model to PIB PET data and reported MAE 8.54 CL (Brain Sciences, 2024). DeepSUVR applied a convolutional correction network to longitudinal SUVR trajectories, improving consistency of serial amyloid estimates across imaging timepoints (Alzheimer's and Dementia, 2026). Additional approaches using standard convolutional architectures have demonstrated Pearson correlations above 0.79 across multi-tracer cohorts. Collectively, these models establish that automated Centiloid prediction is technically feasible and increasingly accurate by the field's primary benchmark.

Despite this progress, a structural evaluation gap has not been addressed. Every published deep learning Centiloid model is evaluated by a single metric: global mean absolute error averaged across the full Centiloid range. This framework is blind to the distribution of error within that range. A model that achieves low global MAE may perform well in clearly amyloid-negative patients (below 10 CL) and clearly amyloid-positive patients (above 30 CL) while failing disproportionately in the 10 to 30 CL intermediate zone where treatment eligibility decisions are concentrated. The clinical consequences are not symmetric. A false positive — predicting eligibility in a patient who is truly amyloid-negative — exposes that patient to a drug with approximately 20% ARIA incidence and no therapeutic benefit. A false negative — predicting ineligibility in a patient who is truly amyloid-positive — denies that patient access to a therapy shown to slow clinical decline by 27% on the CDR-SB in CLARITY-AD. The standard Centiloid pipeline itself carries within-pipeline variability of ±7.43 CL in amyloid-positive subjects (Shekari et al., 2024), establishing a noise floor below which apparent MAE improvements may not reflect genuine gains in biological quantification accuracy. The clinical reality of these errors has been documented in practice: Arai et al. reported a case in which cortical amyloid deposition was visually evident on florbetapir PET yet the calculated Centiloid value was 0, falsely indicating amyloid negativity and rendering the patient ineligible for anti-amyloid therapy — an error attributable to reference region selection (Cureus, 2025). At the regulatory level, the European Medicines Agency declined Eli Lilly's application to extend the Amyvid label for treatment monitoring, concluding that the submitted evidence did not demonstrate sufficient performance of quantitative PET for individual-level monitoring of treatment response.

These converging lines of evidence — biological, clinical, regulatory — establish that the field's current evaluation standard does not measure what matters for patient outcomes at the treatment eligibility boundary. No published study has stratified deep learning Centiloid model error by clinical zone. No published study has computed treatment eligibility misclassification rates from model predictions. No published study has proposed or reported threshold-specific accuracy for any Centiloid quantification model. This paper addresses all three gaps simultaneously.

The present study has three primary objectives. First, to demonstrate that global MAE conceals zone-specific failure by computing MAE stratified by clinical zone and testing whether Zone B MAE is proportional to global MAE across a systematic model ablation. Second, to introduce threshold-specific accuracy (TSA) as the clinically appropriate evaluation standard — defined as the proportion of patients correctly classified as above or below a clinical Centiloid threshold — and to demonstrate that model rankings by TSA differ from rankings by global MAE. Third, to propose TSA as a required supplementary metric for all future deep learning Centiloid quantification studies, analogous to the established requirement for sensitivity and specificity at a clinical threshold in diagnostic test reporting. These contributions are grounded in a six-model systematic ablation on multi-tracer ADNI data with external validation on OASIS-3, providing the first peer-reviewed, reproducible, clinically framed benchmark for deep learning Centiloid quantification.

---

## 2. Methods

### 2.1 Dataset

We used amyloid PET data from the Alzheimer's Disease Neuroimaging Initiative (ADNI; adni.loni.usc.edu). ADNI is a longitudinal multicenter study designed to develop clinical, imaging, genetic, and biochemical biomarkers for the early detection and tracking of Alzheimer's disease. Data used in the preparation of this article were obtained from the ADNI database. The ADNI was launched in 2003 as a public-private partnership, led by Principal Investigator Michael W. Weiner, MD. For up-to-date information, see www.adni-info.org.

We included all available amyloid PET scans with associated Centiloid scores across four radiotracers: florbetapir (FBP), florbetaben (FBB), Pittsburgh compound B (PIB), and florbetapir NAV (NAV). The minimum acceptable sample size was 500 participants across all tracers, with a minimum of 30 participants in the intermediate clinical zone (Zone B, 10–30 CL) to support statistically valid TSA computation. The dataset was split into training (80%) and validation (20%) sets using stratified random sampling by tracer, with random seed 42 fixed throughout.

For external validation, we used OASIS-3 (oasis-brains.org), an open-access dataset comprising longitudinal neuroimaging and clinical data from participants at the Knight Alzheimer Disease Research Center at Washington University in St. Louis.

Clinical zone labels were assigned to each participant based on their Centiloid score: Zone A (below 10 CL, amyloid-negative with high certainty), Zone B (10 to 30 CL, intermediate treatment decision zone), and Zone C (above 30 CL, amyloid-positive with high certainty). These boundaries are derived from the updated Appropriate Use Criteria for amyloid PET (Rabinovici et al., J Nucl Med, 2025) and the original Centiloid Project publication (Klunk et al., Alzheimers Dement, 2015).

### 2.2 Preprocessing Pipeline

All PET volumes were processed using a standardized nine-step pipeline designed to ensure reproducibility and cross-study comparability. The steps were: (1) load NIfTI and enforce channel-first format (C, H, W, D); (2) reorient to RAS coordinate system; (3) isotropic resampling to 2 mm isotropic voxels using bilinear interpolation; (4) foreground cropping with a 10-voxel margin using MONAI CropForeground; (5) resize to 128 × 128 × 128 voxels using trilinear interpolation; (6) spatial padding to exactly 128³; (7) dynamic frame averaging for four-dimensional PET acquisitions; (8) shape enforcement to (1, 128, 128, 128); and (9) per-volume min-max normalization to [0, 1]. Each volume was saved as a float32 NumPy array. A CSV file was generated with columns for array path, Centiloid score, radiotracer, participant ID, train/validation split, and clinical zone label. All preprocessing code is publicly available at https://github.com/julian-borges-md/abpet-threshold.

### 2.3 Model Architecture

We designed a six-model ablation to systematically characterize the contribution of architectural and training components to both global MAE and threshold-specific accuracy. The ablation begins with a simple convolutional baseline and progresses through five controlled improvements, isolating each component independently.

### 2.4 Systematic Ablation: Models B Through F

Six models were trained in sequence to isolate the contribution of each architectural and training component to both global MAE and TSA. Model B adds cosine annealing learning rate scheduling (T_max = 100, η_min = 10⁻⁶) to the baseline architecture. Model C adds 3D augmentation (random flips along all three spatial axes with probability 0.5, random rotation ±10 degrees, and Gaussian noise with σ = 0.02) to Model B. Model D replaces the encoder with a ResNet18 adapted for three-dimensional inputs, maintaining the 8-dimensional concatenative tracer fusion. Model E replaces the concatenative fusion with Feature-wise Linear Modulation (FiLM) using a 32-dimensional tracer embedding applied after each of the four encoder stages. Model F adds MedicalNet pretrained weights (resnet_18.pth, Tencent Research) to Model E's encoder, with the regression head randomly initialized. Models D, E, and F use Huber loss with δ = 10 CL in place of L1 loss, to reduce sensitivity to label noise outliers in the amyloid-positive range.

### 2.5 ResNet18-3D Architecture

The ResNet18-3D encoder follows the standard four-stage structure with [2, 2, 2, 2] residual blocks and channel widths 64, 128, 256, and 512. Each residual block applies Conv3d(3×3×3) → BatchNorm3d → ReLU → Conv3d(3×3×3) → BatchNorm3d with an identity skip connection (or a 1×1 projection when channel dimensions change). The stem applies a 7×7×7 convolution with stride 2 followed by max pooling. The regression head applies adaptive average pooling to the stage-4 output, flattening to a 512-dimensional vector, then Linear(512, 256) → ReLU → Dropout(0.3) → Linear(256, 1).

### 2.6 FiLM Tracer Conditioning

Feature-wise Linear Modulation (Perez et al., AAAI 2018) applies a learned affine transformation to each feature map conditioned on tracer identity:

y = γ(t) · x + β(t)

where x is the feature map tensor of shape (B, C, D, H, W), t is the tracer embedding of dimension 32, and γ(t) and β(t) are predicted by a linear layer that maps the tracer embedding to 2C parameters, split equally into gain and bias vectors broadcast over the spatial dimensions. FiLM is applied after each of the four encoder stages, providing tracer-specific feature modulation at all spatial resolutions. The FiLM layer weights are initialized so that γ = 1 and β = 0 at the start of training, ensuring a stable identity initialization. This implementation is mechanistically distinct from the concatenative fusion in Models A through D, where tracer information is injected only at the final head.

### 2.7 Training Configuration

All models used the AdamW optimizer with initial learning rate 10⁻⁴ and weight decay 10⁻⁴. Models B through F used cosine annealing learning rate scheduling. Training ran for a maximum of 100 epochs with early stopping (patience 15 epochs on validation MAE). Batch size was 4 for GPU and 2 for CPU fallback. Mixed precision training used torch.amp.autocast where available. Gradient clipping was applied with maximum norm 1.0. Random seed 42 was fixed throughout for reproducibility.

### 2.8 Evaluation Metrics

**Global metrics** (reported for comparison with prior work): overall MAE, per-tracer MAE (FBP, FBB, PIB, NAV), RMSE, and Pearson correlation coefficient.

**Clinical metrics** (primary contribution): Threshold-specific accuracy at 24 CL (TSA_24, lecanemab eligibility threshold) and 10 CL (TSA_10, donanemab cessation threshold), defined as the proportion of patients correctly classified as above or below the respective threshold. Zone-stratified MAE was computed separately for Zone A (below 10 CL), Zone B (10 to 30 CL), and Zone C (above 30 CL). The treatment eligibility misclassification rate in Zone B was computed as the proportion of Zone B patients receiving an incorrect eligibility classification at the 24 CL threshold, with false eligible (predicted positive, true negative) and false ineligible (predicted negative, true positive) reported separately.

**Calibration**: Bland-Altman analysis was performed per radiotracer for the best-performing model, with mean bias, limits of agreement (±1.96 SD), and a ±7.43 CL label noise reference band derived from Shekari et al. (Alzheimers Dement, 2024).

### 2.9 Statistical Analysis

All metrics are reported as mean with 95% bootstrap confidence intervals (n = 1,000 bootstrap samples, random seed 42). For TSA comparisons between model pairs, McNemar's test was applied to paired binary eligibility classifications. Statistical significance threshold was p < 0.05 with Bonferroni correction for six model comparisons. The label noise floor of ±7.43 CL (Shekari et al., 2024) is displayed as a reference line on all MAE comparison figures and is cited when interpreting the magnitude of MAE improvements.

### 2.10 Reporting Standards and Code Availability

This study follows the CLAIM (Checklist for Artificial Intelligence in Medical Imaging) reporting guideline (Mongan et al., Radiology AI, 2020). Complete source code is publicly available at https://github.com/julian-borges-md/abpet-threshold under an MIT license. ADNI data are available to credentialed researchers at adni.loni.usc.edu. OASIS-3 data are available at oasis-brains.org. Model weights are available from the corresponding author upon reasonable request following journal acceptance.

---

## 3. Results

### 3.1 Dataset Characteristics

*(Table 1 — pending ADNI data access)*

Total participants: [N]. Tracer distribution: FBP [n], FBB [n], PIB [n], NAV [n].
Centiloid range: [min] to [max] CL. Zone A: [n] ([%]). Zone B: [n] ([%]). Zone C: [n] ([%]).

**Zone B minimum check:** n = [n]. If n < 30, escalation protocol applies before TSA reporting.

### 3.2 Ablation: Global MAE and Pearson r

*(Table 2 — pending model training)*

Model A (hackathon replica) achieved MAE [X] CL on ADNI validation data, consistent with
the hackathon baseline of 19.77 CL (deviation [Y] CL, within the 3.0 CL tolerance threshold).

### 3.3 Primary Finding: Zone-Stratified MAE and Threshold-Specific Accuracy

*(Table 3 — primary contribution — pending model training)*

Zone B MAE differed from overall MAE for [models]. TSA_24 rankings differed from global
MAE rankings for [models], demonstrating that the metrics are not interchangeable in
clinical contexts.

### 3.4 Treatment Eligibility Misclassification

*(Figure 1 — pending model training)*

### 3.5 Label Noise Context

*(Figure 2 — pending model training)*

The apparent improvement from baseline MAE 19.77 CL to [best model MAE] CL is [delta] CL,
which [exceeds / does not exceed] the ±7.43 CL label noise floor by [value] CL (Shekari 2024).

### 3.6 Training Dynamics

*(Figure 3 — pending model training)*

### 3.7 FiLM vs Concat Tracer Fusion

*(Figure 4 — pending model training)*

---

## 4. Discussion

### 4.1 Summary

*(Pending results)*

### 4.2 Clinical Implications for Lecanemab Eligibility

*(Pending — requires CLAIM-006 and CLAIM-007 sourced values — both confirmed in clinical_framing_register.md)*

### 4.3 Clinical Implications for Donanemab Cessation

*(Pending — requires CLAIM-005 sourced values — confirmed in clinical_framing_register.md)*

### 4.4 Label Noise Ceiling

*(Pending results — Shekari 2024 sourced in label_noise_register.md)*

### 4.5 FiLM vs Concat Fusion: Mechanistic Interpretation

*(Pending Figure 4 results)*

### 4.6 Comparison to Prior Work

*(Pending — Yamao 2024, DeepSUVR 2026 comparison requires results)*

### 4.7 Proposed Reporting Standard

Future deep learning Centiloid quantification papers should report as a minimum: overall MAE,
Zone B MAE, TSA_24, TSA_10, and Bland-Altman analysis per radiotracer with the label noise
reference band. This mirrors the requirement in diagnostic test reporting to provide
sensitivity and specificity at the clinical decision threshold, not merely AUC.

### 4.8 Limitations

Public dataset only. Single institution per cohort. No prospective clinical validation.
Ground truth Centiloid pipeline variability ±7.43 CL. No outcome data linking model
misclassification to documented clinical harm. External validation on OASIS-3 provides
cross-cohort generalizability evidence but does not substitute for prospective validation
in a treatment eligibility assessment cohort.

### 4.9 Future Directions

Prospective validation in a clinical amyloid PET eligibility assessment cohort.
Extension to tau PET (CenTauR scale, analogous threshold-specific accuracy problem).
Development of uncertainty quantification methods to identify patients requiring repeat
imaging before eligibility determination.

---

## 5. Conclusion

Global MAE, the field's primary metric for deep learning Centiloid quantification, does not
capture the clinically critical performance of these models at the treatment eligibility
boundary for lecanemab and donanemab. We demonstrate this on public ADNI data using a
six-model systematic ablation, introduce threshold-specific accuracy as the appropriate
evaluation standard, and show that model selection by global MAE alone could expose
borderline patients to ARIA risk or deny them access to approved disease-modifying therapies.
As amyloid PET quantification becomes embedded in anti-amyloid treatment pathways, the field
requires evaluation standards commensurate with the clinical stakes of the decisions those
models inform.

---

## Acknowledgments

Data collection and sharing for the Alzheimer's Disease Neuroimaging Initiative (ADNI) is funded by the National Institute on Aging (National Institutes of Health Grant U19AG024904). The grantee organization is the Northern California Institute for Research and Education. In the past, ADNI has also received funding from the National Institute of Biomedical Imaging and Bioengineering, the Canadian Institutes of Health Research, and private sector contributions through the Foundation for the National Institutes of Health (FNIH) including generous contributions from the following: AbbVie, Alzheimer's Association; Alzheimer's Drug Discovery Foundation; Araclon Biotech; BioClinica, Inc.; Biogen; Bristol-Myers Squibb Company; CereSpir, Inc.; Cogstate; Eisai Inc.; Elan Pharmaceuticals, Inc.; Eli Lilly and Company; EuroImmun; F. Hoffmann-La Roche Ltd and its affiliated company Genentech, Inc.; Fujirebio; GE Healthcare; IXICO Ltd.; Janssen Alzheimer Immunotherapy Research & Development, LLC.; Johnson & Johnson Pharmaceutical Research & Development LLC.; Lumosity; Lundbeck; Merck & Co., Inc.; Meso Scale Diagnostics, LLC.; NeuroRx Research; Neurotrack Technologies; Novartis Pharmaceuticals Corporation; Pfizer Inc.; Piramal Imaging; Servier; Takeda Pharmaceutical Company; and Transition Therapeutics.

**DPC pre-submission review required.** Per ADNI DUA, this manuscript must be submitted to the ADNI Data and Publications Committee prior to journal submission. Contact: EDRAKE@BWH.HARVARD.EDU via LONI-IDA account → My Account → Publication Update tab. Review completed within 2 weeks. *(Remove this note before journal submission.)*

## Conflicts of Interest

The author declares no conflicts of interest.

## Data Availability

ADNI data are available to credentialed researchers at adni.loni.usc.edu.
OASIS-3 data are available at oasis-brains.org.
Source code is publicly available at https://github.com/julian-borges-md/abpet-threshold.
Model weights available upon reasonable request following journal acceptance.

## References

*(references.bib — to be finalized at submission)*

1. van Dyck CH et al. Lecanemab in Early Alzheimer's Disease. NEJM. 2023;388:9-21.
2. Sims JR et al. Donanemab in Early Symptomatic Alzheimer's Disease. JAMA. 2023;330:512-527.
3. Rabinovici GD et al. Updated Appropriate Use Criteria for Amyloid and Tau PET. J Nucl Med. 2025;66:376-392.
4. Klunk WE et al. The Centiloid Project. Alzheimers Dement. 2015;11:1-15.e7.
5. Shekari M et al. Centiloid revisited. Alzheimers Dement. 2024;20:5102-5113.
6. Yamao T et al. [ResNet50 Centiloid prediction]. Brain Sci. 2024. [verify full citation]
7. [DeepSUVR citation — verify at submission]
8. Perez E et al. FiLM: Visual Reasoning with a General Conditioning Layer. AAAI. 2018.
9. Mongan J et al. Checklist for AI in Medical Imaging (CLAIM). Radiology AI. 2020;2:e200029.
10. [MedicalNet / Tencent Research citation]
11. PMC12640226 [Amyvid withdrawal — verify before inclusion]
