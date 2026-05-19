---
document_id: ABPET-GOV-001
type: governance
status: active
created: 2026-04-13
author: Julian Borges, MD
project: RO-2026-002 ABPET Threshold
gate: G0 — required before any manuscript section is drafted
---

# Novelty Statement

## Core Claim

This paper introduces threshold-specific accuracy (TSA) as the first clinically grounded
evaluation standard for deep learning Centiloid quantification models. It demonstrates,
for the first time, that global MAE conceals disproportionate failure in the 10 to 30 CL
intermediate zone where lecanemab and donanemab treatment eligibility is adjudicated,
and quantifies the treatment eligibility misclassification rate that global MAE renders
invisible. The paper fills a gap that exists in every published deep learning Centiloid
model: none report zone-stratified MAE, none report TSA, none compute misclassification
rates at the treatment threshold.

---

## Scientific Origin of This Work

This research originated from the observation that the deep learning Centiloid field
evaluates models exclusively by global MAE while clinical deployment requires accuracy
at a specific threshold. That observation is original. The dataset used (ADNI), the
models designed (BaselineCNN and ResNet18-3D variants), the metrics introduced (TSA,
zone-stratified MAE), and the clinical framing (lecanemab/donanemab eligibility
misclassification) are all products of independent research design. No competition
result, competition code, or competition dataset is used, cited, or referenced in
this paper.

---

## Differentiation from All Prior Work

### 1. Yamao et al. Brain Sci 2024 — ResNet50, PIB only, MAE 8.54 CL
Single tracer only. No zone-stratified analysis. No clinical framing. No TSA.
No misclassification analysis. No label noise analysis. Evaluates technical
performance without reference to treatment decisions. Different question entirely.

### 2. DeepSUVR — Alzheimers Dement 2026 — longitudinal trajectory correction
Addresses longitudinal SUVR consistency across serial scans. Does not address
cross-sectional eligibility classification at a treatment threshold. No zone
analysis. No TSA. Complementary, not redundant.

### 3. All Other Published DL Centiloid Models
A systematic review of the literature through April 2026 identifies no published
deep learning Centiloid paper that:
- Reports zone-stratified MAE for any clinical zone
- Reports TSA at 24 CL or 10 CL
- Computes treatment eligibility misclassification rates
- Discusses ARIA exposure risk of false positive predictions
- References the ±7.43 CL pipeline noise floor as a reference ceiling
- Compares model rankings by TSA versus global MAE

All five absences define the contribution space as genuinely unoccupied.

---

## Why a CS Team Cannot Write This Paper

The primary clinical contributions — treatment eligibility misclassification
quantification, ARIA risk framing, donanemab treat-to-clear implications — require:

- Clinical knowledge of CLARITY-AD and TRAILBLAZER-ALZ 2 trial eligibility criteria
- Understanding of ARIA incidence, grading, and management in anti-amyloid therapy
- Knowledge of the EMA withdrawal of Amyvid treatment monitoring application
- Ability to frame model error statistics as clinical harm risk

These are not derivable from the machine learning literature. They are the physician layer
of this paper, and they constitute the primary novelty of Contributions 2 and 3.

---

## What Makes This an Advancement, Not a Replication

The paper does not improve a model. It changes the question the field asks about models.
The TSA metric is not technically complex — it is a binary classification accuracy at a
threshold. Its novelty lies entirely in the insight that this is the correct question for
clinical deployment, and that no one has asked it before for this problem.

This is the same class of contribution as:
- Proposing sensitivity and specificity instead of AUC for diagnostic tests
- Proposing calibration metrics instead of discrimination for clinical risk scores
- Proposing NNT instead of relative risk reduction for therapy evaluation

The metric itself is simple. The recognition that the field has been measuring the wrong
thing is the contribution.

*G0 gate status: SATISFIED — 2026-04-13*
