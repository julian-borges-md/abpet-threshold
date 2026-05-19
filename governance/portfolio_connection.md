---
document_id: ABPET-GOV-004
type: governance
status: active
created: 2026-04-13
intended_use: PhD applications, EB1 petition, cover letters, grant aims
---

# Portfolio Connection Document

## Thesis
AI systems in clinical medicine are routinely evaluated against metrics that do not
correspond to the decisions they will be used to make. This mismatch creates systematic
and quantifiable risk of harm at the point of clinical deployment.

## Evidence Map
| Project | Domain | Mismatch Demonstrated |
|---|---|---|
| ATI proteomics RO-2026-001 | Biomarker prediction | AUROC 0.939 + BSS -0.029 |
| ABPET threshold RO-2026-002 | Neuroimaging | Global MAE conceals Zone B failure; TSA exposes it |
| IronFF systematic review | Evidence synthesis | Label heterogeneity concealed by pooled estimates |
| DrugSynth AI / MitoCorex | Drug discovery pipeline | Governance architecture preventing mismatch at design |

## EB1 Framing
Original contribution of major significance: identifies structural limitation in the
evaluation framework for AI tools entering clinical practice for FDA-approved drugs
(lecanemab, donanemab) at the exact moment of commercial deployment, when Eli Lilly has
already experienced a regulatory withdrawal (PMC12640226) due to the class of errors this
paper characterizes. Proposes a corrective reporting standard with immediate field impact.

## PhD Statement Paragraph
My second paper examines whether global MAE captures the performance that matters
clinically for DL amyloid PET quantification. Using ADNI data and a six-model systematic
ablation, I demonstrate that global MAE conceals disproportionate failure in the 10-30 CL
zone where lecanemab and donanemab eligibility is adjudicated. I introduce threshold-specific
accuracy as the clinically appropriate evaluation standard, quantify the treatment eligibility
misclassification rate, and show that the full range of improvement from baseline to
competition winner (7.98 CL) exceeds the standard Centiloid pipeline noise floor by only
0.55 CL — reframing what MAE improvements can legitimately claim in this domain.
