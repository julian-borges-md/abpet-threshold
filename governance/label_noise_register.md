---
document_id: ABPET-GOV-003
type: governance
status: active
created: 2026-04-13
author: Julian Borges, MD
---

# Label Noise Register

## Source
Shekari M et al. Centiloid revisited. Alzheimers Dement. 2024;20:5102-5113.

## Values
| Subject Range | 95% CI Half-Width |
|---|---|
| Amyloid-negative | +/-2.70 CL |
| Amyloid-positive | +/-7.43 CL |

## Reference MAE Values for Noise Floor Context

| Metric | Value | Source |
|---|---|---|
| Yamao 2024 best MAE | 8.54 CL | PIB only — Brain Sci 2024 |
| Label noise floor (amyloid-positive) | 7.43 CL | Shekari 2024 |
| Gap between Yamao MAE and noise floor | 1.11 CL | 8.54 − 7.43 |

**Interpretation:** Even the best published model (Yamao 2024, single-tracer PIB) exceeds
the label noise floor by only 1.11 CL. Multi-tracer models with higher MAE may be
operating entirely within the noise floor for some tracer/zone combinations.
This contextualizes what any MAE improvement in this domain can legitimately claim.

## Figure Implementation Standard
All Bland-Altman panels (Figure 2, all four tracer panels) must display +/-7.43 CL
gray reference band labeled "Label noise reference (Shekari 2024)".

## Required Manuscript Language
Results: "The apparent MAE improvement from baseline (19.77 CL) to competition
benchmark (11.79 CL) is 7.98 CL, which exceeds the +/-7.43 CL label noise floor
by 0.55 CL (Shekari et al. 2024)."

## Prohibited Framing
DO NOT write: "The winner's result is not meaningful because of label noise."
CORRECT: contextualizes the improvement; does not invalidate it.
