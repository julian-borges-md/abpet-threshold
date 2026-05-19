# CLAIM Checklist — Checklist for Artificial Intelligence in Medical Imaging
## RO-2026-002 ABPET Threshold
## Reference: Mongan J et al. Radiology AI. 2020;2:e200029

| # | Item | Status | Location in Manuscript |
|---|---|---|---|
| **Title and Abstract** | | | |
| 1 | Identification as a study of AI methodology | Complete | Title |
| 2 | Structured abstract with background, objective, methods, results, conclusions | Partial — results pending | Abstract |
| **Introduction** | | | |
| 3 | Scientific and clinical background | Complete | Intro P1-P2 |
| 4 | Study objectives and hypotheses | Complete | Intro P5 |
| **Methods** | | | |
| 5 | Prospective or retrospective design statement | Complete | Methods 2.1 |
| 6 | Data sources | Complete | Methods 2.1 |
| 7 | Eligibility criteria | Complete | Methods 2.1 |
| 8 | Data preprocessing steps | Complete | Methods 2.2 |
| 9 | Selection of data subsets | Complete | Methods 2.1 (80/20 stratified split) |
| 10 | Definitions of data elements | Complete | Methods 2.1 (zone labels, thresholds) |
| 11 | De-identification methods | N/A — public dataset with existing IRB |
| 12 | Ground truth labeling (reference standard) | Complete | Methods 2.1 (ADNI Centiloid pipeline) |
| 13 | Rationale for ground truth labeling | Complete | Methods 2.1 |
| 14 | Source of ground truth labels | Complete | ADNI data portal |
| 15 | Training, validation, test split | Complete | Methods 2.1 |
| 16 | Model details | Complete | Methods 2.3-2.6 |
| 17 | Software | Pending | To add: PyTorch version, MONAI version |
| 18 | Training details | Complete | Methods 2.7 |
| 19 | Measures of model performance | Complete | Methods 2.8 |
| 20 | Statistical analysis | Complete | Methods 2.9 |
| **Results** | | | |
| 21 | Data flow diagram | Pending | Figure or supplement |
| 22 | Demographic and clinical characteristics | Pending | Table 1 |
| 23 | Model performance metrics | Pending | Tables 2-3 |
| 24 | Estimates of diagnostic performance | Pending | Table 3 (TSA) |
| 25 | Failure analysis | Pending | Figure 1 (misclassification) |
| **Discussion** | | | |
| 26 | Study limitations | Complete | Discussion 4.8 |
| 27 | Implications for practice | Complete | Discussion 4.2-4.3 |
| **Other** | | | |
| 28 | Registration | N/A — not a clinical trial |
| 29 | Ethical approval | Pending — ADNI DUA reference |
| 30 | Conflicts of interest | Complete | Conflicts section |
| 31 | Funding | Complete | Acknowledgments |

## Items Requiring Action Before Submission
- Add PyTorch and MONAI version numbers to Methods 2.2
- Add data flow diagram (PRISMA-style) showing ADNI inclusion/exclusion
- ADNI DUA reference number (after access granted)
- Complete Table 1, Tables 2-3, Figures 1-4 (after model training)
- Verify PMC12640226 before including Amyvid withdrawal claim
