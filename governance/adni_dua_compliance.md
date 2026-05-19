---
document_id: ABPET-GOV-006
type: governance
status: active
created: 2026-04-13
author: Julian Borges, MD
project: RO-2026-002 ABPET Threshold
---

# ADNI DUA Compliance Register
## RO-2026-002 — Data Use Agreement Obligations and Constraints

---

## DUA Status
- DUA reviewed: 2026-04-13
- DUA version: Current (adni.loni.usc.edu/news-publications/)
- Signed by: Julian Borges, MD

---

## Prohibited Activities (Appendix A)

| Activity | Status |
|---|---|
| Input ADNI participant-level data into Claude.ai or any public-facing AI platform | PROHIBITED |
| Input ADNI data into any third-party cloud platform without explicit data containment guarantees | PROHIBITED |
| Redistribute participant-level data in any form | PROHIBITED |
| Use "opt-out" clauses as sufficient protection for sharing with third parties | PROHIBITED — explicitly insufficient per DUA |

---

## Permitted Activities

| Activity | Status |
|---|---|
| Writing code, governance docs, manuscript text in Claude with no participant data present | PERMITTED |
| Running preprocessing, training, and evaluation locally on Mac | PERMITTED |
| Sharing aggregate/summary statistics (Table 2, Table 3 values) with Claude for manuscript writing | PERMITTED — DUA Term 3: "Reports and publication of summary (not participant-level) data are allowed" |
| Training models on ADNI data locally | PERMITTED |
| Using Claude Code locally (execution stays on Mac, no data transmitted) | PERMITTED |

---

## Mandatory Publication Requirements

### Byline
After the named author, must include:
"for the Alzheimer's Disease Neuroimaging Initiative*"

### Footnote (mandatory)
*Data used in preparation of this article were obtained from the Alzheimer's Disease
Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). As such, the investigators
within the ADNI contributed to the design and implementation of ADNI and/or provided data
but did not participate in the analysis or writing of this report. A complete listing of ADNI
investigators can be found at:
http://adni.loni.usc.edu/wp-content/uploads/how_to_apply/ADNI_Acknowledgement_List.pdf

### Methods Section (mandatory)
Must include the following exact language (or variation):
"Data used in the preparation of this article were obtained from the Alzheimer's Disease
Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu). The ADNI was launched in
2003 as a public-private partnership, led by Principal Investigator Michael W. Weiner, MD.
The original goal of ADNI was to test whether serial magnetic resonance imaging (MRI),
positron emission tomography (PET), other biological markers, and clinical and
neuropsychological assessment can be combined to measure the progression of mild
cognitive impairment (MCI) and early Alzheimer's disease (AD). The current goals include
validating biomarkers for clinical trials, improving the generalizability of ADNI data by
increasing diversity in the participant cohort, and to provide data concerning the diagnosis
and progression of Alzheimer's disease to the scientific community. For up-to-date
information, see adni.loni.usc.edu."

STATUS: Already in manuscript Methods 2.1 — COMPLIANT

### Funding Acknowledgment (mandatory)
"Data collection and sharing for the Alzheimer's Disease Neuroimaging Initiative (ADNI)
is funded by the National Institute on Aging (National Institutes of Health Grant U19AG024904).
The grantee organization is the Northern California Institute for Research and Education.
In the past, ADNI has also received funding from the National Institute of Biomedical Imaging
and Bioengineering, the Canadian Institutes of Health Research, and private sector
contributions through the Foundation for the National Institutes of Health (FNIH) including
generous contributions from the following: AbbVie, Alzheimer's Association; Alzheimer's
Drug Discovery Foundation; Araclon Biotech; BioClinica, Inc.; Biogen; Bristol-Myers Squibb
Company; CereSpir, Inc.; Cogstate; Eisai Inc.; Elan Pharmaceuticals, Inc.; Eli Lilly and
Company; EuroImmun; F. Hoffmann-La Roche Ltd and its affiliated company Genentech, Inc.;
Fujirebio; GE Healthcare; IXICO Ltd.; Janssen Alzheimer Immunotherapy Research &
Development, LLC.; Johnson & Johnson Pharmaceutical Research & Development LLC.;
Lumosity; Lundbeck; Merck & Co., Inc.; Meso Scale Diagnostics, LLC.; NeuroRx Research;
Neurotrack Technologies; Novartis Pharmaceuticals Corporation; Pfizer Inc.; Piramal Imaging;
Servier; Takeda Pharmaceutical Company; and Transition Therapeutics."

STATUS: Needs update in manuscript Acknowledgments section — PENDING

### Data Version
Must record ADNI data download date in Methods section.
STATUS: PENDING — add after data download

---

## DPC Pre-submission Review (MANDATORY)

Before submitting to any journal:
1. Submit manuscript to ADNI Data and Publications Committee (DPC)
2. Contact: EDRAKE@BWH.HARVARD.EDU
3. Process: Log into LONI-IDA → My Account → Update → Publication Update tab
4. Timeline: DPC review completed within 2 weeks
5. Purpose: Verify DUA compliance items are correctly implemented (not scientific review)

STATUS: PENDING — required before journal submission

---

## Model Weights Release Decision

Per DUA Appendix A: publicly releasing model weights trained on ADNI data requires
consideration of whether weights could reconstruct participant-level data.
Decision required before release: consult with Dr. Weiner (michael.weiner@ucsf.edu) if
weights are to be made publicly available.

Current plan: weights available on request (not publicly posted) — COMPLIANT

---

## Workflow Architecture (DUA-Compliant)

Claude.ai session: Code, governance, manuscript text, aggregate results only.
                   NO participant-level data ever enters this conversation.

Local Mac execution: All preprocessing, training, evaluation.
                     Data never leaves the machine.

Results transfer: Only aggregate statistics (MAE, TSA, zone counts)
                  pass from Mac to this session for manuscript writing.
                  This is explicitly permitted under DUA Term 3.

---
## Regulatory Sources Verified 2026-04-13

| Claim | Correct Source | Note |
|---|---|---|
| PMC12640226 | Arai H et al. Cureus 2025;17(11):e97398 | Case report: Centiloid=0 in visually positive patient. Stronger than withdrawal. |
| EMA Amyvid withdrawal | EMA EPAR May 2025 | EMA declined treatment monitoring extension. Use this for regulatory caution framing. |
| FDA Amyvid label | FDA / Lilly press release June 25 2025 | FDA EXPANDED label for therapy selection. Opposite of EMA. Do not conflate. |
