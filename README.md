# abpet-threshold

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![ADNI DUA](https://img.shields.io/badge/Data-ADNI%20DUA%20Required-lightgrey)](https://adni.loni.usc.edu)
[![OASIS-3](https://img.shields.io/badge/Validation-OASIS--3-lightgrey)](https://www.oasis-brains.org)
[![CLAIM](https://img.shields.io/badge/Reporting-CLAIM-green)](https://doi.org/10.1148/ryai.2020200029)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--9929--3135-a6ce39)](https://orcid.org/0009-0001-9929-3135)
[![Research Order](https://img.shields.io/badge/Order-RO--2026--002-blueviolet)](governance/decision_log.md)
[![DOI](https://img.shields.io/badge/DOI-pending%20acceptance-blue)](#citation)

> **Frontier Translational Research Lab** (Independent)
> Principal Investigator: Julian Borges, MD (`jyborges@bu.edu`)
> Research Order: RO-2026-002 | 2026-04-13

---

## The Clinical Problem

Two FDA-approved drugs now exist to slow Alzheimer's disease: lecanemab (July 2023) and
donanemab (July 2024). Both require a brain PET scan to confirm amyloid pathology before
treatment begins. The number that comes out of that scan — the Centiloid score — determines
whether a patient qualifies.

Deep learning models that predict Centiloid scores are evaluated by one metric: mean absolute
error (MAE). Lower MAE is reported as better performance. The field has accepted this without
question.

This paper asks a different question: **does global MAE measure the right thing?**

The treatment decision zone spans 10 to 30 Centiloid units. Published models report MAE
values approaching this range. A model that is wrong by 8 CL on average — the best
published result to date — is wrong by nearly half the width of the zone where eligibility
is adjudicated. A false positive exposes a patient to a drug with approximately 20% ARIA
incidence and no benefit. A false negative denies a patient the only therapy shown to slow
Alzheimer's progression.

No published study has measured this. This paper does.

---

## Contributions

| # | Contribution | Field Status |
|---|---|---|
| 1 | Zone-stratified MAE: error separated for Zone A (<10 CL), Zone B (10–30 CL), Zone C (>30 CL) | Not previously reported for any DL Centiloid model |
| 2 | Treatment eligibility misclassification rate at 24 CL and 10 CL | Not previously computed |
| 3 | Threshold-specific accuracy (TSA) as a proposed required reporting standard | New standard proposed |
| 4 | Six-model systematic ablation on ADNI — public reproducible benchmark | First peer-reviewed multi-tracer ADNI ablation |
| 5 | Label noise ceiling: best published MAE (8.54 CL, Yamao 2024) exceeds pipeline noise (±7.43 CL) by only 1.11 CL | Original contextual finding |

---

## Why This Requires a Physician

Contributions 2 and 3 are not derivable from machine learning benchmarks. They require:

- Clinical knowledge of CLARITY-AD and TRAILBLAZER-ALZ 2 eligibility criteria and their
  Centiloid thresholds
- Understanding of ARIA incidence, grading, and management in anti-amyloid therapy
- Knowledge of the EMA's rejection of the Amyvid treatment monitoring application on
  grounds of insufficient performance of quantitative PET for individual-level decisions
- The clinical judgment to recognize that sensitivity and specificity at a threshold —
  not average error across a range — is the correct measure of a decision-support tool

The metric (TSA) is arithmetically simple. The insight that the field has been measuring
the wrong thing is the contribution.

---

## Data Access

**Neither ADNI nor OASIS-3 data are included in this repository.**

| Dataset | Role | Access |
|---|---|---|
| ADNI (FBP, FBB, PIB, NAV + Centiloid scores) | Primary training and validation | [adni.loni.usc.edu](https://adni.loni.usc.edu) — DUA, ~1 week |
| OASIS-3 | External validation cohort | [oasis-brains.org](https://www.oasis-brains.org) — free registration |

Raw data files are excluded from this repository by `.gitignore` and must never be committed.
Per the ADNI DUA, participant-level data must not be shared with any third-party platform
or cloud service. All computation must be performed locally.

---

## Repository Structure

```
abpet-threshold/
├── src/
│   ├── preprocess.py      # NIfTI to npy: 9-step standardized pipeline + zone labeling
│   ├── dataset.py         # PETDataset: loads npy arrays, returns image/centiloid/tracer/zone
│   ├── model.py           # Six model variants: BaselineCNN, ResNet18_3D, +FiLM, +MedNet
│   ├── film.py            # FiLM conditioning layer — Perez et al. AAAI 2018
│   ├── losses.py          # MAELoss (Models A-C), HuberLoss delta=10 (Models D-F)
│   ├── augmentation.py    # RandomFlip3D, RandomRotation3D, GaussianNoise3D
│   ├── train.py           # Training loop: AdamW, cosine LR, early stopping, mixed precision
│   ├── evaluate.py        # TSA, zone-stratified MAE, misclassification rate, Bland-Altman
│   ├── ablation.py        # Sequential A-F runner: produces Tables 2 and 3
│   ├── figures.py         # Figures 1-4, 300 DPI, publication-ready
│   └── label_noise.py     # ±7.43 CL noise floor constants and figure reference bands
│
├── governance/
│   ├── novelty_statement.md         # G0 gate — completed before any writing
│   ├── clinical_framing_register.md # All clinical claims with primary sources
│   ├── label_noise_register.md      # Pipeline noise floor documentation
│   ├── portfolio_connection.md      # Research program thesis map
│   ├── adni_dua_compliance.md       # DUA obligations, prohibited uses, DPC process
│   ├── decision_log.md              # Append-only timestamped modeling decisions
│   └── run_manifest.json            # Per-run metadata
│
├── manuscript/
│   ├── ABPET_threshold_v1.md        # Full IMRAD draft
│   ├── abstract_v1.md               # Structured abstract
│   ├── references.bib               # BibTeX
│   └── CLAIM_checklist.md           # CLAIM reporting compliance
│
├── artifacts/
│   ├── tables/                      # Table 1 (dataset), Table 2 (ablation), Table 3 (clinical)
│   └── figures/                     # Figures 1-4 as PNG
│
├── data/processed/                  # ⛔ GITIGNORED — ADNI-derived files
├── models/                          # Checkpoints — available on request post-acceptance
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Six-Model Ablation

Each model tests one architectural variable in isolation, allowing independent attribution
of MAE and TSA improvements to specific design choices.

| Model | Architecture | LR Schedule | Augmentation | Tracer Fusion | Pretrain |
|---|---|---|---|---|---|
| A | BaselineCNN (4 ConvBlocks) | None | None | Concat 8-dim | None |
| B | BaselineCNN | Cosine | None | Concat 8-dim | None |
| C | BaselineCNN | Cosine | Flip / Rotate / Noise | Concat 8-dim | None |
| D | ResNet18-3D | Cosine | Flip / Rotate / Noise | Concat 8-dim | None |
| E | ResNet18-3D | Cosine | Flip / Rotate / Noise | FiLM 32-dim | None |
| F | ResNet18-3D | Cosine | Flip / Rotate / Noise | FiLM 32-dim | MedicalNet |

**FiLM conditioning** (Models E, F) applies a learned affine transformation per tracer at
each encoder stage: y = γ(t) · x + β(t). This is mechanistically distinct from concatenative
fusion, which injects tracer identity only at the regression head.

---

## Primary Metrics

```python
def threshold_specific_accuracy(y_true, y_pred, threshold):
    """
    Proportion of patients correctly classified above or below a clinical threshold.
    TSA_24: lecanemab eligibility threshold (24 CL)
    TSA_10: donanemab cessation / amyloid negativity threshold (10 CL)
    """
    return ((y_true >= threshold) == (y_pred >= threshold)).mean()

def zone_stratified_mae(y_true, y_pred, low=10, high=30):
    """
    MAE computed separately per clinical zone.
    Zone B is the treatment decision zone. If Zone B MAE >> overall MAE,
    global MAE is concealing the failure that matters clinically.
    """
    mask_b = (y_true >= low) & (y_true <= high)
    return {
        'zone_a': np.abs(y_true[y_true < low] - y_pred[y_true < low]).mean(),
        'zone_b': np.abs(y_true[mask_b] - y_pred[mask_b]).mean(),
        'zone_c': np.abs(y_true[y_true > high] - y_pred[y_true > high]).mean(),
    }
```

---

## Quickstart

```bash
git clone https://github.com/julian-borges-md/abpet-threshold.git
cd abpet-threshold
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

After obtaining ADNI access:

```bash
# Step 1: Preprocess ADNI NIfTI files
python src/preprocess.py \
    --adni_dir /path/to/adni_nifti \
    --metadata_csv /path/to/adni_metadata.csv \
    --output_dir data/processed/

# Step 2: Run full 6-model ablation
python src/ablation.py \
    --train_csv data/processed/adni_train.csv \
    --val_csv data/processed/adni_val.csv \
    --epochs 100 \
    --output_dir results/ablation/

# Step 3: Generate manuscript figures
python src/figures.py \
    --results_dir results/ablation/ \
    --output_dir artifacts/figures/
```

---

## Governance

This project operates under Frontier Translational Research Lab governance standards.
All analytical decisions are documented before execution.

| Document | Purpose | Gate |
|---|---|---|
| `novelty_statement.md` | Establishes that no published work reports zone MAE, TSA, or misclassification rates | G0 — completed before Introduction was written |
| `clinical_framing_register.md` | Sources every clinical claim — lecanemab/donanemab thresholds, ARIA, EMA withdrawal | G3 — completed before Discussion |
| `adni_dua_compliance.md` | Documents DUA obligations, prohibited AI tool uses, DPC pre-submission review | Active throughout |
| `decision_log.md` | Append-only record of every modeling decision with rationale | Active throughout |

---

## Target Journals

| Priority | Journal | Rationale |
|---|---|---|
| Primary | Journal of Nuclear Medicine | Core scope for amyloid PET; published Rabinovici 2025 AUC and DeepSUVR 2026 |
| Secondary | Alzheimer's and Dementia: DADM | Clinical AD biomarker focus; open access |
| Tertiary | NeuroImage: Clinical | Neuroimaging methods with clinical application |

---

## Research Portfolio

This paper is part of a research program characterizing metric-outcome mismatch in
clinical AI across multiple domains.

| Project | Domain | Mismatch |
|---|---|---|
| [ati-proteomics](https://github.com/julian-borges-md/ati-proteomics) | Plasma proteomics | AUROC does not capture calibration failure at individual decision level |
| **abpet-threshold** (this repo) | Amyloid PET neuroimaging | Global MAE does not capture eligibility misclassification at treatment threshold |

---

## Citation

```bibtex
@article{borges2026abpet,
  title  = {Threshold-Specific Accuracy Reveals Systematic Failure of Deep Learning
            Centiloid Quantification at the Anti-Amyloid Treatment Eligibility Boundary:
            A Multi-Tracer {ADNI} Analysis},
  author = {Borges, Julian},
  year   = {2026},
  note   = {Manuscript in preparation.
            https://github.com/julian-borges-md/abpet-threshold}
}
```

Data acknowledgment per ADNI DUA:
> Data used in preparation of this article were obtained from the Alzheimer's Disease
> Neuroimaging Initiative (ADNI) database (adni.loni.usc.edu), NIH Grant U19AG024904,
> PI: Michael W. Weiner, MD.

---

## License

MIT License. See [LICENSE](LICENSE).
Data use governed by the ADNI DUA and OASIS-3 data use agreement independently.

---

<div align="center">

**Frontier Translational Research Lab**

Department of Computer Science · Boston University · Harvard Medical School GCSRT Alumni

[![Lab Website](https://img.shields.io/badge/Lab-frontier--lab-002244?style=flat-square)](https://julian-borges-md.github.io/frontier-lab/)
[![BU CS](https://img.shields.io/badge/BU-Computer_Science-cc0000?style=flat-square)](https://www.bu.edu/cs/)
[![HMS](https://img.shields.io/badge/HMS-GCSRT_Alumni-a51c30?style=flat-square)](https://ghsm.hms.harvard.edu/education/global-clinical-scholars-research-training)
[![ORCID](https://img.shields.io/badge/ORCID-0009--0001--9929--3135-a6ce39?style=flat-square&logo=orcid&logoColor=white)](https://orcid.org/0009-0001-9929-3135)
[![CV](https://img.shields.io/badge/Academic_CV-research--profile-4f46e5?style=flat-square)](https://julian-borges-md.github.io/research-profile/)

*Julian Borges, MD, MS · jyborges@bu.edu*

</div>
