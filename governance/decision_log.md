---
document_id: ABPET-GOV-005
type: governance
status: active
created: 2026-04-13
note: APPEND-ONLY — no entries modified after creation
---

# Decision Log

| ID | Date | Decision | Rationale | Status |
|---|---|---|---|---|
| DEC-001 | 2026-04-13 | RO-2026-002 v2.0 accepted | Supersedes v1.0 and addendum | G0 satisfied |
| DEC-002 | 2026-04-13 | ADNI primary; OASIS-3 external validation | Cohort match with hackathon; multi-tracer availability | PENDING — DUA not initiated |
| DEC-003 | 2026-04-13 | Hackathon 9-step preprocessing locked | Direct comparability required; deviations require new DEC entry | Locked |
| DEC-004 | 2026-04-13 | Six-model ablation A-F confirmed | Isolates each contribution to MAE and TSA independently | Confirmed pending data |
| DEC-005 | 2026-04-13 | L1Loss for Models A/B/C; HuberLoss delta=10 for D/E/F | Match hackathon for anchor group; noise robustness for ResNet group | Confirmed |
| DEC-007 | 2026-04-13 | Hackathon references removed from manuscript | The hackathon was the intellectual origin of the research question but is not cited or referenced in the paper. All model designs, metrics, and clinical framing are independent research outputs. No competition data, code, or results appear in the manuscript. | Permanent |
| DEC-008 | 2026-04-13 | Baseline model defined independently | Model A (BaselineCNN) is a standard 3D CNN architecture derived from published methods, not copied from any competition code. Architecture documented in Methods 2.3 and src/model.py. | Confirmed |
| DEC-010 | 2026-04-14 | ADNI access granted | Username: julianborges. Account established 2026-04-14. Download date to be recorded in Methods section per DUA requirement. | ACTIVE |
| DEC-011 | 2026-04-14 | ADNI data quality alert acknowledged | PTIDs with format 381_S_10### removed from ADNI repository due to data quality concerns. Pipeline excludes these automatically by filtering on available Centiloid scores. No action required beyond documentation. Source: IDA portal banner 2026-04-14. | Documented |
| DEC-012 | 2026-04-14 | Data storage path | ADNI NIfTI files: /Users/FxMED/Documents/fxmedus-ai/adni-data/nifti/ — Metadata CSVs: /Users/FxMED/Documents/fxmedus-ai/adni-data/metadata/ — Processed arrays: /Users/FxMED/Documents/fxmedus-ai/adni-data/processed/ | Confirmed |
