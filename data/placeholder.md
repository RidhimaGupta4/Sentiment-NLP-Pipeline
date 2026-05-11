# data/

This folder holds all datasets used in the project.

| Subfolder | Contents |
|---|---|
| `processed/` | Cleaned, labelled, and aggregated datasets ready for analysis and dashboard |

## Data integrity

- All labels are assigned from review text content only — no target leakage
- Train/val/test splits are performed downstream in `02_train_model.py` — never here
- Priority flags are computed independently of sentiment labels
- No confidence scores are used to create or modify labels
