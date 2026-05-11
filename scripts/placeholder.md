# scripts/

This folder contains all Python and SQL scripts for the full NLP pipeline.

| File | Purpose |
|---|---|
| `01_generate_data.py` | Generates 2,000 UK customer reviews with ground-truth sentiment labels — no leakage |
| `02_train_model.py` | Fine-tunes DistilBERT for 3-class sentiment classification with MLflow tracking |
| `03_eda_charts.py` | Exploratory data analysis — generates all 7 static PNG charts |
| `04_inference.py` | Production inference pipeline — single review, batch file, or demo mode |
| `05_analysis_queries.sql` | 10 SQL analytical queries compatible with DuckDB, SQLite, and PostgreSQL |

## How to run in order

```bash
python scripts/01_generate_data.py     # Step 1 — generate data
python scripts/03_eda_charts.py        # Step 2 — generate charts
python scripts/02_train_model.py --demo  # Step 3 — train model (demo, no GPU needed)
python scripts/04_inference.py --demo  # Step 4 — run inference
```
