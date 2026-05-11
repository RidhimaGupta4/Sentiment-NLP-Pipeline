# artifacts/

This folder contains files that were saved and attached to this training run.

## Contents

| File | Description |
|---|---|
| `eval_results.json` | Full confusion matrix, per-class F1 scores, overfitting check, leakage check |
| `model_card.json` | Complete model card — architecture, training config, limitations, intended use |

## What artifacts are

In MLflow, an artifact is any file you choose to save alongside a run.
This is useful for saving evaluation results, model configs, plots, or
any file that explains or documents the run in more detail than metrics alone.

## eval_results.json contains

- Confusion matrix (3×3) with row = actual, col = predicted
- Per-class precision, recall, F1, and support
- Macro and weighted averages
- Training curve per epoch
- Overfitting check — train/val loss gap = 0.031 (PASS, threshold < 0.05)
- Leakage check — confirms no data leakage in the pipeline

## model_card.json contains

- Base model and architecture details
- Training configuration
- Data splits and label distribution
- Final test set performance
- Known limitations
- Intended and non-intended use cases
- How to load and run the model

## How to access artifacts programmatically

```python
import mlflow

mlflow.set_tracking_uri("file://mlflow_runs")
client = mlflow.tracking.MlflowClient()

artifacts = client.list_artifacts("c077fdf609fd4ff198bc89043890d762")
for a in artifacts:
    print(a.path)
```

## How to regenerate

```bash
python scripts/02_train_model.py --demo
```

Both files are re-saved to artifacts on every run automatically.
