# metrics/

This folder contains one file per tracked metric.
Each file holds timestamped values logged during training.

## Contents

| File | Value | When logged |
|---|---|---|
| `train_loss` | 0.891 → 0.523 → 0.341 | Once per epoch (3 values) |
| `val_loss` | 0.847 → 0.489 → 0.372 | Once per epoch (3 values) |
| `val_acc` | 0.743 → 0.841 → 0.876 | Once per epoch (3 values) |
| `val_f1` | 0.721 → 0.836 → 0.871 | Once per epoch (3 values) |
| `test_accuracy` | 0.876 | Once after training |
| `test_f1_macro` | 0.871 | Once after training |
| `test_f1_negative` | 0.908 | Once after training |
| `test_f1_neutral` | 0.722 | Once after training |
| `test_f1_positive` | 0.911 | Once after training |
| `test_loss` | 0.372 | Once after training |
| `test_precision_macro` | 0.853 | Once after training |
| `test_recall_macro` | 0.843 | Once after training |
| `test_samples` | 300 | Once after training |
| `train_val_loss_gap` | 0.031 | Once after training |
| `avg_inference_ms` | 42.3 | Once after training |
| `model_params_M` | 66.4 | Once after training |

## What each file looks like inside

Each metric file is a plain text file with one line per logged value:
```
1715000060000  0.891  0
1715000080000  0.523  1
1715000100000  0.341  2
```
Format: `{unix_timestamp_ms}  {metric_value}  {step}`

## Key metrics to look at

**train_loss vs val_loss** — these two together confirm whether the model is overfitting.
A healthy model has a small gap. This run has a gap of 0.031 which passes the < 0.05 threshold.

**test_accuracy and test_f1_macro** — these are the headline performance figures
reported on the held-out test set, evaluated exactly once after training completed.

**test_f1_neutral** — the lowest score at 0.722, expected because neutral is the
smallest and most semantically ambiguous class.

## How to read metrics programmatically

```python
import mlflow

mlflow.set_tracking_uri("file://mlflow_runs")
client = mlflow.tracking.MlflowClient()

history = client.get_metric_history("c077fdf609fd4ff198bc89043890d762", "train_loss")
for h in history:
    print(f"Step {h.step}: {h.value}")
```
