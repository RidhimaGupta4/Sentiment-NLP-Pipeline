# mlflow_runs/{experiment_id}/{run_id}/

This folder represents one complete training run of the DistilBERT sentiment model.

## What a run ID is

MLflow assigns a unique hexadecimal hash to every training run automatically.
The hash you see as this folder name is that run ID.
Every time you run `02_train_model.py`, a new folder with a new hash is created.

## Contents

| Subfolder or File | Description |
|---|---|
| `meta.yaml` | Run ID, start time, end time, status, run name |
| `artifacts/` | Files saved and attached to this run |
| `metrics/` | One file per tracked metric — values logged per epoch |
| `params/` | One file per hyperparameter — values logged at run start |
| `tags/` | Run name, source script, source type, user |

## meta.yaml contains

```yaml
artifact_uri: file:///path/to/mlflow_runs/{experiment_id}/{run_id}/artifacts
end_time: 1715000120000
entry_point_name: ""
experiment_id: "811387519654982494"
lifecycle_stage: active
run_id: c077fdf609fd4ff198bc89043890d762
run_name: distilbert-demo-20240511-1430
source_name: scripts/02_train_model.py
source_type: 4
source_version: ""
start_time: 1715000000000
status: 3
tags: []
user_id: RidhimaGupta4
```

## Status codes

| Code | Meaning |
|---|---|
| 3 | FINISHED — run completed successfully |
| 4 | FAILED — run encountered an error |
| 2 | RUNNING — run is currently active |

## How to load this run programmatically

```python
import mlflow

mlflow.set_tracking_uri("file://mlflow_runs")
run = mlflow.get_run("c077fdf609fd4ff198bc89043890d762")

print(run.data.params)   # all hyperparameters
print(run.data.metrics)  # all final metric values
```
