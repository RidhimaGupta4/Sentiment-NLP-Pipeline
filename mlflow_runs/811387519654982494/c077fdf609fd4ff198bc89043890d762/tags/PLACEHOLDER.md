# tags/

This folder contains one file per tag attached to this run.
Tags are metadata labels that describe the run context rather than its performance.

## Contents

| File | Value | Description |
|---|---|---|
| `mlflow.runName` | distilbert-demo-20240511-1430 | Human-readable name for this run |
| `mlflow.source.name` | scripts/02_train_model.py | Script that launched this run |
| `mlflow.source.type` | LOCAL | Where the run was launched from |
| `mlflow.user` | your-system-username | Username of the person who launched the run |

## What each file looks like inside

Each tag file is a plain text file containing just the value:
```
distilbert-demo-20240511-1430
```
## How run names are generated

The run name is set automatically in `02_train_model.py`:

```python
mlflow.start_run(
    run_name=f"distilbert-demo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
)
```

This means every run has a unique, human-readable name based on the timestamp,
so you can identify runs at a glance in the MLflow UI without needing to decode the run ID hash.

## How to read tags programmatically

```python
import mlflow

mlflow.set_tracking_uri("file://mlflow_runs")
run = mlflow.get_run("c077fdf609fd4ff198bc89043890d762")

for key, value in run.data.tags.items():
    print(f"{key}: {value}")
```

## How to add custom tags

```python
with mlflow.start_run() as run:
    mlflow.set_tag("dataset_version", "v1.0")
    mlflow.set_tag("training_environment", "local_cpu")
    mlflow.set_tag("reviewer", "your-name")
```
