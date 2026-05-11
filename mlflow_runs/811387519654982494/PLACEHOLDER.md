# mlflow_runs/{experiment_id}/

This folder represents the MLflow experiment named **sentiment-nlp-pipeline**.

## What an experiment ID is

MLflow assigns a unique numeric ID to every named experiment automatically.
The long number you see as this folder name is that ID.
It is permanent — every training run for this project is stored inside this folder.

## Contents

| Item | Description |
|---|---|
| `meta.yaml` | Experiment name, creation timestamp, artifact storage location |
| `{run_id}/` | One subfolder per training run — each run gets a unique hash ID |

## meta.yaml contains

```yaml
artifact_location: file:///path/to/mlflow_runs/{experiment_id}
creation_time: 1715000000000
experiment_id: "811387519654982494"
last_update_time: 1715000000000
lifecycle_stage: active
name: Sentiment-NLP-Pipeline
```

## How to add more runs

Every time you run the training script, a new `{run_id}/` subfolder is created here automatically:

```bash
# Each of these commands creates a new run subfolder
python scripts/02_train_model.py --demo
python scripts/02_train_model.py --epochs 3 --lr 2e-5
```

## How to compare runs in the UI

```bash
mlflow ui --backend-store-uri file://mlflow_runs
```

In the UI you can select multiple runs and compare their metrics and parameters side by side.
