# params/

This folder contains one file per logged hyperparameter.
Each file holds a single value — the parameter value set before training began.

## Contents

| File | Value | Description |
|---|---|---|
| `model` | distilbert-base-uncased | HuggingFace base model used |
| `epochs` | 3 | Number of training epochs |
| `batch_size` | 16 | Training and evaluation batch size |
| `learning_rate` | 2e-05 | AdamW learning rate |
| `weight_decay` | 0.01 | L2 regularisation coefficient |
| `grad_clip` | 1.0 | Maximum gradient norm for clipping |
| `dropout` | 0.1 | DistilBERT default dropout rate |
| `warmup_ratio` | 0.1 | Proportion of steps used for LR warmup |
| `optimizer` | AdamW | Optimiser with grouped weight decay |
| `random_seed` | 42 | Seed for full reproducibility |
| `split_train` | 1400 | Number of training samples |
| `split_val` | 300 | Number of validation samples |
| `split_test` | 300 | Number of test samples |
| `split_strategy` | stratified_70_15_15 | Stratified split preserving class proportions |
| `class_weights` | computed_on_train_set_only | Class balancing source — no leakage |
| `early_stopping` | val_loss_patience_2 | Stop if val loss does not improve for 2 epochs |
| `num_classes` | 3 | negative, neutral, positive |
| `max_seq_length` | 128 | Maximum token length passed to model |
| `demo_mode` | True | Whether this was a simulated or real training run |

## What each file looks like inside

Each param file is a plain text file containing just the value:
```
distilbert-base-uncased
```
## Why parameters matter

Logging parameters means you can always reproduce any run exactly.
If you change the learning rate or batch size in a future run, MLflow keeps
both sets of parameters so you can see which configuration produced better results.

## How to read params programmatically

```python
import mlflow

mlflow.set_tracking_uri("file://mlflow_runs")
run = mlflow.get_run("c077fdf609fd4ff198bc89043890d762")

for key, value in run.data.params.items():
    print(f"{key}: {value}")
```
