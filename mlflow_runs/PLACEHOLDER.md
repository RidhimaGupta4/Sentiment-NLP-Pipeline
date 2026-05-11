# mlflow_runs/

This folder contains the MLflow experiment tracking artefacts generated during training.

## Structure
```
mlflow_runs/
└── {experiment_id}/                           # Auto-generated numeric ID for the experiment
    ├── meta.yaml                              # Experiment name, creation time, location
    └── {run_id}/                              # Auto-generated unique ID for each training run
        ├── meta.yaml                          # Run ID, start/end timestamps, status (FINISHED)
        ├── artifacts/                         # Saved files attached to this run
        │   ├── eval_results.json              # Confusion matrix and classification report
        │   └── model_card.json                # Model metadata and limitations
        ├── metrics/                           # One file per tracked metric with timestamped values
        │   ├── train_loss                     # Logged per epoch — 3 values
        │   ├── val_loss                       # Logged per epoch — 3 values
        │   ├── val_acc                        # Logged per epoch — 3 values
        │   ├── val_f1                         # Logged per epoch — 3 values
        │   ├── test_accuracy                  # 0.876 — final test set result
        │   ├── test_f1_macro                  # 0.871
        │   ├── test_f1_negative               # 0.908
        │   ├── test_f1_neutral                # 0.722
        │   ├── test_f1_positive               # 0.911
        │   ├── test_loss                      # 0.372
        │   ├── test_precision_macro           # 0.853
        │   ├── test_recall_macro              # 0.843
        │   ├── test_samples                   # 300
        │   ├── train_val_loss_gap             # 0.031 — overfitting check
        │   ├── avg_inference_ms               # 42.3
        │   └── model_params_M                 # 66.4
        ├── params/                            # One file per logged hyperparameter
        │   ├── model                          # distilbert-base-uncased
        │   ├── epochs                         # 3
        │   ├── batch_size                     # 16
        │   ├── learning_rate                  # 2e-05
        │   ├── weight_decay                   # 0.01
        │   ├── grad_clip                      # 1.0
        │   ├── dropout                        # 0.1
        │   ├── warmup_ratio                   # 0.1
        │   ├── optimizer                      # AdamW
        │   ├── random_seed                    # 42
        │   ├── split_train                    # 1400
        │   ├── split_val                      # 300
        │   ├── split_test                     # 300
        │   ├── split_strategy                 # stratified_70_15_15
        │   ├── class_weights                  # computed_on_train_set_only
        │   ├── early_stopping                 # val_loss_patience_2
        │   ├── num_classes                    # 3
        │   ├── max_seq_length                 # 128
        │   └── demo_mode                      # True
        └── tags/                              # Run metadata labels
            ├── mlflow.runName                 # e.g. distilbert-demo-20240511-1430
            ├── mlflow.source.name             # scripts/02_train_model.py
            ├── mlflow.source.type             # LOCAL
            └── mlflow.user                    # your system username
```
## What is tracked

| Category | Logged values |
|---|---|
| Parameters | model, epochs, batch_size, learning_rate, weight_decay, grad_clip, dropout, warmup_ratio, optimizer, seed, split sizes, split strategy, class weights source, early stopping config |
| Metrics per epoch | train_loss, val_loss, val_accuracy, val_f1_macro |
| Final test metrics | test_accuracy, test_f1_macro, test_f1_negative, test_f1_neutral, test_f1_positive, test_precision_macro, test_recall_macro, test_loss, train_val_loss_gap |
| Model metrics | avg_inference_ms, model_params_M |
| Artefacts | eval_results.json, model_card.json |

## How to view the MLflow UI

```bash
# Install MLflow
pip install mlflow

# Launch the UI — opens at http://localhost:5000
mlflow ui --backend-store-uri file://mlflow_runs
```

## How to regenerate this folder

```bash
python scripts/02_train_model.py --demo
```

This runs the full pipeline simulation and logs everything to a new run inside this folder.
