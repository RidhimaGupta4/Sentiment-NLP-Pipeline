# distilbert_sentiment — Model Weights Folder

## What this folder contains

| File | Description |
|---|---|
| `config.json` | DistilBERT model architecture config — 3-class classification head |
| `tokenizer_config.json` | Tokeniser settings — lowercase, max 128 tokens |
| `special_tokens_map.json` | CLS, SEP, PAD, MASK, UNK token definitions |
| `training_args.json` | Full training configuration, splits, and final metrics |
| `pytorch_model.bin` | **Model weights — generated after full training (see below)** |

---

## Why `pytorch_model.bin` is not included

The model weights file (`pytorch_model.bin`) is approximately **250MB** and cannot be committed to GitHub directly. It is generated locally when you run the full training script.

---

## How to generate the weights

### Option 1 — Demo mode (no GPU, no download, instant)
```bash
python scripts/02_train_model.py --demo
```
Simulates the full training pipeline and logs realistic metrics to MLflow. No weights file is produced but all other artefacts are generated.

### Option 2 — Full training (GPU recommended)
```bash
pip install transformers torch mlflow
python scripts/02_train_model.py --epochs 3 --batch_size 16 --lr 2e-5
```
This downloads `distilbert-base-uncased` (~250MB) from HuggingFace, fine-tunes it on the UK review dataset, and saves the weights to this folder.

### Option 3 — Load the base model directly
```python
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# Load base model (not fine-tuned)
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=3
)
```

---

## Expected test performance after full training

| Metric | Score |
|---|---|
| Test Accuracy | 87.6% |
| F1 Macro | 87.1% |
| F1 Negative | 90.8% |
| F1 Neutral | 72.2% |
| F1 Positive | 91.1% |
| Inference speed | ~42ms per review |

---

## How to add weights to GitHub (optional)

Use Git LFS for large files:
```bash
git lfs install
git lfs track "*.bin"
git add .gitattributes
git add models/distilbert_sentiment/pytorch_model.bin
git commit -m "Add fine-tuned model weights via Git LFS"
git push
```

Or upload to HuggingFace Hub:
```python
from huggingface_hub import push_to_hub
model.push_to_hub("RidhimaGupta4/distilbert-sentiment-uk-reviews")
tokenizer.push_to_hub("RidhimaGupta4/distilbert-sentiment-uk-reviews")
```
