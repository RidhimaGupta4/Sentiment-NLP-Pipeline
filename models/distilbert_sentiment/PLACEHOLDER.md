# models/distilbert_sentiment/

HuggingFace model folder for the fine-tuned DistilBERT sentiment classifier.

| File | Description |
|---|---|
| `config.json` | Model architecture — 3-class classification head, id2label, label2id |
| `tokenizer_config.json` | Tokeniser settings — lowercase, max 128 tokens, padding |
| `special_tokens_map.json` | CLS, SEP, PAD, MASK, UNK special token definitions |
| `training_args.json` | Full training configuration, data splits, and final test metrics |
| `pytorch_model.bin` | **Model weights — generated after full training (not included, see below)** |

## Why pytorch_model.bin is not committed

The model weights file is approximately 250MB and exceeds GitHub's file size limit.
It is generated locally when you run the full training script.

## How to generate weights

```bash
# Install dependencies
pip install transformers torch mlflow

# Run full training
python scripts/02_train_model.py --epochs 3 --batch_size 16 --lr 2e-5
```

## How to load the model in Python

```python
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizerFast.from_pretrained("models/distilbert_sentiment")
model = DistilBertForSequenceClassification.from_pretrained("models/distilbert_sentiment")
model.eval()
```

## Upload weights to HuggingFace Hub (optional)

```python
from huggingface_hub import push_to_hub
model.push_to_hub("YOUR_USERNAME/distilbert-sentiment-uk-reviews")
tokenizer.push_to_hub("YOUR_USERNAME/distilbert-sentiment-uk-reviews")
```
