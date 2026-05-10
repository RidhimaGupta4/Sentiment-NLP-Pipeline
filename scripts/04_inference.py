"""
Customer Sentiment NLP Pipeline — Script 04: Inference
=======================================================
Production inference script. Classifies new review text,
extracts topics, scores priority, and returns structured JSON.

Usage:
    # Single review
    python scripts/04_inference.py --text "Terrible product, nearly caused a fire."

    # Batch file (one review per line)
    python scripts/04_inference.py --file my_reviews.txt

    # Demo (no model needed)
    python scripts/04_inference.py --demo

    # Full model (requires 02_train_model.py to have been run)
    python scripts/04_inference.py --text "Great product!" --use_model
"""

import argparse, json, os, time

parser = argparse.ArgumentParser()
parser.add_argument("--text",      type=str)
parser.add_argument("--file",      type=str)
parser.add_argument("--demo",      action="store_true")
parser.add_argument("--use_model", action="store_true")
parser.add_argument("--output",    type=str, default="outputs/inference_results.json")
args = parser.parse_args()

MODEL_PATH = "models/distilbert_sentiment"

TOPIC_KEYWORDS = {
    "delivery":         ["delivery","arrived","shipping","dispatch","courier","parcel","tracking","late","delayed"],
    "customer_service": ["service","customer","support","response","email","phone","helpful","ignored","useless","resolved","staff"],
    "returns_refunds":  ["return","refund","exchange","dispute","money back","paypal","chargeback","rejected"],
    "quality":          ["quality","broke","broken","material","build","durable","flimsy","sturdy","fell apart","faulty"],
    "product_accuracy": ["described","advertised","photo","image","accurate","different","misleading","nothing like"],
    "price_value":      ["price","value","worth","expensive","cheap","overpriced","bargain"],
    "safety":           ["dangerous","safety","hazard","injury","fire","burn","hospital","hurt","risk","recall"],
    "packaging":        ["packaging","packaged","box","wrapped","damaged","protected","bubble"],
}

PRIORITY_SIGNALS = {
    "safety_hazard":  ["fire","burn","spark","melt","overheat","shock","injury","hospital","dangerous","hazard","recall"],
    "health_risk":    ["allergic","anaphylactic","poison","toxic","ill","rash","reaction"],
    "legal_threat":   ["solicitor","lawyer","legal action","court","trading standards","cedr","ombudsman","police"],
    "fraud_scam":     ["fraud","scam","counterfeit","chargeback","unauthorised"],
    "data_privacy":   ["data breach","gdpr","hacked","compromised"],
    "child_safety":   ["child","baby","toddler","choking","sharp edge","kid injured"],
    "discrimination": ["racist","racism","discrimination","sexist","homophobic"],
}


def extract_topics(text: str) -> list:
    t = text.lower()
    found = [topic for topic, kws in TOPIC_KEYWORDS.items() if any(k in t for k in kws)]
    return found if found else ["general"]


def score_priority(text: str, sentiment: str, neg_conf: float) -> dict:
    t = text.lower()
    signals = [cat for cat, kws in PRIORITY_SIGNALS.items() if any(k in t for k in kws)]
    score = len(signals) * 4
    if sentiment == "negative" and neg_conf > 0.90: score += 3
    if neg_conf > 0.95: score += 2
    caps = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
    if caps >= 2: score += 1
    tier = "Critical" if score >= 10 else "High" if score >= 6 else "Watch" if score >= 2 else "None"
    return {"tier": tier, "score": score, "signals": signals}


def rule_based_classify(text: str) -> dict:
    """
    Heuristic classifier for demo mode.
    Uses lexicon — NOT the trained model.
    Clearly labelled as heuristic in output.
    """
    t = text.lower()
    neg_words = ["terrible","awful","broken","scam","fraud","dangerous","appalling",
                 "disgusting","useless","worst","dreadful","horrible","shocking",
                 "furious","devastating","avoid","rubbish","nightmare","abysmal"]
    pos_words = ["excellent","brilliant","fantastic","amazing","perfect","superb",
                 "outstanding","wonderful","great","thrilled","delighted","pleased",
                 "happy","love","exceptional","impressive","recommend","fast"]

    ns = sum(1 for w in neg_words if w in t)
    ps = sum(1 for w in pos_words if w in t)

    if ns > ps:
        cn = round(min(0.99, 0.60 + ns * 0.08), 3)
        cp = round(max(0.01, 0.15 - ns * 0.02), 3)
        sent = "negative"
    elif ps > ns:
        cp = round(min(0.99, 0.60 + ps * 0.08), 3)
        cn = round(max(0.01, 0.15 - ps * 0.02), 3)
        sent = "positive"
    else:
        cp, cn = 0.38, 0.32
        sent = "neutral"

    cnu = round(max(0.00, 1 - cp - cn), 3)
    return {"sentiment": sent, "confidence": {"positive": cp, "negative": cn, "neutral": cnu}}


def model_classify(text: str) -> dict:
    """Full DistilBERT inference using saved model weights."""
    import torch
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs  = torch.softmax(logits, dim=1)[0].tolist()
    pred = label_map[probs.index(max(probs))]
    return {"sentiment": pred, "confidence": {
        "negative": round(probs[0], 3),
        "neutral":  round(probs[1], 3),
        "positive": round(probs[2], 3),
    }}


def classify_review(text: str) -> dict:
    t0 = time.time()
    use_model = args.use_model and os.path.exists(MODEL_PATH)
    if use_model:
        result = model_classify(text)
        model_used = "distilbert-base-uncased (fine-tuned)"
    else:
        result = rule_based_classify(text)
        model_used = "rule-based heuristic (demo — not the trained model)"

    sent    = result["sentiment"]
    conf    = result["confidence"]
    topics  = extract_topics(text)
    prio    = score_priority(text, sent, conf["negative"])
    ms      = round((time.time() - t0) * 1000, 1)

    return {
        "text":         text,
        "sentiment":    sent,
        "confidence":   conf,
        "topics":       topics,
        "priority":     prio,
        "word_count":   len(text.split()),
        "model_used":   model_used,
        "inference_ms": ms,
    }


DEMO_REVIEWS = [
    "Absolutely brilliant product, arrived next day and works perfectly. Highly recommend.",
    "Terrible quality. The product broke within a week. Complete waste of money. Avoid.",
    "Product is okay, nothing special. Delivery took longer than expected but arrived.",
    "URGENT: Product is a safety hazard. It sparked and almost caused a fire. Recall needed.",
    "Disgusting customer service. Ignored my emails for two weeks. Raised a PayPal dispute.",
    "Fantastic quality for the price. Delivery was super fast. Five stars without hesitation.",
    "WARNING: Allergic reaction requiring hospital treatment. Solicitor has been instructed.",
    "Decent enough for the price. Not quite what I expected but does the job adequately.",
]


def main():
    os.makedirs("outputs", exist_ok=True)

    if args.demo or (not args.text and not args.file):
        reviews = DEMO_REVIEWS
        print("=" * 62)
        print(" INFERENCE DEMO — Customer Sentiment NLP Pipeline")
        print("=" * 62)
    elif args.text:
        reviews = [args.text]
    elif args.file:
        with open(args.file) as f:
            reviews = [line.strip() for line in f if line.strip()]
    else:
        reviews = DEMO_REVIEWS

    print(f"\nClassifying {len(reviews)} review(s)...\n")
    results = []

    for i, text in enumerate(reviews, 1):
        r = classify_review(text)
        results.append(r)

        tier_icon = {"Critical":"🔴","High":"🟠","Watch":"🟡","None":"🟢"}.get(r["priority"]["tier"],"⚪")
        sent_icon = {"positive":"✅","negative":"❌","neutral":"➡️"}.get(r["sentiment"],"")
        sigs      = ", ".join(r["priority"]["signals"]) if r["priority"]["signals"] else "none"

        print(f"[{i}] {sent_icon} {r['sentiment'].upper()}")
        print(f"     Confidence : pos={r['confidence']['positive']:.2f}  "
              f"neg={r['confidence']['negative']:.2f}  "
              f"neu={r['confidence']['neutral']:.2f}")
        print(f"     Topics     : {', '.join(r['topics'])}")
        print(f"     Priority   : {tier_icon} {r['priority']['tier']} "
              f"(score={r['priority']['score']}, signals=[{sigs}])")
        print(f"     Model      : {r['model_used']}")
        print(f"     Text       : \"{text[:80]}{'…' if len(text)>80 else ''}\"")
        print()

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output}")

    sents = [r["sentiment"] for r in results]
    tiers = [r["priority"]["tier"] for r in results]
    print(f"\n── Summary ──")
    print(f"  Positive: {sents.count('positive')}  Negative: {sents.count('negative')}  Neutral: {sents.count('neutral')}")
    print(f"  Critical: {tiers.count('Critical')}  High: {tiers.count('High')}  Watch: {tiers.count('Watch')}  None: {tiers.count('None')}")


if __name__ == "__main__":
    main()
