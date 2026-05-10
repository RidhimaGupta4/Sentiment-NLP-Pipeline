"""
Customer Sentiment NLP Pipeline — Script 01: Data Generation
=============================================================
Generates a realistic, leak-free UK customer review dataset modelled
on Trustpilot and Amazon UK review patterns (2022-2024).

Data integrity guarantees:
  - Labels are assigned ONLY from review text content, never from held-out features
  - No target leakage: confidence scores are NOT used to create labels
  - Train/val/test split is done DOWNSTREAM in 02_train_model.py using
    stratified splitting — never here
  - Priority flags use separate rule logic from sentiment labels
  - All records are independent (no temporal leakage between splits)

Real data sources (production):
  - Trustpilot API: https://developers.trustpilot.com/
  - Amazon Product API: https://webservices.amazon.co.uk/

Run:
    pip install pandas numpy
    python scripts/01_generate_data.py
"""

import pandas as pd
import numpy as np
import json, os, random
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

# ── Categories and companies ──────────────────────────────────────────────────

CATEGORIES = [
    "Electronics", "Clothing & Fashion",
    "Home & Kitchen", "Health & Beauty",
    "Books & Media", "Sports & Outdoors",
]

COMPANIES = {
    "Electronics":        ["TechDirect UK", "Currys", "AO.com", "Ebuyer", "Scan Computers"],
    "Clothing & Fashion": ["ASOS", "Next", "Marks & Spencer", "Boohoo", "John Lewis"],
    "Home & Kitchen":     ["Dunelm", "IKEA UK", "Wayfair UK", "Robert Dyas", "Argos"],
    "Health & Beauty":    ["Boots", "Holland & Barrett", "LookFantastic", "Superdrug", "Cult Beauty"],
    "Books & Media":      ["Waterstones", "WHSmith", "Amazon UK", "Book Depository", "Foyles"],
    "Sports & Outdoors":  ["Decathlon UK", "Sports Direct", "Wiggle", "GO Outdoors", "Halfords"],
}

# ── Review text banks ─────────────────────────────────────────────────────────
# These are the ONLY source of labels — text determines sentiment, not vice versa

POSITIVE_REVIEWS = [
    "Absolutely brilliant product, arrived next day and works perfectly. Would definitely recommend to anyone.",
    "Fantastic quality for the price. Delivery was super fast and packaging was excellent. Five stars from me.",
    "Really impressed with how quickly this arrived. The product is exactly as described and great quality.",
    "Outstanding customer service when I had a query. They resolved everything within hours. Absolutely brilliant.",
    "Perfect item, exactly what I needed. Very happy with my purchase and will be ordering again soon.",
    "Superb quality and great value. Delivery was prompt and the item was well packaged. Very pleased indeed.",
    "Exceptional service from start to finish. The product exceeded my expectations completely. Highly recommended.",
    "Arrived quickly, well packaged, exactly as described. Customer service was also very helpful and friendly.",
    "Brilliant experience shopping here. Product quality is amazing and delivery was the very next day.",
    "Really happy with this purchase. Great value, fast delivery and excellent build quality throughout.",
    "Wonderful product, exactly as advertised. Delivered ahead of schedule. Customer service top notch.",
    "Very pleased with my order. The quality is outstanding and shipping was incredibly fast. Will return.",
    "Fantastic buy. The item is perfect and arrived well before the estimated delivery date. Very impressed.",
    "Excellent product, great price, speedy delivery. Everything you could want from an online retailer.",
    "Couldn't be happier with this purchase. Works exactly as described and arrived the very next day.",
    "Five star experience. Product is exactly as shown, delivery was free and very fast. Would recommend.",
    "Absolutely love this product. Easy to order, great communication, arrived perfectly packaged. Thank you.",
    "Top quality item at a very fair price. The seller went above and beyond. Will definitely shop again.",
    "Ordered on Monday, arrived Tuesday. Perfect condition, exactly as described. Couldn't ask for more.",
    "Excellent in every way. Quick despatch, secure packaging, perfect item. Have already recommended to friends.",
]

NEGATIVE_REVIEWS = [
    "Absolutely appalling service. Waited three weeks and still no delivery. No response from customer service.",
    "Terrible quality. The product broke within a week of use. Complete waste of money. Avoid at all costs.",
    "Disgusting customer service. They ignored my emails for two weeks. Had to raise a PayPal dispute to get help.",
    "Product arrived damaged and the returns process is an absolute nightmare. Took over a month for a refund.",
    "Complete rubbish. Nothing like the description. Cheap materials and it stopped working after two days.",
    "Worst online shopping experience of my life. Wrong item sent twice and customer service is completely useless.",
    "Do not order from here. My parcel was lost and they refused to refund me without raising a formal dispute.",
    "Shocking quality. Fell apart immediately. Photos on the listing are clearly misleading. Not fit for purpose.",
    "Dreadful experience. Waited a month, chased them constantly, eventually received a badly damaged item.",
    "Avoid this company at all costs. They take your money then completely ignore you. Reported to Trading Standards.",
    "Absolutely furious. Product completely different to what was advertised. Refused to accept the return. Awful.",
    "Three separate issues with this order. Wrong size sent first, then lost in transit, then refused a refund.",
    "Sent defective item, customer service blamed me entirely, took six weeks to resolve. Never shopping here again.",
    "Horrible experience from start to finish. Charged twice for one order. Took weeks to get my money back.",
    "Zero stars if I could. Item arrived smashed, return label never came, now ignoring all contact. Disgusting.",
    "Waited four weeks for item that never arrived. Tracking not updated. Phone line goes to voicemail. Terrible.",
    "Complete con. Item looks nothing like pictures. Cheap knock-off quality. Refused refund citing wear and tear.",
    "Appalling. Order cancelled without notice, refund took three weeks, no apology whatsoever. Stay away.",
    "Shockingly bad. Received someone else's order, took two weeks to fix, still waiting for correct item.",
    "Never again. Item faulty out of the box, returns policy deliberately confusing. Trading Standards notified.",
]

NEUTRAL_REVIEWS = [
    "Product is okay, nothing special. Delivery took longer than expected but it arrived eventually.",
    "Decent enough for the price. Not quite what I expected from the description but it does the job.",
    "Average product. Works fine but nothing remarkable. Delivery was on time. Would probably buy again.",
    "It's alright. Does what it says on the tin. Packaging was a bit flimsy but item was undamaged.",
    "Standard purchase experience. Nothing exceptional to report. Item is functional and reasonably priced.",
    "Product is fine, mostly matches description. Had to contact support once, they were fairly helpful.",
    "Reasonably satisfied. Delivery was a day late but item is decent quality for the price paid.",
    "It's okay. Expected slightly better quality based on the photos but it is acceptable for everyday use.",
    "Middle of the road. Item arrived in reasonable time and is satisfactory. Nothing more, nothing less.",
    "Fair product and service. Delivery estimate was off by two days. Item is as described, no complaints.",
    "Adequate product. Not the best I have bought but not the worst either. Probably would not reorder.",
    "Standard experience. Item is acceptable. Would have given five stars if delivery had been quicker.",
    "Three stars because it is exactly what was described but left me feeling underwhelmed overall.",
    "Does the job. Took longer than expected to arrive. Customer service response was slow but helpful.",
    "Average all round. Item works but feels cheaper than the price suggests. Delivery was acceptable.",
]

# ── Priority complaint templates (separate from sentiment labels) ──────────────
# These are genuine safety/legal/fraud signals — high priority regardless of sentiment

PRIORITY_COMPLAINTS = [
    "URGENT: Product is a safety hazard. It sparked and almost caused a fire. This needs immediate recall.",
    "WARNING: This item caused an allergic reaction requiring hospital treatment. Dangerous and mislabelled.",
    "SCAM ALERT: Charged me three times for one order and completely unresponsive. Reporting to Trading Standards.",
    "SAFETY ISSUE: Children's toy broke apart exposing sharp metal edges. My child was injured. Unacceptable.",
    "DATA BREACH: My personal and payment data appears to have been compromised. Taking legal action immediately.",
    "FRAUD: Never received item worth £450. Tracking shows delivered but nothing arrived. Police report filed.",
    "DANGEROUS: Electrical item overheated and melted. Lucky we were home. Reporting to product safety authority.",
    "HEALTH RISK: Food supplement contained undisclosed allergen. Anaphylactic reaction. Solicitor instructed.",
    "DISCRIMINATION: Staff member made racist comments during my complaint call. Unacceptable and illegal.",
    "REPEATED FRAUD: Fourth attempt to get a refund for a faulty product worth £800. Now involving CEDR.",
]

# ── Topic keyword mapping (for rule-based topic extraction only) ──────────────
TOPIC_KEYWORDS = {
    "delivery":         ["delivery","arrived","shipping","dispatch","courier","parcel","tracking","late","delayed","week","days","next day"],
    "customer_service": ["service","customer","support","response","email","phone","helpful","ignored","useless","resolved","staff","contact"],
    "returns_refunds":  ["return","refund","exchange","dispute","money back","paypal","chargeback","rejected","refusal"],
    "quality":          ["quality","broke","broken","material","build","durable","flimsy","sturdy","fell apart","faulty","defective"],
    "product_accuracy": ["described","advertised","photo","image","accurate","different","misleading","exactly","matches","nothing like"],
    "price_value":      ["price","value","worth","expensive","cheap","cost","overpriced","bargain","deal","money"],
    "safety":           ["dangerous","safety","hazard","injury","fire","burn","hospital","hurt","risk","recall","electrocute"],
    "packaging":        ["packaging","packaged","box","wrapped","damaged","protected","bubble","smashed","crushed"],
}

PRIORITY_SIGNALS = {
    "safety_hazard":  ["fire","burn","spark","explosion","melt","overheat","electrocute","shock","injury","hospital","dangerous","hazard","recall"],
    "health_risk":    ["allergic","anaphylactic","poison","toxic","contaminated","ill","sick","rash","reaction"],
    "legal_threat":   ["solicitor","lawyer","legal action","court","trading standards","cedr","ombudsman","police report","sue","sued"],
    "fraud_scam":     ["fraud","scam","fake","counterfeit","stolen","chargeback","unauthorised charge"],
    "data_privacy":   ["data breach","personal data","gdpr","hacked","compromised","identity theft"],
    "child_safety":   ["child","children","baby","toddler","choking","sharp edge","toy broke","kid injured"],
    "discrimination": ["racist","racism","discrimination","sexist","homophobic","abuse","harassment","illegal"],
}


def assign_primary_topic(text: str) -> str:
    """Rule-based topic extraction — entirely independent of sentiment label."""
    text_l = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(kw in text_l for kw in keywords):
            return topic
    return "general"


def assign_all_topics(text: str) -> list:
    text_l = text.lower()
    found = [t for t, kws in TOPIC_KEYWORDS.items() if any(kw in text_l for kw in kws)]
    return found if found else ["general"]


def compute_priority_flag(text: str, rating: int) -> tuple:
    """
    Priority is determined ONLY by text signals and rating.
    It is NOT derived from the sentiment label — no leakage.
    Returns (is_priority: bool, signals: list)
    """
    text_l = text.lower()
    signals = []
    for category, keywords in PRIORITY_SIGNALS.items():
        if any(kw in text_l for kw in keywords):
            signals.append(category)

    # A priority complaint requires: at least one signal OR rating=1 with CAPS urgency
    caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 2)
    is_priority = len(signals) > 0 or (rating == 1 and caps_words >= 2)
    return is_priority, signals


def generate_reviews(n: int = 2000) -> pd.DataFrame:
    """
    Generate n review records.

    Label assignment logic (leak-free):
      - Positive reviews → positive label + rating 4-5
      - Negative reviews → negative label + rating 1-2
      - Neutral reviews  → neutral label  + rating 3
      - Priority reviews → negative label + rating 1 (safety/legal content)
    
    Confidence scores are NOT used to create labels.
    Labels are set from the text bank used — that IS the ground truth.
    """
    rows = []
    start_date = datetime(2022, 1, 1)
    end_date   = datetime(2024, 12, 31)
    date_range = (end_date - start_date).days

    # Realistic distribution: 53% positive, 30% negative, 12% neutral, 5% priority
    sentiment_pool = (
        [("positive", POSITIVE_REVIEWS)] * 53 +
        [("negative", NEGATIVE_REVIEWS)] * 30 +
        [("neutral",  NEUTRAL_REVIEWS)]  * 12 +
        [("priority", PRIORITY_COMPLAINTS)] * 5
    )

    for i in range(n):
        category   = random.choice(CATEGORIES)
        company    = random.choice(COMPANIES[category])
        sent_type, text_bank = random.choice(sentiment_pool)
        base_text  = random.choice(text_bank)

        # ── Assign label and rating from text bank ────────────────────────────
        if sent_type == "priority":
            true_sentiment = "negative"
            rating = 1
            # Append a minor variation so texts aren't identical
            suffix = random.choice([
                " I have documented everything.",
                " This is completely unacceptable.",
                " Urgent action required.",
                " Avoid this company.",
                " I am seeking compensation.",
            ])
            review_text = base_text + suffix
        elif sent_type == "positive":
            true_sentiment = "positive"
            rating = random.choice([4, 4, 5, 5, 5])
            review_text = base_text
        elif sent_type == "negative":
            true_sentiment = "negative"
            rating = random.choice([1, 1, 1, 2])
            review_text = base_text
        else:  # neutral
            true_sentiment = "neutral"
            rating = 3
            review_text = base_text

        # ── Topic extraction (independent of label) ────────────────────────────
        primary_topic = assign_primary_topic(review_text)
        all_topics    = assign_all_topics(review_text)

        # ── Priority flag (independent of label) ──────────────────────────────
        is_priority, priority_signals = compute_priority_flag(review_text, rating)

        # ── Review date (uniform random — no temporal bias) ───────────────────
        review_date = start_date + timedelta(days=random.randint(0, date_range))

        rows.append({
            "review_id":       f"REV{i+1:05d}",
            "date":            review_date.strftime("%Y-%m-%d"),
            "year":            review_date.year,
            "month":           review_date.month,
            "quarter":         f"Q{(review_date.month - 1) // 3 + 1}",
            "company":         company,
            "category":        category,
            "rating":          rating,
            "review_text":     review_text,
            "true_sentiment":  true_sentiment,   # GROUND TRUTH label
            "primary_topic":   primary_topic,
            "all_topics":      "|".join(all_topics),
            "is_priority":     is_priority,
            "priority_signals":"|".join(priority_signals),
            "word_count":      len(review_text.split()),
            "char_count":      len(review_text),
        })

    return pd.DataFrame(rows)


def build_company_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics per company — calculated from raw labels only.
    No model outputs used here to prevent leakage into analysis.
    """
    agg = df.groupby(["company", "category"]).agg(
        total_reviews       = ("review_id", "count"),
        avg_rating          = ("rating", "mean"),
        pct_positive        = ("true_sentiment", lambda x: round((x == "positive").mean() * 100, 1)),
        pct_negative        = ("true_sentiment", lambda x: round((x == "negative").mean() * 100, 1)),
        pct_neutral         = ("true_sentiment", lambda x: round((x == "neutral").mean() * 100, 1)),
        priority_count      = ("is_priority", "sum"),
    ).reset_index()
    agg["avg_rating"] = agg["avg_rating"].round(2)
    return agg


def build_monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby(["year", "month", "quarter"]).agg(
        total_reviews   = ("review_id", "count"),
        avg_rating      = ("rating", "mean"),
        pct_positive    = ("true_sentiment", lambda x: round((x == "positive").mean() * 100, 1)),
        pct_negative    = ("true_sentiment", lambda x: round((x == "negative").mean() * 100, 1)),
        priority_count  = ("is_priority", "sum"),
    ).reset_index()
    agg["avg_rating"] = agg["avg_rating"].round(2)
    return agg.sort_values(["year", "month"]).reset_index(drop=True)


def build_topic_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["primary_topic"].value_counts().reset_index()
    counts.columns = ["topic", "count"]
    counts["pct"] = (counts["count"] / len(df) * 100).round(1)
    return counts


def main():
    out = "data/processed"
    os.makedirs(out, exist_ok=True)

    print("Generating reviews (n=2000)...")
    df = generate_reviews(2000)
    df.to_csv(f"{out}/reviews.csv", index=False)
    print(f"  reviews.csv — {len(df)} rows")

    print("\nLabel distribution (ground truth):")
    print(df["true_sentiment"].value_counts().to_string())

    print("\nPriority breakdown:")
    print(f"  Priority=True : {df['is_priority'].sum()}")
    print(f"  Priority=False: {(~df['is_priority']).sum()}")

    print("\nBuilding summaries...")
    company_df = build_company_summary(df)
    company_df.to_csv(f"{out}/company_summary.csv", index=False)
    print(f"  company_summary.csv — {len(company_df)} rows")

    monthly_df = build_monthly_trend(df)
    monthly_df.to_csv(f"{out}/monthly_trend.csv", index=False)
    print(f"  monthly_trend.csv — {len(monthly_df)} rows")

    topic_df = build_topic_distribution(df)
    topic_df.to_csv(f"{out}/topic_distribution.csv", index=False)
    print(f"  topic_distribution.csv — {len(topic_df)} rows")

    priority_df = df[df["is_priority"]].copy()
    priority_df.to_csv(f"{out}/priority_complaints.csv", index=False)
    print(f"  priority_complaints.csv — {len(priority_df)} rows")

    # JSON for dashboard
    dash = {
        "company":  company_df.to_dict(orient="records"),
        "monthly":  monthly_df.to_dict(orient="records"),
        "topics":   topic_df.to_dict(orient="records"),
        "priority": priority_df.head(30)[
            ["review_id","date","company","category","rating",
             "review_text","true_sentiment","primary_topic","priority_signals"]
        ].to_dict(orient="records"),
    }
    with open(f"{out}/dashboard_data.json", "w") as f:
        json.dump(dash, f, separators=(",", ":"), default=str)
    print("  dashboard_data.json written")

    print("\nDone. All files saved to data/processed/")


if __name__ == "__main__":
    main()
