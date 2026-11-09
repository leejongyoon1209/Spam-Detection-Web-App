from pathlib import Path
import json
import zipfile

import joblib
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
MIRROR_URL = "https://raw.githubusercontent.com/debopriyobhowmick/sms_spam_classification/master/SMSSpamCollection"
ZIP_PATH = DATA_DIR / "sms_spam_collection.zip"
DATA_PATH = DATA_DIR / "SMSSpamCollection"

# Small, manually curated supplement for realistic payment and billing phishing.
# The comparison message used in the experiment is intentionally not included.
SUPPLEMENTAL_SPAM_MESSAGES = [
    "Your payment was declined. Confirm your card details now to prevent account suspension.",
    "We were unable to verify your latest payment. Sign in to the billing portal to update your details.",
    "Payment alert: your invoice is overdue. Review and settle the balance at the secure payment page.",
    "Your card payment needs verification. Complete the billing check now to keep your service active.",
    "A problem was found with your payment method. Confirm your billing information immediately.",
    "Your subscription payment failed. Update your payment details today to avoid cancellation.",
    "Billing notice: we could not collect your scheduled payment. Verify your account to retry it.",
    "Your recent transaction is on hold. Confirm the payment information to release it.",
    "Action required: your invoice payment is pending. Open the payment review page before it expires.",
    "We need to recheck your card for the outstanding invoice. Submit your billing details to continue.",
    "Payment verification required. Use the secure account page to confirm your billing information.",
    "Your account billing could not be completed. Update the saved card before access is restricted.",
    "A failed payment may interrupt your service. Confirm your payment method now.",
    "Your electronic invoice is ready but payment details are invalid. Review them immediately.",
    "The bank could not authorize your payment. Verify your billing profile to avoid another failure.",
    "Your payment account needs an urgent security check before the next charge can be processed.",
]


def download_dataset():
    if DATA_PATH.exists():
        return
    print("Downloading UCI SMS Spam Collection...")
    try:
        response = requests.get(DATA_URL, timeout=20)
        response.raise_for_status()
        ZIP_PATH.write_bytes(response.content)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            member = next(name for name in archive.namelist() if name.endswith("SMSSpamCollection"))
            with archive.open(member) as source, DATA_PATH.open("wb") as target:
                target.write(source.read())
    except requests.RequestException:
        print("UCI download unavailable; trying public mirror...")
        response = requests.get(MIRROR_URL, timeout=60)
        response.raise_for_status()
        DATA_PATH.write_bytes(response.content)


def main():
    download_dataset()
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"], encoding="utf-8")
    supplement = pd.DataFrame({
        "label": ["spam"] * len(SUPPLEMENTAL_SPAM_MESSAGES),
        "message": SUPPLEMENTAL_SPAM_MESSAGES,
    })
    df = pd.concat([df, supplement], ignore_index=True)
    x_train, x_test, y_train, y_test = train_test_split(
        df["message"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, strip_accents="unicode", ngram_range=(1, 2), min_df=1)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "dataset_size": int(len(df)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision_spam": round(float(precision_score(y_test, predictions, pos_label="spam")), 4),
        "recall_spam": round(float(recall_score(y_test, predictions, pos_label="spam")), 4),
        "f1_spam": round(float(f1_score(y_test, predictions, pos_label="spam")), 4),
        "classification_report": classification_report(y_test, predictions, digits=4),
    }
    joblib.dump(model, MODEL_DIR / "spam_classifier.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n" + metrics["classification_report"])


if __name__ == "__main__":
    main()
