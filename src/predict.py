import argparse
import re
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "naive_bayes_model.joblib"
VECTORIZER_PATH = PROJECT_ROOT / "models" / "tfidf_vectorizer.joblib"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_sentiment(review):
    model, vectorizer = load_artifacts()
    review_clean = clean_text(review)
    review_vector = vectorizer.transform([review_clean])
    prediction = model.predict(review_vector)[0]

    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = dict(zip(model.classes_, model.predict_proba(review_vector)[0]))

    return review_clean, prediction, probabilities


def main():
    parser = argparse.ArgumentParser(
        description="Predict restaurant review sentiment using the trained TF-IDF + Naive Bayes model."
    )
    parser.add_argument("review", nargs="+", help="Review text to classify.")
    args = parser.parse_args()

    review = " ".join(args.review)
    review_clean, prediction, probabilities = predict_sentiment(review)

    print("Input review:", review)
    print("Clean review:", review_clean)
    print("Predicted sentiment:", prediction)

    if probabilities:
        print("Class probabilities:")
        for label, probability in sorted(probabilities.items()):
            print(f"  {label}: {probability:.4f}")


if __name__ == "__main__":
    main()
