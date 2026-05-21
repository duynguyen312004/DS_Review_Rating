"""
predict.py - Script dự đoán sentiment cho review nhà hàng

Cách dùng:
    python src/predict.py "The food was amazing and the staff was very friendly."
"""

import sys
import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'naive_bayes_model.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')


def load_model():
    """Load model và vectorizer đã train."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("[ERROR] Chưa có model. Hãy chạy notebook 04_train_naive_bayes.ipynb trước!")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict_sentiment(review_text: str) -> str:
    """
    Dự đoán sentiment cho một review.
    
    Args:
        review_text: Nội dung review tiếng Anh
    Returns:
        'Positive', 'Neutral', hoặc 'Negative'
    """
    model, vectorizer = load_model()
    review_vec = vectorizer.transform([review_text])
    prediction = model.predict(review_vec)[0]
    return prediction


def main():
    if len(sys.argv) < 2:
        print("Cách dùng: python src/predict.py \"<nội dung review>\"")
        print("\nVí dụ:")
        print('  python src/predict.py "The food was amazing!"')
        sys.exit(1)

    review = sys.argv[1]
    print(f"\n📝 Review: {review}")
    sentiment = predict_sentiment(review)

    emoji_map = {"Positive": "✅", "Neutral": "😐", "Negative": "❌"}
    emoji = emoji_map.get(sentiment, "❓")
    print(f"{emoji}  Sentiment: {sentiment}\n")


if __name__ == "__main__":
    main()
