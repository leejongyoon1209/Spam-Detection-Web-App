from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "spam_classifier.joblib"
app = Flask(__name__)
model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    try:
        payload = request.get_json(silent=True) or request.form
        message = str(payload.get("message", "")).strip()
        if not message:
            return jsonify({"error": "메시지를 입력해 주세요."}), 400
        if model is None:
            return jsonify({"error": "학습된 모델을 찾을 수 없습니다. 먼저 python train_model.py를 실행해 주세요."}), 500
        label = str(model.predict([message])[0])
        probabilities = model.predict_proba([message])[0]
        classes = list(model.classes_)
        spam_probability = float(probabilities[classes.index("spam")])
        return jsonify({
            "label": "Spam" if label == "spam" else "Ham",
            "spam_probability": round(spam_probability * 100, 2),
        })
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "예측 중 문제가 발생했습니다. 잠시 후 다시 시도해 주세요."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
