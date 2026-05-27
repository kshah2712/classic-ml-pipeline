import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import numpy as np
from flask import Flask, request, jsonify

# create Flask app
app = Flask(__name__)

# load both models when server starts
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
iris_model = joblib.load(os.path.join(BASE_DIR, "models/iris_best_model.pkl"))
titanic_model = joblib.load(os.path.join(BASE_DIR, "models/titanic_best_model.pkl"))

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


# ─── ROUTES ───────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Classic ML Pipeline API",
        "endpoints": {
            "health":  "GET  /health",
            "iris":    "POST /predict/iris",
            "titanic": "POST /predict/titanic"
        }
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": ["iris", "titanic"]})


@app.route("/predict/iris", methods=["POST"])
def predict_iris():
    try:
        data = request.get_json()

        # validate input
        required = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        features = np.array([[
            data["sepal_length"],
            data["sepal_width"],
            data["petal_length"],
            data["petal_width"]
        ]])

        pred = iris_model.predict(features)[0]
        proba = iris_model.predict_proba(features)[0]

        return jsonify({
            "prediction": IRIS_CLASSES[pred],
            "confidence": round(float(proba[pred]), 4),
            "all_probabilities": {
                "setosa":     round(float(proba[0]), 4),
                "versicolor": round(float(proba[1]), 4),
                "virginica":  round(float(proba[2]), 4)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/titanic", methods=["POST"])
def predict_titanic():
    try:
        data = request.get_json()

        # validate input
        required = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
        for field in required:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        features = np.array([[
            data["pclass"],
            data["sex"],
            data["age"],
            data["sibsp"],
            data["parch"],
            data["fare"],
            data["embarked"]
        ]])

        pred = titanic_model.predict(features)[0]
        proba = titanic_model.predict_proba(features)[0]

        return jsonify({
            "survived": bool(pred),
            "result": "Survived 🟢" if pred == 1 else "Did not survive 🔴",
            "survival_probability": round(float(proba[1]), 4),
            "input_received": {
                "pclass": data["pclass"],
                "sex": "female" if data["sex"] == 1 else "male",
                "age": data["age"]
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)