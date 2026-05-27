import joblib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from preprocess import preprocess_iris, preprocess_titanic, get_numeric_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def train_and_evaluate(dataset: str):

    print(f"\n{'='*50}")
    print(f"  Training models on {dataset.upper()} dataset")
    print(f"{'='*50}")

    # load data
    if dataset == "iris":
        X_train, X_test, y_train, y_test, classes = preprocess_iris()
    else:
        X_train, X_test, y_train, y_test = preprocess_titanic()
        classes = ["Did not survive", "Survived"]

    # define models to compare
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "xgboost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss")
    }

    results = {}
    best_acc = 0
    best_model = None
    best_name = ""

    for name, clf in models.items():

        # build pipeline
        pipeline = Pipeline([
            ("preprocessor", get_numeric_pipeline()),
            ("classifier", clf)
        ])

        # train
        pipeline.fit(X_train, y_train)

        # predict
        preds = pipeline.predict(X_test)

        # evaluate
        acc = accuracy_score(y_test, preds)
        results[name] = {"accuracy": round(float(acc), 4)}

        print(f"\n--- {name.replace('_', ' ').title()} ---")
        print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"Classification Report:\n{classification_report(y_test, preds, target_names=classes)}")

        # track best model
        if acc > best_acc:
            best_acc = acc
            best_model = pipeline
            best_name = name

    # save best model
    os.makedirs("models", exist_ok=True)
    model_path = f"models/{dataset}_best_model.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n✅ Best model: {best_name} (accuracy: {best_acc*100:.2f}%)")
    print(f"✅ Saved to: {model_path}")

    # save results to json
    results["best_model"] = best_name
    results["best_accuracy"] = round(best_acc, 4)
    with open(f"models/{dataset}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return best_model


if __name__ == "__main__":
    train_and_evaluate("iris")
    train_and_evaluate("titanic")