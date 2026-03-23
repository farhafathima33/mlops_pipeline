import pandas as pd
import pickle
import json
import os
import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

THRESHOLD = 0.90

def evaluate():
    # Load test data
    test_df = pd.read_csv("data/processed/test.csv")

    X_test = test_df.drop("target", axis=1)
    y_test = test_df["target"]

    # Load model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predictions
    preds = model.predict(X_test)

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, preds)
    }

    # Save metrics
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("📊 Metrics:", metrics)

    # 🚨 Validation check
    if metrics["accuracy"] < THRESHOLD:
        print("❌ Model below threshold! Failing pipeline.")
        sys.exit(1)

    print("✅ Model passed validation!")

if __name__ == "__main__":
    evaluate()