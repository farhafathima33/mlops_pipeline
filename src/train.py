import pandas as pd
import yaml
import pickle
import os

from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import mlflow
import mlflow.sklearn

def train():
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    # Load data
    train_df = pd.read_csv("data/processed/train.csv")

    X = train_df.drop("target", axis=1)
    y = train_df["target"]

    models = {
        "xgb": XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=params["random_state"]
        ),
        "ada": AdaBoostClassifier(
            n_estimators=params["n_estimators"],
            random_state=params["random_state"]
        )
    }

    best_model = None
    best_acc = 0
    
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("mlops_experiment")

    for name, model in models.items():
        with mlflow.start_run(run_name=name):

            model.fit(X, y)
            preds = model.predict(X)
            acc = accuracy_score(y, preds)

            mlflow.log_param("model", name)
            mlflow.log_param("n_estimators", params["n_estimators"])
            mlflow.log_metric("accuracy", acc)

            if acc > best_acc:
                best_acc = acc
                best_model = model

    # Save best model
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    # Log model
    mlflow.sklearn.log_model(best_model, "model")

    print("✅ Training complete!")

if __name__ == "__main__":
    train()