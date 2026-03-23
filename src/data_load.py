import pandas as pd
from sklearn.datasets import load_breast_cancer
import os

def load_data():
    data = load_breast_cancer()

    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    os.makedirs("data/raw", exist_ok=True)

    df.to_csv("data/raw/data.csv", index=False)

    print("✅ Data saved!")

if __name__ == "__main__":
    load_data()