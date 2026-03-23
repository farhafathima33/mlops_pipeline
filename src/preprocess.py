import pandas as pd
from sklearn.model_selection import train_test_split
import os

def preprocess():
    # Load data
    df = pd.read_csv("data/raw/data.csv")

    # Remove duplicates
    df = df.drop_duplicates()

    # Handle missing values
    df = df.fillna(df.mean())

    # Train-test split
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # Create folder
    os.makedirs("data/processed", exist_ok=True)

    # Save files
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)

    print("✅ Preprocessing done!")

if __name__ == "__main__":
    preprocess()