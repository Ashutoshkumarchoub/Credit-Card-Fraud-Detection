from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")

train = pd.read_csv(DATA_DIR / "fraudTrain.csv")
test = pd.read_csv(DATA_DIR / "fraudTest.csv")

print("Training shape:", train.shape)
print("Testing shape:", test.shape)

print("\nColumns:")
print(train.columns.tolist())

print("\nFirst five records:")
print(train.head())

print("\nFraud distribution:")
print(train["is_fraud"].value_counts())