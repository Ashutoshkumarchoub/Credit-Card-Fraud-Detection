
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure paths
DATA_DIR = Path("data")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load datasets
train = pd.read_csv(DATA_DIR / "fraudTrain.csv")
test = pd.read_csv(DATA_DIR / "fraudTest.csv")

sns.set_theme(style="whitegrid")

print("=" * 50)
print("DATASET OVERVIEW")
print("=" * 50)

print("Training shape:", train.shape)
print("Testing shape:", test.shape)

print("\nColumn names:")
print(train.columns.tolist())

print("\nData types:")
print(train.dtypes)

print("\nMissing values:")
print(train.isnull().sum().sort_values(ascending=False).head(15))

print("\nDuplicate rows:", train.duplicated().sum())

print("\nStatistical summary:")
print(train.describe(include="all").T)

# Fraud distribution
fraud_counts = train["is_fraud"].value_counts().sort_index()

print("\nFraud counts:")
print(fraud_counts)

print("\nFraud percentages:")
print((fraud_counts / len(train) * 100).round(4))

# Plot 1: Class distribution
plt.figure(figsize=(7, 5))
sns.barplot(
    x=["Legitimate", "Fraudulent"],
    y=[fraud_counts.get(0, 0), fraud_counts.get(1, 0)]
)
plt.title("Legitimate vs Fraudulent Transactions")
plt.xlabel("Transaction Type")
plt.ylabel("Number of Transactions")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
plt.show()

# Plot 2: Transaction amount distribution
plt.figure(figsize=(9, 5))
sns.histplot(
    data=train,
    x="amt",
    hue="is_fraud",
    bins=60,
    element="step",
    stat="density",
    common_norm=False
)
plt.title("Transaction Amount Distribution by Fraud Status")
plt.xlabel("Transaction Amount")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "transaction_amount_distribution.png", dpi=150)
plt.show()

# Plot 3: Fraud rate by transaction category
category_fraud = (
    train.groupby("category")["is_fraud"]
    .agg(["mean", "count"])
    .sort_values("mean", ascending=False)
)

print("\nCategories with highest observed fraud rates:")
print(category_fraud.head(10))

plt.figure(figsize=(11, 6))
sns.barplot(
    data=category_fraud.reset_index(),
    x="mean",
    y="category"
)
plt.title("Observed Fraud Rate by Transaction Category")
plt.xlabel("Fraud Rate")
plt.ylabel("Category")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fraud_rate_by_category.png", dpi=150)
plt.show()

print("\nEDA completed. Charts saved in the reports folder.")
