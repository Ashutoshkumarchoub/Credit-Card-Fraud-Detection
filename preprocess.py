
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data")
OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load the original datasets
train = pd.read_csv(DATA_DIR / "fraudTrain.csv")
test = pd.read_csv(DATA_DIR / "fraudTest.csv")

print("Original training shape:", train.shape)
print("Original testing shape:", test.shape)

# 1. Check target column
target = "is_fraud"

if target not in train.columns or target not in test.columns:
    raise ValueError("The is_fraud target column must exist in both datasets.")

# 2. Check missing target values
if train[target].isna().any() or test[target].isna().any():
    raise ValueError("Missing target labels found. Investigate before training.")

# 3. Remove duplicate rows from training data only
duplicates = train.duplicated().sum()
print("\nDuplicate training rows:", duplicates)

train = train.drop_duplicates().copy()

# 4. Separate features and target
X_train = train.drop(columns=[target]).copy()
y_train = train[target].astype(int).copy()

X_test = test.drop(columns=[target]).copy()
y_test = test[target].astype(int).copy()

# 5. Remove identifier columns that are not useful as direct predictors
identifier_columns = [
    col for col in ["Unnamed: 0", "trans_num"] if col in X_train.columns
]

X_train.drop(columns=identifier_columns, inplace=True)
X_test.drop(columns=identifier_columns, inplace=True)

# 6. Create transaction date/time features if the timestamp is available
for frame in (X_train, X_test):
    if "trans_date_trans_time" in frame.columns:
        timestamp = pd.to_datetime(
            frame["trans_date_trans_time"], errors="coerce"
        )

        frame["transaction_hour"] = timestamp.dt.hour
        frame["transaction_day"] = timestamp.dt.day
        frame["transaction_month"] = timestamp.dt.month
        frame["transaction_day_of_week"] = timestamp.dt.dayofweek

        frame.drop(columns=["trans_date_trans_time"], inplace=True)

# 7. Remove personally identifying or high-cardinality raw fields
# We will not use names, street addresses, or exact coordinates in this
# initial baseline. More advanced geographic features can be engineered later.
columns_to_drop = [
    col for col in ["first", "last", "street", "city", "zip",
                    "lat", "long", "merch_lat", "merch_long"]
    if col in X_train.columns
]

X_train.drop(columns=columns_to_drop, inplace=True)
X_test.drop(columns=columns_to_drop, inplace=True)

# Ensure training and test feature columns match
X_test = X_test.reindex(columns=X_train.columns)

# Save processed datasets for inspection
X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)

y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

print("\nPreprocessing completed.")
print("Features used:", X_train.columns.tolist())
print("Training features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nMissing values in training features:")
print(X_train.isna().sum().sort_values(ascending=False).head(10))

print("\nSaved processed files to:", OUTPUT_DIR.resolve())
