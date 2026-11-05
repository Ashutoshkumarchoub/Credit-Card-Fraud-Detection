
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    classification_report
)

# ------------------------------------------------
# 1. Paths and settings
# ------------------------------------------------
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 200_000

# ------------------------------------------------
# 2. Load training data only
# ------------------------------------------------
X = pd.read_csv(REPORT_DIR / "X_train.csv")
y = pd.read_csv(REPORT_DIR / "y_train.csv").iloc[:, 0].astype(int)

# Remove unused fields consistently
unused = ["cc_num", "dob", "unix_time"]

X.drop(
    columns=[c for c in unused if c in X.columns],
    inplace=True
)

print("Original training data:", X.shape)
print("Fraud distribution:")
print(y.value_counts())

# ------------------------------------------------
# 3. Take a stratified sample for development
# ------------------------------------------------
if len(X) > SAMPLE_SIZE:
    X, _, y, _ = train_test_split(
        X,
        y,
        train_size=SAMPLE_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

print("\nDevelopment sample:", X.shape)
print("Sample fraud distribution:")
print(y.value_counts())

# ------------------------------------------------
# 4. Split into training and validation data
# ------------------------------------------------
X_fit, X_val, y_fit, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print("\nModel training rows:", len(X_fit))
print("Validation rows:", len(X_val))

# ------------------------------------------------
# 5. Build preprocessing pipeline
# ------------------------------------------------
numeric_features = X_fit.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_fit.select_dtypes(
    exclude=["number"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        min_frequency=100
    ))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# ------------------------------------------------
# 6. Train Random Forest
# ------------------------------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])

print("\nTraining validation model...")
start = time.time()

model.fit(X_fit, y_fit)

print("Training time:", round(time.time() - start, 2), "seconds")

# ------------------------------------------------
# 7. Get validation probabilities
# ------------------------------------------------
probabilities = model.predict_proba(X_val)[:, 1]

# ------------------------------------------------
# 8. Find threshold maximizing validation F1
# ------------------------------------------------
precision, recall, thresholds = precision_recall_curve(
    y_val,
    probabilities
)

f1_scores = (
    2 * precision[:-1] * recall[:-1] /
    (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = np.argmax(f1_scores)
best_threshold = float(thresholds[best_index])

predictions = (
    probabilities >= best_threshold
).astype(int)

print("\nBEST VALIDATION THRESHOLD:", round(best_threshold, 4))

print("\nValidation Precision:",
      round(precision_score(y_val, predictions), 4))

print("Validation Recall:",
      round(recall_score(y_val, predictions), 4))

print("Validation F1-score:",
      round(f1_score(y_val, predictions), 4))

print("Validation PR-AUC:",
      round(average_precision_score(y_val, probabilities), 4))

print("\nValidation classification report:")
print(classification_report(
    y_val,
    predictions,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

# ------------------------------------------------
# 9. Save threshold information
# ------------------------------------------------
threshold_info = {
    "model": "Random Forest",
    "threshold": best_threshold,
    "selection_metric": "F1-score",
    "validation_sample_size": len(X_val),
    "random_state": RANDOM_STATE
}

with open(MODEL_DIR / "threshold.json", "w") as file:
    json.dump(threshold_info, file, indent=4)

# ------------------------------------------------
# 10. Save validation chart
# ------------------------------------------------
plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label="Precision-Recall Curve"
)

plt.scatter(
    recall[best_index],
    precision[best_index],
    label=f"Selected threshold: {best_threshold:.3f}"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Validation Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "validation_precision_recall.png",
    dpi=150
)

plt.show()

print("\nValidation completed successfully.")
print("Threshold saved to models/threshold.json")
