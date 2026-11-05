
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

# Load the saved Random Forest pipeline
model = joblib.load(MODEL_DIR / "random_forest.joblib")

X_test = pd.read_csv(REPORT_DIR / "X_test.csv")
y_test = pd.read_csv(REPORT_DIR / "y_test.csv").iloc[:, 0].astype(int)

# Apply the same removal of unused columns as during training
unused = ["cc_num", "dob", "unix_time"]
X_test = X_test.drop(
    columns=[c for c in unused if c in X_test.columns]
)

# Use exactly the same feature order as the fitted model
X_test = X_test.reindex(
    columns=model.named_steps["preprocessor"].feature_names_in_
)

# Predict fraud probabilities
probabilities = model.predict_proba(X_test)[:, 1]

print("Average precision (PR-AUC):",
      round(average_precision_score(y_test, probabilities), 4))

# Examine several decision thresholds
thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

rows = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    rows.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "Flagged transactions": int(predictions.sum())
    })

results = pd.DataFrame(rows)

print("\nThreshold comparison:")
print(results.round(4).to_string(index=False))

results.to_csv(
    REPORT_DIR / "threshold_comparison.csv",
    index=False
)

# Plot the precision-recall curve
precision, recall, pr_thresholds = precision_recall_curve(
    y_test, probabilities
)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Random Forest Precision-Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    REPORT_DIR / "precision_recall_curve.png",
    dpi=150
)
plt.show()

# Inspect the default threshold of 0.5
default_predictions = (probabilities >= 0.5).astype(int)

print("\nClassification report at threshold 0.5:")
print(classification_report(
    y_test,
    default_predictions,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

print("\nConfusion matrix at threshold 0.5:")
print(confusion_matrix(y_test, default_predictions))

print("\nThreshold analysis saved to:", REPORT_DIR.resolve())
