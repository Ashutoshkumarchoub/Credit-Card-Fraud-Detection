
from pathlib import Path
import json
import time
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# ==========================================
# 2. LOAD PROCESSED DATA
# ==========================================

X_train = pd.read_csv(REPORT_DIR / "X_train.csv")
X_test = pd.read_csv(REPORT_DIR / "X_test.csv")

y_train = pd.read_csv(
    REPORT_DIR / "y_train.csv"
).iloc[:, 0].astype(int)

y_test = pd.read_csv(
    REPORT_DIR / "y_test.csv"
).iloc[:, 0].astype(int)

# Remove unused fields consistently
unused_columns = ["cc_num", "dob", "unix_time"]

X_train.drop(
    columns=[c for c in unused_columns if c in X_train.columns],
    inplace=True
)

X_test.drop(
    columns=[c for c in unused_columns if c in X_test.columns],
    inplace=True
)

# Align test features with training features
X_test = X_test.reindex(columns=X_train.columns)

print("Training dataset:", X_train.shape)
print("Testing dataset:", X_test.shape)

# ==========================================
# 3. LOAD VALIDATION THRESHOLD
# ==========================================

threshold_path = MODEL_DIR / "threshold.json"

if not threshold_path.exists():
    raise FileNotFoundError(
        "Run validation.py first to generate threshold.json"
    )

with open(threshold_path, "r") as file:
    threshold_info = json.load(file)

threshold = float(threshold_info["threshold"])

if not 0 < threshold < 1:
    raise ValueError("Invalid classification threshold.")

print("\nSelected validation threshold:", threshold)

# ==========================================
# 4. IDENTIFY FEATURE TYPES
# ==========================================

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumeric features:", numeric_features)
print("\nCategorical features:", categorical_features)

# ==========================================
# 5. CREATE PREPROCESSING PIPELINE
# ==========================================

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

# ==========================================
# 6. CREATE FINAL RANDOM FOREST MODEL
# ==========================================

random_forest = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", random_forest)
])

# ==========================================
# 7. TRAIN ON COMPLETE TRAINING DATA
# ==========================================

print("\nTraining final Random Forest model...")

start_time = time.time()

final_pipeline.fit(X_train, y_train)

training_time = time.time() - start_time

print("Training completed.")
print("Training time:", round(training_time, 2), "seconds")

# ==========================================
# 8. FINAL TEST PREDICTIONS
# ==========================================

print("\nEvaluating on untouched test dataset...")

probabilities = final_pipeline.predict_proba(X_test)[:, 1]

predictions = (
    probabilities >= threshold
).astype(int)

# ==========================================
# 9. CALCULATE PERFORMANCE METRICS
# ==========================================

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test, predictions, zero_division=0
)

recall = recall_score(
    y_test, predictions, zero_division=0
)

f1 = f1_score(
    y_test, predictions, zero_division=0
)

roc_auc = roc_auc_score(y_test, probabilities)

pr_auc = average_precision_score(
    y_test, probabilities
)

print("\n" + "=" * 55)
print("FINAL RANDOM FOREST MODEL RESULTS")
print("=" * 55)

print(f"Threshold:  {threshold:.4f}")
print(f"Accuracy:   {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-score:   {f1:.4f}")
print(f"ROC-AUC:    {roc_auc:.4f}")
print(f"PR-AUC:     {pr_auc:.4f}")

print("\nClassification Report:")

print(classification_report(
    y_test,
    predictions,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

# ==========================================
# 10. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, predictions)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Final Random Forest Confusion Matrix")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "final_confusion_matrix.png",
    dpi=150
)

plt.close()

# ==========================================
# 11. SAVE FINAL MODEL
# ==========================================

joblib.dump(
    final_pipeline,
    MODEL_DIR / "final_fraud_model.joblib"
)

# Save model evaluation information
metrics = {
    "model": "Random Forest",
    "threshold": threshold,
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(roc_auc),
    "pr_auc": float(pr_auc),
    "training_time_seconds": float(training_time),
    "training_records": int(len(X_train)),
    "testing_records": int(len(X_test))
}

with open(
    REPORT_DIR / "final_model_metrics.json",
    "w"
) as file:
    json.dump(metrics, file, indent=4)

print("\nFINAL MODEL SAVED SUCCESSFULLY")
print("Model:", MODEL_DIR / "final_fraud_model.joblib")
print("Metrics:", REPORT_DIR / "final_model_metrics.json")
print("Confusion matrix:", REPORT_DIR / "final_confusion_matrix.png")
