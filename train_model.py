
from pathlib import Path
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score
)

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
SAMPLE_SIZE = 200_000

# --------------------------------------------------
# 2. Load the processed datasets
# --------------------------------------------------
X_train = pd.read_csv(REPORT_DIR / "X_train.csv")
X_test = pd.read_csv(REPORT_DIR / "X_test.csv")

y_train = pd.read_csv(REPORT_DIR / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(REPORT_DIR / "y_test.csv").iloc[:, 0]

# Remove identifiers and raw timestamp fields.
# These are not used as direct predictors in this baseline.
unwanted = ["cc_num", "dob", "unix_time"]

X_train.drop(
    columns=[c for c in unwanted if c in X_train.columns],
    inplace=True
)
X_test.drop(
    columns=[c for c in unwanted if c in X_test.columns],
    inplace=True
)

# Ensure identical feature order
X_test = X_test.reindex(columns=X_train.columns)

# Remove missing labels, if any
valid_train = y_train.notna()
X_train = X_train.loc[valid_train].reset_index(drop=True)
y_train = y_train.loc[valid_train].astype(int).reset_index(drop=True)

valid_test = y_test.notna()
X_test = X_test.loc[valid_test].reset_index(drop=True)
y_test = y_test.loc[valid_test].astype(int).reset_index(drop=True)

print("Full training shape:", X_train.shape)
print("Testing shape:", X_test.shape)
print("\nTraining class counts:")
print(y_train.value_counts())

# --------------------------------------------------
# 3. Sample the training data for the first experiment
# --------------------------------------------------
if len(X_train) > SAMPLE_SIZE:
    sample_index = (
        y_train.groupby(y_train)
        .sample(
            n=None,
            frac=SAMPLE_SIZE / len(y_train),
            random_state=RANDOM_STATE
        )
        .index
    )

    # Ensure both classes are represented and keep a
    # reproducible stratified sample.
    sample_index = (
        X_train.assign(_target=y_train)
        .groupby("_target", group_keys=False)
        .sample(
            n=None,
            frac=SAMPLE_SIZE / len(X_train),
            random_state=RANDOM_STATE
        )
        .index
    )

    X_fit = X_train.loc[sample_index].reset_index(drop=True)
    y_fit = y_train.loc[sample_index].reset_index(drop=True)
else:
    X_fit = X_train.copy()
    y_fit = y_train.copy()

print("\nRows used for initial training:", len(X_fit))
print("Sample class counts:")
print(y_fit.value_counts())

# --------------------------------------------------
# 4. Identify numeric and categorical features
# --------------------------------------------------
numeric_features = X_fit.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_fit.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nNumeric features:", numeric_features)
print("\nCategorical features:", categorical_features)

# Learn preprocessing from training data only.
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

# --------------------------------------------------
# 5. Define models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=200,
        solver="liblinear",
        random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
}

results = []
trained_models = {}

# --------------------------------------------------
# 6. Train and evaluate both models
# --------------------------------------------------
for name, classifier in models.items():
    print(f"\n{'=' * 55}")
    print("Training:", name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])

    start = time.time()
    pipeline.fit(X_fit, y_fit)
    training_seconds = time.time() - start

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

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
    pr_auc = average_precision_score(y_test, probabilities)

    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, predictions)
    print("Confusion matrix:")
    print(cm)

    # Save confusion matrix for this model
    plt.figure(figsize=(6, 5))
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
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    plt.savefig(
        REPORT_DIR / f"{safe_name}_confusion_matrix.png",
        dpi=150
    )
    plt.close()

    results.append({
        "Model": name,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Training seconds": training_seconds
    })

    trained_models[name] = pipeline

    # Save each fitted pipeline, including preprocessing
    joblib.dump(
        pipeline,
        MODEL_DIR / f"{safe_name}.joblib"
    )

# --------------------------------------------------
# 7. Compare model results
# --------------------------------------------------
results_df = pd.DataFrame(results).sort_values(
    by="PR-AUC", ascending=False
)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

print("\nModels saved in:", MODEL_DIR.resolve())
print("Evaluation reports saved in:", REPORT_DIR.resolve())
