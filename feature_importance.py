
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

MODEL_PATH = Path("models/random_forest.joblib")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

# Load the fitted Random Forest pipeline
model = joblib.load(MODEL_PATH)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

# Get the names of features after preprocessing
feature_names = preprocessor.get_feature_names_out()
importances = classifier.feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values("Importance", ascending=False)

print("\nTop 20 most important features:")
print(importance_df.head(20).to_string(index=False))

# Save all feature importance values
importance_df.to_csv(
    REPORT_DIR / "feature_importance.csv",
    index=False
)

# Plot the top 20 features
top_features = importance_df.head(20).sort_values("Importance")

plt.figure(figsize=(11, 8))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Random Forest Feature Importance")
plt.ylabel("Feature")
plt.title("Top 20 Features for Credit Card Fraud Detection")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "feature_importance.png",
    dpi=150
)
plt.show()

print("\nFeature importance saved to reports/feature_importance.csv")
print("Chart saved to reports/feature_importance.png")
