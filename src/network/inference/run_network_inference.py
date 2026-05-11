"""
run_network_inference.py
Loads runtime network features and runs the trained Random Forest model.
Outputs a predictions CSV to outputs/predictions/network/.

Run from project root:
    python3 src/network/inference/run_network_inference.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import (
    RUNTIME_NET_DIR, NETWORK_MODEL_PATH, PRED_NETWORK_DIR
)

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load aligned runtime network features
features_path = RUNTIME_NET_DIR / "runtime_network_aligned.csv"
print(f"Loading features: {features_path}")
df = pd.read_csv(features_path)

X = df.drop(columns=["file", "label"])
y = df["label"]

# Load model
print(f"Loading model: {NETWORK_MODEL_PATH}")
model = joblib.load(NETWORK_MODEL_PATH)

# Random Forest does not need scaled inputs, so no scaler is applied here.
# The syscall inference script does apply a scaler because Logistic Regression
# is sensitive to feature magnitude.
prob = model.predict_proba(X)[:, 1]
pred = (prob >= 0.5).astype(int)

df["p_net"] = prob
df["pred"]  = pred

# Save predictions
out_path = PRED_NETWORK_DIR / "runtime_network_predictions.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved predictions → {out_path}")

# Print metrics
print("\n=== Runtime Network Evaluation ===")
print(classification_report(y, pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y, pred))
