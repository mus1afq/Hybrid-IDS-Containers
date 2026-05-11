"""
run_syscall_inference.py
Loads runtime syscall features and runs the trained Logistic Regression model.
Outputs a predictions CSV to outputs/predictions/syscall/.

Run from project root:
    python3 src/syscall/inference/run_syscall_inference.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import (
    RUNTIME_SYS_DIR, SYSCALL_MODEL_PATH, SYSCALL_SCALER_PATH, PRED_SYSCALL_DIR
)

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load aligned runtime features
features_path = RUNTIME_SYS_DIR / "runtime_features_aligned.csv"
print(f"Loading features: {features_path}")
df = pd.read_csv(features_path)

# Select only the columns that match the training feature schema.
# Filtering by 'evt_' prefix and 'total_events' avoids accidentally
# including metadata columns (e.g. 'file', 'label', 'container') as features.
feature_cols = [c for c in df.columns if c.startswith("evt_") or c == "total_events"]
X = df[feature_cols]
y = df["label"]

# Load model & scaler
print(f"Loading model:  {SYSCALL_MODEL_PATH}")
print(f"Loading scaler: {SYSCALL_SCALER_PATH}")
model  = joblib.load(SYSCALL_MODEL_PATH)
scaler = joblib.load(SYSCALL_SCALER_PATH)

X_scaled = scaler.transform(X)

# Predict
y_pred = model.predict(X_scaled)
y_prob = model.predict_proba(X_scaled)[:, 1]

# Save predictions
out = df[["file", "label"]].copy()
out["pred"]  = y_pred
out["p_sys"] = y_prob

out_path = PRED_SYSCALL_DIR / "runtime_syscall_predictions.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved predictions → {out_path}")

# Print metrics
print("\n=== Runtime Syscall Evaluation ===")
print(classification_report(y, y_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))
