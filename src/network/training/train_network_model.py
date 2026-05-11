"""
train_network_model.py
Trains a Random Forest classifier on the Bot-IoT network flow dataset.
Best model is saved to models/network/.

Run from project root:
    python3 src/network/training/train_network_model.py
"""

import sys
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import NETWORK_DATASET_PATH, NETWORK_MODEL_PATH

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

# Step 1: Data preparation
print("=" * 65)
print("STEP 1 — Data Preparation")
print("=" * 65)

df = pd.read_csv(NETWORK_DATASET_PATH)
print(f"\nRaw dataset shape: {df.shape}")

assert "label_bin" in df.columns, "ERROR: 'label_bin' column not found!"

# scikit-learn cannot handle boolean dtype natively in all contexts; cast to int.
bool_cols = df.select_dtypes(include="bool").columns.tolist()
if bool_cols:
    df[bool_cols] = df[bool_cols].astype(int)

obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
if "label_bin" in obj_cols:
    obj_cols.remove("label_bin")
if obj_cols:
    df.drop(columns=obj_cols, inplace=True)

y = df["label_bin"].astype(int)
X = df.drop(columns=["label_bin"])

# Bot-IoT contains many duplicate flow records from repeated attack bursts.
# Deduplication avoids the model learning repetition-specific patterns rather
# than genuine attack signatures, which would inflate test accuracy.
before = len(df)
combined = X.copy()
combined["label_bin"] = y
combined.drop_duplicates(inplace=True)
X = combined.drop(columns=["label_bin"])
y = combined["label_bin"]
print(f"Rows after deduplication: {len(X)}  (removed {before - len(X)} duplicates)")

# Constant columns carry no information; remove any feature with only one unique value.
low_var = [c for c in X.columns if X[c].nunique() <= 1]
if low_var:
    X.drop(columns=low_var, inplace=True)

X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
print(f"Final shape: X={X.shape}, y={y.shape}")

vc = y.value_counts().sort_index()
for lbl, cnt in vc.items():
    print(f"  {lbl} ({'benign' if lbl == 0 else 'malicious'}): {cnt:>6,}  ({100*cnt/len(y):.1f}%)")

# Step 2: Split
print("\n" + "=" * 65)
print("STEP 2 — Train/Test Split (80/20 stratified, seed=42)")
print("=" * 65)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

# Step 3: Train
print("\n" + "=" * 65)
print("STEP 3 — Training Random Forest (n=200, balanced)")
print("=" * 65)

# Random Forest was chosen because Bot-IoT has high dimensionality and many
# correlated flow statistics. RF handles both well without feature scaling,
# and n=200 trees gives stable probability estimates for the fusion layer.
rf = RandomForestClassifier(n_estimators=200, max_depth=None,
                             class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
print("Training complete.")

#Step 4: Evaluate
print("\n" + "=" * 65)
print("STEP 4 — Evaluation")
print("=" * 65)

y_pred  = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)
roc  = roc_auc_score(y_test, y_proba)
cm   = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

print(f"{'Accuracy':<25} {acc:>10.6f}")
print(f"{'Precision':<25} {prec:>10.6f}")
print(f"{'Recall':<25} {rec:>10.6f}")
print(f"{'F1 Score':<25} {f1:>10.6f}")
print(f"{'ROC-AUC':<25} {roc:>10.6f}")
print(f"{'FPR':<25} {fpr:>10.6f}")
print(f"{'FNR':<25} {fnr:>10.6f}")
print(f"\nConfusion Matrix:  TN={tn}  FP={fp}  FN={fn}  TP={tp}")

train_acc = accuracy_score(y_train, rf.predict(X_train))
gap = train_acc - acc
print(f"\nTrain Accuracy: {train_acc:.6f}")
print(f"Test  Accuracy: {acc:.6f}")
# A gap > 5% is a reasonable heuristic for overfitting with an uncapped-depth RF.
print(f"Overfitting gap: {gap:.4f}  ({'⚠ > 0.05' if gap > 0.05 else '✓ acceptable'})")

# Step 5: Feature importance
feat_imp = pd.Series(rf.feature_importances_, index=X.columns)
top20 = feat_imp.sort_values(ascending=False).head(20)
print("\n" + "=" * 65)
print("STEP 5 — Top-20 Feature Importances")
print("=" * 65)
for rank, (feat, imp_val) in enumerate(top20.items(), 1):
    print(f"  {rank:<3} {feat:<25} {imp_val:.6f}")

# Step 6: Save
with open(NETWORK_MODEL_PATH, "wb") as f:
    pickle.dump(rf, f)
print(f"\nModel saved → {NETWORK_MODEL_PATH}")
