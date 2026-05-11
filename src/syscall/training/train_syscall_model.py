"""
train_syscall_model.py
Trains multiple classifiers on the CHIDS syscall dataset.
Best model (by F1) is saved to models/syscall/.

Run from project root:
    python3 src/syscall/training/train_syscall_model.py
"""

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import (
    CHIDS_DATASET_PATH, SYSCALL_MODEL_PATH, SYSCALL_SCALER_PATH
)

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TEST_SIZE    = 0.2  # 80/20 split; stratified to preserve the benign/malicious ratio

# Load data
print(f"Loading dataset: {CHIDS_DATASET_PATH}")
df = pd.read_csv(CHIDS_DATASET_PATH)
# 'id' and 'folder' are CHIDS metadata columns that are not predictive features.
df = df.drop(columns=[c for c in ["id", "folder"] if c in df.columns], errors="ignore")

X = df.drop("label", axis=1)
y = df["label"]
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{y.value_counts()}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# Fit the scaler on training data only — scaler.pkl is saved alongside the model
# and must be applied to live runtime features before inference.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Model definitions
# (name, model, use_scaled)
# class_weight="balanced" is used throughout because CHIDS has more benign samples
# than malicious; without it, classifiers tend to bias towards the majority class.
# Tree-based models (RF, ET, GB) operate on raw counts; linear models (LR, SVM)
# need scaled inputs because they are sensitive to feature magnitude.
models = [
    ("Logistic Regression", LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000), True),
    ("Random Forest",       RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), False),
    ("SVM (RBF)",           SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_STATE), True),
    ("Gradient Boosting",   GradientBoostingClassifier(random_state=RANDOM_STATE), False),
    ("Extra Trees",         ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1), False),
]

try:
    from xgboost import XGBClassifier
    models.append(("XGBoost", XGBClassifier(
        scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
        eval_metric="logloss", random_state=RANDOM_STATE, use_label_encoder=False), False))
except ImportError:
    print("XGBoost not installed, skipping.")

try:
    from lightgbm import LGBMClassifier
    models.append(("LightGBM", LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1), False))
except ImportError:
    print("LightGBM not installed, skipping.")

# Train & evaluate
results = []
print("\nTraining and evaluating models...")
print("-" * 80)
print(f"{'Model':<25} | {'Acc':<8} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'ROC-AUC':<8} | {'Time(s)':<8}")
print("-" * 80)

# F1 is used as the selection criterion rather than accuracy because the CHIDS
# dataset is class-imbalanced; accuracy would unfairly favour the majority class.
best_f1, best_name, best_obj = -1, "", None

for name, model, use_scaled in models:
    X_t = X_train_scaled if use_scaled else X_train
    X_v = X_test_scaled  if use_scaled else X_test
    t0  = time.time()
    model.fit(X_t, y_train)
    y_pred = model.predict(X_v)
    y_prob = model.predict_proba(X_v)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_v)
    dur = time.time() - t0

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    roc  = roc_auc_score(y_test, y_prob)
    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1, "ROC-AUC": roc, "Time (s)": dur})
    print(f"{name:<25} | {acc:.4f}   | {prec:.4f}   | {rec:.4f}   | {f1:.4f}   | {roc:.4f}   | {dur:.4f}")

    if f1 > best_f1:
        best_f1, best_name, best_obj = f1, name, model

print("-" * 80)
print(f"\nBest Model: {best_name}  (F1 = {best_f1:.4f})")

# Feature importance
if hasattr(best_obj, "feature_importances_"):
    imp = best_obj.feature_importances_
    top = np.argsort(imp)[::-1][:20]
    print(f"\nTop 20 Features ({best_name}):")
    for i in top:
        print(f"  {X.columns[i]}: {imp[i]:.4f}")
elif best_name == "Logistic Regression":
    imp = np.abs(best_obj.coef_[0])
    top = np.argsort(imp)[::-1][:20]
    print(f"\nTop 20 Coefficients ({best_name}):")
    for i in top:
        print(f"  {X.columns[i]}: {imp[i]:.4f}")

# Save
joblib.dump(best_obj, SYSCALL_MODEL_PATH)
joblib.dump(scaler,   SYSCALL_SCALER_PATH)
print(f"\nModel saved  → {SYSCALL_MODEL_PATH}")
print(f"Scaler saved → {SYSCALL_SCALER_PATH}")
