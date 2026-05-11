"""
evaluate_network.py
Evaluates the trained Random Forest on the Bot-IoT synthetic test set.
Outputs confusion matrix, ROC curve, and metrics to outputs/.

Run from project root:
    python3 src/network/evaluation/evaluate_network.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import (
    BOTIOT_TEST_PATH, NETWORK_MODEL_PATH,
    PLOTS_DIR, METRICS_DIR, PRED_NETWORK_DIR,
)

import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

# Standard 0.5 threshold; the model was trained with class_weight="balanced"
# so the raw probabilities are already calibrated towards the centre.
THRESHOLD  = 0.5
TARGET_COL = "label_bin"

# Load
print(f"Dataset: {BOTIOT_TEST_PATH}")
print(f"Model:   {NETWORK_MODEL_PATH}")

df    = pd.read_csv(BOTIOT_TEST_PATH)
X     = df.drop(columns=[TARGET_COL])
y     = df[TARGET_COL].astype(int).to_numpy()
model = joblib.load(NETWORK_MODEL_PATH)

# The RF model scores samples directly without scaling.
y_prob = model.predict_proba(X)[:, 1]
y_pred = (y_prob >= THRESHOLD).astype(int)

# Metrics
acc  = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred, zero_division=0)
rec  = recall_score(y, y_pred, zero_division=0)
f1   = f1_score(y, y_pred, zero_division=0)
auc  = roc_auc_score(y, y_prob)
cm   = confusion_matrix(y, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
fpr_val = fp / (fp + tn) if (fp + tn) else 0.0
fnr_val = fn / (fn + tp) if (fn + tp) else 0.0

print("\n" + "=" * 55)
print("NETWORK MODEL EVALUATION")
print("=" * 55)
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print(f"FPR={fpr_val:.4f}  FNR={fnr_val:.4f}")
print(f"TN={tn} FP={fp} FN={fn} TP={tp}")
print("=" * 55)

# Save metrics text
metrics_path = METRICS_DIR / "network_synthetic_metrics.txt"
with open(metrics_path, "w") as f:
    f.write(f"NETWORK MODEL EVALUATION\nDataset: {BOTIOT_TEST_PATH}\nModel: {NETWORK_MODEL_PATH}\nThreshold: {THRESHOLD}\n\n")
    f.write(f"Accuracy: {acc:.6f}\nPrecision: {prec:.6f}\nRecall: {rec:.6f}\nF1: {f1:.6f}\nROC-AUC: {auc:.6f}\n\n")
    f.write(f"TN={tn} FP={fp} FN={fn} TP={tp}\nFPR={fpr_val:.6f} FNR={fnr_val:.6f}\n")
print(f"Metrics → {metrics_path}")

# Confusion matrix plot
fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_title("Network Model — Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white")
plt.tight_layout()
cm_path = PLOTS_DIR / "network_confusion_matrix.png"
fig.savefig(cm_path, dpi=200)
plt.close(fig)
print(f"Plot → {cm_path}")

# ROC curve
# ROC curve shows the trade-off between true positive and false positive rates
# across all possible thresholds, not just at 0.5. The network model is a good
# candidate for ROC because it has a smooth probability output.
fpr_c, tpr_c, _ = roc_curve(y, y_prob)
fig2, ax2 = plt.subplots()
ax2.plot(fpr_c, tpr_c, label=f"AUC = {auc:.4f}")
ax2.plot([0, 1], [0, 1], "--", color="grey")
ax2.set_title("Network Model — ROC Curve")
ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR")
ax2.legend()
plt.tight_layout()
roc_path = PLOTS_DIR / "network_roc_curve.png"
fig2.savefig(roc_path, dpi=200)
plt.close(fig2)
print(f"Plot → {roc_path}")

# Save full predictions
pred_df = df.copy()
pred_df["pred_label"]     = y_pred
pred_df["prob_malicious"] = y_prob
pred_path = PRED_NETWORK_DIR / "botiot_predictions.csv"
pred_df.to_csv(pred_path, index=False)
print(f"Predictions → {pred_path}")
