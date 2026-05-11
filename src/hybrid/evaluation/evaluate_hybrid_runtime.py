"""
evaluate_hybrid_runtime.py
Evaluates the hybrid IDS on the runtime paired predictions.
Reads from outputs/predictions/hybrid/runtime_hybrid_predictions.csv.

Run from project root:
    python3 src/hybrid/evaluation/evaluate_hybrid_runtime.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import PRED_HYBRID_DIR, METRICS_DIR

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)

csv_path = PRED_HYBRID_DIR / "runtime_hybrid_predictions.csv"
print(f"Loading: {csv_path}")
# Read the predictions written by run_hybrid_fusion.py so this script evaluates
# the complete pipeline output, not a partial re-run of inference only.
df = pd.read_csv(csv_path)

y_true = df["label"]


def compute_metrics(name: str, y_pred) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr  = fp / (fp + tn) if (fp + tn) else 0.0
    fnr  = fn / (fn + tp) if (fn + tp) else 0.0
    # FPR and FNR are included because accuracy alone is misleading on a
    # small, balanced 6-container sample.
    return {"Model": name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1, "FPR": fpr, "FNR": fnr, "CM": cm}


results = [
    compute_metrics("Syscall (Runtime)", df["pred_syscall"]),
    compute_metrics("Network (Runtime)", df["pred_network"]),
    compute_metrics("Hybrid  (Runtime)", df["pred_hybrid"]),
]

print("\n=== Runtime Hybrid Evaluation ===")
print(f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'FPR':>6} {'FNR':>6}")
print("-" * 60)
rows = []
for r in results:
    print(f"{r['Model']:<22} {r['Accuracy']:>6.4f} {r['Precision']:>6.4f} "
          f"{r['Recall']:>6.4f} {r['F1']:>6.4f} {r['FPR']:>6.4f} {r['FNR']:>6.4f}")
    print(f"  Confusion Matrix:\n{r['CM']}\n")
    rows.append({k: v for k, v in r.items() if k != "CM"})

# Save metrics table
out_path = METRICS_DIR / "runtime_hybrid_metrics.csv"
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"Metrics saved → {out_path}")
