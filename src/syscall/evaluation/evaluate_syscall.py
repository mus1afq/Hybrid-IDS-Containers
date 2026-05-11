#!/usr/bin/env python3
"""
evaluate_syscall.py

Evaluates the syscall IDS against synthetic or captured trace features. Loads
feature CSVs, generates plots, and creates a metrics table.

Outputs saved to outputs/metrics/ and outputs/plots/.

Run from project root:
    python3 src/syscall/evaluation/evaluate_syscall.py
"""

import sys
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import (
    RUNTIME_SYS_DIR, SYSCALL_MODEL_PATH, SYSCALL_SCALER_PATH,
    CHIDS_DATASET_PATH, METRICS_DIR, PLOTS_DIR, PRED_SYSCALL_DIR,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Load model artefacts
print("[*] Loading model and scaler...")
model  = joblib.load(SYSCALL_MODEL_PATH)
scaler = joblib.load(SYSCALL_SCALER_PATH)
print(f"    Model type: {type(model).__name__}")

# Training schema
df_train    = pd.read_csv(CHIDS_DATASET_PATH)
df_train    = df_train.drop(columns=[c for c in ["id", "folder"] if c in df_train.columns])
FEATURE_COLS = [c for c in df_train.columns if c != "label"]

# Capture mode
mode_file    = RUNTIME_SYS_DIR / "capture_mode.txt"
capture_mode = mode_file.read_text().strip() if mode_file.exists() else "synthetic"

WINDOWS = ["N3", "N5", "N10", "N15", "full"]

# Inference helper
def infer(feat_path: Path):
    df         = pd.read_csv(feat_path)
    containers = df["container"].tolist() if "container" in df.columns else [f"c{i}" for i in range(len(df))]
    y_true     = df["label"].values.astype(int)
    X          = df[FEATURE_COLS].fillna(0).values
    X_sc       = scaler.transform(X)
    y_pred     = model.predict(X_sc).astype(int)
    y_prob     = model.predict_proba(X_sc)[:, 1]
    return containers, y_true, y_pred, y_prob

# Run inference on all windows
print("[*] Running inference across all time windows...")
all_results = {}
for tag in WINDOWS:
    path = RUNTIME_SYS_DIR / f"features_{tag}.csv"
    if not path.exists():
        print(f"    [!] Missing {path.name} — skipping")
        continue
    containers, y_true, y_pred, y_prob = infer(path)
    all_results[tag] = {"containers": containers, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}
    print(f"    {tag}: {len(y_true)} containers inferred")

# Training reference metrics
# Re-run the same 80/20 split used during training so we can compare runtime
# performance against held-out training metrics without requiring a separate log file.
_, Xv, _, yv = train_test_split(
    df_train.drop("label", axis=1), df_train["label"],
    test_size=0.2, stratify=df_train["label"], random_state=42,
)
Xv_sc   = scaler.transform(Xv)
yv_pred = model.predict(Xv_sc)
yv_prob = model.predict_proba(Xv_sc)[:, 1]
TRAIN_METRICS = {
    "Accuracy":  accuracy_score(yv, yv_pred),
    "Precision": precision_score(yv, yv_pred, zero_division=0),
    "Recall":    recall_score(yv, yv_pred, zero_division=0),
    "F1":        f1_score(yv, yv_pred, zero_division=0),
}

# Compute metrics
def compute_metrics(res):
    y_true, y_pred, y_prob = res["y_true"], res["y_pred"], res["y_prob"]
    cm      = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr  = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    roc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan")
    # Probability separation = mean P(malicious|malicious) − mean P(malicious|benign).
    # A high separation (close to 1) means the model assigns confidently different
    # scores to the two classes; it is a distribution-level confidence indicator.
    sep  = float(np.mean(y_prob[y_true == 1]) - np.mean(y_prob[y_true == 0]))
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "FPR": fpr, "FNR": fnr, "ROC-AUC": roc, "Separation": sep,
        "CM": cm, "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }

metrics_all = {tag: compute_metrics(res) for tag, res in all_results.items()}
FULL = "full"

# Plot 1: Confusion matrix
if FULL in metrics_all:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(metrics_all[FULL]["CM"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign\n(pred)", "Malicious\n(pred)"],
                yticklabels=["Benign\n(true)", "Malicious\n(true)"],
                annot_kws={"size": 14}, linewidths=0.5, ax=ax)
    ax.set_title("Syscall Model — Confusion Matrix (Full Trace)", fontsize=11, pad=12)
    ax.set_ylabel("True Label", fontsize=10); ax.set_xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "syscall_confusion_matrix.png", dpi=150)
    plt.close()

# Plot 2: Probability distributions per window
fig, axes = plt.subplots(1, len(all_results), figsize=(4 * len(all_results), 4), sharey=True)
if len(all_results) == 1:
    axes = [axes]
palette = {"Benign": "#2196F3", "Malicious": "#F44336"}
for ax, tag in zip(axes, WINDOWS):
    if tag not in all_results:
        continue
    res = all_results[tag]
    ben_p = res["y_prob"][res["y_true"] == 0]
    mal_p = res["y_prob"][res["y_true"] == 1]
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    if len(ben_p): ax.scatter(ben_p, np.ones_like(ben_p) * 0.5, s=120, color=palette["Benign"],    marker="o", label="Benign",    alpha=0.9, zorder=5)
    if len(mal_p): ax.scatter(mal_p, np.ones_like(mal_p) * 0.5, s=120, color=palette["Malicious"], marker="^", label="Malicious", alpha=0.9, zorder=5)
    ax.set_title("Full" if tag == "full" else f"N={tag[1:]}s", fontsize=10)
    ax.set_xlabel("P(Malicious)", fontsize=9); ax.set_xlim(-0.05, 1.05)
    ax.set_yticks([]); ax.set_ylim(0.3, 0.7)
b_patch = mpatches.Patch(color=palette["Benign"], label="Benign")
m_patch = mpatches.Patch(color=palette["Malicious"], label="Malicious")
fig.legend(handles=[b_patch, m_patch], loc="upper right", fontsize=9)
fig.suptitle("Syscall — Probability Distribution by Time Window", fontsize=11, y=1.02)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "syscall_prob_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# Plot 3: N-window ramp-up
ramp_tags   = [t for t in WINDOWS if t != "full" and t in metrics_all]
ramp_labels = [int(t[1:]) for t in ramp_tags]
CAPTURE_SECONDS = 15
if FULL in metrics_all:
    ramp_labels.append(CAPTURE_SECONDS)
    ramp_tags.append("full")
    tick_labels = [str(n) for n in ramp_labels[:-1]] + ["Full\n(15s)"]
else:
    tick_labels = [str(n) for n in ramp_labels]

f1_vals  = [metrics_all[t]["F1"]         for t in ramp_tags]
sep_vals = [metrics_all[t]["Separation"] for t in ramp_tags]
acc_vals = [metrics_all[t]["Accuracy"]   for t in ramp_tags]
rec_vals = [metrics_all[t]["Recall"]     for t in ramp_tags]
x = range(len(ramp_labels))

fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()
ax1.plot(x, f1_vals,  "o-",  color="#1976D2", linewidth=2, label="F1",       markersize=8)
ax1.plot(x, acc_vals, "s--", color="#388E3C", linewidth=2, label="Accuracy",  markersize=7)
ax1.plot(x, rec_vals, "^-.", color="#F57C00", linewidth=2, label="Recall",    markersize=7)
ax1.axhline(TRAIN_METRICS["F1"], color="#1976D2", linestyle=":", alpha=0.5,
            label=f"Train F1 ({TRAIN_METRICS['F1']:.2f})")
ax2.bar(x, sep_vals, alpha=0.2, color="#9C27B0", label="Prob. Separation")
ax2.set_ylabel("Prob. Separation", color="#9C27B0", fontsize=9)
ax2.tick_params(axis="y", labelcolor="#9C27B0"); ax2.set_ylim(0, 1.1)
ax1.set_xticks(x); ax1.set_xticklabels(tick_labels, fontsize=9)
ax1.set_xlabel("Detection Window N (seconds)", fontsize=10)
ax1.set_ylabel("Metric Value", fontsize=10); ax1.set_ylim(0, 1.05)
ax1.set_title("N-Window Ramp-Up Analysis: Performance vs Detection Window", fontsize=11)
l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc="lower right", fontsize=8)
plt.tight_layout()
fig.savefig(PLOTS_DIR / "syscall_n_window_rampup.png", dpi=150)
plt.close()

# Save metrics CSV
summary_rows = []
for tag in WINDOWS:
    if tag not in metrics_all:
        continue
    m = metrics_all[tag]
    label = "Full" if tag == "full" else f"{tag[1:]}s"
    summary_rows.append({
        "Window": label, "Accuracy": round(m["Accuracy"], 4),
        "Precision": round(m["Precision"], 4), "Recall": round(m["Recall"], 4),
        "F1": round(m["F1"], 4), "FPR": round(m["FPR"], 4), "FNR": round(m["FNR"], 4),
        "ROC-AUC": round(m["ROC-AUC"], 4) if not np.isnan(m["ROC-AUC"]) else "N/A",
        "Sep. Gap": round(m["Separation"], 4),
        "TP": m["TP"], "TN": m["TN"], "FP": m["FP"], "FN": m["FN"],
    })
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(METRICS_DIR / "syscall_metrics_summary.csv", index=False)

# Save per-container breakdown
if FULL in all_results:
    res = all_results[FULL]
    breakdown = []
    for name, yt, yp, prob in zip(res["containers"], res["y_true"], res["y_pred"], res["y_prob"]):
        breakdown.append({
            "Container": name, "True_Label": "Malicious" if yt else "Benign",
            "Predicted": "Malicious" if yp else "Benign",
            "P(Malicious)": round(float(prob), 4), "Correct": "✓" if yt == yp else "✗",
            "FP": int(yt == 0 and yp == 1), "FN": int(yt == 1 and yp == 0),
        })
    pd.DataFrame(breakdown).to_csv(METRICS_DIR / "syscall_per_container.csv", index=False)

# Save per-window predictions
for tag, res in all_results.items():
    pd.DataFrame({
        "container": res["containers"], "y_true": res["y_true"],
        "y_pred": res["y_pred"], "p_malicious": res["y_prob"],
    }).to_csv(PRED_SYSCALL_DIR / f"predictions_{tag}.csv", index=False)

# Compute research question answers
full_m = metrics_all.get(FULL, {})
fm_f1, fm_acc  = full_m.get("F1", 0), full_m.get("Accuracy", 0)
fm_fpr, fm_fnr = full_m.get("FPR", 0), full_m.get("FNR", 0)
fm_sep         = full_m.get("Separation", 0)
fm_fp, fm_fn   = full_m.get("FP", 0), full_m.get("FN", 0)
train_f1 = TRAIN_METRICS["F1"]
drop     = train_f1 - fm_f1

best_n = None
for tag in ["N3", "N5", "N10", "N15"]:
    if tag in metrics_all and metrics_all[tag]["F1"] >= 0.7 and metrics_all[tag]["Separation"] >= 0.4:
        best_n = int(tag[1:]); break
if best_n is None:
    best_n = 15





# Console summary
print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)
print(df_summary[["Window", "Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"]].to_string(index=False))
print(f"\nTraining reference F1: {train_f1:.4f}")
print(f"Full-trace F1 (val):   {fm_f1:.4f}  (delta: {drop:+.4f})")
print(f"Recommended N:         {best_n}s")
print(f"False positives:       {fm_fp}")
print(f"False negatives:       {fm_fn}")
print("=" * 60)
print(f"\nOutputs:")
print(f"  {METRICS_DIR / 'syscall_metrics_summary.csv'}")
print(f"  {PLOTS_DIR / 'syscall_confusion_matrix.png'}")
print(f"  {PLOTS_DIR / 'syscall_prob_distributions.png'}")
print(f"  {PLOTS_DIR / 'syscall_n_window_rampup.png'}")
