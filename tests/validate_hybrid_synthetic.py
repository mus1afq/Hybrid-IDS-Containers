#!/usr/bin/env python3
"""
validate_hybrid_synthetic.py
============================
Validates the hybrid fusion logic against the 580-sample synthetic hybrid dataset.
This proves that the rule-based logic (tested iteratively in validate_hybrid.py)
actually performs as expected at scale, corroborating weak signals dynamically.
"""

import sys
import os
import pandas as pd
import numpy as np

# Adjust path to import the fusion logic
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hybrid.fusion.run_hybrid_fusion import hybrid_decision  # type: ignore

# Dataset path
DATA_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "hybrid", "hybrid_synthetic_probs_test.csv"
))

# Output paths for consistency with other synthetic tests
OUT_CSV = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "outputs", "predictions", "hybrid", "hybrid_synthetic_validation.csv"
))
OUT_PLOT = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "outputs", "plots", "hybrid_synthetic_confusion_matrix.png"
))

def main():
    print("=== Running Validation on SYNTHETIC Hybrid Dataset ===")
    print(f"Loading data from: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find {DATA_PATH}")
        sys.exit(1)
        
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["p_sys", "p_net", "y_true"]).reset_index(drop=True)
    df["y_true"] = df["y_true"].astype(int)

    total_samples = len(df)
    n_benign    = (df["y_true"] == 0).sum()
    n_malicious = (df["y_true"] == 1).sum()
    
    print("Running inference...")
    
    # The dataset holds pre-computed (p_sys, p_net) pairs rather than raw features
    # because CHIDS and Bot-IoT were collected independently and cannot be row-aligned.
    # Fusion is evaluated on probability scenarios that cover the decision space.
    y_true    = df["y_true"].values
    p_sys_raw = df["p_sys"].values
    p_net_raw = df["p_net"].values

    # Baselines: threshold each signal at 0.5 to show what either model alone achieves.
    y_pred_sys = (p_sys_raw >= 0.5).astype(int)
    y_pred_net = (p_net_raw >= 0.5).astype(int)

    y_pred_hybrid = np.array([hybrid_decision(s, n) for s, n in zip(p_sys_raw, p_net_raw)])
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, confusion_matrix
    
    acc_sys = accuracy_score(y_true, y_pred_sys)
    acc_net = accuracy_score(y_true, y_pred_net)
    acc_hybrid = accuracy_score(y_true, y_pred_hybrid)
    
    cm = confusion_matrix(y_true, y_pred_hybrid)
    
    print("\n=== Validation Results ===")
    
    # Spot-check 5 representative probability combinations to confirm the fusion
    # rule behaves as designed on real data from the synthetic dataset.
    case_a = df[(df['p_sys'] >= 0.6) & (df['p_net'] < 0.5)].iloc[0]
    case_b = df[(df['p_sys'] < 0.1) & (df['p_net'] >= 0.6)].iloc[0]
    case_c = df[(df['p_sys'] >= 0.1) & (df['p_sys'] < 0.6) & (df['p_net'] >= 0.5)].iloc[0]
    case_d = df[(df['p_sys'] >= 0.6) & (df['p_net'] >= 0.5)].iloc[0]
    case_e = df[(df['p_sys'] < 0.1) & (df['p_net'] < 0.5)].iloc[0]

    cases = [
        ("Case A (High Syscall, Low Network)", round(case_a['p_sys'], 4), round(case_a['p_net'], 4), 1, "Syscall Dominant (p_sys >= 0.6)"),
        ("Case B (Low Syscall, High Network)", round(case_b['p_sys'], 4), round(case_b['p_net'], 4), 0, "Insufficient Syscall (p_sys < 0.1)"),
        ("Case C (Moderate Syscall, High Network)", round(case_c['p_sys'], 4), round(case_c['p_net'], 4), 1, "Network Corroborates (p_net >= 0.5 & p_sys >= 0.1)"),
        ("Case D (High Syscall, High Network)", round(case_d['p_sys'], 4), round(case_d['p_net'], 4), 1, "Network Corroborates (p_net >= 0.5 & p_sys >= 0.1)"),
        ("Case E (Low Syscall, Low Network)", round(case_e['p_sys'], 4), round(case_e['p_net'], 4), 0, "Insufficient Syscall (p_sys < 0.1)")
    ]

    print("Hybrid Fusion Logic Checks (Synthetic Data)")
    for name, p_sys, p_net, expected, reason in cases:
        pred = hybrid_decision(p_sys, p_net)
        status = "PASS" if pred == expected else "FAIL"
        print(f"\n{name}")
        print(f"Inputs: p_sys={p_sys}, p_net={p_net}")
        print(f"Predicted Label: {pred} | Expected: {expected} [{status}]")
        print(f"Reason: {reason}")
    print("\n--------------------------------------------------------------")
    
    print(f"\nTotal samples: {total_samples}")
    print(f"Class counts: Benign (0): {n_benign}, Malicious (1): {n_malicious}")
    print(f"Syscall-Only Accuracy: {acc_sys:.4f}")
    print(f"Network-Only Accuracy: {acc_net:.4f}")
    print(f"Hybrid Fusion Accuracy: {acc_hybrid:.4f}")
    print(f"Confusion Matrix (Hybrid):")
    print(cm)
    
    # Save CSV
    df["pred_sys"] = y_pred_sys
    df["pred_net"] = y_pred_net
    df["pred_hybrid"] = y_pred_hybrid
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nPredictions saved to {OUT_CSV}")
    
    # Generate Confusion Matrix Plot
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign (0)", "Malicious (1)"],
                yticklabels=["Benign (0)", "Malicious (1)"],
                ax=ax, linewidths=0.5, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title("Hybrid Model — Synthetic Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PLOT), exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix plot saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()
