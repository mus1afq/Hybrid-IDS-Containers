import pandas as pd
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Resolve the project root from this file's location so the script is portable.
_ROOT     = Path(__file__).resolve().parents[1]
# Load predictions written by run_hybrid_fusion.py rather than re-running inference,
# so the test reflects exactly what the full pipeline produced.
DATA_PATH = _ROOT / "outputs" / "predictions" / "hybrid" / "runtime_hybrid_predictions.csv"
PLOTS_DIR = str(_ROOT / "outputs" / "plots")

def plot_cm(cm, title, filename):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign (0)", "Malicious (1)"],
                yticklabels=["Benign (0)", "Malicious (1)"],
                ax=ax, linewidths=0.5, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_roc(y_true, p_sys, p_net, filename):
    fig, ax = plt.subplots(figsize=(6, 5))
    # The hybrid model does not have its own ROC curve because it uses a fixed
    # rule rather than a continuous score. Syscall and network curves are shown
    # separately so the individual model signals can be compared.
    fpr_sys, tpr_sys, _ = roc_curve(y_true, p_sys)
    auc_sys = auc(fpr_sys, tpr_sys)
    
    fpr_net, tpr_net, _ = roc_curve(y_true, p_net)
    auc_net = auc(fpr_net, tpr_net)
    
    ax.plot(fpr_sys, tpr_sys, label=f"Syscall (AUC = {auc_sys:.2f})", color="#4C9BE8", linewidth=2)
    ax.plot(fpr_net, tpr_net, label=f"Network (AUC = {auc_net:.2f})", color="#F4A261", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random", alpha=0.5)
    
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("Runtime End-to-End ROC Curves", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)

def main():
    # Predictions are pre-computed by the full runtime pipeline.
    # Re-running inference here would bypass the fusion step and lose context.
    print("Loading existing runtime predictions...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: Could not find {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    
    y_true = df['label']
    y_sys = df['pred_syscall']
    y_net = df['pred_network']
    y_hyb = df['pred_hybrid']
    
    p_sys = df['p_sys']
    p_net = df['p_net']
    
    print("\n=== End-to-End Runtime Evaluation ===")
    print(f"{'Container/Sample':<15} | {'True Label':<10} | {'p_sys':<8} | {'p_net':<8} | {'Pred Hybrid':<11} | {'Status':<10}")
    print("-" * 75)
    
    correct_count = 0
    for idx, row in df.iterrows():
        is_correct = int(row['label']) == int(row['pred_hybrid'])
        status = "Correct" if is_correct else "Incorrect"
        if is_correct: correct_count += 1
        print(f"{row['base']:<15} | {int(row['label']):<10} | {row['p_sys']:<8.4f} | {row['p_net']:<8.4f} | {int(row['pred_hybrid']):<11} | {status:<10}")

    print("\n=== End-to-End Metrics ===")
    def print_metrics(name, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"{name:<15} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
    print_metrics("Syscall-Only", y_sys)
    print_metrics("Network-Only", y_net)
    print_metrics("Hybrid Fusion", y_hyb)
    
    print("\n=== Confusion Matrices ===")
    cm_sys = confusion_matrix(y_true, y_sys, labels=[0, 1])
    cm_net = confusion_matrix(y_true, y_net, labels=[0, 1])
    cm_hyb = confusion_matrix(y_true, y_hyb, labels=[0, 1])
    
    print("Syscall CM:\n", cm_sys)
    print("Network CM:\n", cm_net)
    print("Hybrid CM:\n", cm_hyb)
    
    print("\n[Generating Plots...]")
    plot_cm(cm_sys, "Syscall Model — Runtime Confusion Matrix", "runtime_syscall_cm.png")
    plot_cm(cm_net, "Network Model — Runtime Confusion Matrix", "runtime_network_cm.png")
    plot_cm(cm_hyb, "Hybrid Model — Runtime Confusion Matrix", "runtime_hybrid_cm.png")
    plot_roc(y_true, p_sys, p_net, "runtime_roc_curve.png")
    print("Plots saved to outputs/plots/\n  - runtime_syscall_cm.png\n  - runtime_network_cm.png\n  - runtime_hybrid_cm.png\n  - runtime_roc_curve.png")
    
    # Only 6 containers were used (3 benign, 3 malicious) because the goal is a
    # sanity check on the full end-to-end pipeline, not a statistically significant
    # benchmark. The main performance claims use the 580-sample synthetic dataset.
    accuracy = (correct_count / len(df)) * 100
    print("\n=== Summary ===")
    print(f"The hybrid system evaluated {len(df)} live runtime container scenarios (3 benign, 3 malicious).")
    print(f"Outcome: {correct_count} out of {len(df)} correctly classified (Accuracy: {accuracy:.1f}%).")
    print(f"Of the {len(df)} cases, there were {cm_hyb[0][1]} false positives and {cm_hyb[1][0]} false negatives.")

if __name__ == "__main__":
    main()
