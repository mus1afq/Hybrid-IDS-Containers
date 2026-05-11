"""
evaluate_hybrid_synthetic.py  — v2 (Fixed)
==========================================
Uses the pre-paired (p_sys, p_net, y_true) probabilities from the
synthetic evaluation to compare Syscall-Only, Network-Only, and Hybrid.

Root-cause note: v1 re-ran network inference on unrelated Bot-IoT rows,
breaking label–probability correspondence. This version uses the
pre-generated hybrid_synthetic_probs_test.csv directly.

Outputs saved to outputs/metrics/ and outputs/plots/.

Run from project root:
    python3 src/hybrid/evaluation/evaluate_hybrid_synthetic.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import (
    HYBRID_PROBS_PATH, PRED_HYBRID_DIR, METRICS_DIR, PLOTS_DIR
)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)


# Hybrid decision function
def hybrid_decision(p_sys: float, p_net: float) -> int:
    """
    Late-fusion rule:
      p_sys >= 0.6                     → malicious
      p_net >= 0.5 AND p_sys >= 0.1   → malicious
      else                             → benign
    """
    if p_sys >= 0.6:
        return 1
    elif p_net >= 0.5 and p_sys >= 0.1:
        return 1
    return 0


# Load paired probability data
def load_data() -> pd.DataFrame:
    # Fusion is evaluated on pre-computed (p_sys, p_net, y_true) probabilities
    # rather than raw features because the two models were trained on different
    # datasets that cannot be row-aligned.
    print(f"Loading paired probabilities: {HYBRID_PROBS_PATH}")
    df = pd.read_csv(HYBRID_PROBS_PATH)
    df = df.dropna(subset=["p_sys", "p_net", "y_true"]).reset_index(drop=True)
    df["y_true"] = df["y_true"].astype(int)
    n0 = (df["y_true"] == 0).sum()
    n1 = (df["y_true"] == 1).sum()
    print(f"  {len(df)} samples — Benign: {n0}, Malicious: {n1} ({100*n1/len(df):.1f}%)")
    return df


# Metrics helper
def compute_metrics(y_true, y_pred, name: str) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "FPR":       round(fp / (fp + tn) if (fp + tn) else 0.0, 4),
        "FNR":       round(fn / (fn + tp) if (fn + tp) else 0.0, 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "_cm":     cm,
        "_y_pred": np.array(y_pred),
    }


# N-window simulation
N_WINDOWS = [5, 10, 15, 30]
N_SCALES  = {5: 0.25, 10: 0.50, 15: 0.75, 30: 1.00}


def run_n_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate partial observation windows by interpolating probabilities
    toward 0.5 (uncertainty) as the window shrinks.
    """
    rng = np.random.default_rng(42)
    records = []
    p_sys_full = df["p_sys"].values
    p_net_full = df["p_net"].values
    y_true     = df["y_true"].values

    for n in N_WINDOWS:
        scale     = N_SCALES[n]
        noise_std = (1 - scale) * 0.10
        # Simulate a shorter observation window by pulling probabilities toward 0.5.
        # A scale of 0.25 (N=5s) means the model has seen only a quarter of the full
        # capture, so its confidence is modelled as proportionally lower.
        p_sys_s = np.clip(0.5 + scale * (p_sys_full - 0.5) + rng.normal(0, noise_std, len(p_sys_full)), 0, 1)
        p_net_s = np.clip(0.5 + scale * (p_net_full - 0.5) + rng.normal(0, noise_std, len(p_net_full)), 0, 1)

        y_sys    = (p_sys_s >= 0.5).astype(int)
        y_net    = (p_net_s >= 0.5).astype(int)
        y_hybrid = np.array([hybrid_decision(s, m) for s, m in zip(p_sys_s, p_net_s)])

        records.append({
            "N_Window":   n,
            "Scale":      scale,
            "F1_Syscall": round(f1_score(y_true, y_sys,    zero_division=0), 4),
            "F1_Network": round(f1_score(y_true, y_net,    zero_division=0), 4),
            "F1_Hybrid":  round(f1_score(y_true, y_hybrid, zero_division=0), 4),
        })
        print(f"  N={n:>2} (scale={scale:.2f}): "
              f"syscall={records[-1]['F1_Syscall']:.4f}  "
              f"network={records[-1]['F1_Network']:.4f}  "
              f"hybrid={records[-1]['F1_Hybrid']:.4f}")
    return pd.DataFrame(records)


# Plots
def plot_confusion_matrix(cm: np.ndarray, path: Path):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Benign (0)", "Malicious (1)"],
                yticklabels=["Benign (0)", "Malicious (1)"],
                ax=ax, linewidths=0.5, annot_kws={"size": 14, "weight": "bold"})
    ax.set_title("Hybrid Model — Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {path}")


def plot_model_comparison(results: list, path: Path):
    models = [r["Model"] for r in results]
    accs   = [r["Accuracy"] for r in results]
    f1s    = [r["F1"] for r in results]
    x      = np.arange(len(models))
    w      = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, accs, w, label="Accuracy", color="#4C9BE8", edgecolor="white")
    b2 = ax.bar(x + w/2, f1s,  w, label="F1 Score", color="#2A9D8F", edgecolor="white")
    for bar, val in zip(b1, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar, val in zip(b2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, 1.15); ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison: Accuracy & F1 Score", fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10); ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {path}")


def plot_n_window(nw_df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(nw_df["N_Window"], nw_df["F1_Syscall"], "o-",  color="#4C9BE8", linewidth=2,   label="Syscall-Only")
    ax.plot(nw_df["N_Window"], nw_df["F1_Network"], "s-",  color="#F4A261", linewidth=2,   label="Network-Only")
    ax.plot(nw_df["N_Window"], nw_df["F1_Hybrid"],  "^-",  color="#2A9D8F", linewidth=2.5, label="Hybrid")
    ax.set_xticks(nw_df["N_Window"].tolist())
    ax.set_xlabel("Detection Window N (seconds)", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("F1 Score vs Detection Window (N-Window Analysis)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, 1.05); ax.legend(fontsize=10)
    ax.grid(linestyle="--", alpha=0.5)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {path}")


# Main
def main():
    print("\n" + "=" * 65)
    print("  HYBRID IDS — SYNTHETIC EVALUATION PIPELINE v2")
    print("=" * 65)

    # Load data
    df = load_data()
    p_sys  = df["p_sys"].values
    p_net  = df["p_net"].values
    y_true = df["y_true"].values

    # Compute predictions
    y_pred_sys    = (p_sys >= 0.5).astype(int)
    y_pred_net    = (p_net >= 0.5).astype(int)
    y_pred_hybrid = np.array([hybrid_decision(s, n) for s, n in zip(p_sys, p_net)])

    results = [
        compute_metrics(y_true, y_pred_sys,    "Syscall-Only"),
        compute_metrics(y_true, y_pred_net,    "Network-Only"),
        compute_metrics(y_true, y_pred_hybrid, "Hybrid"),
    ]

    # Print table
    print("\n" + "=" * 75)
    print("COMPARATIVE EVALUATION RESULTS")
    print("=" * 75)
    print(f"{'Model':<16} {'Accuracy':>8} {'Precision':>9} {'Recall':>7} {'F1':>7} {'FPR':>7} {'FNR':>7}")
    print("-" * 75)
    for r in results:
        print(f"{r['Model']:<16} {r['Accuracy']:>8.4f} {r['Precision']:>9.4f} "
              f"{r['Recall']:>7.4f} {r['F1']:>7.4f} {r['FPR']:>7.4f} {r['FNR']:>7.4f}")
    print("=" * 75)

    for r in results:
        print(f"  {r['Model']}: TN={r['TN']} FP={r['FP']} FN={r['FN']} TP={r['TP']}")

    # N-window
    print("\n[N-Window Simulation]")
    nw_df = run_n_window(df)

    # Plots
    print("\n[Generating Plots]")
    hybrid_r = next(r for r in results if r["Model"] == "Hybrid")
    plot_confusion_matrix(hybrid_r["_cm"],   PLOTS_DIR / "hybrid_confusion_matrix.png")
    plot_model_comparison(results,           PLOTS_DIR / "model_comparison.png")
    plot_n_window(nw_df,                     PLOTS_DIR / "n_window_analysis.png")
    # No ROC curve for the hybrid model because it uses a fixed rule, not a
    # continuous score. ROC requires a ranked output, which the fusion function
    # does not produce.

    # Save CSVs
    print("\n[Saving CSVs]")
    eval_rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    pd.DataFrame(eval_rows).to_csv(METRICS_DIR / "evaluation_table.csv", index=False)
    print(f"  Metrics → {METRICS_DIR / 'evaluation_table.csv'}")

    nw_df.to_csv(METRICS_DIR / "n_window_results.csv", index=False)
    print(f"  N-window → {METRICS_DIR / 'n_window_results.csv'}")

    out_df = df.copy()
    out_df["y_pred_syscall"] = next(r["_y_pred"] for r in results if r["Model"] == "Syscall-Only")
    out_df["y_pred_network"] = next(r["_y_pred"] for r in results if r["Model"] == "Network-Only")
    out_df["y_pred_hybrid"]  = hybrid_r["_y_pred"]
    out_df.to_csv(METRICS_DIR / "hybrid_results.csv", index=False)
    print(f"  Full results → {METRICS_DIR / 'hybrid_results.csv'}")

    print("\n[Complete] All outputs saved to outputs/metrics/ and outputs/plots/")


if __name__ == "__main__":
    main()
