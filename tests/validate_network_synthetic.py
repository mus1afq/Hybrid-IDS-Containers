import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


def main():
    # Resolve the project root from this file's location so the script can be
    # run from any working directory, not just /Users/mshk/Desktop/FYP.
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "raw" / "network" / "botiot_synthetic_test_1000.csv"
    out_csv   = base_dir / "outputs" / "predictions" / "network" / "network_synthetic_validation.csv"
    out_plot  = base_dir / "outputs" / "plots" / "network_synthetic_confusion_matrix.png"
    
    print("=== Running Validation on SYNTHETIC Network Dataset ===")

    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        return

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    print("Loading models...")
    model = joblib.load(f"{base_dir}/models/network/best_network_model.pkl")

    # The network model is a Random Forest trained on raw flow counts.
    # It does not need a scaler — RF is not sensitive to feature magnitude.
    feat_cols   = [c for c in df.columns if c != 'label_bin']
    features    = df[feat_cols]
    true_labels = df['label_bin']

    print("Running inference...")
    probs = model.predict_proba(features)[:, 1]
    preds = (probs >= 0.5).astype(int)

    total_samples = len(df)
    class_counts = true_labels.value_counts().to_dict()
    acc = accuracy_score(true_labels, preds)
    cm = confusion_matrix(true_labels, preds, labels=[0, 1])

    df['p_net'] = probs
    df['pred'] = preds
    avg_prob_benign    = df[df['label_bin'] == 0]['p_net'].mean() if 0 in class_counts else 0.0
    avg_prob_malicious = df[df['label_bin'] == 1]['p_net'].mean() if 1 in class_counts else 0.0

    print("\n=== Validation Results ===")
    print(f"Total samples: {total_samples}")
    print(f"Class counts: Benign (0): {class_counts.get(0, 0)}, Malicious (1): {class_counts.get(1, 0)}")
    print(f"Overall Accuracy: {acc:.4f}")
    # Average p_net by class shows whether the model separates the two distributions,
    # not just whether the final accuracy looks good.
    print(f"Average p_net for True Benign: {avg_prob_benign:.4f}")
    print(f"Average p_net for True Malicious: {avg_prob_malicious:.4f}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df = df[['label_bin', 'pred', 'p_net']].copy()
    out_df.to_csv(out_csv, index=False)
    print(f"\nPredictions saved to {out_csv}")

    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Network Model Confusion Matrix (Synthetic)')
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()
    print(f"Confusion matrix plot saved to {out_plot}")

    # Generate and save ROC Curve
    fpr, tpr, _ = roc_curve(true_labels, probs)
    roc_auc = auc(fpr, tpr)
    
    out_roc_plot = f"{base_dir}/outputs/plots/network_synthetic_roc_curve.png"
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Network Model ROC Curve (Synthetic)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_roc_plot)
    plt.close()
    print(f"ROC curve plot saved to {out_roc_plot}")

if __name__ == '__main__':
    main()