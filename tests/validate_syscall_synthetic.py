import pandas as pd
import joblib
import os
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Resolve the project root from this file's location so the script is portable.
    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / "raw" / "syscall" / "chids_synthetic_100.csv"
    out_csv   = base_dir / "outputs" / "predictions" / "syscall" / "syscall_synthetic_validation.csv"
    out_plot  = base_dir / "outputs" / "plots" / "syscall_synthetic_confusion_matrix.png"
    
    print("=== Running Validation on SYNTHETIC Syscall Dataset ===")

    if not os.path.exists(data_path):
        print(f"Error: Dataset {data_path} not found.")
        return

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    print("Loading models...")
    model = joblib.load(f"{base_dir}/models/syscall/best_model.pkl")
    scaler = joblib.load(f"{base_dir}/models/syscall/scaler.pkl")

    # The scaler must match the one fitted during training; using a new scaler
    # would shift the feature distributions and invalidate the model outputs.
    feat_cols = [c for c in df.columns if c != 'label']
    features  = scaler.transform(df[feat_cols])
    true_labels = df['label']

    print("Running inference...")
    probs = model.predict_proba(features)[:, 1]
    preds = model.predict(features)

    total_samples = len(df)
    class_counts = true_labels.value_counts().to_dict()
    acc = accuracy_score(true_labels, preds)
    cm = confusion_matrix(true_labels, preds, labels=[0, 1])

    df['p_sys'] = probs
    df['pred'] = preds
    # Average p_sys by class shows whether the model is separating the two
    # distributions, not just hitting a threshold correctly on the test set.
    avg_prob_benign    = df[df['label'] == 0]['p_sys'].mean() if 0 in class_counts else 0.0
    avg_prob_malicious = df[df['label'] == 1]['p_sys'].mean() if 1 in class_counts else 0.0

    print("\n=== Validation Results ===")
    print(f"Total samples: {total_samples}")
    print(f"Class counts: Benign (0): {class_counts.get(0, 0)}, Malicious (1): {class_counts.get(1, 0)}")
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Average p_sys for True Benign: {avg_prob_benign:.4f}")
    print(f"Average p_sys for True Malicious: {avg_prob_malicious:.4f}")
    print(f"Confusion Matrix:\n{cm}")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df = df[['label', 'pred', 'p_sys']].copy()
    out_df.to_csv(out_csv, index=False)
    print(f"\nPredictions saved to {out_csv}")

    os.makedirs(os.path.dirname(out_plot), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Malicious'], 
                yticklabels=['Benign', 'Malicious'])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Syscall Model Confusion Matrix (Synthetic)')
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()
    print(f"Confusion matrix plot saved to {out_plot}")

if __name__ == '__main__':
    main()