import pandas as pd
import joblib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

print("Loading data...")
df = pd.read_csv(_ROOT / "data" / "raw" / "syscall" / "chids_dataset_clean.csv")

# Get 15 benign and 15 malicious samples
b_sample = df[df['label'] == 0].sample(15, random_state=42)
m_sample = df[df['label'] == 1].sample(15, random_state=42)

# Load the saved model so inference can run without retraining on the full CHIDS dataset.
model  = joblib.load(_ROOT / "models" / "syscall" / "best_model.pkl")
scaler = joblib.load(_ROOT / "models" / "syscall" / "scaler.pkl")

feat_cols = [c for c in df.columns if c != 'label']
b_features = scaler.transform(b_sample[feat_cols])
m_features = scaler.transform(m_sample[feat_cols])

b_probs = model.predict_proba(b_features)[:,1]
m_probs = model.predict_proba(m_features)[:,1]

b_pred = model.predict(b_features)
m_pred = model.predict(m_features)

b_correct = (b_pred == 0).sum()
m_correct = (m_pred == 1).sum()
# 15 samples per class is sufficient to confirm basic model behaviour;
# statistical significance is covered by evaluate_syscall.py on the full split.

print("\n=== Validation Results ===")
print("Benign Syscall Classification")
print(f"Sample Identifier: Index {b_sample.index[0]}")
print(f"True Label: 0 (Benign)")
print(f"Predicted Label: {b_pred}")
print(f"Malicious Probability Score (p_sys): \n{b_probs}\n")

print("\nMalicious Syscall Classification")
print(f"Sample Identifier: Index {m_sample.index[0]}")
print(f"True Label: 1 (Malicious)")
print(f"Predicted Label: {m_pred}")
print(f"Malicious Probability Score (p_sys): \n{m_probs}\n")



print("\n=== Syscall Validation Summary ===")
print(f"Benign: {b_correct}/15 correct")
print(f"Malicious: {m_correct}/15 correct")