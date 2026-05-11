import pandas as pd
import joblib
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parents[1]

print("Loading data...")
df = pd.read_csv(_ROOT / "data" / "raw" / "network" / "network_dataset_cleaned.csv")

# Get 15 benign and 15 malicious samples
b_sample = df[df['label_bin'] == 0].sample(15, random_state=42)
m_sample = df[df['label_bin'] == 1].sample(15, random_state=42)

# Load the saved model so inference can run without retraining on the full Bot-IoT dataset.
model = joblib.load(_ROOT / "models" / "network" / "best_network_model.pkl")

feat_cols = [c for c in df.columns if c != 'label_bin']
b_features = b_sample[feat_cols]
m_features = m_sample[feat_cols]
# No scaler is applied — the network model is a Random Forest, which is
# invariant to feature scaling.

b_probs = model.predict_proba(b_features)[:,1]
m_probs = model.predict_proba(m_features)[:,1]

b_pred = (b_probs >= 0.5).astype(int)
m_pred = (m_probs >= 0.5).astype(int)

b_correct = (b_pred == 0).sum()
m_correct = (m_pred == 1).sum()

print("\n=== Validation Results ===")
print("Benign Network Classification")
print(f"Sample Identifier: Index {b_sample.index[0]}")
print(f"True Label: 0 (Benign)")
print(f"Predicted Label: {b_pred}")
print(f"Malicious Probability Score (p_net): \n{b_probs}\n")

print("\nMalicious Network Classification")
print(f"Sample Identifier: Index {m_sample.index[0]}")
print(f"True Label: 1 (Malicious)")
print(f"Predicted Label: {m_pred}")
print(f"Malicious Probability Score (p_net): \n{m_probs}\n")

print("\n=== Network Validation Summary ===")
print(f"Benign: {b_correct}/15 correct")
print(f"Malicious: {m_correct}/15 correct")
