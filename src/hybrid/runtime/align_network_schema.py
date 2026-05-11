"""
align_network_schema.py
Aligns the raw runtime network features (extracted from pcap files)
to the exact column schema used to train the network model.

Outputs:
  data/runtime/network/runtime_network_aligned.csv

Run from project root:
    python3 src/hybrid/runtime/align_network_schema.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import (
    RUNTIME_NET_DIR, NETWORK_DATASET_PATH
)

import pandas as pd

# Load raw runtime features
raw_path = RUNTIME_NET_DIR / "runtime_network_raw.csv"
print(f"Loading raw features: {raw_path}")
raw = pd.read_csv(raw_path)

# Load training schema
print(f"Loading training schema: {NETWORK_DATASET_PATH}")
train = pd.read_csv(NETWORK_DATASET_PATH)

label_col = "label" if "label" in train.columns else "label_bin"
train_cols = [c for c in train.columns if c != label_col]

# Add any missing columns as zeros
# pcap-derived features are a subset of the Bot-IoT training schema.
# Any column in the training schema that tshark didn't populate is set to zero
# so the model receives a complete, correctly ordered feature vector.
for col in train_cols:
    if col not in raw.columns:
        raw[col] = 0

aligned = raw[["file", "label"] + train_cols]

# Save
out_path = RUNTIME_NET_DIR / "runtime_network_aligned.csv"
aligned.to_csv(out_path, index=False)
print(f"Aligned features saved → {out_path}")
print(aligned.head())
