"""
align_syscall_schema.py
Legacy helper for archived syscall .log traces in data/runtime/logs/.
It extracts syscall counts from those archived traces and aligns them to the
exact feature schema used during training.

The implemented automated runtime pipeline is sysdig/eBPF-based:
run_and_capture.sh writes data/runtime/syscall/sysdig_raw.txt, and
extract_syscall_features.py processes that sysdig output.

Outputs:
  data/runtime/syscall/runtime_features_raw.csv
  data/runtime/syscall/runtime_features_aligned.csv

Run from project root:
    python3 src/hybrid/runtime/align_syscall_schema.py
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import (
    RUNTIME_LOGS_DIR, RUNTIME_SYS_DIR, CHIDS_DATASET_PATH
)

import pandas as pd

SYSCALL_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\(")


def extract_syscall(line: str):
    m = SYSCALL_RE.search(line)
    return m.group(1) if m else None


def process_log(filepath: Path) -> dict:
    counts = {}
    with open(filepath, "r", errors="ignore") as f:
        for line in f:
            sc = extract_syscall(line)
            if sc:
                col = f"evt_{sc}"
                counts[col] = counts.get(col, 0) + 1
    counts["total_events"] = sum(counts.values())
    return counts


def infer_label(filename: str) -> int:
    # Labels are inferred from the filename convention set in run_and_capture.sh.
    # All malicious container logs are prefixed with 'mal_'.
    name = filename.lower()
    return 1 if ("mal_" in name or "malicious" in name) else 0


# Load training schema
print(f"Loading training schema: {CHIDS_DATASET_PATH}")
train_df = pd.read_csv(CHIDS_DATASET_PATH)

label_col = next((c for c in ["label", "label_bin", "true_label", "y"] if c in train_df.columns), None)
train_feature_cols = [c for c in train_df.columns if c != label_col]

if "total_events" not in train_feature_cols:
    raise ValueError("'total_events' not found in training schema.")

# Process archived log files
print(f"Processing archived syscall logs in: {RUNTIME_LOGS_DIR}")
rows = []
for fname in sorted(RUNTIME_LOGS_DIR.iterdir()):
    if fname.suffix != ".log":
        continue
    feats = process_log(fname)
    feats["file"]  = fname.name
    feats["label"] = infer_label(fname.name)
    rows.append(feats)

if not rows:
    raise RuntimeError(f"No .log files found in {RUNTIME_LOGS_DIR}")

runtime_df = pd.DataFrame(rows).fillna(0)

# Save raw
raw_path = RUNTIME_SYS_DIR / "runtime_features_raw.csv"
runtime_df.to_csv(raw_path, index=False)
print(f"Raw features saved → {raw_path}")

# Align to training schema
# Missing feature columns are filled with zero because live captures will rarely
# generate every syscall in the training schema. Zero is safe here — it means
# the syscall simply wasn't observed during the capture window, not that data is missing.
for col in train_feature_cols:
    if col not in runtime_df.columns:
        runtime_df[col] = 0

aligned_df = runtime_df[["file", "label"] + train_feature_cols].copy()
# These assertions are critical: column order must exactly match the scaler and
# model that were fitted on the training schema.
assert list(aligned_df.columns[2:]) == train_feature_cols, "Column order mismatch!"
assert aligned_df.isna().sum().sum() == 0, "NaNs found after alignment!"

aligned_path = RUNTIME_SYS_DIR / "runtime_features_aligned.csv"
aligned_df.to_csv(aligned_path, index=False)
print(f"Aligned features saved → {aligned_path}")
print("\nFiles processed:")
print(aligned_df[["file", "label"]].to_string(index=False))
