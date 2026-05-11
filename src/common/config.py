"""
config.py — Shared path configuration for the Hybrid IDS project.

All paths anchor off the FYP project root, resolved from this file's
location (FYP/src/common/config.py → parents[2] = FYP/).

Usage in any src/ script:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from common.config import *
"""

from pathlib import Path

# Project root
# Resolve the root relative to this file's location so that scripts can be run
# from any working directory without needing to set PYTHONPATH manually.
# parents[0] = common/, parents[1] = src/, parents[2] = FYP/ (project root).
ROOT = Path(__file__).resolve().parents[2]   # FYP/

# Top-level directories
DATA_DIR    = ROOT / "data"
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# Raw data
RAW_SYSCALL_DIR = DATA_DIR / "raw" / "syscall"
RAW_NETWORK_DIR = DATA_DIR / "raw" / "network"
RAW_HYBRID_DIR  = DATA_DIR / "raw" / "hybrid"

# Runtime captured data
RUNTIME_LOGS_DIR = DATA_DIR / "runtime" / "logs"
RUNTIME_SYS_DIR  = DATA_DIR / "runtime" / "syscall"
RUNTIME_NET_DIR  = DATA_DIR / "runtime" / "network"

# Dataset file paths
CHIDS_DATASET_PATH   = RAW_SYSCALL_DIR / "chids_dataset_clean.csv"
NETWORK_DATASET_PATH = RAW_NETWORK_DIR / "network_dataset_cleaned.csv"
BOTIOT_TEST_PATH     = RAW_NETWORK_DIR / "botiot_synthetic_test_1000.csv"
HYBRID_PROBS_PATH    = RAW_HYBRID_DIR  / "hybrid_synthetic_probs_test.csv"

# Trained model / artefact paths
SYSCALL_MODEL_PATH  = MODELS_DIR / "syscall" / "best_model.pkl"
SYSCALL_SCALER_PATH = MODELS_DIR / "syscall" / "scaler.pkl"
NETWORK_MODEL_PATH  = MODELS_DIR / "network" / "best_network_model.pkl"

# Output directories
PRED_SYSCALL_DIR = OUTPUTS_DIR / "predictions" / "syscall"
PRED_NETWORK_DIR = OUTPUTS_DIR / "predictions" / "network"
PRED_HYBRID_DIR  = OUTPUTS_DIR / "predictions" / "hybrid"
METRICS_DIR      = OUTPUTS_DIR / "metrics"
PLOTS_DIR        = OUTPUTS_DIR / "plots"

# Ensure output dirs always exist when config is imported
# Creating dirs here means every script that imports config can safely write
# outputs without needing its own os.makedirs call.
for _d in [PRED_SYSCALL_DIR, PRED_NETWORK_DIR, PRED_HYBRID_DIR,
           METRICS_DIR, PLOTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
