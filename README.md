# Hybrid Intrusion Detection System (IDS) for Containerised Environments

Final Year Project artefact implementing a hybrid, runtime intrusion detection
pipeline for Docker containers. The system combines **sysdig/eBPF-based syscall
monitoring** with `tshark` network telemetry, extracts fixed-schema feature
vectors, and applies independent machine-learning models before a deterministic
late-fusion decision layer.

The research goal is to evaluate whether cross-domain fusion improves detection
reliability over syscall-only or network-only monitoring. In the supplied
evaluation artefacts, the hybrid model improves F1 score while preserving a zero
false-positive rate on the synthetic paired evaluation set.

> **Environment:** Python 3.11+ | macOS / Linux | Docker | sysdig | tshark

## Project Structure

```
FYP/
├── data/
│   ├── raw/
│   │   ├── syscall/          # CHIDS dataset (chids_dataset_clean.csv)
│   │   ├── network/          # Bot-IoT dataset + synthetic test set
│   │   └── hybrid/           # Pre-paired probability CSVs for fusion evaluation
│   └── runtime/
│       ├── logs/             # Archived syscall traces + tshark .pcap files for demos
│       ├── syscall/          # sysdig_raw.txt, feature CSVs (features_N*.csv, aligned)
│       └── network/          # Extracted & aligned network feature CSVs
│
├── models/
│   ├── syscall/              # best_model.pkl (Logistic Regression), scaler.pkl
│   └── network/              # best_network_model.pkl (Random Forest)
│
├── src/
│   ├── common/
│   │   └── config.py         # Shared path constants — import this everywhere
│   ├── syscall/
│   │   ├── training/         # train_syscall_model.py
│   │   ├── inference/        # run_syscall_inference.py
│   │   └── evaluation/       # evaluate_syscall.py, test_synthetic_syscall.py
│   ├── network/
│   │   ├── training/         # train_network_model.py
│   │   ├── inference/        # run_network_inference.py
│   │   └── evaluation/       # evaluate_network.py
│   └── hybrid/
│       ├── fusion/           # run_hybrid_fusion.py
│       ├── evaluation/       # evaluate_hybrid_synthetic.py, evaluate_hybrid_runtime.py
│       └── runtime/          # Docker capture + feature extraction scripts
│
├── tests/                    # Dissertation validation suite (run_all_tests.sh)
│
├── demo/                     # Runtime scenarios for presentation (1-4)
│
├── outputs/
│   ├── predictions/
│   │   ├── syscall/          # runtime_syscall_predictions.csv, syscall_synthetic_validation.csv
│   │   ├── network/          # runtime_network_predictions.csv, network_synthetic_validation.csv
│   │   └── hybrid/           # runtime_hybrid_predictions.csv, hybrid_synthetic_validation.csv
│   ├── metrics/              # evaluation_table.csv, hybrid_results.csv
│   └── plots/                # confusion matrices, ROC curves
│
└── README.md
```

---

## Key Results

Performance on the synthetic evaluation dataset (*n = 580 samples*):

| Model | Accuracy | Precision | Recall | F1 Score | FPR | FNR |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Syscall-Only | 0.6517 | 1.0000 | 0.5609 | 0.7187 | 0.000 | 0.4391 |
| Network-Only | 0.7828 | 1.0000 | 0.7261 | 0.8413 | 0.000 | 0.2739 |
| **Hybrid**   | **0.8534** | **1.0000** | **0.8152** | **0.8982** | **0.000** | **0.1848** |

**N-Window Analysis**: Peak F1 = **0.957** at detection window **N = 15 seconds**.
*(N-window analysis measures how model performance scales depending on the duration of the telemetry capture window before a decision is made).*

---

## Methodology at a Glance

1. **Runtime capture:** `run_and_capture.sh` launches Docker workloads and starts a
   privileged `docker/sysdig` container to collect syscall events from the host
   namespace. This is the implemented syscall capture path.
2. **Syscall feature extraction:** `extract_syscall_features.py` parses
   `data/runtime/syscall/sysdig_raw.txt`, maps raw syscall names onto the CHIDS
   feature schema, and exports full-window and N-window feature CSVs.
3. **Network feature extraction:** `extract_network_features.py` processes
   `tshark`/PCAP-derived network data and aligns it to the Bot-IoT schema.
4. **Inference and fusion:** Logistic Regression models syscall behaviour,
   Random Forest models network behaviour, and a deterministic late-fusion rule
   combines the two probabilities.

The runtime monitoring pipeline is therefore **sysdig/eBPF-based syscall
monitoring**.

---

## Running the Pipelines

All scripts are run from the **project root** (`FYP/`).

### 1 — Train Models (only needed if retraining)
```bash
python3 src/syscall/training/train_syscall_model.py
python3 src/network/training/train_network_model.py
```

### 2 — Evaluate on Synthetic Data
```bash
# Syscall model validation (N-window analysis)
python3 src/syscall/evaluation/evaluate_syscall.py

# Network model evaluation (Bot-IoT test set)
python3 src/network/evaluation/evaluate_network.py

# Hybrid fusion evaluation (pre-paired probabilities)
python3 src/hybrid/evaluation/evaluate_hybrid_synthetic.py
```

### 3 — Runtime Pipeline (Docker containers)
```bash
# Step 1: Capture sysdig/eBPF syscall data and PCAP/network telemetry
bash src/hybrid/runtime/run_and_capture.sh

# Step 2a: Extract syscall features from data/runtime/syscall/sysdig_raw.txt
python3 src/hybrid/runtime/extract_syscall_features.py
# Step 2b: Extract network features
python3 src/hybrid/runtime/extract_network_features.py

# Step 3: Align network features to training schema
python3 src/hybrid/runtime/align_network_schema.py

# Optional: align archived syscall .log traces used by legacy/demo artefacts
python3 src/hybrid/runtime/align_syscall_schema.py

# Step 4a: Run syscall inference over aligned runtime syscall features
python3 src/syscall/inference/run_syscall_inference.py
# Step 4b: Run network inference
python3 src/network/inference/run_network_inference.py

# Step 5: Fuse predictions
python3 src/hybrid/fusion/run_hybrid_fusion.py

# Step 6: Evaluate runtime results
python3 src/hybrid/evaluation/evaluate_hybrid_runtime.py
```


### 4 — Final Dissertation Verification
To validate the model pipelines dynamically and generate all metric outputs, tables, and conceptual validation checks for the dissertation, run the unified verification shell script mapping out all tests in the `/tests` folder.
```bash
# Run all end-to-end dissertation tests (from project root)
bash tests/run_all_tests.sh
```

---

## System Architecture

```
Docker Containers
    │
    ├─── docker/sysdig (eBPF) ──→ sysdig_raw.txt ─→ extract_syscall_features.py ─→ run_syscall_inference.py ─┐
    │                                                                                                       │
    └─── tshark ──→ data/runtime/logs/*.pcap ─→ align_network_schema.py ─────→ run_network_inference.py ──┤
                                                                                                            ↓
                                                                                  run_hybrid_fusion.py → prediction
```

---

## Fusion Logic

The late-fusion decision mechanism is strictly **deterministic and rule-based**, rather than a learned meta-classifier. This design decision ensures explainability and guarantees that specific domain overrides behave predictably.

```python
def hybrid_decision(p_sys, p_net):
    if p_sys >= 0.6:                      # Syscall highly confident
        return MALICIOUS
    elif p_net >= 0.5 and p_sys >= 0.1:  # Network flags + syscall suspicious
        return MALICIOUS
    return BENIGN
```

---

## Requirements

```bash
pip install scikit-learn pandas numpy matplotlib seaborn joblib
```

Optional: `xgboost`, `lightgbm` for additional classifier benchmarking.
Runtime capture requires: `Docker`, `sysdig`, `tshark`.
