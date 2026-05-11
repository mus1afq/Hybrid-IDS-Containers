#!/usr/bin/env python3
"""
demo_runtime_scenario.py
========================
End-to-end runtime demonstration for dissertation viva.

Replays the full Hybrid IDS pipeline for the 'mal_proc' container
(a true-positive malicious scenario) using pre-captured data.

Each stage pauses briefly so the presenter can narrate to examiners.

Run from project root:
    python3 tests/demo_runtime_scenario.py
"""

import time
import os
import pandas as pd

# ── Helpers ───────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def banner(step_num, title):
    print()
    print("=" * 60)
    print(f"  STEP {step_num}: {title}")
    print("=" * 60)
    time.sleep(1)

def pause(msg="Press Enter to continue to the next step..."):
    input(f"\n  ⏸  {msg}\n")

def hybrid_decision(p_sys: float, p_net: float) -> int:
    """
    Late-fusion decision function (from src/hybrid/fusion/run_hybrid_fusion.py):
      - p_sys >= 0.6               → malicious (syscall dominant)
      - p_net >= 0.5 AND p_sys >= 0.1 → malicious (network corroborates)
      - else                       → benign
    """
    if p_sys >= 0.6:
        return 1
    elif p_net >= 0.5 and p_sys >= 0.1:
        return 1
    return 0


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO START
# ══════════════════════════════════════════════════════════════════════════════
print()
print("╔══════════════════════════════════════════════════════════╗")
print("║     HYBRID IDS — RUNTIME SCENARIO DEMONSTRATION        ║")
print("║     Target: mal_proc (Malicious Process Spawning)      ║")
print("╚══════════════════════════════════════════════════════════╝")

# ── STEP 1: Show the raw captured data ────────────────────────────────────────
banner(1, "DATA CAPTURE (sysdig/eBPF + tshark)")

syscall_log = os.path.join(BASE, "data/runtime/logs/mal_proc.log")
network_pcap = os.path.join(BASE, "data/runtime/logs/mal_conn_network.pcap")

log_size = os.path.getsize(syscall_log) / (1024 * 1024)
pcap_size = os.path.getsize(network_pcap) / 1024

print(f"  Syscall log:  data/runtime/logs/mal_proc.log  ({log_size:.1f} MB)")
print(f"  Network pcap: data/runtime/logs/mal_conn_network.pcap  ({pcap_size:.1f} KB)")
print()
print("  Raw syscall trace (first 5 lines):")
print("  " + "-" * 55)
with open(syscall_log, "r") as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(f"    {line.rstrip()}")
print("    ...")
print()
print("  These raw logs were captured from a Docker container")
print("  running a malicious process-spawning simulation.")

pause()

# ── STEP 2: Feature extraction ────────────────────────────────────────────────
banner(2, "FEATURE EXTRACTION")

aligned_csv = os.path.join(BASE, "data/runtime/syscall/runtime_features_aligned.csv")
df_aligned = pd.read_csv(aligned_csv)
mal_proc_row = df_aligned[df_aligned["file"] == "mal_proc.log"].iloc[0]

total_events = int(mal_proc_row["total_events"])
nonzero_feats = (mal_proc_row.drop(["file", "label"]) > 0).sum()

print(f"  Extracted features from mal_proc.log:")
print(f"    Total syscall events:   {total_events:,}")
print(f"    Non-zero feature cols:  {nonzero_feats}")
print(f"    Feature vector length:  {len(mal_proc_row) - 2} columns")
print()

# Show top syscalls
feat_cols = [c for c in mal_proc_row.index if c not in ("file", "label")]
feat_vals = mal_proc_row[feat_cols].astype(float)
top5 = feat_vals.sort_values(ascending=False).head(5)
print("  Top 5 syscall counts:")
for name, val in top5.items():
    print(f"    {name:<25} {int(val):>8,}")

print()
print("  The raw syscall trace was parsed into a fixed-length")
print("  frequency vector matching the CHIDS training schema.")

pause()

# ── STEP 3: Syscall model inference ───────────────────────────────────────────
banner(3, "SYSCALL MODEL INFERENCE")

sys_preds = pd.read_csv(os.path.join(BASE, "outputs/predictions/syscall/runtime_syscall_predictions.csv"))
sys_row = sys_preds[sys_preds["file"] == "mal_proc.log"].iloc[0]
p_sys = float(sys_row["p_sys"])

print(f"  Model:       Logistic Regression (models/syscall/best_model.pkl)")
print(f"  Scaler:      StandardScaler (models/syscall/scaler.pkl)")
print(f"  Input:       mal_proc.log aligned feature vector")
print()
print(f"  ┌────────────────────────────────────┐")
print(f"  │  p_sys = {p_sys:.4f}                    │")
print(f"  │  Syscall verdict: {'MALICIOUS' if p_sys >= 0.5 else 'BENIGN':<18} │")
print(f"  └────────────────────────────────────┘")

pause()

# ── STEP 4: Network model inference ──────────────────────────────────────────
banner(4, "NETWORK MODEL INFERENCE")

net_preds = pd.read_csv(os.path.join(BASE, "outputs/predictions/network/network_predictions.csv"))
net_row = net_preds[net_preds["file"] == "mal_conn_network.pcap"].iloc[0]
p_net = float(net_row["p_net"])

print(f"  Model:       Random Forest (models/network/best_network_model.pkl)")
print(f"  Input:       mal_conn_network.pcap extracted features")
print()
print(f"  ┌────────────────────────────────────┐")
print(f"  │  p_net = {p_net:.4f}                   │")
print(f"  │  Network verdict: {'MALICIOUS' if p_net >= 0.5 else 'BENIGN':<18} │")
print(f"  └────────────────────────────────────┘")
print()
print("  Note: The network model alone does NOT flag this as malicious")
print("  (p_net < 0.5). This is why fusion is critical.")

pause()

# ── STEP 5: Late fusion ──────────────────────────────────────────────────────
banner(5, "LATE FUSION — HYBRID DECISION")

pred_hybrid = hybrid_decision(p_sys, p_net)
label_str = "MALICIOUS" if pred_hybrid == 1 else "BENIGN"

print("  Fusion rules applied:")
print(f"    Rule 1: p_sys >= 0.6?  →  {p_sys:.4f} >= 0.6  →  {'YES ✓' if p_sys >= 0.6 else 'NO'}")
if p_sys < 0.6:
    print(f"    Rule 2: p_net >= 0.5 AND p_sys >= 0.1?  →  {p_net:.4f} >= 0.5 = {'YES' if p_net >= 0.5 else 'NO'}, {p_sys:.4f} >= 0.1 = {'YES' if p_sys >= 0.1 else 'NO'}")
print()

if pred_hybrid == 1:
    print("  ╔════════════════════════════════════════════════════╗")
    print("  ║                                                    ║")
    print("  ║        VERDICT: MALICIOUS — INTRUSION DETECTED     ║")
    print("  ║                                                    ║")
    print("  ╚════════════════════════════════════════════════════╝")
else:
    print("  ╔════════════════════════════════════════════════════╗")
    print("  ║    VERDICT: BENIGN — NO THREAT DETECTED            ║")
    print("  ╚════════════════════════════════════════════════════╝")

print()
print(f"  True label:     1 (Malicious)")
print(f"  Predicted:      {pred_hybrid} ({label_str})")
print(f"  Classification: {'CORRECT ✓' if pred_hybrid == 1 else 'INCORRECT ✗'}")

print()
print("╔══════════════════════════════════════════════════════════╗")
print("║              DEMONSTRATION COMPLETE                     ║")
print("╚══════════════════════════════════════════════════════════╝")
print()
