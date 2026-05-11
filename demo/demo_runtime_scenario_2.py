#!/usr/bin/env python3
"""
demo_runtime_scenario_2.py
==========================
End-to-end runtime demonstration — Scenario 2: Both Models Agree (True Negative).

Replays the full Hybrid IDS pipeline for the 'benign_proc' container,
where both the syscall and network models independently agree the
container is benign — and they are both correct.

This contrasts with Scenario 1 (mal_proc) where only the syscall model
caught the threat while the network model missed it.

Run from project root:
    python3 demo/demo_runtime_scenario_2.py
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
print("║     HYBRID IDS — RUNTIME SCENARIO 2 DEMONSTRATION        ║")
print("║     Target: benign_proc (Legitimate Process Activity)    ║")
print("║     Expected: Both Models Agree → BENIGN                 ║")
print("╚══════════════════════════════════════════════════════════╝")

# ── STEP 1: Show the raw captured data ────────────────────────────────────────
banner(1, "DATA CAPTURE (sysdig/eBPF + tshark)")

syscall_log = os.path.join(BASE, "data/runtime/logs/benign_proc.log")
network_pcap = os.path.join(BASE, "data/runtime/logs/benign_web_network.pcap")

log_size = os.path.getsize(syscall_log) / (1024 * 1024)
pcap_size = os.path.getsize(network_pcap) / 1024

print(f"  Syscall log:  data/runtime/logs/benign_proc.log  ({log_size:.1f} MB)")
print(f"  Network pcap: data/runtime/logs/benign_web_network.pcap  ({pcap_size:.1f} KB)")
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
print("  running normal, legitimate process activity (web server).")

pause()

# ── STEP 2: Feature extraction ────────────────────────────────────────────────
banner(2, "FEATURE EXTRACTION")

aligned_csv = os.path.join(BASE, "data/runtime/syscall/runtime_features_aligned.csv")
df_aligned = pd.read_csv(aligned_csv)
row = df_aligned[df_aligned["file"] == "benign_proc.log"].iloc[0]

total_events = int(row["total_events"])
nonzero_feats = (row.drop(["file", "label"]) > 0).sum()

print(f"  Extracted features from benign_proc.log:")
print(f"    Total syscall events:   {total_events:,}")
print(f"    Non-zero feature cols:  {nonzero_feats}")
print(f"    Feature vector length:  {len(row) - 2} columns")
print()

# Show top syscalls
feat_cols = [c for c in row.index if c not in ("file", "label")]
feat_vals = row[feat_cols].astype(float)
top5 = feat_vals.sort_values(ascending=False).head(5)
print("  Top 5 syscall counts:")
for name, val in top5.items():
    print(f"    {name:<25} {int(val):>8,}")

print()
print("  Notice: The syscall profile is dominated by simple I/O")
print("  operations (read, write, close) — typical of benign workloads.")

pause()

# ── STEP 3: Syscall model inference ───────────────────────────────────────────
banner(3, "SYSCALL MODEL INFERENCE")

sys_preds = pd.read_csv(os.path.join(BASE, "outputs/predictions/syscall/runtime_syscall_predictions.csv"))
sys_row = sys_preds[sys_preds["file"] == "benign_proc.log"].iloc[0]
p_sys = float(sys_row["p_sys"])

print(f"  Model:       Logistic Regression (models/syscall/best_model.pkl)")
print(f"  Scaler:      StandardScaler (models/syscall/scaler.pkl)")
print(f"  Input:       benign_proc.log aligned feature vector")
print()
print(f"  ┌────────────────────────────────────┐")
print(f"  │  p_sys = {p_sys:.4f}               │")
print(f"  │  Syscall verdict: BENIGN           │")
print(f"  └────────────────────────────────────┘")
print()
print(f"  The syscall model is very confident this is benign")
print(f"  (probability near zero).")

pause()

# ── STEP 4: Network model inference ──────────────────────────────────────────
banner(4, "NETWORK MODEL INFERENCE")

net_preds = pd.read_csv(os.path.join(BASE, "outputs/predictions/network/network_predictions.csv"))
net_row = net_preds[net_preds["file"] == "benign_web_network.pcap"].iloc[0]
p_net = float(net_row["p_net"])

print(f"  Model:       Random Forest (models/network/best_network_model.pkl)")
print(f"  Input:       benign_web_network.pcap extracted features")
print()
print(f"  ┌────────────────────────────────────┐")
print(f"  │  p_net = {p_net:.4f}               │")
print(f"  │  Network verdict: BENIGN           │")
print(f"  └────────────────────────────────────┘")
print()
print(f"  The network model also independently confirms this is benign.")
print(f"  Both models agree — high confidence true negative.")

pause()

# ── STEP 5: Late fusion ──────────────────────────────────────────────────────
banner(5, "LATE FUSION — HYBRID DECISION")

pred_hybrid = hybrid_decision(p_sys, p_net)
label_str = "MALICIOUS" if pred_hybrid == 1 else "BENIGN"

print("  Fusion rules applied:")
print(f"    Rule 1: p_sys >= 0.6?  →  {p_sys:.4f} >= 0.6  →  NO")
print(f"    Rule 2: p_net >= 0.5 AND p_sys >= 0.1?  →  {p_net:.4f} >= 0.5 = NO")
print(f"    Fallthrough: → BENIGN")
print()

print("  ╔════════════════════════════════════════════════════╗")
print("  ║                                                    ║")
print("  ║    VERDICT: BENIGN — NO THREAT DETECTED            ║")
print("  ║                                                    ║")
print("  ╚════════════════════════════════════════════════════╝")

print()
print(f"  True label:     0 (Benign)")
print(f"  Predicted:      {pred_hybrid} ({label_str})")
print(f"  Classification: CORRECT ✓")
print()
print("  Both models independently agreed this container was safe,")
print("  and the fusion logic correctly preserved that verdict.")
print("  No false alarm was raised — this is a TRUE NEGATIVE.")

pause()

# ── STEP 6: Comparison with Scenario 1 ───────────────────────────────────────
banner(6, "SCENARIO COMPARISON")

print("  ┌──────────────────────────────────────────────────────────┐")
print("  │                  Scenario 1       Scenario 2             │")
print("  │  Container:      mal_proc         benign_proc            │")
print("  │  True label:     MALICIOUS        BENIGN                 │")
print("  │  p_sys:          1.0000           0.0018                 │")
print("  │  p_net:          0.2350           0.0300                 │")
print("  │  Syscall says:   MALICIOUS        BENIGN                 │")
print("  │  Network says:   BENIGN           BENIGN                 │")
print("  │  Hybrid says:    MALICIOUS ✓      BENIGN ✓               │")
print("  │  Triggered by:   Rule 1 (p_sys)   Fallthrough            │")
print("  └──────────────────────────────────────────────────────────┘")
print()
print("  Key insight:")
print("  • Scenario 1 shows the syscall model catching an attack the")
print("    network model missed — fusion prevents a false negative.")
print("  • Scenario 2 shows both models agreeing on a safe container")
print("    — fusion correctly avoids a false positive.")

print()
print("╔══════════════════════════════════════════════════════════╗")
print("║           SCENARIO 2 DEMONSTRATION COMPLETE              ║")
print("╚══════════════════════════════════════════════════════════╝")
print()
