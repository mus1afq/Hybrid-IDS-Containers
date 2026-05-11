#!/usr/bin/env python3
"""
demo_runtime_scenario_4.py
==========================
End-to-end demonstration — Scenario 4: False Positive Suppression.

This scenario uses simulated probability values to demonstrate why
the syscall threshold (floor) is raised to 0.6 instead of the standard 0.5.

We simulate a benign container generating some noisy or slightly suspicious
syscalls (p_sys = 0.55). 
A standard single-model IDS would flag this as a false positive.
However, because the network is quiet (p_net = 0.10), the hybrid
logic successfully suppresses the false alarm.

Run from project root:
    python3 demo/demo_runtime_scenario_4.py
"""

import time
import os

def banner(step_num, title):
    print()
    print("=" * 60)
    print(f"  STEP {step_num}: {title}")
    print("=" * 60)
    time.sleep(1)

def pause(msg="Press Enter to continue to the next step..."):
    input(f"\n  ⏸  {msg}\n")

def hybrid_decision(p_sys: float, p_net: float) -> int:
    if p_sys >= 0.6:
        return 1
    elif p_net >= 0.5 and p_sys >= 0.1:
        return 1
    return 0

print()
print("╔══════════════════════════════════════════════════════════╗")
print("║     HYBRID IDS — RUNTIME SCENARIO 4 DEMONSTRATION      ║")
print("║     Simulated: False Positive Suppression              ║")
print("║     p_sys = 0.55 (suspicious), p_net = 0.10 (quiet)    ║")
print("╚══════════════════════════════════════════════════════════╝")

banner(1, "SCENARIO CONTEXT")
print("  This scenario simulates a completely BENIGN container that")
print("  is exhibiting slightly unusual process behaviour (e.g., an")
print("  update script or a heavy background task running).")
print()
print("    • Syscall probability: p_sys = 0.55")
print("    • Network probability: p_net = 0.10")
print()
print("  In a standard single-model IDS, the threshold is 0.5.")
print("  Therefore, the syscall model alone would trigger a FALSE POSITIVE.")
print("  Let's see how the hybrid logic handles it.")
pause()

banner(2, "SYSCALL MODEL INFERENCE")
p_sys = 0.55
print(f"  ┌────────────────────────────────────┐")
print(f"  │  p_sys = {p_sys:.2f}                      │")
print(f"  │  Standard verdict: MALICIOUS       │")
print(f"  └────────────────────────────────────┘")
print()
print("  A standalone syscall IDS would INCORRECTLY flag this.")
pause()

banner(3, "NETWORK MODEL INFERENCE")
p_net = 0.10
print(f"  ┌────────────────────────────────────┐")
print(f"  │  p_net = {p_net:.2f}                      │")
print(f"  │  Network verdict: BENIGN           │")
print(f"  └────────────────────────────────────┘")
print()
print("  The network sees absolutely no malicious traffic.")
pause()

banner(4, "LATE FUSION — HYBRID DECISION")
pred_hybrid = hybrid_decision(p_sys, p_net)
print("  Fusion rules evaluated in order:")
print()
print(f"    Rule 1: p_sys >= 0.6?")
print(f"            {p_sys:.2f} >= 0.6  →  NO  (Syscall not high enough to override)")
print()
print(f"    Rule 2: p_net >= 0.5 AND p_sys >= 0.1?")
print(f"            {p_net:.2f} >= 0.5  →  NO  (Network does not corroborate)")
print()
print(f"            Combined  →  BENIGN")
print()
print("  ╔════════════════════════════════════════════════════╗")
print("  ║                                                    ║")
print("  ║    VERDICT: BENIGN — NO THREAT DETECTED            ║")
print("  ║    (False Positive Successfully Prevented!)        ║")
print("  ║                                                    ║")
print("  ╚════════════════════════════════════════════════════╝")
pause()

banner(5, "WHY THIS MATTERS")
print("  This perfectly demonstrates why the syscall threshold")
print("  was raised to 0.6 in the hybrid architecture.")
print()
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  Approach              Verdict       Outcome         │")
print("  │  ─────────────────     ──────────    ──────────────  │")
print("  │  Syscall-only IDS      MALICIOUS     FALSE POSITIVE ✗│")
print("  │  Network-only IDS      BENIGN        TRUE NEGATIVE  ✓│")
print("  │  Hybrid IDS (Ours)     BENIGN        TRUE NEGATIVE  ✓│")
print("  └──────────────────────────────────────────────────────┘")
print()
print("  The hybrid system successfully ignores weak, noisy syscall")
print("  anomalies unless the network traffic backs them up.")
print()
print("╔══════════════════════════════════════════════════════════╗")
print("║           SCENARIO 4 DEMONSTRATION COMPLETE              ║")
print("╚══════════════════════════════════════════════════════════╝")
print()
