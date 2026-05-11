#!/usr/bin/env python3
"""
demo_runtime_scenario_3.py
==========================
End-to-end demonstration — Scenario 3: Network Corroboration (Rule 2).

This scenario uses simulated probability values to demonstrate the
second fusion rule: p_net >= 0.5 AND p_sys >= 0.1 → MALICIOUS.

Neither model is individually highly confident, but together they
corroborate each other to catch an attack that would otherwise be missed.

This rule is critical because:
  - The syscall model alone would say BENIGN (p_sys = 0.15, below 0.5)
  - The network model alone would say MALICIOUS (p_net = 0.65)
  - But Rule 2 requires BOTH signals to agree before flagging

Run from project root:
    python3 demo/demo_runtime_scenario_3.py
"""

import time
import os

# ── Helpers ───────────────────────────────────────────────────────────────────
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
print("║     HYBRID IDS — RUNTIME SCENARIO 3 DEMONSTRATION      ║")
print("║     Simulated: Network Corroboration (Fusion Rule 2)   ║")
print("║     p_sys = 0.15 (weak signal), p_net = 0.65 (strong)  ║")
print("╚══════════════════════════════════════════════════════════╝")

# ── STEP 1: Context ──────────────────────────────────────────────────────────
banner(1, "SCENARIO CONTEXT")

print("  This scenario simulates a stealthy attack where:")
print()
print("    • The attacker's syscall footprint is subtle — only a")
print("      slight deviation from normal behaviour (p_sys = 0.15).")
print("    • However, the network traffic is clearly anomalous —")
print("      the network model flags suspicious packets (p_net = 0.65).")
print()
print("  Neither model alone would confidently raise an alarm with")
print("  standard thresholds (both typically need >= 0.5).")
print()
print("  This is where the fusion logic provides its key advantage:")
print("  cross-domain corroboration catches what single models miss.")

pause()

# ── STEP 2: Syscall model output ─────────────────────────────────────────────
banner(2, "SYSCALL MODEL INFERENCE")

p_sys = 0.15

print(f"  Model:       Logistic Regression")
print(f"  Input:       Simulated syscall feature vector")
print()
print(f"  ┌────────────────────────────────────┐")
print(f"  │  p_sys = {p_sys:.2f}                      │")
print(f"  │  Syscall verdict: BENIGN           │")
print(f"  └────────────────────────────────────┘")
print()
print("  The syscall model sees only minor anomalies in the")
print("  process behaviour — not enough to flag on its own.")
print("  A standalone syscall IDS would MISS this attack.")

pause()

# ── STEP 3: Network model output ─────────────────────────────────────────────
banner(3, "NETWORK MODEL INFERENCE")

p_net = 0.65

print(f"  Model:       Random Forest")
print(f"  Input:       Simulated network feature vector")
print()
print(f"  ┌────────────────────────────────────┐")
print(f"  │  p_net = {p_net:.2f}               │")
print(f"  │  Network verdict: MALICIOUS        │")
print(f"  └────────────────────────────────────┘")
print()
print("  The network model detects unusual traffic patterns —")
print("  high packet rates or connections to suspicious ports.")
print("  On its own, this could be a false alarm.")

pause()

# ── STEP 4: Fusion — the key step ────────────────────────────────────────────
banner(4, "LATE FUSION — HYBRID DECISION (RULE 2)")

pred_hybrid = hybrid_decision(p_sys, p_net)

print("  Fusion rules evaluated in order:")
print()
print(f"    Rule 1: p_sys >= 0.6?")
print(f"            {p_sys:.2f} >= 0.6  →  NO  (syscall not dominant)")
print()
print(f"    Rule 2: p_net >= 0.5 AND p_sys >= 0.1?")
print(f"            {p_net:.2f} >= 0.5  →  YES ✓  (network flags threat)")
print(f"            {p_sys:.2f} >= 0.1  →  YES ✓  (syscall corroborates)")
print(f"            Combined  →  MALICIOUS")
print()

print("  ╔════════════════════════════════════════════════════╗")
print("  ║                                                    ║")
print("  ║        VERDICT: MALICIOUS — INTRUSION DETECTED     ║")
print("  ║                                                    ║")
print("  ╚════════════════════════════════════════════════════╝")

pause()

# ── STEP 5: Why this matters ─────────────────────────────────────────────────
banner(5, "WHY RULE 2 MATTERS")

print("  Without fusion, this attack would be missed or mishandled:")
print()
print("  ┌──────────────────────────────────────────────────────┐")
print("  │  Approach              Verdict       Outcome         │")
print("  │  ─────────────────     ──────────    ──────────────  │")
print("  │  Syscall-only IDS      BENIGN        FALSE NEGATIVE  │")
print("  │  Network-only IDS      MALICIOUS     Possible FP?    │")
print("  │  Hybrid IDS (Ours)     MALICIOUS     CONFIRMED ✓     │")
print("  └──────────────────────────────────────────────────────┘")
print()
print("  The 0.1 threshold on p_sys acts as a sanity check:")
print("  it ensures the network alert isn't triggered by pure noise.")
print()
print("  Contrast with Case B (p_sys=0.05, p_net=0.65):")
print("    → p_sys < 0.1, so Rule 2 does NOT fire → BENIGN")
print("    → This prevents false positives from network-only noise.")

pause()

# ── STEP 6: All three scenarios compared ──────────────────────────────────────
banner(6, "ALL SCENARIOS COMPARED")

print("  ┌───────────────────────────────────────────────────────────────┐")
print("  │              Scenario 1      Scenario 2      Scenario 3       │")
print("  │  Container:  mal_proc        benign_proc     (simulated)      │")
print("  │  True label: MALICIOUS       BENIGN          MALICIOUS        │")
print("  │  p_sys:      1.0000          0.0018          0.1500           │")
print("  │  p_net:      0.2350          0.0300          0.6500           │")
print("  │  Syscall:    MALICIOUS       BENIGN          BENIGN           │")
print("  │  Network:    BENIGN          BENIGN          MALICIOUS        │")
print("  │  Hybrid:     MALICIOUS ✓     BENIGN ✓        MALICIOUS ✓      │")
print("  │  Rule used:  Rule 1          Fallthrough     Rule 2           │")
print("  │              (p_sys≥0.6)     (both low)      (corroborate)    │")
print("  └───────────────────────────────────────────────────────────────┘")
print()
print("  Together, these three scenarios demonstrate all paths through")
print("  the fusion logic and justify every threshold in the design.")

print()
print("╔══════════════════════════════════════════════════════════╗")
print("║           SCENARIO 3 DEMONSTRATION COMPLETE              ║")
print("╚══════════════════════════════════════════════════════════╝")
print()
