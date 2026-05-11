def hybrid_decision(p_sys: float, p_net: float) -> int:
    """
    Late-fusion decision function from src/hybrid/fusion/run_hybrid_fusion.py
    """
    if p_sys >= 0.6:
        return 1
    elif p_net >= 0.5 and p_sys >= 0.1:
        return 1
    return 0

cases = [
    ("Case A (High Syscall, Low Network)", 0.85, 0.10, 1, "Syscall Dominant (p_sys >= 0.6)"),
    ("Case B (Low Syscall, High Network)", 0.05, 0.65, 0, "Insufficient Syscall (p_sys < 0.1)"),
    ("Case C (Moderate Syscall, High Network)", 0.15, 0.65, 1, "Network Corroborates (p_net >= 0.5 & p_sys >= 0.1)"),
    ("Case D (High Syscall, High Network)", 0.85, 0.65, 1, "Network Corroborates (p_net >= 0.5 & p_sys >= 0.1)"),
    ("Case E (Low Syscall, Low Network)", 0.05, 0.10, 0, "Insufficient Syscall (p_sys < 0.1)")
]

print("=== Validation Results ===")
print("Hybrid Fusion Logic Checks")
for name, p_sys, p_net, expected, reason in cases:
    pred = hybrid_decision(p_sys, p_net)
    status = "PASS" if pred == expected else "FAIL"
    print(f"\n{name}")
    print(f"Inputs: p_sys={p_sys}, p_net={p_net}")
    print(f"Predicted Label: {pred} | Expected: {expected} [{status}]")
    print(f"Reason: {reason}")
