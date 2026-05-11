"""
run_hybrid_fusion.py
Merges runtime syscall and network predictions using the late-fusion
decision function and writes the combined output to outputs/predictions/hybrid/.

Run from project root:
    python3 src/hybrid/fusion/run_hybrid_fusion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import PRED_SYSCALL_DIR, PRED_NETWORK_DIR, PRED_HYBRID_DIR

import pandas as pd


def hybrid_decision(p_sys: float, p_net: float) -> int:
    """
    Late-fusion decision function:
      - p_sys >= 0.6               → malicious (syscall dominant)
      - p_net >= 0.5 AND p_sys >= 0.1 → malicious (network corroborates)
      - else                       → benign
    """
    # Syscall threshold is set higher (0.6 vs 0.5) because local process behaviour
    # is a more reliable intrusion signal than network flows in containerised workloads.
    # A high p_sys alone is sufficient to flag malicious activity.
    if p_sys >= 0.6:
        return 1
    # Network evidence only counts when syscall is at least weakly suspicious (p_sys >= 0.1).
    # This prevents a noisy network capture from overriding a clean syscall trace.
    elif p_net >= 0.5 and p_sys >= 0.1:
        return 1
    return 0


def main():
    # Load predictions
    sys_path = PRED_SYSCALL_DIR / "runtime_syscall_predictions.csv"
    net_path = PRED_NETWORK_DIR / "runtime_network_predictions.csv"

    print(f"Loading syscall predictions: {sys_path}")
    print(f"Loading network predictions: {net_path}")

    sys_df = pd.read_csv(sys_path)
    net_df = pd.read_csv(net_path)

    # Extract base names for pairing
    sys_df["base"] = sys_df["file"].str.replace(".log",  "", regex=False)
    net_df["base"] = (net_df["file"]
                      .str.replace("_network.pcap", "", regex=False)
                      .str.replace(".pcap",          "", regex=False))

    # Manual pairing: syscall container ↔ network capture
    # CHIDS (syscall) and the pcap captures were collected independently from
    # separate container workloads, so there is no shared row ID to join on.
    # This map pairs the most semantically comparable workloads:
    # e.g. the process-heavy benign container (benign_proc) is matched with
    # the HTTP-serving capture (benign_web) because both represent idle-ish
    # legitimate container behaviour.
    pair_map = {
        "benign_proc": "benign_web",
        "benign_idle": "benign_ping",
        "benign_loop": "benign_nginx",
        "mal_file":    "mal_flood",
        "mal_proc":    "mal_conn",
        "mal_rename":  "mal_scan",
    }

    rows = []
    for sys_base, net_base in pair_map.items():
        sys_row = sys_df[sys_df["base"] == sys_base]
        net_row = net_df[net_df["base"] == net_base]
        if sys_row.empty or net_row.empty:
            print(f"  [!] No match for pair: {sys_base} ↔ {net_base}")
            continue
        sys_row = sys_row.iloc[0]
        net_row = net_row.iloc[0]
        rows.append({
            "base":          sys_base,
            "network_base":  net_base,
            "label":         sys_row["label"],
            # p_sys and p_net are raw probabilities from each model's predict_proba.
            # The hybrid_decision function uses these directly — individual thresholds
            # (pred_syscall, pred_network) are stored for comparison purposes only.
            "p_sys":         sys_row["p_sys"],
            "p_net":         net_row["p_net"],
            "pred_syscall":  int(sys_row["p_sys"] >= 0.5),
            "pred_network":  int(net_row["p_net"] >= 0.5),
        })

    if not rows:
        raise ValueError("No paired rows found. Check that both inference scripts have been run first.")

    merged = pd.DataFrame(rows)
    merged["pred_hybrid"] = merged.apply(
        lambda r: hybrid_decision(float(r["p_sys"]), float(r["p_net"])), axis=1
    )

    #Save
    out_path = PRED_HYBRID_DIR / "runtime_hybrid_predictions.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved hybrid predictions → {out_path}")
    print(merged.to_string(index=False))

if __name__ == "__main__":
    main()
