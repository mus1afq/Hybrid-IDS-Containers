"""
extract_network_features.py
Extracts flow-level features from pcap files in data/runtime/logs/
using tshark. Outputs raw features to data/runtime/network/.

Run from project root:
    python3 src/hybrid/runtime/extract_network_features.py

Requirements: tshark must be installed and on PATH.
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from common.config import RUNTIME_LOGS_DIR, RUNTIME_NET_DIR

import pandas as pd


def extract_features(pcap: Path) -> dict:
    cmd = [
        "tshark", "-r", str(pcap), "-T", "fields",
        "-e", "frame.len",
        "-e", "ip.proto",
        "-e", "tcp.dstport",
        "-e", "udp.dstport",
        "-e", "frame.time_epoch",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    lines  = result.stdout.strip().split("\n")

    lengths, times, protos, dports = [], [], [], []
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        try:
            if parts[0]: lengths.append(float(parts[0]))
            if parts[4]: times.append(float(parts[4]))
            if parts[1]: protos.append(parts[1])
            if parts[2]:   dports.append(int(parts[2]))
            elif parts[3]: dports.append(int(parts[3]))
        except Exception:
            continue

    pkts        = len(lengths)
    total_bytes = sum(lengths)
    dur         = (max(times) - min(times)) if len(times) > 1 else 0
    rate        = pkts / dur if dur > 0 else 0

    return {
        # Bot-IoT features include source/destination byte counts separately.
        # tshark sees all packets from a single perspective, so destination-side
        # values (dbytes, dpkts, drate) are zeroed. The model is robust to this
        # because the dominant Bot-IoT features are total volume and rate metrics.
        "bytes":     total_bytes, "sbytes": total_bytes, "dbytes": 0,
        "pkts":      pkts,        "spkts":  pkts,        "dpkts":  0,
        "dur":       dur,
        "rate":      rate,        "srate":  rate,        "drate":  0,
        "mean":      pd.Series(lengths).mean()              if lengths else 0,
        "stddev":    pd.Series(lengths).std()               if len(lengths) > 1 else 0,
        "min":       min(lengths)                           if lengths else 0,
        "max":       max(lengths)                           if lengths else 0,
        "sum":       total_bytes,
        # Protocol flags are derived from the IP protocol number (6=TCP, 17=UDP).
        "proto_tcp": 1 if "6"  in protos else 0,
        "proto_udp": 1 if "17" in protos else 0,
        "proto_arp": 0,
        # Only the first destination port is captured; this is a simplification
        # acceptable for single-flow container pcap captures.
        "dport":     dports[0] if dports else 0,
        "seq":       0,
    }


rows = []
print(f"Scanning pcap files in: {RUNTIME_LOGS_DIR}")
for pcap in sorted(RUNTIME_LOGS_DIR.iterdir()):
    if pcap.suffix != ".pcap":
        continue
    print(f"  Processing: {pcap.name}")
    feats         = extract_features(pcap)
    feats["file"] = pcap.name
    feats["label"] = 1 if "mal_" in pcap.name else 0
    rows.append(feats)

if not rows:
    raise RuntimeError(f"No .pcap files found in {RUNTIME_LOGS_DIR}")

df = pd.DataFrame(rows).fillna(0)

out_path = RUNTIME_NET_DIR / "runtime_network_raw.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")
print(df.head())
