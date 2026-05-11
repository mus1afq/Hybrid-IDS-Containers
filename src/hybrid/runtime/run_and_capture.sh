#!/usr/bin/env bash
# =============================================================================
# run_and_capture.sh
# Phase 1: Launch Docker containers and capture syscall traces via sysdig.
#
# USAGE:  bash src/hybrid/runtime/run_and_capture.sh
#
# OUTPUT (written to data/runtime/syscall/):
#   sysdig_raw.txt        – raw "<container> <timestamp_sec> <syscall>" lines
#   capture_manifest.csv  – container,label,start_ts,end_ts
#
# Requirements: Docker Desktop running on macOS.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"   # FYP project root
OUTDIR="${ROOT_DIR}/data/runtime/syscall"

mkdir -p "${OUTDIR}"

RAW="${OUTDIR}/sysdig_raw.txt"
MANIFEST="${OUTDIR}/capture_manifest.csv"
CAPTURE_SECONDS=15

# ---------------------------------------------------------------------------
# Container definitions: name, label, image, command
# ---------------------------------------------------------------------------
declare -a NAMES=( redis nginx ubuntu-idle python-ws alpine-idle
                   file-flood rename-flood port-scan conn-burst privesc-sim )
declare -a LABELS=( 0 0 0 0 0 1 1 1 1 1 )
declare -a IMAGES=(
  "redis:7-alpine"
  "nginx:alpine"
  "ubuntu:22.04"
  "python:3.11-alpine"
  "alpine:latest"
  "alpine:latest"
  "alpine:latest"
  "alpine:latest"
  "alpine:latest"
  "alpine:latest"
)
declare -a CMDS=(
  ""
  ""
  "sleep 20"
  "python3 -m http.server 8080"
  "sleep 20"
  # ---------------------------------------------------------------------------
  # ETHICAL NOTE: The following commands simulate malicious container behaviour
  # within isolated Docker containers on the local host only. They do not
  # exploit real vulnerabilities, make outbound network requests to external
  # targets, or persist any changes outside the container lifetime. All
  # containers are cleaned up immediately after the capture window.
  # ---------------------------------------------------------------------------
  "sh -c 'for i in $(seq 1 8000); do echo x > /tmp/f$i; done; sleep 3'"
  "sh -c 'touch /tmp/a; for i in $(seq 1 4000); do mv /tmp/a /tmp/b; mv /tmp/b /tmp/a; done; sleep 3'"
  "sh -c 'for i in $(seq 1 200); do (nc -z -w1 172.17.0.1 $((RANDOM%1000+1)) 2>/dev/null); done; sleep 3'"
  "sh -c 'for i in $(seq 1 150); do nc -z -w1 172.17.0.1 80 2>/dev/null; done; sleep 3'"
  "sh -c 'id; whoami; cat /etc/shadow 2>/dev/null || true; cat /proc/1/environ 2>/dev/null || true; ls /root 2>/dev/null || true; sleep 3'"
)

# ---------------------------------------------------------------------------
# Clean up any leftover containers
# ---------------------------------------------------------------------------
echo "[*] Cleaning up any leftover containers..."
for NAME in "${NAMES[@]}"; do
  docker rm -f "${NAME}" 2>/dev/null || true
done

# ---------------------------------------------------------------------------
# Start sysdig capture container
# ---------------------------------------------------------------------------
echo "[*] Starting sysdig capture container..."
docker rm -f sysdig-capture 2>/dev/null || true

SYSDIG_FMT='%container.name %evt.rawtime.s %syscall.type'
SYSDIG_FILTER="container.name in ($(IFS=,; echo "${NAMES[*]}"))"

SYSDIG_UP=0
docker run -d --name sysdig-capture \
  --privileged --pid=host --net=host \
  -v /var/run/docker.sock:/host/var/run/docker.sock \
  -v /dev:/host/dev \
  -v /proc:/host/proc:ro \
  -v "${OUTDIR}:/output" \
  docker/sysdig \
  sysdig -p "${SYSDIG_FMT}" "${SYSDIG_FILTER}" > /dev/null 2>&1 && SYSDIG_UP=1 || {
    echo "[!] WARNING: sysdig container failed — falling back to synthetic mode."
    echo "SYSDIG_UNAVAILABLE" > "${RAW}"
  }

if [ "${SYSDIG_UP}" -eq 1 ]; then
  sleep 3
  docker logs -f sysdig-capture >> "${RAW}" 2>/dev/null &
  SYSDIG_PID=$!
fi

echo "container,label,start_ts,end_ts" > "${MANIFEST}"

# ---------------------------------------------------------------------------
# Run each container
# ---------------------------------------------------------------------------
for i in "${!NAMES[@]}"; do
  NAME="${NAMES[$i]}"
  LABEL="${LABELS[$i]}"
  IMAGE="${IMAGES[$i]}"
  CMD="${CMDS[$i]}"

  echo ""
  echo "[*] Launching: ${NAME}  (label=${LABEL})"
  START_TS=$(date +%s)

  if [ -z "${CMD}" ]; then
    docker run -d --name "${NAME}" "${IMAGE}" > /dev/null
  else
    docker run -d --name "${NAME}" "${IMAGE}" ${CMD} > /dev/null
  fi

  sleep "${CAPTURE_SECONDS}"
  END_TS=$(date +%s)
  echo "${NAME},${LABEL},${START_TS},${END_TS}" >> "${MANIFEST}"
  echo "  => Captured ${NAME}: ${START_TS} – ${END_TS}"
  docker rm -f "${NAME}" 2>/dev/null || true
  sleep 2
done

# ---------------------------------------------------------------------------
# Stop sysdig
# ---------------------------------------------------------------------------
if [ "${SYSDIG_UP}" -eq 1 ]; then
  sleep 2
  kill "${SYSDIG_PID}" 2>/dev/null || true
  docker rm -f sysdig-capture 2>/dev/null || true
  echo ""
  echo "[+] Capture complete → ${RAW}"
  echo "    Total events: $(wc -l < "${RAW}")"
else
  echo "[!] Sysdig unavailable — synthetic fallback will be used in step 2."
fi

echo "[+] Manifest → ${MANIFEST}"
echo ""
echo "Next step: python3 src/hybrid/runtime/extract_syscall_features.py"
