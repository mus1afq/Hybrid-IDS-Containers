#!/bin/bash
# Shell script to run all validation tests sequentially for the dissertation

echo "============================================="
echo "   RUNNING ALL DISSERTATION VALIDATION TESTS"
echo "============================================="

# Ensure we use an environment with pandas installed if needed
# You can customize this python path to your active virtual environment
PYTHON_BIN="python3"

# Check if the temporary venv we used earlier exists and use it if so
if [ -f "/tmp/fyp_venv/bin/python3" ]; then
    PYTHON_BIN="/tmp/fyp_venv/bin/python3"
fi

echo ""
echo "--- 1. Syscall Original Dataset Validation (15 Samples) ---"
$PYTHON_BIN tests/validate_syscall_dataset.py

echo ""
echo "--- 2. Network Original Dataset Validation (15 Samples) ---"
$PYTHON_BIN tests/validate_network_dataset.py

echo ""
echo "--- 3. Syscall Synthetic Dataset Validation (Full File + Plots) ---"
$PYTHON_BIN tests/validate_syscall_synthetic.py

echo ""
echo "--- 4. Network Synthetic Dataset Validation (Full File + Plots) ---"
$PYTHON_BIN tests/validate_network_synthetic.py

echo ""
echo "--- 5. Hybrid Logic Validation (5 Controlled Cases) ---"
$PYTHON_BIN tests/validate_hybrid.py

echo ""
echo "--- 6. Hybrid Synthetic Dataset Validation (580 Samples) ---"
$PYTHON_BIN tests/validate_hybrid_synthetic.py

echo ""
echo "--- 7. End-to-End Runtime Validation (6 Containers) ---"
$PYTHON_BIN tests/validate_end_to_end.py

echo ""
echo "============================================="
echo "   ALL TESTS COMPLETED"
echo "============================================="