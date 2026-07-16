#!/usr/bin/env bash
# Bounded PAINT-Q2-K0 GPU runner. The caller must hold the shared ColdCast
# flock around this script; keeping the lock external avoids double-flocking.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BASELINE_ROOT="${1:?usage: bash run_w8a16_gates.sh BASELINE_TENSOR_CUDA_ROOT}"
ARTIFACTS="$HERE/artifacts"
mkdir -p "$ARTIFACTS"

export PYTHONDONTWRITEBYTECODE=1

cd "$BASELINE_ROOT"
PYTHONPATH="$BASELINE_ROOT" python3 -m pytest -q tests \
  --ignore=tests/test_selector_accuracy.py \
  --junitxml="$ARTIFACTS/w8a16_suite_before.xml"
before_status=$?

cd "$HERE"
PYTHONPATH="$HERE" python3 -m pytest -q tests \
  --ignore=tests/test_selector_accuracy.py \
  --ignore=tests/test_w8a16_matmul.py \
  --junitxml="$ARTIFACTS/w8a16_suite_after.xml"
after_status=$?

PYTHONPATH="$HERE" python3 -m pytest -q tests/test_w8a16_matmul.py \
  --junitxml="$ARTIFACTS/w8a16_gate_tests.xml"
gate_test_status=$?

PYTHONPATH="$HERE" python3 tests/test_w8a16_matmul.py \
  --json "$ARTIFACTS/w8a16_gate_report.json"
report_status=$?

printf 'PAINT-Q2-K0 statuses: before=%d after=%d gate_tests=%d report=%d\n' \
  "$before_status" "$after_status" "$gate_test_status" "$report_status"

if (( after_status != 0 || gate_test_status != 0 || report_status != 0 )); then
  exit 1
fi
exit 0
