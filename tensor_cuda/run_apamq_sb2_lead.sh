#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"

PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest "$HERE/tests/test_apamq_sb1.py" -q -rs

for cell in \
  "standard 1 512 prefill 16384" \
  "gemm_apa 1 512 prefill 16384" \
  "gemm_apa 1 512 prefill 65536" \
  "gemm_apa 1 512 decode 65536"
do
  # shellcheck disable=SC2086
  PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" \
    python3 "$ROOT/scripts/apamq_e1_sweep.py" --cell $cell
done
