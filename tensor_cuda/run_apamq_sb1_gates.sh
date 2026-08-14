#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m pytest "$HERE/tests/test_apamq_sb1.py" -q -rs
