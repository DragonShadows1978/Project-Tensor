#!/bin/bash
# ENG3: the steps of one GPU line, run inside forge-gpu-run (from the worktree root). Each step is
#   suite:<label>:<g1|g2>      production suite half (artifacts/eng3/g3/SPLIT.json) through scripts/eng3_suite.py
#   own:<label>[:<-k expr>]    tensor_cuda/tests/test_eng3_host.py (G1, G1b, G2) with records
#   devpath:<label>:<mode>     scripts/eng3_devpath.py
# ENG3_EXPECT (the module sha256) must be set; every step re-checks it. Exit = number of steps whose tool failed
# (test failures are outcomes, recorded per ID, not tool failures). Prior art: none.
set -u
cd "$(dirname "$0")/.."
export TMPDIR=/tmp/eng3_tmp
fails=0
for step in "$@"; do
  IFS=: read -r kind label arg <<< "$step"
  echo "=== step $step $(date -Is)"
  case $kind in
    suite)
      files=$(python3 -B -P -c "import json;print(' '.join(json.load(open('artifacts/eng3/g3/SPLIT.json'))['$arg']))")
      ENG3_LABEL=$label ENG3_FILES="$files" python3 -B -P scripts/eng3_suite.py || fails=$((fails+1)) ;;
    own)
      ENG3_LABEL=$label ENG3_OWN_TESTS=1 ENG3_RECORD=1 ENG3_FILES=tensor_cuda/tests/test_eng3_host.py ENG3_K="${arg:-}" \
        python3 -B -P scripts/eng3_suite.py || fails=$((fails+1)) ;;
    devpath)
      python3 -B -P scripts/eng3_devpath.py "$label" "$arg" || fails=$((fails+1)) ;;
    *) echo "unknown step $step"; fails=$((fails+1)) ;;
  esac
done
echo "=== steps done, tool failures: $fails $(date -Is)"
exit $fails
