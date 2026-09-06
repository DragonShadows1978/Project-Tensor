#!/usr/bin/env bash
# CPU-only foreground verification. Does not create a CUDA context.
set -euo pipefail
if [[ "${APA_SP1_1_CPU_BOUNDED:-0}" != 1 ]]; then
  exec timeout --signal=TERM --kill-after=5s 300s env APA_SP1_1_CPU_BOUNDED=1 bash "$0" "$@"
fi
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
[[ "$ROOT" == /mnt/ForgeRealm/Project-Tensor-wt-apa-sp1 ]] || exit 64
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$ROOT/artifacts/apa_sp1/build:$ROOT/tensor_cuda"
if [[ "${1:-unit}" == diagnose ]]; then
  exec timeout 240s python3 scripts/apa_sp1_1_diagnose.py
fi
[[ "${1:-unit}" == unit ]] || exit 64
stamp=$(timeout 5s python3 -c 'import time;print(time.time_ns())')
log="artifacts/apa_sp1/sp1_1_g1_${stamp}.log"
set +e
timeout 120s python3 -m pytest -q -p no:cacheprovider --tb=short \
  tensor_cuda/tests/test_apa_sp1_reference.py tensor_cuda/tests/test_apa_sp1_host.py \
  tensor_cuda/tests/test_apa_sp1_harness.py tensor_cuda/tests/test_apa_sp1_1.py >"$log" 2>&1
rc=$?
set -e
timeout 30s python3 - "$log" "$rc" <<'PY'
import json,sys
from pathlib import Path
sys.path.insert(0,'scripts')
from apa_sp1_cpu import sha
from apa_sp1_gpu import load_runtime
log=Path(sys.argv[1]);rc=int(sys.argv[2]);tc=load_runtime()
r=dict(evidence_class='CPU tests and host module provenance, no GPU execution',status='PASS' if rc==0 else 'FAIL',
       returncode=rc,log=str(log),summary=log.read_text().splitlines()[-1],runtime_module=str(tc._C.__file__),
       registration_sha256=sha('artifacts/apa_sp1/registration_sp1_1.json'),
       build_manifest_sha256=sha('artifacts/apa_sp1/build/manifest.json'),
       tests={str(p):sha(p) for p in Path('tensor_cuda/tests').glob('*apa_sp1*.py')},
       scope_note='this establishes nothing about model quality')
with log.with_suffix('.json').open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r,indent=2))
PY
exit "$rc"
