# Prior art: AB-1 expected-value receipts (2026), SHA-256 NIST 2015;
# forge-gpu-run (SystemOps) bounded lease wrappers. Configuration, no algorithm.
from pathlib import Path
import json,hashlib,datetime,sysconfig,os
A=Path('artifacts/apa_p1');root=Path.cwd()
load=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files=load(A/'GPU_TEST_FILES.json');ids=load(A/'gpu_collection_cpu.json')['collection'];cases=load(A/'INPUT_CENSUS.json')
script='''#!/usr/bin/env bash
# APA-P1: lead-only G4. Run from worktree root: bash artifacts/apa_p1/STAGED_GPU_LINES.sh
# Prior art: SystemOps forge-gpu-run window lease; AB-1 (2026) exact comparison/exit receipts.
set -uo pipefail
ROOT=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod
cd "$ROOT" || exit 2
export TMPDIR=/tmp/apa_p1_lead PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# GPU visibility is enabled only INSIDE each forge-gpu-run lease.
export CUDA_VISIBLE_DEVICES=''
timeout 10s mkdir -p "$TMPDIR" || exit 2
RUN="artifacts/apa_p1/gpu_$(date -u +%Y%m%dT%H%M%SZ)"
timeout 10s mkdir "$RUN" || exit 2
FAIL=0
run_line() {
  local label="$1"; shift
  # Stop before each line below the order's free-space rails.
  timeout 10s python -c 'import sys;sys.path.insert(0,"artifacts/apa_p1");from gpu_common import disk;disk()' || return 2
  "$@" > "$RUN/$label.log" 2>&1
  local code=$?
  # Only after the GPU command exits: record its exact exit, argv and output hash.
  timeout 15s python artifacts/apa_p1/record_gpu_exit.py "$RUN" "$label" "$code" "$@" || return 2
  if [[ "$code" -ne 0 ]]; then FAIL=1; fi
  printf '%s exit=%s receipts=%s/%s\n' "$label" "$code" "$RUN" "$label"
}
'''
commands=[]
def add(label,args):
    global script
    line=f'run_line {label} timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-{label} -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py {args} --out "$RUN/{label}" || exit 2'
    script+=line+'\n';commands.append(dict(label=label,line=line,expected='PASS; exact equality',max_seconds=600,inner_timeout_seconds=540,outer_timeout_seconds=1750))
for c in cases:
    short='17b' if c['model']=='Qwen3-1.7B' else '4b';add(f'bitwise_{short}_{c["S"]}',f'bitwise --model {c["model"]} --S {c["S"]}')
add('bytes','bytes')
for group in ('apa','dlpack'):
    for f in files[group]:
        label='reg_'+Path(f).stem.removeprefix('test_');add(label,f'regression --test-file {f}')
script+='printf \'GPU verdict exit=%s; receipts=%s\\n\' "$FAIL" "$RUN"\nexit "$FAIL"\n'
(A/'STAGED_GPU_LINES.sh').write_text(script)
# Pinned numerical/source census; reports and mutable ledgers are deliberately not source inputs.
code=[A/x for x in ['gpu_common.py','gpu_worker.py','gpu_driver.py','record_gpu_exit.py','pytest_receipt.py','STAGED_GPU_LINES.sh','build.json','interop_build.json','INPUT_CENSUS.json','reference_g2.json','reference_g0_build.json']]
code+=sorted((A/'refs').rglob('*.py'))
code+=sorted(Path('tensor_cuda/tensor_cuda').rglob('*.py'))
code+=sorted(Path('tensor_cuda/src/apa_spo').glob('*.cuh'))
code+=[Path(f) for f in files['apa']+files['dlpack']]
code+=[Path(p) for p in ('tensor_cuda/src/kernels.cu','tensor_cuda/src/bindings.cpp','tensor_cuda/include/tc/ops.h','tensor_cuda/CMakeLists.txt')]
B=load(A/'build.json');inter=Path(B['lab_module']).parents[1]/'interop'/('_apa_ab1_interop'+sysconfig.get_config_var('EXT_SUFFIX'))
assert inter.is_file()
expected=dict(registered=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='STAGED_NOT_RUN',predictions=dict(P142='all staged bitwise cases exact; score only after lead runs',P143='all production CPU/GPU per-ID outcomes identical; CPU part observed, GPU pending'),numeric_rule='All elements compared through unsigned integer views; no tolerance. Same real Q/K/V captures and exact input-upload bit checks, six cases, both SP/EB3 and SP-tk8/EB4 plus TQ4 packed append histories.',cases=cases,reference_nonvacuity='Every saved integer-view reference array must be nonzero and nonconstant; failure is RED, no waiver. Append valid_rows history must vary.',byte_rule=dict(packed=580,layer0=512,units='bytes per key per KV head',formula='KVH*S*(512+(NL-1)*580)',SP_tk8_note='712 B/key/head incl. int8 V and scale; 580-byte acceptance is SP only, matching AB-1 G2.',expected_totals={model:{str(S):8*S*(512+(NL-1)*580) for S in (2048,8192,32768)} for model,NL in [('Qwen3-1.7B',28),('Qwen3-4B',36)]},receipt='reference_g2.json',exclusions='Shared non-token rotation/codebook tables and transient decode scratch; not an NVML slope claim.'),gpu_test_files=files,gpu_collected_ids=ids,production_outcome_reference='Exact paired GPU baseline is produced in each staged regression line. G3 CPU baseline receipt is g3_production.json; its outcome strings are retained but CPU skips/failures are not substituted for a GPU baseline.',g3_baseline_sha256=sha(A/'g3_production.json'),baselines=dict(production_module_sha256=B['production_sha256'],lab_module_sha256=B['lab_sha256'],port_module_sha256=B['module_sha256']),lab_interop=dict(path=str(inter),sha256=sha(inter)),code_sha256={str(p):sha(p) for p in sorted(set(code))},commands=commands,write_paths=['artifacts/apa_p1/gpu_<UTC>/**','/tmp/apa_p1_lead/**'],disk_rails_bytes=dict(ForgeRealm=25_000_000_000,root=20_000_000_000),scope='Staged kernel identity, production test outcome parity, payload storage arithmetic. No quality/performance/default/promotion/red-team/KV-pitch certification.')
assert not (A/'EXPECTED.json').exists();(A/'EXPECTED.json').write_text(json.dumps(expected,indent=2)+'\n')
(A/'G4_REGISTRATION.md').write_text(f'''# G4 staged registration (immutable, no device data collected)\n\n{len(commands)} lease-wrapped lines: six real captured numerical cases (two models, three S), one bytes line (six full caches), and {len(files['apa'])+len(files['dlpack'])} production regression files ({len(ids)} frozen collected IDs).\n\nEvery line: foreground timeout 1750s; forge-gpu-run --min 120 --max 600; inner timeout 540s. Numerical child timeouts 240s per arm; byte child 80s per cache; regression child 240s per module. Commands, source/module SHA pins, exact expected arithmetic, complete IDs and nonvacuity guards are frozen in EXPECTED.json before the lead's first device execution. Each command exit/output is receipted only after it exits.\n\nNo seat GPU execution. Lead invokes STAGED_GPU_LINES.sh exactly as written from root. If the window/tool times out the line records that failure; it is never counted as equality. No scripts edit another run's receipts. GPU worker output creation is exclusive.\n''')
print('Staged',len(commands),'lines;',len(ids),'GPU test IDs; EXPECTED.json frozen.')
