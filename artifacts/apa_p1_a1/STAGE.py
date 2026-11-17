"""Freeze CPU-rehearsable GPU lines after immutable prediction registration.
Prior art: P1 staging (2026), forge-gpu-run window leases (SystemOps, 2026),
SHA-256 (NIST, 2015). Configuration and exact coverage; no new algorithm.
"""
from gpu_common import *
from schema import payload_bytes
import shlex,sysconfig,datetime
A=ART;reg=load(A/'REGISTRATION.json');digest=sha(A/'REGISTRATION.json')
cases=load(A/'INPUT_CENSUS.json');files=reg['regression_files'];commands=[]
def add(label,args,payload):
 cmd=['timeout','590s','forge-gpu-run','--min','120','--max','600','--label','APA-P1-A1-'+label,'--','timeout','540s','env','APA_P1_A1_LEAD_GPU=1','python3','artifacts/apa_p1_a1/RUN.py',*args]
 line='run_line '+label+' '+shlex.join(cmd)+' --out "$RUN/'+label+'" || exit 2'
 header='# registration_sha256='+digest+'; writes '+payload
 commands.append(dict(label=label,args=args,max_seconds=600,inner_timeout_seconds=540,outer_timeout_seconds=590,header=header,line=line,writes=payload))
for c in cases:add(c['id'],['bitwise','--case',c['id']],f"{2*payload_bytes(c)} raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/{c['id']}/")
add('bytes',['bytes'],'no numerical arrays; six full-cache JSON receipts plus thirteen arena-layout comparisons and logs; target $RUN/bytes/ (expected under 2 MiB)')
for group in ('apa','dlpack'):
 for f in files[group]:add('reg_'+Path(f).stem.removeprefix('test_'),['regression','--test-file',f],'per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_'+Path(f).stem.removeprefix('test_')+'/')
script='''#!/usr/bin/env bash
# APA-P1 amendment A1. Lead: CUDA_VISIBLE_DEVICES=0 timeout 86400s bash artifacts/apa_p1_a1/STAGED_GPU_LINES.sh
# CPU seat: same lines via recorded local forge shim, APA_P1_A1_CPU_REHEARSAL=1, CUDA hidden.
# Prior art: P1/AB-1 staging and SystemOps forge-gpu-run duty-cycle legs (2026).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/tmp/apa_p1_a1_lead CUDA_CACHE_DISABLE=1
timeout 10s mkdir -p "$TMPDIR" || exit 2
RUN="${APA_P1_A1_RUN_ROOT:-artifacts/apa_p1_a1/gpu_$(date -u +%Y%m%dT%H%M%SZ)}"
timeout 10s mkdir "$RUN" || exit 2
FAIL=0
run_line() {
 local label="$1"; shift
 "$@" > "$RUN/$label.log" 2>&1
 local code=$?
 timeout 15s python3 artifacts/apa_p1_a1/record_gpu_exit.py "$RUN" "$label" "$code" "$@" || return 2
 if [[ "$code" -ne 0 ]]; then FAIL=1; fi
 printf '%s exit=%s receipts=%s/%s\\n' "$label" "$code" "$RUN" "$label"
}
cooldown() {
 if [[ "${APA_P1_A1_CPU_REHEARSAL:-}" != 1 ]]; then
  # Foreground cooldown reaches 600 s; timeout's exit 124 is expected at its boundary.
  timeout 600s sleep 600
  local code=$?
  [[ "$code" -eq 0 || "$code" -eq 124 ]] || return 2
 fi
}
'''
script+='# Immutable registration SHA-256: '+digest+'\n'
legs=[]
for i in range(0,len(commands),2):
 batch=commands[i:i+2];num=i//2+1
 script+=f'\n# Leg {num}: summed --max {sum(c["max_seconds"] for c in batch)} seconds.\n'
 if num>1:script+='cooldown || exit 2\n'
 for c in batch:script+=c['header']+'\n'+c['line']+'\n'
 legs.append(dict(leg=num,labels=[c['label'] for c in batch],sum_max_seconds=sum(c['max_seconds'] for c in batch),cooldown_before_seconds=600 if num>1 else 0))
script+='printf \'A1 verdict exit=%s; receipts=%s\\n\' "$FAIL" "$RUN"\nexit "$FAIL"\n'
with (A/'STAGED_GPU_LINES.sh').open('x') as f:f.write(script)
save(A/'LEGS.json',dict(registration_sha256=digest,legs=legs,line_count=len(commands),cooldown_seconds=600,maximum_leg_seconds=1240))
save(A/'STAGED_LINES.json',dict(registration_sha256=digest,commands=commands))
# Pin the full original source closure plus A1 handlers, packages and adapter inputs.
old=load(ROOT/'artifacts/apa_p1/EXPECTED.json')
code={ROOT/p for p in old['code_sha256']}
code.update(A/p for p in ['RUN.py','gpu_common.py','gpu_worker.py','worker_core.py','layout.py','schema.py','pytest_receipt.py','record_gpu_exit.py','build.json','interop_build.json','INPUT_CENSUS.json','reference_g2.json','STAGED_GPU_LINES.sh','STAGED_LINES.json','LEGS.json'])
code.update((A/'refs').rglob('*.py'))
code.add(ROOT/'../tensor_fork_apa_wt_apa2p/artifacts/apa_ds1_a3/recurrence.py')
code.update(ROOT/p for p in reg['reference']['A3_source_sha256'])
labinterop=old['lab_interop'];labinterop['path']=os.path.relpath(labinterop['path'],ROOT)
exp=dict(registration_sha256=digest,code_sha256={os.path.relpath(p,ROOT):sha(p) for p in sorted(code)},gpu_test_files=files,gpu_collected_ids=old['gpu_collected_ids'],lab_interop=labinterop,cases=cases,commands=commands,source='Immutable A1 registration; original P1 collected IDs and real captures retained',status='STAGED_NOT_RUN')
save(A/'EXPECTED.json',exp)
print(dict(lines=len(commands),bitwise=len(cases),regression=sum(map(len,files.values())),legs=len(legs),registration_sha256=digest))
