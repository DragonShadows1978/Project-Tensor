#!/usr/bin/env bash
# APA-MF2.2c: launcher for the copied review probe (one process per ARM:CELL item), each under
# flock -w 900 /tmp/forge-gpu.lock timeout 600; native driver; refuses the 595.84 override.
set -uo pipefail
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
case ":${LD_LIBRARY_PATH:-}:${PATH}:${LD_PRELOAD:-}:" in *595.84*) echo "595.84 override present; refusing" >&2; exit 2;; esac
[[ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)" == 595.91.07 ]] || { echo "driver pin mismatch" >&2; exit 2; }
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TC_FUSED_SDPA_NONCAUSAL=0
for item in "$@"; do
  IFS=: read -r ARM CELL <<< "$item"
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)_${ARM}_$$
  LOG=$W/logs/apa_mf2_2c_probe_${ARM}_${CELL}_$STAMP.log
  ( export APA_MF22REV_ARM=$ARM APA_MF22REV_RUN=$STAMP
    flock -w 900 /tmp/forge-gpu.lock timeout 600 python3 $W/scripts/apa_mf2_2c_probe/gpu_probe.py "$CELL" > "$LOG" 2>&1 )
  RC=$?
  python3 - "$LOG" "$RC" "$item" <<'PY'
import json,sys
log,rc,item=sys.argv[1:4];n=f=0;bad=[]
for line in open(log,errors='replace'):
    try:d=json.loads(line)
    except Exception:continue
    if 'name' in d:
        n+=1
        if not d['passed']:f+=1;bad.append(d['name'])
print(f"{item:28s} exit={rc} rows={n} failed={f} {bad[:8]} log={log.split('/')[-1]}")
PY
done
