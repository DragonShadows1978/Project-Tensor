#!/usr/bin/env bash
# APA-MF2.2 adjudication launcher for the copied re-review probe (gpu_probe.py, edits in
# logs/apa_mf2_2_probe_copy_edits.diff). usage: run.sh REV:CELL[:leaf] ...
# REV in fd32b6d | 048949d (MF2) | c5c468d (MF2.1) | mf2_2. One process per item, each under
# flock -w 900 /tmp/forge-gpu.lock timeout 600; native driver; refuses the 595.84 override.
# Prior art: flock(1)/timeout(1) (util-linux, coreutils). Mine: this launcher.
set -uo pipefail
W=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf2
B=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_mf21rev/artifacts/apa_mf21rev/baselines
P=$W/scripts/apa_mf2_2_probe
case ":${LD_LIBRARY_PATH:-}:${PATH}:${LD_PRELOAD:-}:" in *595.84*) echo "595.84 override present; refusing" >&2; exit 2;; esac
declare -A ENG=([fd32b6d]=$W/artifacts/apa_mf2_2/build_fd32b6d [048949d]=$W/artifacts/apa_mf2/build [c5c468d]=$W/artifacts/apa_mf2_1/build [mf2_2]=$W/artifacts/apa_mf2_2/build)
declare -A PKG=([fd32b6d]=$B/fd32b6d/pkg [048949d]=$B/048949d/pkg [c5c468d]=$B/c5c468d/pkg [mf2_2]=$W/tensor_cuda)
export APA_MF21REV_BEFORE_ENGINE=${ENG[048949d]} APA_MF21REV_PRODUCTION_ENGINE=${ENG[fd32b6d]} APA_MF21REV_AFTER_ENGINE=${ENG[c5c468d]} APA_MF2_2_ENGINE_MF2_2=${ENG[mf2_2]}
export APA_MF2_2_PKGS="{\"fd32b6d\":\"${PKG[fd32b6d]}\",\"048949d\":\"${PKG[048949d]}\",\"c5c468d\":\"${PKG[c5c468d]}\",\"mf2_2\":\"${PKG[mf2_2]}\"}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 TC_ATTN_QTILE=0
for item in "$@"; do
  IFS=: read -r REV CELL LEAF <<< "$item"
  STAMP=$(date -u +%Y%m%dT%H%M%SZ)_${REV}_${LEAF:-orig}_$$
  LOG=$W/logs/apa_mf2_2_probe_${REV}_${CELL}_${LEAF:-orig}_$STAMP.log
  ( export APA_MF21REV_REV=$REV APA_MF21REV_ENGINE=${ENG[$REV]} APA_MF21REV_PKG=${PKG[$REV]} APA_MF21REV_RUN=$STAMP
    export PYTHONPATH=${ENG[$REV]}:${PKG[$REV]}
    [[ ${LEAF:-} == leaf ]] && export APA_MF2_2_BF16_LEAF=1 APA_MF2_2_SKIP_BF16_POOL=1
    [[ -n ${IDREVS:-} ]] && export APA_MF2_2_IDENTITY_REVS=$IDREVS
    flock -w 900 /tmp/forge-gpu.lock timeout 600 python3 $P/gpu_probe.py "$CELL" > "$LOG" 2>&1 )
  RC=$?
  python3 - "$LOG" "$RC" "$item" <<'PY'
import json,sys
log,rc,item=sys.argv[1],sys.argv[2],sys.argv[3]; n=f=0; bad=[]
for line in open(log,errors='replace'):
    try: d=json.loads(line)
    except Exception: continue
    if 'name' in d:
        n+=1
        if not d['passed']: f+=1; bad.append(d['name'])
print(f"{item:45s} exit={rc} rows={n} failed={f} {bad[:12]} log={log.split('/')[-1]}")
PY
done
