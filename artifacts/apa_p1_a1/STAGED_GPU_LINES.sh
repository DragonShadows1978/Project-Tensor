#!/usr/bin/env bash
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
 printf '%s exit=%s receipts=%s/%s\n' "$label" "$code" "$RUN" "$label"
}
cooldown() {
 if [[ "${APA_P1_A1_CPU_REHEARSAL:-}" != 1 ]]; then
  # Foreground cooldown reaches 600 s; timeout's exit 124 is expected at its boundary.
  timeout 600s sleep 600
  local code=$?
  [[ "$code" -eq 0 || "$code" -eq 124 ]] || return 2
 fi
}
# Immutable registration SHA-256: d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f

# Leg 1: summed --max 1200 seconds.
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 40091280 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_17b_2048/
run_line bitwise_17b_2048 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_17b_2048 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_17b_2048 --out "$RUN/bitwise_17b_2048" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 156913296 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_17b_8192/
run_line bitwise_17b_8192 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_17b_8192 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_17b_8192 --out "$RUN/bitwise_17b_8192" || exit 2

# Leg 2: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 624201360 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_17b_32768/
run_line bitwise_17b_32768 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_17b_32768 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_17b_32768 --out "$RUN/bitwise_17b_32768" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 41784720 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_4b_2048/
run_line bitwise_4b_2048 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_4b_2048 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_4b_2048 --out "$RUN/bitwise_4b_2048" || exit 2

# Leg 3: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 163343760 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_4b_8192/
run_line bitwise_4b_8192 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_4b_8192 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_4b_8192 --out "$RUN/bitwise_4b_8192" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 649579920 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_4b_32768/
run_line bitwise_4b_32768 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_4b_32768 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_4b_32768 --out "$RUN/bitwise_4b_32768" || exit 2

# Leg 4: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 750136 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_B1_H2_KV1_S255/
run_line bitwise_B1_H2_KV1_S255 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_B1_H2_KV1_S255 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_B1_H2_KV1_S255 --out "$RUN/bitwise_B1_H2_KV1_S255" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 1509624 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_B1_H4_KV2_S257/
run_line bitwise_B1_H4_KV2_S257 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_B1_H4_KV2_S257 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_B1_H4_KV2_S257 --out "$RUN/bitwise_B1_H4_KV2_S257" || exit 2

# Leg 5: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 1378552 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_B2_H2_KV1_S257/
run_line bitwise_B2_H2_KV1_S257 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_B2_H2_KV1_S257 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_B2_H2_KV1_S257 --out "$RUN/bitwise_B2_H2_KV1_S257" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 2756832 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_B2_H4_KV2_S257/
run_line bitwise_B2_H4_KV2_S257 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_B2_H4_KV2_S257 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_B2_H4_KV2_S257 --out "$RUN/bitwise_B2_H4_KV2_S257" || exit 2

# Leg 6: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 752512 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_B1_H2_KV1_S256/
run_line bitwise_B1_H2_KV1_S256 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_B1_H2_KV1_S256 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_B1_H2_KV1_S256 --out "$RUN/bitwise_B1_H2_KV1_S256" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 754948 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_B1_H2_KV1_S257/
run_line bitwise_B1_H2_KV1_S257 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_B1_H2_KV1_S257 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_B1_H2_KV1_S257 --out "$RUN/bitwise_B1_H2_KV1_S257" || exit 2

# Leg 7: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes 2737584 raw unsigned payload bytes across two retained NPZ files, plus ZIP headers and JSON/log receipts; target $RUN/bitwise_B2_H4_KV2_S255/
run_line bitwise_B2_H4_KV2_S255 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bitwise_B2_H4_KV2_S255 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bitwise --case bitwise_B2_H4_KV2_S255 --out "$RUN/bitwise_B2_H4_KV2_S255" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes no numerical arrays; six full-cache JSON receipts plus thirteen arena-layout comparisons and logs; target $RUN/bytes/ (expected under 2 MiB)
run_line bytes timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-bytes -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py bytes --out "$RUN/bytes" || exit 2

# Leg 8: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_int4_sdpa_noncausal/
run_line reg_apa_int4_sdpa_noncausal timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_int4_sdpa_noncausal -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_int4_sdpa_noncausal.py --out "$RUN/reg_apa_int4_sdpa_noncausal" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_ld1_pdwl/
run_line reg_apa_ld1_pdwl timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_ld1_pdwl -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_ld1_pdwl.py --out "$RUN/reg_apa_ld1_pdwl" || exit 2

# Leg 9: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_ld2_pdwl/
run_line reg_apa_ld2_pdwl timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_ld2_pdwl -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_ld2_pdwl.py --out "$RUN/reg_apa_ld2_pdwl" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf1_quant/
run_line reg_apa_mf1_quant timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf1_quant -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf1_quant.py --out "$RUN/reg_apa_mf1_quant" || exit 2

# Leg 10: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf2_1_definition/
run_line reg_apa_mf2_1_definition timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf2_1_definition -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf2_1_definition.py --out "$RUN/reg_apa_mf2_1_definition" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf2_1_gaps/
run_line reg_apa_mf2_1_gaps timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf2_1_gaps -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf2_1_gaps.py --out "$RUN/reg_apa_mf2_1_gaps" || exit 2

# Leg 11: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf2_1_sentinel/
run_line reg_apa_mf2_1_sentinel timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf2_1_sentinel -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf2_1_sentinel.py --out "$RUN/reg_apa_mf2_1_sentinel" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf2_1_short/
run_line reg_apa_mf2_1_short timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf2_1_short -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf2_1_short.py --out "$RUN/reg_apa_mf2_1_short" || exit 2

# Leg 12: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf2_2_n1/
run_line reg_apa_mf2_2_n1 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf2_2_n1 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf2_2_n1.py --out "$RUN/reg_apa_mf2_2_n1" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf2_2_n2/
run_line reg_apa_mf2_2_n2 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf2_2_n2 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf2_2_n2.py --out "$RUN/reg_apa_mf2_2_n2" || exit 2

# Leg 13: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf2_2a_blocked/
run_line reg_apa_mf2_2a_blocked timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf2_2a_blocked -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf2_2a_blocked.py --out "$RUN/reg_apa_mf2_2a_blocked" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf2_2b_blocked/
run_line reg_apa_mf2_2b_blocked timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf2_2b_blocked -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf2_2b_blocked.py --out "$RUN/reg_apa_mf2_2b_blocked" || exit 2

# Leg 14: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf2_2c_longkeys/
run_line reg_apa_mf2_2c_longkeys timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf2_2c_longkeys -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf2_2c_longkeys.py --out "$RUN/reg_apa_mf2_2c_longkeys" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf2_2c_nits/
run_line reg_apa_mf2_2c_nits timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf2_2c_nits -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf2_2c_nits.py --out "$RUN/reg_apa_mf2_2c_nits" || exit 2

# Leg 15: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf2_mask/
run_line reg_apa_mf2_mask timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf2_mask -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf2_mask.py --out "$RUN/reg_apa_mf2_mask" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_mf2_rule/
run_line reg_apa_mf2_rule timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_mf2_rule -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_mf2_rule.py --out "$RUN/reg_apa_mf2_rule" || exit 2

# Leg 16: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_om135_imm/
run_line reg_apa_om135_imm timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_om135_imm -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_om135_imm.py --out "$RUN/reg_apa_om135_imm" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_phase6/
run_line reg_apa_phase6 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_phase6 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_phase6.py --out "$RUN/reg_apa_phase6" || exit 2

# Leg 17: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_pidv/
run_line reg_apa_pidv timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_pidv -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_pidv.py --out "$RUN/reg_apa_pidv" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_selective/
run_line reg_apa_selective timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_selective -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_selective.py --out "$RUN/reg_apa_selective" || exit 2

# Leg 18: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_selective_int4/
run_line reg_apa_selective_int4 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_selective_int4 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_selective_int4.py --out "$RUN/reg_apa_selective_int4" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_selective_splitk/
run_line reg_apa_selective_splitk timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_selective_splitk -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_selective_splitk.py --out "$RUN/reg_apa_selective_splitk" || exit 2

# Leg 19: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp1_1/
run_line reg_apa_sp1_1 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp1_1 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp1_1.py --out "$RUN/reg_apa_sp1_1" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp1_harness/
run_line reg_apa_sp1_harness timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp1_harness -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp1_harness.py --out "$RUN/reg_apa_sp1_harness" || exit 2

# Leg 20: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp1_host/
run_line reg_apa_sp1_host timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp1_host -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp1_host.py --out "$RUN/reg_apa_sp1_host" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp1_reference/
run_line reg_apa_sp1_reference timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp1_reference -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp1_reference.py --out "$RUN/reg_apa_sp1_reference" || exit 2

# Leg 21: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp2/
run_line reg_apa_sp2 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp2 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp2.py --out "$RUN/reg_apa_sp2" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp3/
run_line reg_apa_sp3 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp3 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp3.py --out "$RUN/reg_apa_sp3" || exit 2

# Leg 22: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp3_a4/
run_line reg_apa_sp3_a4 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp3_a4 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp3_a4.py --out "$RUN/reg_apa_sp3_a4" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp3_a5/
run_line reg_apa_sp3_a5 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp3_a5 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp3_a5.py --out "$RUN/reg_apa_sp3_a5" || exit 2

# Leg 23: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp3_a6/
run_line reg_apa_sp3_a6 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp3_a6 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp3_a6.py --out "$RUN/reg_apa_sp3_a6" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp4g/
run_line reg_apa_sp4g timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp4g -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp4g.py --out "$RUN/reg_apa_sp4g" || exit 2

# Leg 24: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp4g_a2/
run_line reg_apa_sp4g_a2 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp4g_a2 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp4g_a2.py --out "$RUN/reg_apa_sp4g_a2" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp4g_a3/
run_line reg_apa_sp4g_a3 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp4g_a3 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp4g_a3.py --out "$RUN/reg_apa_sp4g_a3" || exit 2

# Leg 25: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp4g_a4/
run_line reg_apa_sp4g_a4 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp4g_a4 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp4g_a4.py --out "$RUN/reg_apa_sp4g_a4" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp4g_a5/
run_line reg_apa_sp4g_a5 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp4g_a5 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp4g_a5.py --out "$RUN/reg_apa_sp4g_a5" || exit 2

# Leg 26: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp4g_a6/
run_line reg_apa_sp4g_a6 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp4g_a6 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp4g_a6.py --out "$RUN/reg_apa_sp4g_a6" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_sp4g_a7/
run_line reg_apa_sp4g_a7 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_sp4g_a7 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_sp4g_a7.py --out "$RUN/reg_apa_sp4g_a7" || exit 2

# Leg 27: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_spd1/
run_line reg_apa_spd1 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_spd1 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_spd1.py --out "$RUN/reg_apa_spd1" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_tiled/
run_line reg_apa_tiled timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_tiled -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_tiled.py --out "$RUN/reg_apa_tiled" || exit 2

# Leg 28: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apa_value_dim/
run_line reg_apa_value_dim timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apa_value_dim -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apa_value_dim.py --out "$RUN/reg_apa_value_dim" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apamq_df1/
run_line reg_apamq_df1 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apamq_df1 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apamq_df1.py --out "$RUN/reg_apamq_df1" || exit 2

# Leg 29: summed --max 1200 seconds.
cooldown || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_apamq_sb1/
run_line reg_apamq_sb1 timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_apamq_sb1 -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_apamq_sb1.py --out "$RUN/reg_apamq_sb1" || exit 2
# registration_sha256=d51c0f482b3a3f2380d5e7770bd1a0be5142e3b10d5dc78019cfb7a409566d3f; writes per-arm pytest JSON/logs, exact per-ID comparison and basetemp tree; size depends on outcomes (round-1 inventory retained); target $RUN/reg_dlpack/
run_line reg_dlpack timeout 590s forge-gpu-run --min 120 --max 600 --label APA-P1-A1-reg_dlpack -- timeout 540s env APA_P1_A1_LEAD_GPU=1 python3 artifacts/apa_p1_a1/RUN.py regression --test-file tensor_cuda/tests/test_dlpack.py --out "$RUN/reg_dlpack" || exit 2
printf 'A1 verdict exit=%s; receipts=%s\n' "$FAIL" "$RUN"
exit "$FAIL"
