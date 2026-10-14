"""P1T-FIX4 G2: join the extracted call sites (sites.json) with the sweep annotations below and write sweep.json and
sweep_table.md. Every site must match exactly one rule (first match wins; an unmatched site fails the script), so the
table is complete by construction. Columns: device the call acts on; how that device is established (guard = switched
to and restored; operand = the operands'/tensor's device; proven current = read with a checked cudaGetDevice; host =
no device); how its return is handled; verdict (OK = correct in 62c8d9f and unchanged in substance; FIXED = changed by
FIX4, with the test that pins it); classes: (a) implicit current device, (b) unchecked/ignored return or error state.
No prior art claimed (a table join)."""
import json
import sys
from collections import Counter, OrderedDict

sys.path.insert(0, '.')
from funcs import enclosing  # noqa: E402

W = '/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_fix4/'
T = {  # tests (core: tensor_cuda/tests/test_dlpack.py; tc_signal: tc_signal/tests/test_fix4_findings.py)
    'v401': 'core test_fix4_v401_export_fence_* (7 cases) + mutant (a)',
    'v401f': 'core test_fix4_v401_export_fence_failure_* / create_or_wait_* / failed_restore_*',
    'label': 'core test_fix4_export_refuses_a_label_* / fill_refuses_a_label_* / export_label_proof_query_failure_*',
    'fill': 'core test_fix4_fill_launches_on_the_tensor_device_* / fill_guard_failures_* (3)',
    'fillpend': 'core test_fix4_fill_reports_a_pending_error_*',
    'drain': 'core test_fix4_storage_drain_failed_restore_is_counted_not_ignored',
    'v403': 'core test_fix4_v403_import_refuses_* (+ empty in no-device) + mutant (b); probe core_import_query_error.py',
    'empty': 'core test_fix4_empty_failed_allocation_is_reported_once / empty_labels_the_current_device_*',
    'v404': 'tcs test_fix4_v404_* (3) + mutant (c); probe sync_existing_error.py',
    'pre': 'tcs test_fix4_op_reports_a_pending_error_as_preexisting_*',
    'guard': 'tcs test_fix4_failed_switch_* / failed_restore_on_the_success_path_* / failed_restore_while_unwinding_*',
    'quiet': 'tcs test_fix4_failed_restore_where_nothing_can_raise_* (QuietDeviceGuard in ~RetainUntilComplete)',
    'dev': 'tcs test_fix4_device_rint_* / device_conv1d_* / device_lstm_* (pin: every logged call on the operand device, caller restored)',
    'checkcuda': 'tcs test_fix4_check_cuda_raise_consumes_the_failed_call',
    'destroy': 'tcs test_fix4_event_is_destroyed_on_its_own_device[ok, fail]',
    'query': 'tcs test_fix4_failed_event_query_in_the_drain_is_consumed_there',
    'cublas': 'tcs test_fix4_failed_cublas_setting_* (conv.cu handle) / failed_lstm_cublas_setting_* (lstm.cu handle)',
    'ccnone': 'no test feasible: this check_cuda copy only ever receives cudaGetLastError()\'s own result (already '
              'consumed; the extra take is a no-op) or, in build_info, calls that fail only with no usable device, where '
              'the runtime keeps cudaErrorNoDevice in the slot permanently (measured), so consumption is unobservable; '
              'the identical lstm.cu copy is tested (checkcuda)',
    'fix1': 'existing FIX1/FIX2 tests (test_fix1_f05_*, tcs test_fix2_findings N1/N3)',
}
HOST_STR = ('host only (error-string lookup)', 'host', 'returns a static string; nothing to check', 'OK', '')
CONSUME = 'error slot (calling thread; per runtime instance, not per device)'
R = [  # (file basename, function or None, call, text fragment or None, (device, established, handling, verdict, test))
    # ------------------------------------------------------------------ core kernels.cu
    ('kernels.cu', 'drain_device_legacy_stream', 'cudaGetDevice', None,
     ('host query (caller\'s current device)', 'proven current', 'checked: failure -> no release, deliberate leak (F05); slot entry consumed', 'OK', 'fix1')),
    ('kernels.cu', 'drain_device_legacy_stream', 'cudaSetDevice', 'e = cudaSetDevice(dev)',
     ('the Storage\'s device (label = producer device, checked = current at import)', 'guard (switch)', 'checked: failure -> leak; consumed', 'OK', 'fix1')),
    ('kernels.cu', 'drain_device_legacy_stream', 'cudaStreamSynchronize', None,
     ('legacy stream of the Storage\'s device', 'guard', 'checked: failure -> leak (no completion proof); consumed', 'OK', 'fix1')),
    ('kernels.cu', 'drain_device_legacy_stream', 'cudaGetLastError', 'if (e != cudaSuccess)',
     (CONSUME, '-', 'consumes the drain\'s own failure (reported by the leak counter)', 'OK', 'fix1')),
    ('kernels.cu', 'drain_device_legacy_stream', 'cudaSetDevice', 'switched && cudaSetDevice(prev)',
     ('the caller\'s device (restore)', 'proven current (read above)', 'was IGNORED in 62c8d9f; now checked: counted (device_restore_failures), consumed, never thrown (destructor path)', 'FIXED (b)', 'drain')),
    ('kernels.cu', 'drain_device_legacy_stream', 'cudaGetLastError', None,
     (CONSUME, '-', 'consumes the failed restore\'s entry', 'FIXED (b)', 'drain')),
    ('kernels.cu', 'raise_cuda_failure', 'cudaGetLastError', None,
     (CONSUME, '-', 'report-and-consume rule for every raise after a failed call (guard, label proof)', 'FIXED (b)', 'fill')),
    ('kernels.cu', 'raise_cuda_failure', None, None, HOST_STR),
    ('kernels.cu', 'CheckedDeviceGuard', 'cudaGetDevice', None,
     ('host query (caller\'s device)', 'proven current', 'checked: RuntimeError, consumed, nothing switched', 'FIXED (a) new guard', 'fill')),
    ('kernels.cu', 'CheckedDeviceGuard', 'cudaSetDevice', None,
     ('the tensor\'s device (label proven against the allocation)', 'guard (switch)', 'checked: RuntimeError naming the device, consumed', 'FIXED (a) new guard', 'fill')),
    ('kernels.cu', 'restore', 'cudaSetDevice', None,
     ('the caller\'s device (success-path restore)', 'proven current (read by the guard)', 'checked: RuntimeError (the write is queued and counted in the revision first)', 'FIXED (a)', 'fill')),
    ('kernels.cu', '~CheckedDeviceGuard', 'cudaSetDevice', None,
     ('the caller\'s device (restore while unwinding)', 'proven current', 'checked: counted, consumed, never thrown', 'FIXED (a)', 'fill')),
    ('kernels.cu', '~CheckedDeviceGuard', 'cudaGetLastError', None,
     (CONSUME, '-', 'consumes the failed restore\'s entry', 'FIXED (b)', 'fill')),
    ('kernels.cu', 'cuda_storage_device_mismatch', 'cudaPointerGetAttributes', None,
     ('host-side pointer query (no device needs to be current; legal for any device)', 'host', 'checked: RuntimeError, consumed; result compared with the label (mismatch / unregistered -> refusal)', 'FIXED (a) new label proof', 'label')),
    ('kernels.cu', 'fill_', 'cudaGetLastError', None,
     (CONSUME, '-', 'pre-launch take: a pending error raises as pre-existing, nothing launched (CUDA PG 3.2.12 rule)', 'FIXED (b)', 'fillpend')),
    ('kernels.cu', 'fill_', None, None, HOST_STR),
    ('kernels.cu', 'fill_', '<<< kernel launch', None,
     ('legacy stream of the tensor\'s device (62c8d9f: the CURRENT device)', 'guard + label proof', 'post-launch cuda_check_last (pre-existing helper: cudaGetLastError, raise); a pending error was taken first', 'FIXED (a)', 'fill')),
    # ------------------------------------------------------------------ core bindings.cpp
    ('bindings.cpp', 'dlpack_fence_consumer', 'cudaGetDevice', None,
     ('host query (caller\'s device)', 'proven current', 'checked: RuntimeError, consumed, nothing switched (62c8d9f: never asked)', 'FIXED (a) V4-01', 'v401f')),
    ('bindings.cpp', 'dlpack_fence_consumer', 'cudaSetDevice', 'e = cudaSetDevice(device)',
     ('the tensor\'s device (label proven against the allocation)', 'guard (switch)', 'checked: RuntimeError, caller unchanged, consumed', 'FIXED (a) V4-01', 'v401f')),
    ('bindings.cpp', 'dlpack_fence_consumer', 'cudaEventCreateWithFlags', None,
     ('the tensor\'s device (62c8d9f: the CURRENT device)', 'guard', 'checked: device restored, consumed, RuntimeError naming the step', 'FIXED (a) V4-01', 'v401')),
    ('bindings.cpp', 'dlpack_fence_consumer', 'cudaEventRecord', None,
     ('legacy stream of the tensor\'s device (62c8d9f: of the CURRENT device)', 'guard', 'checked: device restored, consumed, RuntimeError', 'FIXED (a) V4-01', 'v401')),
    ('bindings.cpp', 'dlpack_fence_consumer', 'cudaStreamWaitEvent', None,
     ('the consumer\'s stream (handle: any device, cross-device wait legal per CUDA PG 3.2.9.3; 2: per-thread stream of the tensor\'s device)', 'guard (issued with the tensor\'s device current)', 'checked: device restored, consumed, RuntimeError', 'FIXED (a) V4-01', 'v401')),
    ('bindings.cpp', 'dlpack_fence_consumer', 'cudaEventDestroy', None,
     ('the event\'s own device', 'guard', 'was IGNORED in 62c8d9f; now checked: RuntimeError after the restore', 'FIXED (b)', 'v401f')),
    ('bindings.cpp', 'dlpack_fence_consumer', 'cudaSetDevice', 'switched ? cudaSetDevice(prev)',
     ('the caller\'s device (restore)', 'proven current (read above)', 'checked: RuntimeError says the thread is still on the tensor\'s device; consumed', 'FIXED (a) V4-01', 'v401f')),
    ('bindings.cpp', 'dlpack_fence_consumer', 'cudaGetLastError', None,
     (CONSUME, '-', 'consumes the fence\'s own failed calls (reported by the raise)', 'FIXED (b)', 'v401f')),
    ('bindings.cpp', 'dlpack_fence_consumer', None, None, HOST_STR),
    ('bindings.cpp', 'tensor_from_dlpack_capsule', 'cudaGetDevice', None,
     ('host query (the import\'s current device)', 'proven current', 'was IGNORED in 62c8d9f (cur stayed 0); now checked BEFORE Storage construction and the capsule rename: RuntimeError naming cudaGetDevice, consumed', 'FIXED (b) V4-03', 'v403')),
    ('bindings.cpp', 'tensor_from_dlpack_capsule', 'cudaGetLastError', None,
     (CONSUME, '-', 'consumes the failed query (reported by the raise)', 'FIXED (b) V4-03', 'v403')),
    ('bindings.cpp', 'tensor_from_dlpack_capsule', None, None, HOST_STR),
    ('bindings.cpp', 'tensor_empty', 'cudaGetDevice', None,
     ('host query (the allocation\'s device = current)', 'proven current', 'checked: RuntimeError, nothing allocated; the label is set to it (62c8d9f: label cuda:0 whatever was current)', 'FIXED (a)', 'empty')),
    ('bindings.cpp', 'tensor_empty', 'cudaGetLastError', 'if (dev.is_cuda()) cudaGetLastError()',
     (CONSUME, '-', 'consumes a failed allocation\'s entry (the engine\'s Storage constructor raises without consuming)', 'FIXED (b)', 'empty')),
    ('bindings.cpp', 'tensor_empty', 'cudaGetLastError', None,
     (CONSUME, '-', 'consumes the failed device query (reported by the raise)', 'FIXED (b)', 'v403')),
    ('bindings.cpp', 'tensor_empty', None, None, HOST_STR),
    # ------------------------------------------------------------------ tc_signal cuda_guard.h
    ('cuda_guard.h', 'raise_cuda_failure', 'cudaGetLastError', None,
     (CONSUME, '-', 'rule 1: every raise after a failed call consumes it (62c8d9f: the four guard copies raised without consuming)', 'FIXED (b)', 'guard')),
    ('cuda_guard.h', 'raise_cuda_failure', None, None, HOST_STR),
    ('cuda_guard.h', 'DeviceGuard', 'cudaGetDevice', None,
     ('host query (caller\'s device)', 'proven current', 'checked: RuntimeError, consumed', 'FIXED (b) (shared guard; was 4 copies)', 'guard')),
    ('cuda_guard.h', 'DeviceGuard', 'cudaSetDevice', None,
     ('the operands\' device (DLPack device_id of every operand, checked equal: common_device)', 'guard (switch)', 'checked: RuntimeError naming the device, consumed, nothing switched', 'FIXED (b)', 'guard')),
    ('cuda_guard.h', 'restore', 'cudaSetDevice', None,
     ('the caller\'s device (success-path restore)', 'proven current', 'NEW: checked, RuntimeError (the call\'s operands stay retained until the queued work completes); 62c8d9f only restored in the destructor and ignored the result', 'FIXED (b)', 'guard')),
    ('cuda_guard.h', '~DeviceGuard', 'cudaSetDevice', None,
     ('the caller\'s device (restore while unwinding)', 'proven current', 'was IGNORED; now checked: consumed, counted, raised once by the next synchronize()', 'FIXED (b)', 'guard')),
    ('cuda_guard.h', '~DeviceGuard', 'cudaGetLastError', None, (CONSUME, '-', 'consumes the failed restore', 'FIXED (b)', 'guard')),
    ('cuda_guard.h', 'QuietDeviceGuard', 'cudaGetDevice', None,
     ('host query (caller\'s device)', 'proven current', 'checked: reported via ok()/error(), consumed (unchanged from 62c8d9f)', 'OK', 'fix1')),
    ('cuda_guard.h', 'QuietDeviceGuard', 'cudaSetDevice', None,
     ('the entry\'s / operands\' device', 'guard (switch)', 'checked: reported via ok()/error(), consumed (unchanged)', 'OK', 'fix1')),
    ('cuda_guard.h', 'QuietDeviceGuard', 'cudaGetLastError', None, (CONSUME, '-', 'consumes the guard\'s own failure (unchanged)', 'OK', 'fix1')),
    ('cuda_guard.h', '~QuietDeviceGuard', 'cudaSetDevice', None,
     ('the caller\'s device (restore)', 'proven current', 'was IGNORED; now checked: consumed, counted, raised once by the next synchronize()', 'FIXED (b)', 'quiet')),
    ('cuda_guard.h', '~QuietDeviceGuard', 'cudaGetLastError', None, (CONSUME, '-', 'consumes the failed restore', 'FIXED (b)', 'quiet')),
    # ------------------------------------------------------------------ tc_signal dlpack_operand.cpp
    ('dlpack_operand.cpp', 'sync_device_legacy', 'cudaStreamSynchronize', None,
     ('legacy stream of the entry\'s device', 'guard (QuietDeviceGuard)', 'checked: failure -> that device\'s entries leaked and reported (N1)', 'OK', 'fix1')),
    ('dlpack_operand.cpp', 'sync_device_legacy', 'cudaGetLastError', None, (CONSUME, '-', 'consumes the failed sync (reported by N1 paths)', 'OK', 'fix1')),
    ('dlpack_operand.cpp', 'destroy_event_on_device', 'cudaEventDestroy', None,
     ('the event\'s own device (62c8d9f: whatever was current)', 'guard (QuietDeviceGuard)', 'was IGNORED; now checked: consumed, counted, raised once by the next synchronize()', 'FIXED (a)+(b)', 'destroy')),
    ('dlpack_operand.cpp', 'destroy_event_on_device', 'cudaGetLastError', None, (CONSUME, '-', 'consumes the failed destroy', 'FIXED (b)', 'destroy')),
    ('dlpack_operand.cpp', 'sync_and_release_all_impl', 'cudaGetDevice', None,
     ('host query (caller\'s current device, also synced)', 'proven current', 'checked: failure is an error of synchronize() (N1)', 'OK', 'fix1')),
    ('dlpack_operand.cpp', 'sync_and_release_all_impl', 'cudaEventQuery', None,
     ('the entry\'s event (any device: cross-device query legal per CUDA PG 3.2.9.3)', 'operand (entry) device, no switch needed', 'checked: not ready -> stop/leak by rule; error -> leak + reported (N1)', 'OK', 'fix1')),
    ('dlpack_operand.cpp', 'sync_and_release_all_impl', 'cudaPeekAtLastError', None,
     (CONSUME, '-', 'defensive NotReady clear (PG 3.2.12: NotReady is never put in the slot)', 'OK', 'fix1')),
    ('dlpack_operand.cpp', 'sync_and_release_all_impl', 'cudaGetLastError', None, (CONSUME, '-', 'consumes its own failures', 'OK', 'fix1')),
    ('dlpack_operand.cpp', 'acquire_event', 'cudaEventCreateWithFlags', None,
     ('the operands\' device', 'guard (QuietDeviceGuard)', 'checked: RuntimeError before any launch, consumed', 'OK', 'dev')),
    ('dlpack_operand.cpp', 'acquire_event', 'cudaGetLastError', None, (CONSUME, '-', 'consumes the failed create', 'OK', 'dev')),
    ('dlpack_operand.cpp', 'acquire_event', None, None, HOST_STR),
    ('dlpack_operand.cpp', 'RetainUntilComplete::RetainUntilComplete', 'cudaGetLastError', None,
     (CONSUME, '-', 'NEW pre-launch take at the one constructor every launch path shares: a pending error raises as pre-existing, nothing launched, operands released at once (62c8d9f: launched, then reported it as its own launch failure)', 'FIXED (b)', 'pre')),
    ('dlpack_operand.cpp', 'RetainUntilComplete::RetainUntilComplete', None, None, HOST_STR),
    ('dlpack_operand.cpp', 'RetainUntilComplete::~RetainUntilComplete', 'cudaEventRecord', None,
     ('legacy stream of the operands\' device', 'guard (QuietDeviceGuard)', 'checked: failure -> synchronous fallback, then release or leak (F01/N1)', 'OK', 'dev')),
    ('dlpack_operand.cpp', 'RetainUntilComplete::~RetainUntilComplete', 'cudaGetLastError', None, (CONSUME, '-', 'consumes the failed record', 'OK', 'fix1')),
    ('dlpack_operand.cpp', 'drain_completed', 'cudaEventQuery', None,
     ('the entry\'s event (any device; cross-device query legal)', 'operand (entry) device, no switch needed', 'checked: not ready -> stop; error -> entry left for synchronize(), which re-queries and reports/leaks it', 'OK', 'query')),
    ('dlpack_operand.cpp', 'drain_completed', 'cudaPeekAtLastError', None, (CONSUME, '-', 'defensive NotReady clear', 'OK', 'query')),
    ('dlpack_operand.cpp', 'drain_completed', 'cudaGetLastError', 'if (cudaPeekAtLastError() == cudaErrorNotReady)', (CONSUME, '-', 'defensive NotReady clear', 'OK', 'query')),
    ('dlpack_operand.cpp', 'drain_completed', 'cudaGetLastError', None,
     (CONSUME, '-', 'NEW: consumes a non-NotReady query failure (62c8d9f left it, so the same call\'s launch check reported it as its own)', 'FIXED (b)', 'query')),
    ('dlpack_operand.cpp', 'synchronize_and_release_all', 'cudaGetLastError', None,
     (CONSUME, '-', 'NEW V4-04 capture: takes the pre-existing error first, drains as before, then raises naming it', 'FIXED (b) V4-04', 'v404')),
    ('dlpack_operand.cpp', 'synchronize_and_release_all', None, None, HOST_STR),
    # ------------------------------------------------------------------ tc_signal launch TUs
    ('lstm.cu', 'check_cuda', 'cudaGetLastError', None,
     (CONSUME, '-', 'NEW: the raise consumes the failed call (62c8d9f left it for the next launch check)', 'FIXED (b)', 'checkcuda')),
    (None, 'check_cuda', 'cudaGetLastError', None,
     (CONSUME, '-', 'NEW: the raise consumes the failed call (same rule as lstm.cu)', 'FIXED (b)', 'ccnone')),
    (None, 'check_cuda', None, None, HOST_STR),
    (None, 'check_launch', 'cudaGetLastError', None,
     (CONSUME, '-', 'post-launch check: reads and consumes; raises; the slot was clean right before the first launch (pre-launch take)', 'OK', 'pre')),
    (None, 'handle_for', 'cublasCreate', None, ('the operands\' device (handle bound to the current device at creation)', 'guard (caller\'s DeviceGuard)', 'checked: RuntimeError', 'OK', 'dev')),
    (None, 'lstm_handle_for', 'cublasCreate', None, ('the operands\' device (handle bound to the current device at creation)', 'guard (caller\'s DeviceGuard)', 'checked: RuntimeError', 'OK', 'dev')),
    (None, None, 'cublasDestroy', None,
     ('the new handle\'s device', 'guard (caller\'s DeviceGuard)', 'NEW (failure path of a setting): status checked and named in the raise', 'FIXED (b)', 'cublas')),
    (None, None, 'cublasSetStream', None, ('handle (its device current)', 'guard', 'checked; on failure the new handle is destroyed, RuntimeError (62c8d9f leaked it)', 'FIXED (b)', 'cublas')),
    (None, None, 'cublasSetMathMode', None, ('handle (its device current)', 'guard', 'checked; on failure the new handle is destroyed, RuntimeError (62c8d9f leaked it)', 'FIXED (b)', 'cublas')),
    (None, None, 'cublasSetAtomicsMode', None, ('handle (its device current)', 'guard', 'checked; on failure the new handle is destroyed, RuntimeError (62c8d9f leaked it)', 'FIXED (b)', 'cublas')),
    (None, None, 'cublasSetPointerMode', None, ('handle (its device current)', 'guard', 'checked; on failure the new handle is destroyed, RuntimeError (62c8d9f leaked it)', 'FIXED (b)', 'cublas')),
    (None, None, 'cublasGetMathMode', None, ('handle (its device current)', 'guard (cublas_config\'s DeviceGuard)', 'checked: RuntimeError', 'OK', 'dev')),
    (None, None, 'cublasGetAtomicsMode', None, ('handle (its device current)', 'guard', 'checked: RuntimeError', 'OK', 'dev')),
    (None, None, 'cublasGetPointerMode', None, ('handle (its device current)', 'guard', 'checked: RuntimeError', 'OK', 'dev')),
    (None, None, 'cublasGetStream', None, ('handle (its device current)', 'guard', 'checked: RuntimeError', 'OK', 'dev')),
    (None, None, 'cublasSgemmStridedBatched', None, ('legacy stream of the operands\' device (handle\'s stream)', 'guard', 'status checked: RuntimeError; then check_launch', 'OK', 'dev')),
    (None, None, 'cublasSgemm', None, ('legacy stream of the operands\' device (handle\'s stream)', 'guard', 'status checked: RuntimeError; then check_launch', 'OK', 'dev')),
    (None, 'build_info', 'cudaRuntimeGetVersion', None, ('host (runtime version)', 'host', 'checked: RuntimeError (check_cuda consumes)', 'OK', '')),
    (None, 'build_info', 'cublasGetProperty', None, ('host (library version)', 'host', 'checked: RuntimeError', 'OK', '')),
    (None, 'build_info', 'cudaGetDevice', None, ('host query (reported as current_device)', 'proven current', 'checked: RuntimeError (now consumed)', 'FIXED (b) via check_cuda', 'ccnone')),
    (None, 'lstm_forward', 'cudaMemcpyAsync', None, ('legacy stream of the operands\' device', 'guard', 'checked: RuntimeError (now consumed by check_cuda)', 'FIXED (b) via check_cuda', 'checkcuda')),
    (None, 'lstm_forward', 'cudaMemsetAsync', None, ('legacy stream of the operands\' device', 'guard', 'checked: RuntimeError (now consumed by check_cuda)', 'FIXED (b) via check_cuda', 'checkcuda')),
    (None, None, '<<< kernel launch', None,
     ('legacy stream of the operands\' device', 'guard (DeviceGuard on common_device)', 'post-launch check (check_launch / check_cuda(cudaGetLastError)); slot clean before the first launch (pre-launch take)', 'OK', 'dev')),
    (None, 'row_norm_forward', 'cudaGetLastError', None, (CONSUME, '-', 'post-launch check, raise', 'OK', 'dev')),
    (None, 'upsample_nearest1d_forward', 'cudaGetLastError', None, (CONSUME, '-', 'post-launch check, raise', 'OK', 'dev')),
    (None, 'reflection_pad1d_forward', 'cudaGetLastError', None, (CONSUME, '-', 'post-launch check, raise', 'OK', 'dev')),
    (None, 'rint_forward', 'cudaGetLastError', None, (CONSUME, '-', 'post-launch check, raise', 'OK', 'dev')),
]


def match(site):
    base = site['file'].split('/')[-1]
    for f, fn, call, frag, ann in R:
        if f is not None and f != base:
            continue
        if fn is not None and fn != site['function']:
            continue
        if call is None:
            if site['call'] not in ('cudaGetErrorName', 'cudaGetErrorString'):
                continue
        elif call != site['call']:
            continue
        if frag is not None and frag not in site['text']:
            continue
        return ann
    return None


sites = json.load(open('sites.json'))
rows, missing = [], []
for s in sites:
    if s['kind'] != 'product':
        continue
    s['function'] = enclosing(W + s['file'], s['line'])
    ann = match(s)
    if ann is None:
        missing.append(s)
        continue
    dev, est, handling, verdict, test = ann
    rows.append(dict(s, device=dev, established=est, handling=handling, verdict=verdict, test=test, test_name=T.get(test, test)))
if missing:
    for m in missing:
        print('UNANNOTATED', m['file'], m['line'], m['function'], m['call'], m['text'][:80])
    sys.exit(1)
# test-code sites
TEST_NOTE = {
    'fix4_cudart_shim.c': ('interposer DEFINITION (LD_PRELOAD shim); forwards to the real call through real_<name>()',
                           'returns the real result, or a GENUINE injected failure (real cudaSetDevice(-1))'),
    'test_dlpack.py': ('test script call (ctypes on the shared libcudart.so.12) or embedded shim definition', 'asserted by the test'),
    'test_fix4_findings.py': ('test script call (ctypes): stages / reads the error slot', 'asserted by the test'),
    'fix4_stage_kernel.cu': ('test helper: stages a real launch-configuration refusal', 'returned to the test, which asserts it'),
    'fix2_native_harness.cpp': ('FIX2 native harness', 'cudaMalloc checked (pre-existing); the pre-drain cudaStreamSynchronize result was IGNORED and is now checked (the run fails, slot consumed): FIXED (b), test code, exercised by tcs test_fix2_findings oom_sweep'),
    'fix3_native_harness.cpp': ('FIX3 native harness (unchanged by FIX4)', 'checked: the harness exits 2 (pre-existing)'),
}
trows = []
for s in sites:
    if s['kind'] != 'test':
        continue
    what, handling = TEST_NOTE[s['file'].split('/')[-1]]
    trows.append(dict(s, function=s['function'], device='test', established='-', handling=handling,
                      verdict='OK (test code)', test='', role=what))
json.dump({'product': rows, 'test': trows}, open('sweep.json', 'w'), indent=1)
# grouped markdown: one row per (file, function, call, annotation), with its lines
groups = OrderedDict()
for r in rows:
    if r['call'] in ('cudaGetErrorName', 'cudaGetErrorString'):   # host string lookups: one row per file
        key = (r['file'], '(message building, several)', 'cudaGetErrorName / cudaGetErrorString', r['device'],
               r['established'], r['handling'], r['verdict'], r['test'])
    else:
        key = (r['file'], r['function'], r['call'], r['device'], r['established'], r['handling'], r['verdict'], r['test'])
    groups.setdefault(key, []).append(r['line'])
out = ['| # | file:lines | function | call | device | device established by | return / error handling | verdict | proving test |',
       '|---|---|---|---|---|---|---|---|---|']
for i, (k, lines) in enumerate(groups.items(), 1):
    f, fn, call, dev, est, h, v, t = k
    lines = sorted(set(lines))
    out.append(f'| {i} | `{f.replace("tensor_cuda/src/", "core/").replace("tc_signal/src/", "tcs/")}:{",".join(map(str, lines))}` | '
               f'`{fn}` | `{call}` | {dev} | {est} | {h} | **{v}** | {("`" + t + "`") if t else "-"} |')
tg = OrderedDict()
for r in trows:
    tg.setdefault((r['file'], r['role'], r['handling']), []).append(f"{r['line']} {r['call']}")
tout = ['| file | role | sites (line call) | handling | verdict |', '|---|---|---|---|---|']
for (f, role, h), items in tg.items():
    tout.append(f'| `{f}` | {role} | {"; ".join(items)} | {h} | {"FIXED (b) (test code)" if "IGNORED" in h else "OK (test code)"} |')
vc = Counter(r['verdict'].split(' ')[0] for r in rows)
cc = Counter('host string' if r['call'] in ('cudaGetErrorName', 'cudaGetErrorString') else 'call' for r in rows)
summary = (f"Product call sites: {len(rows)} ({cc['call']} CUDA runtime/cuBLAS calls and kernel launches + "
           f"{cc['host string']} error-string lookups) in {len(groups)} table rows; verdicts: OK {vc['OK']}, "
           f"FIXED {vc['FIXED']}, other {len(rows) - vc['OK'] - vc['FIXED']}. Test-code sites: {len(trows)} in "
           f"{len(tg)} files (OK, except the FIX2 harness pre-drain sync: FIXED). Unannotated: 0.")
legend = ['Test keys:'] + [f'- `{k}`: {v}' for k, v in T.items()]
open('sweep_table.md', 'w').write(summary + '\n\n' + '\n'.join(out) + '\n\n' + '\n'.join(legend) + '\n\nTest code:\n\n'
                                  + '\n'.join(tout) + '\n')
print(summary)
fixed_rows = Counter(r['verdict'] for r in rows if r['verdict'].startswith('FIXED'))
print(fixed_rows)
