import subprocess
if subprocess.run(["forge-gpu-lease-check", "--require-held"]).returncode != 0: raise SystemExit("not inside a held forge-gpu-run lease")
# BWD-AUD1 staged GPU witness for BWDA-F01: the float16/bfloat16 matmul backward (ops.cpp:785 -> matmul.cu:25-81)
# contracts over an empty extent (forward N = 0 for dA, forward M = 0 for dB); matmul.cu passes lda/ldb = 0 to cuBLAS
# (invalid: ld >= max(1, rows)) and discards the status, so the gradient is the unwritten allocation. Expected by the
# forward's definition: exact zeros (an empty sum). Float32 is the control (ops.cpp:810 returns zeros).
# Writes only artifacts/bwd_aud1/gpu/f01/ (refuses to start if it exists). Prior art: the cuBLAS leading-dimension
# rule (NVIDIA cuBLAS 12.6 documentation); poisoning a stream-ordered pool to expose unwritten memory is ordinary
# test practice, no specific source known to me.
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[5]  # BWD-F3 copy: f3/gpu_lines/<arm>/ (BWD-F1: parents[4])
OUT = ROOT / 'artifacts/bwdf1/f3/gpu' / ('g2_f01_' + pathlib.Path(__file__).resolve().parent.name)  # BWD-F3 copy: output path
if OUT.exists():
    raise SystemExit(f'{OUT} exists; move it aside before a retry')
OUT.mkdir(parents=True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aud1_gpu_load as L  # noqa: E402
import aud1_verdicts as V  # noqa: E402

tc, sha = L.load('tensor_cuda')
import numpy as np  # noqa: E402

SENTINEL = 3.0


def poison(shape, dtype, n=32):
    keep = [tc.tensor(np.full(shape, SENTINEL, np.float32), device='cuda', dtype=dtype) for _ in range(n)]
    tc.synchronize()
    del keep


def leaf(arr, dtype):
    return tc.tensor(arr.astype(np.float32), device='cuda', dtype=dtype, requires_grad=True)


cases = []
for dtype in ('float16', 'bfloat16', 'float32'):
    rng = np.random.default_rng(0)
    try:  # forward N = 0: dA = g[4,0] @ b[3,0]^T contracts over 0
        a, b = leaf(rng.standard_normal((4, 3)), dtype), leaf(np.zeros((3, 0)), dtype)
        loss = tc.matmul(a, b).sum()
        poison((4, 3), dtype)
        loss.backward()
        tc.synchronize()
        cases.append(V.f01_case('n0-dA', dtype, None if a.grad is None else a.grad.numpy(), (4, 3), SENTINEL))
    except Exception as e:  # noqa: BLE001
        cases.append(V.error_case('n0-dA', dtype, e))
    try:  # forward M = 0: dB = a[0,3]^T @ g[0,5] contracts over 0
        a, b = leaf(np.zeros((0, 3)), dtype), leaf(rng.standard_normal((3, 5)), dtype)
        loss = tc.matmul(a, b).sum()
        poison((3, 5), dtype)
        loss.backward()
        tc.synchronize()
        cases.append(V.f01_case('m0-dB', dtype, None if b.grad is None else b.grad.numpy(), (3, 5), SENTINEL))
    except Exception as e:  # noqa: BLE001
        cases.append(V.error_case('m0-dB', dtype, e))

verdict = V.f01_verdict(cases)
(OUT / 'results.json').write_text(json.dumps({'line': 'bwdaud1-f01', 'module_sha256': sha, 'cases': cases,
                                              'verdict': verdict}, indent=1))
for c in cases:
    print(json.dumps(c))
print('VERDICT', verdict)
