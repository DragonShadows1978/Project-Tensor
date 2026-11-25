import subprocess
if subprocess.run(["forge-gpu-lease-check"]).returncode != 0: raise SystemExit("not inside a forge-gpu-run lease")
# BWD-AUD1 staged GPU witness for BWDA-F03: a zero-extent A batch with a broadcast B. The forward returns an empty
# output; the backward's dB has a zero leading extent, and reduce_to (kernels.cu:1364-1376) reduces only axes with
# extent > 1, so the reshape to B's shape raises "reshape: element count mismatch" (kernels.cu:1009-1011) instead of
# returning dB = 0 (the empty sum). Reached from Variable::accumulate_grad (autograd.cpp:89) and the closure's
# reduce_to (ops.cpp:848). Writes only artifacts/bwd_aud1/gpu/f03/ (refuses to start if it exists).
# Prior art: NumPy's sum over an empty axis is 0 (identity of addition); no other prior art known to me.
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[4]  # BWD-F1 copy: one level deeper (gpu_lines/<arm>/)
OUT = ROOT / 'artifacts/bwdf1/gpu' / ('g2_f03_' + pathlib.Path(__file__).resolve().parent.name)  # BWD-F1 copy: output path
if OUT.exists():
    raise SystemExit(f'{OUT} exists; move it aside before a retry')
OUT.mkdir(parents=True)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aud1_gpu_load as L  # noqa: E402
import aud1_verdicts as V  # noqa: E402

tc, sha = L.load('tensor_cuda')
import numpy as np  # noqa: E402

LAYOUTS = (('a3-zero-batch-b2', (0, 4, 3), (3, 5)), ('a4-zero-batch-b3-singleton', (2, 0, 4, 3), (1, 3, 5)))
cases = []
for dtype in ('float32', 'float16'):
    for label, ash, bsh in LAYOUTS:
        try:
            rng = np.random.default_rng(3)
            a = tc.tensor(np.zeros(ash, np.float32), device='cuda', dtype=dtype, requires_grad=True)
            b = tc.tensor(rng.standard_normal(bsh).astype(np.float32), device='cuda', dtype=dtype,
                          requires_grad=True)
            y = tc.matmul(a, b)
            err = None
            try:
                y.sum().backward()
                tc.synchronize()
            except RuntimeError as e:
                err = str(e)
            ga = None if a.grad is None else a.grad.numpy()
            gb = None if b.grad is None else b.grad.numpy()
            cases.append(V.f03_case(label, dtype, err, ga, gb, ash, bsh))
        except Exception as e:  # noqa: BLE001
            cases.append(V.error_case(label, dtype, e))

verdict = V.f03_verdict(cases)
(OUT / 'results.json').write_text(json.dumps({'line': 'bwdaud1-f03', 'module_sha256': sha, 'cases': cases,
                                              'verdict': verdict}, indent=1))
for c in cases:
    print(json.dumps(c))
print('VERDICT', verdict)
