# P1T-M4: `tc_signal` norms (T5) and small ops (T6)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m4`
(branch `voice-m4`, based on `voice-int` bbc4880 = M1 DLPack core + M2 `tc_signal` conv). Edits, builds and GPU runs
inside it are AUTHORIZED. You may also write scratch and mutant builds under `/tmp/p1t_m4_*`.

## Authority

David, 2026-10-12, verbatim:
- "I don't want to build it directly INTO the core of TensorCUDA - It needs to be separate I guess."
- "I'd follow your reccomended option": a separate add-on package, `tc_signal`.
- "Ship it now on Opus we have plenty of inference".

The plan is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md`. Project-Voice
approved it as LD51:
- item 3: `ts.group_norm` replaces a core GroupNorm fix; `ts.layer_norm` is used for fp32 statistics;
- item 5: "Please report µs per call for the small ops too (T6, the norms)".

## Background

M1 gave TensorCUDA core DLPack plus `tensor_cuda.empty`. M2 built `tc_signal` with conv1d/conv_transpose1d.
- Voice's runner already passes T1–T3 on TensorCUDA tensors, through `artifacts/p1t_int/tensorcuda_adapter.py`.
- Read `docs/P1T_M2_REPORT.md` and the package first (`tc_signal/src/dlpack_operand.*`, `bindings.cpp`, `conv.cu`,
  `tc_signal/__init__.py`, `tests/`). Reuse its operand contract, plan/run split and allocator.
- **A parallel seat (M3: LSTM) is editing `tc_signal` right now in another worktree.** To keep the merge trivial:
  - put ALL your code in NEW files: `src/norms.cu`, `src/small_ops.cu`, `src/m4_bindings.cpp` exposing
    `void register_m4(py::module_&)`, `tc_signal/_norms.py`, `tc_signal/_small.py`, `tests/test_norms.py` and
    `tests/test_small_ops.py`;
  - in shared files, add only the minimum: one `register_m4(m);` call in `bindings.cpp`, the source files in
    `CMakeLists.txt`, and import lines (plus `__all__`) in `tc_signal/__init__.py`;
  - do NOT refactor or modify M2's existing code. If you truly need a change there, put it in your report as a
    deviation.

## Mechanism & scope

- **Measures:** whether the six ops match their ONNX semantics, on Kokoro's real operands and on edge cells, with
  the right precision:
  - norms within normwise 1e-5 of an fp64 oracle;
  - upsample, pad and rint bit-exact, signed zero and NaN included.

  Also measured: per-call overhead for each op.
- **Mechanism doc:**
  - `/mnt/ForgeRealm/Project-Voice/research/kernels/TENSOR_IMPL_PLAN_P1T.md` §5 "T5" and "T6", and §4 (numerics);
  - `/mnt/ForgeRealm/Project-Voice/engine/tensor_requests/p1t_tests/adapter_api.py` (exact contracts:
    `instance_norm1d`, `layer_norm`, `group_norm`, `upsample_nearest1d`, `reflection_pad1d`, `rint`);
  - `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md` §4.
- **Exercises:** the new files listed above; minimal hooks in `bindings.cpp`, `CMakeLists.txt` and `__init__.py`;
  your adapter copy `artifacts/p1t_m4/tensorcuda_adapter.py`; Voice's runner `--items T5,T6`.
- **Does not cover:**
  - T4 (M3's);
  - core `tensor_cuda/` (OFF LIMITS), including core's GroupNorm/LayerNorm, which stay as they are;
  - non-integer upsample scales (refused);
  - reflect pads ≥ T (refused, as in ONNX/torch);
  - fp16 compute;
  - autograd;
  - speed beyond correctness-first.
- **Required evidence:**
  - the build log;
  - the own-test log;
  - the runner report JSON with its verdict line, for `--items T5,T6` and for `--items T1,T2,T3,T5,T6` (regression);
  - the mutant table;
  - the overhead table;
  - the diff stat.
- **Operand census:**
  - **Kokoro, from the fixtures manifest:**
    - 70 InstanceNormalizations, eps 1e-5, on [1, C, T] with C up to 1,090 and T up to the generator lengths;
    - LayerNormalization, axis −1, with eps 1e-5 (text encoder) and eps 1e-12 (ALBERT/bert);
    - GroupNorm synthetic cases with groups 1, 8 and 256;
    - Resize nearest ×2 and ×300 (floor, asymmetric);
    - reflection pad (1, 0);
    - Round on duration vectors plus the synthetic rint vectors.
  - **Your own edge cells:**
    - T=1 instance norm, where the variance is 0 and the output is `bias`;
    - constant rows;
    - large-offset rows: mean 1e4, std 1e-2, to catch one-pass variance cancellation;
    - LayerNorm over two axes (axis −2);
    - scale and bias None;
    - rint at ±0.5, ±1.5, ±2.5, −0.4 (gives −0.0), ±Inf, NaN, 2^23 + 0.5 neighbours, and subnormals;
    - pads (0, 0), (2, 3) and (T−1, T−1);
    - upsample ×1 and ×7;
    - N=2 everywhere.
- **Baseline isolation:**
  - The own oracle is fp64 numpy/torch on CPU, in the same process.
  - The exact ops compare bit patterns (`view(np.uint32)`), with NaN equal to NaN.
  - The runner's oracles are Voice's.
  - Mutants are built and run outside the worktree.
- **Forbidden configuration:**
  - `--use_fast_math` and intrinsic transcendentals (`__fsqrt_rn` is fine, but use `rsqrtf`/`sqrtf` from the math
    library, and say which);
  - TF32;
  - `atomicAdd` or any order-dependent reduction (reductions are deterministic block trees);
  - fp16 statistics;
  - edits to core or to M2's existing source;
  - writing outside the target or `/tmp`.

## Spec

```python
tc_signal.instance_norm1d(x, scale=None, bias=None, *, eps=1e-5, out=None)    # x [N, C, T]; per (n, c) over T
tc_signal.layer_norm(x, scale=None, bias=None, *, axis=-1, eps=1e-5, out=None) # normalize over axes [axis, rank)
tc_signal.group_norm(x, num_groups, scale=None, bias=None, *, eps=1e-5, out=None) # x [N, C, T]; per-channel affine
tc_signal.upsample_nearest1d(x, scale, *, out=None)          # integer scale >= 1; y[..., i] = x[..., i // scale]
tc_signal.reflection_pad1d(x, pads, *, out=None)             # pads = (begin, end), each < T; ONNX/torch reflect
tc_signal.rint(x, *, out=None)                               # round half to even (rintf), sign of zero kept
```

**Norms**
- Biased variance (ONNX).
- Statistics are fp32, computed two-pass (mean, then centred sum of squares) or by Welford, with a fixed
  block-tree reduction order.
- `y = (x − mean) · rsqrt(var + eps) · scale + bias`. Say whether you use `rsqrtf` or `1/sqrtf`, and justify it
  against the gate.
- For group_norm, `scale` and `bias` are [C] (GroupNormalization-21 / torch).
- For layer_norm, `scale` and `bias` have shape `x.shape[axis:]`.

**Small ops**
- The small ops are pure data movement or `rintf`, so they are bit-exact by construction.
- `round` in core is untouched.

## How to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout 600 …`. If a lock wait times out,
  retry. The M3 seat shares the lock, so keep each GPU process short. Use the native driver only.
- **Build:**
  `cd <target>/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89`.
  Core is prebuilt in your worktree (sha256 09aa9076…); do not rebuild it.
- **Package path:** `PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda:<target>/tc_signal`, plus
  `PYTHONDONTWRITEBYTECODE=1`.
- **Own tests:** `python3 -m pytest tc_signal/tests -q -p no:cacheprovider`. Every M2 test must still pass. The new
  tests take at most 30 s.
- **Adapter:** copy `artifacts/p1t_int/tensorcuda_adapter.py` to `artifacts/p1t_m4/tensorcuda_adapter.py` and
  implement the six functions per `adapter_api.py`. That file's names differ from tc_signal's: map
  `group_norm(x, scale, bias, *, groups, eps)` and `upsample_nearest1d(x, *, scale)`. Upload with
  `tensor_cuda.tensor`, call `ts.*`, download with `.numpy()`.
- **Voice's runner.** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 900 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<pp> python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <target>/artifacts/p1t_m4/tensorcuda_adapter.py \
    --items T5,T6 --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_m4/p1t_T5T6.json
  ```
  Then run the same command with `--items T1,T2,T3,T5,T6` into `p1t_T1T2T3T5T6.json`. Never edit Voice's files. If
  you think a Voice test is wrong, report the case and evidence.

## Gates

- **G1 (own tests):** every `tc_signal/tests` case passes, M2's included. That means:
  - norms: normwise ≤ 1e-5 against fp64 on every cell (record the max per cell), including the large-offset cell;
  - exact ops: bitwise;
  - determinism;
  - every refusal.
- **G2 (acceptance):** the runner with `--items T5,T6` exits 0 with verdict PASS, and with `--items T1,T2,T3,T5,T6` it
  also exits 0 with verdict PASS.
- **G3 (teeth):** build these mutants outside the worktree and run G1 plus G2 on each. All six must FAIL at least one
  gate; any survivor is an honest RED with your explanation.
  - (a) unbiased variance (÷(T−1));
  - (b) one-pass E[x²]−E[x]² variance; this must fail the large-offset cell;
  - (c) layer_norm with eps forced to 1e-5;
  - (d) rint via `roundf`;
  - (e) reflection pad as symmetric/edge pad;
  - (f) upsample index computed with rounding instead of floor.

  Note whether each of (c) and (e) is caught by Voice's runner, by your own tests, or by both.
- **G4 (scope):**
  - only NEW files plus the minimal hook lines (show `diff -u` of `bindings.cpp`, `CMakeLists.txt` and
    `__init__.py` against bbc4880's copies);
  - nothing under `tensor_cuda/` changed;
  - `grep -n "use_fast_math\|atomicAdd\|__half\|roundf" tc_signal/src/norms.cu tc_signal/src/small_ops.cu` shows no
    forbidden use. Explain any `roundf` hit (there should be none).
- **G5 (overhead, informational; Voice asked for it):** µs per call, median of 200 after 20 warm-ups, as wall time
  and CUDA-event time, with TensorCUDA tensors as the producer (not torch). Measure each of the six ops at:
  - a tiny shape: [1, 8, 16];
  - a Kokoro shape: instance norm [1, 512, 7160], layer norm [1, 82, 768], upsample [1, 1, 1193] ×300, pad
    [1, 22, 7160], rint [1, 82].

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. Your order file is
  immutable.
- NO git (the lead commits). NO subagents. No background waits: run every gate synchronously to completion.
- No deletes, anywhere. Never kill a process you did not start.
- These are READ-ONLY: production `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`,
  the M3 worktree, and every other fork or worktree.
- RED honesty: a failed gate is a result. Report it with receipts; never weaken a gate, test or tolerance to pass.
- Prior Art Directive (AGENTS.md): annotate at the code site, in the ledger and in the report. Known prior art:
  - instance norm: Ulyanov, Vedaldi and Lempitsky, 2016;
  - layer norm: Ba, Kiros and Hinton, 2016;
  - group norm: Wu and He, 2018;
  - variance: Welford, 1962; Chan, Golub and LeVeque, 1983 (pairwise/parallel variance);
  - ONNX operator specs (InstanceNormalization, LayerNormalization, GroupNormalization-21, Resize, Pad, Round);
  - IEEE 754 roundTiesToEven (`rintf`).
- If the AFTERIMAGE memory hook denies a Write, read what it shows you and retry. If a report Write is refused
  outright, put the full report in your final message.

## Done

- Write `docs/P1T_M4_REPORT.md`: the gate table, API, design (reduction trees, rsqrt choice), mutant table, overhead
  table and prior art. Also write `docs/P1T_M4_LEDGER.md`.
- Your final message must contain, verbatim:
  - the gate table;
  - both G2 verdict lines and report paths;
  - the G3 mutant table;
  - the G4 hook diffs and grep output;
  - the G5 table;
  - the built `.so` sha256;
  - every deviation, with its reason.
