# P1T-M3: `tc_signal.lstm`, an ONNX-semantics bidirectional LSTM (T4)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/WorkTree/tensor_fork_voice_wt_m3`
(branch `voice-m3`, based on `voice-int` bbc4880 = M1 DLPack core + M2 `tc_signal` conv). Edits, builds and GPU runs
inside it are AUTHORIZED. You may also write scratch and mutant builds under `/tmp/p1t_m3_*`.

## Authority

David, 2026-10-12, verbatim:
- "I don't want to build it directly INTO the core of TensorCUDA - It needs to be separate I guess."
- "I'd follow your reccomended option": a separate add-on package, `tc_signal`.
- "Ship it now on Opus we have plenty of inference".

The plan is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md`. Project-Voice approved
it as LD51, with amendment A2: "The whole recurrence (about 1,200 steps × 2 directions) runs inside `_tc_signal`.
There are no Python round-trips and no DLPack capsule building per step: one `ts.lstm` call gives one set of capsules."

## Background

M1 gave TensorCUDA core DLPack plus `tensor_cuda.empty`. M2 built `tc_signal` with conv1d/conv_transpose1d.
- Voice's runner already passes T1–T3 on TensorCUDA tensors, through
  `artifacts/p1t_int/tensorcuda_adapter.py` (183/183).
- Read `docs/P1T_M2_REPORT.md` and the package (`tc_signal/src/dlpack_operand.*`, `bindings.cpp`, `conv.cu`,
  `tc_signal/__init__.py`, `tests/`) first. Reuse its operand contract, plan/run split, allocator and cuBLAS handle.
- **A parallel seat (M4: norms and small ops) is editing `tc_signal` right now in another worktree.** To keep the
  merge trivial:
  - put ALL your code in NEW files: `src/lstm.cu`, `src/lstm_bindings.cpp` exposing `void register_lstm(py::module_&)`,
    `tc_signal/_lstm.py`, and `tests/test_lstm.py`;
  - in shared files, add only the minimum: one `register_lstm(m);` call in `bindings.cpp`, the source files in
    `CMakeLists.txt`, and one `from ._lstm import lstm` line (plus `__all__`) in `tc_signal/__init__.py`;
  - do NOT refactor or modify M2's existing code. If you truly need a change there, put it in your report as a
    deviation.

## Mechanism & scope

- **Measures:** whether `tc_signal.lstm` matches ONNX LSTM semantics within normwise 2e-5 of an fp64 oracle
  (tolerances_r2), deterministically, on Kokoro's six real LSTMs (H=256, bidirectional, up to 1,193 steps) and on
  edge cells; plus its run time.
- **Mechanism doc:**
  - `/mnt/ForgeRealm/Project-Voice/research/kernels/TENSOR_IMPL_PLAN_P1T.md` §5 "T4: BiLSTM" (semantics, suggested
    structure) and §4 (numerics);
  - `/mnt/ForgeRealm/Project-Voice/engine/tensor_requests/p1t_tests/adapter_api.py` (`lstm` contract);
  - `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_voice/docs/P1T_TENSOR_ALT_PLAN.md` §4.
- **Exercises:**
  - new `tc_signal/src/lstm.cu`, `tc_signal/src/lstm_bindings.cpp`, `tc_signal/tc_signal/_lstm.py` and
    `tc_signal/tests/test_lstm.py`;
  - minimal hooks in `bindings.cpp`, `CMakeLists.txt` and `__init__.py`;
  - your adapter copy `artifacts/p1t_m3/tensorcuda_adapter.py`;
  - Voice's runner `--items T4`.
- **Does not cover:**
  - T5/T6 (M4's);
  - core `tensor_cuda/` (OFF LIMITS);
  - `sequence_lens` other than None or all-equal-to-T (refused);
  - layout=1 (refused);
  - clip and peepholes (refused);
  - fp16 compute;
  - a persistent kernel or any speed work beyond correctness-first (P2);
  - autograd.
- **Required evidence:**
  - the build log;
  - the own-test log;
  - the runner report JSON with its verdict line, for `--items T4` and for `--items T1,T2,T3,T4` (regression);
  - the mutant table;
  - the timing table;
  - the diff stat;
  - the transcendental code-review note (which functions, with file:line).
- **Operand census:**
  - **Kokoro:** six bidirectional LSTMs, H=256; W [2,1024,640] ×5 and [2,1024,512] ×1; R [2,1024,256]; batch 1; T = 82
    and 1,193 (`/shared/LSTM`). Read the fixtures manifest.
  - **Your own edge cells:** forward, reverse and bidirectional; T=1 and T=2; batch 2 and 3; non-zero `initial_h`
    and `initial_c`; B=None; H not a multiple of 32 (e.g. 20); input size 1; large pre-activations (|z| up to 60) to
    test sigmoid/tanh saturation without overflow or NaN.
- **Baseline isolation:**
  - Own oracle: an independent fp64 numpy ONNX-LSTM reference, written by you, cross-checked against torch fp64
    `nn.LSTM` with the gate order converted (i,o,f,c → i,f,g,o).
  - The runner's oracles are Voice's.
  - Mutants are built and run outside the worktree.
- **Forbidden configuration:**
  - `--use_fast_math`, `__expf`/`__tanhf`/`__fdividef` or other intrinsic transcendentals;
  - TF32;
  - `atomicAdd` or order-dependent reductions;
  - a Python loop over steps;
  - per-step DLPack capsules;
  - edits to core or to M2's existing source;
  - writing outside the target or `/tmp`.

## Spec

```python
tc_signal.lstm(x, W, R, B=None, *, hidden_size, direction="forward", initial_h=None, initial_c=None,
               sequence_lens=None, layout=0, out=None) -> (Y, Y_h, Y_c)
```

- **Shapes:** x [T, N, I] (layout 0); W [D, 4H, I]; R [D, 4H, H]; B [D, 8H] = [Wb; Rb] or None (zeros);
  initial_h and initial_c [D, N, H] or None (zeros).
- **Directions:** `direction` is "forward", "reverse" or "bidirectional". D = 2 means index 0 is forward and index 1
  is reverse.
- **Returns:** Y [T, D, N, H], Y_h [D, N, H], Y_c [D, N, H]. These are new allocations from the allocator. `out` may
  be a 3-tuple.
- **Gates:** in the ONNX order i, o, f, c. Starting from `z = x_t·Wᵀ + Wb + h_{t-1}·Rᵀ + Rb`:
  - `i = σ(z_i)`, `o = σ(z_o)`, `f = σ(z_f)`, `c̃ = tanh(z_c)`;
  - `c = f⊙c_{t-1} + i⊙c̃`;
  - `h = o⊙tanh(c)`.

  σ must be numerically stable for large |z|; use accurate `expf`/`tanhf` from the CUDA math library.
- **Structure (A2):**
  1. One SGEMM for the input projection of all steps, for each direction, plus the Wb bias.
  2. A host C++ loop over steps, inside one `_tc_signal` call. Each step runs a GEMM (or GEMV) of h_{t-1}·Rᵀ for each
     direction, then one fused cell kernel that adds Rb, applies the activations, and updates c, h and Y.
  3. The reverse direction reads step T−1−t and writes `Y[T−1−t, 1]`.
- **Memory and numerics:**
  - All scratch (the projected inputs, gate buffer and state) is allocated by the Python layer through the allocator
    and passed in. C++ allocates nothing.
  - Use the legacy default stream and cuBLAS with default math.
  - Everything is fp32 and deterministic (two calls bitwise equal).

## How to run

- **GPU:** every GPU command runs as `flock -w 900 /tmp/forge-gpu.lock timeout 600 …`. If a lock wait times out,
  retry. The M4 seat shares the lock, so keep each GPU process short. Use the native driver only.
- **Build:**
  `cd <target>/tc_signal && PYBIND11_SOURCE_DIR=/mnt/ForgeRealm/Project-Tensor/tensor_cuda/build/_deps/pybind11-src ./build.sh 89`.
  Core is prebuilt in your worktree (`tensor_cuda/tensor_cuda/_tensor_cuda*.so`, sha256 09aa9076…); do not rebuild
  it.
- **Package path:** `PYTHONPATH=<target>/tensor_cuda/tensor_cuda:<target>/tensor_cuda:<target>/tc_signal`, plus
  `PYTHONDONTWRITEBYTECODE=1`.
- **Own tests:** `python3 -m pytest tc_signal/tests -q -p no:cacheprovider`. Every M2 test must still pass. The new
  LSTM tests take at most 60 s; a slower cell goes behind `TC_SLOW_TESTS=1`.
- **Adapter:** copy `artifacts/p1t_int/tensorcuda_adapter.py` to `artifacts/p1t_m3/tensorcuda_adapter.py` and
  implement `lstm` per `adapter_api.py`: upload with `tensor_cuda.tensor`, call `ts.lstm`, download with `.numpy()`.
  Assert the runner's `layout == 'onnx'`.
- **Voice's runner.** Run it from `/mnt/ForgeRealm/Project-Voice`, which is READ-ONLY to you:
  ```
  cd /mnt/ForgeRealm/Project-Voice && flock -w 900 /tmp/forge-gpu.lock timeout 900 env PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=<pp> python3 engine/tensor_requests/p1t_tests/run_p1t.py --adapter <target>/artifacts/p1t_m3/tensorcuda_adapter.py \
    --items T4 --fixtures /mnt/ForgeRealm/Project-Voice/artifacts/p1t/fixtures \
    --tolerances engine/tensor_requests/p1t_tests/tolerances_r2.json --out <target>/artifacts/p1t_m3/p1t_T4.json
  ```
  Then run the same command with `--items T1,T2,T3,T4` into `p1t_T1toT4.json`. Never edit Voice's files. If you think
  a Voice test is wrong, report the case and evidence.

## Gates

- **G1 (own tests):** every `tc_signal/tests` case passes, the M2 ones included. That is normwise ≤ 2e-5 against
  the fp64 oracle on every LSTM cell (record the max per cell), Y_h/Y_c included, determinism, and every refusal.
- **G2 (acceptance):** the runner with `--items T4` exits 0 with verdict PASS, and with `--items T1,T2,T3,T4` it also
  exits 0 with verdict PASS.
- **G3 (teeth):** build these mutants outside the worktree and run G1 plus G2 on each:
  - (a) torch gate order (i, f, g, o);
  - (b) the reverse direction not reversed;
  - (c) TF32 on the cuBLAS handle;
  - (d) initial_h/initial_c ignored;
  - (e) `expf` → `__expf` in the cell.

  (a), (b) and (c) must FAIL. (d) must FAIL your non-zero-initial-state cells. For (e), report the N change; the
  tests may not catch it, which is why the code review exists.
- **G4 (scope):**
  - only NEW files plus the minimal hook lines (show `diff -u` of `bindings.cpp`, `CMakeLists.txt` and
    `__init__.py` against bbc4880's copies);
  - nothing under `tensor_cuda/` changed;
  - `grep -n "__expf\|__tanhf\|__fdividef\|use_fast_math\|atomicAdd" tc_signal/src/lstm*` is empty.
- **G5 (time, informational):**
  - `ts.lstm` wall time and CUDA-event time for `/shared/LSTM` at T=1,193 bidirectional, median of 20;
  - torch cuDNN `nn.LSTM` fp32 on the same shape, for reference;
  - µs per step.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. Your order file is
  immutable.
- NO git (the lead commits). NO subagents. No background waits: run every gate synchronously to completion.
- No deletes, anywhere. Never kill a process you did not start.
- These are READ-ONLY: production `/mnt/ForgeRealm/Project-Tensor` (outside `Fork/`), `/mnt/ForgeRealm/Project-Voice`,
  the M4 worktree, and every other fork or worktree.
- RED honesty: a failed gate is a result. Report it with receipts; never weaken a gate, test or tolerance to pass.
- Prior Art Directive (AGENTS.md): annotate at the code site, in the ledger and in the report. Known prior art:
  - Hochreiter and Schmidhuber, 1997 (LSTM);
  - Gers, Schmidhuber and Cummins, 2000 (forget gate);
  - the ONNX LSTM operator spec;
  - Appleyard, Kocisky and Blunsom, 2016 (input projection hoisted out of the recurrence; the cuDNN RNN design);
  - numerically stable sigmoid: the standard split on the sign of z.
- If the AFTERIMAGE memory hook denies a Write, read what it shows you and retry. If a report Write is refused
  outright, put the full report in your final message.

## Done

- Write `docs/P1T_M3_REPORT.md`: the gate table, API, design, the transcendental review note, mutant table, timing
  table and prior art. Also write `docs/P1T_M3_LEDGER.md`.
- Your final message must contain, verbatim:
  - the gate table;
  - both G2 verdict lines and report paths;
  - the G3 mutant table;
  - the G4 hook diffs and grep output;
  - the G5 table;
  - the built `.so` sha256;
  - every deviation, with its reason.
