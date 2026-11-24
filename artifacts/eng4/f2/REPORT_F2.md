<!-- Filed by the lead from the seat's final message (the harness refused the seat's own write). Content unchanged
except where marked LEAD; rows and bullets merged for length. Receipts: LEDGER_F2.md. -->

# ENG4-F2 report — the three ENG4-REV2 findings hardened

Seat: Claude Opus 5.5 (Agent tool). Order `orders/ENG4_F2_REV2_HARDENING.md` at 3f8bffe80 (sha256 c6ca7096…).

> **LEAD verification (2026-11-23 10:1x EST).**
> - The diff is limited to `matmul.cu`, `ops.cpp` and the test.
> - Module sha256s recomputed: F2 `c913e8d9` is installed.
> - CPU re-run from `tensor_cuda/` (`test_eng4_matmul.py` plus the BWD matmul and native tests): 156 passed, 97
>   skipped (GPU).
> - The seat's statement on the F1 incident's cause (compute-sanitizer removes `CUDA_VISIBLE_DEVICES` from its
>   target) is confirmed by a lead GPU line inside a forge lease (label `tensor-cvd-probe`).

## J1 to J6

- **J1 PASS:** CPU, CUDA hidden, on F2: 153 passed, 82 GPU tests skipped (`f2/j1/J1_F2_final.xml`). The same file on
  F1 fails 20 (the 4 over-2^63 witnesses and the 16 Y2 cases).
- **J2 PASS:** clean checkout (`git archive` of 3f8bffe80 minus `scripts/eng4_*` and `artifacts/eng4`, plus F2's three
  changed files and the F2 module): 153 passed, 82 skipped; outside the package it opened only `pyproject.toml` and the
  test file.
- **J3 PASS:** CPU suite per ID, B vs F2: 4038 IDs each, 0 differences (`f2/J3_CPU_COMPARE.json`).
- **J4 PASS (CPU parts):** m09 killed 4/4, m10 16/16, each on its named assertion with the executed-marker seen; m10's
  GPU line STAGED.
- **J5 STAGED:** 25 lines, each dry run 78.
- **J6 PASS:** writes only in the writable set; no GPU use of any kind (no line, no step function, nothing that removes
  `CUDA_VISIBLE_DEVICES`); no git writes; nothing deleted.

## Diff summary (`f2/F2_source.diff`, sha256 8cd855c7…; the 7 new scripts as new-file hunks)

- **Y1 (F01), `matmul.cu`:** out_shape is built first. A zero leading extent of A gives batch 0; otherwise the extents
  are multiplied one at a time, refusing (`BatchCount`) as soon as `extent > INT_MAX / batch`, so no product above
  `INT_MAX` is formed. The message prefix "GEMM batch count above INT_MAX" is kept, now naming the failing dimension.
  Values are unchanged wherever the old product did not overflow.
- **Y3 (F03), `matmul.cu`:** the K = 0 path keeps `NDArray::zeros`, then `cuda_check_last("matmul zero fill")` for CUDA
  outputs.
- **Y2 (F02), `ops.cpp`, `Tensor matmul`'s gradient:** A's (empty) gradient is accumulated first, as before. Then, if
  the ranks differ (B shared) and A has a zero leading extent, B gets zeros of its own shape and dtype (fill checked),
  and the function returns before the dB GEMM and `reduce_to`. `reduce_to`, `NDArray::zeros` and the other core
  members are unchanged.
- **Tests, `test_eng4_matmul.py`:** Y1 zero-leading witnesses, over-2^63 refusals and the BatchCount boundary; the Y2
  host float16/bfloat16 zero-gradient test; `test_gpu_bwd0_zero_gradient`. The file stays self-contained.

## Module sha256s

| Arm | sha256 |
|---|---|
| F2 | c913e8d9e1eff88e6cbc0f68f6f2d03076baa022cc6addd92843215dea7fbd32 (installed worktree module) |
| m09_f2 | a2e81be14dd9a0e2b933a8c93f0833ce5ec785bf4bacb129ba3b41993f89b76c |
| m10_f2 | 5e20548f8ef6769a480a209b74629c00ce4ea77f707b53d9f77e92a21ba4106b |

B (20881ace…) and F1 (7b80c6dc…) unchanged.

## Intended changes (`f2/INTENDED_CHANGES_F2.json`, sha256 fa558007…, 09:54, before both J3 runs)

- Suite IDs: only ENG4's three M7 IDs, carried over; none new (J3 observed none).
- **Y2 (raise → zero gradient):** classes a and c at batch 0 with a shared, nonempty B — A (0,2,3) × B (3,4);
  A (0,2,3) × B (1,1,3,4); A (2,0,2,3) × B (3,4); A (0,3,2,3) × B (1,3,4) — `trans_b` both ways, float32/float16/
  bfloat16. Before: F1 raised "reshape: element count mismatch" (REV2's CPU witness and the seat's J1 run on F1);
  Build B to be recorded by `bwd0_B`. After: B's gradient is positive zeros of B's shape and dtype; A's is empty.
- **Y1 (over-2^63 leading products now refused):** nonzero leading extents (2^32, 2^32), (2^16, 2^48), (3, 2^62) and
  (2^31−1, 2^31−1, 4). Before (F1): the wrapped product passed the check and the call stopped at the output byte bound
  with an OverflowError. After: a BatchCount refusal.
- G3: no cells change.

## Mutants (`f2/MUTANTS_F2.json`, sha256 db6b5ed9…, 09:54:07, before any mutant build; `f2/MUTANT_RESULTS_F2.json`)

| Mutant | Killing tests | Result |
|---|---|---|
| m09_f2_naive_product | `test_j1_y1_refuses_before_overflow` × 4 | KILLED 4/4 (the mutant's OverflowError escapes `pytest.raises(RuntimeError)`, as registered) |
| m10_f2_empty_batch_raise | 16 CPU (`test_j1_y2_empty_batch_shared_b_zero_gradient`) + 24 GPU | KILLED 16/16 on CPU ("zero-gradient check: … reshape: element count mismatch"); GPU line mut_m10_f2 STAGED |

m09 on the order's 2^32 × 2^32 × 0 witness is unkillable by test and recorded so: the naive product of a list holding
a 0 is 0 (2^32 · 2^32 wraps to 0, then × 0), so the assertion holds for the mutant too. The mutant forms 2^32 · 2^32 in
int64 (signed overflow) before reaching the 0; the over-2^63 witnesses are the ones a test can see, and they kill it.
For that witness, "accepted as batch 0" is asserted as: the predicate accepts (no BatchCount refusal), and the call then
stops at the inherited output byte bound, naming exactly the predicate's out_shape (those extents can never pass the
output constructor, which counts the 0 as 1). Where the output fits, e.g. leading (2^32, 0), it returns NumPy's empty
result; the L55 probe shows plan batch 0 directly.

## L55 (`f2/l55/`; F2's real `matmul.cu` with stubbed cuBLAS, no CUDA call)

| Witness | Predicate | Batch vs output |
|---|---|---|
| zero leading (2^32, 2^32, 0) | accepted, batch 0 | equals the output's batch product (0) |
| zero leading (0, 2^32) | accepted, batch 0 | equals 0 |
| (2147483647) | accepted, batch 2147483647 | equal |
| (1, 2147483647) | accepted, batch 2147483647 | equal |
| (32768, 65535) | accepted, batch 2147450880 | equal |
| (65536, 32768) | refused at dimension 1 | — |
| (2^31) | refused at dimension 0 | — |
| (2^32, 2^32), naive product over 2^63 | refused at dimension 0 | — |
| (3, 2^62), naive product over 2^63 | refused at dimension 1 | — |

## Staged lines (`artifacts/eng4/GPU_LINES_F2.json`, sha256 7dc296c6…)

Run from the worktree root as `python3 -B -P scripts/eng4_f2_gpu_line.py run <line>` (inner
`forge-gpu-run --min <s> --max 540 --label eng4f2-<line> -- timeout <s>s python3 -B -P scripts/eng4_f2_gpu_line.py step <line>`);
receipts in `artifacts/eng4/f2/gpu/<line>/`.

| Line | --min | inner timeout | peak host | dry run |
|---|---|---|---|---|
| bwd0_B, bwd0_F2, g3_F2, own_F2 | 120 | 480 s | 3 GiB | 78 each |
| mut_m10_f2 | 180 | 480 s | 3 GiB | 78 |
| g4_B_01 … g4_B_10, g4_F2_01 … g4_F2_10 | 480 | 500 s | 6 GiB | 78 each |

CPU commands after the lines:
- `python3 -B -P scripts/eng4_f1_g3_compare.py artifacts/eng4/gpu/g3_B/g3 artifacts/eng4/f2/gpu/g3_F2/g3 artifacts/eng4/f2/G3_F2_COMPARE.json`
- `python3 -B -P scripts/eng4_f2_bwd0.py verdict artifacts/eng4/f2/gpu/bwd0_B/bwd0 B`
- `python3 -B -P scripts/eng4_f2_bwd0.py verdict artifacts/eng4/f2/gpu/bwd0_F2/bwd0 F2`
- `python3 -B -P scripts/eng4_f2_g4_compare.py f2gpu_B f2gpu_F2 artifacts/eng4/f2/J3_GPU_COMPARE.json`
- `python3 -B -P scripts/eng4_f2_mutant_results.py gpu`

L48: the post-device steps of these lines were driven on the CPU with constructed returns; `f2/POST_DEVICE_STEPS.json`
lists 34 steps, each reaching its expected terminal label.

## L58 census (`grep -n CUDA_VISIBLE_DEVICES scripts/eng4_*.py`)

- This round's scripts: `eng4_f2_gpu_line.py:240` `step_env()` pops the variable — the only removal site — reached only
  from `step()` after `lease_guard()` finds `forge-gpu-run` among its ancestor processes (otherwise exit 7; shown to
  return False outside forge). `eng4_f2_gpu_line.py:184` (`run`) and `stage` pass the environment through unchanged.
  `eng4_f2_mutant_results.py:65` sets it to `""` for CPU pytest children. Read/record only: `eng4_f2_suite.py:177`,
  `eng4_f2_postdevice.py:262`, `eng4_build.py:85`. `eng4_f2_bwd0.py` and the unchanged scripts the lines execute
  (`eng4_g3.py`, `eng4_f1_g3_compare.py`) have no site.
- Not run this round: `eng4_gpu_line.py:163/208/310` and `eng4_f1_gpu_line.py:179/223/311` pop it without a guard (the
  route of the F1 incident); `eng4_mutant_results.py:62` and `eng4_f1_mutant_results.py:64` set it to `""`.
- On the F1 incident's cause: in both earlier smokes the seat's wrapper had replaced `pyenv` with a version that set
  `CUDA_VISIBLE_DEVICES=""`; plain-python children honoured it, the compute-sanitizer children still reached the GPU,
  so compute-sanitizer belongs in the L58 route too. (LEAD: confirmed by measurement; see the note above.)

## Deviations

1. `REPORT_F2.md` refused by the harness; this message is the report.
2. L56(b) repair 1 of 3: the m09 classifier's text list was narrower than its own registration (the OverflowError
   escaping `pytest.raises`); aligned and re-run; run 1 kept in `f2/mutants/superseded_try1/`.
3. L56(a) slips: a sed `#` collision left `eng4_f2_mutant_results.py` empty (regenerated at once); a pgrep `\|` pattern
   ended a wait early (waited again); the L55 stub `numel_of` copied from F1's probe multiplied before checking for 0
   (fixed before compiling).
4. Interpretation, lead to confirm: "accepted as batch 0" for the 2^32 × 2^32 × 0 witness, as described under
   Mutants. (LEAD: confirmed; those extents cannot pass the output constructor.)
5. Extra line `own_F2` beyond J5's list: F2's own tests on the GPU.
6. Moved, not deleted: run 1 of MUTANT_RESULTS_F2, and a first `F2_source.diff` that included the lead's file (now in
   `/tmp/eng4_tmp/`).

## Prior art

Y1: checked multiplication before overflow (CERT C INT32-C; GCC/Clang `__builtin_mul_overflow`, GCC 5, 2015). Y2: the
zero gradient of a broadcast operand over an empty axis (reverse-mode autodiff; PyTorch's `sum_to_size`); NumPy's empty
sum. Y3: the CUDA runtime's last-error API and the engine's `cuda_check_last`. Process: mutation testing (DeMillo,
Lipton and Sayward, 1978). Ours: the zero-first rule, the placements, the ancestor-process lease guard.

## Files written

Engine: `tensor_cuda/src/matmul.cu`, `tensor_cuda/src/ops.cpp`, `tensor_cuda/tests/test_eng4_matmul.py`; the installed
module (F2; swapped to B and back during f2cpu_B). New scripts: `scripts/eng4_f2_make_arms.py`, `eng4_f2_suite.py`,
`eng4_f2_g4_compare.py`, `eng4_f2_mutant_results.py`, `eng4_f2_bwd0.py`, `eng4_f2_gpu_line.py`,
`eng4_f2_postdevice.py` (all `python3 -B -P`; no stdlib names). `artifacts/eng4/GPU_LINES_F2.json`. `artifacts/eng4/f2/`:
`LEDGER_F2.md`, `preflight_L01_F2.txt`, `before_F2/`, `F2_src/`, `F2_source.diff`, `build/`, `j1/`,
`J2_clean_checkout.txt`, `J2_junit.xml`, `l55/`, `tools/l55_f2_probe.cpp`, `MUTANTS_F2.json`,
`killmap_used_for_registration.json`, `mutants/`, `MUTANT_RESULTS_F2.json`, `INTENDED_CHANGES_F2.json`,
`g4/f2cpu_B`, `g4/f2cpu_F2`, `J3_CPU_COMPARE.json`, `POST_DEVICE_STEPS.json`, `gpu_dry/` (25). Builds:
`tensor_cuda/build-eng4/F2/`, `m09_f2/`, `m10_f2/`; `pkg/` for F2 and m10_f2. Logs: `logs/eng4_f2_build_F2.log`,
`eng4_f2_build_mutants.log`, `eng4_f2_j3_cpu_B.log`, `eng4_f2_j3_cpu_F2.log`. Scratch left and listed: `/tmp/eng4_j2/`,
`/tmp/eng4_l55f2/`, `/tmp/eng4_src_m09_f2_naive_product/`, `/tmp/eng4_src_m10_f2_empty_batch_raise/`, `/tmp/eng4_tmp/`
(incl. `f2_postdevice/` and `F2_source_with_lead_file.diff`).
