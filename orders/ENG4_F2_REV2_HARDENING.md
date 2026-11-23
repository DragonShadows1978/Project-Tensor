# ENG4-F2: harden the three ENG4-REV2 findings (batch product order, empty-batch gradient, K=0 fill status)

Follows: ENG4_F1_REV1_FIXES.md
ID reuse: ENG4's fix rounds (F1, F2) continue the ENG4 lane and share its id

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng4`
(branch `eng4`, at the commit carrying this order). The writable set is ENG4's, with these differences:
- the engine files you may change are `tensor_cuda/src/matmul.cu` (the predicate and the forward's K = 0 path) and
  `tensor_cuda/src/ops.cpp` (only `Tensor matmul`'s gradient function, for Y2). `NDArray::zeros` and every other core
  member in `kernels.cu` stay unchanged;
- tests: `tensor_cuda/tests/test_eng4_matmul.py` only;
- the new build arm is `tensor_cuda/build-eng4/F2/`, mutant arms `tensor_cuda/build-eng4/m09_f2/` and `m10_f2/`;
- receipts go under `artifacts/eng4/f2/`; new GPU lines go in `artifacts/eng4/GPU_LINES_F2.json`, with new names;
- the installed worktree module becomes F2 at the end of the round.

**Read-only:** `build-eng4/B/`, `E/`, `F1/` and every existing mutant arm; `GPU_LINES.json`, `GPU_LINES_F1.json`, every
directory under `artifacts/eng4/gpu/` and `artifacts/eng4/f1/gpu/`, and every script those lines run. If an F2 line
needs other logic, add a new script. The review worktrees `tensor_fork_apa_wt_eng4rev1` and `…_eng4rev2` are read-only.

## GPU: you run none, not even as a smoke

You run no GPU work at all. A "smoke" of a staged line is its `FORGE_GPU_DRY_RUN=1` dry run (exit 78) and nothing else.
Never call a staged line's step function directly. Never run code that removes, clears or overrides
`CUDA_VISIBLE_DEVICES` outside `forge-gpu-run`. ENG4-F1 broke this rule by calling step functions as CPU smokes. Those
functions pop the variable by design, so the GPU became visible outside the lease.

## Why

The Codex review ENG4-REV2 (run 20261123T142624Z-4153772) was stopped by the provider's content filter before its
final report. Its `findings.json` (in `tensor_fork_apa_wt_eng4rev2/artifacts/eng4_rev2/`) has three findings, none
blocking. Under ENG4-REV2's finding rule they are hardening: one fix, lead-verified, no re-review of the fix itself.
- **F01 (MAJOR):** `matmul_operand_plan` multiplies A's leading extents in signed int64 before its `INT_MAX` check. A
  zero-element tensor can be reshaped to leading extents such as 2^32 × 2^32 × 0. The product then exceeds int64 before
  the check (signed overflow), even though the output allocation bound keeps such shapes away from the GEMM.
- **F02 (MAJOR):** for an accepted forward with batch 0 and a shared, nonempty B (classes a and c), the backward's dB
  intermediate has zero elements. `reduce_to` (`kernels.cu`) then fails to reshape it to B's shape, and the gradient
  raises instead of giving B a zero gradient.
- **F03 (MINOR):** F1's K = 0 path returns through `NDArray::zeros`, which discards `cudaMemset`'s status.

## The fixes (binding)

- **Y1 (F01).** The predicate decides first whether any leading extent of A is 0. If so, the batch is 0 and the
  `INT_MAX` batch check passes. Otherwise it multiplies extent by extent, and refuses (`BatchCount`, the same message
  prefix) as soon as the next product would exceed `INT_MAX`. No signed overflow is reachable. The forward and the
  fp32 backward share it, as before.
- **Y2 (F02).** In `Tensor matmul`'s gradient function, when the forward batch is 0 and B is shared (classes a and c),
  B receives a zero gradient of B's own shape and dtype, and A receives its empty gradient. `reduce_to` is unchanged.
  First record what production's backward does on these pairs, as a staged GPU line on Build B (it raises, per the
  review). If production raises there, the change is an intended change from a raise to a zero gradient. List it.
- **Y3 (F03).** The forward's K = 0 path checks that its zero fill succeeded, using the engine's checked CUDA error
  handling (for example `cuda_check_last("matmul zero fill")` right after it), and raises if it did not.
  `NDArray::zeros` itself is unchanged.
- **Y4.** Everything else in ENG4 and F1 stands. No knob.

## Gates

- **J1, CPU** (`CUDA_VISIBLE_DEVICES` empty), on F2, through the public API on host operands:
  - all of F1's H1 tests pass;
  - Y1: leading extents (2^32, 2^32, 0) built by reshaping a zero-element tensor are accepted as batch 0, with no
    overflow. Leading extents (2^16, 2^15) and (2^31) at a nonzero size are refused with the `BatchCount` message, from
    zero-element tensors where that applies;
  - Y1 refuses before the product could overflow: show it with a witness whose naive product exceeds 2^63, where the
    tensor can be built.
- **J2, the clean-checkout run** of `test_eng4_matmul.py`, as F1's H2.
- **J3, CPU suite per ID,** Build B against F2: identical outcomes, except intended changes listed before the after-run.
- **J4, mutants,** written in `artifacts/eng4/f2/MUTANTS_F2.json` before any runs:
  - `m09_f2_naive_product`: Y1 reverted to the plain product (J1, the 2^32 × 2^32 × 0 witness: UBSan is not required.
    The test asserts the accepted batch-0 result and the refusal message, and the mutant must fail one of them. If no
    assertion can see it, say so and record it as unkillable by test, with the reading that shows the overflow);
  - `m10_f2_empty_batch_raise`: Y2 reverted (a staged GPU line; it must fail on the zero-gradient check).
- **J5, staged GPU lines** for F2, in `GPU_LINES_F2.json`, each dry-run once (78), in ENG4's form (≤ 500 s, `--min` ≤ 480,
  write-once paths listed, expected peak host bytes):
  - `bwd0_B`: production's backward on the Y2 pairs (classes a and c, batch 0), recorded;
  - `bwd0_F2`: the same pairs on F2: B's gradient is zeros of B's shape, A's gradient is empty;
  - `g3_F2`: ENG4's G3 grid on F2. Nonzero cells bitwise equal to `artifacts/eng4/gpu/g3_B/` outputs; zero-size cells
    equal NumPy's result;
  - `mut_m10_f2`;
  - `g4_B_01` … `g4_B_10` and `g4_F2_01` … `g4_F2_10`: the suite on GPU for Build B and F2, on this round's test tree.
- **J6, scope:** writes only in the writable set; no GPU use of any kind; no git writes; nothing deleted.

**Registered acceptance:** the seat's part passes when J1 to J4 (CPU parts) and J6 pass and J5 is staged and dry-run
clean. ENG4 with F1 and F2 is ACCEPTED when the lead's GPU lines pass. A short Codex confirm then reads the F2 diff and
answers ENG4-REV2's remaining questions (callers, tests and mutants), without new probes.

## Mechanism & scope

- **Measures:** whether REV2's F01, F02 and F03 are fixed without changing any accepted nonzero-size result.
- **Mechanism doc:** `orders/ENG4_MATMUL_CONTRACT.md`, `orders/ENG4_F1_REV1_FIXES.md` and this order (Y1 to Y4).
- **Exercises:** the predicate's batch arithmetic, the forward's K = 0 path, `Tensor matmul`'s gradient function;
  Builds B and F2, plus two mutant arms; CPU tests; a clean-checkout copy; staged GPU lines.
- **Does not cover:** `reduce_to` and other core members; new broadcasting; host-tensor compute (HOST-COMPUTE-HMM); a
  current-device check; performance; promotion.
- **Required evidence:** the F2 diff against the commit carrying this order; F2 and mutant build receipts with module
  sha256s; J1 to J4 results; `INTENDED_CHANGES_F2.json` (written before the after-run); `GPU_LINES_F2.json` with
  dry-run exits; `artifacts/eng4/f2/REPORT_F2.md`.
- **Operand census:** Y1's leading-extent cases (a zero anywhere, products just under and over `INT_MAX`, a naive
  product over 2^63); Y2's pairs (classes a and c at batch 0, `trans_b` both ways, three dtypes); the `CUDA_VISIBLE_DEVICES`
  overrides in every script this round runs or adds (L58).
- **Baseline isolation:** Build B (production `f7282b1d`) for results; F1 for the predicate's other refusals.
- **Forbidden configuration:** any change to a nonzero-size accepted result; changes to `NDArray::zeros`, `reduce_to`
  or other core members; any GPU use; editing the B, E or F1 lines or their scripts; edits outside the writable set;
  git writes; deleting anything yourself.

## Machine rules

As in ENG4: one CPU-busy process at a time at `nice -n 19`; builds and jobs over 1 GB in their own scope with
`MemoryMax=12G` and `OOMPolicy=continue`, started only with `MemAvailable` ≥ 12 GB; stop under 40 GB free on
`/mnt/ForgeRealm`. The ceiling is 3 hours of wall time.

## Registration checklist (order-lint)

- **L01:** the instruments are ENG4's. Re-run their `--version` checks once.
- **L02:** GPU lines carry an explicit `--min`.
- **L04:** no fallback. Y1 refuses; Y2 gives B the zero gradient its forward implies (an empty sum is 0).
- **L06:** n/a. No count threshold; `MemAvailable` ≥ 12 GB is a memory floor.
- **L09:** `g3_F2` compares per element on same-width unsigned views after dtype and shape checks. Zero-size cells
  compare against NumPy: empty arrays by shape and dtype, zeros by value. `bwd0_F2` compares B's gradient with zeros by
  value, after dtype and shape checks.
- **L10:** nonzero-size accepted results are held to identity with Build B.
- **L12:** inherited gates: F1's H1 and the suite per ID. Intended changes are listed before the after-run.
- **L16:** the cheapest passing work would fix Y3 and leave Y1's product and Y2's raise. Excluded by J1's Y1 witnesses,
  `bwd0_F2`, `m09_f2` and `m10_f2`.
- **L18:** n/a. No owner requirement behind an option; no knob (Y4).
- **L19:** every bitwise comparison first checks that each side is nonzero and non-constant, at full width (zero-size
  and zero-gradient checks use L09's rule instead).
- **L21:** public remote: no personal names, no raw quotes, home paths as `$HOME/...`.
- **L25:** `g3_F2`'s sides run the same cuBLAS calls (the GEMM block is unchanged), each in its own process on the same
  device, with no host thread pool on the path. The grid and production call shapes are as in ENG4-F1's L25.
- **L32:** n/a. "Overflow" here is integer overflow of the batch product; there is no bounded buffer. The peak host
  bytes in `GPU_LINES_F2.json` are estimates for the lead's scope, not a figure handed to a consumer.
- **L48:** each staged line runs as written, through its `run` wrapper, from the worktree root. J1 also drives every
  post-device step of `g3_F2`, `bwd0_B`, `bwd0_F2` and `mut_m10_f2` (readback, comparators, verdict code) on
  constructed returns of each dtype and shape class, including the zero-size, empty and largest cells. List those
  steps with a terminal label each in `artifacts/eng4/f2/POST_DEVICE_STEPS.json`.
- **L81:** Y2's "production raises" comes from ENG4-REV2's reading and its CPU witness log
  (`tensor_fork_apa_wt_eng4rev2/artifacts/eng4_rev2/CPU_WITNESSES.log`, Build F1 on CPU). Source records: the
  gradient call on batch-0 pairs with a shared B, and the exception it raised. Build B's behaviour is recorded by
  `bwd0_B` before any claim about it; the order asserts nothing about Build B beyond that record.
- **L92:** n/a. There is no finite-difference check; `bwd0_F2` checks an exact zero gradient by value.
- **L35:** registrations may fix cases and seeds; they may not narrow anything a gate covers. If a clause cannot hold,
  stop and report RED.
- **L37:** n/a. No lever, knob or spec key.
- **L38:** existing tests are unchanged except `test_eng4_matmul.py`, listed.
- **L40:** pytest with `-p no:cacheprovider`, `--basetemp` and `TMPDIR` under `/tmp/eng4_*`; TERRAIN-TEST-WRITES as in
  ENG4.
- **L41:** each mutant names its expected failing assertion before it runs. Show that the killing test executes the
  mutated line.
- **L45:** staged lines record `forge-gpu-run`'s exit, the inner `timeout` exit and the log's sha256.
- **L47:** the mutants' expected failures come from this order.
- **L55:** Y1 holds only if the predicate's batch equals the forward's output batch shape product. Show one witness each
  for a zero leading extent, a product equal to `INT_MAX`, and one over it.
- **L56:** (a) a slip in running your own tools goes under Deviations; (b) an exception in your own scripts is repaired
  at most 3 times, each recorded.
- **L58:** CUDA_VISIBLE_DEVICES overrides: census every pop, del, unset or override in the scripts this round runs or
  adds (`grep -n "CUDA_VISIBLE_DEVICES" scripts/eng4_*.py`), list each site and its guard (it runs only inside
  `forge-gpu-run`), and never call one outside it. J1 runs with `CUDA_VISIBLE_DEVICES` empty, and no child removes it.
- **L67:** measured at `e9b0ffb89`: one module build arm is 46–65 MB, and the receipts under `artifacts/eng4/` are
  tens of MB. This round adds 3 arms plus receipts, under 0.5 GB. Stop at 40 GB free, checked by you before each build.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** each mutant is run (GPU ones as staged lines). If one cannot fail, say why (J4).
- **L79:** `FORGE_GPU_DRY_RUN=1` returns 78. Use it once per staged line.
- **L99:** the lead's readings at `e9b0ffb89`, run before sealing:
  - `matmul.cu`: the predicate multiplies `p.batch *= a.shape[d]` over A's leading dimensions, then checks
    `p.batch > INT_MAX`;
  - `matmul.cu`: the K = 0 path is `return NDArray::zeros(plan.out_shape, a.dtype, a.device);`;
  - `ops.cpp`: `Tensor matmul`'s gradient computes dB with `matmul_backward` over A's batch. For class (c) it calls
    `reduce_to(grad_b, weight.shape)` itself; for class (a) `accumulate_grad` reduces it to B's shape through its own
    `reduce_to` path. Both paths meet the empty-batch case.
- **L103:** each staged line refuses, before the lease, a module whose sha256 differs from its build receipt.
- **L104:** no ratio bar; every gate is exact.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L114:** big jobs in the seat's own scope (Claude seat). Staged GPU lines set no RLIMIT_AS; the lead runs them in a
  `MemoryMax=12G` scope, and `GPU_LINES_F2.json` lists each line's expected peak host bytes.
- **L118:** the consumers of the changed code are ENG4's census rows. Re-check only callers that can pass batch 0 with a
  shared B (none expected among product callers).
- **L119:** n/a. The diff is product code and its test. Findings follow ENG4-REV2's rule.

## Rules

- This order is immutable. NO git writes (read-only git, including `git archive` for J2, is allowed). NO subagents.
  No background waits; every command under an explicit timeout.
- No deletes, in the leave-and-list form. Never kill a process you did not start, and never touch the lead's lines.
- **Report:** write `artifacts/eng4/f2/REPORT_F2.md`. If the write is refused, your final message is the report.
- **Prior Art Directive:** annotate each code site and the report. Known: checked multiplication before overflow (CERT
  C INT32-C; GCC/Clang `__builtin_mul_overflow`); the gradient of a broadcast operand as a sum over the broadcast axes,
  which is zero over an empty axis (standard reverse-mode autodiff, e.g. PyTorch's `sum_to_size`); NumPy's empty sum.

## Done

Your final message gives, verbatim: J1 to J6 (PASS, FAIL or STAGED); the diff summary; every module sha256; the
intended changes; the mutant table; each staged line with its dry-run exit; the L58 census; deviations; every file
written.
