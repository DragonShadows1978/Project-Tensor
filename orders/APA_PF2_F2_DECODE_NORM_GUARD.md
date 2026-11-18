# APA-PF2-F2: make packed decode keep docs/APA.md §8 clause 3 for a raw non-finite stored norm, as PF2-F1 made prefill (Codex; build and CPU; GPU lines staged)

Follows: none in this directory (it follows APA-PF2-F1, `9c024ab7` on `apa-p1pf2`)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2f2`
(branch `apa-pf2f2`, from `apa-p1pf2` at `9c024ab7`). Edits, builds and CPU runs are AUTHORIZED there, limited to:
- `tensor_cuda/src/apa_spo/packed.cuh`, only the decode kernel's stored-norm load (`norms[tid]=scales[...]`, line 203);
- `tensor_cuda/tests/test_apa_pf2_nonfinite.py`, adding decode cases;
- new `scripts/apa_pf2f2_*.py`, `artifacts/apa_pf2f2/` (new), build directories `artifacts/apa_pf2f2/build_*`,
  `logs/apa_pf2f2_*`;
- scratch under `/tmp/pf2f2_*`. Point `TMPDIR` there.

**Read-only:**
- every other product file, including `packed_prefill.cuh` and the append at `packed.cuh:121-123`;
- `docs/APA.md`;
- `artifacts/apa_pf2f1/` here (PF2-F1's harness, its lines and the lead's GPU receipts, which you may import or copy);
- `tensor_fork_apa_wt_p1pf2/` (the lead's PF2-F1 GPU lines are running there; never write in it);
- every other worktree;
- production.

Never walk `/`, `/mnt` or `/mnt/Shared` recursively.

## Why

`docs/APA.md` §8 clause 3: "A key whose stored norm is non-finite, for any reason, makes every row that reads it
non-finite."

PF2-F1 made packed prefill keep clause 3, by turning a non-finite stored norm into a quiet NaN before the multiply.
The lead's GPU line `G2_norm` (11-18 05:47; `tensor_fork_apa_wt_p1pf2/artifacts/apa_pf2f1/gpu/G2_norm_real.log`) ran
REV1's raw +Inf-norm cases. All 16 cases failed the same check:
- prefill makes every reading row NaN (`inf_prefill_nan_elements` 128 to 1,024);
- packed decode, P1's production path, gives `inf_decode_nan_elements` 0;
- the NaN-norm control is NaN on both.

With a +Inf norm and a negative dot, the decode score is −Inf, so the key's weight is 0 and the row stays finite. The
key is dropped silently.

P1's append (`packed.cuh:123`) already stores a quiet NaN for a non-finite element or an overflowed norm. So a raw +Inf
reaches the store only by a direct write, as REV1's cases do. Clause 3 says "for any reason", so decode must keep it,
and prefill and decode must agree.

## The fix (ruled by the lead)

At the decode kernel's stored-norm load (`packed.cuh:203`), a non-finite value becomes a quiet NaN, as PF2-F1 did in
prefill. Every multiply that reads `norms[]` (lines 208, 211, 215, 216) then propagates NaN.

Nothing else changes: no finite arithmetic, margin, route, Python surface or append change.

## Gates

- **G1, finite identity:** on Build F2 against Build F1 (PF2-F1's build), packed decode outputs are bitwise equal on
  every finite case. Use P1's packed decode identity grid (P1-F3's G1 shapes and thread settings), and PF2's G1 prefill
  grid, since prefill shares the store. GPU lines are staged.
- **G2, non-finite:** REV1's sink and norm reproduction sets, rerun on Build F2. Both prefill and decode give the same
  non-finite row sets. The NaN-norm control stays NaN. The new test cases ship in `test_apa_pf2_nonfinite.py`, which
  skips with no GPU.
- **G3, mutant:** `pf2m01_decode_norm_finite_only` reverts the guard. Built incrementally, its killing test (G2's decode
  norm cases) executes the edit and fails.
- **G4, regression:**
  - SASS: against Build F1, only the packed decode instantiations change;
  - P1's packed decode non-finite tests (clauses 1 and 2) pass on Build F2.
- **G5, scope:** writes only in the writable set; no git writes; nothing deleted; no GPU.

Every staged GPU line runs `forge-gpu-run --min <n> --max 540` with `--min` ≤ 510. Each has a launcher that sets its
full environment and checks the module sha before the lease, and a dry run with exit 78 and `"would_start": true`.

**Registered acceptance:** ACCEPT when G3's CPU part and G5 pass, and the lead's GPU lines pass G1, G2, G3 and G4. A
narrow Claude cross-model review of PF2-F1's and this round's diffs follows, before the PF2 promotion.

## Mechanism & scope

- **Measures:** whether packed decode keeps clause 3 for a raw non-finite stored norm, with finite behaviour bitwise
  unchanged.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2f2/docs/APA.md`
  (§8), with PF2-F1's order `orders/APA_PF2_F1_REV1_FIXES.md`.
- **Exercises:** the decode norm load; the decode and prefill non-finite cases; Build F2 and one mutant build; the
  staged lines.
- **Does not cover:**
  - prefill speed;
  - the append;
  - the dense route;
  - promotion.
- **Operand census:** `grep -n 'norms\[' tensor_cuda/src/apa_spo/packed.cuh`: lines 22, 111, 123, 195, 203, 208, 211,
  215 and 216. Line 203 is the edit site. Lines 208 to 216 read it. Lines 22, 111 and 123 are append and packing,
  read-only.
- **Baseline isolation:** Build F1 is the finite reference. Clause 3 is the non-finite reference.
- **Required evidence:**
  - the diff;
  - the build and mutant receipts with module sha256s;
  - the CPU results;
  - `STAGED_LINES.md`;
  - `REPORT.md`.
- **Forbidden configuration:**
  - any change outside line 203's load and the test file;
  - GPU use;
  - git writes;
  - deletes.

## Seat and rules

Codex gpt-6.2-sol, xhigh effort.

**Rules:** read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is
immutable.
- No git writes, no subagents, no background waits; every command under a `timeout` (builds under 60 minutes).
- **No deletes:** never `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call. Leave and list.
- Never call `forge-gpu-run` without `FORGE_GPU_DRY_RUN=1`.
- **Builds:** at most `-j4` under `nice -n 19`.
- **CPU jobs:** one thread, its own `RLIMIT_AS` of at most 4,000,000,000 bytes; never a systemd scope.
- Stop below 40 GB free on `/mnt/ForgeRealm`.
- **Wall:** 6 hours. Your final message is the report; also write `artifacts/apa_pf2f2/REPORT.md`.
- **Prior art:** follow `AGENTS.md` (IEEE 754 NaN propagation; PF2-F1's guard).

## Registration checklist (order-lint)

- **L01:** cmake, nvcc, pytest and python3, each run with `--version` first. Also run `forge-gpu-run` once in dry-run
  mode (exit 78, `would_start` true).
- **L02:** the first G2 line is the smoke.
- **L03:** n/a. No share is gated.
- **L09:** G1 compares outputs as same-width unsigned views after dtype and shape checks.
- **L10:** G1 is bitwise, on the inherited grids.
- **L12:** inherited: P1-F3's decode identity grid, PF2's G1 grid and REV1's reproduction sets.
- **L16:** the cheapest pass would guard only the cases REV1 constructs. Excluded: the census lists every `norms[]`
  read, and G2 covers both D (64, 128), both dtypes and both G values.
- **L19:** compared arrays are checked non-empty and nonconstant first, except structurally constant non-finite rows,
  which are counted.
- **L35:** your registration may fix cases and line splits. It may not narrow a gate.
- **L40:** PF2-F1's harness writes into `artifacts/apa_pf2f1/` by default. Copy what you reuse into
  `scripts/apa_pf2f2_*.py` and rebind every output to `artifacts/apa_pf2f2/`. List them in `COPY_DEPS.json`.
- **L41:** the mutant's killing test executes the edited load.
- **L45:** the lead records each staged line's exits and log sha256.
- **L48:** staged lines run as written, from the worktree root.
- **L55:** finite identity holds by construction: the guard selects the original value whenever it is finite.
- **L56:** (a) tool slips go under Deviations; (b) at most 3 repairs of your own scripts.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty.
- **L65:** no deletes.
- **L79:** every line has `--min` ≤ `--max 540` − margin 30. Each dry run shows exit 78, no lease and
  `"would_start": true`.
- **L82:** the census above lists every site; only line 203 is edited.
- **L99:** the lead's reading: `G2_norm_real.log`'s 16 `G2_CASE` lines, each failing `raw_inf_decode_row_sets` with
  `inf_decode_nan_elements` 0.
- **L103:** each staged line rechecks the module sha before the lease.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L114:** CPU jobs are bounded by their own `RLIMIT_AS`; none needs systemd in the seat.

## Done

Your final message gives, verbatim:
- the diff;
- the module sha256s;
- the CPU results;
- the mutant row;
- `STAGED_LINES.md` in full;
- the deviations;
- every file written.
