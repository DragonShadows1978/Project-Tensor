# APA-PF2-F1: fix APA-PF2-REV1's findings in the packed prefill kernel (Codex; build and CPU; GPU lines staged)

Follows: APA_P1PF2_AMENDMENT_2.md

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2`
(branch `apa-p1pf2`, at the commit carrying this order; PF2's receipts are `abb1770c`, and the lead added `docs/APA.md`
§8 as `22d59897`). Edits, builds and CPU runs are AUTHORIZED, limited to:
- `tensor_cuda/src/apa_spo/packed_prefill.cuh`;
- `scripts/apa_p1pf2_g45.py`, only the visible-row refusal row (F04);
- new `tensor_cuda/tests/test_apa_pf2_nonfinite.py`;
- new `scripts/apa_pf2f1_*.py`, `artifacts/apa_pf2f1/` (new), build directories `artifacts/apa_pf2f1/build_*`,
  `logs/apa_pf2f1_*`;
- scratch under `/tmp/pf2f1_*`. Point `TMPDIR` there.

**Read-only:**
- every other product file;
- `artifacts/apa_p1pf2/` (PF2's receipts and harness, which you may import or copy);
- `docs/APA.md`;
- `tensor_fork_apa_wt_pf2rev1/artifacts/pf2_rev1/` (the review, its findings and reproductions);
- every other worktree;
- production.

Never walk `/`, `/mnt` or `/mnt/Shared` recursively.

## Why and rulings

APA-PF2-REV1, a blind Codex review of the Claude-written packed prefill, is RED: 2 MAJOR and 3 MINOR. The lead ran both
staged reproductions, and both CONFIRMED, 16 of 16 cases each (`artifacts/pf2_rev1/repro/runs/lead-gpu-*.json`).
- **F01:** a NaN or +Inf sink gives a finite prefill output where packed decode gives NaN.
- **F05:** a raw +Inf stored norm gives finite output, where the NaN-norm control stays NaN.

Both break `docs/APA.md` §8's non-finite contract (clauses 2 and 3). Real findings are fixed. The lead ruled:
- **PF2R1-F01 (MAJOR), fix.** A non-finite sink must propagate as packed decode does.
  - Keep the finite reciprocal-and-multiply arithmetic unchanged. Only the guard changes: compute the reciprocal for
    any non-zero denominator, so a NaN or Inf denominator propagates.
  - Finite outputs must stay bitwise as Build PF.
- **PF2R1-F02 (MAJOR), fix.** Widen the diagnostic tail's initializer and loop counter to int64 before the addition
  (`smax + lane`, `j += 32`). Do not add a new refusal.
  - Factor the index arithmetic into one inline helper that host code can call.
  - A CPU test of the helper at S = INT_MAX and INT_MAX − 31 shows no wrap.
- **PF2R1-F03 (MINOR), closed by the lead.** `docs/APA.md` §8 is on this branch now (`22d59897`, from production
  main's P1 merge). Nothing to do.
- **PF2R1-F04 (MINOR), fix.** The visible-row refusal row binds the same exact K, isolates the visible-row mismatch,
  and asserts the visible-rows message and zero device calls. A separate exact-K mismatch row is kept.
- **PF2R1-F05 (MINOR), fix.** At both norm reads, the sequential one and the lane one, a non-finite stored norm becomes
  a quiet NaN before the multiply. That keeps clause 3: every row reading the key goes non-finite. Finite arithmetic is
  unchanged.
- **Coverage (REV1 question 4).** REV1 found 17 of the 28 reachable template combinations with no recorded execution.
  This round runs at least one finite case of each of the 28 combinations, under PF2's G1 comparison rules: masks
  equal to the oracle, and outputs within PF-I3's τ (Amendment 1).

Nothing else changes: no change to finite arithmetic, a margin, the route or the Python surface.

## Gates

- **G1, finite identity:** on Build F1 (this round's build) against Build PF, outputs and masks are bitwise equal on
  every finite case of PF2's G1 grid (4,848 cases), on GPU. Staged as split lines.
- **G2, non-finite:**
  - REV1's two reproduction sets (sink: 16 cases; norm: 16 cases) now give non-finite rows on prefill, matching packed
    decode;
  - the NaN-norm control stays NaN.

  These checks also ship as `test_apa_pf2_nonfinite.py`, which skips with no GPU.
- **G3, index helper:** a CPU test shows no wrap at S = INT_MAX and INT_MAX − 31.
- **G4, coverage:** all 28 template combinations execute at least one finite case and pass PF2's G1 comparison rules.
  The census is in a file.
- **G5, mutants.** Each is built incrementally, and its killing test executes the edit:
  - `pf1m01_sink_guard_positive`: F01 reverted. It must fail G2's sink cases.
  - `pf1m02_norm_finite_only`: F05 reverted. It must fail G2's norm cases.
  - `pf1m03_index_int32`: F02 reverted in the helper. It must fail G3.
- **G6, regression:**
  - SASS: against Build PF, only the `spq_pf::prefill` instantiations change. Use PF2's mapping rules.
  - PF2's G4 refusal rows (with F04's corrected row) and G5 memory rows pass on Build F1. Both are staged.
- **G7, scope:** writes only in the writable set; no git writes; nothing deleted; no GPU.

**Every staged GPU line** runs `forge-gpu-run --min <n> --max 540`. GRAPA's owner-approved pilot holds GPU 0 today,
with 600 s cooldowns. Each line has a launcher that sets its full environment and checks the module sha before the
lease. Split any shape that cannot fit.

**Registered acceptance:** ACCEPT when G3, G5's CPU part and G7 pass, and the lead's GPU lines pass G1, G2, G4, G5 and
G6. A narrow cross-model review of this round's diff follows, by a Claude seat.

## Mechanism & scope

- **Measures:** whether F01, F02, F04 and F05 are fixed with finite behaviour bitwise unchanged, and whether every
  template combination is exercised.
- **Mechanism doc:** `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2rev1/artifacts/pf2_rev1/findings.json`,
  with `docs/APA.md` §8 and this order's rulings.
- **Exercises:**
  - the sink guard;
  - the two norm reads;
  - the tail index helper;
  - the F04 row;
  - the new tests;
  - Build F1 and three mutant builds;
  - the staged GPU lines.
- **Does not cover:**
  - prefill speed (APA-SP-PREFILL-SPEED, a separate item);
  - grouped reuse;
  - non-finite behaviour of the dense route (P1's, unchanged);
  - promotion.
- **Operand census:**
  - every read of the sink, the denominator and the stored norm in `packed_prefill.cuh`;
  - every index computation that adds to S or `smax`;
  - the 28 template combinations.
- **Baseline isolation:** Build PF is the reference for finite behaviour, and packed decode for non-finite behaviour.
- **Required evidence:**
  - the diff;
  - the build and mutant receipts with module sha256s;
  - the CPU results;
  - `COVERAGE.json`;
  - `STAGED_LINES.md`;
  - `REPORT.md`.
- **Forbidden configuration:**
  - any change to finite arithmetic, a margin, the route or the Python surface;
  - edits outside the writable set;
  - GPU use;
  - git writes;
  - deletes.

## Seat

Codex gpt-6.1-sol, xhigh effort.

**Rules:** read `/mnt/Shared/HOUSE_RULES.md`, `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` and `docs/APA.md` §8 first. This
order is immutable.
- No git writes, no subagents, no background waits; every command under a `timeout` (builds under 60 minutes).
- **No deletes:** never `rm`, `rmdir`, `unlink`, `git clean`, `find -delete` or an inline delete call. Leave and list.
- Never call `forge-gpu-run` without `FORGE_GPU_DRY_RUN=1`.
- Builds use at most `-j4` under `nice -n 19`. Stop if MemAvailable falls below 12 GB or `/mnt/ForgeRealm` below 40 GB
  free.
- **Wall:** 8 hours. Your final message is the report; also write `artifacts/apa_pf2f1/REPORT.md`.
- **Prior art:** follow `AGENTS.md`.

## Registration checklist (order-lint)

- **L01:** cmake, nvcc, pytest and python3, each run with `--version` first. Also run `forge-gpu-run` once in dry-run
  mode (exit 78).
- **L02:** every staged line carries an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L09:** G1 compares outputs and masks as same-width unsigned views after dtype and shape checks. Logs and timings
  are never compared.
- **L10:** G1 is the bitwise identity, on PF2's G1 grid shapes and thread settings.
- **L12:** inherited: PF2's G1 comparison rules and PF-I3 τ (Amendment 1), PF2's G4 and G5 rows, and the SASS mapping
  rules, copied by reference.
- **L16:** the cheapest pass would add the guards only where REV1's cases reach. Excluded: G4 runs all 28 template
  combinations, and the operand census lists every sink, denominator and norm read.
- **L19:** compared arrays are checked non-empty and nonconstant first, except structurally constant non-finite rows,
  which are counted.
- **L25:** G1 pins the thread counts and call shapes of PF2's grid.
- **L32:** n/a.
- **L35:** your registration may fix cases and line splits. It may not narrow a gate.
- **L40:** PF2's harness under `scripts/apa_p1pf2_*.py` writes into `artifacts/apa_p1pf2/` by default, and that
  folder is read-only now. Copy what you reuse into `scripts/apa_pf2f1_*.py`, rebind every output path to
  `artifacts/apa_pf2f1/`, and list the output paths in `COPY_DEPS.json`. The F04 edit to `apa_p1pf2_g45.py` is a source
  edit only. Run it through a copy.
- **L41:** each mutant's killing test executes the edit.
- **L45:** the lead records each staged line's exits and log sha256.
- **L47:** the expected failures come from this order.
- **L48:** staged lines run as written, from the worktree root.
- **L55:** G3 uses constructed S = INT_MAX and INT_MAX − 31. G2 uses REV1's constructed cases.
- **L56:** (a) tool slips go under Deviations; (b) at most 3 repairs of your own scripts.
- **L58:** CPU commands run with `CUDA_VISIBLE_DEVICES` empty.
- **L65:** no deletes.
- **L67:** builds go to `artifacts/apa_pf2f1/build_*`. Each mutant build is incremental. Stop below 40 GB free.
- **L69:** one reference per class: finite results come from Build PF (G1); non-finite results from packed decode
  (G2).
- **L79:** staged lines' dry runs stop before the lease (exit 78). Each launcher states its full environment.
- **L82:** `grep -n 'denom\|inv\|norm\|smax\|j += 32' tensor_cuda/src/apa_spo/packed_prefill.cuh` lists the edit sites.
  REV1 cites lines 96, 116, 151 and 162.
- **L99:** the lead's readings:
  - REV1's `findings.json` (final sha `c85a49b7…`);
  - the lead's GPU receipts, sink and norm both 16/16 CONFIRMING;
  - Build PF's sha `c3c27f8651d82390aac89d62ef704e20dade16d38e9ac8f6cf0e3fb456f4f831`.
- **L103:** each staged line refuses, before the lease, a module whose sha256 differs from its build receipt.
- **L104:** units: G1 per case; G2 per reproduction case; G4 per template combination.
- **L109:** helpers run as `python3 -B -P`; none shares a standard-library module name.
- **L111:** n/a.

## Done

Your final message gives, verbatim:
- the diff summary;
- the module sha256s;
- the CPU results (G3, G5's CPU part, G7);
- `COVERAGE.json`'s summary;
- the mutant table;
- `STAGED_LINES.md` in full;
- the deviations;
- every file written.
