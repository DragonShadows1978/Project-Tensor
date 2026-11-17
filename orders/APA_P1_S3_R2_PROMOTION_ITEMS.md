# APA-P1-S3 r2: the port's three hardening items (per-device stream-safe tickets, the deterministic early-counter test, the exact-tie test), and the design for making packed storage the default

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod`
(branch `apa-2p-prod`, at the commit carrying this order). This is r2: r1 stopped at its own scope stop (see the S1
note). Edits, CPU builds and CPU runs are AUTHORIZED only in:
- `tensor_cuda/src/apa_spo/eb3.cuh`, `eb4.cuh` and `eb4_tuning.cuh`, for the ticket ownership (S1) and a test-only
  hook for S2;
- `tensor_cuda/src/bindings.cpp`, for S2's test-only hook binding, if one is needed;
- new test files `tensor_cuda/tests/test_apa_p1_*.py`;
- `artifacts/apa_p1_s3_r2/` (new): builds, receipts, staged GPU lines and the design note;
- scratch under `/tmp/apap1s3r2_*`.

**Read-only:**
- every other file in this worktree, including `tensor_cuda/tensor_cuda/` (Python) and `apa_sp2.py`. This round
  changes no public API;
- every other worktree;
- the other projects' repositories (GRAPA, GraftRepositoryMemory, Project-Voice), which S4's caller census only
  reads.

**No GPU:** `CUDA_VISIBLE_DEVICES` is empty for every command. GPU lines are staged for the lead. **Searching:** never
walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`. Search inside named
repo roots only.

## Why

**The owner's 11-02 direction:** APA memory back to the two-pass (2P) level. P1 ported the lab's packed SP path to
production's base, additively and opt-in.
- **The P1 ruling:** `lead/LEAD_RULING_A1_GPU.md`, cdb7d21a.
- **On the GPU:** P149 is bitwise on 13 cases, P150 gives 0 outcome changes across 2,078 IDs, and P151 gives exact
  bytes.

**What stands between P1 and production:**
- **The survey's promotion items** (`$APA2P/artifacts/apa_2p/SURVEY.md` §S3, lines 138-158, with designs, where
  `$APA2P` is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2p`):
  - per-device, stream-safe tickets;
  - a deterministic early-counter test;
  - the exact-tie suite test;
  - a blind red team.
- **The packed default (L18),** shipped as the default, not as an opt-in (orchestrator rule 1).

This round does the first three in code, and designs the fourth. The red team and the default switch follow.

## Sections (binding)

- **S1, per-device tickets under the engine's single-stream contract** (survey row 1, narrowed by
  `artifacts/APA_P1_S3_LEAD_RULING.md`). The engine launches on the CUDA legacy default stream throughout, and the
  ticket code states that contract (`eb3.cuh:683-687`). This round does not add multi-stream support.
  - **The ticket buffers:** `eb3::counters` (`eb3.cuh:688`) and `eb4::counters` (`eb4.cuh:645`) become per CUDA
    device:
    - a map from device ordinal to its own merge buffer and its own choose buffer;
    - each allocated, zeroed and grown on that device;
    - growth synchronizes that device only.
  - **The L2 size:** `eb3::l2_bytes()` (`:700`) is cached per device.
  - **The device:** the device is the operands' device, guarded (current device set and restored) around allocation
    and launch.
  - **A host mutex** covers map lookup and growth.
  - **At the code site,** state the contract: one stream, the legacy default stream; a launch on any other stream is
    out of contract.
  - Kernels, reductions and numerics are unchanged.
  - The preflight census from r1 (`artifacts/apa_p1_s3/TICKET_BOUNDARY.json`, `SCOPE_STOP.json`) may be reused.
- **S2, the deterministic early-counter test** (survey row 2). A test-only two-CTA case (one bh, NC = 2):
  - the delayed CTA's partials are poisoned;
  - chunk 1 is held before its partial stores until chunk 0 signals that every possible merge and output store is
    done;
  - the correct code passes;
  - EB3's mutant 3 (`prev==NC-2` for `prev==NC-1`, `eb3.cuh:464-496`) and EB4's analogue (`eb4.cuh:504-512`) fail
    deterministically.

  The hook compiles in only for tests: a build flag or a test-only binding, off in the default build. Cite the
  survey's design.
- **S3, the exact-tie suite test** (survey row 3). Promote the existing lead fixture unchanged
  (`/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eb4/artifacts/apa_eb4/lead_gate/lead_check.py:59-68`,
  from commit 88cbd784; read-only).
  - **The fixture:** H16/KV8, D128, S2048, bit-identical key blocks 5 and 20, fraction 1/32.
  - **The test requires:** equal fp32 scores at the cutoff; exactly one high block per head; block 5 selected; bitwise
    EB2b/EB4 flags and output.
  - **The control:** the reversed-tie mutant selects block 20 and fails.
- **S4, the default-switch design.** Write `artifacts/apa_p1_s3_r2/DEFAULT_SWITCH_DESIGN.md`. It covers how production's
  SP path makes packed storage the default:
  - **(a) Production's SP entry today:** `apa_sp2.apa_selective_attention_sp(q, k, kq, v, scale, epsilon, *,
    bulk_bits, margins, ...)` (`apa_sp2.py:72`). It is gated by `TC_APA_SP=1`, takes kq from `quantize_sp2_keys`, and
    selects under registered error margins. The packed path (`_apa_spq1_entry`, `_apa_spq4_entry`) takes a delta.
    State exactly how the two selection guarantees differ, and what "the packed path by default" means for each
    argument.
  - **(b) The caller census:** every caller of `apa_selective_attention_sp`, `quantize_sp2_keys` and the raw SP
    `_C` entries, in this repo, GRAPA, GraftRepositoryMemory and Project-Voice. Give file:line and what each passes.
    Paste the grep commands and their output.
  - **(c) The design:** the API, with or without a signature change, the fallbacks for unsupported shapes, the
    opt-out, and the memory and quality evidence that would be required. Name existing evidence where it exists
    (AB-1, EB4), and the measurement each gap needs.
  - **(d) The tests and GPU lines** the implementing round would stage.
- **S5, GPU staging.** Stage S1's two-thread test and S2's and S3's tests as lead GPU lines:
  - each `forge-gpu-run --min 30 --max …` with an inner timeout, on this round's build;
  - expected: pass on the correct build, and fail under each named mutant;
  - dry-run every line as written with `FORGE_GPU_DRY_RUN=1`, so that each exits 78.
- **S6, no regression.** On the CPU:
  - production's existing test outcomes are unchanged (before and after);
  - the build's SASS is identical outside the APA kernels S1 touches. List the changed kernels.

## Gates

- **G1, tickets:**
  - S1 is implemented, and a host-side unit test checks per-device ownership and separate merge and choose slots;
  - a GPU line is staged that runs the packed path from two host threads on the one stream, and checks the outputs
    bitwise against a serial run.
- **G2, early counter:** S2's test passes on the correct build in a CPU-compiled structural check. Its GPU line is
  staged with mutant 3's expected failure.
- **G3, ties:** S3's test is in the suite; its CPU-side fixture checks pass; its GPU line is staged with the reversed
  mutant's expected failure.
- **G4, the design:** S4's note covers (a) to (d), with the census commands and output pasted.
- **G5, staging:** S5's lines, each dry-run to 78.
- **G6, no regression:** S6.
- **G7, scope:** writes only in the writable set. No GPU, no git writes, nothing deleted.

**Registered acceptance:** ACCEPT when G1 to G7 pass. Then the lead's GPU lines; then the default-switch round and one
blind red team over both rounds; then the promotion ticket.

## Mechanism & scope

- **Measures:** whether the ported packed path's ticket state is safe across devices under the engine's single-stream
  contract, and whether its
  early-counter and tie behaviours are pinned by deterministic tests. Also, what making it the default requires.
- **Mechanism doc:**
  `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2p/artifacts/apa_2p/SURVEY.md`
  (§S3), and `lead/LEAD_RULING_A1_GPU.md`.
- **Exercises:** engine edits to ticket ownership, two new tests with test-only hooks, a design note with a
  cross-repository caller census, a CPU build, and GPU staging.
- **Does not cover:**
  - the GPU run (the lead's);
  - the default switch itself;
  - the blind red team;
  - promotion.
- **Required evidence:**
  - the diff;
  - the build receipt;
  - `SASS_DIFF.json`;
  - the test files and their CPU results;
  - `DEFAULT_SWITCH_DESIGN.md`;
  - `STAGED_GPU_LINES.sh` with `EXPECTED.json`, written first, and `DRYRUN.json`;
  - `OUTCOMES_before.json` and `OUTCOMES_after.json`;
  - `REPORT.md`.
- **Source records (L81):**
  - S1 to S3's designs and mutants come from the survey and the EB3/EB4 orders and lead notes it cites;
  - expected GPU outcomes come from those mutants' registered behaviour.
- **Operand census:** every caller of the two ticket buffers and of `l2_bytes()`, both ticket buffers, S2's two-CTA case, and
  S3's fixture.
- **Baseline isolation:** A1's build and receipts are read-only. The new build goes to `artifacts/apa_p1_s3_r2/`.
- **Forbidden configuration:**
  - kernel math or reduction changes;
  - public API changes;
  - GPU use;
  - git writes;
  - deleting anything yourself.

## Budget (L42)

- **Units:** one engine change across the APA launch chain, two tests with hooks, one design note with a census, one
  CPU build, and about 6 staged GPU lines.
- **Ceiling:** 180 minutes. Every single command stays under 10 minutes, except the build, which stays under 30.
- If it cannot finish inside the ceiling, stop and report `RED-BUDGET` with what was done.

## Registration checklist (order-lint)

- **L01:** the instruments all exist, checked by running them: the build script that A1 used, `cuobjdump`, pytest,
  and `forge-gpu-run` with `FORGE_GPU_DRY_RUN=1`.
- **L02:** GPU lines are staged with an explicit `--min`.
- **L03:** n/a. No share is gated.
- **L04:** n/a. No builder fallback is gated; fallbacks appear only in S4's design note.
- **L08:** builds and receipts go under `artifacts/apa_p1_s3_r2/`, and scratch under `/tmp/apap1s3r2_*`.
- **L09:** S6 compares test outcomes per ID, and SASS per kernel. Logs are not compared.
- **L10:** n/a. S1 changes buffer ownership only, not any stage's numerics. The GPU tests are bitwise on the same
  build.
- **L16:** the cheapest passing work adds the tests without the ownership change. G1's unit test and the staged
  concurrency line exclude it.
- **L18:** this round changes no shipped default. The packed default is the next round (S4 designs it). The owner was told
  on 11-02, in session, that P1 alone does not change the default.
- **L19:** S3's fixture outputs are compared bitwise against the same build. The test checks that each saved reference
  array is nonzero and non-constant, stored at full bit width.
- **L21:** the remote is public. No personal names, no raw quotes, and home paths as `$HOME/...`.
- **L25:** n/a on the CPU. The GPU tests are bitwise against the same build.
- **L26:** the lead touches nothing in the writable set while the seat runs.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh), CPU only. The red team that follows is Claude, so it is cross-model.
- **L38:** new test files only.
- **L40:** pytest runs with `-p no:cacheprovider`, and `--basetemp` and `TMPDIR` under `/tmp/apap1s3r2_*`.
- **L41:** each mutant's expected failure is named at its test line.
- **L42:** see Budget.
- **L47:** the mutants' expected behaviour comes from the EB3/EB4 receipts the survey cites, never from this round's
  run.
- **L48:** every staged line is dry-run as written (S5), from the stated cwd, with no wrapper. The CPU host records
  each line's first unmet need.
- **L55:** the bitwise checks (S3's flags and output, and the GPU tests' outputs) compare one build against itself on
  the same inputs. S1 changes no kernel math, so identity holds by construction. EB4's lead fixture recorded the
  instance (`lead_check.py:59-68`, positive and negative receipts).
- **L60:** S1 adds a device guard where the ticket buffers are allocated and used. For each such site, census its
  boundary as the change touches it: the current device versus the operands' device, first-use allocation, and growth.
  Record it in `TICKET_BOUNDARY.json`. No other precondition changes.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty.
- **L64:** reads are this worktree, `$APA2P` (the survey and EB receipts), and the other repos for S4's census only.
  The production test suite's reads are its own; record any read outside this worktree that a test run makes.
- **L65:** leave-and-list form for any file you would remove.
- **L72:** the write target is the named files, `artifacts/apa_p1_s3_r2/`, plus scratch.
- **L74:** plain wording. This is engine hardening and tests.
- **L76:** the lead writes nothing in the writable set while the seat runs.
- **L77:** every mutant can fail, and each is staged.
- **L79:** dry runs use `FORGE_GPU_DRY_RUN=1`, each line with an explicit `--min`.
- **L82:** `rg -n -e counters -e l2_bytes tensor_cuda/src/apa_spo/eb3.cuh tensor_cuda/src/apa_spo/eb4.cuh
  tensor_cuda/src/apa_spo/eb4_tuning.cuh` gives these hits:
  - `eb3.cuh:468`, `:688`, `:700`, `:804`, `:848`, `:903`;
  - `eb4.cuh:642`, `:645`, `:735`, `:736`, `:744`, `:756`, `:779`.

  All are in the writable files. r1's census shows that explicit streams would reach `packed_launcher.cuh`,
  `eb2b.cuh`, `launcher.cuh`, `ops.h` and `bindings.cpp`. r2 threads no stream, so none of them is touched. Each
  stays read-only, because the single-stream contract needs no change there.
- **The report:** write `artifacts/apa_p1_s3_r2/REPORT.md` and repeat it in your final message.
- **Prior Art Directive:** the survey's annotations at each code site:
  - CUDA Samples' threadFenceReduction, and the stream-ordered allocator;
  - DeMillo, Lipton and Sayward 1978;
  - Batcher 1968, through EB4's fixture.

## Done

Your final message contains, verbatim:
- G1 to G7, each PASS or FAIL;
- the diff summary and the changed-kernel list;
- the tests and their CPU results;
- the design note's summary and the caller census;
- the staged lines and their dry runs;
- every file written.
