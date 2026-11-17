# APA-P1: port the packed SP decode path (INT4 bulk + full-precision tail, 2P's storage) onto production main

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod`
(branch `apa-2p-prod`, at the commit carrying this order; it branches from production main `0e1dea3`). Edits, builds and
CPU runs there are AUTHORIZED, limited to:
- **new files:** `tensor_cuda/src/apa_spo/` (the 16 headers named in D1);
- **additive hunks** in `tensor_cuda/src/kernels.cu`, `tensor_cuda/include/tc/ops.h`, `tensor_cuda/src/bindings.cpp`,
  `tensor_cuda/tensor_cuda/__init__.py` and `tensor_cuda/CMakeLists.txt`;
- **the build outputs:** `tensor_cuda/build-p1/`, plus the module the build drops into the package (git-ignored);
- `artifacts/apa_p1/`: receipts, ledger, report, staged GPU lines;
- `logs/`, and scratch under `/tmp/apa_p1_*`. Point `TMPDIR` at it.

The code under test may manage its own files inside these paths.

**Read-only:**
- every other file in this worktree, including all production tests;
- the lab worktree `…/WorkTree/tensor_fork_apa_wt_apa2p` (the source of the port, at branch `apa-ab1` `aaa9b4ea`;
  read it through `git show aaa9b4ea:<path>`, never its working files, because DS1 is editing them);
- production `/mnt/ForgeRealm/Project-Tensor`, by named reads;
- every other worktree.

**No GPU here.** Set `CUDA_VISIBLE_DEVICES` empty: the lead runs every GPU line in a GRAPA window. **Searching:**
never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.

## Why

The owner's goal (2026-11-02): bring APA's memory back to the two-pass version's. In 2P the bulk stayed packed INT4,
the precision pass stayed full precision, and the kernel fused them.

The survey (`…/tensor_fork_apa_wt_apa2p/artifacts/apa_2p/SURVEY.md`, commit `e5538604`) measured from code at D128:

| Path | Per key per KV head | Bulk stored | Bulk computed |
|---|---:|---|---|
| Production APA-SP | 768 B | bf16 reconstruction | fp32 |
| The lab's packed path (EB3/EB4, AB-1's SP arm) | 580 B | packed TQ4 codes (64 B) plus a norm (4 B) | bf16 hi/lo, fp32 accumulation |
| 2P-INT4 | 580 B | packed INT4 | — |

Both production and the lab path keep exact bf16 K and V (256 B each). AB-1 measured the lab path's quality at the
fp32 floor on Qwen3-1.7B and 4B.

The survey's dry-run merge (S5) found no engine conflicts. This round is S5's steps 1 to 3: the port, unchanged in
numerics, beside production's existing APIs. The promotion items (S3) and the K/V pitch ABI (DS1, on the lab line) are
later rounds.

## Design (binding; the lead chose these)

- **D1, the headers.** Copy these 16 headers from `aaa9b4ea:tensor_cuda/src/apa_spo/` byte for byte:
  - `d1.cuh`, `parallel.cuh`, `tensor.cuh`, `packed.cuh`, `launcher.cuh`, `baseline_dump.cuh`;
  - `merge.cuh`, `packed_launcher.cuh`, `spq4.cuh`, `spq6.cuh`, `eb2b_values.cuh`, `eb2b.cuh`;
  - `eb3_tuning.cuh`, `eb3.cuh`, `eb4_tuning.cuh`, `eb4.cuh`.
- **D2, the seams.** Add only S5 step 3's additive hunks:
  - **`kernels.cu`:** the guarded includes, per the lab tail `:8191-8197`;
  - **`ops.h`:** the declarations from `:302-342`;
  - **`bindings.cpp`:** the packed, EB3 and EB4 bindings from `:986-1064`. Append them, and leave production's
    DLPack, import, export and lifetime code untouched;
  - **`__init__.py`:** the loader, workspace and wrappers;
  - **`CMakeLists.txt`:** the feature defines AB-1's main build used.

  Do not import the K1 AOT tail. Every production line keeps its bytes.
- **D3, nothing else changes.** Production's existing SP/SP2 APIs, its quant/functional fixes, its tests and its
  numerics are untouched. The packed path is a new opt-in entry, not a redirect of the calibrated SP2 API.
- **D4, prior art.** Carry the headers' existing annotations, and add a port note at each seam: Git's three-way merge
  for the dry run (S5), and the existing SPQ1, SPQ4, SPQ6, EB2b, EB3 and EB4 lineage.

## Gates

- **G1, scope.**
  - The `0e1dea3` to working-tree diff is exactly the 16 new headers, byte-identical to `aaa9b4ea`'s, plus additive
    hunks in the five seam files, each mapped to D2.
  - No existing production line changed: check with `git diff --stat` and an inspection for removed lines.
- **G2, the build.** The module builds from `tensor_cuda/build.sh 89` (or its CMake equivalent) into
  `tensor_cuda/build-p1/`, with the lab main build's defines.
  - **Receipts:** the build JSON and the module sha256.
  - **SASS:** for the ported kernels, an instruction-count comparison against AB-1's main build
    (`…/wt_apa2p/artifacts/apa_ab1/build/main/`). Every difference is listed with its cause, and the expected answer
    is none beyond paths.
- **G3, CPU regression.** Production's CPU-runnable tests that import the module (the APA SP/SP2, DLPack and core
  files) give the same per-ID outcomes under the port build as under production's installed module (sha256
  `d6b1b7ad1d87e0d7d5eed70cff94b33e469e6f18b857d78679110e6b2dce3c24`, copied in only for the comparison run).
- **G4, staged GPU lines.** `artifacts/apa_p1/STAGED_GPU_LINES.sh` and `EXPECTED.json`, written before the lead's run,
  with every line through `forge-gpu-run`, `--max` and an inner timeout.
  1. **Bitwise identity, port vs lab.** The packed SP (EB3) and SP-tk8 (EB4) entries and the packed append, on
     identical inputs at S = 2,048, 8,192 and 32,768. The port build and the lab main build produce bit-identical
     outputs, masks or flags, codes, norms and append histories.
  2. **Production regression on GPU.** Production's GPU-run APA and DLPack test files give the same per-ID outcomes
     under the port build as under production's module.
  3. **Bytes.** The packed path's allocation for a cache of S keys equals 580 × S bytes per KV head at D128, plus the
     exact layer-0 bf16 layer, as in AB-1's G2 arithmetic.

**Registered acceptance:** ACCEPT when G1 to G3 pass and G4 is staged. GPU verdicts come from the lead's run. The
promotion items (S3) and a blind cross-model red team come next, then the owner's approval.

## Lead predictions (registered here, before any data)

- **P142:** the port build is bitwise identical to the lab build on every staged case.
- **P143:** no production test outcome changes, on CPU or GPU.

## Mechanism & scope

- **Measures:** whether the lab's packed SP decode path can live in production beside the existing APIs, with its
  numerics unchanged and production's behaviour untouched. That path stores the bulk as INT4 codes beside exact K/V,
  580 B per key per KV head, the same as 2P.
- **Mechanism doc:**
  - `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2p/artifacts/apa_2p/SURVEY.md`;
  - `docs/APA.md`.
- **Exercises:**
  - the port build;
  - production's CPU tests under both modules;
  - the staged GPU lines (bitwise vs lab, production regression, bytes).
- **Does not cover:**
  - the promotion items (S3);
  - the K/V pitch ABI (DS1);
  - a red team;
  - model quality, which AB-1 already measured on the lab build that this round's bitwise gate ties to;
  - performance;
  - promotion itself.
- **Required evidence:**
  - the hunk map;
  - the header identity check;
  - the build JSON and SASS comparison;
  - the G3 table;
  - the staged lines with `EXPECTED.json`.
- **Operand census:**
  - the 16 headers;
  - every seam hunk;
  - every compared test ID;
  - every staged S and arm.
- **Baseline isolation:** production's installed module and the lab main build are the two references. Both are
  read-only.
- **Forbidden configuration:**
  - any write outside the writable list;
  - changing any existing production line or test;
  - any numerical change to the ported code;
  - GPU use;
  - git writes;
  - deleting anything yourself.

## Budget (L42)

- **Units:** a production build takes about 10 minutes. The CPU test files run in minutes.
- **Ceiling:** 3 h. Every single command stays under 30 minutes.
- If a gate cannot finish inside the ceiling, stop and report `RED-BUDGET` with what was done.

## Registration checklist (order-lint)

- **L01:** the instruments all exist:
  - `tensor_cuda/build.sh`;
  - nvcc (CUDA 12.6);
  - the lab branch `aaa9b4ea`, with AB-1's main build;
  - production's module, at the sha256 above;
  - pytest 9.0.2.
- **L02:** the first device execution is the lead's G4 run. G3 is the CPU smoke.
- **L03:** n/a. No coverage bar.
- **L04:** n/a. No builder with fallback rungs.
- **L05:** the owner is the production engine's new packed decode entries.
- **L06:** n/a. No threshold.
- **L07:** n/a. Nothing visual.
- **L08:** tests and staged lines run from the worktree root.
- **L09:**
  - outputs and codes are compared bitwise, through integer views;
  - test outcomes per ID, as exact strings;
  - files by sha256.
- **L11:** G2 compares per-kernel instruction counts against the lab build, and lists every difference with its
  cause.
- **L12:** n/a. No rule changes, only a port.
- **L14:** n/a. No gain gate.
- **L15:** P142 and P143 are predictions, scored after the lead's run.
- **L16:** the cheapest passing work copies the lab's whole engine tree over production. G1's additive-only check and
  the header identity exclude it.
- **L18:** this slice adds the packed path as a separate entry. It is a step, not the shipped state.
  - **The shipped default:** the packed storage for APA-SP decode, per the owner's 2026-11-02 requirement (memory back
    to 2P's).
  - **When:** the default switch is a later round, after the S3 promotion items, the red team and a check of existing
    callers.
  - **Told:** the owner was told in session on 2026-11-02 that P1 alone does not change production's default.
- **L20:** n/a. No parameter is calibrated. δ, row carry and tk8's rule stay at EB3/EB4's registered values.
- **L19:** bitwise dumps save integer views. Each reference is checked to be nonzero and non-constant.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`.
- **L25:** threads at 1 for CPU tests.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L30:** the baselines are commit `0e1dea3` (production, tracked) and `aaa9b4ea` (lab headers), both read through
  git, plus production's module at the sha256 above.
- **L33:** every staged GPU line goes through `forge-gpu-run` with `--max` and an inner timeout.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh). It builds and stages on CPU only.
- **L38:** n/a. No test changes. The bitwise lines are new staged checks, not suite edits.
- **L40:** outputs go under `artifacts/apa_p1/`, `tensor_cuda/build-p1/` or `/tmp/apa_p1_*`.
- **L41:** every test outcome difference, if any, records its message.
- **L42:** see Budget.
- **L45:** each staged line records its exit and outputs after the process exits.
- **L47:** the expected values, with their receipts:
  - layout bytes: AB-1's `artifacts/apa_ab1/g2/result.json` in the lab worktree;
  - lab outputs: produced by the lab main build in the staged line itself, at the lead's run;
  - production outcomes: G3's own baseline run.
- **L48:** the lead runs the staged lines exactly as written.
- **L52:** this round forbids GPU use. G1 to G3 are build and CPU checks, and G4 is staged.
- **L55:** exact equalities.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty for every seat command.
- **L64:** reads: this worktree, the lab branch through git, and production by named paths. The read-only list permits
  them.
- **L65:** leave-and-list form used (Rules).
- **L67:** the build tree takes about 1 GB, and the staged dumps at most about 2 GB. Lines stop below 25 GB free on
  `/mnt/ForgeRealm` and 20 GB on `/`.
- **L69:** one reference governs each class:
  - numerics: the lab main build;
  - production behaviour: production's module;
  - bytes: AB-1's G2 arithmetic.
- **L70:** n/a. The bitwise gate has no tolerance.
- **L72:** the write targets are the 16 new headers, the five seam files, `tensor_cuda/build-p1/`, the package module
  drop, `artifacts/apa_p1/`, `logs/` and `/tmp/apa_p1_*`. The registered invocations are the build, the CPU tests and
  the staging.
- **L73:** n/a. No scorer of receipts.
- **L74:** no harness edits receipts or interrupts a process.
- **L75:** no file moves. D1 copies.
- **L76:** the lead writes nothing inside `artifacts/apa_p1/` until the receipts are committed.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- **APA preflight:** read `docs/APA.md`, `docs/APA_PAPER_DRAFT.md` §2 to §3, and the survey's S2 and S5 before
  porting. State the bulk-operand precision, stored and computed, in the report.
- NO git writes. NO subagents. No background waits. Run everything synchronously, each command under an explicit
  timeout.
- **No deletes by you,** in the leave-and-list form. Never set `DELETE_ME_ROOTS`.
- Never kill a process you did not start. GPU never.
- **Read-only inspection** is always allowed. How long your harness waits between polls is not governed by this order.
- A failed gate is a result. Report it with receipts.
- Prior Art Directive (AGENTS.md): D4.
- **Write early:** create `artifacts/apa_p1/REPORT.md` and `artifacts/apa_p1/LEDGER.md` within your first few steps.

## Done

Your final message contains, verbatim:
- G1 to G3, each PASS or FAIL, with receipts, and G4 staged (yes or no);
- the hunk map and header identity table;
- the SASS comparison summary;
- the G3 outcome table;
- the staged lines with their expectations;
- every file written.
