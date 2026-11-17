# APA-P1 amendment A1: port the arena alignment fix, fix the zero-test scorer, extend the bitwise set, restage

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_apa2pprod`
(branch `apa-2p-prod`, at the commit carrying this order). Edits, CPU builds and CPU runs are AUTHORIZED only in:
- the six host arena sites in `tensor_cuda/src/apa_spo/{eb2b,eb3,eb4,launcher,spq4,spq6}.cuh`;
- `artifacts/apa_p1/gpu_driver.py` (the zero-test scoring rule and the bitwise case list only);
- `artifacts/apa_p1_a1/`;
- `logs/`;
- scratch under `/tmp/apa_p1_a1_*`.

**Read-only:**
- everything else in this worktree, including `artifacts/apa_p1/` apart from `gpu_driver.py`, and GPU run 1's receipts
  `artifacts/apa_p1/gpu_20261103T062254Z/`;
- the DS1 worktree `tensor_fork_apa_wt_apa2p`, including:
  - A3's fixed sources at commit ffa547e9;
  - A3's build `artifacts/apa_ds1_a3/build/main/` (`.so` sha256 `4255faa5…`);
  - A3's recurrence proof;
  - A4's rulings;
- every other worktree, and production through read-only git.

**No GPU.** Set `CUDA_VISIBLE_DEVICES` empty for every command. GPU lines are staged for the lead.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.

## Why

APA-P1 ports the lab's packed SP decode path (EB2b/EB3/EB4 and the SPQ kernels) onto production main as an opt-in
entry. Its GPU run 1 (`lead/LEAD_RULING_GPU1.md`) held P142 (6 of 6 bitwise against the lab build) and P143 on 43 test
files. It was RED as registered on one scorer defect: `gpu_driver.py:56` treats pytest's exit 5 ("no tests collected")
as a process failure, so the stub `test_apa_tiled.py`, which defines no tests, failed its line.

Since then the DS1 line found and fixed two pre-existing defects in the same code. The port carries both:
- **The arena alignment defect.** The six workspace arenas align views to 8 bytes while the fused merge reads `pa` with
  16-byte vector loads. 21 of 60 shapes faulted on the GPU with misaligned addresses. Root cause shown by memcheck.
- **The EB4 layout assertion.** Under the 8-byte rule, EB4's size loop pads absent optional views, so with diagnostics
  off it throws its layout assertion whenever B·KV·⌈S/64⌉ is not a multiple of 8.

**The DS1 fix:** 16-byte view alignment, with zero-size entries skipped in the size loop.
- **Source:** DS1 A3, commit ffa547e9.
- **Proof:** a recurrence over 36.2M cases per rule.
- **Kernels:** SASS identical.
- **GPU:** confirmed by A4 (`lead/LEAD_RULING_DS1_A4_GPU1.md`, f9c01984). P144 held; every computed output on the
  previously clean shapes is bitwise unchanged.

## Sections (binding)

- **S1, port the fix.** Apply A3's six host-side changes to the port's six arena functions, as a mechanical port.
  Show it: `diff` each port function body against the corresponding A3 lab function body, and show they now match
  wherever the pre-A1 port matched the pre-A3 lab. Port every difference, or explain it. No device code changes.
- **S2, the scorer.** In `gpu_driver.py`, a test file whose production arm collects zero test IDs (pytest exit 5) is
  scored by comparing the two arms' collected ID sets. Equal sets, including both empty, PASS; anything else is a
  difference. The port arm still runs.
- **S3, the bitwise set.** Keep the existing six cases. Add DS1's four identity shapes `(B, H, KV, S)` =
  `(1,2,1,255)`, `(1,4,2,257)`, `(2,2,1,257)` and `(2,4,2,257)`, and three of A1's previously faulting shapes chosen
  from A3's mismatch classes. The bitwise reference becomes the fixed lab build (A3's `4255faa5…`), not the pre-fix lab
  build.
- **S4, build and identity.** One CPU build of the port (up to two if the first fails for an environmental reason;
  record each).
  - **SASS:** `cuobjdump -sass` of the fixed port equals the pre-A1 port's for every kernel.
  - **CPU checks:** the port's CPU gates (G3: CPU-runnable regression IDs and outcomes) pass as in round 1.
- **S5, registration and staging,** in `artifacts/apa_p1_a1/`.
  - **`REGISTRATION.json`,** written before any staged line exists, with:
    - **P149:** the fixed port is bitwise identical to the fixed lab build on every staged bitwise case, the new
      shapes included;
    - **P150:** no production test outcome changes under the fixed port, with zero-test files compared as ID sets;
    - **P151:** bytes and layout per S follow the fixed recurrence, with no layout assertion on any staged shape,
      diagnostics on or off.
  - **Staged lines:** the bitwise lines (the extended set), the bytes line, and all 44 regression lines. The port
    changed, so they all rerun. Each goes through `forge-gpu-run` with `--max` and an inner timeout, and runs as
    written on the CPU first.
  - **`LEGS.json`:** legs of at most 1,240 s summed `--max`, with 600 s cooldowns.
  - **The re-score:** GPU run 1's `reg_apa_tiled` result is re-scored with the fixed driver from its own directory with
    `--out` into `artifacts/apa_p1_a1/rescore_gpu1/`. It is report only and does not replace the new run.

## Gates

- **G1, the port of the fix:** S1's per-function comparison shows the port matches A3 at all six sites. The tracked diff
  touches only those sites and `gpu_driver.py`.
- **G2, the scorer:** a CPU test of the driver's scoring on constructed pytest outcomes passes:
  - both arms empty: PASS;
  - one arm empty and the other not: difference;
  - normal IDs: unchanged behaviour.
- **G3, build and identity:** S4 passes. SASS is identical for every kernel.
- **G4, staging:** S5's files exist; the registration predates them (by mtime, plus its sha256 in the staged headers);
  every line runs as written on the CPU; `LEGS.json` covers each line once. The post-device code (comparisons and
  verdict) is driven on constructed outputs of the smallest and largest staged shapes (L48 as extended 11-03).
- **G5, scope:** writes only in the writable set. No GPU.

**Registered acceptance:** ACCEPT when G1 to G5 pass. The lead's GPU run scores P149 to P151.

## Mechanism & scope

- **Measures:** whether the production port, carrying DS1's arena fix, is bitwise identical to the fixed lab build on
  the original and new shapes, and leaves production's tests unchanged.
- **Mechanism doc:**
  - `lead/LEAD_RULING_GPU1.md`;
  - DS1's `lead/LEAD_RULING_DS1_A1_GPU.md`, `lead/LEAD_RULING_DS1_A4_GPU1.md` and
    `artifacts/apa_ds1_a3/RECURRENCE.md` (DS1 worktree).
- **Exercises:** a mechanical port of six host functions, a driver rule, one CPU build, a SASS comparison, the CPU
  gates, and staging.
- **Does not cover:**
  - GPU runs;
  - the promotion items S3 (per-device and stream tickets, early-counter and exact-tie tests);
  - the red team;
  - the default switch to packed storage;
  - speed.
- **Required evidence:**
  - the diff;
  - the per-function comparison;
  - the driver test;
  - the build log and module sha256;
  - the SASS comparison;
  - the CPU gate logs;
  - `REGISTRATION.json`, the staged lines, `LEGS.json`;
  - the GPU run 1 re-score;
  - `REPORT.md`.
- **Operand census:**
  - six arena functions;
  - the driver's scoring branches;
  - every staged case and shape.
- **Baseline isolation:** A3's lab build and sources, and GPU run 1's receipts, are read-only.
- **Forbidden configuration:**
  - device-code edits;
  - edits outside the writable set;
  - GPU use;
  - git writes;
  - deleting anything yourself.

## Budget (L42)

- **Units:** six host functions, one driver rule, one build, one SASS comparison, and a restage.
- **Ceiling:** 180 minutes. Every single command stays under 10 minutes, except the build and the SASS dump, which stay
  under 30 minutes each.
- If it cannot finish inside the ceiling, stop and report `RED-BUDGET` with what was done.

## Registration checklist (order-lint)

- **L01:** the instruments all exist: the port's build script (round 1's), `cuobjdump`, pytest, A3's lab build and
  sources.
- **L02:** GPU lines are staged for the lead through `forge-gpu-run` with `--max`.
- **L03:** n/a. No coverage share.
- **L04:** n/a. No fallback rungs. One or two recorded attempts.
- **L05:** each change is attributed to S1 or S2.
- **L06:** n/a. Bitwise.
- **L07:** n/a. Nothing visual.
- **L08:** the re-score and the staged lines write to fresh folders.
- **L09:** bitwise outputs, exact ID sets, exact SASS text.
- **L10:** identity is held on the kernels (SASS). Host layout is the only change.
- **L11:** binaries are named by path and build command. SASS is compared per kernel. Host ELF sections are expected to
  differ.
- **L12:** S2 changes one scoring rule, stated above, for this order only.
- **L14:** n/a. No gain gate.
- **L15:** P149 to P151 are in the registration before staging.
- **L16:** the cheapest passing work ports eb3 only. G1's six-site comparison excludes it.
- **L18:** the owner's requirement here is APA memory back to the two-pass level, through packed storage. In this
  round the shipped default stays production's current SP path; the packed path is an opt-in entry, as in round 1.
  Making packed storage the shipped default is a registered later step before any promotion (round 1's ruling, Next).
  The owner was told in session on 11-02 that the port lands opt-in first and that the default switch follows.
- **L19:** bitwise lines keep their reference arrays as in round 1.
- **L21:** the remote is public. No local paths in records.
- **L25:** n/a. Bitwise.
- **L26:** the lead touches nothing in the writable set while the seat runs.
- **L28:** n/a. Registered seeds.
- **L32:** the arena is a bounded buffer whose size equals its use (A3's proof). P151 restates it for the port's
  staged shapes.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh), CPU only.
- **L35:** the registration keeps every round-1 case and line.
- **L38:** n/a. No test file changes.
- **L40:** the build writes under `artifacts/apa_p1_a1/`. pytest runs with `-p no:cacheprovider` and `--basetemp`
  there. `TMPDIR` points under `/tmp/apa_p1_a1_*`.
- **L41:** n/a. No mutant.
- **L42:** see Budget.
- **L44:** summaries are regenerated from records.
- **L48:** see G4, as extended on 11-03: a static plan-handler check, and readback driven on the smallest and largest
  shapes.
- **L52:** one or two CPU builds.
- **L55:** exact equalities.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty.
- **L59:** registration mtime before the staged files, plus its sha256 in each staged header.
- **L64:** reads are this worktree, the DS1 worktree read-only, and read-only git.
- **L65:** leave-and-list form used (Rules).
- **L67:** staged lines state the bytes they write.
- **L69:** one reference governs each class:
  - correct outputs: the fixed lab build `4255faa5…`;
  - production tests: production's module, as in round 1.
- **L72:** the write targets are those named in the writable set.
- **L73:** the driver scores every honest pytest outcome, zero-collection included.
- **L74:** plain wording.
- **L75:** no file moves.
- **L76:** the lead writes nothing inside `artifacts/apa_p1_a1/` while the seat runs.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- **APA preflight:** read `docs/APA.md` and `docs/APA_PAPER_DRAFT.md` §2 to §3. State the bulk-operand precision,
  stored and computed, in every table that names a path.
- NO git writes. NO subagents. No background waits. Run everything synchronously, each command under an explicit
  timeout.
- **No deletes by you,** in the leave-and-list form. Never set `DELETE_ME_ROOTS`.
- Never kill a process you did not start. GPU never.
- **Prior Art Directive:** cite A3's annotations at the ported sites, or write "no prior art known to me" for anything
  new.
- **Write early:** create `artifacts/apa_p1_a1/REPORT.md` within your first few steps.

## Done

Your final message contains, verbatim:
- G1 to G5, each PASS or FAIL;
- the per-function port comparison;
- the driver test results;
- the SASS result;
- the staged-line summary and `LEGS.json`;
- every file written.
