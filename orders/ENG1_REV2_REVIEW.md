# ENG1-REV2: narrow cross-model review of amendment A1 (the five REV1 fixes), before the GPU run

YOUR WRITABLE TARGET is
`/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1/artifacts/eng1/rev2/` (branch `eng1`, at the
commit carrying this order). Writing there, CPU runs, and scratch under `/tmp/eng1_rev2_*` are AUTHORIZED. Point
`TMPDIR` at the scratch directory for every command.

**Read-only:** everything else in this worktree, every other worktree, and production. Read-only git is allowed.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## What you review

A Codex seat (gpt-6.1-sol) wrote amendment ENG1-A1, which answers the five findings of ENG1-REV1:
- the order is `orders/ENG1_A1_REV1_FIXES.md`;
- the receipts are commit `7cf46bd5`, under `artifacts/eng1/a1/`;
- the review it answers is `artifacts/eng1/rev1/` (`REPORT.md`, `findings.json`).

A1 did not touch the engine sources, and the candidate module pin is still `3e6d5805…`. **The review is narrow:**
check only that A1 fixed the five findings correctly and introduced nothing new. Do not re-review the six C++ fixes;
REV1 found no fault in them.

## Questions (answer each, with receipts)

1. **R1-F01, the E3 census.**
   - Is the NumPy emulation in `tensor_cuda/tests/test_eng1.py` a faithful emulation of the operation sequence that
     `nn.py` runs, rounding to nearest-even in the half dtype after every op, with float32 where a class upcasts?
   - Check one class end to end against your own emulation.
   - Are the old tolerances now labelled as accuracy targets, and is no E3 ID a gate?
2. **R1-F02, binding subprocesses to the arm.**
   - Does every staged line's child put its arm's pinned package first on the `PYTHONPATH` it passes on?
   - Does the origin preflight refuse a wrong pin before pytest runs?
   - Rerun REV1's `artifacts/eng1/rev1/d4_child_probe.py` scenario on CPU, once per arm.
   - Is the list of 16 excluded source-reading tests (`artifacts/eng1/a1/SOURCE_READING_TESTS.md`) complete? Check it
     with a grep of your own.
3. **R1-F03, the bfloat16 witness and M15.**
   - Does the new non-power-of-two bfloat16 witness, with its derived tolerance, separate the final-cast decision?
   - Would M15 (a bfloat16-only early cast) be killed by it? Re-derive this arithmetically.
4. **R1-F04, the OW = 0 cases and M16.** Do the four new cases have OH > 0 and OW = 0 for each caller, and does M16's
   CPU kill receipt show all four failing?
5. **R1-F05, the caller census.** Spot-check 10 entries of `artifacts/eng1/a1/CALLER_CENSUS.md` against the code.
6. **Nothing new broken.**
   - `git diff 5031c765 7cf46bd5 -- . ':!artifacts/eng1' ':!orders' ':!logs'` touches only `test_eng1.py`.
   - The tooling diff under `artifacts/eng1/tool/` keeps the parent-after-child receipt property.
   - Rerun the CPU subset on the candidate with CUDA hidden. Expect 47 passed.

## Method

- **CPU only:** `CUDA_VISIBLE_DEVICES` empty, `nice -n 19`, one thread, every command under 10 minutes.
- **A finding** has an ID (`R2-Fnn`), a severity (BLOCKER, MAJOR, MINOR or NIT), the file and line, a reproduction or
  a precise argument, and a suggested fix. A question checked and found sound is reported as checked, with what you
  ran.
- **NITs follow the 11-03 rule:** they are fixed once and closed without another review round.
- **Write early:** create `artifacts/eng1/rev2/REPORT.md` and `artifacts/eng1/rev2/findings.json` in your first
  steps. If your tool refuses a large write, keep `findings.json` complete, and put the full report in your final
  message.

## Mechanism & scope

- **Measures:** whether A1's five fixes are correct and complete, and whether A1 changed nothing else.
- **Mechanism doc:**
  - `orders/ENG1_A1_REV1_FIXES.md`;
  - `artifacts/eng1/a1/REPORT.md`;
  - `artifacts/eng1/rev1/REPORT.md`;
  - `artifacts/eng1/rev1/findings.json`.
- **Exercises:**
  - the test-file diff;
  - the emulation, on CPU;
  - the tooling diff;
  - REV1's probe scenario, rerun;
  - the CPU subset.
- **Does not cover:**
  - the six C++ fixes (reviewed in REV1);
  - GPU execution (G5);
  - promotion.
- **Required evidence:**
  - `findings.json` and `REPORT.md`;
  - the scripts behind each check, with their outputs.
- **Operand census:**
  - the five findings;
  - 40 E3 emulation rows (one class checked end to end);
  - 2 arms for the probe;
  - 16 excluded tests;
  - 4 OW = 0 cases;
  - 10 census entries;
  - the 47-ID CPU subset.
- **Baseline isolation:** the review reads frozen receipts. Its runs write only under `rev2/` and scratch.
- **Forbidden configuration:**
  - any write outside `rev2/` and scratch;
  - editing the product, the tests or receipts;
  - GPU use;
  - git writes;
  - deleting anything.

## Registration checklist (order-lint)

- **L01:** the instruments are the receipts above, NumPy, pytest and the pinned modules. All exist at `7cf46bd5`.
- **L05:** findings are attributed to the test file, the tooling or the report.
- **L07:** n/a. Nothing visual.
- **L09:** pins are compared as exact sha256 strings, test IDs as exact strings, and emulated values within a stated
  tolerance.
- **L16:** the cheapest passing review reads A1's report and agrees. Questions 1–4 and 6 require your own runs.
- **L21:** the remote is public. No personal names, no raw quotes, and home paths written as `$HOME/...`.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while you run.
- **L34:** a Claude seat (Opus 5.5) reviews Codex's amendment, which makes the review cross-model.
- **L40:** outputs go under `artifacts/eng1/rev2/` or `/tmp/eng1_rev2_*`.
- **L41:** M15's and M16's witnesses are shown reaching their mutated lines, by trace or by an argument from the call
  path.
- **L42:** budget 90 minutes. If it runs out, report what was done.
- **L45:** your probe runs record each child's exit after it exits.
- **L46:** before your first write, record a sha256 map of the worktree without `artifacts/eng1/rev2/`. Show it
  unchanged at the end.
- **L58:** `CUDA_VISIBLE_DEVICES` is empty for every run.
- **L65:** n/a. You are a Claude Agent-tool seat, and the delete guard covers you. Delete nothing anyway.
- **L80:** the scope diff uses `5031c765..7cf46bd5`, the commits named above.
- **L92:** n/a. No gradient check.
- **L93:** n/a.
- **L47:** the expected values are A1's own receipts: the 47-ID CPU subset (`artifacts/eng1/a1/gates.json`, which
  the lead reran with 47 passed), pin `3e6d5805…` and the 16 excluded tests. All are at `7cf46bd5`.
- **L81:** expectations come from A1's receipts and REV1's probes. Re-derive the M15 separation and the OW = 0
  geometry from the code, not from the receipts.
- **L90:** each set gets one treatment: all five findings, both arms, every one of the 16 exclusions, all four
  callers. Only the census (10 of 37 entries) is sampled, as stated in question 5.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. GPU never. Never kill a process you did not start.
- **Prior Art Directive:** name your sources for any method you use (IEEE 754 rounding, error bounds, mutation
  testing).

## Done

Your final message contains:
- the verdict: ACCEPT (no findings) or REJECT (with the findings);
- every finding with its ID, severity, file:line, reproduction and suggested fix;
- for each of questions 1–6, what you checked and ran;
- the files you wrote.
