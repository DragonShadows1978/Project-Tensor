# GN1 amendment A8: per-line device evidence and demote-only modes, under the narrowed contract

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1`
(branch `gn1`, at the commit carrying this amendment). Edits and CPU runs there are AUTHORIZED, limited to:
- `artifacts/gn1/a8/`: the tooling copy, arms, receipts, ledger and report;
- `logs/`, and scratch under `/tmp/gn1_a8_*`. Point `TMPDIR` at it for every command.

**Read-only:**
- the product (`nn.py` at `cf922891`'s bytes) and the test file;
- `artifacts/gn1/` outside `a8/`: every earlier round and review, all frozen;
- the pinned module and `tensor_cuda/build-gn1/`, with no rebuild;
- every other worktree, and production.

Read-only git is allowed. You may run the installed `forge-gpu-run` only on CPU children, with its summary, window and
lock redirected to scratch. Never touch the real lock or window, and leave `/mnt/ForgeRealm/grapa_run7` strictly
alone.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## Why

A7 (`orders/GN1_A7_MODE_EVIDENCE.md`) ended early: the model provider stopped the seat. The lead's ruling,
`artifacts/gn1/LEAD_RULING_A7.md` (commit `58a26d5a`), records what A7 established:
- per-line device evidence, built from the CUDA runtime the engine links (`a7/device_evidence.py`,
  `a7/DEVICE_API_SOURCES.json`, `a7/DEVICE_HIDDEN_PROBE.json`);
- G2 and G5 passing on CPU.

The ruling also changes the scorer's contract (C below). This round finishes A7's design under C. It is a short,
ordinary verification round: no new process-interruption or receipt-editing harnesses are needed or wanted.

## Contract (C, binding)

- **The scorer's job:** to give the right verdict on every root that an honest run of the staged script can produce.
  That includes:
  - wrong programs, a wrong PATH or `FORGE_GPU_*` environment, or a wrapped runner;
  - a hidden or missing GPU;
  - interruptions and timeouts;
  - missing, empty, truncated or unparsable receipts left by an abnormal ending.
- **The bar:** it must never accept a root unless every staged line ran the sealed programs on the device and passed.
- **Out of scope:**
  - deliberate edits to well-formed receipts after the run. The lead's GPU wrapper writes a sha256 manifest of the
    run root at run end, checks it, scores those bytes and commits the manifest. That is lead-side and not this
    round's work;
  - A6's out-of-scope items.
- **Earlier rows:** edited-receipt rows from earlier rounds are history. List them, but do not run them as gate
  targets.

## Design (binding; the lead chose these)

- **D1, the tooling copy.** Copy `a7/`'s tooling to `a8/` and change it there only. `a7/` stays frozen. Finish any
  D2 or D3 work A7 left incomplete. The `a6` to `a8` tooling diff holds only D2 and D3 hunks.
- **D2, positive device evidence on every line.**
  - **What the scorer requires:** every admitted line carries evidence, written by the child into its own receipt,
    that it ran on the device:
    - the child program's resolved path and sha256, equal to the sealed `run_gpu_tests.py`;
    - `cudaGetDevice` status 0, with a device ordinal;
    - a non-empty device name from `cudaGetDeviceProperties`;
    - the runtime and driver versions;
    - a non-empty `CUDA_VISIBLE_DEVICES` in the child's environment.
  - **The source:** these come from A7's `device_evidence.py`. Cite its source lines.
  - **The consequence:** a run with the GPU hidden can never be accepted.
- **D3, demote-only modes.** Any mode field in a parent receipt or in the root can turn acceptance off but never on.
  Each line's mode is derived from D2's child evidence. A line whose parent receipt declares a different mode is
  refused.
- **D4, the controls, classified under C.**
  - **Classification:** sort every target in A6's `G4_summary.json` and A7's added controls into one of three
    classes, using the control names, targets and deciding rules in those JSON receipts. Do not open or re-run the
    old control scripts to do this.
    - **(i) run condition:** program, PATH, environment, runner wrapping, interruption or timeout;
    - **(ii) abnormal-ending artifact:** a missing, empty, truncated or unparsable receipt;
    - **(iii) edited well-formed receipt.**
  - **Must be refused, with the deciding rule named:**
    - every class (i) and (ii) target;
    - a CPU run of the staged lines with the GPU hidden (D2);
    - a line whose child program differs from the seal;
    - a line whose parent receipt declares GPU mode while its child evidence says no device (D3).
  - **Class (iii):** listed as history, not run.
  - **The accept path:**
    - call the scorer's line check directly, as a unit test, with a device-evidence record shaped like A7's probe
      output but with status 0 and a device name;
    - the end-to-end acceptance is the lead's GPU run (G5 on the device).

## Gates

- **G1, scope.**
  - The product and the test file are byte-identical to `cf922891`. Git records their blobs: `52e17bd2…` for
    `nn.py`, `14815dcc…` for the test file.
  - Every earlier round and review folder is byte-identical before and after.
  - The `a6` to `a8` tooling diff holds only D2 and D3 hunks, each mapped.
- **G2, CPU checks.** As A6, with the same IDs.
- **G3, the module.** Pin and source map unchanged, and every arm copy matches the pin.
- **G4, the controls.** The D4 classification table. Every class (i) and (ii) target, and every new D4 control, meets
  its target, with its deciding rule named. The unit-level accept path passes.
- **G5, mutants on CPU.** Unchanged from A6, re-verified.
- **G6, staging.**
  - the staged script and the line expectations;
  - the regeneration, verified against the seal;
  - one audit-mode line;
  - the honest 58-line CPU run: it passes all line checks, and its root is refused only for lack of device evidence.

**Registered acceptance:** ACCEPT when G1 to G6 pass. The lead's GPU run follows a clean GN1-REV6, judged against
contract C.

## Lead predictions (registered here, before any data)

- **P125:** every class (i) and (ii) target, and every new D4 control, is refused by its named rule.
- **P126:** the honest 58-line CPU root passes every line check and is refused only for lack of device evidence.

## Mechanism & scope

- **Measures:** whether GN1's verdict can be accepted only by a run whose every line ran the sealed programs on the
  device and passed, under contract C.
- **Mechanism doc:**
  - `artifacts/gn1/LEAD_RULING_A7.md`;
  - `orders/GN1_A7_MODE_EVIDENCE.md`;
  - `artifacts/gn1/a7/LEDGER.md`.
- **Exercises:**
  - `a8/`'s parent, child and scorer on CPU, under class (i) and (ii) conditions;
  - the device-evidence query with the GPU hidden;
  - the regeneration.
- **Does not cover:**
  - C's out-of-scope items, including the lead's post-run manifest;
  - GPU execution (G5, the lead's run);
  - the product;
  - performance.
- **Required evidence:**
  - the `a6` to `a8` diff with its hunk map;
  - the device-evidence fields and their source lines;
  - the D4 classification and G4 tables;
  - the staged script.
- **Operand census:**
  - every A6 and A7 target, classified;
  - every mode field;
  - every device-evidence field.
- **Baseline isolation:** the product is frozen at `cf922891`. Earlier receipts are frozen.
- **Forbidden configuration:**
  - any edit outside the writable list;
  - any product or test change;
  - a rebuild;
  - touching the real GPU lock or window;
  - GPU use;
  - git writes;
  - deleting anything yourself.

## Budget (L42)

- **Units:** A6 took 2,935 s. A7 stopped at 1,423 s.
- **Ceiling:** 90 minutes. Every single command stays under 20 minutes.
- If a gate cannot finish inside the ceiling, stop and report `RED-BUDGET` with what was done.

## Registration checklist (order-lint)

- **L01:** the instruments all exist:
  - A6's and A7's tooling and receipts;
  - the installed forge-gpu-run;
  - the pinned module;
  - NumPy 2.4.2 and pytest 9.0.2.
- **L02:** the smoke is G2 and G4 on CPU. The lead's GPU run is the first device execution, and it shows D2's device
  evidence for real.
- **L03:** n/a. The mutants stand in for a coverage bar.
- **L04:** n/a. No builder with fallback rungs.
- **L05:** the owner is the GPU tooling under `artifacts/gn1/`.
- **L06:** n/a. No size threshold.
- **L07:** n/a. Nothing visual.
- **L08:** the staged lines and the scorer run from `a8/`'s own directory.
- **L09:**
  - modes and device fields are compared as exact strings and integers;
  - programs by path and sha256;
  - argv as exact lists.
- **L12:** the rules this round changes are acceptance (D2) and mode handling (D3). G4 re-runs every class (i) and
  (ii) target.
- **L14:** n/a. No gain gate.
- **L15:** P125 and P126 are predictions, scored in the report.
- **L16:** the cheapest passing work moves targets into class (iii) to avoid running them. The classification table,
  with each target's name, target and deciding rule, is checked by GN1-REV6.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`. Name the runner
  generically.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L30:** the baseline is git commit `cf922891c612effeabc3c5461f3cfcd7f8ac77e5`, with the blobs in G1. Both are
  shown by `git ls-tree cf922891`. A6's last check is `a6/G1_scope.json` line 6.
- **L33:** every staged GPU line runs through the pinned `forge-gpu-run`. This round runs the installed runner only on
  CPU children, redirected to scratch.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh), on CPU. GN1-REV6 is a Claude seat, judged against contract C.
- **L38:** n/a. No test changes.
- **L40:** outputs go under `artifacts/gn1/a8/` or `/tmp/gn1_a8_*`.
- **L41:** the mutants are unchanged from A6.
- **L42:** see Budget.
- **L44:** the scorer derives each line's mode from the child's own evidence, never from a declared field.
- **L45:** each line counts only with the parent's record of the child's exit, written after the child exits,
  together with the child's own device evidence.
- **L47:** A6's results are in `a6/G4_summary.json`. A7's state is in `a7/LEDGER.md`. Requirements on new code are
  marked as requirements.
- **L48:** G4 runs on CPU before any GPU line.
- **L55:** exact equalities: strings, integers, sha256 digests and argv lists.
- **L58:** every CPU run sets `CUDA_VISIBLE_DEVICES` empty. The unit-level accept path passes a record as data and
  runs nothing.
- **L59:** the seal is this order's commit. `START.json` is written by the GPU run's parent before its first line.
- **L64:** reads: the worktree, earlier receipts, and the CUDA headers A7 cites. The read-only list permits them.
- **L65:** leave-and-list form used (Rules).
- **L67:** the GPU lines write a few MB each. The lead's runner stops below 25 GB free on `/mnt/ForgeRealm` and 20 GB
  on `/`.
- **L69:** one reference governs each class:
  - line mode: the child's evidence;
  - programs: the seal;
  - line commands: `gpu_lines.json`.
- **L70:** n/a. No numeric bound between device computations.
- **L71:** n/a. No input evidence is derived.
- **L72:** the round runs its own tooling copy under `a8/`. Its write targets are under `a8/`, `/tmp/gn1_a8_*` and the
  redirected runner scratch. The registered invocations are the gate scripts, the controls and the audit-mode line.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No background waits. Run everything synchronously, each command under an explicit
  timeout.
- **No deletes by you,** in the leave-and-list form. Anything that must go stays where it is and is listed under
  `## Done` for the lead to bin. Scratch goes under `/tmp` (exempt). Never set `DELETE_ME_ROOTS`.
- Never kill a process you did not start. GPU never.
- **Read-only inspection** is always allowed. How long your harness waits between polls is not governed by this order.
- **Reading:** read this order, the ruling and A7's `LEDGER.md`. Open other reports only for one specific detail,
  not as background reading. Reuse the existing class (i) and (ii) control scripts by running them. Never open or run
  class (iii) scripts.
- A failed gate is a result. Report it with receipts.
- Prior Art Directive (AGENTS.md): as A7.
- **Write early:** create `artifacts/gn1/a8/REPORT.md` and `artifacts/gn1/a8/LEDGER.md` within your first few steps,
  and update the report after each gate.

## Done

Your final message contains, verbatim:
- G1 to G6, each PASS or FAIL, with receipts;
- the device-evidence fields with their source lines;
- the tooling diff summary;
- the D4 classification and G4 tables;
- P125 and P126, each HELD or NOT HELD;
- every file written, and anything left for the lead to bin.
