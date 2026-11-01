# GN1 amendment A7: acceptance rests on per-line device evidence; a mode declaration can only demote

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1`
(branch `gn1`, at the commit carrying this amendment). Edits and CPU runs there are AUTHORIZED, limited to:
- `artifacts/gn1/a7/`: the new tooling copy, arms, receipts, ledger and report;
- `logs/`, and scratch under `/tmp/gn1_a7_*`. Point `TMPDIR` at it for every command.

**Read-only:**
- the product (`nn.py` at `cf922891`'s bytes) and the test file;
- `artifacts/gn1/` outside `a7/`: every earlier round and review, all frozen;
- the pinned module and `tensor_cuda/build-gn1/`, with no rebuild;
- every other worktree, and production.

Read-only git is allowed. You may run the installed `forge-gpu-run` only on CPU children, with its summary, window and
lock redirected to scratch. Never touch the real lock or window, and leave `/mnt/ForgeRealm/grapa_run7` strictly
alone.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## Why

A6 (`orders/GN1_A6_REV5_FIXES.md`, receipts `artifacts/gn1/a6/`, commit `84641cbb`) fixed REV5's findings. All 321
registered controls pass, and an honest run of all 58 staged lines passes under the pinned programs (P118).

The seat also found and reported an in-scope defect (`a6/D0_MODE_PROMOTION.json`). Editing only the control-mode
metadata of the root and of every parent receipt promotes a CPU fixture root to acceptance. That is 59 edits, with the
seal, the child receipts, the logs, the programs and the argv all unchanged. Under A6's D0(a), a forgery of receipts
short of a coordinated rewrite of everything together with the seal is in scope. So this is a defect. P117 did not
hold. House rule: it is fixed and reviewed again.

## Design (binding; the lead chose these)

- **D0, the threat model:** unchanged from A6's D0.
- **D1, the tooling copy.** Copy `a6/`'s tooling to `a7/` and change it there only. `a6/` stays frozen.
- **D2, acceptance needs positive device evidence on every line.** The scorer accepts a root only when every admitted
  line carries positive evidence that its child ran on the device. That evidence lives in the child's own receipt,
  which the child writes and the parent never rewrites. At least:
  - the child program's resolved path and sha256, equal to the sealed `run_gpu_tests.py`;
  - the device identity the pinned engine reports at run time: the device name and the CUDA runtime and driver
    versions the engine exposes, recorded before the tests run;
  - a non-empty `CUDA_VISIBLE_DEVICES` in the child's received environment.

  A CPU stand-in child cannot produce this evidence, so its lines can never be accepted. Choose the exact fields from
  what the engine exposes, and cite the source lines.
- **D3, mode can only demote.** Any mode field in a parent receipt or in the root (control, audit) can turn acceptance
  off but never on. The scorer derives each line's mode from D2's child evidence, and refuses a line whose parent
  receipt declares a mode that disagrees with that evidence.
- **D4, controls.**
  - **Must be refused:**
    - A6's promotion forgery (`D0_MODE_PROMOTION`), in its original 59-edit form and in every partial form: root
      only, parent receipts only, a subset of lines;
    - a child receipt hand-edited to claim device evidence while the child program identity says stand-in;
    - a stand-in child placed under the real child's path name (sha256 mismatch);
    - device evidence with `CUDA_VISIBLE_DEVICES` empty.
  - **Must be accepted:** a constructed fully honest GPU-shaped root, whose child receipts carry plausible device
    evidence and whose programs all match the seal. If it cannot be built honestly on CPU, document what can be shown
    and how G5 on the GPU will show the rest.

## Gates

- **G1, scope.**
  - The product and the test file are byte-identical to `cf922891`.
  - Every earlier round and review folder is byte-identical before and after.
  - The `a6` to `a7` tooling diff holds only D2 and D3 hunks, each mapped.
- **G2, CPU checks.** As A6, with the same IDs.
- **G3, the module.** Pin and source map unchanged, and every arm copy matches the pin.
- **G4, hostile controls.** A6's 322 targets, the promotion forgery now refused, and every D4 control. Each meets its
  target, with its deciding rule named.
- **G5, mutants on CPU.** Unchanged from A6, re-verified.
- **G6, staging.** The staged script, the line expectations, the regeneration verified against the seal, and one
  audit-mode line. The honest 58-line CPU run passes all line checks, and its root is refused as not a device run.

**Registered acceptance:** ACCEPT when G1 to G6 pass. The lead's GPU run follows a clean GN1-REV6, judged against D0.

## Lead predictions (registered here, before any data)

- **P121:** every G4 target is met, including every partial form of the promotion forgery.
- **P122:** an honest CPU root passes every line check and is refused acceptance only for lack of device evidence.

## Mechanism & scope

- **Measures:** whether GN1's verdict can be accepted only by a run whose children actually ran on the device, within
  D0.
- **Mechanism doc:**
  - `artifacts/gn1/a6/D0_MODE_PROMOTION.json`;
  - `artifacts/gn1/a6/REPORT.md`;
  - `orders/GN1_A6_REV5_FIXES.md`.
- **Exercises:**
  - `a7/`'s parent, child and scorer under the promotion forgery and D4's controls;
  - the engine's device-identity calls, read from source and probed with the GPU hidden;
  - the regeneration.
- **Does not cover:**
  - D0's out-of-scope items;
  - GPU execution (G5, the lead's run);
  - the product;
  - performance.
- **Required evidence:**
  - the `a6` to `a7` diff with its hunk map;
  - the device-evidence fields and their source lines;
  - the G4 table;
  - the staged script.
- **Operand census:**
  - every mode field in every receipt;
  - every combination of root, parent and child edits in D4;
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

- **Units:** A6 took 2,935 s.
- **Ceiling:** 90 minutes. Every single command stays under 20 minutes.
- If a gate cannot finish inside the ceiling, stop and report `RED-BUDGET` with what was done.

## Registration checklist (order-lint)

- **L01:** the instruments all exist: A6's tooling and receipts, the installed forge-gpu-run, the pinned module, NumPy
  2.4.2, and pytest 9.0.2.
- **L02:** the smoke is G2 and G4 on CPU. The lead's GPU run is the first device execution, and it shows D2's device
  evidence for real.
- **L03:** n/a. The mutants stand in for a coverage bar.
- **L04:** n/a. No builder with fallback rungs.
- **L05:** the owner is the GPU tooling under `artifacts/gn1/`.
- **L06:** n/a. No size threshold.
- **L07:** n/a. Nothing visual.
- **L08:** the staged lines and the scorer run from `a7/`'s own directory.
- **L09:**
  - modes and device fields are compared as exact strings;
  - programs by path and sha256;
  - argv as exact lists.
- **L12:** the rules this round changes are acceptance (D2) and mode handling (D3). G4 re-runs every earlier control.
- **L14:** n/a. No gain gate.
- **L15:** P121 and P122 are predictions, scored in the report.
- **L16:** the cheapest passing work refuses only A6's exact 59-edit form. D4's partial forms exclude it.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`. Name the runner
  generically.
- **L25:** threads at 1.
- **L30:** the baseline is git commit `cf922891c612effeabc3c5461f3cfcd7f8ac77e5`. Git records its hashes:
  - blob `52e17bd2…` for `tensor_cuda/tensor_cuda/nn.py`;
  - blob `14815dcc…` for `tensor_cuda/tests/test_groupnorm_gn1.py`.

  Both are shown by `git ls-tree cf922891`. A6's last check is `a6/G1_scope.json` line 6. The lead re-checked before
  sealing: `git diff --quiet cf922891 HEAD` over both files gives rc 0, and both are clean in the worktree.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L33:** every staged GPU line runs through the pinned `forge-gpu-run`. This round runs the installed runner only on
  CPU children, redirected to scratch.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh), on CPU. GN1-REV6 is a Claude seat.
- **L38:** n/a. No test changes.
- **L40:** outputs go under `artifacts/gn1/a7/` or `/tmp/gn1_a7_*`.
- **L41:** the mutants are unchanged from A6.
- **L42:** see Budget.
- **L44:** the scorer derives each line's mode from the child's own evidence, never from a declared field.
- **L45:** each line counts only with the parent's record of the child's exit, written after the child exits,
  together with the child's own device evidence.
- **L47:** the defect is `a6/D0_MODE_PROMOTION.json`, and A6's results are in `a6/G4_summary.json`. Requirements on
  new code are marked as requirements.
- **L48:** G4 runs on CPU before any GPU line.
- **L55:** exact equalities: strings, sha256 digests and argv lists.
- **L58:** every CPU run sets `CUDA_VISIBLE_DEVICES` empty, except the constructed GPU-shaped receipts in D4, which are
  data, not runs.
- **L59:** the seal is this order's commit. `START.json` is written by the GPU run's parent before its first line.
- **L64:** reads: the worktree, earlier receipts, and the engine's source for its device-identity calls. The read-only
  list permits them.
- **L65:** leave-and-list form used (Rules).
- **L67:** the GPU lines write a few MB each. The lead's runner stops below 25 GB free on `/mnt/ForgeRealm` and 20 GB
  on `/`.
- **L69:** one reference governs each class:
  - line mode: the child's evidence;
  - programs: the seal;
  - line commands: `gpu_lines.json`.
- **L70:** n/a. No numeric bound between device computations.
- **L71:** n/a. No input evidence is derived.
- **L72:** the round runs its own tooling copy under `a7/`. Its write targets are under `a7/`, `/tmp/gn1_a7_*` and the
  redirected runner scratch. The registered invocations are the gate scripts, the controls and the audit-mode line.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No background waits. Run everything synchronously, each command under an explicit
  timeout.
- **No deletes by you,** in the leave-and-list form. Anything that must go stays where it is and is listed under
  `## Done` for the lead to bin. Scratch goes under `/tmp` (exempt). Never set `DELETE_ME_ROOTS`.
- Never kill a process you did not start. GPU never.
- **Read-only inspection** is always allowed. How long your harness waits between polls is not governed by this order.
- A failed gate is a result. Report it with receipts.
- Prior Art Directive (AGENTS.md): as A6. Add the CUDA runtime device-query API, as the engine exposes it.
- **Write early:** create `artifacts/gn1/a7/REPORT.md` and `artifacts/gn1/a7/LEDGER.md` within your first few steps.

## Done

Your final message contains, verbatim:
- G1 to G6, each PASS or FAIL, with receipts;
- the device-evidence fields with their source lines;
- the tooling diff summary;
- the G4 table;
- P121 and P122, each HELD or NOT HELD;
- every file written, and anything left for the lead to bin.
