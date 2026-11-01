# GN1 amendment A5: fix every GN1-REV4 finding (exact-form bound binding, kill-grace ordering, receipts that never crash)

YOUR WRITABLE TARGET is `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1`
(branch `gn1`, at the commit carrying this amendment). Edits and CPU runs there are AUTHORIZED, limited to:
- `artifacts/gn1/a5/`: the new tooling copy, arms, mutants, receipts, ledger and report;
- `logs/`, and scratch under `/tmp/gn1_a5_*`. Point `TMPDIR` at it for every command.

**Read-only:**
- the product (`nn.py` at `cf922891`'s bytes) and the test file;
- `artifacts/gn1/` outside `a5/`: every earlier round and review, all frozen;
- the pinned module and `tensor_cuda/build-gn1/`, with no rebuild;
- every other worktree, and production.

Read-only git is allowed. You may run the installed `forge-gpu-run` only on CPU children, with its summary, window and
lock redirected to scratch as REV2 to REV4 did. Never touch the real lock or window.

**Searching:** never walk `/`, `/mnt` or `/mnt/Shared` recursively, and never enter `/mnt/Shared/ForgeRealm_Link`.
Search inside this worktree, by path.

## Why

GN1-REV4 (`artifacts/gn1/rev4/REPORT.md` and `findings.json`, commit `ef337e40`) is a Claude cross-model re-review.
It found A4 (`a14c0aad`) fixed R3-F01 to R3-F06, and rejected it on 2 MINOR and 2 NIT findings:
- **R4-F01 (MINOR):** D7's token-search bound check accepts commands whose real bounds differ: `--max=900`,
  abbreviated `--mi 1`, `--min=1`, `timeout -k 240 900s`, and `timeout --kill-after 240 900s`.
- **R4-F02 (MINOR):** in the staged setup, the runner's default kill grace (10 s) equals the outer kill delay. An outer
  timeout during a run is then recorded as exit -9, with no runner row and no cause (N2b, N2c).
- **R4-F03 (NIT):** `timeout_cause` can raise in the not-admitted branch (N5), so the parent dies without a receipt.
- **R4-F04 (NIT):** `parent_result.json` is written non-atomically. A truncated one crashes every later parent for its
  label (N4).

House rule: every finding is fixed, NITs included, and the fix is reviewed again.

## Design (binding; the lead chose these, following REV4's proposed fixes)

- **D1, the tooling copy.** Copy `a4/`'s tooling to `a5/` and change it there only. `a4/` stays frozen. The product
  and the test file do not change.
- **D2, exact-form bound binding (R4-F01).** The scorer accepts a line only when two checks pass:
  - the runner arguments are exactly `['--min', str(min), '--max', str(max), '--label', label]`, from that line's
    `gpu_lines.json` row;
  - the inner command is exactly `['timeout', '--kill-after=10s', f'{inner}s', 'python3', 'run_gpu_tests.py', arm,
    suite]`.

  Every other spelling is refused, whether or not the runner's argparse or `timeout` would accept it.
- **D3, kill-grace ordering (R4-F02).**
  - The staged script sets `FORGE_GPU_KILL_GRACE` explicitly. Its value is chosen so that the runner's TERM, grace and
    KILL sequence, plus one poll interval, finishes before the outer `timeout` sends KILL.
  - Do this by raising the outer `--kill-after` above the grace plus the poll. If the outer delay grows, raise the
    script-level `timeout` too, to keep its margin.
  - Exits are normalised before attribution: `rc < 0` becomes `128 - rc`.
  - Show the arithmetic from the runner's source lines (grace 115, TERM/KILL 209-215, poll interval).
- **D4, `timeout_cause` never raises (R4-F03).** It returns `outer-timeout` for any non-start row with exit 124 or 137,
  `unattributed` for anything else it cannot attribute, and the existing causes otherwise. The parent wraps every
  evaluation step, so a problem becomes a recorded reason.
- **D5, atomic parent receipts (R4-F04).** `parent_result.json` is written to a temporary file, then `os.link`ed to
  its final name, which keeps no-overwrite. The prior-attempt check catches read errors and refuses with a JSON error
  line; it never prints a traceback.

## Gates

- **G1, scope.**
  - `nn.py` and the test file are byte-identical to `cf922891`.
  - Every earlier round and review folder is byte-identical before and after.
  - The `a4` to `a5` tooling diff holds only D2 to D5 hunks, each mapped.
- **G2, CPU checks.** The earlier reference checks pass. Both arms import and collect with the GPU hidden, with the
  same IDs as A4.
- **G3, the module.** The pin and source map are unchanged, and every arm's module copy matches the pin.
- **G4, hostile controls.** Each control runs through `a5/`'s real parent and scorer, with the installed runner
  redirected. Each must meet its target, and its deciding rule is named. The set:
  - A4's 131 targets;
  - REV4's five D7 spellings, all refused;
  - N2b and N2c with the grace left unset, as in a pre-D3 staging: refused by D2 or attributed correctly;
  - N2b and N2c under D3's staging: attributed `outer-timeout`, with no orphan;
  - N5: a receipt with a recorded reason;
  - N4: a JSON refusal, with no traceback;
  - one control for each spelling argparse's prefix matching or `timeout` accepts that D2 must refuse. The seat
    enumerates these from the runner's and coreutils' option sets.
- **G5, mutants on CPU.** Unchanged from A4, re-verified against the new tooling.
- **G6, staging.** The staged script, the line expectations, the arm regeneration verified against the seal, and one
  audit-mode line. An honest run of every label passes the scorer.

**Registered acceptance:** ACCEPT when G1 to G6 pass. The lead's GPU run follows a clean GN1-REV5.

## Lead predictions (registered here, before any data)

- **P115:** every G4 control meets its target, REV4's reproductions included.
- **P116:** under D3's staging, no outer-timeout case leaves a surviving child.

## Mechanism & scope

- **Measures:** whether GN1's GPU verdict reports only real kills, real regressions, and runner outcomes bound to the
  exact registered command, under every interruption, with receipts that are always written.
- **Mechanism doc:**
  - `artifacts/gn1/rev4/REPORT.md`;
  - `orders/GN1_A4_REV3_FIXES.md`;
  - `artifacts/gn1/a4/REPORT.md`.
- **Exercises:**
  - `a5/`'s parent, child and scorer under hostile inputs, with the real runner redirected;
  - the D3 arithmetic;
  - the regeneration.
- **Does not cover:**
  - GPU execution;
  - the product;
  - other layers;
  - performance.
- **Required evidence:**
  - the `a4` to `a5` diff with its hunk map;
  - the sha256 maps;
  - a per-finding fix table;
  - the G4 table;
  - the D3 arithmetic;
  - the staged script.
- **Operand census:**
  - every REV4 finding and reproduction;
  - every accepted spelling of the runner's and `timeout`'s options;
  - every exit path, including signal deaths.
- **Baseline isolation:** the product is frozen at `cf922891`. Earlier receipts are frozen.
- **Forbidden configuration:**
  - any edit outside the writable list;
  - any product or test change;
  - symlinks in arms;
  - a rebuild;
  - touching the real GPU lock or window;
  - GPU use;
  - git writes;
  - deleting anything yourself.

## Budget (L42)

- **Units:** A4 took 2,640 s.
- **Ceiling:** 2 h. Every single command stays under 20 minutes.
- If a gate cannot finish inside the ceiling, stop and report `RED-BUDGET` with what was done.

## Registration checklist (order-lint)

- **L01:** the instruments all exist: A4's tooling, REV4's scripts, the installed forge-gpu-run, the pinned module,
  NumPy 2.4.2, and pytest 9.0.2.
- **L02:** the smoke is G2 and G4 on CPU. The lead's GPU run is the first device execution.
- **L03:** n/a. The mutants stand in for a coverage bar.
- **L05:** the owner is the GPU tooling under `artifacts/gn1/`.
- **L06:** n/a. No size threshold. D3's grace values come from the runner's source.
- **L07:** n/a. Nothing visual.
- **L08:** the staged lines and the scorer run from `a5/`'s own directory.
- **L09:**
  - commands are compared as exact argv lists;
  - exits are integers after normalisation;
  - files are compared by sha256.
- **L12:** the rules this round changes are bound binding, the kill-grace staging, cause attribution and receipt
  publication. G4 re-runs every earlier control.
- **L14:** n/a. No gain gate.
- **L15:** P115 and P116 are predictions, scored in the report.
- **L16:** the cheapest passing work special-cases REV4's five spellings. G4's enumerated spelling controls exclude it.
- **L21:** the remote is public. No personal names, no raw quotes, home paths as `$HOME/...`. Name the runner
  generically.
- **L25:** threads at 1.
- **L26:** the lead touches nothing in this worktree while the seat runs.
- **L33:** every staged GPU line runs through `forge-gpu-run` with `--max` and an inner timeout. This round runs the
  installed runner only on CPU children, redirected to scratch.
- **L34:** a Codex seat (gpt-6.1-sol, xhigh), on CPU. GN1-REV5 is a Claude seat.
- **L38:** n/a. No test changes.
- **L40:** outputs go under `artifacts/gn1/a5/` or `/tmp/gn1_a5_*`.
- **L41:** the mutants are unchanged from A4, with witness reach as REV3 and REV4 verified.
- **L42:** see Budget.
- **L44:** the scorer recomputes every verdict, cause and bound check from the raw receipts and the runner's log.
- **L45:** each line counts only with the parent's record of the child's exit, written after the child exits,
  together with the runner's own outcome and the child's start marker.
- **L47:** the findings and reproductions are REV4's receipts under `rev4/`. The runner's lines (grace 115, TERM/KILL
  209-215) are re-read in D3. Requirements on new code are marked as requirements.
- **L48:** G4 runs on CPU before any GPU line.
- **L55:** exact equalities: argv lists, exit integers, runner tokens and sha256 digests.
- **L58:** every CPU run sets `CUDA_VISIBLE_DEVICES` empty.
- **L59:** the seal is this order's commit. `START.json` is written by the GPU run's parent before its first line.
- **L64:** reads: the worktree, earlier receipts, the installed runner's source, and coreutils' documentation or
  `--help`. The read-only list permits them.
- **L65:** leave-and-list form used (Rules).
- **L67:** the GPU lines write a few MB each. The lead's runner stops below 25 GB free on `/mnt/ForgeRealm` and 20 GB
  on `/`.
- **L69:** one reference governs each class:
  - GroupNorm values: the float64 reference;
  - tracked files: `cf922891`;
  - the module: round 1's pin;
  - line commands and bounds: `gpu_lines.json`.
- **L70:** n/a. No bound in this round separates rounding from a defect between device computations.
- **L71:** n/a. No input evidence is derived.
- **L72:** the round runs its own tooling copy under `a5/`. Its write targets are under `a5/`, `/tmp/gn1_a5_*` and the
  redirected runner scratch. The registered invocations are the gate scripts, the controls and the audit-mode line,
  each recorded at launch.

## Rules

- Read `/mnt/Shared/HOUSE_RULES.md` and `/mnt/ForgeRealm/Project-Tensor/AGENTS.md` first. This order is immutable.
- NO git writes. NO subagents. No background waits. Run everything synchronously, each command under an explicit
  timeout.
- **No deletes by you,** in the leave-and-list form. Anything that must go stays where it is and is listed under
  `## Done` for the lead to bin. Scratch goes under `/tmp` (exempt). Never set `DELETE_ME_ROOTS`. The code under test
  may manage its own files inside the writable set.
- Never kill a process you did not start. GPU never.
- **Read-only inspection** is always allowed. How long your harness waits between polls is not governed by this order.
- A failed gate is a result. Report it with receipts.
- Prior Art Directive (AGENTS.md): as A4. Add argparse's `allow_abbrev` and GNU coreutils `timeout(1)`.
- **Write early:** create `artifacts/gn1/a5/REPORT.md` and `artifacts/gn1/a5/LEDGER.md` within your first few steps.

## Done

Your final message contains, verbatim:
- G1 to G6, each PASS or FAIL, with receipts;
- the per-finding fix table;
- the tooling diff summary;
- the G4 table;
- the D3 arithmetic;
- P115 and P116, each HELD or NOT HELD;
- every file written, and anything left for the lead to bin.
