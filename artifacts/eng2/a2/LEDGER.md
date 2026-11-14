# ENG2-A2 ledger (seat: Claude Opus 5.5, Agent tool; order orders/ENG2_A2_REV2_FIXES.md, branch eng2 at d44af925)

Times EST, 2026-11-14. No git was run (not even read-only). No GPU: CUDA hidden for every CPU run; the device-mode
dry run stops at the governor (FORGE_GPU_DRY_RUN=1, exit 78). Scratch: /tmp/eng2_a2_* only. Nothing deleted.

## Setup
- 11:40 read the A2 order (sha256 50c35152...4e91), REV2 REPORT.md / findings.json (sha256 45301021...8a772d,
  verified, matches the order), the A1 ledger and its scripts, HOUSE_RULES.md, AGENTS.md, stage.py, evidence.py.
- 11:44 L01 (`preflight_L01.txt`): Python 3.12.3, NumPy 2.4.2, pytest 9.0.2, forge-gpu-run --help rc 0;
  /mnt/ForgeRealm 97,809,694,720 B free; 16 GB available. No-op CPU collection: slip (L56a, Deviation D1) on the
  worktree path from /tmp (pytest importlib mode bound tensor_cuda as a namespace package; rc 2); re-run as A1 did on
  the candidate arm with the package preloaded: 157/298 selected, rc 0.
- 11:45 before-map `worktree_before_a2.json` (13,569 files; artifacts/eng2/snapshot.py, unchanged) and pre-edit
  copies in `before/` (__init__.py, test_eng2.py, dtype_census.py, seal, ids, lines, expected failures, script).
  Seal before A2: 08dcde8b79285f1cd47628af2ac3d5e6d99f4a77cf3a54cfbce1b769d3de34cf (A1's).

## Fixes (diff: `A2_SOURCE.diff`)
- E2R2-F01, `make` (`__init__.py` :69-80): line 70 is the dtype check `_np_dtype(dtype, name=True) in
  _DEVICE_CAST_DTYPE` (non-strings refused with F1's TypeError) BEFORE any shape handling; the cast branch (bfloat16)
  then normalizes the shape (one-line conditional, same rule) and runs unchanged. For every other name, line 75
  (`_np_dtype(dtype)`, F1's refusal of unsupported strings, MF2's registered hunk, text unchanged) now also runs
  before the shape normalization, which moved below it (:76-77). Supported-dtype behaviour is the same expression in
  the same order of NumPy calls. Line positions kept: MF1's hunk stays at :63 (tensor, untouched), MF2's at :75, so no
  re-registration was needed (mutants.json read-only). Two comment lines (:74, :76) carry the A2 note and prior art.
  Choice (ledgered, reasoning): the order's literal minimum (name check first) would still let a side-effecting
  list shape run before the refusal of an unsupported STRING ("float64"); moving the normalization below :75 closes
  that too at no cost to the registered lines. Prior art: validate-arguments-before-side-effects (no specific source
  known to me); REV2's dtype_boundaries.py sentinel (this repo).
- Tests (`test_eng2.py`, additions only, byte prefix kept): `test_a2_shape_sentinel_refused_cpu` (4 factories x A1's
  6 non-strings, REV2's None among them; shape = REV2's list-subclass sentinel holding F1_TOO_BIG; asserts F1's
  TypeError, sentinel calls == [], RNG state unchanged, spy untouched) = 24 IDs; `test_a2_list_shape_supported_cpu`
  (4 factories x 6 supported dtypes x {list, tuple, sentinel} zero-byte [0, 2] shapes on "cpu"; dtype and shape kept,
  the sentinel iterated exactly once) = 72 IDs (L16: a fix that breaks list shapes for supported dtypes fails these;
  no earlier CPU test passed a list shape). Unsupported strings are NOT parametrized in the sentinel test: under MF2
  they would fail and become MF2 killers outside the read-only registration; the A2-G1 copy probes them instead.
- E2R2-F02, `dtype_census.py` (additions only, inserted before the summary): `apa_sp2.RegisteredMargins.lookup`,
  role metadata validation, float32-only contract; probed with the class's own constructor on a sentinel table
  (bulk_bits 2, D 64, scale 1/8) for the 19 strings and 6 non-strings = 25 rows per arm, appended after the 337
  earlier rows. Outcomes: refused / handled (str "float32") / accepted-equal (another object equal to "float32").
  Product behaviour unchanged (apa_sp2.py not edited). Prior art: REV2's lookup probe; otherwise none known to me.

## Registration (written 11:58, before any gate ran; immutable from here)
- R1 IDs: test_eng2.py 298 -> 394 (+96, all CPU-named); CPU subset 157 -> 253; both arms' eng2 ID files equal.
- R2 candidate CPU: 253/253 passed, pytest rc 0.
- R3 baseline CPU: failing IDs == A1's 96 (a1/baseline_cpu.json) + the 24 test_a2_shape_sentinel_refused_cpu IDs = 120
  exactly; the 72 test_a2_list_shape_supported_cpu IDs pass. tool/baseline_expected_failures.json == A1's 142 + those
  24 = 166.
- R4 MF1: failing IDs == its 12 registered killers, each traced through hunk [63, 63]; MF2: 48 killers, [75, 75].
  The stage.apply_mutant hunk check re-derives [63,63] / [75,75] on the new __init__.py (no line shift).
- R5 treatment effect: A1's __init__.py (before/) with the new tests: the 24 sentinel IDs fail, the 72 supported pass.
- R6 A2-G1 (REV2 copies): for zeros/ones/randn/rand with Shape([2,3]) and dtype=None, TypeError with F1's message,
  sentinel calls == [], RNG state unchanged (REV2's assertions inverted to the fixed condition; ledgered); the four
  unsupported strings with the same sentinel shape: TypeError, no __iter__, RNG unchanged. Census: 362 rows per arm,
  the 337 A1 rows a prefix, 25 RegisteredMargins.lookup rows per arm, baseline rows == candidate rows; refused for
  every request except "float32" (handled) and the objects equal to "float32" (accepted-equal).
- R7 A2-G3: scope map shows changes only in __init__.py (make only), test_eng2.py (prefix), dtype_census.py (no line
  removed), the re-staged files under tool/ and a2/**; added only a2/**, tool/gpu_runs/dry_3/**, audit_3/**; nothing
  removed. Dry run dry_3: 21/21 labels ok, start records bound to the new seal. Audit route audit_3: 10 steps ok with
  253 CPU IDs; score_gpu.py exit 1 with "audit run cannot establish GPU acceptance" (designed).

## Runs (in order; all CPU, CUDA hidden except the dry run's CUDA_VISIBLE_DEVICES=0 + FORGE_GPU_DRY_RUN=1)
- 11:47 R5 treatment (`R5_PREFIX_TREATMENT.json`, `receipts/r5_prefix_treatment.log`): scratch copy
  /tmp/eng2_a2_prefix of the A1 candidate arm (its __init__.py == before/ copy, sha256 ce08bd7a...e116) + the A2
  test_eng2.py, `-k test_a2`: 24 failed (all sentinel IDs, at `assert shape.calls == []`), 72 passed. R5 ok.
  First attempt was a harness slip (D2).
- 11:49 `restage.py arms` (copy of a1/restage.py, diff `restage.diff`): 17 arms rewritten; make_arm's hunk check
  re-derived MF1 [63,63] and MF2 [75,75] (no shift). Log `restage_arms.log`.
- 11:50 `cpu_runs.py` (byte copy of a1/cpu_runs.py) baseline candidate MF1 MF2: baseline 120 failed / 253, candidate
  0 / 253, MF1 12 (22 tests traced through :63), MF2 48 (148 traced through :75). Log `cpu_runs.log`.
- 11:50 `restage.py census lines`: eng2 394 (both arms), audit 253, regnorm 578, regdtype 1779; 21 lines,
  166 registered baseline failures of 394.
- 11:51 `restage.py script seal`: STAGED_GPU_COMMANDS.sh and gpu_lines.json byte-identical to before/;
  seal `tool/staging_seal.json` sha256 a9175ff93acf6f8c71a76aadecef916c92077f1ac489993eaec5c6e281bdf28f
  (was 08dcde8b...34cf; old copy in before/). Registration list: a2/baseline_cpu.json, both A1 and A2 orders.
- 11:51 A2-G2 `ag2_check.py` (diff `ag2_check.diff`) -> `A2G2_CPU.json`: G2a 253/253 passed; G2b failed 120 ==
  96 + 24 exactly, 72 supported pass on baseline, expected failures 142 -> 166 == +24; G2c MF1 12/12 traced, hunk
  [63,63] re-derived; MF2 48/48 traced, [75,75]; G2d prefix, +96 IDs, none lost. all_ok True, rc 0.
- 11:51 dry run `g4_dry.py dry_3` (diff `g4_dry.diff`, scratch path only) -> `G4_DRY_dry_3.json`: 21/21 ok (4 roots:
  seal+arms verified, governor 78 not admitted, no child; 17 dependents refused at the dependency check; S1 78 x21).
- 11:52 audit route: `env -u PYTHONOPTIMIZE RUNS=.../tool/gpu_runs/audit_3 TMPDIR=/tmp/eng2_a2_audit timeout 1190s
  bash artifacts/eng2/tool/STAGED_GPU_COMMANDS.sh --audit` -> staged rc 0 (`audit.*`); `RUNS=<same>
  CUDA_VISIBLE_DEVICES= timeout 300s python3 score_gpu.py` -> rc 1, "audit run cannot establish GPU acceptance"
  (designed; `audit_score.*`). `audit_check.py` (diff `audit_check.diff`: root audit_3, 253 IDs, dry_3) ->
  `A2G3_AUDIT_RECEIPT.json`: 11/11 steps ok. (score_gpu.py ran without -B, as staged: tool/__pycache__ refreshed;
  Python cache, exempt, skipped by snapshot.py.)
- 11:53 `mutant_eval_cpu.py` (diff `mutant_eval_cpu.diff`; root audit_3/cpu_mutants; A1's stand-in and pre-A1
  matcher read from a1/, unchanged) -> `MUTANT_EVAL_CPU.json`: MF1 K1/K3 (12/12 killers via evaluate(), traced),
  MF2 K1/K3 (48/48), candidate K2, S1/S2 refused "no registered killer ID failed". all_ok True.
- 11:55 A2-G1 REV2 copies (`rev2_copies/`, diffs `*.py.diff`, outputs `rev2_out/`), env as REV2 (TMPDIR
  /tmp/eng2_a2_review, CUDA hidden, 1 thread, nice 19, timeout 590s):
  - census_review.py rc 0: 362 rows per arm, A1's 337 a prefix; 35 supported/default pairs (17 nontrivial byte
    comparisons) equal; 25 lookup rows per arm, baseline == candidate; outcomes: 22 refused, "float32" handled,
    UserString('float32') and StrEqualUnhashable() accepted-equal. Census receipts copied to
    `L60_dtype_census_{baseline,candidate}.json`.
  - dtype_boundaries.py rc 0: 36 standard non-string refusals, 6 mask pairs, 90 supported str/np.str_/subclass pairs
    equal to baseline; shape sentinel: 20/20 cases (zeros/ones/randn/rand x None + the 4 refused strings) TypeError
    with F1's exact message, hooks [], RNG unchanged, no C++ call; census lookup rows 25/25; apa_sp2.py source equal.
  - dtype_boundaries_asis.py (paths and the source() reader only) rc 1: AssertionError at REV2's defect assertion
    `calls==["shape.__iter__ before dtype guard"] and not stable` (line 101) -- REV2's reproduction no longer holds.
- 11:57 after-map `worktree_after_a2.json` (13,688 files); `scope_check.py` (diff `scope_check.diff`) ->
  `A2G3_SCOPE.json` passed: changed outside a2/ = __init__.py (owner make only), test_eng2.py (prefix),
  dtype_census.py (additions only, 37 inserted lines), staging (seal, ids_eng2/ids_eng2_candidate/ids_audit,
  collect_*eng2*, baseline_expected_failures); arms: __init__.py in 16 arms (not baseline), test_eng2.py in 17;
  added: a2/ 73, dry_3/ 13, audit_3/ 33; removed none; violations none.

## Gates
- A2-G1 PASS (`rev2_out/dtype_boundaries.json`, `rev2_out/census_review.json`, `rev2_out/asis/`).
- A2-G2 PASS (`A2G2_CPU.json`, `MUTANT_EVAL_CPU.json`, `R5_PREFIX_TREATMENT.json`). MF1 killed at :63, no re-registration.
- A2-G3 PASS (`A2G3_SCOPE.json`, `G4_DRY_dry_3.json`, `A2G3_AUDIT_RECEIPT.json`).
- Repairs (L56b): none on any gate.

## Deviations
- D1 (L56a): L01 no-op collection on the worktree test path from /tmp failed (namespace-package binding, rc 2);
  re-run on the candidate arm (A1's method), rc 0. Both in preflight_L01.txt.
- D2 (L56a): R5 harness slip (pytest hookimpl argument named `r`, PluginValidationError); re-run, ok.
- D3: REV2's scripts were changed beyond output-path rebinding (diffs in rev2_copies/): `git show` replaced by file
  reads (no-git rule; base_pkg = bc02d3a1, as A1); census prior rows compared as a prefix (as A1); REV2's two
  defect-presence assertions inverted to the fixed condition, the four refused strings added to the shape probe, and
  lookup-row counts/equality asserted. The paths-only copy was also run (rc 1 at the defect assertion) so the
  unmodified REV2 logic is on record.
- D4: the product fix goes past the order's literal minimum: besides the name check first, the shape
  normalization of the non-cast path moved below :75, so refused strings are also refused before the shape is read.
  Within `make`; MF1/MF2 lines and texts unchanged.
- D5: the registered sentinel test parametrizes non-strings only (24 IDs); the string cases are probed in A2-G1,
  because a string sentinel test would fail under MF2 outside its read-only killer registration.

## Residuals (not fixed; outside this order)
- RegisteredMargins.lookup returns the fp32 margin for non-str objects that compare equal to "float32"
  (UserString("float32"), a str-equal object): recorded as "accepted-equal" in both arms; product behaviour left
  unchanged as the order requires. Its production caller passes q.dtype (a str).
- *.json receipts are git-ignored in this repo (lead: git status --ignored before add).
- Scratch kept (no deletes): /tmp/eng2_a2_{scratch,prefix,stage,cpu,dry,dry_s1,audit,mut,review}.

## Prior art
- Checking arguments before any side effect (validate-then-act): no specific source known to me; unverified -- lead
  to check ("validate arguments before side effects", "fail fast precondition"). Taken: the ordering idea. Ours: the
  line-preserving layout that keeps MF1/MF2's registered hunks.
- REV2's dtype_boundaries.py shape sentinel and lookup sentinel table (this repo, 2026-11): reused as the test
  operand and census probe. Differential testing (McKeeman 1998, as REV2 cited; unverified -- lead to check).
- Mutation testing (DeMillo, Lipton and Sayward 1978): MF1/MF2 kill checks, unchanged registration.
- Manifest diffing (BSD mtree(8), unverified) and difflib for the scope map; SHA-256 (NIST FIPS 180-4) for the seal.
