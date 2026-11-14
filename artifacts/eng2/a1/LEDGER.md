# ENG2-A1 ledger (seat: Claude Opus 5.5, Agent tool; order orders/ENG2_A1_REV1_FIXES.md, branch eng2 at 00789558)

Times EST, 2026-11-14. No git was run (not even read-only). No GPU: CUDA hidden for every CPU run; the device-mode
dry run stops at the governor (FORGE_GPU_DRY_RUN=1, exit 78). Scratch: /tmp/eng2_a1_* only. Nothing deleted.

## Setup
- 10:57 read the A1 order, the ENG2 order, REV1 REPORT.md / findings.json (sha256 1938c6e0...1a67, verified),
  HOUSE_RULES.md, AGENTS.md, the ENG2 ledger and the tool.
- 11:04 L01 (`preflight_L01.txt`): Python 3.12.3, NumPy 2.4.2, pytest 9.0.2, forge-gpu-run --help rc 0; no-op CPU
  collection of the candidate arm (121/262 selected, rc 0). /mnt/ForgeRealm free 77,664,841,728 B.
- 11:05 before-map `worktree_before_a1.json` (13,121 files; artifacts/eng2/snapshot.py, unchanged) and pre-edit copies of
  every file this round changes in `before/`.

## Fixes (diff: `A1_SOURCE.diff`)
- E2R-F01 `__init__.py`: `_np_dtype(dtype, name=False)` checks `isinstance(dtype, str)` first (short-circuit, so a
  non-string never reaches `in` / `[]` and none of its __eq__/__hash__ runs); `name=True` returns the string after only
  that check. `tensor` (:60) and `make` (:72) call `_np_dtype(dtype, name=True)` in their `_DEVICE_CAST_DTYPE` test,
  so every non-string gets the registered TypeError before conversion, allocation or a random draw. The message format
  is unchanged. Line counts kept: MF1's hunk stays at :63, MF2's at :75 (mutants.json is read-only and pins them).
  Prior art: NumPy np.dtype TypeError, PyTorch unknown-dtype refusal (unverified -- lead to check); the
  isinstance-before-lookup idiom (no specific source known to me).
- E2R-F05 `functional._causal_mask`: after `_check_causal_lengths`, `tc._np_dtype(dtype, name=True)` (str check) and, for
  any name outside `_DEVICE_CAST_DTYPE`, `tc._np_dtype(dtype)` (F1's refusal), before the cache key and before any tensor.
  Supported names keep today's mask (float16/bfloat16 cast; other supported names the float32 bias). Callers pass
  Tensor.dtype (always one of the six; core.h:30). Prior art: as F1; no other known to me.
- E2R-F03 `evidence.matches`: observed ID and every pattern normalised by `norm_id` and compared whole (file path kept);
  `literal_brackets` also escapes '?' (only '*' is a wildcard). Prior art: fnmatch / POSIX fnmatch(3); no other.
- E2R-F02 `evidence.regression_reasons` pin branch -> `pin_failure_reasons`: exactly one failed phase (the call), report
  `exception_class` == TypeError (pytest call.excinfo.type, recorded by the sealed child), the attempt's JUnit testcase
  unique with exactly one element, a `<failure>`, whose `message` (pytest reprcrash, the terminal exception's
  "Class: message") equals the registered text exactly. No longrepr substring. Signature unchanged (run_gpu_line.py
  calls it with two receipts; the JUnit path is the receipt's own attempt_dir, which evaluate() already bound).
  Prior art: pytest junitxml failure message (unverified -- lead to check "pytest junitxml reprcrash"); exact oracles.
- E2R-F04 `score_gpu.py`: `keyed_rows` keys rows by `norm_id` on both arms, refuses duplicates and other-kind rows, and
  requires each arm's ID set == the registered census (ids_eng2.txt filtered to the writing test functions), whose size
  must be the order's 35 / 20. Hashes compared unchanged. Prior art: keyed equi-join with a uniqueness check (no
  specific source known to me).
- E2R-F05 `dtype_census.py`: 136 rows appended after the 201 earlier rows (which stay a byte-for-byte prefix after
  object-address normalisation): E2R-F01's non-strings on every entry, `_causal_mask` (all strings + non-strings) and
  the three affine NumPy-dtype parameters (unchanged; NumPy semantics recorded, e.g. "f16" -> float128, "u8" ->
  uint64, None -> float64). Receipts `L60_dtype_census_{baseline,candidate}.json` (337 rows each).
- Tests (`test_eng2.py`, additions only, byte prefix kept): `test_a1_nonstring_refused_cpu` (5 entries x 6 objects:
  UserString float32/bfloat16, an unhashable str-equal object, None, np.dtype("float32"), a spy recording
  __hash__/__eq__) and `test_a1_causal_mask_nonstring_refused_cpu` (6 objects; tc.tensor patched to fail, cache
  checked empty). 36 IDs, all CPU.

## Re-staging (`restage.py`, driving the sealed stage.py)
- arms: 17 arms rewritten in place (make_arm's recipe; deletes forbidden, so no fresh directories), file sets checked.
  Changed arm files: baseline test_eng2.py; candidate and 15 mutants __init__.py, functional.py, test_eng2.py.
- CPU runs (`cpu_runs.py` = byte copy of g2/cpu_runs.py): baseline 96 failed / 157, candidate 0 / 157, MF1 12, MF2 48.
- census (stage.step_census unchanged): eng2 298 (both arms), audit 157, regnorm 578, regdtype 1779 (unchanged files).
- lines (step_lines with a1/baseline_cpu.json): 21 lines (gpu_lines.json byte-identical), 142 registered baseline
  failures (106 + 36).
- script (stage.step_script unchanged): STAGED_GPU_COMMANDS.sh byte-identical.
- seal (step_seal, mode "w"; registration list: a1/baseline_cpu.json replaces g2/baseline_cpu.json, A1 order added):
  `tool/staging_seal.json` sha256 08dcde8b79285f1cd47628af2ac3d5e6d99f4a77cf3a54cfbce1b769d3de34cf
  (was 8de89b78...2a72d; old copy in before/).

## Gates
- A-G1 (`receipts/ag1_*`): probe_edges: UserString refused in all 5 entries, spy allocation [], all 4 pin cases refused,
  every killer self-matches (15/15). cross_arm_review: IDs equal True, no differing rows, no errors (35/20).
  census_review: 337 rows/arm, earlier 201 a prefix; _causal_mask refuses the 4 order strings, None, np.float32.
- A-G2 (`AG2_CPU.json`, `MUTANT_EVAL_CPU.json`, `TOOL_CHECKS.json`): all ok.
- A-G3 (`AG3_SCOPE.json`, `G4_DRY_dry_2.json`, `AG3_AUDIT_RECEIPT.json`): all ok.

## Harness copies and repairs (L56)
- REV1 scripts copied to `rev1_copies/` (diff `rev1_copies/REV1_COPIES.diff`): output path -> a1/rev1_out (running
  them unchanged would overwrite REV1's receipts); census_review: `git show` replaced by file reads (no-git rule),
  prior rows compared as a prefix; cross_arm_review: score_gpu's module globals imported into the exec namespace.
  A-G1 repair 1 of 3: cross_arm_review's collection pin 262 -> 298 (its printed label still says 262).
- `g4_dry.py` copy (diff `g4_dry.diff`): scratch /tmp/eng2_dry* -> /tmp/eng2_a1_dry*.
- `cpu_runner_standin.py` (diff `cpu_runner_standin.diff`): the audit stand-in's ENG2_AUDIT assertion replaced (non-audit
  CPU child for the mutant/pin checks).
- A-G3 repair 1 of 3: scope_check.py counted blank/comment lines between added definitions as unowned; they are now
  allowed only if blank or a comment, and listed.
