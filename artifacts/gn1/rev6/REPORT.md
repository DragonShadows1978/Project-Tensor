# GN1-REV6 review report

- **Reviewer:** Claude Opus 5.5 (Agent-tool seat). This is a cross-model review of Codex's A8 (`76f52a05`), under
  contract C.
- **Order:** `orders/GN1_REV6_REVIEW.md` (`81da599b`).
- **Run:** 2026-11-01, 17:46 to about 18:20 EST, CPU only. That is about 35 of the 60 budgeted minutes.

**Note from the lead.** The seat's tool rules forbid report files, so the lead filed its final message here, re-laid
out in sections with its wording kept. `findings.json` is the seat's record.

**Verdict: ACCEPT.** There are no in-scope findings: 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT. Nothing lets the scorer accept
a root unless every line ran the sealed programs on the device and met its expectation. There are 7 observations.

| Q | Verdict |
|---|---|
| Q1 | R5-F01 to R5-F05 FIXED. R5-F04 is also OUT OF SCOPE under C, because its reproduction edits a well-formed receipt. |
| Q2 | SOUND |
| Q3 | SOUND |
| Q4 | CONFIRMED, with qualification R6-O1 |
| Q5 | SOUND |
| Q6 | ACCEPT |

## Observations (not findings)

**R6-O1, a qualification on Q4.** 3 of the 32 class (iii) targets are not run receipts.
- **The targets:** `Import_helper_pin_parent`, `_child` and `_scorer` (`a6/extra_controls.py:38-53`). Each edits the
  digest in `staging_seal.json`, which is a sealed input written before the run. No honest run produces a seal edit,
  and A8's basis text says what they are.
- **Their in-scope counterpart:** a changed import module, which is a wrong program. The reviewer ran it live
  (`rev6_cases_i.json`). START, the parent before launch, and the scorer all refuse it with "import module seal
  mismatch: scripts/apa_om135_core.py".
- **If "receipt" means strictly a file in the run root:** this becomes a NIT against D4's labelling of those 3 rows.
  The fix would be to relabel them, or to move them to class (i) with the live case as the executed target.

**R6-O2, a gap in A8's G4 evidence.** None of A8's runs shows the environment rule refusing a real run. It is not a
tooling defect.
- A8's 6 live environment controls (`Actual_env_FORGE_GPU_*`) ran under a control START.
- Under a control START, `program_binding.reasons` returns before it compares the environment
  (`program_binding.py:72-73`). Their only refusal in A8 is the root device rule.
- The only A8 rows that exercise the environment comparison are the class (iii) `Recorded_env_*` edits, which were not
  run.
- The reviewer's production-mode runs fill the gap. `Q2_live_gpu_hidden_*` (redirected summary, window and lock) and
  `F02_new_live_unknown_var_and_grace18` are both refused with "child environment differs from staged environment".

**R6-O3, what G6/P126 actually runs.** The "honest 58-line CPU run" is the declared CPU control path, not real tests.
- It uses the real runner (redirected), the staged argv exactly, and the real `run_gpu_tests.py` entry.
- `GN1_CPU_FIXTURE=1` sends the child to `rev2_forge.py`, which makes up the test reports. Programs and environment are
  not compared under a control root.
- The reviewer reproduced it: 58 of 58 lines pass their checks, and the root is refused only for absent device
  evidence.

**R6-O4, environment outside C's named scope.**
- Variables other than PATH and `FORGE_GPU_*` (PYTHONPATH, PYTHONHOME, LD_LIBRARY_PATH, LD_PRELOAD) pass unrecorded to
  the runner and the child (`run_gpu_line.py:57-60`).
- The device evidence checks the runtime and driver versions only as positive decimals (`device_evidence.py:97-99`).
  It does not compare them with the runtime 12060 that A7 observed.
- **Optional hardening:** pin `cuda_runtime_version`, or record the LD_* and PYTHON* variables.

**R6-O5, the public-remote screen (L21).** The literal home path, which contains the account name, appears in three
places, unchanged since A6:
- `a8/program_binding.py:16`;
- `a8/STAGED_GPU_COMMANDS.sh:21`;
- the programs block of `a8/staging_seal.json`.

Screen them before any public push.

**R6-O6, reporting quality only.** The read at `evidence.py:441` is new in A8. When `child_result.json` is missing,
`recompute_line` raises, and the row comes from `score_gpu`'s fallback path: the raw "[Errno 2]" text, with no
admission fields. The verdict is still a refusal. The reviewer saw this live in `NEW_grace_21`.

**R6-O7, REV5's own expectations.**
- **The cases:** some REV5 cases expected a line to pass: Q_honest, Q2_control_real_runner_honest, the N5 retry, the
  lock-wait retry, D5_unsealed and D6.
- **Why they cannot pass now:** under the A6-onward design, a stand-in runner, or an inner command with a bare
  interpreter name, is not the pinned program. A production line on CPU has no device evidence.
- **What A8 does:** it refuses them by the pinned-program rules.
- **Not covered by this review:** D5_unsealed's and D6's own rules were not re-read under a control root.
  `runtime_evidence.py` is byte-identical from A6 to A8.

## What the reviewer checked and ran

**Setup.**
- A fresh mirror in `/tmp/gn1_rev6_m/W`, built with A8's own `regenerate_arms.py --verify-seal`
  (`mirror_regen_a8.py`). The seal and all 56 arms verify (`regen_compare.json`).
- `parent_wrap6.py` and `checks_wrap6.py` change only the scratch-directory prefix the guard accepts (`gn1_a8_` becomes
  `gn1_rev6_`), as REV5 did. Everything else is A8's bytes.
- The installed runner always had its summary, window and lock in `/tmp/gn1_rev6_forge`.
- `git diff --quiet cf922891` over `nn.py` and the test file gives rc 0. No tracked file outside `rev6/` changed.

**Q1: REV5's reproductions, re-run against A8's real parent, child and scorer.** Receipts:
`hostile_rev6_{q,c,n,p}.json` and `rev6_probe_{rawexit,postexit}.json`, with the per-case table in `analysis.json`.

| Finding | Result | REV5 reproduction under A8 | New REV6 cases |
|---|---|---|---|
| R5-F01, runner identity | FIXED | Q2_wrapped_runner_command0 is refused by "runner command[0] differs from pinned runner". | A byte-identical copy of the runner at another path, and a wrapped runner with the production inner command. Both are refused by the same rule. |
| R5-F02, PATH and `FORGE_GPU_*` | FIXED | Q2_path_shadow_inner_timeout is refused by "program pin mismatch: timeout". The redirected summary, window and lock are refused by the environment rule. | A shadow `python3` reached through the runner's `/usr/bin/env` shebang is refused by "program pin mismatch: runner_interpreter". A live unknown `FORGE_GPU_` variable plus a kill grace of 18 are recorded and refused. |
| R5-F03, deeply nested JSON | FIXED | Nested parent receipt and START: a JSON refusal, a summary is written, no traceback. | A nested child result on a focus line. REV5's regression case re-run under a control root: it now reaches `regression_reasons`, and the recursion error is recorded. |
| R5-F04, `raw_exit` | FIXED | raw_exit -9 with exit 0 is refused ("normalisation mismatch"). A missing raw_exit is refused. | A live run where the child was killed (-9) is recorded as 137 and accepted as consistent (NEW_grace_21). |
| R5-F05, post-exit seal hash | FIXED | An unreadable seal while the child runs still gives a published receipt with the reason recorded. | Unreadable seal and read-only attempt directory together: a JSON error on stderr, no receipt, and the scorer marks the line incomplete. |

- REV5's spelling cases (Q) and runner-row cases (C, N) still refuse, or record the same causes as REV5 saw.
- The interruption cases N2b, N2c, grace 18 and grace 21 behave as REV5 expected.

**Q2: device evidence (D2). SOUND.**
- **GPU hidden, live:** production lines with the real child, with `CUDA_VISIBLE_DEVICES` empty and with it set to
  "-1".
  - The child records `cudaGetDevice` status 100, no device name, runtime 12060 and driver 13020.
  - The line is mode cpu, with "mode binding: parent declaration disagrees with child device evidence".
  - The root is refused: "not a device run".
- **Who writes the evidence:**
  - the evidence in `invocation.json` equals the child result's;
  - the parent receipt contains no device evidence;
  - `invocation.json` is written before `junit.xml`;
  - the capture time falls between the parent's start and the child's exit;
  - the recorded program is the real child;
  - in the code, the parent writes only `parent_result`, `regressions.json` and `stdout.log`.
- **The staged script:** with `CUDA_VISIBLE_DEVICES` unset or empty, it exits at line 29 before START, and nothing is
  created.
- **33 data-only cases on `check_line_device`,** seeded from the real hidden-GPU capture:
  - only the complete positive record under a production declaration is accepted;
  - every variant is refused: unset, empty or mismatched device selection; empty, blank or non-string device name;
    failed `cudaGetDevice` or properties query; a missing call, or integer statuses; a query error; zero or
    non-decimal versions; a negative, missing or Unicode-digit ordinal; a wrong engine; a wrong provider; a bad
    timestamp; a child at the right path with wrong bytes; a stand-in path; a missing program; and evidence as a list
    or absent.
- **A child at the right path with wrong bytes, live:** the parent refuses before launch and the scorer refuses the
  root, both on "tool seal mismatch: run_gpu_tests.py". The reviewer restored the file and re-verified the seal.
- **A stand-in child, live (part Q):** refused by "device evidence: child program differs from sealed
  run_gpu_tests.py".

**Q3: demote-only modes (D3). SOUND.**
- **Mismatches before launch:** a production START with a control environment, and the reverse, are both refused
  before launch ("START control-mode mismatch").
- **An audit root:** refused at root level.
- **A fixture in a production root:** the CPU fixture requested in a production root fails, with no child result, and
  the line is refused.
- **A control root:** its line passes as mode cpu, and the root is refused.
- **`score_gpu.main` with every line device-mode and passing:**
  - a production root is accepted, which shows the accept branch is reachable;
  - a control root is refused ("control run cannot establish");
  - an audit root is refused;
  - a root with one cpu line, or one line without a mode, is refused.
- **Parent fields:** a parent marked control or audit on top of positive evidence is refused, and so is a non-boolean
  field.

**Q4: the classification. CONFIRMED, with R6-O1.**
- The reviewer read the code behind all 32 class (iii) targets (citations in `analysis.json`).
  - **29 are edits of run receipts after a run:** Stored_parent, Expected_exit, START_seal, Stored_timeout_cause,
    D7 ×5, raw_exit ×4, Recorded_pin ×7, Recorded_env ×8, and the D0 mode promotion.
  - **3 edit the staging seal** (R6-O1).
- All 290 class (i) and (ii) targets were replayed in A8, and every one met its target. None of the 270 roots was
  accepted. A7's one class (i) control was run.
- No run-condition target is hidden in class (iii). Each live counterpart is in class (i) or among the reviewer's
  cases.

**Q5: regression. SOUND.**
- **Regeneration:** matches the seal and A8's arms on disk: 56 arms, with byte-identical maps and mutants.
- **The honest 58-line run:** run with the staged argv exactly as written and the real runner redirected. All 58 lines
  pass their checks, and the root's only error is absent device evidence (`hostile_rev6_p.json`).
- **Mutants** (`mutant_reach_compare.json`):
  - 54 of 54 pass, identical to A8 field by field;
  - 53 reach their edited line, and M14 reaches its edit through default-argument binding (eps = 0.001 in the witness
    call);
  - 51 are killed on CPU and 3 are equivalent (E_npbool_c, E_npbool_g, V3).

**Q6:** ACCEPT.

## Disclosed adaptations and harness failures

- **One Q2 input set `CUDA_VISIBLE_DEVICES=-1`.** The CUDA driver initialised with no visible device, the same contact
  as A7's empty-value probe. No GPU was used.
- **The staged script ran only up to its device-selection guard.** It was given TMPDIR
  `/tmp/gn1_a8_rev6_never_created`. That path is only resolved by the guard, never created, and the reviewer checked
  that it does not exist.
- **Two mirror files were edited for live cases and then restored, with the seal re-verified:** the mirror's
  `run_gpu_tests.py` and `scripts/apa_om135_core.py`.
- **The CPU mutant checks failed twice before the good run:**
  - the first failure was a missing `rev1/gn_emulate.py`, which was copied in (`mirror_extra_copies.sha256`); that log
    was overwritten;
  - then a wrong working directory, kept as `*.attempt1_wrongcwd.log`;
  - then a file-exists error, kept as `*.attempt2_fileexists.log`.
- **Disk:** `/` was at 55 GB free and `/mnt/ForgeRealm` at 87 GB at the end.

## Prior art

- **Mirror and regeneration:** carried over from REV4 and REV5 (2026). SHA-256 (FIPS 180-4). Reproducible-builds
  digest comparison; unverified, lead to check: "reproducible builds sha256 comparison".
- **Fault injection and test stubs:** Meszaros (2007).
- **Mutant reach:** the reach/infect/propagate model of mutation testing (Offutt et al., 1990s); unverified, lead to
  check: "RIP model mutation reachability".
- **Device selection:** CUDA_VISIBLE_DEVICES semantics from NVIDIA's CUDA programming guide; unverified, lead to check:
  "CUDA_VISIBLE_DEVICES invalid index hides devices".
- **Interpreter shadow:** POSIX execvp PATH lookup, and env(1) shebang resolution.
- **The reviewer's own contribution:** only the case list. No new algorithm.

## Files

All in `artifacts/gn1/rev6/`:
- **Record:** `findings.json`.
- **Regeneration:** `mirror_regen_a8.py`, `regen_compare.json`, `regen_compare.log`, `regen_maps_regenerated.json`,
  `regen_maps_a8_disk.json`, `mirror_extra_copies.sha256`.
- **Wrappers:** `parent_wrap6.py`, `checks_wrap6.py`.
- **Mutant checks:** `mirror_cpu_candidate.log`, `mirror_cpu_mutants.log`, and their `*.attempt1_wrongcwd.log` and
  `*.attempt2_fileexists.log`; `mutant_reach_compare.py`, `mutant_reach_compare.json`.
- **REV5 re-runs:** `hostile_rev6.py`, `hostile_rev6_{q,c,n,p}.{json,log}`, `rev6_probe_{rawexit,postexit}.{py,json,log}`.
- **The reviewer's cases:** `rev6_cases.py`, `rev6_cases_{d,m,f,i}.{json,log}`.
- **Analysis and classification check:** `analyze_rev6.py`, `analysis.json`.

Scratch is left under `/tmp/gn1_rev6_m` (590 MB, the mirror) and `/tmp/gn1_rev6_attempt1`, `_forge`, `_obs`, `_tmp`,
`_tools` and `_scratch`.
