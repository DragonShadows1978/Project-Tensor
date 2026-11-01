# GN1 A6 report

**RED-D0 — G4 FAIL; G1, G2, G3, G5 and G6 PASS in their finite CPU scope. The required 321 controls pass, but one supplemental in-scope receipt-forgery control fails: the scorer accepts a promoted CPU fixture root without any seal, child-receipt, log, program or argv change. P117 NOT HELD. P118 HELD. Overall acceptance and lead GPU execution are not cleared.**

Evidence class: CPU test/suite execution, installed-runner CPU interruption controls, exact byte/hash/argv/environment comparisons, and source reasoning. No device execution, product change, rebuild, performance measurement or model-quality claim. Order commit: `1de3ed5b63addceb1a012deccb94fd8f2721b763`, branch `gn1`. Delivery seal: `dd49febf342bd5d0a3421bb882980c339d8e080c2cd9977480c44a0de6cd457a`. Registration precedes every gate. The supplied plan is unchanged at `orders/GN1_A6_REV5_FIXES.md`.

## Gates

| Gate | Result | Receipts and scope |
|---|---|---|
| G1 | PASS | G1_scope.json: 29,271 frozen file hashes identical before/after; product and test match cf922891; tracked diff empty. tooling_a5_to_a6.diff and TOOLING_DIFF_MAP.md map all 40 hunks in 19 files to D1–D4 or mechanical A6 relocation. |
| G2 | PASS | G2_checks.json: NumPy 2.4.2, pytest 9.0.2; native imports/collection with CUDA hidden; A5 IDs unchanged (focus 412, mutation 405, regression 468); 44 native reference/guard calls, 57 guard cases, 405 tests pass in the explicit NumPy stand-in lane. |
| G3 | PASS | G3_pin.json and engine_sources_verified.json: original pin and 72-source map unchanged; 56 module copies match c9b7dc610421010a8b11e8e3af74d1a9ad99928dabb98835aa6c265754daf8ce; zero arm symlinks and zero builds. |
| G4 | FAIL | G4_summary.json and G4_TABLE.md: 283 inherited +38 new registered targets pass, but D0_MODE_PROMOTION.json demonstrates an in-scope control-root promotion. Scorer exit0, all58 lines passing after59 metadata edits; seal and child/log/program/argv evidence unchanged. |
| G5 | PASS | G5_checks.json, G5_mutants.json and final_mutants_gate.json: all 54 unchanged mutants/witnesses reached; 51 CPU kills and 3 registered equivalences; rechecked with the new tooling. No GPU mutation kill claim. |
| G6 | PASS | G6_staging.json, staging_exact_bounds.json, regeneration_verified.json and honest_labels_verified.json: 58 exact staged commands, 56 sealed arms regenerated, audit 29/29 reference calls (child/parent 0), all 58 honest installed-runner CPU control lines pass. The unaltered declared control root refuses acceptance; receipt forgery defeats that declaration (G4). |

The pre-probe consolidation run is `final_verify_gate.json` / `.log`, with its exact registered command. The authoritative final RED assessment is `FINAL_ASSESSMENT.json`, verified by `final_d0_assessment_gate.json` / `.log`. Its driver is retained at `/tmp/gn1_a6_seat/final_verify.py`. The report's file inventory and SHA-256 maps cover all retained workspace and scratch artifacts.

## In-scope residual — control-mode promotion

`ADDITIONAL_CONTROL_REGISTRATION.md` registered this probe before it ran. `D0_MODE_PROMOTION.launch.json` records its launch; `D0_MODE_PROMOTION.json` records the failure; `D0_MODE_RESTORED.json` verifies restoration.

Starting from the real-parent honest58-label CPU control root, edit only59 JSON receipt files: set START.control=false; set each of58 parent.control fields false; remove FORGE_GPU_SUMMARY, FORGE_GPU_WINDOW_FILE and FORGE_GPU_LOCK from the58 recorded parent environments. No seal, child receipt, runner log, executable or argv is changed. The scorer returns exit0, passed=true, and58 passing lines. Those programs really were pinned, but the CPU child fixtures ran under scratch ownership overrides and control mode; the forged metadata claims production mode/environment. After the probe, every edited byte was restored and the honest root's summary was recomputed to root rejection with all58 lines passing.

Evidence class: actual scorer execution on forged receipts from the actual parent/control run. This is inside D0(a). D0's excluded coordinated rewrite of every receipt/log together with the seal does not apply: the seal, all child receipts and all runner logs stayed byte-identical. It also directly defeats D1's statement that control mode can never establish acceptance. D1's comparisons are implemented, but their input records are themselves mutable under the stated adversary.

**Not claimed fixed under all of D0:** program/environment provenance (R5-F01/R5-F02). Their plain wrapper/PATH reproductions now refuse; a consistent receipt forgery can hide the real mode/environment. The selected D1–D4 design provides consistency checks, not trusted run provenance. More comparisons among the same editable records do not close this case. A trusted provenance boundary or a change to D0 requires a lead design decision; neither is introduced by this amendment. The frozen product, earlier evidence, installed programs and GPU resources remain untouched. This is the stopping result under the binding design, not an ACCEPT.

## Per-finding fix table

| Finding | Result | Fix and deciding evidence |
|---|---|---|
| R5-F01 MINOR | D1 implemented; D0 closure not claimed | D1 checks the pinned runner at command[0], plus absolute/resolved path and SHA-256. G4_a6.json: Q2_wrapped_runner_command0, D0c_runner_substitute, Recorded_pin_runner. |
| R5-F02 MINOR | D1 implemented; D0 closure not claimed | D1 fixes PATH before shell lookups, uses absolute runner/timeout/Python and Bash paths, unsets summary/window/lock overrides on non-audit lines, and binds every received PATH/FORGE_GPU_* value. G4_a6.json: Q2_path_shadow_inner_timeout, D0c_* substitutions, seven isolated program-pin controls, eight isolated environment controls, six actual environment controls. |
| R5-F03 NIT | FIXED | D2 catches Exception at both scorer read boundaries and parent dependency reads. G4_a6.json: all three NEW_*deep* cases write refusing summaries with recorded recursion reasons. |
| R5-F04 NIT | FIXED | D3 requires type(raw_exit) is int and recomputes exit = raw_exit if nonnegative, else 128 - raw_exit. G4_a6.json: minus9/exit0, missing, bool and string edits all refuse. |
| R5-F05 NIT | FIXED | D4 hashes the seal inside protection, records errors, retries publication once with a minimal error receipt, and records double-write failure. G4_a6.json: REV5_postexit_seal_failure, D4_write_retry, D4_double_write_incomplete. The final case has no parent receipt and the scorer explicitly marks it incomplete. |

## Tooling diff summary

The source comparison spans 43 Python/shell sources, including three new support files: `program_binding.py`, `a6_controls.py`, and `p116_probe.py` (the unchanged A5 direct-PID supplement, relocated and declared as a control). The raw diff has 40 hunks across 19 files. `tooling_hunks.json` and `TOOLING_DIFF_MAP.md` enumerate every hunk. `tooling_functional.diff` removes only mechanical namespace/order/scratch relocation.

D1 adds program/environment operands, strict runner command[0] and absolute inner argv, sealed pins, control-mode binding at START and parent receipts, mandatory control-root rejection, and exact-stage CPU fixture support. The shell sets PATH before any external lookup, uses an absolute Bash shebang and built-in path expansion, and calls runner, timeout, Python and mkdir by absolute path. D2 broadens the scorer and parent JSON/evaluation catches to Exception. D3 independently recomputes raw exit normalization. D4 protects the post-exit seal hash and full receipt publication, with exactly one minimal retry. Numerical code, tolerances, IDs, mutant edits and witnesses are unchanged.

Two inherited positive target names retain their old root-pass wording in raw JSON. As registered before replay, their D1 predicates require every line to pass and the CPU control root to refuse GPU acceptance. The consolidated table states that deciding rule explicitly; no production-acceptance result is inferred from a fixture root.

## Program pins and environment

| Program | Staged absolute path (home abbreviated in this narrative) | SHA-256 |
|---|---|---|
| Installed runner | `$HOME/.local/bin/forge-gpu-run` | `ad6c0a22edc3baf8fd513f1123e6d460266a7ede77b91c496b3ea38e8724e107` |
| timeout | `/usr/bin/timeout` | `12690a043dfd555a6c14ccc1564f5649c18f1762c0d6ff66729948130898ec52` |
| Python | `/usr/bin/python3` (resolves to python3.12) | `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f` |
| Runner shebang env | `/usr/bin/env` | `886e5fa8be716b03eeda48856e0f04493f2969fc148fb7812594ffac647fa050` |

`program_pins.json` and the seal preserve the exact absolute/resolved paths required by D1; narrative home paths are abbreviated. Parent receipts record runner, inner/outer timeout, inner interpreter, runner-shebang interpreter, parent interpreter and env executable, plus the parent's kernel executable identity. Resolution uses the exact child PATH and files are hashed immediately before launch; these are file identities, not a claim of kernel attestation of every descendant instruction.

The sealed production environment is exactly PATH=`/usr/bin:/bin`, FORGE_GPU_KILL_GRACE=`10`, FORGE_GPU_POLL=`5`, and FORGE_GPU_RUN_LOG=`{attempt_dir}/forge.log`. Summary/window/lock overrides are absent on non-audit staged lines. Every unknown FORGE_GPU_* variable also causes a mismatch. `OPERAND_CENSUS.json` derives all six runner reads from its installed source: KILL_GRACE, POLL, RUN_LOG, SUMMARY, WINDOW_FILE, LOCK. The runner's env shebang is the only remaining PATH program resolution; it resolves Python under the fixed PATH and is separately pinned.

## G4 table

| G4 batch | Targets | Result | Receipt |
|---|---:|---|---|
| Inherited hostile payloads | 52 | PASS | G4_hostile.json |
| Installed-runner outcomes/interruption | 13 | PASS | G4_runner.json |
| REV2 replay | 38 | PASS | hostile_rev2_acd.json |
| Supplementary seal/cause | 4 | PASS | G4_extra.json |
| A4/REV3 obligations | 24 | PASS | G4_a4.json |
| A5/REV4 grammar, grace and publication | 150 | PASS | G4_a5.json |
| Direct-PID interruption supplement | 2 | PASS | G4_p116.json |
| REV5 and D0(c), D2–D4 | 38 | PASS | G4_a6.json |
| Supplemental control-root receipt promotion | 1 | FAIL | D0_MODE_PROMOTION.json |
| Total | 322 | FAIL (321 met) | G4_summary.json; G4_TABLE.md |

`G4_TABLE.md` lists every individual target and deciding rule. The 38 new targets comprise one all-label control-root case, three deep-nesting cases, four raw-exit edits, six executable/PATH attacks, seven isolated recorded program pins, eight isolated recorded environments (PATH, all six runner variables and an unknown FORGE_GPU_*), six actual environment captures, and three post-exit publication cases.

The actual program/PATH attacks may trigger more than one D1 refusal. The isolated recorded-pin and recorded-environment cases calibrate each equality independently, avoiding credit for a different pin's failure. Actual ownership overrides always point inside task scratch. An incoming RUN_LOG override is replaced by the parent's staged attempt log path; its actual received value is checked. Synthetic transports are declared controls and their unaltered mode records refuse GPU acceptance; the supplemental attack defeats those records.

REV5's post-exit seal failure is reproduced by a scratch launcher faulting the real parent's seal hash after its real child exits; it leaves the actual seal bytes unchanged. The write cases similarly fault the real mkstemp call once or twice. The first retry publishes a minimal failure receipt; double failure leaves an incomplete line with no acceptance. All three use the real parent main and real scorer, with launch receipts.

## Threat model and predictions

`D0_THREAT_MODEL.md` explains the immutable order's D0 scope. In-scope classes are receipt/log edits and malformed inputs; arbitrary staged argv changes; substitute programs and differing PATH/FORGE_GPU_* values; and installed-runner/timeout interruptions. The order excludes coordinated rewriting of every receipt/log together with the seal, root installed-runner races beyond the hash check, kernel hiding, and GPU/driver defects. These gates provide finite execution evidence and exact consistency checks, not a formal authentication proof.

**P117 NOT HELD:** the321 original predicates pass, but the registered D0(a)/D1 requirement that CPU control mode never establish acceptance fails under the supplemental receipt-only promotion. The broader G4 requirement is not met.

**P118 HELD:** all 58 exact staged argv lists run under the installed runner and pinned timeout/interpreter and pass line recomputation; consolidation independently checks every recorded program identity against the seal. Only ownership-file redirection and the explicitly declared child stand-in use control mode. The unaltered root refuses GPU acceptance. The supplemental forgery defeats that mode declaration, so this honest positive result does not establish D0 soundness. This is the authorized CPU sensitivity result; production environment/GPU execution remains the lead's run after GN1-REV6.

A5's grace ordering is unchanged: grace10 + poll5 =15 < outer kill-after20; script margin40 leaves20 after outer KILL for publication. Both direct-PID controls observe no live child immediately after return. This is a configured ordering and finite observation, not a hard bound on arbitrary scheduler/filesystem stalls.

## Retained failures and corrections

- Initial sealing launch **FAIL**: `FileNotFoundError` for the omitted A6 `gpu_lines.json` setup copy. The frozen A5 file was copied unchanged. `seal_initial_gate.log` and receipt retain the failure.
- Initial hostile launch **FAIL**: `import module seal mismatch: artifacts/gn1/a6/program_binding.py` after its source was edited following the initial census. The failed launch is in `pre_census_refresh/`; import/runtime censuses were refreshed before replay.
- First REV2 launch **FAIL**: `FileNotFoundError` at the task scratch `rev2_forge/summary.log`. Created the missing scratch directory; source and target predicates unchanged. The failed launch is retained in `pre_shell_binding/rev2_gate.*` and its partial run root.
- A final source census found the inherited pre-PATH dirname lookup. D1 now removes it and sets PATH first. All 321 targets, the audit and regeneration were replayed against the final seal. Earlier successful batches remain in `pre_shell_binding/`; they are not delivery-seal evidence.
- An intermediate commentary total of327 was a counting error. The original G4 batches contain321 targets (283+38); the count was corrected before consolidation. The supplemental failing probe raises the final table to322.

## Prior art

- POSIX execvp PATH lookup and shell parameter expansion, IEEE Std1003.1: taken executable-resolution semantics and built-in path derivation; A6 applies the order's exact path/environment conjunction. Year unverified — lead to check: POSIX execvp PATH search; POSIX shell parameter expansion history. Code comments are in program_binding.py and the staged shell.
- Linux procfs exe/cmdline/environ: taken kernel process identity/state inspection; the new parent records `/proc/self/exe`, inherited PID controls inspect procfs state, and REV5 supplies cmdline/environ observation provenance. History unverified — lead to check: Linux procfs executable cmdline environ history. A6 does not claim to sample every descendant through procfs.
- SHA-256, NIST FIPS180-4 (2015): inherited file maps/seals, extended to executable pins. GNU coreutils timeout(1) and Python subprocess negative-signal returns supply exit/supervision semantics; the POSIX128+signal convention supplies D3 normalization. Historical GNU year unverified — lead to check: coreutils timeout kill-after history.
- CPython json recursion handling, Python Software Foundation: taken Exception/RecursionError behavior, verified by REV5's 200,000-level payloads. Historical year unverified — lead to check: CPython json recursion limit history.
- POSIX link(2) no-overwrite publication, IEEE Std1003.1: unchanged A5 write/fsync/link mechanism; A6 adds the mandated one-retry error receipt. No prior art known to me for that order-specific retry policy. Year unverified — lead to check: POSIX link atomic no replace.
- Meszaros (2007), test stubs/fault injection, and independently authored GN1-REV5 (2026) reproductions supply control techniques. Unchanged numerical/mutation provenance remains in copied sources: Higham (1996/2002), PEP357 (2006), DeMillo/Lipton/Sayward (1978), Offutt/Voas reach/equivalence (year unverified — lead to check), pytest/JUnit (Krekel et al.,2004+, historical attribution unverified), and argparse allow_abbrev (PSF,Python3.5,2015, historical attribution unverified). No novel algorithm or external literature verification is claimed. LEDGER.md and code-site comments carry the same provenance.

## Done

The superseded pre-probe ACCEPT report and summaries are retained in pre_D0_promotion/; they are historical results, not the final verdict. Every retained workspace file written is enumerated in `files_written.txt` and `files_written.json`; task scratch is enumerated in `scratch_files_written.txt` and `.json`. `sha256_maps.json` and `scratch_sha256_maps.json` hash those files, excluding their own self-hashes. `LEFT_FOR_LEAD_TO_BIN.md` and `left_for_lead_to_bin.json` list retained superseded batches, old seals, linked temporary receipts, empty control scratch and the task scratch tree. Nothing was deleted by this seat. All writes are within `artifacts/gn1/a6/` and `/tmp/gn1_a6_*`; `logs/` was unused. Frozen earlier artifacts/product/test/module/source maps match their before bytes; no rebuild, GPU use, real ownership-file access, git write or subagent was used.

Work began at20:18 UTC and final gate consolidation completed at20:54:49 UTC on2026-11-01; synthesis prepared 2026-11-01T20:57:43.609285+00:00. This is within the2h ceiling. Every command has an explicit foreground timeout below20min; running returns were resumed with the maximum300000ms wait and no intervening status message. The lead still owns the D0/provenance design decision, GN1-REV6 and any later GPU execution. Overall acceptance is withheld.
