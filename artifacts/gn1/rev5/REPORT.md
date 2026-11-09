# GN1-REV5 review report

- **Reviewer:** Claude Opus 5.5 (Agent-tool seat). This is a cross-model re-review of Codex's A5 (`24ddbff3`).
- **Order:** `orders/GN1_REV5_REVIEW.md` (e914aa49).
- **Run:** 2026-11-01, 14:53 to 15:15 EST, CPU only.

**Note from the lead.** The seat's tool rules forbid report files, so the lead filed its final message here, re-laid
out in sections with its wording kept. `findings.json` is the seat's record.

**How the seat ran (disclosed adaptations):**
- A5's real parent, child stand-in and scorer ran byte for byte in seal-verified mirrors. `parent_wrap5.py` and
  `checks_wrap5.py` widen only the TMPDIR check.
- The installed runner always had its summary, window and lock redirected to scratch.
- Instead of running the staged script, the seat parsed every staged line's argv and ran it exactly, with the
  script's exported environment.

**Verdict: REJECT.** 5 findings: 0 BLOCKER, 0 MAJOR, 2 MINOR, 3 NIT. All four REV4 findings are FIXED.

## Findings

**R5-F01, MINOR. The scorer never checks which runner ran.**
- **Where:** `a5/evidence.py:330-331` compares only `runner[1:]`.
- **Reproduction:** case `Q2_wrapped_runner_command0`. A wrapper at `command[0]` runs the real runner with `--max 900
  --self-leasing` appended. The kernel's cmdline and the runner's row confirm the bound actually applied. The scorer
  passes the line.
- **Fix:**
  - require `runner[0] == "forge-gpu-run"`;
  - the parent records the resolved runner path and its sha256, which the scorer compares with a sealed pin of the
    installed runner (`ad6c0a22…e107`);
  - CPU controls use a declared control mode that cannot establish acceptance;
  - add the wrapper case to G4.

**R5-F02, MINOR. The environment can change the applied bound or the honoured window and lock, unseen by the
scorer.**
- **Where:**
  - `a5/run_gpu_line.py:54-58`: the child inherits PATH and every `FORGE_GPU_*` variable;
  - `:62`: `timeout` is found through PATH;
  - `:99-104`: the receipt records none of this;
  - `STAGED_GPU_COMMANDS.sh:15-17`: pins only grace and poll.
- **Reproduction:** `Q2_path_shadow_inner_timeout`. A shadow `timeout` rewrites the inner 240 s to 900 s, and the
  scorer passes the line.
- **Fix:**
  - the staged script sets PATH explicitly, or uses absolute paths, and unsets the summary, window and lock overrides
    for non-audit lines;
  - the parent records the resolved programs with sha256, and the `FORGE_GPU_*` environment;
  - the scorer refuses unless these equal the sealed and staged values;
  - CPU roots declare a control mode;
  - add PATH-shadow and redirected-window controls.

**R5-F03, NIT. A deeply nested JSON receipt crashes the scorer.**
- **Where:** `a5/score_gpu.py:22`, `:27`. `RecursionError` is not caught. This is unchanged since A4.
- **Reproduction:** 200,000 nested `[` in `parent_result.json`, `START.json`, or a regression `child_result.json`. The
  scorer exits 1 and writes no summary. This fails closed.
- **Fix:** catch `Exception` at both sites, and add the three cases.

**R5-F04, NIT. The scorer does not recompute the exit normalisation from `raw_exit`.**
- **Where:** `a5/evidence.py:397`.
- **Reproduction:** an edited receipt with `raw_exit` -9 and exit 0 passes, and so does one with `raw_exit` removed.
- **Fix:** require an integer `raw_exit`, with `exit == (raw if raw >= 0 else 128 - raw)`.

**R5-F05, NIT. D4 is not complete after the child exits.**
- **Where:** `a5/run_gpu_line.py:99-104`. The seal hash can raise after the protected block, and the temporary write
  at `:108-112` is unprotected.
- **Reproduction:** an unreadable seal while the child runs leaves no receipt. This fails closed.
- **Fix:** compute `seal_sha256` inside the protected block, falling back to a recorded reason. Retry the write with a
  minimal error receipt.

## Per question

- **Q1:** R4-F01 to R4-F04 are all FIXED.
  - Bound spellings: REV4's 5 spellings and 10 new ones are all refused.
  - Interruptions: N2b and N2c under the staging record `outer-timeout` and leave no survivor. So do grace 18 and 21
    and the staged lock-wait kill.
  - Malformed inputs: N5 and 6 malformed row shapes are recorded.
  - Truncated receipts: N4 and its variants are refused as JSON.
- **Q2, exact-form binding:** UNSOUND (F01, F02).
- **Q3, kill-grace ordering:** SOUND. The runner finishes about 10.1 s after the outer TERM, about 9.9 s before the
  outer KILL. The script margin is 20 s.
- **Q4, robustness:** UNSOUND (F03, F04, F05).
- **Q5, regression:** SOUND.
  - Two fresh mirrors regenerate all 56 arms byte-identically.
  - All 58 staged lines run honestly with the real runner, are admitted, and pass the scorer.
  - The runtime seal, the equivalences and admission hold.
  - Mutants: 54 of 54 reached, 51 killed, 3 equivalent, matching A5's `G5_mutants.json`.
- **Q6:** REJECT.

The seat's harness failures are retained as `*.attempt1.log` and `*.attempt2_wrongcwd.log`.

## Files

All in `artifacts/gn1/rev5/`:
- `findings.json`;
- `mirror_regen_a5.py`, `parent_wrap5.py`, `checks_wrap5.py`, `hostile_rev5.py`;
- `rev5_probe_rawexit.*`, `rev5_probe_postexit.*`;
- `hostile_rev5_{q,n,c,p}.*`;
- the regeneration and mutant-reach receipts.

Scratch is left under `/tmp/gn1_rev5_*`.
