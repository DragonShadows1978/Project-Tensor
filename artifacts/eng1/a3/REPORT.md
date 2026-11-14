# ENG1-A3 report (lead copy of the seat's final message)

The Claude Code harness refused the seat's own write of this file; the lead copied the final message verbatim on 2026-11-14.

The routing fix works, but the round is not ACCEPT: G1 fails its registered bar. In device mode all 24 labels now verify the A3 seal and all 20 arms. Only 6 of the 24 reach the governor's dry-run exit 78; the other 18 stop earlier, at a dependency check in a file I may not edit. G2, G3 (in a no-git form) and G4 pass.

**Why G1 fails.** `run_gpu_line.py` lines 39-42 (forbidden file, unchanged) refuse a line until an earlier line has passed:
- the 16 mutant labels (M01-M16) need a passing `candidate_eng1`: `{"error": "mutations require a passing candidate_eng1"}`;
- `candidate_regression` and `candidate_terrain` need their baseline to have passed: `{"error": "candidate regression requires a valid baseline"}`.

A dry run never passes a line (exit 78 means "not admitted"), so these 18 labels cannot reach the governor before G5. I predicted this from the code and ledgered the prediction before running G1. I attempted no repair (L56(b) allows 3): every route past the check is either a forbidden edit or a fabricated passing receipt.

**Decision for you:** either re-register G1's bar for these 18 labels as "seal and arms verified, refusal at the dependency check, and S1 exit 78 for the line's argv", or accept that G5 is the first place they reach the governor. In G5 they run after their predecessors, in `gpu_lines.json` order, as your wrapper already does. Never reuse `a3_dry_1` for G5.

**REPORT.md was not written.** The harness refused my Write of `artifacts/eng1/a3/REPORT.md` ("Subagents should return findings as text"). I did not work around it; this message carries the report, and the refusal is in the ledger.

## Route and changed bytes
I took route (a). The only change to an existing file is `evidence.py` lines 23-24:
```
-ORDER = "orders/ENG1_A1_REV1_FIXES.md"
-SEAL_NAME = "staging_seal_a1.json"
+ORDER = "orders/ENG1_A3_DEVICE_SEAL_ROUTING.md"  # A3 route (a): re-pointed from ENG1_A1_REV1_FIXES.md
+SEAL_NAME = "staging_seal_a3.json"  # A3 route (a): was staging_seal_a1.json; prior art: none known to me (a constant re-point)
```
- **Hashes:** `evidence.py` f3ae12c2… before, c7201f7d06e2dd7094b4775333ccb44df06abe8d9ff5764dcb864379e9a3cc11 after. Lines 62, 88, 245 and 348 keep their numbers. `STAGED_GPU_COMMANDS.sh` is untouched.
- **New seal:** `tool/staging_seal_a3.json` (38708 B), written by `a3/staging.py`, which runs stage.py's own seal step as `a2/staging.py` does. It differs from the A2 seal only in these fields (`a3/seal_a2_to_a3.diff`):
  - `a3_note` replaces `a2_note`; `base_commit` is 97edd1fd;
  - `order_commit` 46676057f27fedcad214e214d20f5fd699f89fe3, `order_sha256` b38dd90b…;
  - `tool_files['evidence.py']` c7201f7d…;
  - two registration entries added: `a3/staging.py` 484718b5… and `a3/sitecustomize.py` a0cc8278….
  - The arm maps and engine pins equal the A2 seal's; the staging script asserts this.
- **Order commit:** read with `cat` from the ref file `refs/heads/eng1`; no git command was run.
- **G2 routing (new file):** `a3/sitecustomize.py`, active only with `ENG1_A3_AUDIT_ROUTING=1`. It checks that the defaults already point at the A3 seal and order, changes neither, and only restricts the allowed run root to `a3/audit`.

## A3 seal sha256
**e0d48543e796de8e3825ea789e1016983bfbde4cdadc61a308179709e2a8de44**

## Before G1: nothing before the governor touches CUDA
- **Code reading:** `start_gpu_run.py`, `evidence.py`'s imports, and `run_gpu_line.py` up to its runner call (line 66) only hash files, read JSON and create directories.
  - The only `ctypes.CDLL` call (`device_evidence.capture`), the only `tensor_cuda` import (`arm_origin.preflight`) and the only git call (`e6_record.snapshot`) are reached only from `run_gpu_tests.py`, the child process after the governor.
  - With the dry-run switch set, `forge-gpu-run` returns 78 at lines 248-264, before any lock, lease, log line or spawn.
- **Mechanical check (`cuda_preflight.py`):** I imported the pre-governor modules and ran the seal check and all 20 arm checks. No CUDA or engine library was mapped and no `tensor_cuda` module was loaded.

## G1 table (run root `tool/gpu_runs/a3_dry_1`, fresh)
Each label ran as `env -u PYTHONOPTIMIZE -u ENG1_AUDIT RUNS=<root> CUDA_VISIBLE_DEVICES=0 TMPDIR=/tmp/eng1_a3_dry FORGE_GPU_DRY_RUN=1 bash artifacts/eng1/tool/STAGED_GPU_COMMANDS.sh <label>`, with the same inherited environment as your `eng1_g5_env.txt`.

| label | staged rc | seal verified | arms verified | governor rc |
|---|---|---|---|---|
| baseline_eng1 | 1 | yes | 20/20 | 78 (DRY RUN, no run-log line) |
| candidate_eng1 | 1 | yes | 20/20 | 78 |
| baseline_regression | 1 | yes | 20/20 | 78 |
| candidate_regression | 1 | yes | 20/20 | not reached (baseline check) |
| baseline_terrain | 1 | yes | 20/20 | 78 |
| candidate_terrain | 1 | yes | 20/20 | not reached (baseline check) |
| C_COPY | 1 | yes | 20/20 | 78 |
| C_UBSAN | 1 | yes | 20/20 | 78 |
| M01 … M16 (16 labels) | 1 each | yes | 20/20 | not reached (candidate_eng1 check) |

- **Totals:** seal and arms verified on 24 of 24; governor exit 78 on 6 of 24. **G1: FAIL** against "every label returns 78".
- **Supplementary (not a substitute for G1):** each label's exact runner argv, passed straight to `forge-gpu-run` with the dry-run switch, returns 78 on all 24. No usage error and no run-log line, so the 18 labels' governor arguments are well-formed (`S1_GOVERNOR_ARGV.json`).

## G2 audit result: PASS
- `STAGED_GPU_COMMANDS.sh --audit` (the real staged script, CPU stand-in runner, CUDA hidden), run with `PYTHONPATH=a3:tool`, `ENG1_A3_AUDIT_ROUTING=1` and `RUNS=artifacts/eng1/a3/audit`, exited 0.
- All 10 steps after the device call pass (`G2_AUDIT_RECEIPT.json`):
  - START bound to e0d48543;
  - child start marker, origin check (engine 3e6d5805…), invocation record, junit.xml;
  - child result: pytest exit 0, 47 test IDs;
  - runner log row "ran 0";
  - parent receipt passed, recorded after the child exited;
  - re-scoring with the A3 seal passed;
  - `score_gpu.py` exits 1 with "audit run cannot establish GPU acceptance", which is the designed outcome for an audit root.

## G3: diff summary and engine pins (no-git form): PASS
The dispatch forbids git, so you need to run `git diff 97edd1fd -- . ':!artifacts/eng1/a3' ':!orders' ':!logs'` yourself. My stand-in compares whole-file sha256 of the full worktree before any edit (11657 files) against the state after (`G3_SCOPE_r2.json`):
- **Changed:** only `tool/evidence.py`. Nothing removed.
- **Added outside `a3/`:** `tool/staging_seal_a3.json` and 19 files in `tool/gpu_runs/a3_dry_1/`.
- **Forbidden files:** 1066 checked, all byte-identical.
- **State at dispatch against the A2-close records:** 0 mismatches. The records are the A2 seal maps, `a2/protected_after.json`, `a2/tensor_cuda_after.json` and `a2/FILES_WRITTEN.json`.
- **Engine pins** (A1, A2 and A3 seals and the files on disk all agree):
  - baseline 73ededa0e73357c8578e58e5eadd87df9f2f3b2d64d631eba32499f2fdb9523e
  - candidate 3e6d5805fb8b89ebfeb2b32c173fd097ef1cfe4a01b3349dc177d4b0c8c0ad53
- **Repair 1 (on my own G3 script):** its first run reported a fail while every check was clean, because I had wrongly expected the pre-edit `evidence.py` to differ from the A2 seal. The first receipt `G3_SCOPE.json` is kept and the fix is in the ledger.

## G4 refusal text: PASS
I ran G1's first label in two byte-copy scratch trees under `/tmp/eng1_a3_m17`. With the live `evidence.py`, the seal verifies and the governor returns 78. With M17 (the pre-edit `evidence.py`, routing on the A1 seal), the line exits 1, creates no run root, and refuses with the same text as your 05:39 log:
```
  File ".../tool/start_gpu_run.py", line 8, in <module>
    seal = verify_seal()
  File ".../tool/evidence.py", line 66, in verify_seal
    assert sha(TOOL / name) == digest, "tool seal mismatch: " + name
AssertionError: tool seal mismatch: ids_a1_regression.txt
```

## FILES_WRITTEN.json sha256
**f5ada0654673364956c30f382a198fd529c50d27a15c4791250023c1386c9704** (119 worktree files; the scratch roots are listed inside it).

## Other notes
- **L82:** the grep gives 22 hits, not 21. They are exactly the lines your order lists, which also add up to 22.
- **G1 environment:** I did not add `PYTHONPYCACHEPREFIX` or `XDG_CACHE_HOME` to G1 lines, because L48 allows only the dry-run switch as an addition; the staged script already disables bytecode writing. My own commands set all three to `/tmp/eng1_a3_*`.
- **Scratch kept under `/tmp`:** eng1_a3_work, _dry, _audit, _m17, _m17_tmp, _s1, _pycache, _xdg.
- **Free space at the end:** /mnt/ForgeRealm has 109956141056 B free.

## Prior art
- **Seal:** ENG1 `stage.py` seal step and `a2/staging.py` (2026), reused unchanged; SHA-256, NIST FIPS 180-4. New here: only the checks that the A3 seal equals the A2 seal except where route (a) changed bytes.
- **Audit routing:** Python's site/sitecustomize startup hook (PSF; unverified, lead to check "python site sitecustomize") and `a2/sitecustomize.py`. New here: checking the seal and order instead of rebinding them.
- **Dry run:** the GW3 contract (System-Claude, GPU_WINDOW.md, 2026).
- **M17:** mutation testing, DeMillo, Lipton and Sayward (1978), with a paired control as in ENG1 C_COPY.
- **No-git identity check:** manifest diffing as in BSD mtree(8) (year unverified, lead to check "mtree BSD").
- **Route (a) constant re-point:** no prior art known to me; it is not an algorithm.

Files are in `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng1/artifacts/eng1/`:
- `tool/evidence.py`
- `tool/staging_seal_a3.json`
- `tool/gpu_runs/a3_dry_1/`
- `a3/` holds the ledger, gate receipts and scripts:
  - `LEDGER.md`, `FILES_WRITTEN.json`
  - `g1_a3_dry_1/G1_RESULT.json`, `S1_GOVERNOR_ARGV.json`
  - `G2_AUDIT_RECEIPT.json`, `G3_SCOPE_r2.json`, `G4_M17.json`
  - `staging.py`, `sitecustomize.py`, `evidence_py.diff`, `seal_a2_to_a3.diff`
