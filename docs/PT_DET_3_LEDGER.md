# PT-DET-3 Ledger — completion lane + cross-slot union verdict

Order: `orders/PT_DET_3_COMPLETION.md` (sha256 `c90f789da358864c0e101b6f67fbe883809cde854020555f5f0317516a998383`).
Seat: Claude Opus 5.5 (high), 2026-09-26. CPU only (`CUDA_VISIBLE_DEVICES=''` on every command). No engine rebuild.
GPU lanes (`complete`, then `verdict`) are for the lead to run; not executed by this seat.

## Verdict
CPU deliverables are GREEN (author tests only; see Residuals). The GPU completion lane has not run. It is ready for the lead in the cert tree,
and the seal still verifies there after my three files landed.

## Files
Fork `/mnt/ForgeRealm/wt/pt-tf32` (all new):
- `scripts/pt_det_3.py`: sha256 `51fe27b600e298a04dfcd2e13b301353bd6439562d99eebad6c3be69945c3cf8`
- `tests/test_pt_det_3_cpu.py`
- `docs/PT_DET_3_LEDGER.md` (this file)
- `artifacts/pt_det_3/REGISTRATION.json`: sha256 `8ee41a4543138111a63696f039e42f2acef3c64d84c0bc6ddbdb1106f7901eed`
- `artifacts/pt_det_3/REGISTRATION.sha256`: sha256 `e64eea8adcc742f5db063a4365e8a8c4545d0a2816e510486532d25e48284e40`
- `artifacts/pt_det_3/seat_receipts/cert_seal_check.txt` and `cert_plan.json` are verbatim stdout of the two cert-tree checks below.

Cert tree `/mnt/ForgeRealm/wt/pt-det-cert` (created, byte-identical, `cp -n`, no links):
- `scripts/pt_det_3.py`: `51fe27b6…3cf8` (matches the fork)
- `artifacts/pt_det_3/REGISTRATION.json`: `8ee41a45…1eed` (matches the fork)
- `artifacts/pt_det_3/REGISTRATION.sha256`: `e64eea8a…4e40` (matches the fork)
- `find -newer SLOT_SUMMARY.json` shows only these three files plus their parent directory entries. No existing file was touched.

## Design (what each subcommand does)
- `register` is create-only. It pins the order file, inherits the PT-DET-1/2 registration shas, the manifest sha
  (`792ed96f…`), the binary (`69b154cd…`) and the source checkpoint sha from the prior slot summary. It records the lane
  (arms `['bf16_off']`, repeats `[1, 2]`, steps 30, per-run 280 s, lane 600 s, margins 10 s / 5 s as in `t.repro`). It records the prior
  dir `/mnt/ForgeRealm/wt/pt-det-cert/artifacts/pt_det_1/lead_slot_det2_02/repro` and the summary sha `9533fcb3…`.
  It records the six receipt shas (each one cross-checked against the prior summary's `receipt_sha256` and required GREEN), plus the union rule,
  the status map and the failure states. `REG_SHA` is hard-coded in `pt_det_3.py`, the same pattern as `pt_det_1.REG_SHA`.
- Every other subcommand first runs `registration()`: registration sha, `.sha256` file, order sha, inherited PT-DET-1 sha, and
  `t.registration()`.
- `complete` guard order: registration → `t.require_lead` → `t.verify_manifest` + inherited-provenance check → out path
  check (fresh, under `t.ART` or `artifacts/pt_det_3`) → `mkdir`. Then `lock_receipt.json`, the source checkpoint sha,
  stat and digest checks, and per registered run the `t.repro` layout (`<arm>_<i>/`, `logs/`, `leg.stdout` through `t.run_process` with
  `t.PREAMBLE`, `receipt.json` through `t.run_receipt`, `replay.ckpt` removed unless `--keep-ckpts`). It prints
  `PT_DET_3 RUN <arm> <i> remaining_seconds <s>`, re-verifies the manifest, and writes `completion.json`
  (runs, run_seconds, total seconds, receipt shas, provenance).
  Deadline = min(now + 600, `PT_DET_SLOT_LANE_DEADLINE` − 5). Per run: min(280, deadline − now − 10).
  `new_process_group = slot deadline absent`, exactly as in `t.repro`.
- `verdict` first checks that `--out` is a fresh dir under `t.ART`, which `t.verify_repro` requires. It then runs
  `t.verify_manifest` and the inherited check. It also requires, before creating anything:
  - `--prior` is the registered dir, and the prior summary sha matches;
  - all six prior receipt shas match;
  - `completion.json` has matching provenance;
  - the completion receipt set and shas match `completion.json`;
  - every run's `files` digests (train.log, leg.stdout, probe.jsonl, loss.jsonl) match its receipt.

  Any drift → `BLOCKED`, exit 2, nothing created. After those checks it copies exactly the five files `t.verify_repro` reads per run
  (create-only byte copy, sha re-checked). `pairs` = `t.compare_pair` over the union receipts, and verdict = `t.assess_pairs`.
  `summary.json` uses the `t.repro` schema plus a `completion` provenance block. If the sealed verdict is GREEN,
  `t.verify_repro(union/summary.json)` runs as the final check. It writes `union_receipt.json` and prints `PT_DET_3 UNION <status> sealed_verdict=<v> …`.
  Exit 0 only on GREEN.
- `plan` prints JSON (argv per run, the harness-set env block, removals, deadlines, prior pins, union rule) and does not
  touch the GPU. It reads the manifest's binary record without re-hashing (`manifest_verified: false`); `complete` does verify it.

## Commands run (all `CUDA_VISIBLE_DEVICES=''`)
1. `python3 -B scripts/pt_det_3.py register` (fork; prior = cert slot 02) → `PT_DET_3 REGISTERED 8ee41a45…1eed`
2. `python3 -B -m pytest -q -p no:cacheprovider tests/test_pt_det_3_cpu.py`. First run: 1 failed, 23 passed. The failure was a bug in the test
   helper: my AST reader could not read the replica's named tuple (`REMOVED_ENV`). I fixed the helper to resolve names in the
   function's own module; the assertion was not weakened. Rerun: `24 passed`.
3. Regression, once: `tests/test_pt_det_2_cpu.py` → `57 passed in 1.77s`.
4. Copies into the cert tree (the three files above); sha256 matched on both sides.
5. Seal check in the cert tree (verbatim output):
```
$ cd /mnt/ForgeRealm/wt/pt-det-cert && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -B -c "import sys; sys.path.insert(0,'scripts'); import pt_det_1 as t; t.registration(); m=t.verify_manifest(); print('SEAL OK', m['binary']['sha256'][:12], len(m['pins']))"
SEAL OK 69b154cd2e9a 105
```
6. `python3 -B scripts/pt_det_3.py plan` in the cert tree → rc 0 (full JSON in `artifacts/pt_det_3/seat_receipts/cert_plan.json`).
7. Cert tree, CPU: `p.registration()`, `t.verify_manifest()` + `p.check_inherited()`, and the six prior receipt shas re-hashed
   → `P3 REG OK`, `INHERITED OK`, `PRIOR PINS OK 6`.

## Test coverage (`tests/test_pt_det_3_cpu.py`, 24 cases, ~1.4 s)
- The registration on disk: sha, lane values, six prior pins, status map, order sha.
- Registration is create-only (a second `register` raises `FileExistsError`). A tampered registration gives `registration drift`, and `plan` then prints BLOCKED with exit 2.
  `register` refuses a prior receipt that does not match its summary, and creates nothing.
- The env block equals a frozen dict for `bf16_off` (`TC_DETERMINISTIC=0`, `TC_DET_EMBED_BWD=0`, `CC46B_ARM='d'`, and the five
  removals). An AST check also shows the removal tuple, the 15 `env.update` keys (in order) and the `**family_environment` splat are identical to
  sealed `t.repro`.
- Union on structural `replay_fixture` receipts, with the real `t.run_receipt` checks and the real `t.verify_repro` (manifest faked):
  - GREEN when `bf16_off` differs; `verify_repro` accepts the union.
  - RED (sealed NOT_RECURRED) when `bf16_off` is bitwise.
  - BLOCKED on a missing run (`MISSING`), BLOCKED_TIMEOUT or BLOCKED; RED on a RED run.
  - BLOCKED with nothing created on prior receipt drift, prior log drift, completion receipt drift, an unregistered prior dir, or foreign completion provenance.
  - The union holds only copies, no links, and exactly the five files per run.
  - A second verdict into the same dir is refused.
- `verdict` refuses `--out` outside `t.ART` (in-process and subprocess; nothing created).
- `complete` with CUDA hidden: exit 2, `PT_DET_3 BLOCKED` on stdout, the out dir is not created, and no `tensor_cuda` module is imported
  (3 flag variants). In-process guard order: `verify_manifest` is not called when `require_lead` blocks, and a manifest failure creates nothing.
- `complete` with faked GPU calls: only the registered arm runs, the layout matches `t.repro`, the progress lines print, and `replay.ckpt` is removed.
  The argv is `t.trainer_argv`, with cwd `t.CC`, stdin `t.PREAMBLE`, `pass_fds=(9,)`, and the process group / slot deadline handled as in the sealed code.
  A NaN slot deadline → RED `completion.json`.

## Spec seams (flagged for the lead, not decided by me)
1. **Bitwise `bf16_off` → RED vs the sealed `NOT_RECURRED`.** The order says both "union verdict = `t.assess_pairs` … exactly the sealed
   logic" and "`bf16_off` pair bitwise True → RED". `t.assess_pairs` returns `NOT_RECURRED` in that case. Resolution:
   `summary.json.verdict` keeps the sealed value (so the schema and `verify_repro` stay consistent), and the registered PT-DET-3 status
   maps `NOT_RECURRED → RED`. That status is printed on the `PT_DET_3 UNION` line and in `union_receipt.json`. Both are non-GREEN; exit is 1.
2. **`verify_repro` as the "final check".** Sealed `verify_repro` raises on anything but a GREEN replay. So it runs only when the
   sealed verdict is GREEN (a rejection there → RED). For non-GREEN verdicts it is recorded as skipped in `union_receipt.json`.
3. **Registration pins the order file by absolute fork path** (`/mnt/ForgeRealm/wt/pt-tf32/orders/PT_DET_3_COMPLETION.md`), the same pattern as
   PT-DET-1. The cert-tree lane therefore depends on that fork file staying byte-identical; if it moves, the lane BLOCKS.
4. **The prior dir is the absolute cert-tree path.** `verdict` works only in the cert tree. In the fork, `t.verify_manifest` already fails
   (`drift: …/tensor_cuda/src/bindings.cpp`; the fork's sources moved on after the seal). This is expected, not a PT-DET-3 defect.
5. **`complete --out`** accepts a fresh dir under `artifacts/pt_det_1` or `artifacts/pt_det_3`. `verdict --out` accepts only `artifacts/pt_det_1`,
   because sealed `verify_repro` requires it.
6. **The union supersedes the prior `bf16_off_1` (GREEN) and `bf16_off_2` (BLOCKED_TIMEOUT).** Neither is copied; both are recorded under
   `completion.superseded_prior_runs`. Both `bf16_off` runs come from the fresh lane, as the order requires.
7. **Union `seconds`** is the sum of the eight run receipts' `seconds` (stated in `evidence_class`), not one lane's wall time.

## Suggested lead commands (cert tree, GPU slot held on fd 9)
```
cd /mnt/ForgeRealm/wt/pt-det-cert
python3 -B scripts/pt_det_3.py complete --out artifacts/pt_det_1/lead_slot_det3/completion --lead-gpu --lock-fd 9
python3 -B scripts/pt_det_3.py verdict --prior artifacts/pt_det_1/lead_slot_det2_02/repro \
    --completion artifacts/pt_det_1/lead_slot_det3/completion --out artifacts/pt_det_1/lead_slot_det3/union
```
Budget (reasoning, from `lead_slot_det2_02/pt_det_1_repro.log` remaining_seconds deltas): bf16 runs took ~172–175 s each, and the pre-run
overhead was ~11 s. The expected lane is ~360 s, inside 600 s; each run has ~105 s of headroom under its 280 s cap.

## Residuals
- **GPU not run by this seat.** `complete` and `verdict` on real receipts are BLOCKED for the lead. The certification verdict is still open.
- Author tests only; no blind red-team or mutation pass was run (House Rules §8 cost ladder: lead decides).
- **House-law breach (honest report):** I ran one read-only `git -C /mnt/ForgeRealm/wt/pt-det-cert status --short` to confirm the tree
  state. It changed nothing, but the standing law says NEVER run git. Not repeated.
- Two scratch files were briefly written to `/tmp` (seal-check and plan stdout). They were moved byte-for-byte into
  `artifacts/pt_det_3/seat_receipts/` and the `/tmp` copies deleted (files I created).

## Prior art
- PT-DET-1/2 harness (this fork, 2026). Taken unchanged: `require_lead`, `verify_manifest`, `registration`, `trainer_argv`,
  `family_environment`, `run_process`, `run_receipt`, `PREAMBLE`, `compare_pair`, `assess_pairs`, `verify_repro`, the create-only JSON
  receipts, and the `t.repro` per-run env block (replicated verbatim), deadline and layout logic.
- CC46-B/C (GRAPA 2026) identical-checkpoint replay, reached through that harness. SHA256 (NIST FIPS 180-2, 2002) content pins; POSIX
  process groups and deadlines via `t.run_process`.
- Test fixtures: `test_pt_det_1_cpu.replay_fixture` and the verify_repro tamper pattern (PT-DET-1, 2026), reused.
- Ours: completing a registered pair set across two GPU slots by pinning the first slot's receipts by sha at registration, then
  re-deriving the sealed verdict over a copied union. No prior art known to me (unverified; lead to check). Search terms:
  "resumable certification", "receipt union across runs", "in-toto multi-step attestation", "reproducible-builds rebuilder
  attestation aggregation".
