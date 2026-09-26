# PT-DET-3 — REGISTERED 2026-09-26 12:20 EDT by the lead (Fable 5.1). Seat: Claude Opus 5.5 (high). Completion lane for the PT-DET run-to-run certification: arm-selectable runner + cross-slot union verdict, ADDITIVE to the sealed PT-DET-1/2 harness.

## Why (read first)
The PT-DET GPU certification slot ran today in the CLEAN sealed tree `/mnt/ForgeRealm/wt/pt-det-cert`
(binary 69b154cd…, seal `artifacts/pt_det_1/SOURCE_MANIFEST_003.json`). Receipts:
`/mnt/ForgeRealm/wt/pt-det-cert/artifacts/pt_det_1/lead_slot_det2_02/` (SLOT_SUMMARY.json, repro/summary.json).
- embedding lane GREEN, gather lane GREEN.
- repro lane (30 steps, two fresh processes per arm): `v3_on` BITWISE, `bf16_on` BITWISE, `v3_off` NOT bitwise
  (control diverges, as registered), `bf16_off`: run 1 GREEN, run 2 `BLOCKED_TIMEOUT` — the lane deadline
  (1,320 s) cannot hold 8 runs at ~175–190 s each. Verdict BLOCKED by the all-four-pairs rule.
The remedy is not in question; the registration under-budgeted the lane. We complete the certification with a
SMALL second lane that runs ONLY the missing `bf16_off` pair and computes the registered verdict over the union
of the six measured runs + the two new ones. Everything is additive: NO pinned file may change.

## YOUR WRITABLE TARGET (explicit grant)
- `/mnt/ForgeRealm/wt/pt-tf32` (this fork worktree, branch tf32-fast-path): edits/creates AUTHORIZED for NEW files
  only: `scripts/pt_det_3.py`, `tests/test_pt_det_3_cpu.py`, `docs/PT_DET_3_LEDGER.md`, `artifacts/pt_det_3/**`.
  Do NOT modify `scripts/pt_det_1.py`, `scripts/pt_det_2.py`, anything under `tensor_cuda/`, or any existing file.
  No rebuild of the engine.
- `/mnt/ForgeRealm/wt/pt-det-cert` (the sealed certification tree): you may CREATE exactly these new files there:
  `scripts/pt_det_3.py` (byte-identical copy of the fork file) and `artifacts/pt_det_3/REGISTRATION.json` +
  `REGISTRATION.sha256` (byte-identical copies). NEVER modify, rename or delete any existing file in that tree —
  the seal pins them, and the lead's GPU lane runs there.
Read-only: everything else, in particular `/mnt/ForgeRealm/Project-Tensor` (LIVE engine), `/mnt/ForgeRealm/grapa_run7`
(live run), `/mnt/ForgeRealm/wt/grapa-cc46` (trainer sources the seal depends on), `/mnt/ForgeRealm/GRAPA-Native-LLM`.

## Hard boundaries (no sandbox — these are the sandbox)
- NO GPU: every command you run carries `CUDA_VISIBLE_DEVICES=''`. Never open, lock, or read `/tmp/forge-gpu.lock`.
- No git (the lead commits). No subagents. Never kill/signal any process you did not start; a second orchestrator
  and a live training run share this machine. No deletions outside files you created this seat.
- Run ONLY the named test file, synchronously, in the foreground:
  `CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/mnt/ForgeRealm/wt/pt-tf32/tensor_cuda python3 -B -m pytest -q -p no:cacheprovider tests/test_pt_det_3_cpu.py`
  (the sealed harness tests `tests/test_pt_det_2_cpu.py` may be run once as a regression check, nothing else).
- RED honesty: if anything blocks, write it in the ledger and the final message; never widen scope to "fix" it.

## Deliverables
1. `scripts/pt_det_3.py` — imports the sealed module (`sys.path.insert(0, str(Path(__file__).resolve().parent))`,
   `import pt_det_1 as t`), uses ITS functions and constants (`t.ARMS`, `t.trainer_argv`, `t.family_environment`,
   `t.run_process`, `t.run_receipt`, `t.compare_pair`, `t.assess_pairs`, `t.verify_manifest`, `t.require_lead`,
   `t.registration`, `t.create_json`, `t.sha`, `t.PREAMBLE`, `t.CC`, `t.ROOT`, `t.ART`) — never a copy of their logic
   except the per-run environment block of `t.repro` (lines ~590–605), which you replicate exactly (same keys,
   same removals) and unit-test against a frozen expected dict for `bf16_off`. Subcommands:
   - `register` — create-only `artifacts/pt_det_3/REGISTRATION.json` + `.sha256`: the completion lane definition —
     arms `['bf16_off']`, repeats (1, 2), steps 30, per-run deadline ≤ 280 s, lane deadline 600 s; the PRIOR slot dir
     (absolute: `<ROOT>/artifacts/pt_det_1/lead_slot_det2_02/repro`) and the sha256 of each of its six measured run
     receipts (`v3_on_1/2`, `bf16_on_1/2`, `v3_off_1/2` `receipt.json`) pinned at registration; the union verdict rule
     = `t.assess_pairs` over the four pairs (exactly the sealed logic); registered failure states: any new run not
     GREEN → BLOCKED/RED as the sealed receipt says; `bf16_off` pair bitwise True → RED (the control must diverge);
     prior receipt drift → BLOCKED. Registration is checked (sha) before every other subcommand.
   - `complete --out <ART>/<fresh dir> --lead-gpu --lock-fd N` — same guards as `t.repro`: `t.require_lead`,
     `t.verify_manifest`, source-checkpoint sha + stat checks, `PT_DET_SLOT_LANE_DEADLINE` handling; runs only the
     registered arm(s), each repeat into `<out>/<arm>_<repeat>/` exactly as `t.repro` lays it out (run dir, `logs/`,
     `leg.stdout`, `receipt.json` via `t.run_receipt`, replay.ckpt removed unless `--keep-ckpts`); prints the same
     `PT_DET_1 RUN …` progress lines prefixed `PT_DET_3`; writes `<out>/completion.json` (runs, seconds, receipts sha).
     Guard order: registration → require_lead → verify_manifest → mkdir out (a CPU-hidden call must exit 2 with
     `BLOCKED` in stdout and create nothing).
   - `verdict --prior <slot02 repro dir> --completion <out of complete> --out <ART>/<fresh union dir>` — verifies the
     six prior receipts against the registered shas, COPIES (not symlinks) the six prior run dirs and the two new
     ones into the union dir (same `<arm>_<i>/` layout, only the files the sealed `t.verify_repro` reads: `receipt.json`,
     `logs/train.log`, `leg.stdout`, `probe.jsonl`, `loss.jsonl`), recomputes `pairs` with `t.compare_pair` from the
     receipts, verdict = `t.assess_pairs(pairs)`, writes `<union>/summary.json` in the SAME schema as `t.repro`'s
     summary (all provenance fields: registration_sha256, binary, steps 30, pt_det_2_registration_sha256,
     deterministic_sites, manifest_sha256, source_checkpoint_sha256, pairs, runs, receipt_sha256, plus
     `completion` provenance: prior dir, completion dir, PT-DET-3 registration sha), then calls
     `t.verify_repro(<union>/summary.json)` as the final check and prints `PT_DET_3 UNION <verdict>`; exit 0 only on
     GREEN. Everything create-only.
   - `plan` — prints the completion lane (argv per run, env, deadlines) as JSON without touching the GPU.
2. `tests/test_pt_det_3_cpu.py` (CPU only, ≤ 60 s): registration create-only + sha check; the replicated env block
   equals the frozen expected dict for `bf16_off` (TC_DETERMINISTIC=0, TC_DET_EMBED_BWD=0, CC46B_ARM='d', removals);
   union verdict on synthetic receipts — GREEN when `bf16_off` differs, RED when bitwise, BLOCKED when a run is
   missing/non-GREEN, BLOCKED on prior receipt drift; guard-before-CUDA/output for `complete` (exit 2, nothing
   created); `verdict` refuses an `--out` outside `t.ART`.
3. `docs/PT_DET_3_LEDGER.md` — commands run, results, the exact seal-check receipt below, residuals.
4. Prove the seal still verifies in the cert tree AFTER your copies land, and paste the output verbatim:
   `cd /mnt/ForgeRealm/wt/pt-det-cert && CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -B -c "import sys; sys.path.insert(0,'scripts'); import pt_det_1 as t; t.registration(); m=t.verify_manifest(); print('SEAL OK', m['binary']['sha256'][:12], len(m['pins']))"`
   and `python3 -B scripts/pt_det_3.py plan` there (CPU-hidden).

## Prior art
Annotate at the code site + ledger + report: PT-DET-1/2 harness (this fork, 2026) — taken: guards, receipts,
pair logic; cross-slot union of receipts is ours; say "none known" where true.

## Done (final message must contain, verbatim from disk)
- pytest summary line for `tests/test_pt_det_3_cpu.py` (and the PT-DET-2 regression line if run).
- The seal-check output from the cert tree after your copies, and the `plan` JSON head.
- sha256 of the fork `scripts/pt_det_3.py` and of the cert-tree copy (must match), and of
  `artifacts/pt_det_3/REGISTRATION.json` in both trees.
- Residuals and anything BLOCKED, honestly.
