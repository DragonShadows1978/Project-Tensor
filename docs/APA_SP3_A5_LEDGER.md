# APA-SP3 amendment 5 implementation ledger

Immutable plan: `orders/APA_SP3_AMENDMENT_5.md`. Seat: gpt-6-astra,
xhigh, confirmed by `logs/apa_sp3_a5_r1.log`. No git or subagents.

## Registration — 2026-09-06

- Read `/mnt/Shared/HOUSE_RULES.md`, local AGENTS and orders. Snapshot source,
  registry, per-kind closures and all local receipt hashes in `a5_before.json`
  and `a5_baseline/`. No access to the lead's running worktree.
- Registered 18 `decode_pool` cells in immutable
  `artifacts/apa_sp3/amendment_009_decode_pool.json` before gates. Original
  registration and amendments remain unchanged. Worker TERM 290s for all new
  cells; 32K planning derives from same-arm/bitwidth pool-on 8192 measurements.
  Formula: setup + guard + 16*prefill + 4*decode_work + 15s allowance. Equal to
  or above rail is terminal non-fit before lease, no retry. Missing measurement
  blocks; no timing invented from June or pool-off observations.
- CPU code inspection found current TensorCUDA API documents enabling pooling
  AFTER loading raw persistent weights. The cited floor script enables before
  loading. Use the current engine's documented load/runtime split, explicitly
  recorded in registration; pool ON for every control and measured forward.
- Pool peak is default CUDA pool ReservedMemHigh/UsedMemHigh; excludes raw
  persistent weights and driver/context allocations. Existing raw observer and
  decode arithmetic stay byte-identical. Dense A>=8192 remains registered non-fit.
- 32K capture manifests stay registered; only explicit cell runs or the lead
  opt-in flag select them. Raw `decode` removed from default execution order.

## Prior art

- Project-Tensor TensorCUDA (2026), `src/kernels.cu` and Python allocator API:
  reuse existing NVIDIA stream-ordered allocation; no allocator/kernel change.
  NVIDIA CUDA 12.6 local `driver_types.h` and `cuda_runtime_api.h`: default-pool
  reserved/used high-water getters/reset; no whole-device exact peak claim.
- Existing SP3 teacher-forced decode, synchronized per-token wall clock and
  32 identical continuation tokens reused unchanged. No novel timing method.
- Conventional quadratic prefill/linear decode cost extrapolation: no prior
  art known to me for this exact planning formula; new registered rail policy.
- A4 dependency-directed content hashing / explicit semantic bridge and Kahn
  scheduling reused. Historical attributions Make (Feldman 1979), Nix (Dolstra
  et al. 2004), Kahn (1962) are unverified — lead to check: “Make dependency
  invalidation”, “Nix content addressed builds”, “Topological sorting of large
  networks”. New work is experiment wiring and a reviewed source transition.
- Constructed-input and mutation tests are standard practice; no new method.
  DeMillo/Lipton/Sayward (1978), unverified — lead to check “Hints on Test Data
  Selection”. Author gates are baseline evidence; blind verification lead-owned.

## Implementation and first CPU gates

- New worker delegates to the byte-identical `Model.decode`; enables pool ON
  immediately after raw weight loading and replaces only its peak observer with
  CUDA pool counters. G0 controls remain inside the worker before decode.
- Immutable fingerprint amendment `amendment_010_pool_fingerprint.json` sealed
  before CPU gates. Legacy execute AST (minus new-kind dispatch), base registry,
  G0 guard and receipt worker AST are identical; model bytes identical. Shared
  dispatcher, preflight, shell and validation closure edits are explicitly
  reviewed for all 17 prior kinds; new validation/registry dependencies are
  pinned. No numerical closure is waived. No prior amendment modified.
- First CPU suite: **RED**, `1 failed, 112 passed, 2 warnings in 9.60s`;
  `a5_cpu_initial.log`. Exact error: `test_a5_registration_preserves_all_prior_cells_and_rails`
  asserted a raw manifest list equalled a topologically reordered list.
  Fix the oracle to compare exact id-to-cell mappings; independent dependency
  ordering and uniqueness assertions remain. No source, registration, threshold,
  timing requirement or tolerance changed by this test-only repair.
- Initial source/receipt audit: 147 source eligible, zero source invalidated,
  one original RED preserved, all 148 original receipt hashes unchanged.
  Local runtime-valid count zero due to inherited isolated build identity;
  this is not waived. Synthetic per-kind a4 receipts cover previously registered
  kinds even though no completed `jobs_a4` receipts are present locally.

## Accepted gates and handoff

- Combined CPU suite PASS: **113 passed, 2 upstream deprecation warnings,
  10.00s**, `a5_cpu_accepted.log`; machine receipt `A5_CPU_GATES.json`.
- Registered mutation baseline passed; **5/5 killed**, zero invalid, >=0.80
  threshold. Pool-enable removal fails specifically inside worker guard at
  `assert model.tc.state is True`. Shortened decode, rail boundary relaxation,
  accidental default 32K capture and raw-decode P5 substitution also fail.
  Disposable copies only; real sources hash-identical after mutations.
- Source gates PASS: 24 source files, 107 pre-existing kernel bodies; shell
  syntax PASS. Direct dry-run and leased-shell `list` agree on **877 cells**,
  **18 new**. Default 733 cells exclude 18 raw decode and 126 32K capture
  ranges/aggregations. Explicit lead flag/cell selection preserves opt-in.
- Final receipt audit: 147 source eligible, zero invalidated; one original RED;
  148/148 original bytes unchanged. All 17 prior per-kind fingerprints accepted
  in synthetic a4 checks, including five a4 kinds. Unknown edits and wrong
  bridge identity rejected. Audit source-only comparison removes build-manifest
  key on both sides for a4 receipts; actual runtime validation retains it.
- Original-build CPU simulation: **147/147 recursively valid in 0.593s**;
  only current build hash substituted in memory. Live local build remains stale;
  no build/receipt bytes changed. Preserve lead build at merge; rerun audit and
  summary there. This is not GPU validation.
- A4 disk rail unchanged: first 32K range requires 490,783,899,647 bytes,
  exceeding current free disk and order's ~93GB estimate. Captures OFF by
  default; resolving optional capacity is the lead's decision.
- Complete delivery: `A5_CELLS.md`, immutable amendments 009/010,
  `A5_CPU_GATES.json`, `A5_GPU_BLOCKED.json`, `a5_receipt_audit.json`,
  `a5_lead_identity_simulation.json`, `A5_REPORT.md`, continuing `RESULTS.md`
  and dependency-ordered `lead_commands.txt`. Separate 32K opt-in commands
  generated, never run. No numeric timings invented; P5 UNASSESSED. Observed
  raw-pool timeout not claimed fixed by GPU evidence.
- Report-file write initially rejected an empty apply_patch hunk; no source
  change occurred. Corrected the patch and wrote the complete report.
- Process safety: no git, subagents, shell background jobs, GPU leases/queries,
  signals/kills, live-service changes or live-worktree access. Bounded foreground
  CPU commands completed. Lead-owned blind verification remains pending.
