# APA-SP3 amendment 6 implementation ledger

Immutable plan: `orders/APA_SP3_AMENDMENT_6.md`. Seat gpt-6-astra, xhigh,
confirmed in `logs/apa_sp3_a6_r1.log` lines 6 and 10. No git or subagents.

## Registration and inspection — 2026-09-07

- Read house rules, order, local AGENTS, actual harness, adapter, June results,
  and A5 receipts. Snapshot `a6_before.json` / `a6_baseline`: 877 prior cells,
  736 receipts. First snapshot attempt failed with `FileNotFoundError` for
  `artifacts/apa_sp3/build/manifest.json`; no gate ran. Retried recording that
  absent input explicitly. Local build is missing; no lead build is accessed.
- Source inspection: SP3 disables fused INT4 decode, fused RMSNorm, fused
  softmax, and absorbed MLA. Its cache loop DOES pass one token, position and
  returned caches correctly; `last_token_only=True` is already present. Its
  full-logit host copy occurs OUTSIDE its forward timer; record both forward
  and complete-step walls to isolate the effect honestly. Diagnostic attention
  wrapper only assigns a layer index when observation is off. No measured
  attribution to any candidate is claimed.
- June reference is 21.6 ms/token at S approximately 360, engine 8501a5c;
  order requests S2048/current pinned engine. Threshold stays 43.2 ms/token.
- Register amendment 011 before gates: one greedy reproduction, twelve
  teacher-forced ladder jobs (reference, ten single toggles, combined endpoint),
  eighteen clean cells. Worker 290s; one lease each; no automatic RED retry.
  One discarded warmup preserves initial cache and measured contexts.
- Dense A8192/32768 retains registered non-fit. B/C32K requires same-arm/bit
  clean8192 timing; setup + 16*prefill + 4*(warmup+decode_work) + 15 >=290 is
  non-fit before lease. Existing A5 B/C projections 874.174/593.829s are
  reasoning evidence only, not substituted for clean measurements.
- Clean/repro use June flags and no layer/blend diagnostic hooks. C needs the
  existing native SP binding substitution (diagnostics=False), not a per-layer
  observer. A/B retain original selective API. Frozen delta and validated G0
  dependencies retained; historical PPL controls are not rerun under the changed
  numerical flags. Final finite check outside timed loop is not a quality gate.

## Prior art

- GraftRepository MiniCPM3/TensorCUDA June 2026 results and pinned adapter:
  reuse documented fused flags, latent cache, absorbed weights, tied head.
  DeepSeek-AI DeepSeek-V2 (2024) MLA attribution appears in local sources;
  external bibliography unverified — lead to check `DeepSeek V2 MLA absorption`.
- NVIDIA CUDA 12.6 default-pool high-water counters and A5 PoolPeak reused;
  before-load pooling can include persistent allocations in pool peak. No
  whole-device resident peak claim. No new allocator or CUDA kernel.
- Standard one-factor controlled experiments, cached/teacher-forced timing;
  no prior art known to me for this exact ladder or conservative rail formula.
- Existing A4/A5 content hash bridge and dependency DAG reuse. Make (Feldman
  1979), Nix (Dolstra et al. 2004), Kahn (1962): unverified — lead to check
  `dependency invalidation content addressed builds topological sorting`.
- Constructed-input and mutation testing reused; DeMillo/Lipton/Sayward
  (1978), unverified — lead to check `Hints on Test Data Selection`. Author
  CPU gates are baseline evidence; blind review remains lead-owned.

## Build, seal and initial CPU gate

- Host CUDA compile/link completed in `a6_build.log`; no CUDA source edits.
  Build script root allowlist extended for this worktree. Foreground shell
  session only; no shell background jobs or GPU context. Local compiled build
  is for CPU probes and must not replace the lead build at integration.
- Sealed immutable amendment 012 (seven reviewed shared source transitions).
  Legacy execute AST after removing only A6 dispatch, worker/G0/base registry
  AST and legacy model/pool-worker bytes are identical.
- Initial combined CPU gate: **RED — 3 failed, 154 passed, 2 warnings in
  12.42s**, `a6_cpu_initial.log`. Failures: old rail test expected 480 for new
  kinds; old substring `run decode_b` also matches `decode_bisect`; old next
  selector expected pool decode, now superseded by clean. These are outdated
  test assumptions. Repair their scope/expected new kinds without changing
  numerical tolerances, worker rails, source implementation or registration.

## Accepted CPU suite and receipt audit

- Combined suite **157 passed, 2 upstream deprecation warnings, 10.60s**,
  `a6_cpu_accepted.log`. Registered no-hook gate:
  `test_decode_clean_no_attention_hook_installed` (A/B/C constructor and loop).
- Audit initial `KeyError: 'cell'`: two historical RED timeout receipts have
  no cell metadata. Audit now preserves them as unclassified RED, with no
  inferred source/runtime pass. Final audit: **736/736 bytes unchanged**,
  **727 source eligible, zero source invalidated, nine original RED**.
  Runtime-valid locally zero due to new seat build; no waiver applied.
- First mutation gate **RED**: six executable mutants killed; the raw-P5
  mutation changed the registered cell lookup too and raised `KeyError`, an
  invalid error mutant. Original `a6_mutations/results.json` remains RED.
  Register separate replacement in `a6_mutations_p5_followup/registration.json`
  before execution: change only receipt namespace, keep registered cell lookup
  valid. Threshold 0.80, zero invalid allowed unchanged. No production edits.

## Accepted mutation followup and delivery

- Accepted set **7/7 executable mutants killed**, zero invalid; six prior valid
  mutants plus registered P5 receipt-namespace replacement. Original invalid
  run remains RED. Production sources remain hash-identical after both runs.
- CPU original-build identity simulation: **727/727 prior PASS receipts**
  recursively validate. Only current fingerprint's build-manifest hash was
  replaced in memory; source/protocol/dependency checks unchanged. Simulation
  is not runtime or GPU validation. Nine RED receipts remain RED.
- Shell syntax and leased-shell `list` pass; dry-run exactly equals all 877
  old cells plus 31 new cells. Product-source pins: **24 files, 107 kernel
  bodies byte-identical**. Weight-stat verification passes. Nine new Python
  modules compile. No measurement source changed after fingerprint seal.
- Wrote `A6_REPORT.md` with the four Done items verbatim, `A6_CELLS.md`,
  `A6_TABLES.md`, `A6_CPU_GATES.json`, `A6_GPU_BLOCKED.json`, `a6_results.json`,
  source deltas, receipt audit and identity simulation. Refreshed
  `lead_commands.txt` with one command per new registered cell. Appended A6
  narrative to the existing `RESULTS.md`; historical P5 sections are explicitly
  superseded by clean-kind evaluation. All GPU timing fields are null.
- Residuals: GPU reproduction/bisect and P5 unmeasured, clean8192 ratio absent;
  32K clean planning needs own8192 receipts; dense long STANDARD remains
  non-fit. Independent blind verification is lead-owned and not performed.
  Speed regression is **not claimed fixed**. Preserve lead's original build
  on integration; the seat-local CPU build has no runtime identity waiver.
- Process safety: no git, subagents, background shell jobs, GPU leases/jobs/
  queries, live-service edits, external writes or process signals/kills. Host
  build and bounded CPU child processes ran in foreground and exited normally.
  One early patch attempt had a context mismatch; no partial edit occurred.
  Agent `gpt-6-astra`, reasoning `xhigh`, verified from the seat launch log.
