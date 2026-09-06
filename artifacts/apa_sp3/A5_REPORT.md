# APA-SP3 amendment 5 — CPU implementation complete; GPU timing blocked

Evidence class: author CPU gates, source/receipt audit and planning. **113 tests
passed; 5/5 registered mutants killed; zero receipts invalidated by A5 source
edits.** GPU pool timing, measured 32K planning and P5 remain UNRUN/UNASSESSED.
The pool-off 32K timeout is **not claimed fixed by GPU evidence**.

## 1. New cell ids/estimates; pool-state pin; peak meaning

- 18 ids: `decode_pool_b{4,8}_{A,B,C}_{2048,8192,32768}`. Expanded ids,
  dependencies and estimates: [A5_CELLS.md](A5_CELLS.md). Each depends on `g0`,
  `ppl_b4_D_1024`, `freeze_b{bits}`; 32K also requires its same-arm/bit pool8192
  receipt. Existing 859 cells remain registered unchanged.
- S<=8192 planning: 60–280 seconds, unmeasured; worker TERM 290s +5s grace.
  Same 290s TERM rail registered for 32K. Dense A at 8192/32768 records the
  lead's registered non-fit before GPU lease, without model execution.
- 32K planning is create-only and SHA-pins its actual pool8192 source:
  `setup_s + guard_s + 16*prefill_s + 4*decode_work_s + 15`. Setup/guards stay
  constant, prefill scales quadratically, decode linearly. This is a conservative
  planning assumption. Missing measurement blocks; estimate >=290s records
  terminal `fit=false` before lease, without retry. No numeric 32K estimate
  exists before the lead measures pool8192. Plans: `plans_a5/<cell>.json`.
- `test_decode_pool_worker_pool_state_pin` calls the real worker/dispatcher
  and unchanged `Model.decode` loop with a stateful CPU fake. It asserts pool
  TRUE inside G0 guard, cache prefill and all 32 decode forwards. Disabling the
  setter is a killed mutant. Timing remains per-token CUDA-synced wall time,
  reporting tokens/s and ms/token. This is CPU plumbing evidence, not native
  GPU state or performance validation.
- Peak source: CUDA default-pool `cudaMemPoolGetAttribute`, ReservedMemHigh
  and UsedMemHigh, reset before measured prefill. `pool_reserved_peak_mib` and
  `pool_used_peak_mib` are pool counters. Legacy-shaped `peak_resident_mib`
  aliases **reserved pool high water**, not device resident memory; raw weights
  and driver/context allocations are excluded. Receipts explicitly say so.
  Legacy pool-off decode and intercepted allocation peaks remain unchanged.
- P5 uses valid primary bulk4 pool-on B/C 32K receipts: C.tokens_s / B.tokens_s
  >=2. Non-fit, missing, stale, RED or short measurements leave it UNASSESSED.
  Starting 32K measures contexts 32769–32800 beyond the trained window.
  G2/G3 rows establish nothing about model quality by themselves.

## 2. Fingerprint amendment; receipts affected

- New cells: `amendment_009_decode_pool.json`, SHA256
  `8f0809b8f00cf1db333048f63da30f1a10a907c2c9890b97733693efaa209bfc`.
- Fingerprint: `amendment_010_pool_fingerprint.json`, SHA256
  `23af444cfd89bed69f79bc766681a680315fb801cea702fe313c6214c7d6dc1f`.
  Both immutable, bound to A5 order and original registration. All prior
  amendment checksums remain valid. Exact per-kind closures and reviewed
  transition endpoints are in amendment 010. Original registration SHA256:
  `d9b6511702a894f72174795141c72b2097b1bbe6110e810a1d4d8bfc3cd3498c`.
- **Source-invalidated receipts: none.** 147/147 local PASS source-eligible;
  all 148 original receipt bytes unchanged, including one original RED.
  `a5_receipt_audit.json` scans `jobs` and `jobs_a4`. No completed a4-kind
  receipts exist locally; synthetic prior per-kind receipts for all 17 kinds
  pass bridge checks, including range/aggregate/T/ceiling/long-PPL.
- Every prior kind's shared closure is touched by dispatcher, preflight,
  shell and provenance edits. The pool branch is separate. Original execute
  AST after removing that branch, base registry, G0 guard and receipt worker
  are identical. Model, scorer, a4 worker modules and kernels are unchanged.
  New validation/registry dependencies are pinned. Unknown edits reject reuse.
- Local runtime remains stale because the inherited isolated a4 build differs
  from the lead's build; no waiver. A CPU simulation substituting only the
  original build-manifest hash in memory validates all 147 PASS receipts and
  recursive dependencies in 0.593s (`a5_lead_identity_simulation.json`). Preserve
  the lead build on merge and re-render there. No rebuild/receipt replacement.

## 3. CPU gates; blocked-report; exact lead commands

- [A5_CPU_GATES.json](A5_CPU_GATES.json): 113 passed, 2 upstream deprecation
  warnings, 10.00s; `a5_cpu_accepted.log`. Initial RED retained in
  `a5_cpu_initial.log`: manifest-order vs topological-order test oracle. Fixed
  to exact id-to-cell comparison; independent dependency checks unchanged.
- `a5_mutations/results.json`: 5/5 killed, zero invalid, threshold >=0.80.
  Mutants: pool OFF, 31 steps, rail >= changed to >, default 32K captures ON,
  raw decode substituted for P5. Disposable copies only; sources unchanged.
- Source pins: 24 files, 107 existing kernel bodies PASS; shell syntax PASS.
  Direct `--dry-run` and shell `list` agree on 877 cells, 18 new. Default list
  has 733 cells, excluding 18 raw decode and 126 32K capture cells.
- [A5_GPU_BLOCKED.json](A5_GPU_BLOCKED.json): no GPU device nodes or GPU
  execution here. Native timings/counters, pool-on G0 parity, 32K fit and P5
  require lead execution; blind verification remains lead-owned.

After merge, preserving the lead's build and receipts:

```bash
cd /mnt/ForgeRealm/Project-Tensor-wt-apa-sp3
timeout 30s python3 scripts/apa_sp3_a5_audit.py
timeout 30s bash scripts/apa_sp3_lead_gpu.sh list
timeout 30s bash scripts/apa_sp3_lead_gpu.sh summary
```

Use [lead_commands.txt](lead_commands.txt), one foreground command at a time
after its dependencies PASS. Exact pool4 commands:

```bash
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run decode_pool_b4_A_2048
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run decode_pool_b4_B_2048
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run decode_pool_b4_C_2048
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run decode_pool_b4_A_8192
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run decode_pool_b4_B_8192
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run decode_pool_b4_C_8192
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run decode_pool_b4_A_32768
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run decode_pool_b4_B_32768
timeout --kill-after=2s 590s bash scripts/apa_sp3_lead_gpu.sh run decode_pool_b4_C_32768
timeout 30s bash scripts/apa_sp3_lead_gpu.sh summary
```

Bulk8 equivalents are in the full commands file after `freeze_b8`. These are
individual commands, not an automatic batch. Existing receipts are never
retried. Pool-off decode is excluded from default resume too.

32K captures stay OFF. Explicit `run capture_..._32768...` is lead cell-list
inclusion; `APA_SP3_INCLUDE_32K_CAPTURES=1` opts into resume/command generation
(`--include-32k-captures` on Python `--next`). Separate
`lead_commands_32k_captures.txt` is a lead-only opt-in list; this seat ran none.

## Prior art

- TensorCUDA, Project-Tensor (2026), existing `src/kernels.cu` and Python
  allocator API: reuse stream-ordered transient pooling; no allocator/kernel
  changes. NVIDIA CUDA 12.6 pool high-water APIs verified in local headers.
- Existing SP3 synchronized teacher-forced cached decode: same 32 tokens,
  arithmetic and timing loop; no new benchmark algorithm or selection rule.
- Conventional quadratic prefill/linear decode planning: no prior art known
  to me for this exact formula. New registered allowance and non-fit fork.
- A4 dependency-directed hashing / manual semantic bridge reused. Make
  (Feldman 1979), Nix (Dolstra et al. 2004), Kahn scheduling (1962): unverified —
  lead to check “Make dependency invalidation”, “Nix content addressed builds”,
  “Topological sorting of large networks”. No general equivalence proof.
- Mutation testing: DeMillo/Lipton/Sayward (1978), unverified — lead to check
  “Hints on Test Data Selection”. Author baseline, not independent review.
  Code sites and ledger carry the same distinctions. Existing APA/BLASST/
  TurboQuant selectors are unchanged and retain their original attributions.

## 4. Deviations; RED; process safety; model and effort

- Current TensorCUDA API documents pool enable AFTER raw persistent weight
  loading; the cited floor script enables before loading. Follow current engine
  lifecycle and explicitly record the difference. Pool ON for all controls,
  prefill and measured decode steps.
- A4's unchanged conservative first-32K-range disk gate requires
  490,783,899,647 free bytes, above available space and the lead's ~93GB per-arm
  layout estimate. The rail is not relaxed; optional captures remain lead-owned.
- Existing RED retained. No measured speedup, 32K timing fit, P5 hit, quality
  improvement, native pool-state validation or blind verification claimed.
- No git, subagents, GPU leases/queries, shell background jobs, process
  kills/signals, live-service edits or live-worktree access. All host child
  commands bounded below 10 minutes and completed in the foreground.
- Model **gpt-6-astra**, effort **xhigh**, confirmed by `logs/apa_sp3_a5_r1.log`.
  Ledger: `docs/APA_SP3_A5_LEDGER.md`; continuing synthesis: `RESULTS.md`.
