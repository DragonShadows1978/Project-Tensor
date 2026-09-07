# APA-SP3 amendment 6 — CPU complete; GPU timings BLOCKED

**Host build PASS; 157 CPU tests PASS; seven executable mutations killed;
736 inherited receipt files unchanged; zero source invalidations.** No GPU
measurement was attempted. The decode slowdown is **not claimed fixed**.
The lead runs the registered reproduction and ladder on the card.

1. The reproduction result (ms/token, configuration) and the bisect
   ladder with numbers or the blocked-report for each rung.

   **Reproduction: BLOCKED; ms/token unavailable.**
   `decode_repro_b4_A_2048` uses the pinned MiniCPM3 adapter, INT4 weights,
   BF16 compute, pooling enabled **before** loading weights, `tc.no_grad()`,
   `QuantLinearTC.FUSED_DECODE=True` (global `USE_FUSED=False`), fused RMSNorm,
   fused causal softmax and STANDARD absorbed MLA decode. The existing tied
   head already uses `trans_b=True`. No diagnostic wrappers or interposer.
   The pinned prefix has 2048 tokens, followed by greedy generation. One
   discarded warmup retains the original prefix cache; 32 cached steps are
   timed with scalar device-argmax host transfers. A single final full-logit
   finite check occurs after all timed work. Pool reserved/used high water
   resets before prefill; it is **not whole-device resident peak**.

   The June source is
   `/mnt/ForgeRealm/GraftRepository/docs/MiniCPM3-MLA_Results.md`,
   “Decode speed pass (2026-06-10)”: 21.6 ms/token at S≈360, RTX 3070,
   engine `8501a5c`. Its document SHA is pinned in amendment 011. The order's
   decision fork is unchanged: ≤43.2 ms/token gives the harness finding;
   >43.2 gives the engine-on-card finding. Neither fork has been evaluated.
   Different context and engine revision limit attribution to hardware alone.

   | Rung | Change from reference 00 | ms/token | Status |
   |---|---|---:|---|
   | 00 reference | June flags; fixed teacher-forced continuation | — | GPU BLOCKED |
   | 01 wrapper | Install the original layer-index `owner.original_attn` hook | — | GPU BLOCKED |
   | 02 host_logits | Full `float().numpy()` / finite check each token | — | GPU BLOCKED |
   | 03 last_token_only | Set `last_token_only=False` | — | GPU BLOCKED |
   | 04 cache_recompute | Full-prefix recompute instead of cached single token | — | GPU BLOCKED |
   | 05 pool_after | Enable pooling after weight load | — | GPU BLOCKED |
   | 06 interposer | Load the existing cudaMalloc observer | — | GPU BLOCKED |
   | 07 int4_eager | Disable fused INT4 decode | — | GPU BLOCKED |
   | 08 norm_eager | Disable fused RMSNorm | — | GPU BLOCKED |
   | 09 expanded_mla | Disable absorbed MLA | — | GPU BLOCKED |
   | 10 softmax_eager | Disable fused softmax | — | GPU BLOCKED |
   | 11 legacy_stack | Combine old flags/instrumentation; descriptive endpoint | — | GPU BLOCKED |

   Rungs 01–10 each toggle **one** field from 00, then return to the same
   reference configuration for the next independent job. Rung 11 is not a
   one-factor attribution or byte-for-byte replay of A5's complete workload.
   A failed rung does not block unrelated ladder rungs. Every measured job
   records complete-step and forward-only timing series.

   Source inspection, **not performance evidence**: the legacy hook only
   assigns a layer index with observation off; the full-logit finite copy was
   outside its reported forward timer. Its cached input/offset/last-token
   arguments are already correct. SP3 explicitly disables the fused decode,
   fused norm, fused softmax and absorbed MLA paths used by the June stack.
   The ladder tests these differences without presuming their measured costs.

2. New cells, rails, estimates; the no-hook pin test name.

   **31 new cells; 908 total, all 877 prior cell records unchanged**:
   one greedy reproduction, twelve bisect jobs, and eighteen
   `decode_clean_b{4,8}_{A,B,C}_{2048,8192,32768}` cells. Bulk8 remains optional;
   model weights are INT4 in all cells. Full list, estimates and dependencies:
   [A6_CELLS.md](A6_CELLS.md). Machine blocked report covers **every** new cell:
   [A6_GPU_BLOCKED.json](A6_GPU_BLOCKED.json).

   All new worker rails are **290 seconds**, including load, prefill, warmup
   and measurement; existing five-second termination grace remains. One
   foreground lease per executed cell, up to 20 seconds to acquire it, 30-second
   foreground cooldown. Outer 585-second bound plus three-second grace stays
   below ten minutes; lead command wrapper is 590 seconds plus two-second grace.
   No lease or model starts for a terminal preflight non-fit.

   S≤8192 runnable cells have an unmeasured **20–280 second** worker estimate.
   Dense A8192/32768 retains the prior registered memory non-fit. For B/C32K,
   use the same-arm/bit **clean8192** result:
   `setup + 16*prefill + 4*(warmup + decode_work) + 15` seconds.
   Missing/invalid measurements block; estimate **≥290** is terminal non-fit.
   Existing A5 B/C32K projections are 874.174/593.829 seconds, consistent with
   the warning that full prefill cannot fit, but are not clean timing evidence.

   **P5 UNASSESSABLE. Clean 8192 C/B ratio also unavailable (GPU BLOCKED).**
   Only validated, measured bulk4 clean C/B at 32768 can evaluate ≥2×. If 32K
   is non-fit, reporting retains UNASSESSABLE and shows the clean8192 ratio.
   No raw/pool-kind or shorter-context substitution can score P5. Starting at
   32768 means the 32 decode contexts exceed the model's trained window.

   No-hook gate:
   `tensor_cuda/tests/test_apa_sp3_a6.py::test_decode_clean_no_attention_hook_installed`
   runs for A/B/C and checks constructor and every fake-engine forward.
   Clean A/B retain native attention and blend functions. C installs only
   the required native SP API dispatch with diagnostics explicitly false;
   it installs no per-layer attention or blend diagnostic hook.

3. Fingerprint amendment; receipts affected (expected none).

   Create-only registration: [amendment_011_decode_clean.json](amendment_011_decode_clean.json),
   SHA-256 `69dbdb19302f1c34b591b7e8b39183baeceb0cdba316ce2ac2a2cb0212f353ca`.

   Create-only fingerprint bridge:
   [amendment_012_clean_fingerprint.json](amendment_012_clean_fingerprint.json),
   SHA-256 `127962b235c73c8dbf429765d72325a8aebd213689f069dbeebab3dfea956ae1`.
   Seven exact shared-source transitions are reviewed. Legacy execute AST
   after removing the new dispatch, original worker/G0/base-registry AST,
   `apa_sp3_model.py` and `apa_sp3_a5_decode.py` remain identical. All 24
   registered product source files and 107 pre-existing kernel bodies pass pins.
   Unknown source edits or wrong bridge identities fail closed.

   **Receipts affected by source invalidation: none.**
   [Audit](a6_receipt_audit.json): 736/736 bytes unchanged, 727 source-eligible
   PASS receipts, nine inherited RED receipts retained. Local runtime-valid
   count is zero because this fresh worktree had no build and its new host
   build has a different identity. **Preserve the lead's original build;
   do not integrate this seat's build directory.** No runtime waiver exists.
   The separately labeled [CPU identity simulation](a6_lead_identity_simulation.json)
   substitutes only the original build-manifest hash in memory and validates
   **727/727 PASS receipts recursively**, including protocol and dependencies.

   Refreshed [lead_commands.txt](lead_commands.txt) lists the 31 new jobs in
   execution order. Run one command at a time after integration. Existing
   receipts are never overwritten or retried automatically.

4. Prior art; deviations; RED; process safety; model id and effort.

   **Validation:** [A6_CPU_GATES.json](A6_CPU_GATES.json): host CUDA build PASS;
   157 tests PASS, two upstream deprecation warnings, 10.60 seconds; shell
   syntax and dry-run PASS. Seven executable mutants killed, threshold 0.80,
   zero invalid in the accepted set. These are author CPU gates, not blind
   review or GPU speed/quality validation.

   **Deviations made explicit:** current pinned engine/adapter and requested
   S2048 reproduce the documented June configuration, not historical engine
   bytes/S≈360. Added four fast-stack toggles after the five requested
   candidates (cache/last-token split into two rungs). One discarded warmup
   excludes lazy absorbed-weight materialization from steady timing. Complete
   token-step timing includes argmax; the separately recorded forward-only
   series preserves the old comparison boundary. Existing G0/kernel/freeze
   dependency validation remains; old in-process PPL is not repeated under
   changed numerical flags. No new quality claim is made. Before-load pooling
   may include persistent allocations in the pool counters.

   **RED retained:** initial suite 3 failed / 154 passed (old test assumptions,
   repaired without changing measurement policy); initial audit `KeyError:
   'cell'` (two metadata-free RED timeouts, now explicitly retained); first
   mutation run had an invalid raw-P5 `KeyError`. Its original RED result is
   preserved. A separately registered executable P5 replacement passed with
   the other six valid mutants. Initial missing-build snapshot failure and
   patch-context mismatch are in the execution record; no source/receipt loss.
   GPU timings, clean8192 ratio, P5 and independent blind review remain pending.

   **Process safety:** no git, subagents, GPU jobs/queries/leases, background
   shell jobs, process kills/signals, service changes or external writes.
   Foreground host build and bounded CPU subprocesses completed normally;
   executor sessions were resumed until completion. No process was killed.
   Existing shell timeout rails were retained for lead-owned future workers.

   **Model under test:** `openbmb/MiniCPM3-4B`, snapshot
   `d6b14ddaefdb11c624dd75c3c779549bc90b08cb`, INT4 weights/group128/BF16 compute.
   Existing weight-stat and source identity pins verified on CPU.
   **Author:** `gpt-6-astra`, reasoning effort **xhigh**, recorded in
   `logs/apa_sp3_a6_r1.log`.

## Prior art

- **GraftRepository MiniCPM3/TensorCUDA, June 2026**, local results and pinned
  adapter: reused fused INT4 GEMV routing, fused RMSNorm/softmax, tied-head
  transpose flag, latent caching and absorbed decode. No new CUDA algorithm.
- **DeepSeek-AI, DeepSeek-V2 (2024)**: absorbed MLA attribution appears in
  local sources. External bibliography **unverified — lead to check**:
  `DeepSeek V2 MLA weight absorption`. We reuse the adapter implementation.
- **Existing SP3 native C dispatch (2026)** and A5 pool observation reused.
  The unchanged SP comparator's BLASST and online-softmax credits remain in
  `apa_sp3_model.py`; the unchanged TurboQuant key quantizer remains credited
  there as well. This amendment contributes measurement plumbing only.
- **NVIDIA CUDA 12.6** default-pool ReservedMemHigh/UsedMemHigh APIs, verified
  in local headers and reused through A5 PoolPeak. No novel allocator.
- Standard one-factor controlled experiments, teacher-forced benchmarking
  and warmup. **No prior art known to me for this exact ladder or planning
  formula**; both are registered experiment policy, not performance evidence.
- A4/A5 dependency-directed hash bridge reused. **Make, Feldman (1979);
  Nix, Dolstra et al. (2004); Kahn (1962)**: external bibliographies
  **unverified — lead to check** `dependency invalidation content addressed
  builds topological sorting`. Exact reviewed endpoints, not a general proof.
- Constructed-input/mutation testing reused. **DeMillo, Lipton, Sayward
  (1978)**, **unverified — lead to check** `Hints on Test Data Selection`.

Append-only execution record: [APA_SP3_A6_LEDGER.md](../../docs/APA_SP3_A6_LEDGER.md).
