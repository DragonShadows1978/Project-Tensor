# APA-SP3 amendment 4 implementation ledger

Order (immutable): `orders/APA_SP3_AMENDMENT_4.md`, SHA256
`e2b9dd3c074688d9ac457183f9219ea9ad08a29f006232942e6001cf19c0b236`.
Seat: gpt-6-astra, xhigh, confirmed by `logs/apa_sp3_a4_r1.log`.

## Registration and source inspection — 2026-09-06

- Read HOUSE_RULES and local AGENTS. No git, subagents, GPU calls, process
  intervention, or reads/writes of the lead's live SP3 worktree.
- Evidence class: code inspection. Preserved baseline harness files and all
  148 local job hashes in `artifacts/apa_sp3/a4_before.json` and `a4_baseline/`.
  Local receipts: 147 PASS, one RED; build manifest/shared objects and capture
  arrays were not carried into this worktree. These are execution prerequisites,
  not grounds for declaring the measured results invalid due to source edits.
- Registered 172 new/changed cells in create-only
  `amendment_005_long_context.json` before formal CPU gates. Original
  `registration.json` and order unchanged. Checkpoints, estimates, decision
  forks, CPU/mutation thresholds and the D/T interpretation are registered there.
- Evidence class: diagnostic CPU import, not a gate. Stock transformers 5.12
  remote-code loading failed first with `FileNotFoundError` for
  `blobs/configuration_minicpm.py` (snapshot symlink relative-import resolution),
  then on a byte-identical local staging copy with `ImportError: cannot import
  name 'is_torch_fx_available' from 'transformers.utils.import_utils'`.
  T needs explicit compatibility handling and CPU import/forward validation.

## Prior art carried into implementation

- Checkpoint/restart, memory-mapped block writes, SHA256 content addressing and
  per-dependency cache invalidation are established systems techniques; no
  prior art known to me for this specific SP3 layer-range layout. Reuse the
  original OpenBMB MiniCPM3 (2024) adapter layer and residual computations.
- Kahn (1962) topological scheduling: reuse DAG ordering, new registered SP3
  dependencies. Unverified lead to check: "Topological sorting of large networks".
- PyTorch SDPA (2023) uses fused flash and xFormers efficient implementations;
  backend control documented at
  https://docs.pytorch.org/docs/main/generated/torch.nn.attention.sdpa_kernel.html
  and https://pytorch.org/blog/accelerated-pytorch-2/ (checked through web tool).
  FlashAttention-2, Tri Dao (2023), https://arxiv.org/abs/2307.08691, supplies
  the existing fused algorithm; this amendment adds reference harness wiring.
- Standard teacher-forced NLL (Shannon 1948 attribution retained from SP3),
  pooled PROTOCOL-2 scorer reused unchanged; no new quality metric.
- Existing selectors remain Perry APA (2026), TurboQuant (Zandieh et al. 2025),
  BLASST (Yuan et al. 2025/2026), with SP3 finite error measurements only.

## CPU implementation gates and diagnosed failures

- Host CUDA build/link passed in the isolated a4 worktree; receipt
  `artifacts/apa_sp3/a4_build.log`. This is host compilation, no GPU execution.
- First new-piece CPU gate: **RED**, 22 passed / 1 failed. Tiny HF reload
  produced nonfinite logits. Inspection showed nonpersistent rotary buffers
  uninitialized by Transformers 5.12 meta loading. Rebuild with the unchanged
  remote `_init_rope` method, preserving original positional arithmetic.
- Second new-piece CPU gate: **RED**, 22 passed / 1 failed. A strengthened
  parameter identity check found loaded embedding weights differed entirely.
  Root cause: direct dynamic class loading had `_auto_class=None`, hence
  `is_remote_code=False`; the 5.12 initialization guard then reran the legacy
  in-place random initializer even though loading_info reported no missing keys.
  Standard `register_for_auto_class` activates the remote-code guard. Parameter
  equality and finite full-vs-sliced-logit equality remain exact, no tolerance
  relaxation. Prior art: standard Transformers loader lifecycle, no new model
  arithmetic. Evidence: initial and rotary-fix logs retained.
- Legacy CPU gate initially 56 passed / 3 failed: lazy registry import captured
  a test's patched temporary ART root. Move the registry import to driver module
  initialization. No scoring, selector, threshold or assertion changed.
- Separate immutable amendment_007_execution_details.json corrects only the
  unintended T dependency on B/C 32768: own ceiling fit suffices as ordered.
  D still requires T. It also records head-row projection, capture aggregation
  stat verification and the limited D layer-0 diagnostic scope before gates.

- Third combined CPU gate: 81 passed / 1 failed, solely the old assertion that
  every worker rail equals 480s. A4 explicitly requires <300s ceiling/range jobs;
  assertion now requires exactly 290s for those two new kinds and 480s for all
  remaining kinds. This strengthens the amended rail; no numerical criterion
  was relaxed. The new test independently checks 290+5<300 and all estimates.
- HF compatibility outcome: register the dynamic class with Transformers'
  standard AutoModel mechanism, request `dtype=bf16`, reject incomplete loading
  info, rebuild only the remote class's nonpersistent rotary buffers. CPU gate
  compares EVERY tiny checkpoint parameter exactly after bf16 conversion and
  the full 1024 vs 512-row projection outputs bitwise. No alternate weights.
- Source fingerprint design keeps the lead's original build required. The
  isolated host rebuild is not granted a semantic-equivalence waiver; the audit
  separates source-edit eligibility from local runtime identity. There is no
  justification to rebuild or rerun the lead's existing model cells for this
  registry/source-only amendment.

## Final validation and handoff

- Source compatibility audit: 147/147 local PASS receipts remain eligible,
  zero invalidated by the source edit; all 148 original receipt files unchanged.
  Per-kind closure and r3-to-a4 transition hashes are in immutable amendment 006;
  amendment 008 appends the bounded validation-cache correction without editing
  006. All registered cells and thresholds stay fixed.
- Read-only lead-identity CPU simulation initially hit its 55s rail: **RED**,
  `a4_lead_identity_simulation.log` (no numbers produced). Repeated ancestor
  protocol/source hashing was the cause, not model work. Per-action memoization
  fixes this with no cross-action cache. Final simulation validates all 147
  PASS receipts in 0.734s; first B8192 range preflight 0.436s. It substitutes ONLY
  the known original build-manifest hash in memory, explicitly not live runtime
  validation; no files or GPU state modified. Next-action ancestor mutation is
  a CPU negative gate and must reject reuse.
- Disk inspection: first 32768 capture range requires 490,783,899,647 free bytes
  (registered float size estimate truncates one byte below its exact rational
  value). About 392GB free at inspection, so full 32K captures are disk-blocked.
  A test's initially exact-rational byte oracle was corrected to the unchanged
  registered float estimate after a one-byte assertion failure (88 passed /
  1 failed). No implementation, sample size, tolerance or scientific threshold
  was relaxed. Arrays remain unsampled; the disk rail now runs before GPU lease.
- Final CPU suite: **89 passed**, two upstream deprecation warnings, 12.09s;
  `a4_cpu_accepted.log`. Final registered mutation run: **8/8 killed**, zero
  invalid, >=0.80 threshold unchanged; disposable copies only, under
  `a4_final_verification/a4_mutations/`. Earlier failure logs remain preserved.
- Host compile/link and compiled CPU API/shape guards PASS; 107 source kernel
  bodies and 25 pre-existing source files stay pinned. No production kernel or
  model/scorer file edited. Both shell scripts pass `bash -n`.
- Dry-run and leased-shell list agree on **859 cells**, **172 new/changed**:
  144 range captures, seven aggregations (five replace 8192 capture identities),
  three T cells, 15 ceiling cells, three 32K PPL cells. Immutable original
  registration and order hashes unchanged. Exact dependencies and estimates in
  `A4_CELLS.md`, machine JSON and refreshed `lead_commands.txt`.
- Full GPU reference/perplexity, native range output parity, measured fit/peak
  and runtime estimates remain **UNRUN** here. The original lead timeout is
  **not claimed fixed by GPU evidence**. T actual backend is pending; flash is
  tried for eligibility on actual 96/64 tensors, efficient otherwise; neither
  means RED. BF16/INT4 quality gaps are reported without equality gates.
- No git commands, subagents, shell background jobs, foreign-process signals,
  live-service edits, GPU leases, or access to the lead's live worktree. All
  started host commands had sub-10-minute foreground bounds and finished.
  Lead-owned blind verification remains pending.
