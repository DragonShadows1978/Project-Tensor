# APA-PF2-F2 r2 gate registration

The immutable orders are orders/APA_PF2_F2_DECODE_NORM_GUARD.md and
orders/APA_PF2_F2_R2_ALL_DECODE_LOADS.md. SHIM-RUN 20261118T110458Z-2788309
sets the deadline at 2026-11-18 17:04:58 UTC. DISPATCH_TRACE.json fixes the
two permitted load edits and classifies the entire pre-edit census.
The F1 module is pinned to
641a3139181cdee3e800f62afad08c0881b3018c26e57d953fa6c13413f7cbe7
and its build receipt to 7827ce708e92a3f36539058a091b25f9f55eded0cf2a6b9b3dcbbf1654ddd242.

- G1. Compare F2 and F1 for all 512 inherited P1-F3/P1-D public decode cases,
  both default and explicit-packed, plain and diagnostics, and the inherited
  native r2_tq4/r2_int4 LF1 cases using the same seeds and switches. Extend
  finite decode to every public bulk CAP and shared D/G instantiation with
  B=1, KV=2, S=257, VD in {64,128,256,512}, D in {64,128}, G in {1,2,4,8},
  both diagnostics settings and the native BF16 grid. Prefill uses all 4,848
  PF2 G1 cases at unchanged indices, seeds, shapes, sink/causal settings and
  delta. Each output comparison requires equal dtypes and shapes and
  nonempty, nonconstant operands before comparing same-width unsigned bits.
  Structurally constant non-finite rows are counted under G2. Diagnostic data
  retained with inherited informative checks. Sufficiently short GPU line
  splits can be registered separately before staging, with unchanged cases.
- G2. First GPU line is the 16-case sink smoke. Run all 16 REV1 sink and all
  16 REV1 raw +Inf norm cases on F2, with same operands, carry and settings
  and the original NaN controls. Add decode partition-carry checks for the
  raw non-finite norms on both D, native FP32/BF16 and both diagnostics,
  including G=8, bulk CAP256/512 and finite controls. Every reading row
  must be non-finite. Check unrelated KV group rows stay finite where present.
- G3. Once F2 builds, build two separate incremental mutants on copies. Each
  reverts exactly one load to the inherited source. On GPU the clean F2
  control and alternate route must pass, and all eight matching inherited
  cases must fail only raw_inf_decode_row_sets on the revert. An exception,
  launch error, timeout or missing row is a gate failure. GPU verdict pending
  lead execution. CPU gate requires successful independent incremental
  builds, correct one-site source deltas, distinct pinned modules, import
  checks and host arithmetic checks on the extracted load guards.
- G4. CPU cuobjdump sm_89 compares exact normalized Function blocks under
  PF2-F1 name mappings. F1 and F2 symbol sets must match. Expected changes
  are bulk<CAP,W> for all four CAP/W pairs and bulk_shared<D,G,TQ> for both
  D, both G and both TQ values emitted by the native dispatcher. Each mutant
  must only change the instantiations of its reverted kernel against F2.
  G4 GPU runs the inherited P1 packed decode clause 1/2 cases from
  LF2 and its P1-F3 overflow follow-up with their passing row-set criteria.
- G5. Scope checked with a before/after worktree inventory and content hashes
  for permitted product edits and the F1 build inputs. No promotion, GPU,
  git or deletes. Report every written file and scope deviation.
- CPU. CUDA_VISIBLE_DEVICES empty. Non-build commands use at most
  4,000,000,000 bytes RLIMIT_AS and one BLAS/OpenMP thread. Builds use
  CUDA 12.6.85, sm_89 and F1 CMake flags under timeout (<60 min) and nice 19,
  at most -j4. Stop below 40,000,000,000 bytes free. Scratch /tmp/pf2f2_r2.
- Each staged GPU line pins the module before lease, checks dry-run exit 78
  and would_start true, has max 540 and min at most 510, and records the
  actual forge exit and log SHA when run by the lead. No seat GPU execution.
- Up to three repairs of own scripts. No gate threshold changes.
  Acceptance requires all applicable CPU gates and every lead GPU line.
- Prior art. IEEE 754-2008 quiet-NaN propagation and Project-Tensor PF2-F1
  (2026), reusing the prefill non-finite conversion at both decode loads.
  External reference unverified — lead to check IEEE 754-2008 6.2 quiet NaN
  multiplication. Test grids, exact SASS comparison and isolated incremental
  reverts reuse Project-Tensor PF2-F1/P1-F3 (2026).
