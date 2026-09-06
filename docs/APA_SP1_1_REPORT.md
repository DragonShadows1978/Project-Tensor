# APA-SP1.1 delivery

Implemented and host-built; **G1 CPU PASS, G2/G3 BLOCKED, RED**. No new GPU timing or equivalence result is claimed. All 51 lead-supplied SP1 GPU receipts remain hash-pinned. Evidence class for numerical performance is **kernel sweep**: **this establishes nothing about model quality**.

Registration: `artifacts/apa_sp1/registration_sp1_1.json`, SHA256 `382194c1f9d94c146bddcf533b7becadee60fe7d212a2029fabf133243c683b4`; immutable parent SHA256 `12059d49f39abe9450b34989f06ac5ca8a736d80e3127c9d41c1ed97e650c9fe`. [Registration](../artifacts/apa_sp1/registration_sp1_1.json), [separate amendment](../artifacts/apa_sp1/amendment_sp1_1_001.json). No delta, z threshold, matching tolerance, or numerical tolerance changed.

Part A — proof and implementation

For key j in ordered partition p, let M_p(j) be its local inclusive bulk-prefix maximum, and G(j) the whole-row inclusive bulk-prefix maximum. The partition's visited keys are a subset of the global prefix, so M_p(j) <= G(j). If the global rule selects j, bulk_j >= G(j)-delta >= M_p(j)-delta, hence the local rule selects it too. Equivalently a local skip satisfies bulk_j < M_p(j)-delta <= G(j)-delta and remains a global skip. Replacing G(j) with the final whole-row maximum also proves containment of the final-max window. This assumes identical finite bulk scores and delta>=0; changing dot accumulation order can still move an fp32 boundary.

The output claim needs qualification. Extra local refinements replace bulk scores with exact scores and can change output. They are not guaranteed to cost only work. For bulk=[2,0], exact=[2,10], V=[0,1], delta=0, separate singleton partitions refine both keys; the global prefix refines only the first. With sink=0 the outputs differ by more than 0.8. This explicit counterexample is pinned.

For the actual partition-selected mixed scores s_j, each partial holds m_p=max(s_j), l_p=sum(exp(s_j-m_p)), a_p=sum(exp(s_j-m_p)*V_j). With M=max_p(m_p), summing l_p*exp(m_p-M) and a_p*exp(m_p-M) gives exactly the materialized softmax numerator and denominator in real arithmetic. The sink contributes once to the denominator at merge. CPU tests check this identity within the registered atol=rtol=0.001, including empty partitions, random noncontiguous ordered subsets, ties, ragged tails and sinks. Equality is to partition-selected materialization, not global-prefix output or dense attention.

[Kernel](../tensor_cuda/src/apa_sp1_1.cuh:5), [launcher](../tensor_cuda/src/apa_sp1_1.cuh:149), [entry routing](../tensor_cuda/src/kernels.cu:7477), [unchanged merge](../tensor_cuda/src/kernels.cu:1672), [CPU pin](../tensor_cuda/tests/test_apa_sp1_1.py:42). All L=1 SP calls use num_parts=max(1,ceil(S/2048)); part_keys is the existing TC_APA_SPLITK_PART_KEYS=128*16 via the existing fixed-layout helper. Thus S=2048/8192/32768 gives P=1/4/16. The inherited 128-thread DMAX dispatch uses per-thread dots for cap<=64 and four warp-cooperative dots per tile otherwise. Inclusive tile prefixes carry forward across tiles. Prefill retains the original SP1 kernel. The same TC_APA_SP=1 opt-in applies, and existing attention kernel bodies—including SP1—are byte-identical. The optional sink uses the original merge exactly once.

The final host build has 24 split-K specializations, zero stack/spill bytes. FP32 D64 uses 95 registers, D128 uses 40. These are compiler resource observations, not achieved occupancy or speed. [Build log](../artifacts/apa_sp1/sp1_1_build_final.log), [compiler resources](../artifacts/apa_sp1/sp1_1_compiler_resources.json), [source patch](../artifacts/apa_sp1/source_additions_sp1_1.patch).

Part B — adjudication

Verdict: **EMULATOR_ORDER_SENSITIVE**, with strong CPU reconstruction and GPU confirmation still unrun. Both implement the intended mathematical selector, but the NumPy reference does not reproduce CUDA lane/FMA/stats order at threshold boundaries. The existing kernel and original mathematical emulator are unchanged; a separate ordered diagnostic emulator and read-only GPU diagnostic hook were added. The original G2 FAIL remains a FAIL.

The exact seed is SeedSequence([20260907,10]); H=KVH=4, group=1 rules out GQA head remapping, and S=L=2048 gives the same causal bound query+1. [Diagnosis receipt](../artifacts/apa_sp1/sp1_1_diagnosis.1788660692574177856.json), [CPU reconstruction](../scripts/apa_sp1_1_diagnose.py), [ordered reference](../tensor_cuda/tests/apa_sp1_1_reference.py), [baseline SASS showing FFMA](../artifacts/apa_sp1/sp1_1_baseline_sass.txt).

| Head | Query | Key | Original abs bulk / threshold bits | CUDA-order abs bulk / threshold bits | Decision | Output effect |
|---|---|---|---|---|---|---|
| 1 | 1271 | 1199 | 0x3f9db1d6 / 0x3f9db1d6 | 0x3f9db1d6 / 0x3f9db1d7 | refine -> skip | max_abs 0.00002630055 |
| 3 | 1137 | 737 | 0x3fc7c231 / 0x3fc7c230 | 0x3fc7c230 / 0x3fc7c232 | refine -> skip | max_abs 0.00158654526 |

Three further differences are key 0/query 0 on heads 0, 1, 3: fused variance arithmetic leaves a tiny positive residual for a one-key row, lifting the threshold. Those decisions have zero output effect without a sink because there is only one value. The receipt records all five keys and all float values. At the dominant h3 witness, changing dots alone or stats alone does not flip the key; their combined fp32 order does. This is a threshold-boundary/order effect, not changing >= into >.

Reconstructed full-output max_abs=0.0015865452587604523; supplied GPU max_abs=0.0015865489840507507. Reconstructed RMSE=6.030458721079378e-06; observed RMSE=6.030553212643941e-06. Agreement of max_abs within 3.8e-9 localizes the failure. The old receipt contains aggregate errors, not saved baseline masks; these keys are reconstructed CPU evidence until the lead runs `partb`. Nothing is widened to hide the original failure.

Part C — matching and scoring

There are **14 unmatched decode rows**, tabulated individually with delta, calibration residual, realized fractions, signed holdout gap and excess beyond 0.02 in [the per-row explanation](../artifacts/apa_sp1/sp1_matching_explanations.md). [JSON](../artifacts/apa_sp1/sp1_matching_explanations.json) includes all 24 decode rows and their per-head counts, query norms, final bulk maxima and CPU-versus-receipt reconstruction. Calibration aggregates 128 independent query histories; each GPU decode class has 4 or 8. A long shared key list does not eliminate query-history variation in prefix records. Calibration grid residuals are small, while holdout gaps exceed the tolerance in both directions. Causal/noncausal decode sees all S keys but the registered shape seeds differ. GQA and MHA shapes also use distinct seeds and 4/8 query rows.

**Every unmatched row invalidates its matched-budget speed/deviation comparison.** Its raw timing and raw dense deviation remain measurements for that actual, different work budget. All 24 decode classes reconstruct both the SP and z selection counts exactly from their registered seeded inputs, including every unmatched row. No holdout recalibration was performed. There are 47 G2 passes, 34 matched fractions, 33 jointly eligible classes out of 48. The summary now uses the immutable original receipt manifest; changes to the implementation fingerprint cannot erase SP1 history.

| Prediction | Verdict | Qualification |
|---|---|---|
| P2_prefill_overlap_ge_0_8 | MISS | All 24 count-bearing prefill rows; no timing/matching filter on an overlap prediction. |
| P2_decode_overlap_less | MISS | Microaggregate prefill=0.454184277, decode=0.472696150; all rows. |
| P3_half_shapes_deviation | HIT | 33/48 registered shapes meet deviation<=2 with G2 PASS and matched fraction; threshold 24. |
| P3_all_prefill_speed | MISS | 5 eligible counterexamples below1.4; eligible 23/24. |
| P3_decode_speed_smaller | MISS | Coverage MISS unless both groups fully eligible; eligible medians prefill=1.497820, decode=0.180685; eligible 23/24 and 10/24. Observed subset supports direction but does not establish all-class prediction. |
| A3_prefill_overlap_le_0_65 | HIT | All 24 prefill rows with complete counts. |
| A4_decode_speed_le_1 | MISS | Coverage MISS: eligible 10/24. Every raw decode speed <=1 is separately visible; unmatched work budgets do not count as matched-comparison hits. |
| A5_prefill_majority_below_1_4 | MISS | 5/24 eligible rows below1.4; majority requires at least13. |
| P2_monotone | HIT | Bulk-prefix safety proof and 40,000 CPU draws; not an exact-score guarantee. |

[Final prediction table with every receipt path](../artifacts/apa_sp1/gpu_summary.md), [machine-readable scoring](../artifacts/apa_sp1/gpu_summary_sp1_final.json). P3 decode-smaller and A4 are explicitly coverage MISS, not numerical counterexamples: their eligible medians/directions support the hypotheses, but missing matched-class coverage cannot earn a full-class hit. All requested P2/P3/A3/A4/A5 verdicts are HIT/MISS; no PENDING remains in the finalized scoring. SP1.1's new GPU predictions remain BLOCKED because its receipts do not yet exist.

Gates and lead commands

G1: **52 passed in 3.44s**, including all 37 SP1 tests, 15 SP1.1 tests, 1,000 randomized partition trials, the key-specific diagnosis, all previous kernel bodies and source/registration/receipt pins. [G1 receipt](../artifacts/apa_sp1/sp1_1_g1_1788661437075136419.json), [final audit](../artifacts/apa_sp1/G1_SP1_1_FINAL.json). Reproduce with `timeout 300s bash scripts/apa_sp1_1_cpu.sh unit`; full reconstruction with `timeout 300s bash scripts/apa_sp1_1_cpu.sh diagnose`. Host module provenance matches the final build manifest. The supplied original legacy GPU receipt reports 95 passed, 3 skipped; it is prior lead evidence, not a new execution on this seat.

G2/G3: **BLOCKED_NO_GPU**. [Blocked report](../artifacts/apa_sp1/GPU_BLOCKED_SP1_1.json) contains exact commands and bounds. [One command per class](../artifacts/apa_sp1/sp1_1_lead_commands.txt) lists the sink/dtype boundary, all 24 decode classes, Part B confirmation, and summaries. Run each as a separate foreground invocation from this worktree. Example:

```bash
timeout --signal=TERM --kill-after=5s 590s bash scripts/apa_sp1_lead_gpu.sh splitk boundary
timeout --signal=TERM --kill-after=5s 590s bash scripts/apa_sp1_lead_gpu.sh splitk decode_s8192_d64_c0_h4_kv4
timeout --signal=TERM --kill-after=5s 590s bash scripts/apa_sp1_lead_gpu.sh partb
timeout 30s bash scripts/apa_sp1_lead_gpu.sh splitk-summary
timeout 30s bash scripts/apa_sp1_lead_gpu.sh summary
```

Calling `splitk` without a class runs only the next incomplete class and returns. Each GPU invocation takes `/tmp/forge-gpu.lock` with a 5-second wait, rejects existing compute PIDs, bounds the worker at 510 seconds and the full invocation at 590 seconds plus 5 seconds termination grace, and cools for 30 seconds in foreground Python while holding the lease. No automatic retry or retuning. G2 requires full-output emulator equivalence, exact SP diagnostic masks, and diagnostic/timing output identity; G3 follows G2 per class. Separate split sink/dtype boundaries cover 12 cases, including ragged partitions and separate value widths.

Deviations, residual risks, RED and process safety

The request's global-output equality/work-only interpretation is false; the registered and tested claim is selection containment plus partition-selected merge equality. The old emulator was not semantically fixed because the diagnosis is arithmetic-order sensitivity. The second build statically unrolled only the new D64 accumulator after the first compiler receipt exposed local-memory indexing; final stack/spills are zero. Separate amendment records added diagnostics and stricter mask gates. Original thresholds and kernel bodies did not change.

RED remains: new CUDA runtime behavior, split sink/low-precision equivalence and speed gates are unrun. D64 register pressure and D128 tile barriers may miss speed predictions. Resetting partition maxima increases refinement at frozen delta and may make more classes unmatched; extra refinement does not mathematically guarantee lower dense deviation. CPU random tests and SASS inspection cannot prove CUDA race freedom or performance. No model/corpus quality test was performed.

No git, subagents, shell background jobs, GPU workloads, service changes, or foreign process termination were used. All compiler/test commands ran as timeout-bounded foreground children, each under 10 minutes; all have completed. All mutations stayed in the writable worktree. Initial read-only AGENTS filename discovery extended to parent directories; no other-checkout source, runtime or build was used or edited. The required memory quick pass was read-only. Registration and supplied receipt bytes are unchanged.

Actual model **gpt-6-astra**, effort **xhigh**, verified from `logs/apa_sp1_1_astra_r2.log:5-12`; [execution metadata](../artifacts/apa_sp1/execution_metadata_sp1_1.json). Branch/HEAD are dispatch-supplied and not git-verified.
