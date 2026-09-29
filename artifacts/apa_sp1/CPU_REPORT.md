# APA-SP1 CPU result and lead handoff

**Q1:** impossible for irrevocable selective streaming; a buffered all-exact one-input-read construction exists. No unrestricted impossibility or GPU equivalence is claimed.
**Q2:** implemented the distinct bulk-prefix-maximum rule, with a proof of conservative coverage of the terminal bulk-max tail. It changes selection/output semantics.
**Q3:** GPU sweep blocked by no CUDA device. Runnable gates are delivered; no GPU timing, output, peak-memory or model-quality result exists.

Evidence class for every number in the following table: **CPU randomized unit test**, never GPU kernel sweep. **this establishes nothing about model quality**.

Four registered row classes (prefill_d64, prefill_d128, decode_d64, decode_d128) each completed 10,000 independent full-key (q,K) draws; total 40,000. All class lengths were exercised, causal flags alternated, with per-draw values of width 4. These are per-row workloads, not 10,000 complete attention tensors per GPU shape.
Online selection masks matched offline prefix scans exactly: 0 mismatches. Maximum output absolute difference 3.57627869e-07, with frozen atol=rtol=1e-3. Separate tensor tests cover causal/GQA/MQA/sinks and distinct D/VD.
Microaggregated z-tail recall: prefill 45.193999%; decode 45.703051%.

| row geometry | draws | frozen delta | SP fraction | z fraction | recall overlap | matched | CPU dense-error ratio |
|---|---:|---:|---:|---:|---:|---|---:|
| decode_s2048_d128_c0 | 1660 | 2.00 | 0.152311 | 0.154994 | 0.449016 | True | 0.896594 |
| decode_s2048_d128_c1 | 1680 | 2.05 | 0.159640 | 0.154821 | 0.454763 | True | 0.892929 |
| decode_s2048_d64_c0 | 1660 | 1.95 | 0.141878 | 0.154660 | 0.436428 | True | 0.931462 |
| decode_s2048_d64_c1 | 1680 | 2.00 | 0.155618 | 0.154809 | 0.449879 | True | 0.911796 |
| decode_s32768_d128_c0 | 1692 | 2.75 | 0.148394 | 0.154894 | 0.457082 | True | 0.899968 |
| decode_s32768_d128_c1 | 1628 | 2.75 | 0.149575 | 0.154844 | 0.461479 | True | 0.883294 |
| decode_s32768_d64_c0 | 1692 | 2.75 | 0.152481 | 0.154793 | 0.454078 | True | 0.895073 |
| decode_s32768_d64_c1 | 1628 | 2.80 | 0.162751 | 0.154822 | 0.459342 | True | 0.880867 |
| decode_s8192_d128_c0 | 1648 | 2.45 | 0.163938 | 0.154811 | 0.467507 | True | 0.855690 |
| decode_s8192_d128_c1 | 1692 | 2.40 | 0.150345 | 0.154798 | 0.455701 | True | 0.888542 |
| decode_s8192_d64_c0 | 1648 | 2.35 | 0.145886 | 0.154869 | 0.447769 | True | 0.931074 |
| decode_s8192_d64_c1 | 1692 | 2.40 | 0.154533 | 0.154780 | 0.452111 | True | 0.910368 |
| prefill_s2048_d128_c0 | 1648 | 2.00 | 0.153344 | 0.154917 | 0.451114 | True | 0.894605 |
| prefill_s2048_d128_c1 | 1692 | 1.80 | 0.143748 | 0.155113 | 0.432022 | True | 0.511563 |
| prefill_s2048_d64_c0 | 1648 | 2.00 | 0.154841 | 0.154830 | 0.449018 | True | 0.902553 |
| prefill_s2048_d64_c1 | 1692 | 1.75 | 0.139739 | 0.155153 | 0.429165 | True | 0.549899 |
| prefill_s512_d128_c0 | 1660 | 1.55 | 0.154456 | 0.155208 | 0.442470 | True | 0.908633 |
| prefill_s512_d128_c1 | 1680 | 1.35 | 0.160155 | 0.155804 | 0.431158 | True | 0.582061 |
| prefill_s512_d64_c0 | 1660 | 1.50 | 0.145833 | 0.155399 | 0.430370 | True | 0.944204 |
| prefill_s512_d64_c1 | 1680 | 1.35 | 0.160242 | 0.155443 | 0.428105 | True | 0.596406 |
| prefill_s8192_d128_c0 | 1692 | 2.45 | 0.161978 | 0.154874 | 0.462679 | True | 0.851174 |
| prefill_s8192_d128_c1 | 1628 | 2.25 | 0.149221 | 0.154917 | 0.448679 | True | 0.646175 |
| prefill_s8192_d64_c0 | 1692 | 2.40 | 0.160464 | 0.154868 | 0.455599 | True | 0.888808 |
| prefill_s8192_d64_c1 | 1628 | 2.25 | 0.153938 | 0.154824 | 0.445523 | True | 0.455459 |

Recall = intersection / z-selected count. Matched requires absolute realized-fraction difference <=0.02. The last column is the ratio of aggregate CPU output-error norms versus the dense fp32 reference; it does not score GPU P3. Jaccard and raw counts/squared errors are in cpu_summary.json.

Registered predictions scored from CPU evidence:

- P1_restricted_prefix: HIT
- P1_unrestricted_impossibility: MISS (buffered construction)
- P2_monotone: HIT (proof + suffix tests)
- P2_prefill_overlap_0.8: MISS
- P2_decode_lower: MISS
- A1: HIT
- A2: HIT
- A3_prefill_overlap_le_0.65: HIT
- A6_no_cpu_equivalence_failures: HIT
- P3_and_A4_A5_gpu_speed: BLOCKED: no GPU; CPU deviations do not score G3

The P2 overlap scores apply to the explicitly registered **bulk** running maximum. A variant using the mixed-score softmax maximum was not benchmarked. P3 GPU deviation/speed and A4/A5 speed remain BLOCKED.

Q1 witness (CPU unit evidence; q=[1], exact scores [2,0,0,-10], V=[1,0,0,0]):

| last bulk | threshold | refine mask | output |
|---:|---:|---|---:|
| 0 | 0.698788762 | [True, False, False, False] | 0.71123451 |
| 10 | 7.10887671 | [False, False, False, True] | 0.576111376 |

The first key changes selection despite an identical prefix. Tests also force the wrong first decision while holding each suffix fixed and check an output discrepancy beyond tolerance. See THEORY.md for the restricted proof, buffered loophole, general monotone characterization and limits.

Build / preservation (code-inspection and host-compile evidence):

- CUDA/C++ build and link succeeded for sm_89. The new compiled family has 24 specializations, zero stack/spill bytes; production fp32 caps 64/128 use 46/54 registers. These are compiler receipts, not measured occupancy.
- kernels.cu:7351 adds the new kernel and launcher; bindings.cpp:662 adds the explicit tc._C.apa_selective_attention_sp entry; ops.h:12 adds one declaration. TC_APA_SP must equal 1; default OFF. No existing dispatcher routes into SP.
- Every original byte of all three edited existing files is recoverable exactly by removing marked additions. In particular, ALL existing kernel bodies/families remain byte-identical. Hash test and original snapshots are in the registration.
- Production candidate allocates only output, O(B*H*L*VD), with no S-dependent workspace. Diagnostic uint8 masks are optional, quadratic and excluded from timings. The baseline and candidate share floating Kq residency; no KV-compression claim.
- No split-K candidate: preserving incoming prefix state requires a scan or ordered carry; decode underfill remains unresolved.

Registration sha256: `12059d49f39abe9450b34989f06ac5ca8a736d80e3127c9d41c1ed97e650c9fe`. Calibration sha256: `49a561b807b95ce621cfcc97ee9bcc30a4c7dbe827715db650d38f47fa9d9991`. Registration remains immutable; amendment_001.json documents instrumentation/build choices without threshold changes.

G1: final CPU unit log g1_cpu_final_r2.log reports 37 passed. Mutation gate: 5/6 nonerror mutants killed (83.33%); the exclusive-prefix survivor is decision-equivalent for delta>=0. Original source never mutated. The unchanged legacy suite reports 49 failed, 1 passed, 48 skipped on this no-GPU seat. The selector fails with `RuntimeError: cudaMalloc failed: no CUDA-capable device is detected`. Full G1 is therefore **BLOCKED**, not green. Raw logs and source hashes are in the timestamped g1 receipt.

Lead commands (run from this worktree; each GPU call handles only one class):

```bash
timeout 540s bash scripts/apa_sp1_build.sh > artifacts/apa_sp1/build_lead.log 2>&1
bash scripts/apa_sp1_lead_gpu.sh run boundary
bash scripts/apa_sp1_lead_gpu.sh run legacy
bash scripts/apa_sp1_lead_gpu.sh run selector
bash scripts/apa_sp1_lead_gpu.sh list
bash scripts/apa_sp1_lead_gpu.sh run prefill_s512_d64_c0_h4_kv4
bash scripts/apa_sp1_lead_gpu.sh resume
bash scripts/apa_sp1_lead_gpu.sh summary
```

Repeat resume to run the next pending class, once per invocation. It checks source/module/registration/calibration fingerprints, retains previous receipts and never interprets missing/failed/unmatched rows as hits. The worker is bounded to 510 seconds inside the registered 540-second ceiling; the wrapper bounds the invocation to 590 seconds plus at most 5 seconds signal grace. One numeric GPU selection, exclusive /tmp/forge-gpu.lock, 30-second foreground Python cooldown while holding the lease, and busy-process refusal are enforced. No shell background jobs or foreign-process kills. G2/G3 remain blocked until the lead executes them.

Residuals/deviations: no Rust source in the dispatched worktree, so the existing independent NumPy reference and original host z-threshold are the available pins. Q1 proof explicitly assumes irrevocable selective streaming; buffered Q1 construction is CPU-only and not a profitable candidate in G3. Prefix rule safety concerns terminal BULK scores, not dense exact dominance; unbounded quantization error has no guarantee. Finite tested fp32 domain does not prove all-input CUDA float equivalence. Synthetic perturbed Kq matches the existing test mechanism but is not quantizer/model evidence. CPU class counts are pooled across the listed lengths/causal flags. G2/G3 and independent blind verification remain RED/pending. Author-run mutation/unit tests do not replace lead-owned blind verification.

Process safety: no git, no subagents, no main/other-worktree changes, no live-service changes, no GPU workload here, no foreign process terminated. Model `gpt-6-astra`; actual configured effort `xhigh` according to the launch log; ultra was authorized but not configured. Ledger: docs/APA_SP1_LEDGER.md.
