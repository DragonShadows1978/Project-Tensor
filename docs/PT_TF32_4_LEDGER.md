# PT-TF32-4 implementation ledger

**Fork rebuilt; 78 current CPU tests pass. GPU certification is BLOCKED for
this seat.** Historical CPU checks retain three obsolete binary-pin failures
(146 pass). This is an author handoff, not blind verification or a new GPU pass.

Immutable plan: [orders/PT_TF32_4_FINAL.md](../orders/PT_TF32_4_FINAL.md).
Read `/mnt/Shared/HOUSE_RULES.md` first. No git, subagents, GPU discovery or GPU
execution, live-engine imports, external-tree writes, or lock operations.
All executed Python/build/test commands use `CUDA_VISIBLE_DEVICES=''`;
compiler temporaries and pytest directories are inside this worktree.

## Registration and evidence order

1. Read the order, previous ledger, sources, and slot-11 receipts. The order
   already disclosed failures, so this is a registered correction after the
   old experiment, not a claim of blind pre-observation registration.
2. `scripts/pt_tf32_4_register.py` derives thresholds using only the previous
   registration's dimensions/rounding model and a fixed family tail risk.
   It never loads slot-11 numerical results. It creates
   [REGISTRATION.json](../artifacts/pt_tf32_4/REGISTRATION.json), hash
   `e4667c4a67f525dd24c390efa9b8c859fcdd431e831916821a296cf9043de5ac`,
   before implementation, CPU tests, or historical re-scoring. The four
   derived rows were printed before the comparison command ran.
3. [BASELINE_SHA256.json](../artifacts/pt_tf32_4/BASELINE_SHA256.json) pins
   902 files, including historical registrations/receipts. Eight large legacy
   files have size/mtime pins only; no byte-hash claim is made for those dumps.
4. The separate [AMENDMENT_001.json](../artifacts/pt_tf32_4/AMENDMENT_001.json)
   corrects an RN-only WMMA accumulation assumption from NVIDIA's PTX
   documentation. Hash:
   `e140d67e24ec9d525f623a096d383773647048a840c0a7b85bd0afca20b7fc78`.
   It was written after the first CPU run and before implementing/testing
   the amended term. No native result prompted it. The original registration,
   first edge derivation and failed CPU receipt remain intact.

All other numerical/model/selection/attention-speed/default/storage rules
remain unchanged. GEMM speed becomes a diagnostic by this order's explicit
authorization; no prior RED verdict is overwritten.

## dK tail derivation, before comparison

Evidence class: rounding-model reasoning. Let `uT=2^-11`, `u32=2^-24`, and
`Q=max(L,S,D,VD)`. The inherited six-event dK expectation is

```text
epsilon = sqrt(6) * uT/sqrt(3) + sqrt(Q)*u32
relative_L2_bound = 3*epsilon                         (unchanged)
N = B*KVH*S*D                                      (output elements)
R = max(abs(reference))
```

Normalization matters. Under a homoscedastic model, the per-element absolute
standard deviation would be `epsilon*RMS(reference)`, and its peak-normalized
value would be `epsilon*RMS(reference)/R`, not simply epsilon. Causal dK
entries are heteroscedastic: their sums have different numbers of contributing
queries and selected pairs. We therefore register the **additional explicit
engineering assumption** that every centered error has sub-Gaussian scale
at most `sigma=epsilon*R`. This conservative common peak-scale envelope is
**not implied by an L2 average**. Cancellation, selection discontinuities,
and biased errors can invalidate the model; there is no worst-case theorem
or measured failure-probability claim. The code reports reference RMS and
peak separately, plus the tighter homoscedastic sigma as a diagnostic.

Gumbel/David–Nagaraja give the leading Gaussian-max scale
`sigma*sqrt(2 ln N)`; this is a leading envelope here, not an exact finite-N
expected value. The finite two-sided sub-Gaussian union bound is

```text
P(max_i |error_i| > sigma*t) <= 2*N*exp(-t^2/2).
alpha_family = 0.001; comparisons = 4 shapes * 3 arms = 12.
t_bound = sqrt(2 ln(2*N*12/0.001)).
normalized_max_bound = epsilon*t_bound.
margin = epsilon*(t_bound - sqrt(2 ln N)).
```

The 0.1% family risk is a stated engineering choice, fixed before re-scoring,
not estimated from observed maxima. Inter-element independence is unnecessary
for this union bound; the stated marginal scale assumption is necessary.
Only dK receives this replacement. Out/LSE/dQ/dV retain their previous maximum
bars, every L2 bar is unchanged, and zero-reference arrays require exact zero.

These numbers were registered and printed **before** historical comparison:

| Case; L/D | N | epsilon | Unchanged L2 bar | Leading max envelope | Added margin | Max bar |
|---|---:|---:|---:|---:|---:|---:|
| 0; 2048/96 | 3145728 | 0.0006932313643071851 | 0.0020796940929215554 | 0.003792115714553032 | 0.001114411631931218 | 0.00490652734648425 |
| 1; 2048/128 | 4194304 | 0.0006932313643071851 | 0.0020796940929215554 | 0.0038283996893140695 | 0.0011062242264851664 | 0.004934623915799236 |
| 2; 4096/96 | 6291456 | 0.0006943486632681129 | 0.0020830459898043387 | 0.0038852146620742527 | 0.001096756321480305 | 0.004981970983554558 |
| 3; 4096/128 | 8388608 | 0.0006943486632681129 | 0.0020830459898043387 | 0.003920750893644804 | 0.0010889825803635034 | 0.005009733474008307 |

### Comparison with immutable slot 11

Evidence class: CPU re-scoring of existing GPU measurements, **not new GPU
certification**. [HISTORICAL_REASSESSMENT.json](../artifacts/pt_tf32_4/HISTORICAL_REASSESSMENT.json)
stores the original fields beside the new rule. All 12 retain passing L2 and
fall inside the registered tail envelope. No exceeded-tail branch was taken.

| Case | Arm | Original rel-L2 | Original normalized max |
|---|---|---:|---:|
| 0 | isolated | 0.0003509975124118612 | 0.0016735099879850019 |
| 0 | downstream | 0.0003581262858441721 | 0.0013148795653161564 |
| 0 | same native state | 0.00071077219951232 | 0.003862075850069606 |
| 1 | isolated | 0.00029557128611958886 | 0.00037145962444432793 |
| 1 | downstream | 0.0004500038680808381 | 0.002203109868528126 |
| 1 | same native state | 0.0004136411833421751 | 0.0017873926051520687 |
| 2 | isolated | 0.0003712706125180213 | 0.0015990581492353847 |
| 2 | downstream | 0.0006050142236766192 | 0.004581897533772324 |
| 2 | same native state | 0.0008468804390934714 | 0.00458190274851278 |
| 3 | isolated | 0.00035171815077387217 | 0.0009089299309363516 |
| 3 | downstream | 0.0003123389362514542 | 0.0005620973594186283 |
| 3 | same native state | 0.00043987358254233384 | 0.002807431244792845 |

All unchanged out/LSE/dQ/dV, selection and attention-time rules also pass when
applied to these stored values. A future exceeded tail stays RED; the runner
records its worst coordinate/value and reference scale, with no bound fitting.
Full native saved-state capture for a focused diagnosis would be lead work.

## GEMM: measure dispatch and accuracy

Evidence class: installed CUDA headers, official NVIDIA documentation, source
inspection and historical kernel receipts. Every one of 54 old GEMM cells
(nine shapes, three directions, two allocation policies) has algorithm 21,
required flags and accuracy <=1e-3, **including every dWeight cell**.
Maximum rel-L2 by direction is 0.0002941052281717468 (forward),
0.00029417958973728054 (dInput), 0.000294404476267947 (dWeight).
No non-HMMA case or missing dWeight route was found; the existing mandatory
HMMA filter is preserved.

```text
0x40202 = 0x40000 INPUT_TF32 | 0x200 ACCUMULATOR_32F | 0x2 HMMA
compute_type = 77 = CUBLAS_COMPUTE_32F_FAST_TF32
CUBLAS_STATUS_INVALID_VALUE = 7
```

The old query cast integer 8 to a capability enum absent from the installed
12.6 header. Its status 7 was an invalid/unsupported-attribute usage error,
not evidence of CUDA-core execution. Removed that call. The supported named
`CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS` query (attribute 15, uint64_t) is now
the sole capability query: actual status, returned bytes and flags are
recorded. Old mathmode fields are -1, explicitly unsupported/not attempted;
we do not fabricate a successful deprecated-query response. Header locations:
`/usr/local/cuda-12.6/include/cublas_api.h:97`,
`/usr/local/cuda-12.6/include/cublasLt.h:938` and `:2230`.

The new harness captures readback **immediately after the shape's actual
TF32 matmul**, before interleaved timing can change last-call metadata.
It requires the exact M/N/K, transpose/layout, alignment, compute type,
successful 8-byte capability query, actual algorithm ID, required flags,
workspace range and <=1e-3 FP64 relative L2. A missing/stale/wrong-mode
receipt fails. Both allocation policies and all 54 cells remain mandatory.
CPU tests inject FMA-only, wrong input precision, wrong shape/transposes,
invalid status, truncated capability data and low alignment; all are rejected.

### Why the 5x bar was wrong

Evidence class: roofline reasoning, not a bandwidth/profiler measurement.
For M/N/K GEMM, `F=2MNK` and the ideal compulsory FP32 traffic is
`B=4(MK+KN+MN)` bytes. Thus `I=F/B`; attainable rate is bounded by
`min(effective_compute_ceiling, bandwidth*I)` before launch/allocation and
other overhead. TF32 keeps FP32 input/output storage; it does not divide the
traffic by its peak compute ratio. M=4096 alone does not establish saturation.

| Forward shape K/N, M=4096 | Ideal FLOP/byte | Pooled SGEMM TFLOP/s | Pooled TF32 TFLOP/s | Speedup |
|---|---:|---:|---:|---:|
| 1024/1024 | 227.56 | 18.42 | 29.64 | 1.610 |
| 1024/4096 | 341.33 | 20.17 | 32.37 | 1.605 |
| 1024/1792 | 281.10 | 20.60 | 33.18 | 1.610 |
| 1792/1024 | 281.10 | 18.81 | 31.10 | 1.654 |
| 1024/768 | 198.19 | 20.16 | 27.84 | 1.381 |
| 768/2048 | 245.76 | 19.97 | 29.82 | 1.493 |
| 1024/320 | 115.06 | 16.54 | 27.31 | 1.651 |
| 256/2048 | 107.79 | 19.02 | 25.72 | 1.352 |
| 1024/8192 | 372.36 | 21.41 | 31.90 | 1.490 |

The raw-global model-forward SGEMM receipt is 12.38 TFLOP/s; pooled and other
directions are faster, reaching 25.58. Across all receipts speedup is
1.2108160895204498–3.676670982171929 (the order's 1.2–2.4 description was not
the full range). dWeight spans 1.2108160895204498–1.8496732001278287.
cuBLASLt's heuristic picks eligible algorithms for these actual dimensions,
layouts and workspaces; an HMMA selection does not imply a 5x speedup over
an already optimized SGEMM. The roofline is an upper ceiling, never a promised
speedup floor. No claim that a particular shape is DRAM-bound is made without
traffic counters. Allocation asymmetry in the raw-global arm remains labeled;
the pooled arm pools both outputs. Speed/time samples are retained in full.

## The two edge cases: an input-derived dV interval

Evidence class: source/rounding analysis and CPU fixtures. The old failures
are dV, with the same three recorded coordinates and values. No attention
kernel arithmetic changed in this order. The new test applies the same
derived rule to dV in **all 40** padding/grouped-head cases; dQ/dK retain
`rtol=0.003, atol=2e-5`. The two additional independent FP64 edge gates are
unchanged. There are no shape IDs or observed-error constants in the budget.

For each all-selected visible pair, use the *native saved FP32 LSE* as fixed
API state, FP64 input dot `s`, and `z=s-LSE`. With `u=2^-24` and
`gamma_n=n*u/(1-n*u)`, derive:

```text
E_s = abs(scale)*gamma_(2D+1)*sum_d abs(q_d*k_d)
E_z = E_s + u*(abs(z)+E_s)             (plus CPU FP64 evaluation allowance)
U   = 2 + floor(1.173*(abs(z)+E_z))
p_lo = exp(z-E_z)*(1-U*2^-23)
p_hi = exp(z+E_z)*(1+U*2^-23)
```

NVIDIA documents the `__expf` ULP envelope. Normal-probability ULP relative
spacing is <=2^-23. Round interval endpoints outward to FP32, then apply
monotone RNA_TF32; the installed CUDA `mma.h` emits `cvt.rna.tf32.f32`.
Let those coefficients be `Plo/Phi`, and the CPU center `P0`. For each pair,
`deltaP=max(abs(Plo-P0),abs(Phi-P0))`. Away from a possible midpoint crossing,
deltaP is zero; near one, it includes a full TF32-bin jump. Singleton p=1 and
masked p=0 are handled exactly. Sum over both grouped heads:

```text
J = sum_heads deltaP.T @ abs(RNA_TF32(dO))
A = sum_heads max(abs(Plo),abs(Phi)).T @ abs(RNA_TF32(dO))
m = (H/KVH)*ceil(L/16)*16
atol_per_element = J + gamma_m(u_acc)*A + FTZ_floor + CPU_FP64_allowance
```

AMENDMENT_001 sets `u_acc=2^-23`, with
`FTZ_floor=m*2^-126/(1-m*u_acc)`: one-ULP faithful accumulation covers RN and
directed rounding models. PTX leaves WMMA accumulation order/rounding
unspecified, so this remains an explicitly conditional FP32 error model,
not an unconditional vendor guarantee of internal elementary additions.
The code rejects nonfinite/subnormal-probability/product or non-all-selected
fixtures rather than applying this budget outside its domain. It uses no
near-zero relative tolerance and no arbitrary absolute floor.

The amended CPU fixture calculation, made without native candidate values,
gives these witnesses. CPU reference values equal the corresponding desired
values printed in the old log. The old native saved arrays were not retained,
so this does **not** replay the full old native test.

| L/S; dV coordinate | Midpoint contribution J | Accumulation term | Derived absolute tolerance | Old observed absolute discrepancy |
|---|---:|---:|---:|---:|
| 17/31; (0,1,1,30) | 0.00010612607002258301 | 0.000006259018563784371 | 0.00011238508858636738 | 0.000024646520614624 |
| 17/31; (0,1,1,42) | 0.0001047775149345398 | 0.000005012658530998692 | 0.00010979017346553849 | 0.0000332072377 |
| 33/35; (0,1,0,22) | 0.00011371448636054993 | 0.000023426330048795547 | 0.00013714081640934546 | 0.0000230306759 |

The interval allows all possible contributing midpoint crossings, so it is
more conservative than the single observed crossing. These tolerances are
not claimed minimal. [EDGE_DERIVATION_CPU_002.json](../artifacts/pt_tf32_4/EDGE_DERIVATION_CPU_002.json)
contains the numbers, with its registration/amendment pins; the unamended
derivation remains as history. A rebuilt-native pass still requires the lead.

Verbatim old GPU failures, preserved at `pt_tf32_3/lead_slot_01/gpu_units.log`:

```text
Not equal to tolerance rtol=0.003, atol=2e-05
Mismatched elements: 2 / 3968 (0.0504%)
[0, 1, 1, 30]: -0.0013424102216959 (ACTUAL), -0.001367056742310524 (DESIRED)
[0, 1, 1, 42]: -0.0016745328903198242 (ACTUAL), -0.0016413256525993347 (DESIRED)
Max absolute difference among violations: 3.32072377e-05
Mismatched elements: 1 / 4480 (0.0223%)
[0, 1, 0, 22]: 0.0009828303009271622 (ACTUAL), 0.0010058609768748283 (DESIRED)
Max absolute difference among violations: 2.30306759e-05
2 failed, 58 passed in 0.83s
```

## Execution receipts and preservation

Commands, all in `/mnt/ForgeRealm/wt/pt-tf32` with CUDA hidden:

```bash
CUDA_VISIBLE_DEVICES='' python3 -B scripts/pt_tf32_4_register.py
CUDA_VISIBLE_DEVICES='' python3 -B scripts/pt_tf32_4_build.py
CUDA_VISIBLE_DEVICES='' python3 -B scripts/pt_tf32_4_grapa.py prepare
CUDA_VISIBLE_DEVICES='' python3 -B scripts/pt_tf32_4.py host
CUDA_VISIBLE_DEVICES='' python3 -B scripts/pt_tf32_4.py seal
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q \
  tests/test_pt_tf32_4_cpu.py -p no:cacheprovider \
  --basetemp=artifacts/pt_tf32_4/cpu_final_tmp
CUDA_VISIBLE_DEVICES='' python3 -B scripts/pt_tf32_4.py blocked
```

Actual test command arrays/return codes are in `CPU_FINAL_RECEIPT.json` and
`CPU_HISTORICAL_RECEIPT.json`. The first preseal run had a test expectation
typo (475 was mis-added as 575); it did not expose a changed gate budget:

```text
E           assert 475 == 575
1 failed, 76 passed, 1 deselected in 1.28s
77 passed, 1 deselected in 1.16s
```

The second preseal run corrected that assertion and exercised the amended
edge model. After the final source seal, the full current suite includes the
manifest test. Verbatim final receipts:

```text
BUILD_RC=0 SOURCE_PINS_UNCHANGED=True RECEIPT=/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_tf32_4/build_01/receipt.json
HOST_DESCRIPTOR_RC 0 CASES 27
78 passed in 1.08s
CPU_FINAL_RC 0
60 tests collected in 0.12s
3 failed, 146 passed in 3.26s
CPU_HISTORICAL_RC 1
E           ValueError: fork binary drift
E           ValueError: fork binary drift
E       ValueError: fork engine drift
PRESERVED 898 OF 902 UNEXPECTED_CHANGES []
HISTORICAL_RECEIPTS_BYTE_IDENTICAL True
BF16_FP16_MATMUL_SOURCE_BYTE_IDENTICAL True
KERNELS_CU_AND_OBJECT_BYTE_IDENTICAL True
ATTENTION_TF32_SOURCE_BYTE_IDENTICAL True
```

The three historical failures are respectively the PT-TF32-2, -3 and -1
engine-registration checks. Their historical binary hashes cannot match the
rebuilt binary. They remain RED; nothing is skipped or updated to conceal
them. The new PT-TF32-4 registration, exact model-state pins, native host
contract and manifest pass. 78 tests cover all 40 edge fixtures, separate
max/L2 defects, normalized scale, zero/nonfinite rejection, grouped-head
omission, off-midpoint tightness, invalid dispatch and GPU-hidden guards.

[PRESERVATION_RECEIPT.json](../artifacts/pt_tf32_4/PRESERVATION_RECEIPT.json)
confirms only four baseline files changed: `tf32_gemm.cu`, the two GPU test
files, and the extension. Existing attention source, complete matmul source,
BF16/default kernels and the `kernels.cu.o` object are identical. Historical
artifact receipts are byte-identical. Runtime BF16 parity remains a lead GPU
test; source/object identity is not represented as runtime certification.

Binary SHA256:
`cbb11e8c5cc7c6ffba4e603183310e830d03df31ef1ef4390740d1c91166b3c5`.
Final source manifest SHA256:
`4743587808bd8aa54cb2f8141f693b7ef24eeef04f2b185a2df055972988b3b1`.

Slot-11 model receipts remain historical GREEN: onset registered cosine
0.999865554869887 / rel-L2 0.09613019724704594; healthy cosine
0.9995156202599452; control cosine 0.9991513145280014. The actual frozen bar
in those files is **0.9985468607788921**, not the brief's approximate 0.99843.
Step time is 5.105 to 5.66 seconds, ratio 1.1087169441723799. The three
sanitizer summaries have zero errors. No new model-level result is claimed.

## Lead's slot: at most 20 minutes

These GPU commands are provided, **not executed by this seat**. The lead
supplies the exclusive slot. The runner never acquires/touches the lock,
runs lanes sequentially, and signals only process groups it starts. The
1200-second deadline also covers optional model lanes.

```bash
cd /mnt/ForgeRealm/wt/pt-tf32
CUDA_VISIBLE_DEVICES='' python3 -B scripts/pt_tf32_4_slot.py \
  --print-sequence --out artifacts/pt_tf32_4/lead_slot_01
CUDA_VISIBLE_DEVICES=0 python3 -B scripts/pt_tf32_4_slot.py \
  --lead-gpu --out artifacts/pt_tf32_4/lead_slot_01
```

To include model completeness reruns, add `--include-model` to the **single**
GPU invocation above. If the output directory exists, use a new name; receipts
are create-only. The optional model lanes use the new engine registration
and a newly frozen noise floor, never the previous binary's floor.

| Sequential lane | Budget, seconds |
|---|---:|
| Dispatch self-check, all 27 descriptors/algorithms | 15 |
| Units, all 60 tests | 30 |
| GEMM, all 54 cells | 60 |
| Attention cases 0 / 1 / 2 / 3 | 35 / 35 / 90 / 90 |
| memcheck / racecheck / synccheck | 30 / 60 / 30 |
| Required lanes total | 475 |
| Optional noise / onset / healthy / control / step time | 120 / 60 / 60 / 60 / 260 |
| All lanes total / global deadline | 1035 / 1200 |

Verbatim sequence receipts:

```text
LEAD_SEQUENCE.json LANES 10 BUDGET_SECONDS 475 DEADLINE_SECONDS 1200 DEFAULT_DUMP_BYTES 0
LEAD_SEQUENCE_WITH_MODEL.json LANES 15 BUDGET_SECONDS 1035 DEADLINE_SECONDS 1200 DEFAULT_DUMP_BYTES 0
```

Both JSON files contain complete argv arrays. The inherited 8-GiB free-space
rail, exact pipe/RAM gradients, suppressed timing checkpoint saves, separate
sanitizer verdicts, `--keep-grads` opt-in and ENOSPC BLOCKED handling remain.
The default persistent gradient/checkpoint dump total is zero; small JSON,
text, token batches and CUDA caches are separately identified. At timeout,
unrun lanes receive BLOCKED receipts. No timeout or missing coverage passes.

[BLOCKED_REPORT.json](../artifacts/pt_tf32_4/BLOCKED_REPORT.json) is this
seat's certification result, pinned to the new binary/manifest/amendment.
**Not claimed fixed:** rebuilt-native edge passes, new GPU tail/dispatch
certification, runtime BF16 parity, blind verification or training stability.
The statistical and interval models are explicit assumptions, with native
validation pending; historical re-scoring is not a replacement for that slot.

## Prior art

- **Gumbel (1958), [Statistics of Extremes](https://doi.org/10.7312/gumb92958);
  David & Nagaraja (2003), [Order Statistics, chapter 4](https://doi.org/10.1002/0471722162.ch4).**
  Taken: Gaussian leading maximum scale; the two-sided union-bound derivation
  is displayed above. Publisher bibliographic pages verified; no claim of
  checking a numbered theorem inside inaccessible book text. Ours: explicit
  peak-scale assumption, 12-array risk allocation and registration placement.
- **NVIDIA TF32 (2020), [TF32 format](https://developer.nvidia.com/blog/accelerating-ai-training-with-tf32-tensor-cores/);
  [CUDA 12.6 guide](https://docs.nvidia.com/cuda/archive/12.6.2/cuda-c-programming-guide/index.html#intrinsic-functions),
  [PTX ISA 8.5](https://docs.nvidia.com/cuda/archive/12.6.2/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma-mma),
  [cuBLASLt](https://docs.nvidia.com/cuda/archive/12.6.2/cublas/index.html#cublasltmatmulalgocapgetattribute),
  all 2024.** Taken: format/conversion, exp error, accumulation limitations,
  descriptors/capability flags. Verified official pages and installed headers.
  Ours: replacing the invalid query, exact-shape receipt binding and dV
  coefficient intervals. No new GEMM selection algorithm was needed.
- **Higham (2002), [Accuracy and Stability of Numerical Algorithms](https://epubs.siam.org/doi/10.1137/1.9780898718027.fm).**
  Taken: unit-roundoff, gamma_n, absolute dot-product error/interval propagation.
  Ours: its placement around native saved-LSE and grouped dV accumulation.
- **Williams, Waterman & Patterson (2009), [Roofline](https://digital.library.unt.edu/ark:/67531/metadc934195/).**
  Taken: compute/bandwidth ceilings, FLOP/byte reasoning. Ours: these shape
  calculations; no measured DRAM-traffic or hardware-peak assertion.
- **PT-TF32-1/2/3, CC39/CC41, BP-KERNEL-2/3/4 (2026), NumPy (Harris et al.,
  2020), pytest, POSIX/Python subprocess, SHA256 (NIST, 2001).** Taken:
  fixture/reference, pipe transport, CPU/build and immutable receipt mechanisms.
  Ours: new versioned registration, short slot budgets, fail-closed assertions.
  Existing residual-product/attention methods remain attributed at their
  unchanged code sites to Ootomo & Yokota (2022) and Dao (2022/2023).
  No prior art known to me for the inherited user's exact four-arm cosine
  calibration rule; it is unchanged. No novelty claim for any underlying method.
