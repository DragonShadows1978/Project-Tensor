# BWD-REV4B re-check 2

**RED for closure. Seven original findings are fixed and six are partly fixed.** The three r5 planted edits are refused in the retained CPU receipt. A new CPU check on the real r5 bundle showed that the same candidate can go from 39/40 to 40/40 in G2 when a re-stamped merged oracle disagrees with its unchanged FD part. The G5 run passed its implemented estimate gate at **3,544,383,488 B**. The hard process-memory bound remains unverified. [Verifier output](commands/34-verifier-check.log) · [Receipt recount](commands/58-recount-execution.log)

I reviewed every hunk of both named diffs. R4 has 43 hunks in 11 files, including the two newly tracked launch/suite scripts. R5 has seven hunks in seven files. Evidence below is source reasoning, retained self-test and lead receipts, or the isolated CPU verifier call. The lead's r4 receipts contain 18 passing GPU lines and four passing reference steps. R5 contains three passing restaged GPU lines and four passing reference steps, with 15 r4 lines retained. I verified the 35 retained r4 pins, the 43 r5 pins and the launcher log SHA-256s. The retained G6 result is per-ID parity, with 100 expected changes against r4a and 512 against production in each round. [R4 diff](commands/05-r4-diff.log) · [R5 diff](commands/06-r5-diff.log) · [Recounts](commands/58-recount-execution.log)

## Original findings

The case names below point to the retained r4 and r5 CPU self-tests, the public test file, or the specified lead receipt. Where a finding concerns report arithmetic or coverage, the test exercises the corrected behaviour and the receipts identify the remaining boundary. [R4 cases and receipt](commands/25-r4-cases.log) · [R5 cases and receipt](commands/14-r5-selftest-source.log)

| ID | Status | Source | What the check establishes |
|---|---|---|---|
| F01 | **fixed** | [scripts/bwd_r4b_gpu.py:96,194–214,723–737](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L96) | The real digest functions are stubbed through backward and replay. The retained check counts five r3 calls and zero r4 calls. The G5 receipt also records zero. Check: `F01_digest_scope_covers_replays`. |
| F02 | **partly fixed** | [scripts/bwd_r4b_refwave.py:55–75,189–227](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_refwave.py#L55) · [scripts/bwd_r4b_verify.py:238–274,299–307](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_verify.py#L238) | Same-wave stale capture ID and operand drift cases refuse, but self-consistently re-stamped part provenance and contradictory reference metadata pass. Capture still resolves from newest passing receipt, with no check of its staged pins or worker operand record. Check: `F02_r4bm06_stale_bundles_refused; f02_cases_still_refused`. |
| F03 | **partly fixed** | [scripts/bwd_r4b_verify.py:70–167,278–294](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_verify.py#L70) · [scripts/bwd_r4b_gpu.py:376–386](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L376) | Exact probe keys, count, directions, finite slopes, numerical caps and linear sealed file bytes are checked. MFCC crossing observations and in-bound slopes remain supplied by an unpinned merged oracle. Check: `F03_g2_fails_closed; f03_cases_still_refused; r4bm07`. |
| F04 | **partly fixed** | [scripts/bwd_r4b_gpu.py:425–492,708–750](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L425) · [scripts/bwd_r4_gpu.py:29–71](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4_gpu.py#L29) | Separate 2 ms monotonic sampler and 50 ms maximum raw-poll gap close the observed 2.116 s stall. Missing PID polls still count toward gap/bracket checks, the median allowance is not gated and the formula does not bound physical peak. Check: `F04_sampler_coverage`. |
| F05 | **fixed** | [tc_signal/tc_signal/_conv_backward.py:153–161,196–222](../../../../tensor_fork_bwd_wt_r4b/tc_signal/tc_signal/_conv_backward.py#L153) · [tc_signal/src/bindings.cpp:339–370,662–667](../../../../tensor_fork_bwd_wt_r4b/tc_signal/src/bindings.cpp#L339) | The shared geometry and range code runs before tc.empty and the all-off return. A separate operand-validation gap remains for address and device provenance. Check: `F05_native_preflight_before_all_off; test_host_native_preflight_refuses_beyond_int_max; test_host_preflight_runs_before_needs_and_device`. |
| F06 | **fixed** | [scripts/bwd_r4b_gpu.py:389–410](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L389) · [scripts/bwd_r4b_gpu_r5.py:55](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu_r5.py#L55) | The residual is L2(candidate-reference at wave_c), plus a relative value divided by L2(reference at wave_c) when nonzero. The signed norm difference is labeled separately. Check: `F06_residual_is_norm_of_difference`. |
| F07 | **fixed** | [scripts/bwd_r4b_host.py:433–466,515–566](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_host.py#L433) · [scripts/bwd_r4b_refwave.py:178–184](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_refwave.py#L178) · [scripts/bwd_r4b_gpu_r5.py:55–69](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu_r5.py#L55) | Nonlinear MFCC is evaluated on the selected A2-1 probe endpoints. Both lead r5 schedules report three nonlinear slopes on all 40 matching keys. Round 1 is separately labeled. Check: `mfcc_paths_on_stand_in; retained stand-in and lead MFCC oracle and GPU receipt`. |
| F08 | **partly fixed** | [tc_signal/tests/test_bwd_r4b_public_conv_backward.py:303–385](../../../../tensor_fork_bwd_wt_r4b/tc_signal/tests/test_bwd_r4b_public_conv_backward.py#L303) · [scripts/bwd_r4b_gpu.py:852–894](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L852) | Tests cover the two missing masks, the dilation-dominant transpose case, and simulated noncurrent CUDA. The real mixed-device test skipped and constructs the cuda:1 operand using a factory that labels it cuda:0. Check: `pytest_path_selftest-r4bm05; G-API; r4bm05; r4bm06`. |
| F09 | **partly fixed** | [scripts/bwd_r4b_gpu.py:787–808,811–849](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L787) · [scripts/bwd_r4b_stage.py:261–267](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_stage.py#L261) | All completed public pytest children are bracketed by pinned 58-file censuses. A timeout unwinds before the after-child hash census and detailed pytest receipt. Check: `F09_fixture_census; G-API and mutant worker receipts`. |
| F10 | **partly fixed** | [tc_signal/tc_signal/_conv_backward.py:123–150,209–222](../../../../tensor_fork_bwd_wt_r4b/tc_signal/tc_signal/_conv_backward.py#L123) · [tensor_cuda/src/bindings.cpp:666–674,1289–1317](../../../../tensor_fork_bwd_wt_r4b/tensor_cuda/src/bindings.cpp#L666) | The current-device contract is documented and checked before tc.empty for correct labels. It relies on labels, which can disagree with allocation residency. Private native run with valid preallocated buffers can guard and restore a noncurrent operand device. Check: `test_host_current_device_rule; test_gpu_public_refuses_noncurrent_device_simulated; two-device case skipped`. |
| F11 | **fixed** | [scripts/bwd_r4b_gpu.py:285–319](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L285) | Structural zeros get fp32, finite, shape and absolute error checks. Exact zeros and tiny constants pass the retained check, and the excessive error fails. Zero norms outside the set are refused before relative comparison. Check: `F11_zero_rows_absolute_only`. |
| F12 | **fixed** | [scripts/bwd_r4b_host.py:365–388](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_host.py#L365) | The synthetic over-bar and malformed cases record a red receipt. IDs are read from row.probe. Check: `F12_red_receipt_written`. |
| F13 | **fixed** | [scripts/bwd_r4b_stage.py:33](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_stage.py#L33) · [scripts/bwd_r4b_suite.py:32–49](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_suite.py#L32) · `git ls-tree 4f4c6168` | Both pinned acceptance scripts are tracked. The screen has no hits and the current bytes match the staged pins. Check: `F13_trackability_screen; both paths tracked at 4f4c6168`. |

## Consistency rules

| Rule | General verdict | Evidence |
|---|---|---|
| Registered values | **Partly fixed** | The verifier derives ρ, the strength floor and the numeric bound from the sealed settings and reference. It checks the declared chosen triple and T(1) for nonzero crossings. It does not require an observation for every factor, verify the actual step sizes or reproduce the endpoint masks. The clean real copy passed. Empty crossing dictionaries passed, too. For `kmodel.decoder.encode.norm1.fc.weight`, coordinate 23423, changing the three slopes within the recomputed cap changed the result from 39/40 to 40/40. |
| Recomputed digests | **Partly fixed** | R5 recomputes named source/output/part byte hashes, both NPZ hashes and all direction hashes. The inconsistent-digest edit refuses, as does the re-stamped 2× direction in the retained self-test. For the MFCC bundle those file hashes are compared to the same bundle's manifest, and the manifest is compared to its own verdict. The sealed inventory already contains the independent r4 manifest SHA. The actual r4 and r5 manifests match it exactly. The verifier ignores that pin and accepts contradictory nested metadata and a stale, empty part after re-stamping. The live/reference recomputation also admitted a re-stamped NaN direction. |
| Exact inventory | **Holds for unique parsed names on the full worker path. Partly fixed in general.** | The pinned inventory enforces exact registered output, part, script, Voice and named operand entries. Omitted and renamed keys refuse in the retained test. An extra occurrence of an existing key was accepted because JSON parsing collapsed it. Crop and source-basis sequences have only registered lengths at the launcher, with values checked by the worker's operand equality. A part filename can remain while the part's rows disappear. |

The CPU check recomputed the real crop reference and all 16 directions from the pinned wave. It used the fp32-cast reference and a constructed coordinate as candidate proxies. The threshold witness used original slopes of about 0.013739650, replacement slopes of about 0.013739840 and a fixed candidate of 0.013753450. It changed only the merged oracle and its self-reported hashes, leaving the producer parts intact. This is a verifier counterexample. The lead's actual r4 and r5 merged receipts match their corresponding parts and both schedules passed. [Verifier output](commands/34-verifier-check.log) · [Manifest hashes](commands/59-r4-r5-content-links.log) · [Lead receipts](commands/58-recount-execution.log)

I recomputed 21/21 passing r4 CPU self-test entries and 6/6 passing r5 entries from the files. The r5 planted edit refusal counts were 7 for r4bm07, 2 for r4bm08 and 3 for r4bm09. I read the self-test sources and receipts in place. These author self-tests were not executed in the read-only round worktree. [R4 and r5 receipt counts](commands/58-recount-execution.log)

## G5 process memory and coverage

The NVML series and the R4a accounting rows give these independent recounts. The per-phase durations and counts are reported by the worker. Absolute phase marks are absent from the raw receipt, so the phase assignment and endpoint bracketing cannot be independently replayed. [G5 code](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L460) · [G5 recount](commands/58-recount-execution.log)

| Arm | Valid readings | Median / max gap | Largest NVML observation | Pool reserved high + largest observed nonpool | Binding estimate |
|---|---:|---:|---:|---:|---:|
| Checkpoint | 2,076 | 2.000 / 21.608 ms | 3,063,939,072 B | 2,147,483,648 + 1,396,899,840 B | 3,544,383,488 B |
| Plain, report only | 1,898 | 2.000 / 17.620 ms | 9,640,607,744 B | 8,724,152,320 + 1,396,899,840 B | 10,121,052,160 B |

Both traces had zero missing PID values and zero gaps over 50 ms. The checkpoint worker reports observations in every phase and zero real digest calls. This closes the old worker-GIL stall on the retained trace. In general a missing PID value has a timestamp and still advances both the max-gap and endpoint checks. For example, a dense run of empty polls between valid readings can span most of forward and still satisfy the test with a single valid sample in the phase. The 50 ms rule cannot replace the sealed 5 ms median requirement, and a query-start timestamp alone cannot establish where the reading lies relative to a phase transition. [Sampler source](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_nvml_sampler.py#L49) · [Coverage source](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L460) · [Sealed bars](../../../../tensor_fork_bwd_wt_r4b/artifacts/bwd_r4b/REGISTRATION_R4B.md#L34)

R4a adds a scoped default-pool reserved high-water to the largest of several synchronized, device-wide nonpool observations. The latter subtracts the current default-pool reservation from the device's reported usage. It may include memory attributed to other processes. A transient allocation outside that pool can occur between NVML polls and between device-wide boundaries. The larger estimate still misses that transient if both measures miss it. A hard bound requires coverage of all relevant pools and a bound on every process-attributed nonpool category across the interval. The checkpoint receipt establishes the sampled peak and the passing estimate for this run. [R4a accounting](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4_gpu.py#L29) · [G5 verdict](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L723)

## Public preflight and native parity

Shared `backward_geom` validates native attribute ranges, extent arithmetic, upstream shape and fp32 byte counts. Python calls it via `_conv_backward_scratch` before validating the mask and device and before the all-off return or any `tc.empty`. The retained F05 case refuses stride 2³¹ and 2⁶³ with all slots off. Appended host cases reach the same preflight with both active and all-off masks and exercise both geometries. This closes the original geometry finding. [Wrapper](../../../../tensor_fork_bwd_wt_r4b/tc_signal/tc_signal/_conv_backward.py#L153) · [Native planner](../../../../tensor_fork_bwd_wt_r4b/tc_signal/src/bindings.cpp#L339) · [Host tests](../../../../tensor_fork_bwd_wt_r4b/tc_signal/tests/test_bwd_r4b_public_conv_backward.py#L277)

Full operand parity is wider than geometry. The native run validates DLPack pointer, alignment and device information as well as destination and scratch. The public current-device check runs before allocation using the three TensorCUDA labels and `cudaGetDevice`. In the supported, correctly labeled case, mixed labels and a noncurrent label refuse early, including with all slots off. The private path can run valid preallocated operands on a noncurrent device through a guard, so the public contract is intentionally tighter there. Conversely, an inconsistent label or a misaligned imported pointer can pass public validation first. Active masks allocate before that mismatch is detected at export or native consumption. The all-off mask never reaches either check. Source traces show the two-GPU test uses the labeling-unsafe constructor, and the retained G-API run skipped that case. The label and alignment conclusions are code-path reasoning. [Public device check](../../../../tensor_fork_bwd_wt_r4b/tc_signal/tc_signal/_conv_backward.py#L123) · [Native operand checks](../../../../tensor_fork_bwd_wt_r4b/tc_signal/src/dlpack_operand.cpp#L440) · [Native device guard](../../../../tensor_fork_bwd_wt_r4b/tc_signal/src/conv_backward.cu#L245) · [Constructor and allocation trace](commands/54-allocation-label-path.log)

## Reference and probe read paths

| Path | Reads and verification |
|---|---|
| Staged r5 `g1-linear` | `bwd_r4b_gpu_r5.py:23,28` calls `verify.linear_pins` and `g2_registered`. `verify.py:170–177` delegates to the whole-file-pinned round-1 loader before returning the per-direction hashes. |
| Staged r5 `g1-mfcc` | `bwd_r4b_stage_r5.py:152–160` calls the light verifier before the lease. `bwd_r4b_gpu_r5.py:41,53–54` uses the full `bundle_r5` and registered scoring. Report-only round-1 probe reads use the sealed round-1 loader at line 44, and the A1-2/A2-2 fields are read at lines 45–67 from the already loaded record and oracle. These fields inherit the manifest trust gap. |
| Staged r5 `r4bm03` | Same pre-lease and full bundle paths. `bwd_r4b_gpu_r5.py:80–93` does G1 and checks that the edit executes once. It consumes the reference and does not score the oracle. |
| r5 `ref-at-wave` | `bwd_r4b_refwave_r5.py:11–20` delegates to r4 producer code after checking all r5 stage pins on a real invocation. `bwd_r4b_refwave.py:130–136` loads round-1 oracle/directions for the report. FD reads its reference record and operands at lines 156–179, while merge reads that record, directions, reference archive and parts at lines 189–216. These checks are the producer's own, with full staged file pins for the round-1 inputs. |
| Superseded r4 entry points | `bwd_r4b_stage.py:208–235,261–271` and `bwd_r4b_gpu.py:414–420,601–657,755–760` retain the old loaders and scorer. Those labels still dispatch if the old launcher is invoked directly. R4b G0, other G1 mutants, G3, G4 and G5 do not use the superseded MFCC bundle or G2 oracle. |
| Offline creation, reports and self-tests | `bwd_r4b_host.py:312–313,606–635` reads old probe data for report/merge. `bwd_r4b_register.py:39–40` reads summaries. `bwd_r4b_verify.py:200–212` reads manifests when creating an inventory. R4 self-test reads the old bundle through its old reader. R5 self-test reads raw copied fixtures for mutation setup and checks its cases through the r5 verifier. Old self-tests and registration are historical tooling. |

All three documented restaged GPU paths enter the r5 verifier. The literal rule that every read uses that verifier has exceptions in the reference producer, offline reports and still-callable r4 launcher. R5's stage pins protect the old round-1 files read during actual r5 production. Internal FD part and report joins need the stronger checks above. [Read census](commands/19-consumers.log) · [R5 worker and launcher](commands/15-r5-entry.log) · [Producer](commands/20-refwave.log) · [Other readers](commands/56-census-details.log)

## New findings

### BWDR4BR5-F01 · MAJOR

The MFCC oracle can change a G2 verdict without matching its FD parts. Empty crossing maps pass. An alternative finite slope within the independent numeric cap changed one constructed candidate's result from 39/40 to 40/40 after the oracle, manifest and verdict were re-stamped, with the FD parts untouched. [scripts/bwd_r4b_verify.py:98–125,259–284](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_verify.py#L98) · [scripts/bwd_r4b_gpu_r5.py:53–54](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu_r5.py#L53)

**Fix.** Bind the capture and manifest to an independent registered digest. Parse and validate part identities and stitch the checked parts, then compare the full merged oracle. For reusable generation without a fixed pin, verify actual endpoint observations, the step schedule and complete factor-keyed crossing records before scoring.

Evidence recorded in `commands/34-verifier-check.log (CROSSINGS_OMITTED, MERGED_ORACLE_CHANGED)`.

### BWDR4BR5-F02 · MAJOR

The verifier compares manifest, outputs, parts and sources against digests supplied by the same bundle, ignores the manifest digest in the sealed inventory's derived_from record, and leaves the capture's worker operand record and certain embedded digests unjoined. A wrong cotangent digest, wrong stored wave array digest, bogus crop and empty FD part from another capture passed together. [scripts/bwd_r4b_verify.py:194–197,229–274,299–307](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_verify.py#L194) · [scripts/bwd_r4b_refwave.py:55–75](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_refwave.py#L55)

**Fix.** Use the independently sealed manifest SHA already present for this capture. Register a new capture ID and manifest digest when inputs change. Validate the capture's pins and worker operand record and bind embedded metadata/digests to the live wave, cotangent and operands.

Evidence recorded in `commands/34-verifier-check.log (STALE_PART_AND_RECORD); commands/59-r4-r5-content-links.log (actual r4/r5 manifest SHA)`.

### BWDR4BR5-F03 · MINOR

A re-stamped NaN direction passes the fresh-product comparison because a NaN difference is not greater than the tolerance. A single NaN also poisons a tensor reduction and arrays are not required to have the fresh sample's shape/dtype before subtraction. The G2 strength checks provide a further barrier, but the bundle verifier already returned it, and r4bm03 consumes only the gradient. [scripts/bwd_r4b_verify.py:275–296](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_verify.py#L275)

**Fix.** Validate exact shape, dtype and finiteness per stored tensor and direction, then require each finite difference norm to be at or below its bound.

Evidence recorded in `commands/34-verifier-check.log (NAN_DIRECTION)`.

### BWDR4BR5-F04 · MINOR

An additional occurrence of an inventory key passes because JSON decoding collapses duplicates. The launcher also checks crop and source bases by length only and accepts changed crop values at the same count. The worker's live operand equality rejects that crop change. Normal missing/extra/renamed parsed inventory keys are covered. [scripts/bwd_r4b_verify.py:181–197,230–251](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_verify.py#L181) · [scripts/bwd_r4b_stage_r5.py:152–160](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_stage_r5.py#L152)

**Fix.** Use strict duplicate-free parsing and an explicit schema. Verify the fixed crop and ordered source bases in preflight or describe that admission check as structural only.

Evidence recorded in `commands/34-verifier-check.log (COUNT_ONLY_OPERAND, DUPLICATE_INVENTORY_NAME)`.

### BWDR4BR5-F05 · MAJOR

Coverage can pass across a missing-PID interval because raw polling timestamps drive max gap and endpoint checks. The original 5 ms median bar is absent, and absolute phase marks are not persisted. Default-pool high-water plus the maximum observed device-wide nonpool sample cannot bound the unobserved process peak. [scripts/bwd_r4b_gpu.py:460–492,727–747](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L460) · [scripts/bwd_r4_gpu.py:29–71](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4_gpu.py#L29) · [scripts/bwd_r4b_nvml_sampler.py:49–61,93–96](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_nvml_sampler.py#L49)

**Fix.** Require PID-valid endpoint brackets and gaps, restore the registered median check, persist phase marks and query intervals, and use that result as a sampled estimate. Claim a hard cap only with scoped high-water accounting or a proven upper bound for all process pools, native/driver and nonpool memory across the interval.

Evidence recorded in `source reasoning; commands/17-memory.log; commands/31-memory-source.log; commands/58-recount-execution.log`.

### BWDR4BR5-F06 · MINOR

Public preflight checks geometry and labels before allocations, but it does not check operand address alignment or actual residency. TensorCUDA can wrap an external DLPack allocation after checking only byte_offset alignment. Native consumption then rejects a misaligned effective pointer, or export refuses a storage/label mismatch; active masks allocate first and all-off can accept. [tc_signal/tc_signal/_conv_backward.py:136–161,209–222](../../../../tensor_fork_bwd_wt_r4b/tc_signal/tc_signal/_conv_backward.py#L136) · [tc_signal/src/bindings.cpp:353–379](../../../../tensor_fork_bwd_wt_r4b/tc_signal/src/bindings.cpp#L353) · [tc_signal/src/dlpack_operand.cpp:500–507](../../../../tensor_fork_bwd_wt_r4b/tc_signal/src/dlpack_operand.cpp#L500) · [tensor_cuda/src/bindings.cpp:666–674,1230–1269](../../../../tensor_fork_bwd_wt_r4b/tensor_cuda/src/bindings.cpp#L666)

**Fix.** Expose a no-device-allocation native operand check using the same DLPack/device and pointer validation before public allocation and all-off. Keep destination/scratch validation in the run entry.

Evidence recorded in `source reasoning; commands/21-api-focus.log; commands/23-native-detail.log; commands/51-dlpack-alignment.log; commands/54-allocation-label-path.log`.

### BWDR4BR5-F07 · MINOR

The mixed-device test creates w1 with tc.tensor after a device switch. The constructor assigns cuda:0 metadata while CUDA allocates on the current device. After restoring 0, the test cannot exercise the wrapper's mixed-label refusal and may hit the later exporter failure. The retained qualification skipped the test. [tc_signal/tests/test_bwd_r4b_public_conv_backward.py:370–385](../../../../tensor_fork_bwd_wt_r4b/tc_signal/tests/test_bwd_r4b_public_conv_backward.py#L370) · [tensor_cuda/src/bindings.cpp:87–96](../../../../tensor_fork_bwd_wt_r4b/tensor_cuda/src/bindings.cpp#L87) · [tensor_cuda/src/kernels.cu:60–67,174–180](../../../../tensor_fork_bwd_wt_r4b/tensor_cuda/src/kernels.cu#L60)

**Fix.** Create the negative-test operand with tc.empty on the correct current device, assert its label, check cudaSetDevice status and restore the prior device.

Evidence recorded in `source reasoning and retained G-API one-device skip; commands/53-tensor-device-construction.log; commands/54-allocation-label-path.log; commands/58-recount-execution.log`.

### BWDR4BR5-F08 · MINOR

The three documented r5 workers use the new verifier. The old r4 launcher/dispatch still accepts those affected line labels with the legacy checks. Producer steps and earlier report helpers also read these files through legacy checks. The staged r5 producer wrapper pins the round-1 input bytes, so this is a coverage and supersession gap rather than evidence of an unverified lead read. [scripts/bwd_r4b_stage.py:43–67,208–235,261–271](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_stage.py#L43) · [scripts/bwd_r4b_gpu.py:414–420,601–657,755–760](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L414) · [scripts/bwd_r4b_refwave.py:130–136](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_refwave.py#L130) · [scripts/bwd_r4b_refwave_r5.py:11–20](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_refwave_r5.py#L11)

**Fix.** Enforce superseded labels in the active admission path or route them to r5, preserve historical replay explicitly, and use shared verified loaders for probe and part reads. Keep the seal checks on producer inputs.

Evidence recorded in `source reasoning; commands/19-consumers.log; commands/47-stage-review.log; commands/56-census-details.log`.

### BWDR4BR5-F09 · MINOR

A subprocess timeout raises at pytest_run before the second fixture SHA census and pytest record. The worker's generic mtime audit still runs, but the child after-hash verification is absent on that exit path. [scripts/bwd_r4b_gpu.py:817–846](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_gpu.py#L817) · [scripts/bwd_r4b_stage.py:261–267](../../../../tensor_fork_bwd_wt_r4b/scripts/bwd_r4b_stage.py#L261)

**Fix.** Run the after-census in a finally path and retain a red timeout or fixture-change receipt with the available child output.

Evidence recorded in `source reasoning; commands/35-r4-caller-and-ui-gates.log`.


## Deviations and receipts

The round's retained receipts include five r4 and two r5 launcher attempts with exit 76 and no worker start. The passing counts above use the actual worker verdicts and verified launcher log hashes. This re-check ran no GPU lines, builds or Voice tests. It retained one scratch directory at `/tmp/bwd_rev4b_r5_bundle_b_8zadi_`. The Voice audit of the targeted CPU check recorded 86 read paths, with zero changes or write attempts. ONNX Runtime reported `Failed to persist telemetry device ID; using an in-memory identifier`. A few read-only path guesses missed. The shell rejected one attempt to assemble command receipts with `Argument list too long (os error 7)` before execution. I split the reads and preserved their outputs. Neither CPU check needed a post-run repair. [Ledger](LEDGER.md) · [Commands and full outputs](COMMANDS.md) · [Verifier output](commands/34-verifier-check.log) · [Receipt recount](commands/58-recount-execution.log)

## Prior art

- DeMillo, Lipton and Sayward (1978), mutation testing. I used a clean control followed by edits to real registered artifacts. The changed oracle and the candidate coordinate demonstrating a verdict change are specific to this review.
- SHA-256 (NIST FIPS 180-4, 2015) and independently pinned build inputs (Dolstra, Nix, 2006). The proposed adaptation reuses the already sealed manifest pin for this capture and checks the produced FD parts against it. The file and metadata joins come from R4b.
- Higham (2002) for finite-difference and floating-point error analysis, and IEEE 754-2019 for unordered NaN comparisons. The proposed checks apply these to the registered finite-difference and fresh-product comparisons.
- NVIDIA CUDA and NVML API documentation supplies pool high-water and per-process query semantics. The coverage argument and requested scope for nonpool accounting are specific to G5.
- DLPack (dmlc, 2017 onward) supplies tensor layout, address and device metadata. Reusing the native DLPack validation before public allocation applies that contract to the mask path.
- No prior art known to me for this harness's specific timeout receipt arrangement. Unverified — lead to check “DeMillo Lipton Sayward 1978 mutation testing”, “Dolstra Nix 2006”, “NIST FIPS 180-4 SHA-256”, “Higham 2002 finite difference”, “IEEE 754 NaN comparison”, “NVIDIA CUDA ReservedMemHigh NVML process usedGpuMemory”, and “DLPack effective pointer alignment”.

[JSON](recheck.json) · [SHA-256 and UTC date](recheck.sha256) · [Execution registration](REGISTRATION.md) · [Check amendment](CHECK_REGISTRATION.md)
