# APA-P1-REV2 — RED before promotion

Two MAJOR, four MINOR and two NIT findings require disposition. One blind finding (F03) was withdrawn after checking the explicit row-classification contract. No BLOCKER is claimed. GPU reproductions are staged only.

## Blindness, scope and checks

The immutable user order is the plan; REGISTRATION.md records execution constraints. The first findings snapshot was sealed before prior orders/reports/known items: findings.blind.v1.json sha256 39dbf435e8dc127d224d03e4973c746ad0784d6799a468ae250f3d2e414ce2c1. LEDGER.jsonl records the seal before receipt 017. The mechanism documents were deliberately deferred to respect the explicit blind-first instruction; after reading them, no scope mismatch was found. The revision premises match: 17 files, +1,927/-92 and the four named commits. No subagents, builds, GPU execution, git writes or deletes. Commands ran with empty CUDA_VISIBLE_DEVICES, nice 19, single-thread variables, TMPDIR scratch and explicit timeouts <=1200s. nvcc and C++ were used for preprocessing only, with no object/module build.

findings.json sha256: `781131d78e024aa8643a301f47adb49db254c7d63e8686cb1d60506207b6fbf5`.

## Findings (verbatim)

P1R2-F01 — MAJOR — apa_packed.py:390. The packed margin pin is checked only by the default loader. Public construction and the expected_sha256 override allow unpinned entries to reach lookup/attention; an unpinned zero margin was accepted on CPU. Fix: enforce registered immutable provenance and entries at use, and separate unchecked experimental construction.

Evidence: receipts/016_blind_host.log; apa_packed.py:354-395,404-419. Prior art: SHA-256 (NIST FIPS 180-4, 2015) for pinning; no prior art known to me for this exact API repair.

P1R2-F02 — MAJOR — packed.cuh:95. With D64, q=ones and an infinite key coordinate, TQ4 can produce bulk=-Inf and skip the key, returning a finite row where dense SP returns NaN. The host construction has 33 positive and 31 negative infinite dense components. Fix: carry non-finite TQ4 keys as a NaN stored norm. Independently found; matches P1F1-INF. I disagree with its MINOR rating.

Evidence: receipts/016_blind_host.log; packed.cuh:14-23,89-99,148-151,178-188; quant.py:122-127. Prior art: IEEE 754-2008 NaN propagation; mirrors the F1 non-finite sentinel in its INT4 packer (Project-Tensor, 2026), no new algorithm.

P1R2-F04 — MINOR — apa_packed.py:225. Capacity has no upper bound before native int narrowing. At capacity=2**31, a second KV head can index before codes/norms; the D64 two-head case requires 144 GiB, so smaller devices fail allocation first. Fix: reject unsupported capacities/products before allocation or widen the entire indexing chain.

Evidence: receipts/016_blind_host.log; apa_packed.py:223-239; packed_launcher.cuh:8-12,34-43; launcher.cuh:184-191. Prior art: ISO C++ integer conversion/range rules (C++17, 2017); no prior art known to me for the exact boundary repair.

P1R2-F05 — MINOR — eb3.cuh:935 and packed_launcher.cuh:14-45. Registered EB3 timeline calls with G=3/5/6/7/8 refuse after rotation and bulk. Raw rotate/prepare/append also omit positive leading-extent checks and can attempt a zero-grid launch. Fix: validate these combinations and extents before device work.

Evidence: eb3.cuh:931-935,1001-1032,1053-1069,1083; packed_launcher.cuh:14-30,33-45; BOUNDARY_CENSUS.md; repro/r03_empty_helpers.py. Prior art: Ordinary precondition validation; no prior art known to me.

P1R2-F06 — MINOR — merge.cuh:27. Finite inputs can overflow scores to Inf and make den=NaN: B returns zeros while F returns NaNs. NaN/+Inf sinks also make F non-finite where dense returns zeros. Fix: explicitly define these behavior changes and narrow the finite-identity claim, or implement the required parity; retain q/k/v NaN propagation.

Evidence: receipts/039_numeric_boundaries.log; merge.cuh:14-30; kernels.cu:2172-2194; eb3.cuh:781-791; eb4.cuh:644-653; repro/r02_finite_overflow.py; receipts/050_nonfinite_extremes_script.log; repro/r01_inf_key_default.py. Prior art: IEEE 754-2008 overflow and NaN arithmetic; Higham (2002) finite-precision analysis. No new algorithm.

P1R2-F07 — NIT — ops.h:328. The old 14-argument EB2b overload remains declared without a definition; Python works through a global bridge. Fix: update the shared declaration and bind the current tc function directly. Known F1-D3; fix once and close.

Evidence: receipts/038_ops_cmake_tests.log; eb2b.cuh:138-141,221-228; bindings.cpp:2807-2816. Prior art: ISO C++17 (2017) external linkage and declarations; ordinary API plumbing, no new algorithm.

P1R2-F08 — NIT — scripts/apa_p1f1_sass.py:118. The SASS receipt pins the original expectation rather than the active override. Its 17 suffix renames comprise 14 paired renames plus three additions. Fix: record both expectation hashes and separate those counts; preserve original G7 FAIL. The pin issue is known F1-RCPT.

Evidence: receipts/023_sass_audit.log; receipts/041_regression_recompute.log; scripts/apa_p1d_sass.py:60-71; scripts/apa_p1f1_sass.py:57-64,118. Prior art: SHA-256 (NIST FIPS 180-4, 2015); ordinary provenance bookkeeping, no new algorithm.

P1R2-F09 — MINOR — apa_sp2.py:85. The default factory reserves capacity=S, so its result cannot append another token. Direct construction with capacity=N works. Fix: forward a validated capacity keyword. Known P1Q-API; not independently recorded before the known-item list.

Evidence: apa_sp2.py:61,85; apa_packed.py:223-225,305-307; receipts/045_public_route_census.log. Prior art: Capacity reservation in standard containers (std::vector::reserve, ISO C++ 1998) and inherited ApaSpq1Workspace (Project-Tensor 2026); no new algorithm.

### Withdrawn blind finding

P1R2-F03 initially reported V-sentinel column spreading as MAJOR. After reading F1 R1, it is withdrawn: that rule explicitly prescribes row-scale carrying and gates row sets, not matching finite columns. The behavior remains reported in Q3. The blind snapshot is retained unchanged; no retrospective rewriting.

## Answers to all ten questions

### Q1. Default switch

With storage=None, the environment selects dense/packed if set; otherwise the shipped choice is packed. Eligible CUDA fp32 4-bit D64/D128 K becomes PackedSpKeys, packed_prepare/append creates TQ4 codes and norms, and attention uses spq1 rotate/bulk (bulk_shared at G4/8), spo2/spo4 ordered selection, and spo2 merge under the packed table. Two-bit or missing-backend defaults fall back to dense with a reason; explicit ineligible packed raises. Dense opt-out keeps the dense quantizer and native epsilon/split-K path/table. The dense wrapper body is byte-identical (574 bytes). Old four-bit callers now receive an owned object, need packed margins, get a prefill refusal, and raw dense entries reject that object. Diagnostics still return output/mask. Identity/generation/revision and bounded append checks work under the documented unchanged-prefix contract, but the factory cannot reserve growth (F09), native capacity can narrow (F04), and the table pin is bypassable at use (F01). On identical bulk scores, margins and partition width, no default packed selection rule skips a key selected by dense: both use inclusive prefix max and >=. L=1 causal exposes all S keys; sinks do not enter selection. Read apa_sp2.py:61-136, apa_packed.py:88-146,215-437, packed_launcher.cuh:14-55, launcher.cuh:169-245, parallel.cuh:90-106 and apa_sp1_1.cuh:56-73; ran receipts 016,036,039,045.

### Q2. Boundary census

The complete guarded/impossible/unguarded table is in BOUNDARY_CENSUS.md and reproduced below. The source census has 43 vector-cast lines: 21 uint4, 8 uint2, 11 float4, three float2, zero int4/int2; this includes stores and two mutant branches. All caller vector operands have matching host guards; 17 internal sites use aligned allocations/views. The supplied alignment census has 32 rows and LF3 has 38 rows, including all 32 IDs and six acceptance controls. Noncontiguous TensorCUDA views are impossible through its API. Capacity narrowing and extreme index products remain unguarded (F04); raw empty helpers and EB3 timeline can reach device work before refusing (F05). General scalar alignment of an arbitrary external DLPack base is also an inherited unguarded precondition; checking byte_offset alone does not establish base alignment. The table states baseline behavior for every refusal class. Read all casts and launchers, core.h:103-138 and bindings.cpp:1177-1212; ran receipts 009-015,034,038,045-047 and retained LF3/4 checks.

### Q3. Non-finite values

Full dense parity does not hold. NaN q/k/v row propagation is repaired at all three named merges; the retained LF2 NaN cases agree in row sets. TQ4 +/-Inf keys can yield -Inf bulk and finite output where dense is non-finite: r2_tq4, tc_tq4, EB2b, EB3 and EB4 (F02; public fp32 r2 included). INT4 r2/tc instead carry NaN scales and can be non-finite where dense is finite. Constructed +/-Inf queries with opposite-sign finite keys give dense zero versus packed NaN; the ordered D1/r2 and TC/EB normalizers likewise can differ on all -Inf logits. NaN/+Inf sinks add another dense-zero/packed-NaN case (F06; EB2b has no sink argument). Zero-norm keys stay finite: TQ4 norm is zero, INT4 zero codes use safe scale one. The V quantizer intentionally poisons an entire low-precision token, so finite column sets differ although the row remains non-finite; blind F03 was withdrawn under F1's explicit row contract. Read packed.cuh:14-23,36-99,148-188, eb2b.cuh:8-27, eb2b_values.cuh:12-48, merge.cuh:14-30, kernels.cu:2172-2195; ran host receipts 016,039,048/050 and audited LF2/LF2S in 031-032. Negative-Inf and value-Inf device cases are staged, not run; exact receipt row differences are below.

### Q4. Tickets, hooks and streams

Per-device ownership and normal reset are sound under the single legacy-default-stream contract: CheckedDeviceGuard selects q's device, the recursive mutex covers growth/enqueue, new slots are zeroed, and the last atomicInc wraps at NC-1. A rejected launch that starts no CTA takes no ticket; a sticky execution fault does not promise context recovery. The hook timeout synchronizes then resets merge slots before throwing; both EB3/EB4 pass that pointer/count. The choose kernel has completed beforehand, or choose tickets are absent in default CHOOSE_SITE=1. Retained FH LF5 has 11 successful checks; its same-process next-output bytes are not retained, so that particular fresh-process equality is receipt-only. Build F has 99 dynamic exports, zero hook-symbol matches, no device EarlyCounterHook signatures, 13 real spo2 entries and no hook/timeout strings. FH has 34 device hook signatures and two hook methods among 15 entries. Ordinary nm reports stripped host symbols, so it cannot establish hidden bridge symbols. All launches use stream 0; Python exposes no alternate stream. Unsupported concurrent external/non-default streams may race; they are out of contract. Actual nvcc -E --default-stream per-thread fails at the new #error; legacy preprocessing succeeds. Read eb3.cuh:91-164,568-607,811-869,944-947,1102 and eb4.cuh:207-218,719-720,824; ran 024,027,028,033,035.

### Q5. Linkage and loading

The global block-scope bridge declarations are valid C++ external declarations in the enclosing global namespace. Their types match the single definitions reached through kernels.cu's include chain; no duplicate definition or competing name was found. The hooks-only bridge is conditional on the same macro. The stale 14-argument tc::apa_eb2b_run declaration in ops.h:328 has no definition (F07); the other in-scope APA declarations have implementations. RTLD_NOLOAD|LAZY probes the already-loaded _C library; Linux fallback dlopen and symbol lookup are valid here. Packed initializer errors propagate; nonpacked builds return None and public packed stubs/workspace refuse before allocation. Real packed, nonpacked and simulated-failure CPU tests passed. Read bindings.cpp:2741-2816, eb2b.cuh:213-228, eb3.cuh:1139-1147, ops.h:308-348 and __init__.py:1578-1610,1758-1773; ran 024-025,033,038,045. Successful binary loading supports the linkage; stripped nm does not independently expose the hidden bridge.

### Q6. Code generation

Independent streaming comparison of sha-checked B/F dumps found B=778 functions, F=781: 750 equal, 28 changed, zero removed, three added, exactly the stored substantive counts. The changed groups are 18 EB3 fused selects, six EB4 mixed selects, three spo2 merges and one EB2b quantizer; additions are three spq1::pack_int4_rows types. Only nvcc's _cu_<8hex> with optional _<n> was mapped, including any call targets; names stayed bijective, instructions/encodings/control words were preserved. Fourteen paired names changed suffix; the stored 17 includes three additions (F08). The six retained spq6 bulk variants (D64/D128 x V0/1/3) cannot be reached through a supported default-build dispatch: var is fixed to 2 and non-default environment values refuse before rotation; all other launch sites explicitly use V2. Their presence is not itself a product defect. Original G7 still FAILS because its six registered removals did not occur; the attempt-specific override passes a different predicate. Read spq6.cuh:222-276, CMakeLists.txt:77-81 and every bulk_mma_tq launch; ran 020,023,024,041,045. See SASS_AUDIT.json for every exact name/body hash.

### Q7. Determinism and numerics

Source reasoning supports bitwise repeatability for valid, unchanged inputs on one device under the default-stream contract: output writes are disjoint; integer tickets select the last CTA but do not set summation order; merges visit chunks in fixed ascending order; no floating-point atomics are used. This CPU-only review did not measure run-to-run GPU output. I independently inventoried 6,922 retained B/F LF1 arrays and, after excluding one combined timing/byte-census structure, rechecked 6,921 using dtype/shape checks, informative same-width unsigned views, and np.array_equal: all equal. That is finite-case evidence, not universal determinism. The receipt uses one build per side and the engine's default logical CUDA device; it does not record a physical GPU UUID/model, so that identity cannot be independently reconstructed. For finite nonnegative denominators, >0 and !=0 give identical results, including +/-0 behavior; finite operands alone can overflow and change B's finite zero into F's NaN (F06). Read merge.cuh:14-30 and EB3/4 fused merge chains/tickets; ran 023,028,035,037,039,041. No model-quality or hardware-portability claim follows.

### Q8. Tests and evidence

Both requested CPU suites pass: 40 policy IDs and 15 F1 IDs (55 total), recomputed from the verbose log. They cover policy/forwarding, identity, loader and macro checks, mainly with fakes. Existing GPU receipts cover registered finite identity, 32 vector-refusal IDs plus six controls, 63 refusal/knob checks and 11 FH timeout/knob checks. I recomputed 6,921 array identities from 6,922 retained arrays (one timing structure excluded), 44 regression files/2,078 IDs and 729 CPU regression IDs; outcomes match their before arms, including 36 GPU failures/15 errors and 125 CPU failures/15 errors, not an all-green suite. LF2 remains FAIL: 246/260 checks true, including two failed L19 controls and 12 Inf predictions; supplementary LF2S has 17/26 true and is PASS only because its nine Inf misses are predictions. Missing coverage includes pin bypass, reserved factory growth and large capacities, raw empty helpers, EB3 timeline at G3/5/6/7/8, -Inf and V Inf, all -Inf logits, finite overflow/non-finite sinks, and column-level non-finite behavior. These lines do not re-establish ticket growth, multiple devices, alternate streams or repeatability; no physical-device UUID is recorded. Read both entire CPU test files and scripts/apa_p1f1_gpu.py LF1-LF6; ran 021-022,025,029-032,035-037,041,043. Full gap matrix and all exit/log hashes follow.

### Q9. REV1 completeness

F01: all three named NaN guards and INT4/V carries are implemented; infinite-key parity remains incomplete (F02), and finite-overflow/sink parity is narrower than claimed (F06). F02: caller vector alignment fixes are complete across the source census; G=1 safety refusals are acceptable compatibility limits. F03: Python and binding EB3/EB4 defaults are partition. F04: EB2b delta/carry are required and forwarded/validated at every layer; header remains F07 NIT. F05: timeout reset is complete at both hook sites. F06: int4_stats rank/dtype/device/divisor checks are complete at the named diagnostic. F07: default-build A/B dispatch is disabled/refused; hooks retain controls; dead variants remain unreachable. F08: workspace wrappers share exact object/generation/revision checks; real append still assumes unchanged prefix by contract. F09: packed/nonpacked/failed-load handling is fixed. F10: dead first wrapper is gone. F11: named attention checks moved before rotation, empty attention extents and cursor equality are fixed, FP32 tensor dots now refuse; helper empties and timeline ordering remain incomplete (F05). F12: actual per-thread nvcc preprocessing is refused. Read REV1 findings only after blind sealing, then all named source sites; evidence is 018,025,027-028,034,038,043,045,049 and the detailed table below.

### Q10. Other inherited behavior

No additional BLOCKER was found. The default factory's append limit (F09), stale native declaration (F07), and evidence provenance/count NIT (F08) need disposition. Packed prefill refusal is the registered temporary limitation, not an accidental fallback; G=1 misaligned K/V refusal is a safe implementation limit. General ambient-device assumptions and scalar alignment of arbitrary external DLPack bases are inherited engine preconditions, not fixed here. The original SASS removal gate remains failed and is not relabeled as passing. No product source was edited, no build or GPU run was performed, and no file was deleted. Read all 17 diff files and their called mechanisms; 044 confirms an empty product working diff. Pending lead GPU reproductions are supportive checks, not substitutes for the source arguments.

## Default route matrix

| storage=None context | Built Kq | Attention path | Margin table |
|---|---|---|---|
| env unset; eligible CUDA fp32, 4-bit D64/128 | PackedSpKeys/TQ4 codes+norms | packed r2: scalar rotate/bulk; shared bulk at G4/8; ordered spo2 select at staged G1 or spo4 head select otherwise; spo2 merge | PackedRegisteredMargins |
| env=packed, eligible | same | same | packed |
| env=dense | dense fp32 TurboQuant reconstruction | native epsilon -> dense SP split-K at L1, dense prefill otherwise | RegisteredMargins |
| env unset/packed, 2-bit or missing backend | dense with recorded fallback reason | dense | dense |
| invalid env | none; ValueError | no launch | none |
| raw _C dense/d1 or explicit EB/SPQ entries | caller-supplied representation; no storage argument | explicit entry's kernel; no auto-policy conversion | direct delta, no registered-table governance |

Factory signature remains compatible for old two positional arguments; storage is keyword-only. Return type changes on eligible default calls; explicit storage=dense restores the old representation and dense margins. Default owned memory per key is 36 B at D64 and 68 B at D128, plus fixed tables (code width D/2+4 norm bytes; arithmetic, not a new residency measurement). Default packed margins are 1.010646104812622 (D64), 0.8716110587120056 (D128), and the frozen file's sha matches 1897fedd2a8d05945e34433c75a7afd3b33feb64e0a6b2ca410d611d18868e98. This is finite calibration, not a universal error bound.

## Full precondition census

# Boundary census

Evidence class: source reasoning, with executed CPU tests and retained GPU receipts identified below. G = guarded; I = impossible through the Python Tensor API; U = unguarded. Baseline means e2fb85a1, not Build B. File references without a prefix are in tensor_cuda/src/apa_spo/. Every item below was read; this is not a claim of a GPU boundary run.

## Shared representation and device preconditions

All native operands are NDArray/Tensor handles over one contiguous allocation plus an element offset (include/tc/core.h:103-138). Noncontiguous views are I: TensorCUDA represents no strides; DLPack imports must be contiguous (bindings.cpp importer). Arbitrary element-aligned offsets are possible, so vector alignment is a real precondition. Natural scalar alignment is G for own cudaMalloc/Async allocations. It is U for an arbitrary external base: bindings.cpp:1192-1197 checks byte_offset%itemsize, but not dl.data%itemsize. The bundled DLPack header:222-231 explicitly warns not to rely on base alignment. Normal typed producers satisfy scalar alignment; an odd external base remains an inherited producer/engine boundary affecting dense as well as packed. This is not an additional regression introduced by the reviewed diff. A raw C++ caller that manufactures an invalid NDArray is outside the Python ABI contract.

Matching CUDA device and dtype are G where listed below. Selecting the current CUDA device is a different issue: EB3/EB4 guard q.device with CheckedDeviceGuard; other inherited launchers rely on the engine's ambient current-device convention (kernels.cu:174-180 does not switch devices). This review does not extend that inherited multi-GPU behavior or claim it tested.

q/k/v values themselves may be non-finite: no finite-data guard is intended. Scale and delta are finite positive/nonnegative on attention entries. Positive extents do not imply products or a capacity fit int. Allocation failure on a small GPU is not a general range proof.

## Native entries and downstream kernels

| Entry / code site | G: device, dtype, rank and geometry | Alignment and pointer indexing | Extents, capacity, overflow | First launch / refusal; baseline behavior |
|---|---|---|---|---|
| apa_spq1_core_check/core; apa_spo2_run / _spo2.run, launcher.cuh:59-159; packed_launcher:4-6 | Rank4 q/k/kq/v; common CUDA/dtype f32/f16/bf16; kq.shape=k.shape; B/D and V KV/S match; H%KV=0; L=1; max(D,VD)<=TC_APA_MAXD; B*H*L<=INT_MAX; sinks [H] matching type/device; causal S>=L; TC_APA_SP=1, finite scale>0/delta>=0; modes restricted. | Staged ordered G=1 and (cap>64 or S<=2048): kq/k 16B if D%(16/sizeof(T))=0, v 16B if VD divisible. Scalar spo4, spo5 and baseline paths need scalar alignment only. Chunked offsets widened to int64. | Positive per-axis <=INT_MAX G. S+part_keys-1 in fixed partition helper, nc*256 row PK and arena byte sums are U at extreme ranges. Each view is 16B aligned, exact cursor==total G. | All public argument checks precede launch. New alignment refusals replace possible sticky vector faults; FP32 dots=tensor previously silently ran ordered. Other mode/shape/feature refusals existed at baseline. |
| apa_spq1_buffer_check, packed_launcher:8-12, shared by every packed attention/append entry | k rank4 CUDA f32/bf16 D64/128. codes uint8 rank4 [B,KV,C,D/2]; sc fp32 rank3 [B,KV,C]; C>=visible S; same devices. | TQ/int4 tuple layout G. Codes/scales use capacity, exact K and V use visible S. No reads beyond visible S in attention. | C>INT_MAX U: native int capacity/dst_s narrows (F04). Individual B/KV/S positive/range not checked here; attention checks supply them. | No launch; baseline same helper. It validates shapes, not actual packed table values or prefix identity. |
| apa_spq1_rotate / _spo2.packed_rotate, packed_launcher:14-20 | q rank4 CUDA f32/bf16, L=1, D64/128; KV>0, H%KV=0; r fp32 [KV,D,D] same device. | Scalar input loads; own pq fp32 allocation 16B. | B=0/H=0 U (zero grid); extent/product q.B*q.H may narrow U. | On public packed attention called after full attention checks (F1 repair). Raw helper still launches invalid zero grid for empty q (F05); baseline raw helper did the same. |
| apa_spq1_prepare / _spo2.packed_prepare, packed_launcher:24-30 | k rank4 CUDA f32/bf16; D64/128; 0<=start<S; r fp32 [KV,D,D] same device. | Scalar K/R; norm/rot internal allocations. Source row offset int64; row/grid and N/H/S are int. | B=0/KV=0 and B*KV*N overflow U; per-axis narrowing U. N>0 G. | All explicit checks before launch, but empty B/KV yield zero grid rather than host refusal (F05). Baseline same. New packed default reaches this helper. |
| apa_spq1_pack / _spo2.packed_append; new spq1::pack_int4_rows, packed_launcher:33-45, packed.cuh:36-75 | buffer_check; 0<=start<=S; TQ rotated [B,KV,N,D], norms.numel=B*KV*N, bnd.numel=15, all fp32 same CUDA. INT4 ignores preparation buffers. | Scalar loads/stores; codes destination (bh*C+row)*D/2; scale bh*C+row. New int4 kernel stores NaN scale on any non-finite row; D even guaranteed by helper. | C and B*KV narrowing/product U. N=0 intentionally returns without a launch. B/KV zero with N>0 U: zero grid. | TQ/int4 invalid prep checks before launch; raw empty leading axes still launch error. Baseline identical refusal order; INT4 non-finite scale carry is the changed device behavior. |
| apa_spq1_run / packed r2, packed_launcher:47-55 | fmt tq4/int4; buffer_check; cb fp32 [16] matching q device; core_check runs first; rotate checks r before its launch. | r2 bulk codes/R/q scalar. q rotated into owned pq. Same staged G=1 k/v checks as core. G>1 uses scalar exact-K/V selection. GQA any positive integral ratio (bulk_shared only G4/8). | C narrowing U; remaining core constraints as above. | F1 moved all core checks before rotate. Baseline rotated q before feature/scale/shape refusal. |
| apa_spq1_int4_stats / _spo2.int4_stats, packed_launcher:62-87 | q rank4 float32/half/bf16; codes rank4 uint8; scales rank3 fp32; same CUDA device; positive axes <=INT_MAX; H%KV=0; matching B,C,D/2; B*H<=INT_MAX; L1 D64/128, 1<=S<=C. | Scalar kernel from kernels.cu. q flattened row*D; codes/scales use capacity. | C fit int G here. scale/zthr are U (diagnostic accepts them unchecked); arbitrary stats may be NaN/Inf. Extreme S loop arithmetic remains inherited. | New validation prevents rank indexing and KV=0 SIGFPE before launch. Baseline read shapes before ranks and did H%KV without zero guard; wrong dtype/device also unchecked. |
| apa_spq4_run dispatcher, spq4.cuh:454-464 | gqa_path r2/tc and fmt int4/tq4; MHA tc delegates r2; buffer_check for grouped tc. | Delegates r2, SPQ4 INT4 or SPQ6 TQ4; no direct pointer loads. | Per downstream entry. | F1 moved TQ rotation into fully validated SPQ6. Baseline could rotate before G/range refusals. |
| apa_spq4_tc, spq4.cuh:362-430 | rank4 BF16 q/k/v same CUDA, positive axes<=INT_MAX, L1; B/D/KV/S match; D/VD64/128; G2..8 integral; B*H<=INT_MAX/4; modes/scale/delta/feature/sinks checked. | q8B, codes16B(D128)/8B(D64), k/v16B G before first launch. Row strides multiples of width; owned pa8B stores, shared V tiles16B. | Codes C narrowing U; extreme S+partition/row PK and arena sizes U; arena exact and aligned G. | All explicit checks before bulk. Empty axes now host refusal; baseline possible zero-grid errors. Alignment was unguarded at baseline. |
| apa_spq6_tq4_tc, spq6.cuh:189-289 | SPQ4 geometry + cb fp32[16] same device; rotation fully checked by apa_spq1_rotate; default A/B knob refused before rotation. | q8B, codes16/8B, k/v16B G; owned pq float4 16B; shared/pv as SPQ4. | Same C/large-S/arena U as SPQ4. Post-rotation pq check is I to fail with the implemented rotate (own shape/type/device); allocation/driver errors may still occur. | Every input refusal before first rotation, unlike baseline SPQ6 G-range/knob refusals after rotation. |
| apa_eb2b_quantize / _spo2.eb2b_quantize, eb2b.cuh:120-126; changed eb2b::quantize | CUDA BF16 rank3/4, final extent128, numel>0, rows=numel/128<=INT_MAX. | Scalar BF16 input. vi same shape uint8, scales prefix shape fp32; row*128 int64. | Nonempty and row-grid fit int G; no general need for per-axis int here because kernel takes flattened rows only. | All explicit checks before launch; baseline same validation. F1 adds NaN scale per non-finite token (R1). |
| apa_eb2b_values / _spo2.eb2b_values, eb2b.cuh:106-115 | v BF16 rank4 D128; vi same shape uint8; vs fp32[B,KV,S]; flags uint8[B,KV,ceil(S/64)]; same CUDA; v positive per-axis int. | v16B / vi8B G; output fresh16B. Flags scalar, scale scalar, per-row strides multiple of 16/8. | total=v.numel()/8 narrowed to int U; total+127 and kernel block*128 int overflow U. | All explicit shape/align checks prelaunch. Empty/align refused now vs invalid launch/sticky fault at baseline. Huge totals not refused. |
| apa_eb2b_run / apa_p1f1_eb2b_run bridge, eb2b.cuh:138-226 | q/k/v rank4 BF16 D=VD128 L1, B/H/KV positive int; G2..8; S1..65536; K/V shape match; vi/vs match; all 9 operands same CUDA; cb16 fp32; rotation checked in rotate; finite scale/delta/fraction, fraction(0,1]; TC_APA_SP=1; carry partition/row. | codes/k/v16B, q/vi8B G; scalar R/CB/scale. pq own float4; view pa float2. | B*H guard itself evaluates int multiplication U on extreme product; C int narrowing U. Arena cursor==total and 16B views G. | Explicit argument checks before rotation. delta/carry newly required; baseline entry hardcoded 8.1/row. Empty checks added. Bridge introduces no new pointer arithmetic. |
| apa_eb3_run, eb3.cuh:904-1105 | BF16 rank4 CUDA q/k/v, codes/sc/rot/cb same device; positive int axes; L1 D=VD128; G2..8; B*H<=INT_MAX/4; buffers, rot[K V,D,D], cb16, sinks; valid variant and diagnostic/timeline restrictions; feature/scale/delta/modes/knobs checked. | q8B, k/v/codes16B; R16B iff VA.rot=1; all G. Owned pq/pa16B; shared vector tiles aligned. CheckedDeviceGuard and enqueue lock G. | C narrowing, huge S partition arithmetic/row PK U. Arena 16B and cursor exact G. Tickets n=B*KV bounded by B*H. | F05: registered timeline with G3/5/6/7/8 passes prechecks and refuses in byG after rotate+bulk (and possibly poison). Baseline had same missed combination. Other new extent/alignment/knob checks precede first launch. |
| apa_eb4_run, eb4.cuh:677-827 | EB3-like geometry but G in {2,4,8}, S<=65536; vi uint8 V-shape, vs fp32[B,KV,S]; optional inject uint8[B,KV,ceil(S/64)] matching device; fraction(0,1]; force flag; rot/cb/sinks checked. | R/codes/k/v16B, q/vi8B G; pq/pa internal16B; flags/inject/scales scalar. Device guard and lock G. | C narrowing U; leading products use int64 with B*H bound. Arena exact/aligned G. | Input refusals before poison/rotation. Driver allocation or ticket-growth failures after rotation are runtime failures, not missed input guards. Baseline lacked alignment/empty checks and used row carry by default. |
| spo2::merge_kernel; EB3 select_pv and EB4 select_mixed fused merge changes | Only called by validated host entries above; NC positive, rows/VD/output shapes established; PM/PL/PA internal workspace. | Per-call 16B arena, pa vector strides VD64/128 for TC entries; scalar merge handles arbitrary allowed VD. Output owned. Fixed ascending chunk order; no float atomics. | Per-host range U above; no separate device bound check intended. | Guard changed den>0 to den!=0 at all three sites. Finite nonnegative denominator results unchanged; finite inputs can overflow (F06). |
| EarlyCounterHook setup/check/finish, eb3.cuh:91-164, hooks only | bh=1 NC=2 registered fused path; device clock>0; occupancy>=2 resident CTAs checked. State12B holds three words. | Owned aligned state; first word polled, second timeout flag, third forced unread signal; ctr slots int positive from guarded host. | Poison byte products use tiny registered rows/NC; timeout forces c1 return before ticket. | Occupancy refusal happens after bulk and hook setup after poison: hooks-only validation is not universally before first device work. Baseline handshake did likewise. Finish now syncs then zeros merge slots before throwing; driver failure before reset is reported. |
| Ticket buffers/counters/l2 and hooks accessor, eb3.cuh:811-869; EB4 counters: see eb4.cuh | Locked map keyed by current device; EB3/4 guard switches to operand device. Separate merge/choose slots; n=B*KV checked by callers. Accessor n<=0 returns empty; n>0 may grow and copies on current device. | cudaMalloc-aligned uint buffer; zeroed before publication. Growth synchronizes this device, frees old buffer only under enqueue lock. Process-lifetime map. | Per-slot capacity grows max(n,1024), never shrinks. Accessor allows any positive int; allocation failure raises. Size product sizeof(unsigned)*size_t(n) safe on this 64-bit host. | Normal atomicInc(limit=NC-1) wraps to zero. Failed pre-ticket launch leaves it zero; asynchronous device faults are sticky and cannot promise recovery. Timeout reset covers merge after sync; choose completed in prior kernel or is absent at CHOOSE_SITE=1. No silent timeout pass. |

## Python entries

| Entry | Preconditions and classification | Refusal / baseline |
|---|---|---|
| quantize_sp2_keys, apa_sp2.py:61-94; resolve_storage, apa_packed.py:88-122 | Storage string G (argument > env > default); bits exactly int2/4; K rank4 fp32 D64/128 G. Packed eligibility additionally CUDA/backend. Per-axis positivity/native capacity-range U here; downstream helpers apply partial checks above. | New representation switch; explicit invalid packed raises, default/environment packed on ineligible input falls back with reason. Baseline always dense and had no storage keyword. Non-packed build fallback is intentional. |
| PackedSpKeys init/properties/provenance, apa_packed.py:215-280 | Rank/type/D/CUDA G; capacity exact Python int >=visible S and >=1 G. Codes/scales owned via workspace; tables shared/cache; no dense K copy, weak source. Per-buffer tuple shapes G natively. | New entry. C upper bound U (F04); factory fixes C=S (known P1Q-API). Exposed mutable buffers are caller-controlled; raw tuple/table value validity is a caller contract, not cryptographic protection against deliberate mutation. |
| PackedSpKeys append/invalidate/check_source, apa_packed.py:293-324 | Generation/source object/storage revision G; append same geometry/type/device, monotone S<=capacity; no-op append requires same unedited K. Prefix immutability during a true append is explicitly caller-owned (docstring:296-298). | New entry. Generation increments before native append, so failed append leaves object stale (fail-closed). No auto-growth. Identity does not prove a new extended tensor retained its old prefix; that is documented, not an inferred proof. |
| ApaSpq1Workspace init/append/check_source and _workspace_tuple, __init__.py:1628-1690 | Nonpacked build refused before allocation. Capacity>=S,D64/128,fmt G; shared exact-K helper rejects stale/different/no-growth-edited K. True append documents unchanged prefix contract inherited from port. Device consistency for true append is checked natively; no-op requires exact object. | F1 now checks identity; baseline compared only row count and could silently use stale codes. Empty leading dimensions and capacity upper bound omitted. |
| PackedRegisteredMargins load/lookup, apa_packed.py:354-395 | Default loader hash check, registered metadata, exact two entries, finite nonnegative measured values G. Lookup bits/D/dtype/scale G. | New entry. Hard pin before every use U (F01): public constructor and expected_sha override bypass it. frozen dataclass does not validate a mutable entries dict supplied directly. |
| derive_delta, apa_packed.py:125-146 | epsilon finite(0,1], e_q finite>=0, derived double finite<=FLT_MAX G; upward fp32 rounding. | New packed helper, inherited APA-SP2 formula. No device work. |
| public apa_selective_attention_sp / packed_attention, apa_sp2.py:97-136; apa_packed.py:398-437 | Exact packed K/source/bits/device and table type/lookup G; L=1 G; feature flag before native call. Native validates full q/k/v shapes/dtype/sinks/alignment. Dense branch body remains byte-identical. | New packed branch; prefill refuses with dense opt-out message before native work. Dense table now errors with default packed K; raw dense _C gets TypeError on PackedSpKeys. Selection/diagnostic return schema unchanged (Tensor or output/mask). |
| _spo2_initializer_present/_spo2_load/_require_packed_build and public packed wrappers, __init__.py:1578-1628,1699-1773 | Loaded _C path probed via RTLD_NOLOAD|LAZY (fallback dlopen); initializer presence, init failure propagation; stubs and workspace refuse non-packed. | F1 fixes hidden ImportError and allocation-before-None error. Real CPU loader tests pass packed, nonpacked, simulated failure. Global Linux/POSIX loader behavior only; no cross-platform claim. Wrappers forward required EB2b args and partition defaults to native checks. |

## Alignment completeness

43 vector-cast source lines recomputed from raw source (21 uint4, 8 uint2, 11 float4, 3 float2, 0 int4/int2). This count includes 2 mutant-only branches and vector stores; it is not a load-instruction count. VECTOR_CASTS.json contains every line and its classification. There are 17 internal-only cast sites; the EB3 D64 code-load branch is not instantiated by its D128-only callers. The remaining caller accesses map to guarded operands below. d1:26 contains both a caller load and a shared store.

- d1.cuh:26: Kq/K/V at the conditional staged route; launcher:110-115.
- spq4.cuh:86,220 q8B; 99/100 codes16/8B; 232 K16B; 168/296 V16B. spq4_tc:384-387, or calling SPQ6/EB2b/EB3 guards.
- spq6.cuh:130/131 codes16/8B; 105 owned pq16B. SPQ6:215-218.
- eb3.cuh:201 rotation16B; 275/276,370/371 codes16/8B; 480 q8B; 492 K16B; 440/535 V16B. EB3:939-943.
- eb4.cuh:149/150 codes16B; 407 q8B; 419 K16B; 474 V16B or vi8B via eb2b::value8. EB4:711-716.
- eb2b_values.cuh:22 vi8B and 43 V16B; EB2b:166-170, values:111-112, EB4:715-716.
- Internal pq/pa float4 loads, pa float2 stores, and shared uint4 stores have 16B aligned allocations/views and width-multiple offsets/strides. Every float2 cast is a store. Scalar 16-bit pair loads in tensor.cuh do not introduce a wider memory-access requirement.

The supplied ALIGNMENT_CENSUS.json has 32 rows (recomputed); its LF3 receipt has 38 rows (recomputed): all 32 refusal IDs plus six acceptance controls. No missing caller vector operand found. Its internal paragraph is a summary, not the requested complete source-line census; the supplied detailed census also includes a duplicate dtype-specific d1 K row. My source-line list supplies the internal and mutant sites. Guarding G=1 K/V is acceptable as a safety refusal on this implementation; it is a compatibility limit relative to dense. Supporting those views requires a scalar route or aligned handling in a separate change. The baseline packed vector path could fault, so this is not a loss of a valid baseline packed result.

## Arena and growth census

Core, SPQ4, SPQ6, EB2b, EB3 and EB4 each allocate a fresh per-call uint8 arena, round each nonzero slot and each view to 16B, and require cursor==total. Core optionally adds prefix and wide accumulation; EB2b optionally adds prefix/scores/logZ/ticks; EB3 optionally prefix/two timelines; EB4 optionally prefix/used/scores/logZ. A zero-byte optional slot is skipped in the size sum and has no corresponding view. Output is a separate allocation; diagnostics retain arena storage through their views. No arena is a bounded reusable cache and no arena grows across calls. PackedSpKeys never silently grows; it refuses above capacity. The only mutable persistent native arenas are the two per-device ticket slots, whose growth synchronizes and zeroes a new allocation before replacing the old one.

Prior art: CUDA C++ Programming Guide natural vector alignment (NVIDIA; edition/year unverified, lead to check); contiguous storage handles and tensor version counters (PyTorch/Paszke et al., 2019); ordinary aligned bump allocation (ISO C++17, 2017); CUDA Samples threadFenceReduction (NVIDIA; date unverified); no new algorithm in this census.


## Non-finite entry/case census

| Entry family | NaN q/k/v | +/-Inf cases that differ | Zero norm / values |
|---|---|---|---|
| D1 ordered (f32/f16/bf16), tensor dots (f16/bf16) | New shared merge propagates NaN instead of zero; LF2 covers ordered f32/bf16 and tensor bf16 | All -Inf logits: dense online sentinel may give zero; chunk normalization gives NaN. Non-finite sink parity also differs. | Finite zero keys stay finite. |
| r2 tq4 (public f32; raw f32/bf16) | K norm NaN reaches bulk/softmax; q and V NaN reach output row | Infinite K norm times a negative code dot gives -Inf: key skipped and packed finite while dense can be NaN (F02). Infinite q with opposite-sign finite keys gives dense-zero/packed-NaN. | TQ norm zero, dequant key zero; finite result. |
| tc tq4 BF16 / EB2b / EB3 / EB4 | Same norm-carry and NaN merge behavior | Same sign-dependent infinite-key loss via their packed TQ bulk; q all -Inf/NaN cases can add non-finite rows. | Finite zero keys stay finite; EB2b has no sink API. |
| r2 int4 (f32/bf16), tc int4 BF16 | Explicit any-nonfinite vote stores NaN scale | +/-Inf K becomes NaN in every score; extra non-finite heads where dense accepted/skipped -Inf. No sign-based finite concealment of that key. | Zero INT4 key stores safe scale one with zero decoded codes; finite. |
| packed_prepare / packed_append TQ4 | norm NaN and code-zero bins for normalized NaNs | Norm Inf is retained, not canonicalized to NaN; this is F02's source | Norm0, unit0; code ties use >, norm zero cancels centroid. |
| packed_rotate | NaN q reaches rotated coordinates; Inf q can produce mixed +/-Inf or NaNs | Subsequent packed dot can turn a dense -Inf dot into NaN | Finite zero q rotates to zero. |
| packed_append INT4 | New packer stores NaN scale for any non-finite key coordinate | Both Inf signs canonicalized by scale carry | Zero codes represent zero with safe scale one. |
| eb2b_quantize / eb2b_values; EB2b/EB4 int8 V fetch | Any NaN/Inf V element stores NaN token scale; low fetch makes every token coordinate NaN | Sign and finite-column parity is not preserved on low V rows, but the row stays non-finite. BF16 flags fetch original V. | V zero -> scale0/codes0; finite. |
| int4_stats | Diagnostic bulk/statistics only, not an SP output row; rank/type/device/divisor validation now exists | Scale and zthr themselves remain unchecked; non-finite stats are possible | Not a dense-SP output comparator. |

These are source deductions and constructed cases, not an exhaustive GPU test. Q/K/V non-finite row sets of the recorded cases follow. The LF2 dense comparator uses exact K as Kq (scripts/apa_p1f1_gpu.py:462), whereas public default/dense quantizer parity is separately staged in r01. This distinction matters: the receipt is not a public-API quantizer comparison.

| Receipt / dtype / case | Entry | Dense non-finite rows | Entry non-finite rows |
|---|---|---|---|

| LF2/RESULT.json / bfloat16 / k_inf_rev1 | r2_int4 | [] | [0, 1, 2, 3] |

| LF2/RESULT.json / bfloat16 / k_inf_rev1 | tc_int4 | [] | [0, 1, 2, 3] |

| LF2/RESULT.json / bfloat16 / k_inf_more | r2_tq4 | [6] | [] |

| LF2/RESULT.json / bfloat16 / k_inf_more | r2_int4 | [6] | [4, 5, 6, 7] |

| LF2/RESULT.json / bfloat16 / k_inf_more | tc_tq4 | [6] | [] |

| LF2/RESULT.json / bfloat16 / k_inf_more | tc_int4 | [6] | [4, 5, 6, 7] |

| LF2/RESULT.json / bfloat16 / k_inf_more | eb2b | [6] | [] |

| LF2/RESULT.json / bfloat16 / k_inf_more | eb3 | [6] | [] |

| LF2/RESULT.json / bfloat16 / k_inf_more | eb4 | [6] | [] |

| LF2/RESULT.json / float32 / k_inf_rev1 | r2_int4 | [] | [0, 1, 2, 3] |

| LF2/RESULT.json / float32 / k_inf_more | r2_tq4 | [6] | [] |

| LF2/RESULT.json / float32 / k_inf_more | r2_int4 | [6] | [4, 5, 6, 7] |

| LF2S/RESULT_F_supp.json / bfloat16 / k_inf_key100_dim0 | r2_tq4 | [0, 2, 3] | [2, 3] |

| LF2S/RESULT_F_supp.json / bfloat16 / k_inf_key100_dim0 | r2_int4 | [0, 2, 3] | [0, 1, 2, 3] |

| LF2S/RESULT_F_supp.json / bfloat16 / k_inf_key100_dim0 | tc_tq4 | [0, 2, 3] | [2, 3] |

| LF2S/RESULT_F_supp.json / bfloat16 / k_inf_key100_dim0 | tc_int4 | [0, 2, 3] | [0, 1, 2, 3] |

| LF2S/RESULT_F_supp.json / bfloat16 / k_inf_key100_dim0 | eb2b | [0, 2, 3] | [2, 3] |

| LF2S/RESULT_F_supp.json / bfloat16 / k_inf_key100_dim0 | eb3 | [0, 2, 3] | [2, 3] |

| LF2S/RESULT_F_supp.json / bfloat16 / k_inf_key100_dim0 | eb4 | [0, 2, 3] | [2, 3] |

| LF2S/RESULT_F_supp.json / float32 / k_inf_key100_dim0 | r2_tq4 | [0, 2, 3] | [2, 3] |

| LF2S/RESULT_F_supp.json / float32 / k_inf_key100_dim0 | r2_int4 | [0, 2, 3] | [0, 1, 2, 3] |


## Coverage gaps by question

| Question | Existing evidence covers | Gap / practical limit |
|---|---|---|
| 1 | Policy precedence/fallbacks, type matching, forwarding; registered finite public grid | Unchecked margin construction/pin override; factory growth; capacities above native int; actual future distribution is outside calibration |
| 2 | Full 32-row supplied vector refusal census plus six acceptance controls; 63 refusal/knob checks | Raw helper empty axes; EB3 timeline at G3/5/6/7/8; extreme product overflow; scalar external-base alignment; capacity int narrowing; not every wrong tuple/rank/device combination has a device test |
| 3 | NaNs in q/k/v and positive Inf in q/k at two positions, H8/KV2/S1024/D128 G4 | Negative Inf; V +/-Inf; D64 and G1 non-finite matrix; all -Inf logits; non-finite sinks; column behavior; zero norms are not in F1 LF2 and only inherited REV1 evidence exists |
| 4 | Hook timeout/recovery on slot0, BKV1/NC2; source ownership; actual PTDS preprocess refusal | Multiple devices; ticket growth past1024; choose-site0 GPU recovery; nonsticky failed-launch injection; no retained next-output array to recheck LF5 equality; alternate streams out of contract |
| 5 | Real binary loader success, nonpacked refusal, simulated init failure; source linkage | Stale header prototype (F07); stripped hidden host symbols prevent direct bridge nm confirmation; cross-platform loader behavior untested |
| 6 | Independent full B/F dump and ELF symbol census | Original G7 six-removal predicate fails; receipt pins override incorrectly; only sm89 dump/these modules covered |
| 7 | Fixed source order; B/F recorded finite arrays re-compared | No newly run GPU repeatability experiment; finite overflow boundary missing; physical GPU UUID/model not recorded |

## REV1 disposition by ID

| REV1 ID | Disposition | Source / check |
|---|---|---|
| F01 | Named NaN merge repairs complete; broader Inf parity incomplete | merge:30; eb3:788-791; eb4:650-653; INT4/V carry; F02/F06 and host cases |
| F02 | Vector caller guards complete; scalar external-base issue inherited | VECTOR_CASTS.json; 32 supplied IDs covered by38 LF3 rows |
| F03 | Complete defaults | __init__.py:1732,1748 and bindings.cpp:2829,2845; CPU signature tests |
| F04 | Behavior complete; header NIT | eb2b.cuh:138-171,203; binding:2813-2816; wrapper:1718-1723; CPU and LF4 |
| F05 | Complete at both timeout finish sites | eb3:155-164,1102; eb4:824; FH LF5 |
| F06 | Complete at named int4_stats rank/divisor/type/device sites | packed_launcher:67-80; LF4 child exit0; scale/zthr diagnostic domain remains unchecked |
| F07 | Runtime default knobs removed/refused; dead variants not removed | launcher:41-51,106; spq6:222-231; eb3:917-919; eb4:691-693; symbol evidence |
| F08 | Shared exact-K helper applied at all Python workspace wrappers | apa_packed:170-191; __init__:1647-1685,1700,1722,1734,1751; true append prefix remains caller contract |
| F09 | Complete on valid public packed/nonpacked/failure routes | loader/stubs and real CPU tests |
| F10 | Complete | One _C packed wrapper assignment; CPU exact source test |
| F11 | Named attention checks fixed; helper/timeline completeness still partial | core checks pre-rotate, empty attention extents, cursor!=, FP32 tensor refusal; F05 |
| F12 | Complete for actual nvcc flag and explicit macro | receipt027 and the two CPU preprocess cases |

## Lead known-item rulings

| Known item | Review ruling |
|---|---|
| P1F1-INF | Independently found before list; MAJOR rather than MINOR (F02) because the default conceals a dense non-finite row. The proposed NaN norm carry is appropriate; Inf set equality is not currently established. |
| P1Q-API | Agree MINOR (F09); observed after known list. |
| F1-D3 | Agree NIT (F07); bridge sound, obsolete native declaration not sound for clients. |
| F1-D15 | Safety refusal acceptable on actual vector-loaded G1; document compatibility difference. G>1 scalar r2 accepts those views. |
| F1-G7 | Verified substantive counts and unreachability. Original G7 FAIL remains; no separate product defect claimed for retained kernels. |
| F1-RCPT | Agree NIT (F08); also recomputed paired suffix count14 versus recorded17. |
| P1-RES | Registered temporary prefill refusal, sound fail-closed behavior with dense opt-out; no second persistent K representation. |

## Modules, tools and retained evidence

| Build | sha256 |
|---|---|
| B | 8728e67e26027dbafe8a61592decb38e68171fa7b5634239ce0af7897c4e8c4e |
| F | 5c17147715ffc422b0aa7ba8fae826c554a4caff63b1231c22ce580867efc243 |
| FH | 19b4a1783e981302522e79654c278e0e3c462606c2b3687c11876395756be2b6 |

All named modules matched before use. Additional nonpacked loader reference: d6b1b7ad1d87e0d7d5eed70cff94b33e469e6f18b857d78679110e6b2dce3c24. B dump sha8864363cf0d75fc0ddf6052c134da8bf96bbab4239c78510e86ccecb0deca43c; F dump shab36e77cae62835462f547f30e13f2d358ce896d9736bff7745c9cdc43af9e77a. Tools: git2.43.0, Python3.12.3, NumPy2.4.2, pytest9.0.2, GNU nm/c++filt2.42, cuobjdump12.0.140, nvcc12.6.85 (preprocess only). See INSTRUMENTS.log.

Commands and outputs: LEDGER.jsonl stores argv/environment/start/end and receipts/*.log stores full outputs, including negative results. Shell source reads have complete commands in the ledger. Interactive Python here-doc input is in the session transcript; the decisive arithmetic, SASS, symbol and array checks additionally have reproducible scripts in this folder. Those scripts have prior-art comments. No shell transcript is a replacement for the retained raw receipts.

Key replayable CPU commands (use the registered empty-CUDA/nice/single-thread/TMPDIR environment): `timeout 120s python3 blind_host_checks.py`; `timeout 120s python3 numeric_boundaries.py`; `timeout 120s python3 nonfinite_extremes.py`; `timeout 300s python3 sass_audit.py`; `timeout 600s python3 symbol_audit.py`; `timeout 600s python3 receipt_audit.py`. Paths here are relative to this artifact folder; scripts derive the worktree where needed. These are review checks, not production validators. The executed pytest command, module precheck and PYTHONPATH are in receipt025.

### GPU receipt exits and log hashes (read-only; not our runs)

| Receipt | forge exit | inner timeout exit | verified log sha256 |
|---|---|---|---|

| BND_r1.exit | 0 | 0 | 15bd05b98febac824ca14f6c74cd54bfa621763c7194f2ba78cc4df86a2fbe59 |

| LF1_r1.exit | 1 | 1 | 5e7a07f15d92ad4b39c9222328c07350cdd407032bee9ffbea0cb8004c200ebc |

| LF1_r2.exit | 0 | 0 | ed7dec5e8bfb3fe8dd904fe51690606d07f966d680b8c35cc0c1c2aba95fe8ac |

| LF2S_r1.exit | 0 | 0 | f7bedd7c2eeaf3178f1e4a063038429df4eaf799fb42c7a1a34c9c3bc294a8b4 |

| LF2_r1.exit | 76 | None | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |

| LF2_r2.exit | 1 | 1 | f29024e8eb231e95de3f3e01e0ead6ba1947eecf6d54974ec8cc9917ac37e68c |

| LF3_r1.exit | 76 | None | e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 |

| LF3_r2.exit | 0 | 0 | b59a156dc367f140a8dd718e734f463398b3f990b564baac66159a6329bfdc4d |

| LF4_r1.exit | 1 | 1 | 1f2e7a6186e5264df060c90467c13a8cfe69f248a6da6647820481bda43e42b6 |

| LF4_r2.exit | 0 | 0 | b110feaafab5093cfb37b0e86157c3664bd76c1ac1209dce008ceaee74b0e150 |

| LF5_r1.exit | 0 | 0 | a7ff74d956ab2eed86c85effffcf50bf1843a62955ac0d7738a84bf079db3bb9 |

| LF6a_r1.exit | 0 | 0 | 4e752b7de8062dda6f8a3d4e2c31f11980c661e44434a77e835f34c8e38fe9e6 |

| LF6b_r1.exit | 0 | 0 | 1decaf1f2f15b1fbb81ec2b72d29d3a728d8b986c131e91b19829982af4fc851 |

| MUT_r1.exit | 0 | 0 | 4c9849e8f6922a13fcafc93e2b4c04e8628fce69dab9614665404e67038213f9 |


LF2 r1 and LF3 r1 were lock failures (76), not device runs. Failed first LF1/4 attempts remain retained. No timeout is converted to pass. LF2's overall FAIL and Inf prediction misses are retained. Regression outcomes are equality to the historical before results, not every test passing.

## Staged GPU reproductions — compile only

### repro/r01_inf_key_default.py

sha256 `3e4b6e46dd403fc28195e7ffd2dffd547c78f69b003f1a867117a8ea9d345595`. Status: STAGED ONLY; compiled, not executed.

```text
P1R2-F02, GPU staging only. Run unchanged from this worktree root.
CONFIRM: for either Inf sign, default packed output is finite while the dense
opt-out output on the same exact Q/K/V is non-finite. REFUTE for this instance:
both Inf cases keep the dense non-finite row non-finite in packed output.
A finite dense control is INCONCLUSIVE, never a refutation. Other q/k/v NaN,
Inf and zero-key cases report classification only; they are not universal proof.
Prior art: IEEE 754-2008; TurboQuant (Zandieh et al., 2025) inherited tables;
F1 path/sha loader (Project-Tensor, 2026); no new algorithm.
```

Lead command: `forge-gpu-run --min 30 --max 600 --label p1rev2-01 -- timeout 560s python3 artifacts/apa_p1_rev2/repro/r01_inf_key_default.py`.

### repro/r02_finite_overflow.py

sha256 `bc5b40085ef6427c53295a4ac2e9532a7bcb441fba80218bc3b1d757ab185910`. Status: STAGED ONLY; compiled, not executed.

```text
P1R2-F06, GPU staging only. Run unchanged from this worktree root.
CONFIRM finite-input identity counterexample: finite Q/K/Kq/V and FLT_MAX
scale give all-zero finite D1 output on B and a non-finite output on F.
Also CONFIRM sink parity gap if a NaN/+Inf sink gives finite dense output and
non-finite F D1 output. REFUTE for these instances: no B/F classification
change on overflow and no F/dense sink divergence, with finite regular controls.
B and F load in separate child processes; each sha is checked before import.
This is a behavior/claim counterexample, not a proposal to restore NaN masking.
Prior art: IEEE 754-2008 overflow/NaNs; F1 path/sha loader (Project-Tensor, 2026).
```

Lead command: `forge-gpu-run --min 30 --max 600 --label p1rev2-02 -- timeout 560s python3 artifacts/apa_p1_rev2/repro/r02_finite_overflow.py`.

### repro/r03_empty_helpers.py

sha256 `8b9f52067eff058b310b41798ac9660cb737370327f2c4fea08cdad99eef9738`. Status: STAGED ONLY; compiled, not executed.

```text
P1R2-F05 empty-helper portion, GPU staging only; worktree-root invocation.
CONFIRM: raw packed_rotate or packed_prepare on B=0 reaches a CUDA invalid
configuration error (attempted zero-grid launch), while a valid control still
works. REFUTE for these instances: both calls refuse positive-extent
preconditions on the host, with no CUDA launch error, and control stays finite.
Other errors or failure to exercise the control are INCONCLUSIVE/ERROR.
The separate EB3 timeline/G refusal is established by source, not this script.
Prior art: CUDA launch geometry validation (NVIDIA CUDA Programming Guide;
edition unverified); F1 path/sha loader (Project-Tensor, 2026); no new algorithm.
```

Lead command: `forge-gpu-run --min 30 --max 600 --label p1rev2-03 -- timeout 560s python3 artifacts/apa_p1_rev2/repro/r03_empty_helpers.py`.

Compilation receipts: 040 (initial two scripts), 042 (three scripts), and final 051 after r01 was extended. Caches were redirected to /tmp/p1_rev2_review/pycache. No staged script was executed here. The lead must file forge/inner exits and log sha for each run; those outcomes are pending, never counted as a pass.

## Deviations and residuals

- Bootstrap rules read used TMPDIR=/tmp before scratch existed; every subsequent command used /tmp/p1_rev2_review.
- Bootstrap quoting failed before any script started: `/bin/bash: -c: line 4: syntax error near unexpected token '('`. Replaced once with a direct Python here-doc bootstrap. No gates ran before registration.
- Read-only source probes named absent `tensor_cuda/src/tc/ndarray.h`, `tensor_cuda/src/core.cpp` and `tensor_cuda/tests/conftest.py`; correct core.h/kernels.cu paths were subsequently read. The missing conftest is not required by pytest. Probe errors remain in receipts.
- Receipt023 yielded while its command was running, but the displayed wrapper omitted the returned session metadata. It finished in9.24s; the next action read its completed ledger entry before further audit work. No process was killed, no background wait was started. Subsequent tool returns retained metadata.
- Receipt030 inspection raised `TypeError: object of type 'int' has no len()` while inspecting an integer arrays field. Repaired once in031 and continued.
- Git diff/show short help was recorded later than their first use because L99 required the two premise reads first; git --version and all listed instrument versions preceded CPU tests.
- The explicit blind-first order deferred mechanism docs and prior claims until after findings v1, as logged. This takes precedence over the general mechanism-doc-first house sequence.
- The first constructed selector instances worked on scores; later host cases expanded the argument to actual pure table arithmetic for q/k and all non-finite directions. No device assertion is based solely on those host constructions.
- The initial array audit included EB2b tuple item10, a combined timing/byte-census structure. Source spq4.cuh:328-343 proves its clock fields are zero in non-profile mode, refuting uninitialized-clock speculation. To honor L09 strictly, the final replay excludes the whole structure: 6,921 equal comparisons from 6,922 retained arrays.
- NITs are reported for one cleanup and closure, with no separate NIT review round requested. Product/evidence script edits are outside this seat's authorized writable scope.
- No model-quality, speed, residency, DS1, certification, multi-stream support, multi-device recovery or exhaustive non-finite GPU claim is made. The remaining untested space is stated in Q8.

## Cleanup / files

Nothing deleted or moved. The only disposable work is `/tmp/p1_rev2_review/` (receipt runner, pytest temp files, preprocessor output and pycache); the lead may reclaim it after review. No deliverable is proposed for deletion. All review writes are under artifacts/apa_p1_rev2/; product working diff is empty (044). MANIFEST.sha256 lists the final files.

## Prior art

- IEEE754-2008: non-finite propagation, maxNum behavior, overflow and comparisons; used for the source counterexamples and carry/guard recommendations.
- TurboQuant (Zandieh et al.,2025): inherited rotations/Lloyd-Max tables, extracted unchanged from quant.py; no new quantizer.
- BLASST (Yuan et al.,2025) running-max criterion; Hillis/Steele(1986) scan; inherited APA-SP2 two-error margin (Project-Tensor,2026). Used to restate and compare selection semantics; no new selection rule.
- Milakov/Gimelshein(2018), FlashAttention (Dao et al.,2022), Flash-Decoding(2023): inherited online-normalizer/partial merge structure. Higham(2002): finite-precision/overflow analysis. No new merge.
- CUDA C++ Programming Guide / runtime APIs (NVIDIA), vector natural alignment, legacy/PTDS stream rules, and CUDA Samples threadFenceReduction. Used for alignment/ticket reasoning; exact guide edition/date unverified — lead to check.
- SHA-256 (NIST FIPS180-4,2015), GNU nm/c++filt and NVIDIA cuobjdump; P1-D exact per-function comparison (Project-Tensor,2026). Independent streaming implementation; no instruction normalization beyond permitted suffixes.
- NumPy (Harris et al.,2020), same-width unsigned-view equality from P1-D; Python subprocess (standard library,2003 onward) receipt capture. No novel algorithm.
- PyTorch/Paszke et al.(2019) version counters and ISO C++ external linkage/range rules; std::vector reserve (ISO C++,1998). Used for identity/capacity/API recommendations.
- Test doubles (Meszaros,2007) and mutation testing (DeMillo/Lipton/Sayward,1978) describe the authors' existing tests; no new mutant run was performed.
Bibliographic details inherited from source were not checked over the network. Unverified — lead to check search terms: TurboQuant2504.19874, BLASST2512.12087, online softmax1805.02867, CUDA threadFenceReduction, CUDA vector natural alignment, nvcc default-stream per-thread, PyTorch version counters. No prior art known to me for the exact API cleanup/document template; no novelty is claimed. Each finding, review script and ledger disposition carries its source annotation.
