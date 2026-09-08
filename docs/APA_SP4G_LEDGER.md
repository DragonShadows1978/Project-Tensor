# APA-SP4G ledger (append only)

2026-09-07 — Registration and source inspection; evidence class: code inspection / preparation.
Read immutable orders/APA_SP4G_GEMMA4_MODEL_TEST.md, AGENTS.md and /mnt/Shared/HOUSE_RULES.md. No gates preceded registration. Registration SHA: 099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e. Both prediction sets are in registration.json. Offline Arrow -> June newline join -> Gemma -it default tokenizer: 292282 int64 tokens; SHA 8bb85a61060d4221fef55f0b134914c55e6110ffbfd10913aed54e99f130d1cb. Full QAT weight SHA faff1a63667fac17ac5e777f47114688fcefea96e220e211aaa8d62c2c4561f1. Receipt logs/apa_sp4g_registration.log. No bf16 model loaded.

Adapter facts: gemma4_tc.py:568/658 strict context threshold; force apa_min_context=0, fast_max_seq=0 in all APA cells. Use reconstructed-kq fused calls :609/:712, INT4/GEMM opt-ins off. B/C/D/E share D=VD=512 MQA ABI, scale1; bf16 K/V unchanged. Incremental kq_count at :353-388, only new rows quantized in decode. Prefill continues whole-prefix quantization, chunked when S>4096. SP prefill and splitK both already instantiate512 (kernels.cu:7496, apa_sp1_1.cuh:179); no product edits authorized/needed. June floor off-by-one corrected to mandated1024 targets; fp64 scoring; cached64 feeding, adaptive prefix chunking preserved. See registration for precise boundaries.

Prior art: June Gemma port ledger/floor (2026) adapter, cache, flags and evaluation protocol; SP3 (2026) calibration/provenance/clean decode; BLASST (Yuan et al.2025/2026, https://arxiv.org/abs/2512.12087) running-max comparator; APA retains all denominator keys and promotes precision. ThriftAttention (Joe Sharratt2026, https://arxiv.org/abs/2605.23081) weight-sensitive precision motivation, no FP4 port. FlashAttention-2 (Tri Dao2023, https://arxiv.org/abs/2307.08691) existing online softmax/work partitioning, no FA2 implementation port. TurboQuant (Zandieh et al.2025, https://arxiv.org/abs/2504.19874) existing rotation/scalar-codebook reconstructed BF16 keys, no QJL residual. SP2 conditional log(1/epsilon)+2eq reused, empirical finite max cannot certify unseen queries. arXiv abstracts verified live; detailed comparator receipt is lead's /mnt/Shared/APA_SP_Prior_Art_Comparison_2026-09-06.md. Standard bisection, nearest-rank statistics, NLL (Shannon1948), SHA-256 (NIST2001), create-only hardlinks/leases, Make (Feldman1979) dependency invalidation, Kahn1962 DAGs reused; historical DAG attribution unverified lead to check. New contribution is Gemma experiment wiring, no novel algorithm.

Implementation in progress: new scripts/apa_sp4g_* only; source-preserving diagnostic copies in artifacts/apa_sp4g/build, D512/MQA tests under tensor_cuda/tests. Calibration uses all causal pairs at prefix0 S2048 (SP3's calibration population), separate from four-window PPL. G2 covers all rows in128-query bands and aggregates full error populations, no sampling. Freeze first actual-C fraction within.01 of B, max12 trials, then terminal RED. Author gates are baseline only; blind verification is lead-owned, unrun. Model system family GPT-6; exact serving variant and configured effort not exposed, not invented.

Host build completed: logs/apa_sp4g_build.log, sealed artifacts/apa_sp4g/build/manifest.json. Native and separate diagnostic extensions compile/link for sm89. Existing source pins pass. CPU import receipt adapter_cpu_import.json: correct workspace extension and read-only Gemma adapter; cudaGetDeviceCount=100, device_count=0. GPU jobs=0. A first ad-hoc import command had a Python SyntaxError (`import core if False else sys`) before execution; corrected import passed. No GPU allocation/model load attempted.

First CPU baseline: logs/apa_sp4g_cpu_baseline.log:31 passed,1 failed. Failure was AST walk enumeration order in test_adapter_live_threshold_same_tensor_callsite_and_incremental_kq: ast.walk is breadth-first, so the nested decode call appears after prefill despite earlier source line. Fix is sorting inspected calls by lineno; both exact argument lists remain asserted. No production/harness behavior or tolerance changed by this test correction.

Execution details before next gates: exactness_2048 aggregates A/D at immutable.005 tolerance and blocks calibration if RED. Full G2 arrays use creation SHA then exact stat identity (inode/device/size/mtime/ctime) on reuse, SP3 a4 practice; metadata JSON still fully rehashed. Owned-cell disk rails:8GiB for capture/trial/band and12GiB for population aggregation. Current disk available159013404672 bytes (inspection, not long-run guarantee). Full G2 float64 errors need about73GB plus captured activations; no sampling fallback. Receipt validation is memoized only within one action; source closures are per kind. This follows Make/Feldman1979 and SP3 a4 dependency invalidation, no novel data structure.

Author baseline after correction:32 passed (logs/apa_sp4g_cpu_baseline_r2.log). First mutation gate:8/8 non-error mutants killed, rate1.0>=.80, no survivors; artifacts/apa_sp4g/mutations/results.json. Mutants are independent copies and original files were never altered. Added behavioral tests then passed36 tests (logs/apa_sp4g_cpu_final.log): real KVRing method quantizes4x512 at cold2048, then1 new row at2049, then0 without append; cached feeding and direct binding object identity pinned. Two SWIG import DeprecationWarnings retained unsuppressed. Source imports do not instantiate a model.

Review before delivery: worker checks lease fd9 device/inode matches /tmp/forge-gpu.lock; only then nonblocking flock confirms ownership. Exactness aggregation added at each requested long length as well as short, fixed.005. Scheduling puts short A/B/D and calibration ahead of long PPL;32K cells last. All1408 cells remain dependency ordered. Computational-cost residual:1280 G2 bands imply10.67hours of cooldown alone if dispatched separately, plus work. No claim that full8192 all-pair analysis is cheap in wall time; the conservative split honors290s hard worker bounds. No GPU execution performed to tune batching.

Final CPU handoff:38 passed,0 failed,0 skipped; logs/apa_sp4g_cpu_handoff.log. Fresh seeded-copy mutation pass on current Python source identities:8/8 non-error mutants killed, rate1.0; artifacts/apa_sp4g/mutations_sealed/results.json. Repeated runs were triggered by new behavioral tests and strengthened gate/lease guards, not tolerance changes. Original failure and all earlier receipts retained. Resume's selection phase is now inside the same588s outer timeout. CPU_GATES.json is create-only and pins every execution source plus tests; SHA362f26ec3f25f67624c9e9caee6e4274739bdf5928b6efc7ad2f70cd52de4239. Final build and all62 registered pre-existing sources verified unchanged. No production edits, no compiled numerical GPU gate claimed.

Final artifact verification: list output exactly matches cells.json;1408 unique exact lead run commands in dependency order, each with worker estimate and lease/cooldown bounds. Kernel preflight resolves to scheduling kind GPU after source/build/protocol/CPU checks; next returns kernel512. No lease or GPU worker was started by these dry runs. GPU_BLOCKED.json: cudaGetDeviceCount=100, device_count=0, actual error `no CUDA-capable device is detected`;0 model loads and0 GPU cells. Lead commands include the unmeasured30–120s QAT load estimate alone. GPU_BLOCKED.json and DELIVERY_CHECKS.json create-only; the latter pins report, commands, cell manifest, CPU gate, registration and amendment. RESULTS.md and receipt_audit.json generated successfully,0 valid/RED/stale GPU receipts, all cells UNRUN. Prior art, both prediction sets, model/effort exposure limits, protocol deviation, diagnostic overhead/disk/cooldown risks and blind-review residual are in RESULTS.md.

Delivery scope: scripts/apa_sp4g_register.py, common.py, registry.py, make_diag.py, build.sh, model.py, metrics.py, gpu.py, lead_gpu.sh, mutations.py, report.py (all share apa_sp4g_ prefix); tensor_cuda/tests/test_apa_sp4g.py; artifacts/apa_sp4g/; logs/apa_sp4g_*; this ledger. No git, subagents, shell background jobs, foreign-process signals, services, model writes or edits to reused SP3 files. Only bounded foreground host build/tests were executed. GPU evidence and independent blind review remain BLOCKED/UNRUN, not claimed completed.

2026-09-07 — Amendment 1 registration BEFORE gates; evidence class: source inspection / CPU symbol diagnosis. Immutable order orders/APA_SP4G_AMENDMENT_1.md; separate amendment_002_a1_execution.json registers whole-layer 2048 estimates10–90s and8192 estimates60–270s, unchanged285s worker TERM. Original registration, receipts and sources snapshotted in a1_baseline with a1_before.json hashes. Diagnosis a1_symbol_diagnosis.json reproduces undefined symbol cudaGetDeviceDefaultMemPool; correct pool/get/set/current-device symbols all resolve without invoking CUDA. CUDA12.6 local headers verify ABI and high-water attributes6/8. No gates yet. Kernel measurement body/build will stay unchanged; shared dispatch/provenance closure changes require an exact reviewed bridge, not an unchanged-whole-file claim. Prior art: SP3 a4(2026) provenance, existing SP4G/June(2026) all-pairs replay/statistics, NVIDIA CUDA Runtime12.6(2024) API; nothing new, no new algorithm. Seat gpt-6-astra, reasoning xhigh, exposed in logs/apa_sp4g_a1_r1.log. No git/subagents/background work/signals.

A1 first CPU suite:47 passed,1 failed,0 skipped (logs/apa_sp4g_a1_cpu.log). Failure: STALE_OR_RED_RECEIPT: kernel512 inside test_kernel_bridge_exact_endpoint_and_unknown_changes_rejected. Diagnosed test isolation: earlier monkeypatched fingerprint was captured by provenance lazy import; standalone live kernel require_pass returned PASS. Test-only correction imports provenance before fixtures. amendment_004_a1_validation.json records exact before/after test hash and retained failure log; no assertion/tolerance or execution source changed. Prior art unchanged; standard test isolation, no new algorithm.

A1 baseline48 passed,0 failed,0 skipped (logs/apa_sp4g_a1_cpu_final.log); two SWIG warnings retained. First mutation pass met threshold:7/8 non-error killed=.875>=.80 (mutations_a1/results.json); stale_fingerprint survived because first-match text replacement moved into the new fallback RAIL guard, while its selected test exercises ordinary receipt reuse. This is an incorrectly located mutation, not demonstrated coverage of that guard. amendment_005_a1_mutation_target.json scopes the same defect to the actual receipt fingerprint/bridge clause. Gate unchanged; no live execution source changed. Repeat copied-source mutations justified by this named unresolved targeting concern. Prior art unchanged.

A1 implementation complete — evidence class: source edits / author CPU suite / provenance dry checks (no new GPU measurement).
Fix: scripts/apa_sp4g_model.py:14-28 resolves all four NVIDIA CUDA12.6 worker symbols at import time against installed libcudart.so.12, binding argument types and cudaError_t return types before any model load. Correct cudaDeviceGetDefaultMemPool at:20/:48; cudaGetDevice, cudaMemPoolGetAttribute and cudaMemPoolSetAttribute also resolved. PoolPeak uses shared resolved handle at:47; high-water6/8 reads/reset ABI retained. Original failure reproduced without a CUDA API call (a1_symbol_diagnosis.json). Tests test_worker_ctypes_symbols_resolve_at_import_without_cuda_calls (:271), test_misspelled_pool_symbol_reds_during_import (:294), test_pool_counters_use_registered_high_water_abi cover symbol inventory, actual dlsym/import failure, and getter/reset signatures.

Final CPU gates:48 passed,0 failed,0 skipped in0.54s;2 SWIG import warnings and final swigvarlink warning unsuppressed. logs/apa_sp4g_a1_cpu_final.log; CPU_GATES_A1.json SHA cd50ac3498f2155c440efeabc78b890846a7ac34dc71be0784ded4d74e6ee7fb. Final copied-source mutation gate8/8 non-error killed=1.0>=.80; mutations_a1_final/results.json. Earlier47/48 baseline failure and7/8 mutation result preserved with amendment_004/005 explanations. Original implementation files were never replaced by mutants. No blind-review claim; lead-owned UNRUN. No rebuild: build manifest and its binaries/sources verified unchanged, all62 registered pre-existing sources and36 inputs verified. Model/effort gpt-6-astra / xhigh confirmed from this dispatch log.

Margin schedule: scripts/apa_sp4g_registry.py:12/:27 registers margin_layer ids margin_{B,C}_{2048,8192}_l{05,11,17,23,29,35,41,47} —32 cells total,16 per S. Estimates2048 10–90s;8192 60–270s, unmeasured, includes replay/exact dots/population partition, no model load. scripts/apa_sp4g_metrics.py:87 reuses128-query internal tiles and existing all-pair error arrays/nearest-rank partition; per-tile bitwise replay failure and incomplete population reject. Every16 query heads retained despite sharedKV=1; causal pair counts33,570,816 at2048 and536,936,448 at8192 per layer. No percentile averaging, sampling or GPU-timing claim. Default DAG128 cells. S8192 cooldown16*30s=8min; both lengths32*30s=16min. Optional margin_band *_qNNNNN and margin_summary *_bands cells only through explicit lead run after current whole-layer RED RAIL;1280 bands+32 summaries excluded from default resume. Fallback success can satisfy downstream layer dependency with validated summary hash while original whole-layer RED stays unchanged. Complete fallback and dependency routing CPU tests passed. Retained errors about73GB plus captures; whole-layer12GiB headroom unchanged metric population, including scratch. Planning estimates are not measured rail clearance.

Fingerprint amendment: amendment_003_a1_fingerprint.json SHA 5b7e2d490a46e3f00ba1cc0a2e58e99c5275bfab2e0f0a30391af074a4ece50a. Raw kernel per-kind shared harness closure DID change (common/registry/gpu, plus new provenance and execution-amendment dependencies); never claimed byte-identical whole closure. Exact reviewed before/after bridge only accepts original PASS kernel512: kernel512 function bytes, initial execute branch AST, runtime/source/token helpers, cell recipe, original CPU gate, runner and build unchanged. Unknown path/hash transitions reject. Source patch a1_source_delta.patch; machine-check receipts inside amendment. Original kernel512 bytes/SHA unchanged, PASS still eligible; zero prior PASS receipts invalidated. Original ppl_A_2048_w0 remains RED, byte-identical with traceback: AttributeError: /usr/local/cuda-12.6/lib64/libcudart.so.12: undefined symbol: cudaGetDeviceDefaultMemPool. It is never bridged to PASS. Corrected same id now writes create-only jobs_a1/ppl_A_2048_w0.json. Original registration SHA 099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e unchanged, and original CPU_GATES.json/GPU_BLOCKED.json/amendment_001_execution.json retained as historical evidence.

Handoff: RESULTS.md, lead_commands.txt, lead_commands_fallback.txt and cells.json refreshed; cells.json has explicit A1 amendment hashes plus128 default/1312 optional recipes. receipt_audit_A1.json inventories both original card receipts and exact reuse decision. GPU_BLOCKED_A1.json is the current blocked report; DELIVERY_CHECKS_A1.json pins refreshed artifacts. Dry list exactly matches cells.json default cells;128 default exact commands and1312 conditional fallback commands verified in dependency order. CPU preflight kernel512=DONE; ppl_A_2048_w0=GPU (classification only). next=ppl_A_2048_w0. Next lead command: bash scripts/apa_sp4g_lead_gpu.sh run ppl_A_2048_w0; then resume runs one cell, or explicit dependency-ordered commands. QAT load alone remains30–120s unmeasured; worker285s TERM/290s hard, outer588s, lease20s, foreground cooldown30s untouched.

RED / not claimed fixed: end-to-end model execution and whole-layer GPU rail remain UNRUN in this seat. CPU visibility cudaGetDeviceCount=100, device_count=0, error 'no CUDA-capable device is detected' in a1_device_visibility.json;0 GPU workers and0 model loads. Original card RED is retained, not erased by a CPU pass. Prior art: nothing new introduced; SP3 a4(2026) provenance/replay and existing SP4G/June(2026) measurement/statistics reused; NVIDIA CUDA Runtime12.6(2024) declarations checked locally. No new selector/algorithm/optimization claim. Prior literature verification text retained from immutable r1 registration is historical, not a new A1 browse. Safety: no git, subagents, background work/waits, process kills/signals, service changes, model writes or product/kernel edits. All foreground calls <10min; original compiled card-tested build preserved.

2026-09-07 — A2 hypotheses registered BEFORE source edits or gates. amendment_006_a2_hypotheses.json SHA 9ffb2675acbb59814df91cc3989941505723d04a82795d74e865748890677bc3. Six ordered diagnostic cells, same-state source/scale/value/RoPE/FP32 and original-shape replay. 285s worker; original .005 PPL gate unchanged; AP conditional on named storage cause; no replay tolerance registered. Lead rail directive recorded; no long retries. a2_before.json preserves source and all existing receipt hashes. Prior art: June Gemma (2026), SP3 (2026) controlled same-tensor reference/replay/provenance reused; no novel algorithm. Evidence class: registration and source inspection, no card diagnostic run. Model gpt-6-astra / xhigh from dispatch log. No git/subagents/background/signals.

A2 source diagnosis BEFORE implementation gates: storage flags explicitly off; prefill tuples exact k/v shared at fork; qk-norm and p-RoPE occur before branch; V shared raw projection then scale-free RMSNorm/no RoPE; scale1/no attention softcap. Standard materializes bf16 logits/probabilities; SP uses fused FP32 accumulation. Numerical effect not yet measured, .005 remains. Old G2 changes call shape and substitutes final-prefix Kq; original per-call Kq/masks were not saved. No atomics in active D512 B reduction; no cuBLAS blend under forced fused flags. Cannot yet distinguish quantizer-context differences from repeat nondeterminism on device. amendment_007_a2_execution.json registers original-call replay probes, actual PPL-linked lossless captures and strict mask+output checks before code. Prior art: June/SP3 (2026), IEEE754/NumPy bit payload storage; ordinary controlled precision ablation, no novel algorithm.

A2 first author CPU suite:64 passed,2 failed,0 skipped; logs/apa_sp4g_a2_cpu_01.log preserved. A1 namespace test failed because lead archived original symbol RED to jobs_stale_r1; verified archived bytes equal a1_before SHA47bbd0615c54e8bd0187b0651ef827573322db6b6c0fea8a3ed547d775655471 and restored create-only copy at original path, no measured receipt changed. A2 rail expected30 was author arithmetic error:5 PPL +15 ceilings =20; fixed test expected count, no rail narrowed. amendment_009_a2_validation.json records exact changes before next gates; thresholds unchanged. No new prior art.

A2 final implementation / evidence class: source inspection, author CPU suite, copied-source mutations, provenance dry checks. New isolated scripts/apa_sp4g_a2_{registry,common,model,metrics,gpu,report,mutations}.py and a2_lead_gpu.sh; test_apa_sp4g_a2.py. ALL preexisting SP4G scripts/tests, A1 CPU closure and compiled build verified byte-identical; report-only delegate considered during implementation was removed before final seal to preserve even the A1 report/CPU pins. No adapter/product/kernel/SP3/model edits.

Hypothesis cells (registration006/007 before code/gates): diag_a2_{source,scale,value,rope}_2048; diag_a2_fp32_A_2048_w0 then diag_a2_fp32_2048_w0; diag_a2_replay_B_8192_l05. Four source cells repeat first global call per layer from same x/cos/sin/offset/tuple cache, compare actual Q/K/V and standard-vs-D outputs, propagate original A. Standard-over-APA-exact-K/V comparator exists; no AP reference adopted. Source evidence: quantized storage flags off; shared exact K/V; scale1/no attention softcap; shared raw K=V projection but distinct normalization and K-only RoPE before branch. Standard bf16 score/probability materialization versus fused FP32 is the remaining concrete numerical-path difference, not a measured cause of +4.144481/+0.518396 PPL. FP32 global-attention-only window0 treatment prepared (weights/projections/sliding layers still QAT/bf16). Original abs(D-A)<=.005 and both REDs remain. No model-specific widened rule or tolerance replay.

Replay finding: A1 replaces early-call Kq with final-prefix Kq and reshapes adaptive calls into128-query bands without saving original per-call masks/Kq. Active B reduction has no atomics; forced-fused path bypasses blend. Device probe must still distinguish native repeat variation from Kq regeneration/call-geometry effects. No root cause claimed confirmed. Correction: actual PPL captures store per-call lossless bf16 Q/K/Kq/V/out plus packed native mask; native output is propagated and instrumented output must match bitwise. G2 repeats complete original calls, checks both output and mask bitwise, then tiles FP64 statistics without reranking. PPL source/dependency hash and selected/pair totals are checked. Population explicitly2047/8191 executed PPL queries, all16 heads/all eligible pairs, window0 short; final input token target-only. Old prefix capture remains historical evidence, cannot certify original PPL selection. C cells remain gated by original exactness/calibration.

CPU final:67 passed(48 original+19 new),0 failed,0 skipped; logs/apa_sp4g_a2_cpu_final.log; SWIG warnings retained. Additional behavioral regression verifies original129/64-query calls with distinct per-call Kq survive before128 statistic tiles, complete population and PPL counts. Initial64/2 failures retained with validation009; corrected66 and then67 baseline runs justified by fixes/new seam test. Copied-source mutations8/8 non-error killed, rate1.0>=.80; mutations_a2/results.json. Mutated source identities rechecked at delivery; no live source replaced. Blind review lead-owned UNRUN. Shell syntax and8 Python ASTs pass.

Rails/context:20 lead rail non-fits (5 ppl_*_16384 +15 ceilings16K/24K/32K),5 measured returncode124 terminations versus15 administratively deferred/unrun. RAIL time-budget non-fit, memory capacity unknown, no OOM claim and no retries. A2 schedule rejects historical/long IDs and has43 cells all<=8192. RESULTS.md retains exact historical model PPL and clean decode/FIT rows; records1024 targets/window4096total, correcting lead header. June GEMMA4_PORT_LEDGER.md:287-295 standard121.74/raw -it template context verified; lead's52-to538 spread contextualized, no new template experiment.

Prior art: June Gemma port/floor(2026) standard MQA, normalization/RoPE/quantizer/protocol; SP3(2026) same-tensor reference/native masks/exact provenance; Make/Feldman1979 and SHA256/NIST2001 fingerprints; IEEE754(2019) float representation, NumPy packbits(system year unverified — lead to check NumPy packbits bitorder release); standard controlled precision ablation, no prior art known to me for a distinct novel method introduced here, no novelty claim. Existing BLASST/Yuan2025 running-max, ThriftAttention/Sharratt2026 weight-sensitive precision, FA2/Dao2023 online softmax, TurboQuant/Zandieh2025 rotation/scalar reconstruction retained with registration URLs. No new selector/kernel; code-site annotations and RESULTS Prior art agree.

Fingerprint amendment008 SHA f484d1619e9010756575800e8fcda37c5df4c930dbc6e84bbf85002c60798b6f; CPU_GATES_A2 SHA 79f710a3b8c2ad61b9c1075328932624e16a586e0462a800b99c17063af3b326. No compatibility waiver:30 historical PASS receipts retain exact closures;15 card RED receipts unchanged plus original symbol RED restored byte-identically from lead archive. New jobs_a2 namespace, separate full A2 fingerprint, unknown changes fail closed. Original registration SHA 099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e unchanged. a2_before and008 enumerate exact source/receipt hashes.

Handoff artifacts verified: RESULTS.md, lead_commands.txt, cells.json, RAIL_NONFITS_A2.json, receipt_audit_A2.json, GPU_BLOCKED_A2.json, DELIVERY_CHECKS_A2.json.43 unique dependency-ordered commands match live list/manifest. First preflight=GPU classification only; next=diag_a2_source_2048. Command: bash scripts/apa_sp4g_a2_lead_gpu.sh run diag_a2_source_2048; resume executes one cell. Use A2 summary; original runner/report retained as historical code. Estimates parity/precision/capture60–285s; replay5–90s; margins2048 10–90s,8192 60–270s. A1 measured QAT load mostly75–76s(first84.7s), planning30–120s retained. Worker285s TERM/290s hard, outer588s, lease20s, cooldown30s, >=12GiB artifact headroom capture/margins. No GPU replay/precision/rail clearance claim.

BLOCKED / Not claimed fixed: cudaGetDeviceCount100, device_count0, no CUDA-capable device is detected; a2_device_visibility.json.0 A2 GPU workers,0 model loads. D/A root cause and replay bug-versus-nondeterminism require lead card receipts; A′ conditional, .005 unchanged. No tolerance accepted; C/E and long memory capacity remain RED/unrun. Process safety: no git/subagents/background jobs or waits, no process kills/signals, no services touched, all actual calls foreground<10min. Seat gpt-6-astra / xhigh from logs/apa_sp4g_a2_r1.log.

2026-09-07 — A3 registration BEFORE implementation / gates / card runs.
Evidence class: source inspection and existing lead card receipts. Immutable
`amendment_010_a3_registration.json` SHA
`d2da08a10e08413887bced5da9829cbfbca0e1ffb5545f1cdae3f63db5e46852`;
`a3_before.json` pins all existing execution files and receipts. Order A3
unchanged. Prior handoff outputs copied to `a3_baseline/` before refresh.

### A3 fork/merge audit — prediction registered before card

Adapter references below are `/mnt/ForgeRealm/GraftRepository/core/gemma4_tc.py`;
SP seam is `scripts/apa_sp4g_model.py`; native sources in this worktree.
Every active branch operation/argument from inputs through merge is listed;
dead paths are identified to distinguish actual calls from alternatives.

| Stage / source | A standard | D refine-all through SP seam | Consequence / suspect |
|---|---|---|---|
| Model flags, seam Model.__init__ | attention_mode standard; apa_min_context=0, fast_max_seq=0 | attention_mode apa_selective; same thresholds; TC_APA_SP=1 | Thresholds select attention implementation only; they do not enter outer chunk formula. |
| Adapter :849-891; harness perplexity/scoring_blocks | ids[:1023] prefix, last_token_only=True; adaptive chunk512 then511; 16 subsequent64-query scoring calls | Identical feeding, token stream, offsets and chunk formula | c0=(L512,S512,offset0), c1=(L511,S1023,offset512); not two512 calls. Last input token2047 is target-only for ALL arms, not an A-only mask. |
| :513-524 projection | qkv_proj slice q; kraw; global vsrc=kraw; or separate q_proj/k_proj | Same | K=V projection does not mean normalized/roped K equals V. |
| :526-536 normalization/layout | q_norm_w/k_norm_w RMSNorm eps; reshape B,L,H/KV,D then transpose1,2; V scale-free RMSNorm, no RoPE | Same | Q=(1,16,L,512), K/V=(1,1,L,512), bf16. |
| :538-553 position/hook | rope_apply Q/K using cos/sin and absolute position_offset; storage hook None | Same | V never roped; a2 source/scale/value/rope probes bitwise on first calls. |
| :557-659 cache | L>1 bypasses ring; cache None or immutable exact tuple; concatenate old/current K,V dim2; S_all=k.shape[2] | Same | QUANT_V/QUANT_KV4 false. Decode full-cap bias and sinks irrelevant to these calls. Repeat receives SAME tuple objects before consumed-list outer return. |
| :661-667 fork/cache return | apa_active false; new_kv=(k,v) | apa_active true iff global and S_all>0; same new_kv | Fork occurs after norm/RoPE/cache append. |
| :668-705 APA preparation | None | zthr=_norm_ppf(1-.15); gemm=false; fused=S_all>0 true; int4_fused=false; _tables(D512,bits4,KV1,True,dev); kq=_quantize_keys(k,...) for S<=4096, else2048-key chunks+cat | Kq is additional input, exact K/V unchanged. Kq affects refine selection, not selected exact dot. |
| :712-714 call arguments | See two GEMMs below | tc.apa_selective_attention(q,k,kq,v,1.0,float(zthr),L>1) | q/k/kq/v object order explicit; scale1, causal True, H16/KV1. |
| Seam native_sp :117-118 / diagnostic_dispatch | No interposer in ordinary A | _C.apa_selective_attention_sp(q,k,kq,v,scale,delta,is_causal,None,False); D delta=float32 max, diagnostics True only for mask validation | zthr intentionally replaced by delta, None is sink absence; output returned is attention B,H,L,D. No output scalar. |
| bindings.cpp :664-680 | matmul / causal_softmax bindings | q/k/kq/v Tensor.data; double scale/delta narrowed float; is_causal preserved; None -> nullptr; diagnostics -> optional selected output | A zero tensor is an extra zero-logit denominator contribution, not absence. Source shows no None-to-zero folding. |
| :729-731 A scores | q.reshape(B,1,H*L,D); matmul(qf,k,alpha=1,trans_b=True); reshape(B,H,L,S_all) | Warp row maps i=row%L; h=(row/L)%H; kvh=h/(H/KVH)=0; dots against exact K when refined | Same head-major order. SP bulk maximum selection is inherited BLASST-related running-max; refine-all asserted on actual mask. |
| native matmul.cu :25-84 | cuBLAS row-major swap operands; OP_T for K; bf16 output with FP32 accumulation, or SGEMM for FP32; alpha1,beta0; batch1 | Fused warp FP32 dot; multiply scale1 | No attention softcap/post-dot hidden scalar in source. GEMM status is not inspected by existing kernel; no change here. |
| :732-745 and kernels.cu :6710-6760 | causal_softmax(sc) under no_grad; visible=S-L+i+1; max FP32, expf, sum FP64, reciprocal FP32; store probability in input dtype; masked entries zero | kernels.cu :7363+: smax=S-L+i+1 if causal else S; online FP32 max/denominator/value accumulator; __expf; no probability tensor | Same mask bound. Numerical materialization/reduction differs; a2 FP32 treatment did not close PPL. Fallback explicit causal mask is inactive. |
| :744-745 A value GEMM | p.reshape(B,1,H*L,S) @ exact V; reshape(B,H,L,D); dtype of input | Fused weighted V; optional sink denominator only if non-null; divide by denominator; store q.dtype | No A-only post-attention scaling. Native SP and diagnostic SP output bitwise required. |
| :765-766 merge | transpose1,2; reshape(B,L,H*D); _cast compute bf16; o_proj; return new_kv | SAME operations after SP replacement | Capture BOTH before-merge and o_proj outputs; FP32 candidates additionally cast to bf16 then same o_proj. Prediction focuses on this boundary/propagation. |
| :794-802 downstream block | post_attention_layernorm, cast, residual; pre-FF norm/cast, MLP, post-FF norm/cast, residual; multiply whole block by layer_scalar | SAME outside attention branch | layer_scalar is after BOTH residuals, not an omitted SP operation. No kernel patch warranted by source. |
| :894-915 outer layer/cache handling | same position_offset, cache consumed-list replacement; final norm/logits after layers | Same | Diagnostic stops by caught local exception after selected attention finishes; no diagnostic prefix is reported as PPL. |

Prediction (reasoning, not measured A3 finding): no single scale/sink/causal/
offset argument correction is supported by the inspected call. Existing
`jobs_a2/diag_a2_fp32_{A_2048_w0,2048_w0}.json` contain144 probes EACH,
all FP32 D-vs-standard max_abs <=4.38690185546875e-5, despite PPL
49.92117893813879 vs53.472390467473765. Thus the current evidence does
not establish a LARGE local FP32 semantic mismatch. Suspect unresolved
boundary: small differences at cast-to-bf16/o_proj and subsequent propagation.
This does not reassert bf16 intermediate precision ALONE as the cause; A2
eliminates that simple explanation. A's gain from52.48049 to49.92118 is about
2.56 PPL; D is effectively unchanged. A3 will test an independent dense FP32
reference and measure cast/projected differences, rather than assume the
aggregate PPL diagnoses the per-call math.

Registered cells: diag_a3_call_l05_c0 / c1 (A-propagated exact window0 first /
second prefix calls), diag_a3_sweep_l05 (both saved calls, 24 FP32 combinations
each). Registered numerical agreement BOTH max_abs<=1e-4 and relative
Frobenius<=1e-5; bf16 descriptive; native replay bitwise; original .005 PPL
unchanged. Single change must rescue nominal disagreement on BOTH calls;
causal-off offset duplicates cannot count as independent improvements.
Explicit offset replay uses L1 prefix calls and is labelled geometry-changing.
No offset parameter is invented for the SP ABI.

Decision rail: nominal agreement -> no argument fix justified, keep RED and
request lead propagation investigation. Named harness culprit -> additive
source/treatment registration and D-fixed window0, then original full
short/8192 gates before C calibration/freeze/PPL/margins. Named A-only semantic
op -> leave kernel untouched, additive A-prime exact removed-op registration,
A-prime minus A cost and D-vs-A-prime .005 comparator; lead chooses reference
and successor DAG. Neither / ambiguous -> RED. Conditional successor IDs
are reservations, not executable cells or authorization to bypass exactness.

Prior art: June Gemma port/floor(2026) attention/chunking, SP3(2026) same-tensor
reference/native replay/provenance, Vaswani et al.(2017) dense scaled dot
attention, ordinary max-shift softmax; inherited BLASST/Yuan(2025/2026),
ThriftAttention/Sharratt(2026), FlashAttention2/Dao(2023),
TurboQuant/Zandieh(2025) as in original registration. IEEE754(2019) bf16 bit
rounding, NumPy; DeMillo/Lipton/Sayward(1978) mutation testing (unverified —
lead to check: Hints on Test Data Selection). New work is diagnostic wiring;
no prior art known to me for a distinct novel method; no novelty claim.
No git, no subagents, no background work/waits, no kills/signals. Seat
 gpt-6-astra / xhigh (logs/apa_sp4g_a3_r1.log). No gates/card runs yet.

A3 implementation and CPU handoff complete — GPU diagnostic finding UNRUN.
Evidence class: source inspection / author CPU tests / copied-source mutations /
provenance dry checks. New isolated `scripts/apa_sp4g_a3_{registry,common,math,
model,sweep,gpu,report,mutations}.py`, `apa_sp4g_a3_lead_gpu.sh`, and
`tensor_cuda/tests/test_apa_sp4g_a3.py`. No preexisting execution file edited.

Actual-call instrumentation observes the two standard GEMMs and intervening
native causal_softmax; saves native scores/probabilities/output. Same-state
APA replay checks Q/K/V bitwise and returns isolated standard for a native
post-projection identity check, then the saved SP result for an actual D merge.
Tuple K/V return bytes are checked too. Both bf16 and fp32 SP native/diagnostic
outputs must be bitwise, with exactly all eligible mask entries selected.
Dense NumPy FP32 references are independent of TensorCUDA; staged BF16 is
labelled descriptive. The FP32→BF16 and shared o_proj outputs are saved to
quantify boundary amplification. First capture ends after c0; second propagates
A's complete c0 then reaches c1, using the real ids[:1023] adaptive prefill.
No partial forward is called PPL. Every saved payload is SHA-checked on replay.

The sweep replays both saved calls, 24 FP32 combinations each, with their own
NumPy specifications plus nominal NumPy and standard comparisons. None vs
zero sink is explicit. Absolute/lost-offset rows use existing L1 split-K with
prefix K/Kq/V, labelled a geometry intervention; there is no invented offset
ABI. Causal-off offset labels are no-op controls and share the identical
actual kernel call. No kernel or original harness argument has been patched;
no measured cause exists in this seat. Conditional D-fixed/A-prime successor
IDs remain non-executable until a named treatment/removed-op registration.
C/E stay behind the original .005 gate. These diagnostics can justify the
next branch; they cannot themselves unblock calibration.

Commands / gates:
- `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -q -p no:cacheprovider tensor_cuda/tests/test_apa_sp4g.py tensor_cuda/tests/test_apa_sp4g_a2.py tensor_cuda/tests/test_apa_sp4g_a3.py`
  First:87 passed,0 failed,0 skipped, logs/apa_sp4g_a3_cpu_01.log.
  Final:88 passed,0 failed,0 skipped in0.90s,
  logs/apa_sp4g_a3_cpu_final.log. Existing two SWIG import warnings plus final
  swigvarlink warning retained, unsuppressed.
- `python3 scripts/apa_sp4g_a3_mutations.py` and later `--final`, single-thread
  CPU / foreground. First and final each10/10 non-error mutants killed,
  rate1.0 >= registered.80; mutations_a3/ and mutations_a3_final/ receipts.
  No live sources were replaced; exact mutant source hashes rechecked.
- First passing baseline exposed no failure, but author inspection found a
  concrete classifier concern: finite error tolerances are not transitive.
  amendment_012_a3_validation.json registered the correction BEFORE repeat
  gates: directly require standard–NumPy as well as SP–standard and SP–NumPy.
  New boundary regression demonstrates the two other comparisons can pass
  while standard–NumPy fails. Bounds unchanged. Original87-pass and original
  mutation receipts retained. No new algorithm/prior art.
- Nine Python ASTs and `bash -n scripts/apa_sp4g_a3_lead_gpu.sh` PASS.
  `list`, `next`, two call preflights and `summary` exercised without GPU.
  First/second preflight=GPU (classification only), next=c0; sweep rejects its
  missing c0 receipt with FileNotFoundError, as required by dependencies.
  No receipt created for that dry dependency rejection.

Fingerprint amendment011 SHA
`fdd3cf12deea588373c6b3e804255bb2c53dd0a9985c31022f9e4e115e8cbcb1`.
CPU_GATES_A3 SHA
`203a7abf9e815b18d75aec0a9c425778c629c3a0d2f0d82b243a072f829e916f`.
Original registration SHA
`099a8bd9e1bb94909a81c110521d2d3a9d642d5fac27d49f6820b97cb3fbbd1e`
unchanged. All23 preexisting SP4G execution files and72 receipt files are
byte-identical to a3_before.json; build/product/adapter/input pins also verified.
No historical receipt closure changed or RED upgraded. New namespace jobs_a3
has0 receipts. `receipt_audit_A3.json` inventories preservation. Fresh handoff
RESULTS.md, cells.json, lead_commands.txt, fallback note, audit copy, 010/011/
012 amendments, CPU/GPU blocks and visibility are pinned by
DELIVERY_CHECKS_A3.json. Live list/manifest/three commands match exactly in
dependency order. Generated report prints every pair's max-abs/relative-F and
all sweep rows once validated card receipts exist. A2 report snapshot is kept
in a3_baseline; current RESULTS supersedes its pending-state narrative.

Lead commands, each a separate foreground lease, stop on RED:
`bash scripts/apa_sp4g_a3_lead_gpu.sh run diag_a3_call_l05_c0`
`bash scripts/apa_sp4g_a3_lead_gpu.sh run diag_a3_call_l05_c1`
`bash scripts/apa_sp4g_a3_lead_gpu.sh run diag_a3_sweep_l05`
`bash scripts/apa_sp4g_a3_lead_gpu.sh summary`
Capture planning90–250s including30–120s load (historical roughly75–85s);
sweep30–270s without model load. Estimates unmeasured on these new cells.
Worker285s/hard290s; outer588s; lease20s; foreground cooldown30s; disk≥2GiB.
No >=16384 jobs or retries, no C bypass. First two capture cells intentionally
independent except kernel512; sweep requires BOTH create-only capture receipts.

Prior art: code sites, initial ledger audit and RESULTS Prior art carry the
June Gemma/SP3 (2026) branch/chunking/reference/replay/provenance precedents,
Vaswani et al. (2017) dense attention, stable softmax/Frobenius standard methods,
IEEE754 (2019) ties-to-even BF16, Make/Feldman (1979), SHA256/NIST (2001),
DeMillo/Lipton/Sayward (1978) mutation tests (unverified — lead to check Hints
on Test Data Selection). No prior art known to me for a distinct novel method
introduced here; no novelty claim. Inherited BLASST/Yuan (2025/2026) running
maximum, ThriftAttention/Sharratt (2026) weight-sensitive precision,
FA2/Dao (2023) online softmax and TurboQuant/Zandieh (2025) quantization were
not modified. A3 web tool checked primary arXiv author/title/year records:
https://arxiv.org/abs/1706.03762 ; https://arxiv.org/abs/2512.12087 ;
https://arxiv.org/abs/2307.08691 ; https://arxiv.org/abs/2504.19874 ;
https://arxiv.org/abs/2605.23081 . No external benchmark reproduced.

RED / Not claimed fixed: D/A model exactness, named argument/semantic culprit,
A-prime adoption, C/E calibration/quality, A3 GPU numerics/rail clearance,
long-context memory capacity. B replay fix is now card-confirmed16/16; old
REDs remain. A's FP32 PPL gain2.56 and D's precision insensitivity are stated;
local FP32 max4.3869e-5 versus global3.5512 PPL gap is a finding requiring the
new per-call/merge evidence, not a root cause. G2/G3 alone establish nothing
about model quality. CPU baseline/mutations are NOT blind verification;
lead-owned blind review UNRUN.

Sandbox visibility fresh receipt a3_device_visibility.json:
cudaGetDeviceCount=100, device_count=0, error verbatim
`no CUDA-capable device is detected`. Zero GPU workers, zero model loads.
No git, no subagents, no background jobs/waits, no process kills/signals,
no services, model writes, product/kernel/SP3 edits. All actual commands
foreground<10min. Model under test Gemma-4-12B-it QAT q4_0 symmetric-8 g32,
bf16 engine compute. Seat gpt-6-astra / reasoning xhigh, verified current
`logs/apa_sp4g_a3_r1.log` header.

A4 registration BEFORE implementation/gates, 2026-09-07. Evidence: source
inspection and existing card receipts, not new GPU execution.
Immutable amendment013 SHA9fce2b05849eb4e69e8cb5cc2a3434e4c05510677128473aa1f49a4c28af2b1a.
Order remains untouched; a4_before.json pins all historical execution files
and job receipts; a4_baseline saves the previous synthesis/commands/cells.

Suspect-cell audit contradicts the lead's proposed missed-cast diagnosis:
apa_sp4g_a2_model.py:85-93 explicitly converts q/k/kq/v via .float(), calls
native SP, and returns d.astype('bfloat16') for treatment D32. Both A2 receipts
contain144 fp32 comparisons. bindings.cpp:222 calls ops::cast(Float32),
ops.cpp:1013 allocates/converts a changed dtype, SP binding forwards data,
kernels.cu:7441+ validates all dtypes and dispatches on q.dtype. SP's bf16
kernel already uses float loads/dots/softmax/accumulators before its final
bf16 store. Therefore identical D32/bf16 PPL does not establish a missed cast.
Historical dtype was not receipted; source evidence supports fp32, runtime
assertion on the original cell cannot be retroactively claimed. No confirmed
arithmetic bug is fixed. A4 adds actual native-boundary input/output dtype
pins, schedule/coverage checks and explicit A32 scope. Do not extend fp32
through o_proj because the A2 A32 comparator did not do so.

Lead prediction registered: abs(D32-A32)<=.005; A32 exact receipt49.92117893813879.
Seat prediction: unchanged arithmetic may repeat D32=53.472390467473765 and
FAIL. The gate is unchanged from lead a4; failure stops all successors.
A4 propagation splits into A/D leases and a named aggregate, saving all2047
executed residual rows at each of8 complete global blocks plus final norm.
C/E successors are registered behind D32: actual-PPL fraction calibration,
C/E four short windows plus8192, C16 full-population margins and clean decode,
extra A32 reference windows1-3/8192 for a like-for-like comparison table.
No >=16384 retries; empirical E bound remains conditional and finite.

Prior art: June Gemma port/floor and SP3/SP4G A2/A3 (2026): same attention seam,
feeding, bitwise replay, capture and clean decode; inherited BLASST/Yuan et al.
(2025/2026) running maximum, ThriftAttention/Sharratt (2026) weight-sensitive
precision, FlashAttention2/Dao (2023) online softmax, TurboQuant/Zandieh et al.
(2025) reconstructed quantization, SP2 conditional delta. Primary arXiv records
checked via web:2512.12087,2605.23081,2307.08691,2504.19874. Standard FP dtype
assertions, Frobenius norm, bisection, SHA256/NIST (2001), Make/Feldman (1979),
DeMillo/Lipton/Sayward (1978) mutation tests (unverified — lead to check:
Hints on Test Data Selection). New work is diagnostic/dispatch wiring;
no prior art known to me for a distinct novel method; no novelty claim.
CUDA visibility source probe: rc100, device_count0, 'no CUDA-capable device
is detected'. Will record blocked GPU handoff; no model loaded. Seat
 gpt-6-astra / xhigh, logs/apa_sp4g_a4_r1.log. No git/subagents/background
waits/kills; large A2 captures and error arrays retained.

A4 handoff review found a concrete new calibration lookup defect before GPU:
B PPL capture result has global_fraction.fraction; A4 trial/freeze read a
nonexistent top-level fraction. This would fail closed with KeyError after
D32, not alter a numerical result. Amendment015 registers the schema fix and
regression BEFORE correction. Initial100-pass CPU gate and8/8 mutations,
fingerprint014 and initial DELIVERY_CHECKS_A4 remain intact, superseded for
execution by fingerprint016 / CPU_GATES_A4_V2. No metric, tolerance, selection
population or algorithm change. Prior art: SP4G A2 (2026) actual-PPL schema;
standard regression testing. No new prior art. All GPU work still UNRUN.

A4 final CPU handoff complete; GPU gate UNRUN / C/E BLOCKED. Evidence classes:
source inspection, CPU comparisons of saved A3 card payloads, author CPU gates,
source-copy mutations, dependency/provenance dry checks. No new card result.

New load-bearing finding, a4_suspect_cell_audit.json and logs/apa_sp4g_a4_audit.log:
on BOTH A3 layer5 c0/c1, saved SP_fp32_cast_bf16 vs SP_bf16 is BITWISE equal
(max_abs0,relF0), and SP_fp32_projected vs actual_D_projected is also BITWISE
(max_abs0,relF0). This supports unchanged D output after the return cast at
these calls. It does not prove all-layer equality or retroactively receipt
A2's native dtype. The proposed missed-cast diagnosis is not established;
source evidence supports A2 already executing the requested precision scope.
Fix delivered is observed native input/output pins and full-call coverage,
not a claimed attention arithmetic rescue. Registered lead prediction and
.005 D32/A32 gate remain exact; seat contrary prediction retained.

Implementation: additive scripts/apa_sp4g_a4_{register,registry,common,model,
metrics,gpu,report,mutations,audit}.py, apa_sp4g_a4_lead_gpu.sh, test_apa_sp4g_a4.py.
PrecisionModel inherits the original A2 attention interception and PROTOCOL-G,
asserts bf16 source, fp32 Q/K/Kq/V immediately before native entry, fp32 native
output and bf16 return, receipts all144 calls on window0, checks their geometry
against A2 A32. ResidualModel observes complete global-block output including
layer_scalar and post-final-norm compute cast, restoring the original class
and norm after each worker. No kernel/product/adapter edit. C capture IS its
PPL; A2 bitwise native replay copied with only lookup/id/output namespace
changes; old captures and margin errors kept. New B target schema correction
is amendment015, reproduced before fix as `KeyError: 'fraction'` in
logs/apa_sp4g_a4_schema_regression_before.log, then fixed to
result.global_fraction.fraction. This was a new harness defect; no GPU had run.

52 immutable cells in amendment013 and cells.json; lead_commands.txt contains
all52 in exact dependency order, each a separate foreground lease:
- ppl_a4_D32_2048_w0 first; any RED stops all successors.
- diag_a4_propagation_A_2048_w0 / diag_a4_propagation_D_2048_w0, aggregate
  diag_a4_propagation_2048_w0; separate loads keep each within285s.
- trial_a4_00..11, freeze_a4: actual-PPL window0 B fraction, match±.01;
  completed-match carry executes no model; one frozen delta.
- ppl_a4_A32_2048_w1..3 and ppl_a4_A32_8192, for C's matching reference table.
- ppl_a4_C_2048_w0..3 / ppl_a4_C_2048 / ppl_a4_C_8192; w0/8192 also capture.
- margin_a4_C_{2048,8192}_l{05,11,17,23,29,35,41,47}; eq_a4 using B16+C16.
- ppl_a4_E_2048_w0..3 / ppl_a4_E_2048 / ppl_a4_E_8192.
- decode_a4_C_2048 / decode_a4_C_8192: original clean32-step decode unchanged.

Planning: load75–130s (prior75–126s observations), model cell90–275s total;
margin10–90s@2048 /60–270s@8192, CPU aggregates1–30s. All new timings
UNMEASURED. Worker285s/hard290, outer585s/hard588, lease wait20, foreground
cooldown30, disk>=12GiB. No>=16384 retry or long lease; old rail non-fits
are not OOM or hardware capacity evidence. All trials/quality/decode after
current D32 pass only, propagation completion additionally precedes trials.

CPU gates:
- Initial full baseline100 passed,0 failed,0 skipped; logs/apa_sp4g_a4_cpu_01.log
  and apa_sp4g_a4_cpu_final.log. Initial mutations8/8; immutable CPU_GATES_A4,
  fingerprint014 and initial delivery manifest intentionally superseded by015.
- Final full baseline101 passed,0 failed,0 skipped in1.11s;
  logs/apa_sp4g_a4_cpu_v2.log. Two existing SWIG import warnings and final
  swigvarlink warning retained unsuppressed.
- Final mutations8/8 non-error, rate1.0>=.80; mutations_a4_v2/results.json and
  logs/apa_sp4g_a4_mutations_v2.log. Source-copy only; no live file replacement.
- Dtype test test_a4_fp32_native_dispatch_dtype_pin_and_returned_arm; negative
  tests reject silent failed cast and native bf16 output. New schema regression
  test_a4_calibration_uses_actual_ppl_fraction_receipt covers trial/freeze on
  real B PPL receipt shape. Norm aggregate test distinguishes full-population
  Frobenius from averaging chunk ratios; gaps and mismatched dtype RED.
-10 Python ASTs and shell syntax PASS. List/next/D32 preflight/summary PASS;
  preflight GPU is classification only. Six representative propagation/trial/
  C/E/margin/decode preflights reject missing D32 receipt. Existing B16 margin
  dependencies validate PASS. Live52-cell list, manifest and command order match.

Immutable registration013 SHA
9fce2b05849eb4e69e8cb5cc2a3434e4c05510677128473aa1f49a4c28af2b1a.
Final fingerprint016 SHA
c1e024c9d6924ffb834af389c9dfff53e272c54ca187912f4b953cd747e3ea20.
CPU_GATES_A4_V2 SHA
79891cdfb6b8a8156c4a88e107ee09f38bffe0df9deaeef0bc7bf5f242af7091.
DELIVERY_CHECKS_A4_V2 SHA
a827c9f8f8286163d2e6e35f6ab189aa5d97af9bc6c99b571f872b90b5c19e85.
33 preexisting execution/test files and75 job receipts preserved byte-for-byte;
original registration099a8bd9... unchanged, build/product/adapter/token/weight
pins checked. New jobs_a4 receipts0. GPU_BLOCKED_A4_V2 is the current handoff;
initial GPU_BLOCKED_A4 / DELIVERY_CHECKS_A4 are superseded by015/016, not current
source seals. Narrative RESULTS now shows all baseline rows, A3 call table,
bitwise cast/projected audit, pending propagation/C/E and dual A/A32 columns.

Prior art: unchanged from initial A4 entry and code sites; June Gemma/SP3/SP4G
A2/A3 (2026) seam, schedule, replay, capture, calibration, provenance and clean
decode; SP2 conditional empirical delta; BLASST/Yuan(2025/2026),
ThriftAttention/Sharratt(2026), FA2/Dao(2023), TurboQuant/Zandieh(2025)
inherited and untouched. Primary arXiv records checked; no external benchmark
claim. Standard Frobenius/precision ablation/bisection/hash/dependency methods;
DeMillo/Lipton/Sayward(1978) unverified — lead to check Hints on Test Data
Selection. No new algorithm; no prior art known to me for a distinct novel
method introduced here. New work is observation/dispatch/report wiring.

RED / Not claimed fixed: D32/A32 exactness, missed-cast root cause, numerical
propagation causality, C/E quality or capacity. Finite per-call agreement is
not model exactness; G2/G3 establish nothing about model quality themselves.
Author CPU tests/mutations are NOT blind review; lead-owned review UNRUN.
a4_device_visibility: cudaGetDeviceCount100, device_count0, error verbatim
`no CUDA-capable device is detected`. Zero GPU workers/model loads and zero
signals in this seat. No git, subagents, background jobs/waits, kills,
services, model writes, kernel/product/SP3 edits; actual commands foreground
<10min. Lead runner retains bounded owned-worker timeouts from prior orders;
no timeout fired during this seat. Model under test Gemma4-12B-it QAT q4_0
exact symmetric-8 g32, bf16 engine. Seat gpt-6-astra / reasoning xhigh,
verified logs/apa_sp4g_a4_r1.log header.

Receipt transcription correction: the final101-pass CPU log and
CPU_GATES_A4_V2 record1.10s (JSON1.1), not1.11s in the preceding entry.
Counts, hashes and verdict unchanged.
