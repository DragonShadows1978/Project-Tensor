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
