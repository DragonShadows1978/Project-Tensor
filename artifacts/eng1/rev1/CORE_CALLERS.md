# E4, E5 and C++ reshape call-path audit

Reasoning evidence: the three non-topk functions have identical nonempty paths; metadata allocation uses the same input shape, dtype and device. An empty output has no elements to read. No caller can legitimately require the former host division by zero. Topk changes only out-of-range k and no launch for empty output. Unknown external users are not covered.

## Every direct E4 call found in worktree source

| File:line | Source | Caller-specific conclusion |
|---|---|---|
| core/trinity_nano_tc.py:81 | `_, selected = (scores + bias).topk(int(top_k), True)` | Model routing chooses configured k expert scores; valid 0<=k<=expert_count results unchanged. Invalid configs now refuse instead of emitting -1 sentinels; no intentional sentinel use found. |
| core/trinity_nano_tc.py:84 | `top_scores, selected = scores.topk(int(top_k), True)` | Model routing chooses configured k expert scores; valid 0<=k<=expert_count results unchanged. Invalid configs now refuse instead of emitting -1 sentinels; no intentional sentinel use found. |
| core/trinity_nano_tc.py:292 | `return tc.rms_norm(x, self.weight, self.eps)` | RMS inference passes fp32 weights and preserves x dtype; nonempty feature dimension path unchanged; an empty batch retains shape. |
| tensor_cuda/tensor_cuda/functional.py:189 | `weights = _guarded_softmax(scores, lambda s: tc.causal_softmax(` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| external/kimi-swarm/core/mistral7b_tc.py:144 | `return tc.rms_norm(x, self.weight, self.eps)` | RMS inference passes fp32 weights and preserves x dtype; nonempty feature dimension path unchanged; an empty batch retains shape. |
| external/kimi-swarm/core/mistral7b_tc.py:196 | `weights = tc.apa_blend_softmax(b_flat, r_flat, zthr)` | Attention scores/bulk/rank share (...,S). Nonempty launch identical; empty returns same score shape/dtype and can be reshaped as before. |
| external/kimi-swarm/core/gemma4_runner.py:238 | `return tc.rms_norm(x.reshape([B, L * H, D]), wt,` | RMS inference passes fp32 weights and preserves x dtype; nonempty feature dimension path unchanged; an empty batch retains shape. |
| external/kimi-swarm/core/gemma4_runner.py:463 | `p = tc.causal_softmax(` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| external/kimi-swarm/core/gemma4_runner.py:500 | `p_ = tc.causal_softmax(sc)` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apa_mf2_2b_probe/gpu_probe.py:431 | `put(key+'/causal_softmax',TC.causal_softmax(TC.matmul(*[tensor(x,dt) for x in raw[:2]],alpha=D**-.5,trans_b=True)))` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apa_sp4g_a2_model.py:36 | `p=tc.causal_softmax(sc)` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/trinity_nano_first_divergence.py:192 | `_, sel2 = torch.topk(scores_sig + expert_bias.float(), k=moe.router.top_k, dim=1)` | PyTorch call, does not invoke NDArray topk; unaffected. |
| scripts/trinity_nano_first_divergence.py:195 | `gathered, sel2 = torch.topk(scores_sig, k=moe.router.top_k, dim=1)` | PyTorch call, does not invoke NDArray topk; unaffected. |
| scripts/apamq_e1_sweep.py:254 | `weights = tc.causal_softmax(scores)` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apamq_e1_sweep.py:536 | `"tensor_cuda.causal_softmax(scores)",` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apamq_e1_sweep.py:613 | `"- STANDARD is configured to invoke `tensor_cuda.matmul(q_grouped, k, trans_b=True)`, `tensor_cuda.causal_softmax(scores)`, then `tensor_cuda.matmul(weights_grouped, v)`. Q and weights are grouped as `(B, kv_heads, (q_heads/kv_heads)*L, ...)`; K/V are not expanded.",` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apa_mf2_2a_census.py:150 | `put("causal_softmax/" + base, lambda: {"out": tc.causal_softmax(mk(sc, dt))})` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apa_mf2_2b_mutation.py:11 | `"B1": (FN, """        weights = _guarded_softmax(scores, lambda s: tc.causal_softmax(` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apa_mf2_2b_mutation.py:13 | `"""        weights = tc.causal_softmax(` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apa_mf2_2b_mutation.py:18 | `value)""", "        return tc.matmul(tc.causal_softmax(scores), value)"),` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apa_sp3_gpu.py:214 | `native=tc.apa_blend_softmax(bulk,rank,1.2815515655446004,L,0,0)` | Attention scores/bulk/rank share (...,S). Nonempty launch identical; empty returns same score shape/dtype and can be reshaped as before. |
| scripts/apa_spd1_common.py:176 | `weights = tc.causal_softmax(scores) if shape['causal'] else scores.softmax(-1)` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apa_mf2_2c_probe/gpu_probe.py:495 | `with tc.no_grad():o=tc.causal_softmax(s)` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apa_mf2_1_identity.py:93 | `put(base + "/causal_softmax", tc.causal_softmax(sc))` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apa_mf2_1_identity.py:146 | `put("design/causal_softmax_-3.3e38", tc.causal_softmax(tc.tensor(w)))` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| tensor_cuda/tensor_cuda/__init__.py:321 | `return _C.causal_softmax(scores)` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| tensor_cuda/tensor_cuda/__init__.py:999 | `return _C.rms_norm(x, w, eps)` | RMS inference passes fp32 weights and preserves x dtype; nonempty feature dimension path unchanged; an empty batch retains shape. |
| tensor_cuda/tensor_cuda/__init__.py:1331 | `return _C.apa_blend_softmax(bulk, rank, zthr, Lq, row0, window)` | Attention scores/bulk/rank share (...,S). Nonempty launch identical; empty returns same score shape/dtype and can be reshaped as before. |
| tensor_cuda/tests/test_phase9.py:39 | `vals, idx = x.topk(3, True)` | phase9 uses k=3 on last extent 5; range is valid, nonempty kernel path unchanged. |
| scripts/apa_mf2_1_reviewer/gpu_probe.py:203 | `w=host(tc.causal_softmax(a))` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| external/kimi-swarm/gemma4_tc.py:89 | `return tc.rms_norm(x.reshape([B, L * H, D]), wt, eps).reshape([B, L, H * D])` | RMS inference passes fp32 weights and preserves x dtype; nonempty feature dimension path unchanged; an empty batch retains shape. |
| external/kimi-swarm/gemma4_tc.py:485 | `p = tc.causal_softmax(` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| external/kimi-swarm/gemma4_tc.py:524 | `p_ = tc.causal_softmax(sc)` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| tensor_cuda/tests/bench_wo1a_sdpa.py:101 | `weights = tc.causal_softmax(scores.reshape([B, H * L, 1, S])).reshape([B, H, L, S])` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| tensor_cuda/tests/test_apa_mf2_1_sentinel.py:99 | `w = tc.causal_softmax(x).float().numpy()[0, 0]` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| tensor_cuda/tests/test_apa_mf2_1_sentinel.py:105 | `w = tc.causal_softmax(tc.tensor(np.array([[-np.inf, 0.0]], np.float32))).numpy()` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| scripts/apa_mf2_2_probe/gpu_probe.py:431 | `put(key+'/causal_softmax',TC.causal_softmax(TC.matmul(*[tensor(x,dt) for x in raw[:2]],alpha=D**-.5,trans_b=True)))` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| tensor_cuda/bench/harness.py:138 | `tc.causal_softmax(scores_t)` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| tensor_cuda/bench/harness.py:163 | `tc.apa_blend_softmax(bulk_t, rank_t, zthr)` | Attention scores/bulk/rank share (...,S). Nonempty launch identical; empty returns same score shape/dtype and can be reshaped as before. |
| tensor_cuda/bench/harness.py:319 | `tc.rms_norm(x, w, 1e-6)` | RMS inference passes fp32 weights and preserves x dtype; nonempty feature dimension path unchanged; an empty batch retains shape. |
| tensor_cuda/bench/harness.py:455 | `"causal_softmax": lambda shape, dtype, reps, warmup, extra: bench_causal_softmax(shape, dtype, reps, warmup),` | Causal score primitive on rank>=2 scores. Existing S<L refusal retained; nonempty path identical, empty accepted shapes retain shape/dtype. |
| tensor_cuda/bench/harness.py:456 | `"apa_blend_softmax": lambda shape, dtype, reps, warmup, extra: bench_apa_blend_softmax(shape, dtype, reps, warmup, extra["sink"]),` | Attention scores/bulk/rank share (...,S). Nonempty launch identical; empty returns same score shape/dtype and can be reshaped as before. |
| tensor_cuda/bench/harness.py:465 | `"rms_norm": lambda shape, dtype, reps, warmup, extra: bench_rms_norm(shape, dtype, reps, warmup),` | RMS inference passes fp32 weights and preserves x dtype; nonempty feature dimension path unchanged; an empty batch retains shape. |

Callback site functional.py:170 passes tc.causal_softmax through _guarded_softmax; same S>=L admission guard, same nonempty path. Script interposer apa_sp4g_a3_model.py:29/62/68 saves/restores the callable without changing its metadata contract. Mutation-template string hits in apa_mf2_2b_mutation.py are source text, not runtime callers. The sink variant (kernels.cu:4012–4044) uses rows=B*H*L with no host division by S, so D2 correctly leaves it unchanged.

## Every make_dims caller

| Call | Entry line in conv.cu | Allocation/launch and regression argument |
|---|---|---|
| im2col | 158 | Helper first; then allocate (N,C*kh*kw,OH*OW), launch iff nonzero. Public refusal IDs and valid im2col reference pin this. |
| col2im | 166 | Helper first; then zero input gradient, scatter from cols. Conv/im2col backward routes here; invalid geometry cannot get past helper. |
| avgpool2d | 173 | Helper first; then allocate (N,C,OH,OW). Positive geometry uses unchanged kernel, checked against independent windows. |
| avgpool2d_bwd | 180 | Helper first with captured input shape and forward geometry. Invalid forward already refuses; valid backward scatter unchanged and value-tested. |
| maxpool2d | 187 | Helper first; then values and argmax allocation. Invalid geometry refused before either; valid forward/backward value-tested. |

maxpool2d_bwd (conv.cu:198) does not call make_dims: it consumes the captured valid-forward argmax and x_shape. The order premise that both backward pools directly call make_dims is imprecise; max-pool backward is protected by its forward.

For positive strides, floor(A/s)+1>0 iff A>=0. Thus every candidate-accepted geometry has the same extent under old truncation. The old accepted set also contained -s<A<0; those cases are now correctly refused (H=3,k=4,s=2,p=0). Invalid zero/negative strides and >int32 image sizes are inherited limits, not claimed repaired.

## Additional C++ metadata reshape sites

| File:line | Check |
|---|---|
| ops.cpp:812 | sum backward restores reduced singleton axes kshape; product preserved. |
| ops.cpp:827,830 | public reshape passes shape to NDArray; backward restores the captured original nonnegative shape. |
| ops.cpp:912,925 | extrema backward restores keepdim layout; product preserved. |
| ops.cpp:943,944 | product backward restores gradient and product to keepdim layout; same count. |
| ops.cpp:1008 | squeeze removes only size-one axes. |
| ops.cpp:1015 | unsqueeze inserts a size-one axis. |
| ops.cpp:1034 | flatten replaces a contiguous axis range by its explicit product. |
| kernels.cu:1127 | reduction removes singleton keepdim axes. |
| kernels.cu:1154 | reduce_to restores target after same-count reduction. |
| kernels.cu:5093 | grouped KV: rows*G*group = B*KV*S*D under D%group==0. |
| kernels.cu:5099 | scales rows*G restored to B*KV*S*G. |
| gemm_apa.cu:778 | grouped heads: H*L = KVH*M; probability element count preserved. |
| gemm_apa.cu:780 | output grouping reversed with the same head/query identity. |

All previously valid explicit nonnegative shapes take the same final count comparison and share the same storage. Negative extents only arose from already-invalid metadata; their new refusals are intentional. No Python validator intercepts -1: bindings.cpp:1879 forwards the int64 vector.
