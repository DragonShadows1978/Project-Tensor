# Selective-training census (source inspection / reasoning)
Baseline kernels.cu SHA-256 is
f1951814eae4f8183c3a2d3cd9b86eda9a5e0df74def7f07e5393e3c3f67fb2b,
matching the immutable engine's source. Survey covered all branches before
implementation; scratch/original holds the four original modified source files.
Line numbers below refer to the final source.

| Entry point / dispatcher | Targets and conditions |
|---|---|
| Python `tc.apa_selective_train` -> binding -> `tc::ops::apa_selective_train` (`ops.cpp:778`) | Differentiable public entry. Forward getter defaults `a`; backward getter defaults `a`. Captures q/k/v, detached kq, saved lse/thr; non-a backward also captures out. |
| Python/binding `apa_selective_fwd_train` -> `tc::apa_selective_fwd_train` (`kernels.cu:2639`) | Base forward `apa_selective_fwd_train_kernel<T,DMAX,WCOOP>` (`:2350`), 128 threads/query row; DMAX=64/128/256/512. WCOOP=false only cap<=64 and decode-shaped L=1, rows<2*SM_COUNT (default56); otherwise true. |
| Binding `apa_selective_fwd_train_variant`, ops forward switch | `a` -> base; `h` -> `tc::apa_selective_fwd_train_variant` (`:8220`) -> `tc::apa_selective_fwd_bk4` (`:8193`) -> `bk4::forward<DIAG=false>` (`:8065`), BF16 D/VD<=128; `h_tf32` -> `tc::apa_selective_fwd_tf32` (`attention_tf32.cu:407`) -> `pt_tf32::forward<DIAG=false>` (`:229`), FP32 SM80+. |
| Binding `bp_kernel_4_diagnostic` | `tc::apa_selective_fwd_bk4` or `tc::apa_selective_fwd_tf32` with DIAG=true. `a` -> base plus `bk4::a_mask` / `pt_tf32::a_mask`; `h` / `h_tf32` -> corresponding forward<true>. Masks/clocks diagnostic only. |
| Python/binding `apa_selective_bwd` -> `tc::apa_selective_bwd` (`kernels.cu:2684`) | `apa_selective_bwd_kernel<T,DMAX,DK_STE>` (`:2530`): query-owned dQ, dtype dK/dV atomics, explicit rowdot recomputation. 128 threads; DMAX=64/128/256/512. |
| Binding `apa_selective_bwd_variant` -> `tc::apa_selective_bwd_variant` (`ops.cpp:1159`) | `a` -> base; `b/c/d` -> bk1; `f/f_pass_a` -> bk2; `g1/g2` -> bk3; `g1_tf32` -> TF32. Validation/branches unchanged. |
| `tc::apa_selective_bwd_bk1_cuda` (`kernels.cu:2856`) | `apa_selective_bwd_bk1_kernel<T,DMAX,WRITE_KV,OUTPUT_DOT,DK_STE>` (`:2736`). b=true/false (FP32 dK/dV atomics, recomputed rowdot); c=false/false (deletion diagnostic); d=true/true (saved-output dot rowdot). Final casts unchanged off. |
| `tc::apa_selective_bwd_bk2_cuda` (`:2980`) | Query prepass reuses bk1<WRITE_KV=false,OUTPUT_DOT=(f),DK_STE=false>, writes saved_rowdot. Key pass `apa_selective_bwd_bk2_kv_kernel<T,DMAX,DK_STE>` (`:2915`), one block per batch/KV-head/key, fixed FP32 reductions, no atomics. f uses output dot; f_pass_a recomputes rowdot. |
| `tc::apa_selective_bwd_bk3_cuda` (`:3207`) | BF16 `bk3::backward<KEY,DENSE,DK_STE>` (`:3076`), D/VD<=128. g1 DENSE=false selective scalar exact scores; g2=true dense exact MMA scores. Query half KEY=false,STE=false; key/value half KEY=true,STE=switch. Owner tiles, BF16 coefficient MMA, saved-output rowdot. |
| `tc::apa_selective_bwd_tf32` (`attention_tf32.cu:447`) | FP32 `pt_tf32::backward<KEY,DK_STE>` (`:87`), D/VD<=128, SM80+. Query half false/false; key/value half true/switch. Existing TF32 coefficient products, saved-output rowdot, singleton identity unchanged. |
| Backward training setter `bp_kernel_2_set_variant` | Accepts a/b/d/f/f_pass_a/g1/g2/g1_tf32. c is bare diagnostic only and cannot be selected for autograd. Forward setter accepts a/h/h_tf32. No TC environment variable changes either getter. |

This is the complete kernel/launcher graph for this op: three training-forward
kernel families, two diagnostic-mask kernels, five backward kernel families
(base, bk1, bk2, bk3, TF32). Inference selective/sink/split-K/int4/SP and composed
blend-softmax/GEMM attention are separate APIs, not dispatched by this op.

GRAPA's specified call dispatches base forward
`apa_selective_fwd_train_kernel<__nv_bfloat16,128,true>` and base backward
`apa_selective_bwd_kernel<__nv_bfloat16,128,STE>`, grid=65536, block=128,
KVH=16, group=1. Reason: B=1,H=16,L=S=4096,D=128,VD=64 means cap=128;
L>1 rules out decode; both variant getters initialize to a. zthr is the
engine `_norm_ppf(0.85)` cast to float, exactly as bindings already do.

MLA confirmation (read-only real model code): GRAPA `grapa/attention_mla.py`
sets qk=nope+rope at :58; kv_b reshapes to B,L,H,nope+vdim at :103;
k_nope and v are per-head, k_pe broadcasts to H, and cat assembles k_h with H
heads at :119-120. Therefore KVH=H=16 after up-projection, not one latent head.
`grapa/attention.py:66-72` selects tc.apa_selective_train for causal grad-enabled
refine<1 with kq supplied. `/mnt/ForgeRealm/grapa_run7/registration.json` records
checkpoint-carried qk_rope_dim=64 (overriding the repo's older default32);
nope64+rope64 gives D128 and VD64. No trainer/registration file was changed.

`TC_DETERMINISTIC=1` / legacy `TC_DET_EMBED_BWD=1` initialize one shared policy
in embed_deterministic.cu:21-25. Only embedding and gather/scatter use it.
They do not redirect APA. Base/bk1 still use atomics; bk2/bk3/TF32 have owners.
Neither global APA determinism nor FFN gate behavior is established here.

# Operand census and off-path argument

| Touched site | Only on change | Off-path identity reasoning |
|---|---|---|
| base backward + launcher | Template DK_STE makes dK guard `sel || DK_STE`. | false folds exactly to original `if(sel)`; same dK cast, q/scale/dscore, dQ/dV, grids and reductions. All 12 off SASS specializations have exact original instruction encodings. |
| bk1 backward + launcher | Same guard in WRITE_KV paths; diagnostic c adds only unselected dK, using FP32 scratch and one cast. | false preserves WRITE_KV=false elimination for c, original pointers, allocations, zeroing/casts and guard for b/d. Query prepass of bk2 always instantiates STE=false. |
| bk2 KV backward + launcher | `selected || DK_STE` admits unselected ds*q to existing local ak. | false recovers original guard; owner/thread traversal and fixed reduction trees unchanged. Query prepass and saved_rowdot unchanged. |
| bk3 backward + launcher | Only KEY=true coefficient sel receives ds for unselected keys. | false yields original selected?ds:0; query half is the unchanged false specialization. p, ds, bulkds, value coefficients and WMMA order unchanged. |
| TF32 backward + launcher | Same KEY-only coefficient admission. | false restores original coefficient; query half stays false; singleton, probability, ds and dV coefficient arithmetic unchanged. |
| ops switch helper + extension init | Strict process-wide frozen getenv parse; once-only stderr receipt; host dispatch before loops. | Unset/0 selects false; no CUDA work, forward edits or launch-shape changes. Invalid strings fail during import before use. |
| restored deterministic support files | None: exact base source copies missing from delivered worktree. | Hashes equal base build's source pins; no algorithm modification. |

`test_source.py` checks every forward body byte-for-byte and backward operand
bodies after substituting STE=false. Static SASS evidence: 75/78 off and 15/19
forward specializations have identical encodings. The remaining 3/4 differ
only in immediates of MOV/UMOV/LDS/STS address instructions; arithmetic/register
instruction sequence and instruction count match. This is consistent with
changed shared allocation (reasoning), not literal encoding identity. The raw
all-specialization SASS identity diagnostic remains RED. Runtime equal output
bytes require L-G1/L-G2; atomic ordering cannot be proved by source equivalence.

# Prior art
Bengio, Leonard and Courville (2013) STE: identity derivative for detached
quantizer, integrated here only into key accumulation. Dao et al. (2022)
FlashAttention: saved-lse backward recomputation already in this engine;
Dao (2023) FlashAttention-2 owner partitioning already in bk2/bk3. NVIDIA
CUDA/WMMA/Nsight systems: existing kernels and static/timing instrumentation.
Unverified - lead to check these names/years. No new estimator/attention
algorithm claimed; switch integration and evidence harness are this order.
