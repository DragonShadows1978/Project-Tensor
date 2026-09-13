# BP-SCOUT-1 — cheaper backward on this stack

2026-09-12. Read-only scout; recommendations, no implementation or GPU execution.
Dispatch identity: **Codex Astra, `gpt-6-astra`, reasoning high** (as specified in the order; independent launch metadata is not exposed to this scout).

**Verdict:** first measure the current backward, then remove redundant work in the native APA backward and fuse training intermediates. Do not start by “adding checkpointing”: GRAPA already has block checkpointing, a training driver, BF16 activations, and an O(sequence)-storage APA backward. Aggressively dropping tail gradients remains an approximation requiring new evidence. For full 7B training, persistent model state already defeats 12 GB; activation tricks alone cannot make it fit.

**New measurements:** none on GPU; only source inspection, small standard-library scalar calculations, checkpoint header/tail inspection, and historical receipt reading. All proposed experiment thresholds below are registrations for a successor, not passed gates. No performance improvement is claimed delivered. **Not claimed fixed:** any kernel, training quality, memory ceiling, or cross-backend gradient mismatch.

## Evidence and corrections to the starting premise

Evidence labels throughout: **C = code reading**, **R = existing receipt** (historical, not rerun), **D = reasoning/calculation** (unmeasured), **L = literature recollection or local literature attribution** (external verification unavailable). A number derived from a code dimension is D with its C source, not a benchmark. GF = 10^9 FLOPs, TF = 10^12 FLOPs; MiB/GiB use powers of two. FMA counts as two FLOPs. The card and electrical constraint are **order-supplied**, not re-probed.

| Starting claim | What the inspected files actually establish | Class / source |
|---|---|---|
| No training driver or gradient checkpointing | `grapa/train.py` exists and performs backward/AdamW. `MLAConfig200.checkpoint_blocks=True`; each block uses native `tc.checkpoint`. Native binding replays the closure with a seeded VJP. | C: [model configuration and block loop][C1], [training loop][C2], [checkpoint binding][C3] |
| APA training must materialize quadratic attention | The native `refine<1`, causal path saves Q/K/Kq/V and two FP32 row statistics, recomputes probabilities, and returns dense-shaped dQ/dK/dV. It does **not** save an S×S probability or selector matrix. The composed full-attention path still does. | C: [GRAPA dispatch][C4], [APA op closure][C5], [APA backward and buffers][C6], [standard SDPA][C7] |
| “Head set” means selected attention heads | The relevant selector chooses **key positions within each query/head row**. `refine=0.15` is a z-threshold control, not a guarantee that exactly 15% of keys, mass, or heads are selected. Training uses absolute bulk-score statistics; SP running-max is a different selector. | C: [training selector][C8], [SP rule derivation][R5] |
| Low-bit tail means low-bit training buffers/MACs | GRAPA reconstructs quantized keys into the original floating dtype. The native training kernel loads floating arrays into float accumulators; it is not a packed INT4 backward or tensor-core low-bit GEMM. | C: [key reconstruction][C9], [kernel loops][C6] |
| APA has “no output change” | All keys remain in the denominator, but replacing scores can change outputs. The APA overview explicitly records negative/boundary models. Forward parity at an operating point is neither universal equality nor gradient parity. | C/R: [APA scope][R1], [SP1.1 counterexample][R4], [SP5 amendment][R7] |
| Step 1750 checkpoint | The named `STEP1750-copy-rule.ckpt` is a 2,805,440,129-byte protocol-4 pickle whose small header/tail were read without deserializing model arrays. The training log and evaluation's embedded checkpoint metadata both say **1720**, not 1750. | C: [checkpoint serialization][C10]; R: [step-1720 save][R2], [evaluation metadata][R3] |
| Bolt-on distillation is a proven cheap behavioral replacement | Offline adapter training exists, on CPU, with no base backward. E2 achieved 90.714% validation activation-MSE improvement, but behavioral recovery was RED and WikiText PPL rose 16.4795%. E3 found no qualifying install layer. Mechanism exists; general useful substitution is unproven. | C: [adapter training][C11]; R: [E2][R8], [E3][R9] |

The historical status receipt reports: uncheckpointed W=2048 OOM; checkpointed FP32 W=2048 passed two steps, W=4096/8192/12288 one step each, W=16384 backward OOM; BF16 W=20480 passed a one-step smoke and subsequently a two-step loop-fix smoke, W=24576 backward OOM. These are **R, smoke-scale**, not current capacity measurements. The available `grapa_mla_w20480_r015_bf16_train_loopfix.log` contains only a startup line, so the two-step success here is supported by the status narrative, not independently by that log. The original W=20480 failure log explicitly ends `RuntimeError: cudaMallocAsync failed: out of memory` at `log_softmax` on the next step. [Status lines 71–100][R10], [raw failure][R11].

W=12288 training has a stronger historical receipt: steps 1701–1720 report **365.64–404.34 s/step**, **30.39–33.61 tokens/s**, B=1, BF16, refine=.15. This includes model/loss/backward/update and the driver's cleanup/synchronization, not isolated backward; hardware sharing, software drift, and current kernel attribution are unmeasured. Early logs rounded throughput to `0.0k tok/s`; do not read that as zero throughput. [R2], [C2]. The status document's “optimizer state ~3.7GB” is imprecise: approximately 3.72 decimal GB corresponds to **weights + gradients + both Adam moments**, not moments alone (D below).

SP1–SP5 were consulted as inference lineage, not training-speed evidence. SP1.1 reports eligible original-SP1 medians of 1.498× prefill and 0.181× decode versus its baseline, with only 10/24 decode rows eligible; new split-K GPU gates in that delivery were blocked. The split-K construction uses 1/4/16 partitions at S=2048/8192/32768 and proves containment plus partition-selected merge identity, not unchanged global-selector output. SP2's bound concerns score/probability error under a uniform score-error premise, not backward. SP3/SP4G ledgers preserve provenance/precision/rail failures; SP5's expanded evaluation reports A/B/C PPL 232.27/273.47/244.21 and decode 82.58/84.02/84.18 ms/token at S=2048. These do not license a “free tail” or a training speedup. **R:** [SP1.1][R4], [SP2][R5], [SP3][R6], [SP4G][R12], [SP5][R7].

## 1. Cost model table (with evidence classes)

### Geometry and accounting boundaries

**GRAPA (C):** B=1 for this analysis; S=2048; 24 blocks, width d=1024, H=16, score width D=96, value width Dv=64, query rank=768, KV rank=256, SwiGLU width f=1792, vocabulary=8192, tied output embedding. Configuration's formula gives **232,598,016 parameters (D from C)**. Per block: 3,833,856 attention-matrix weights, 5,505,024 MLP weights, 3,136 norm weights. [C1], [MLA projections][C12]. MLA compresses the cached latent; this training forward expands K/V to all heads before attention. A latent-cache size is not a saved-training-activation budget.

**7B comparator (D, explicit hypothetical):** conventional LLaMA-2-7B-like dense MHA geometry, B=1, S=2048, 32 blocks, d=4096, H=32, D=Dv=128, f=11008, vocab=32000, untied input/output embeddings, no linear biases. This arithmetic geometry gives **6,738,415,616 parameters**. No runnable 7B training artifact or 7B measurement was established. Its literature lead is Touvron et al., LLaMA 2 (2023), **unverified — lead to check**. A GQA variant changes K/V projection and storage costs; it does not eliminate per-query-head attention work or MLP costs. The nominal 7.000B state row below is a separate convenient budget, not this exact parameter count.

Let M=BHS² be full rectangular attention pairs and CausalPairs=BH S(S+1)/2. Ordinary composed GEMMs compute the rectangle even under the causal mask. An explicitly triangular implementation approaches half that attention MAC work; MLP/projection costs do not halve. “Backward” below excludes checkpoint replay; training total includes it separately. Transcendentals are reported as calls or rough scalar operations, never converted to alleged GPU cycles.

### Per block FLOPs, plus global embedding/loss

Every numeric entry in this table is **D, unmeasured**. GRAPA geometry is C; 7B geometry is the assumption above. These are algebraic dense-equivalent costs, not measured instructions for the scalar native APA kernel.

| Component | Forward formula | Backward formula / what it computes | GRAPA forward / backward GF | 7B comparator forward / backward GF | Code basis |
|---|---|---|---:|---:|---|
| QKᵀ scores | 2MD | 4MD: dQ=dZ K, dK=dZᵀ Q | 12.885 / 25.770 | 34.360 / 68.719 | [matmul VJP][C13], [C7] |
| Softmax | About 5M scalar ops, including one exp/pair; maxima/reductions extra | About 4M scalar ops for dZ=P⊙(dP−rowsum(P⊙dP)); no D×D Jacobian | 0.336 / 0.268 | 0.671 / 0.537 | [composed softmax][C14]; native formula [C6] |
| PV | 2M Dv | 4M Dv: dP=dO Vᵀ, dV=Pᵀ dO | 8.590 / 17.180 | 34.360 / 68.719 | [C7], [C13] |
| Attention projections | 2BS A, A=sum of projection matrix sizes | 4BS A for input + weight gradients | 15.703 / 31.407 | 274.878 / 549.756 | MLA A above; 7B A=4d². [C12] |
| SwiGLU's three linear maps | 6BSdf | 12BSdf | 22.549 / 45.097 | 554.051 / 1108.102 | [SwiGLU][C15], [C13] |
| SwiGLU pointwise | O(BSf), roughly 6BSf scalar ops including sigmoid/exp | Roughly 10BSf, excluding framework temporaries | 0.022 / 0.037 | 0.135 / 0.225 | [SiLU backward][C16]; operation convention is D |
| All block RMSNorms | Approx. 4E scalar ops + O(rows) rsqrt | Approx. 10E scalar ops; explicit chain may do more | 0.030 / 0.074 | 0.067 / 0.168 | E=BS·3616 for MLA; E=2BSd for 7B. [C12], [RMSNorm][C17] |
| Input embedding lookup, once/model | Copies BSd elements, not a dense GEMM | BSd scatter additions + zeroing Vocab·d buffer | 0 GEMM / 0.0021 additions | 0 GEMM / 0.0084 additions | Dense gradient buffer [C18] |
| Output vocabulary projection, once/model | 2BSd·Vocab | 4BSd·Vocab | 34.360 / 68.719 | 536.871 / 1073.742 | GRAPA tied map [C1], [C13]; 7B assumed untied |
| Final RMSNorm, once/model | Approx. 4BSd | Approx. 10BSd | 0.0084 / 0.0210 | 0.0336 / 0.0839 | [C1], [C17] |
| Vocabulary log-softmax + gather loss, once/model | O(BS·Vocab), about 5 operations/element incl. exp | O(BS·Vocab), about 4/element in a fused analytical VJP | 0.084 / 0.067 | 0.328 / 0.262 | Actual composition and FP32 cast [C19], [C20]; these counts are not a census of temporary ops |

Dense GEMM-only model totals, including the output head: **GRAPA 1.468 TF forward + 2.936 TF backward = 4.403 TF/step**; **7B 29.262 + 58.523 = 87.785 TF/step** (D). All-block checkpointing adds one block-body forward: **1.433 TF** and **28.725 TF** respectively, giving **5.837 TF** and **116.510 TF**, before scalar ops, optimizer, quantization, allocator/copy work. These are rectangular full-attention comparators, not the current refine=.15 execution.

At S=2048, rectangular attention-core backward is approximately **36.0%** of GRAPA block GEMM-backward FLOPs and **7.7%** of the 7B comparator's (D). Even deleting that whole core would leave projections/MLP/head; tail selection cannot touch all backward FLOPs. Wall shares can be much larger for unfriendly scalar kernels than these arithmetic shares.

### What native APA actually computes

For native training let r be the **measured selected-pair fraction**, which is unavailable here. The following leading-MAC formulas use the causal pair count. They exclude scalar scale/threshold operations, exp/log, reductions, memory traffic, atomics overhead and some online-normalizer work. They count what the loops do, not a tensor-core performance equivalent. [C6], [C8].

| Native APA component | Leading work (D from C) | S=2048 MLA illustration with assumed r=.15 (D, not a measured fraction) |
|---|---|---:|
| Forward threshold statistics + mixed scores + weighted values | CausalPairs·[(4+2r)D+2Dv] | 18.155 GF |
| Additional online numerator rescaling | Approximately 2·CausalPairs·Dv; reductions/normalizer scalars still extra | 4.297 GF |
| Backward pass A | 2·CausalPairs·[(1+r)D+Dv], to rebuild scores/probabilities and sum P(dO·V) | 11.710 GF |
| Backward pass B | Score rebuild + dO·V + dV for every key, dQ through K or Kq for every key, dK only selected | 23.419 GF |
| Backward total A+B | CausalPairs·[6(1+r)D+6Dv] | 35.129 GF |
| Detached key quantization | Two dense D×D rotations per head: about 4BHS D², plus norm/search/reconstruction | 1.208 GF/quantization |

The backward has **O(S) global tensor storage but O(S²) arithmetic**. The r=.15 example is not a 6.67× backward saving: dQ/dV, probabilities and row coupling remain dense in key support; Kq is floating. A block-checkpointed current-path leading-MAC subtotal is approximately **5.548 TF/step**, plus approximately **0.206 TF** of the listed online rescaling over initial/replay forwards and remaining overhead (D). This is not directly comparable to a hardware FLOP counter or wall measurement.

A code-specific opportunity follows: in real arithmetic, `rowdot = dot(dO, O)` with O=PV. Retain O and avoid pass A, removing about **one third** of the leading backward work in the illustration (D), for at most a 4 MiB BF16 or 8 MiB FP32 output reference per active MLA block. The output is already needed downstream, but retaining a reference can extend its lifetime. BF16 rounding and current forward/backward dot-order differences mean this is not a promise of bit identity; see ranked item 1.

### Activation memory and where peak can live

These are **D byte inventories of named tensors**, not measured peak or disjoint totals. Views/share references must be deduplicated by storage; allocator reserve, cast copies, replay lifetime, per-op gradients, atomics buffers and temporaries can overlap. BF16=2 bytes/element, FP32=4. B=1, S=2048.

| Tensor group / scaling | GRAPA | 7B assumed geometry | Materialized today? / evidence |
|---|---:|---:|---|
| One residual boundary BSd, BF16 | 4 MiB/block | 16 MiB/block | Yes; checkpoint inputs retain storage. [C3] |
| N checkpoint block inputs, BF16 | 96 MiB; boundary output(s) extra | 512 MiB; output(s) extra | GRAPA yes; 7B only hypothetical port. Avoid double counting shared adjacent outputs. |
| Q, K, V each, BF16 | 6 / 6 / 4 MiB | 16 / 16 / 16 MiB | Yes, expanded MLA Q/K/V [C12]. Extra copies from cat/permute can occur. |
| Native APA saved Q+K+Kq+V + LSE/threshold | 22.25 MiB/block BF16; 44.25 MiB FP32 | 64.50 MiB/block BF16; 128.50 MiB FP32 if using analogous APA | Yes native selective; stats are 0.25/0.50 MiB FP32. Storage references, not necessarily new copies. [C5], [C6] |
| Native APA output / incoming dO, each BF16 | 4 MiB | 16 MiB | Yes. Saved-output proposal can extend output lifetime. |
| Native APA dQ+dK+dV, BF16 | 16 MiB | 48 MiB | Full-shaped buffers, even though dK has selectively accumulated contributions. FP32 doubles these. [C6] |
| ONE rectangular attention matrix BHS², BF16 / FP32 | 128 / 256 MiB | 256 / 512 MiB | Composed SDPA: scores, masked scores, shifted scores, exponential and probability can each occupy this size. Native selective: none. [C7], [C14] |
| Five such forward matrices, rough composed chain inventory | 640 MiB/block BF16; 1.25 GiB FP32 | 1.25 GiB/block BF16; 2.5 GiB FP32 | D inventory, not exact peak. If retained across all blocks: 15/30 GiB for GRAPA and 40/80 GiB for 7B, before anything else. Mask cache/reduction rows extra. Checkpoint replay limits retained blocks, not the size of one block. |
| Causal mask S², BF16 / FP32 | 8 / 16 MiB | 8 / 16 MiB | Cached and shared across heads; do not multiply by H or N. [C7] |
| SwiGLU gate/up/SiLU/product, four BSf BF16 arrays | 28 MiB/block | 172 MiB/block | Yes in composed forward, plus input/output/residual references and backward scratch. [C15], [C16] |
| Norm input-sized buffers | One buffer across block's norm calls: 14.125 MiB BF16; three illustrative buffers 42.375 MiB | One 32 MiB; three 96 MiB | Composed RMSNorm retains x, x*x and scaled intermediates; overlap with other rows means this is not additive. Row reductions extra. [C17] |
| Input embedding output, FP32 then optional BF16 | 8 then 4 MiB | 32 then 16 MiB if same policy | GRAPA casts after lookup; output may overlap first residual boundary. [C1], [C18] |
| One output-logit matrix, BF16 / FP32 | 32 / 64 MiB | 125 / 250 MiB | GRAPA computes all logits, then casts for loss. [C1], [C19] |
| Four full-vocab FP32 loss arrays (cast, shifted, exp, log-prob), rough inventory | 256 MiB/model, BF16 original extra | 1000 MiB/model, BF16 original extra | Composed loss retains several such buffers; backward adds full-shaped arrays. This is an inventory, not a measured count of simultaneous unique allocations. [C19], [C20] |
| Dense input-embedding gradient scatter buffer, FP32 | 32 MiB | 500 MiB | Yes for trainable lookup; zeroed even when few token ids occur. GRAPA tied output map also contributes a dense gradient, so sparse lookup alone cannot make the shared parameter gradient sparse. [C18] |

The active graph is native C++ autograd, not the legacy Python engine. `Tensor::from_op` keeps parent variables and closure captures; backward clears each consumed **op gradient** but does not clear all saved forward data/closures. Leaf gradients remain for optimization. The source explicitly describes the consumed-gradient fix. Therefore “free op gradients after use” is already implemented; a new proposal must target saved-data lifetime or optimizer timing. [C21]. The per-step GRAPA loop already clears grads before forward and after update, deletes logits/loss/batches, then clears cache. [C2].

| Persistent state | GRAPA 232,598,016 params | Nominal 7.000B params | Class / implication |
|---|---:|---:|---|
| FP32 parameters | 0.8665 GiB | 26.077 GiB | D from 4P; C: parameters remain FP32 |
| FP32 leaf gradients | 0.8665 GiB | 26.077 GiB | D; C: casts return gradients to source dtype [C20] |
| Adam m+v, FP32 | 1.7330 GiB | 52.154 GiB | D from 8P; C: two same-dtype buffers per param [C22] |
| Above combined | **3.4660 GiB** | **104.308 GiB** | D; excludes activation/workspace/reserve. Exact 6.738B geometry: **100.410 GiB**. |
| BF16 weight representation alone | 0.4332 GiB | 13.039 GiB | D; not an extra FP32 master. Exact 6.738B geometry: **12.551 GiB**, already exceeds 12 GiB. |
| 8-bit m+v replacing FP32 moments | About 0.4332 GiB, saving 1.2997 GiB | About 13.039 GiB, saving 39.116 GiB | D ideal payload, excludes scales/outlier exceptions; no native implementation established. Still does not fit full 7B. |
| 4-bit frozen nominal-7B weights | — | 3.260 GiB payload | D before scales/metadata/workspaces. This is a PEFT/offload route, not full FP32 Adam training. |

**Peak interpretation:** historically GRAPA was activation/lifetime-bound, and checkpointing plus BF16 moved its ceiling (R). At current S=2048, no measured allocation census identifies whether the high-water instant is a replayed block, loss backward, dense leaf-gradient accumulation or allocator retention. Native APA eliminates the quadratic saved matrix; it does not eliminate expanded projections, FP32 loss buffers, MLP/norm intermediates or optimizer state. At S=12288/20480, linear activation/loss inventories multiply by 6/10, while attention arithmetic multiplies by approximately 36/100 (D). The recorded late W=20480 ~10.34 GB process/~10.57 GB board readings are narrative R, not a synchronized allocator peak. For 7B, **persistent state is the decisive full-training blocker before activations**.

### Engine boundaries that must not be conflated

`tensor.py:16` is NumPy/FP64 with eagerly allocated grads; `tensor_gpu_v2/_core.py:16` imports CuPy; `test_autograd_regressions_phase6.py:3,19–23` targets that CuPy engine. Reading these establishes prior autograd coverage, not a current native CUDA training gate. No module was imported or test executed here.

The separate PyTorch `apa_cuda` path saves Q/K/V/Kq and a refine mask ([wrapper][C23]). Its backward rebuilds **full-precision QK softmax**, splits dQ with the mask, but computes dK using unmasked dS. This differs from native mixed-score probabilities and selected-only dK; tiling only bounds each backward workspace, and the saved mask can remain quadratic. [C24]. It must not serve as the native APA gradient oracle without reconciling the intended objective. No bug fix or full mathematical diagnosis is claimed from this scout.

## 2. Ranked idea ledger — Prior art

Ranking is an engineering recommendation (**D**), conditional on the first measurement. Exact-work removal precedes approximate gradients for continued GRAPA training; the rank-64 route moves to first place if the real task is adding a narrow capability to a frozen base. None of these is a claim of novel backpropagation.

**Registration common to every proposed experiment:** future work only; no cells ran in this order. Use one exclusively leased RTX 4070 SUPER, no overlap with the current campaign, no second card. Refuse an unavailable lease and report deferred; do not signal another process. Each cell is foreground, planned for at most 480 seconds with admission/cleanup below 10 minutes; larger budgets below mean separate short leases, not sustained multi-hour occupancy. Use cooperative between-iteration budget checks, not kill/timeout commands under this order's no-signal rule. If the first bounded cell cannot finish the registered sample count, report INCONCLUSIVE and stop; do not weaken thresholds, shorten sequences or extrapolate a pass. GPU-hour figures below are **D upper budgets**, not runtimes or measurements, and do not include code development. No independent blind validation is claimed; lead dispatch owns it.

Freeze source/build identity, checkpoint, tokenizer, token arrays, dtype, refinement rule, seed, loss weighting and sample ids before any successor gate. For exact implementation comparisons use interleaved same-input A/B, separate diagnostics from timing, and measure peak live and reserved bytes separately. Report optimizer-inclusive time and tokens-to-quality, not just kernel GFLOPs. For approximate-training entries, “Q32” below means 32 matched updates at S=512 on the actual 232.6M model (three paired seeds) plus disjoint fixed held-out samples from **each** of copy/arithmetic/brackets/reverse/sort/pattern; hard fail for nonfinite values or >1% relative PPL increase in any family against the matched baseline. This is a cheap rejection screen, not a convergence/generalization certificate. Failure to complete within its budget is inconclusive. Existing step-1720 family regressions make a copy-only acceptance gate particularly inappropriate. [R3], [R13].

All external paper identifications below are **L: unverified — lead to check**; search terms are given explicitly. Local system behavior is C/R at the linked sites. The “site comment” is the proposed attribution text to place at the attachment point in a successor. **No actual code-site comments were edited**: doing so would violate this order's read-only constraint. These recommended comments, this ledger, and this report's Prior art section preserve the three-way attribution intent without pretending implementation occurred.

### 1. Replace backward pass A with the output-dot identity

- **Mechanism/savings:** compute D_i=dot(dO_i,O_i) rather than rescanning every key to compute sum_j P_ij(dO_i·V_j). Keep all tail gradients and the original selector. Saves score reconstruction, exp, V-dot reads and one global key traversal; approximately 11.710 GF/block in the illustrative S=2048 MLA backward. Extra retained output is linear, not quadratic. Expected wall gain is unmeasured.
- **Cost/failure:** stored BF16 O and different reduction orders can change D and gradients; native forward uses warp-cooperative dots while backward uses serial per-thread dots, so selector/probability reconstruction is already a finite-precision seam. Start FP32, then test BF16 separately. A FP32 output side buffer may be needed; that changes the memory tradeoff. Do not call the BF16 substitution exact-bitwise.
- **Prior art / ours:** Dao et al., **FlashAttention (2022)** uses the output-dot softmax derivative identity and recomputation. Take that algebra; our work would be adapting it to the existing mixed-score MLA op and preserving its selector/dtype contract. Search: `FlashAttention backward Di sum dO O 2022`.
- **Cheapest falsifier:** one real GRAPA attention layer, B=1, S=256 then 2048, H=16, D=96, Dv=64, refine=.15. Freeze captured Q/K/Kq/V/dO; 5 warmups, 20 interleaved replays. Predict ≥15% backward-kernel time reduction and ≤8 MiB extra peak at S=2048. Reject if either fails, FP32 any dQ/dK/dV relative L2 error exceeds 1e-5, or BF16 exceeds 1e-2; also report max-absolute error and selector mismatches without averaging away boundary cases. Budget **0.15 GPU-h**. No training-quality claim from replay.
- **Attach / site comment:** [kernels.cu:2558][C6] and [ops.cpp:740][C5]. `// Prior art: Dao et al., FlashAttention (2022), output-dot backward row term; adapt to native APA mixed scores. External reference unverified — lead to check.`

### 2. Tile training attention and provide an exact full-attention streaming backward

- **Mechanism/savings:** native selective already has recomputed softmax, but uses per-query blocks and dK/dV atomics, without GEMM-like cooperative backward tiling. Build a tiled gradient accumulation schedule; add a separately validated full-attention streaming path for refine=1. Saves quadratic saved activations for the full-attention twin; mainly targets memory traffic/atomics and wall time for APA. Recomputing softmax trades added FLOPs for fewer HBM reads/writes. No “first FlashAttention on GRAPA” claim: part of its principle is already present.
- **Cost/failure:** substantial CUDA implementation; reduction workspaces can cancel memory savings. MLA D=96/Dv=64, BF16 atomics, causal bounds and GQA sharing must be explicit. SP split-K **decode** results do not establish training partition efficiency; changing selector partitioning can change the forward objective.
- **Prior art / ours:** Dao et al., FlashAttention (2022); Dao, **FlashAttention-2 (2023)**; Milakov and Gimelshein, **Online normalizer calculation for softmax (2018)**. Take IO-aware tiling, stable online normalization and recomputed backward; ours would be native C++ integration with MLA/mixed-score semantics. Search those titles plus `FlashAttention backward work partitioning`.
- **Cheapest falsifier:** one GRAPA MLA layer S=512/2048, causal FP32 then BF16. Compare dense refine=1 reference separately from native refine=.15 baseline. Predict full-attention peak scratch drops ≥75% at S=2048; APA backward ≥1.2× faster with ≤10% peak increase. Require FP32 gradient relL2≤1e-5/BF16≤1e-2 in every output gradient. A failed branch stays failed; do not substitute APA results for full attention. **0.25 GPU-h**.
- **Attach / site comment:** [kernels.cu:2524][C6], [functional.py:81][C7], [ops.cpp:734][C5]. PyTorch comparison seam [apa.cpp:360][C24], only after semantic reconciliation. `// Prior art: FlashAttention (Dao et al. 2022), FA2 (Dao 2023), online normalizer (Milakov/Gimelshein 2018); native MLA/APA backward integration, not a new attention algorithm. Unverified — lead to check.`

### 3. Do not compute gradients for frozen operands

- **Mechanism/savings:** native matmul's closure unconditionally launches both operand VJPs, and `accumulate_grad` has no requires-grad guard. When x is trainable but W frozen, compute dX only. When an input is frozen and only W trainable, compute dW only. This removes an actual GEMM and unwanted buffer for those cases. It saves nothing for two trainable operands in full GRAPA training. Do not skip dX merely because W is frozen: earlier adapters still need it.
- **Cost/failure:** small engine change, broad semantic impact; audit other op closures too, but do not assume all can be changed identically. Gradients for tied parameters, broadcasting and accumulation must remain correct.
- **Prior art / ours:** **PyTorch autograd (Paszke et al., 2017/2019)** and reverse-mode activity analysis; take requires-grad/needed-gradient pruning. Ours is enforcing the existing native contract at this code site. Search: `PyTorch autograd needs_input_grad Paszke 2017 automatic differentiation`.
- **Cheapest falsifier:** a 1024×1792 GRAPA MLP linear, B=1, S=2048, all four input/weight activity combinations, same seed. Predict exactly one GEMM disappears in each one-active-operand case, inactive `.grad` stays absent, active gradient matches FP32 at relL2≤1e-6, and that isolated backward is ≥25% faster. All-trainable case must be unchanged within timing noise (≤5% median regression). **0.05 GPU-h**.
- **Attach / site comment:** [ops.cpp:624][C13], [autograd.cpp:14][C21]. `// Prior art: PyTorch autograd needed-gradient pruning (Paszke et al. 2017/2019); honor native requires_grad before each VJP GEMM. Unverified — lead to check.`

### 4. Fuse vocabulary loss, RMSNorm and SwiGLU training intermediates

- **Mechanism/savings:** start with gather-NLL fused log-softmax backward, preserving weighted-target normalization; next chunk the vocabulary projection/loss to avoid all-token logits; separately fuse RMSNorm and SiLU×up VJPs. Saves memory traffic, allocations and saved intermediates, mostly wall/memory rather than large GEMM FLOPs. The existing inference-only RMSNorm/causal-softmax functions cannot be enabled during training without a real backward.
- **Cost/failure:** weighted loss, tied embedding accumulation and FP32 loss math are load-bearing. Chunked loss may reconstruct logits and add compute; BF16 normalization can shift numerics. Keep these as separately falsifiable steps, not an opaque bundle.
- **Prior art / ours:** Zhang and Sennrich, **RMSNorm (2019)**; Shazeer, **GLU Variants Improve Transformer (2020)**; LinkedIn's **Liger Kernel (2024)** system; **Cut Your Losses in Large-Vocabulary Language Models (2024/2025)** (paper author list and publication year remain leads to verify). Take fused training kernels and materialization avoidance; ours is native weighted-NLL/tied-head plumbing. Search the titles with `fused linear cross entropy`.
- **Cheapest falsifier:** GRAPA B=1 S=2048 vocab=8192 loss replay, both uniform and recorded answer weights. First fused-loss variant must save ≥128 MiB peak and improve loss forward+backward ≥20%, FP32 loss abs error≤1e-5 and logits-grad relL2≤1e-5. If it cannot meet this without projection chunking, record the first variant failed. A separate one-block norm/MLP replay predicts ≥25% pointwise temporary-byte reduction at the same numerical limits. **0.20 GPU-h** total, no combined speed claim until full-step measurement.
- **Attach / site comment:** [ops.cpp:999][C20], [ops.cpp:435][C25], [ops.cpp:133][C16], [nn.py:286][C17]; caller [grapa/loss.py:19][C19]. `// Prior art: RMSNorm (Zhang/Sennrich 2019), GLU variants (Shazeer 2020), Liger (2024), Cut Your Losses (2024/25); preserve native weighted loss and tied gradients. Bibliography unverified — lead to check.`

### 5. Selective checkpointing, including attention-only and MLP-only variants

- **Mechanism/savings:** tune the already-built all-block policy. At short sequences, save expensive attention/projection results and replay cheap pointwise subgraphs; compare attention-only, MLP-only, whole-block and none. Attention-only replay of an already-streaming attention core can be a poor trade, since its saved data are linear. MLP-only checkpointing targets the four wide SwiGLU arrays. Whole-block replay remains a robust long-context option.
- **Cost/failure:** checkpointing saves bytes by **adding** recompute FLOPs/wall. A saved-GEMM policy is not automatically optimal once casts, lifetimes and CUDA kernels are included. Native `checkpoint_py` does not capture RNG state; stochastic functions require RNG-preserving semantics. It also decides whether to checkpoint from its explicit inputs, so a frozen-prefix output with only captured trainable parameters needs careful handling.
- **Prior art / ours:** Chen et al., **Training Deep Nets with Sublinear Memory Cost (2016)**; Griewank and Walther, **Revolve (2000)**; Jain et al., **Checkmate (2020)**. Take rematerialization/selective schedules; ours is a measured policy for this fixed 24-block graph. Search titles plus `selective activation checkpointing save matmul`.
- **Cheapest falsifier:** GRAPA S=512 then 2048, BF16, fixed refinement and token batch, five same-weight gradient evaluations/policy. Predict at least one selective policy gives ≥10% optimizer-inclusive speed improvement over current whole-block policy while peak stays ≤10.5 GiB and per-parameter gradient relL2≤1e-2. If none do, keep current policy. **0.35 GPU-h**. This does not re-certify W=12288.
- **Attach / site comment:** [bindings.cpp:66][C3], [model_mla.py:100][C1]. `// Prior art: Chen et al. 2016 rematerialization, Revolve 2000, Checkmate 2020; choose measured native GRAPA checkpoint boundaries. Unverified — lead to check.`

### 6. Keep mixed activations; quantize optimizer moments if memory is still binding

- **Mechanism/savings:** BF16 activations with FP32 leaves/moments already exist. Add blockwise 8-bit Adam states with scales and high-precision exceptions; ideal GRAPA moment saving is ~1.30 GiB. Audit transient BF16 casts rather than assuming every weight needs a permanent duplicate. A single-moment optimizer is another memory option, but native Lion already exists and changes optimization behavior.
- **Cost/failure:** moment quantization/dequantization and small-tensor exceptions; small gradients/outliers can underflow. BF16 activations do not imply FP32 GEMM accumulation can be discarded. State/checkpoint serialization must change with the optimizer. This mainly attacks persistent bytes, not backward FLOPs.
- **Prior art / ours:** Micikevicius et al., **Mixed Precision Training (2017/2018)**; Dettmers et al., **8-bit Optimizers via Block-wise Quantization (2021/2022)**; Chen et al., **Symbolic Discovery of Optimization Algorithms / Lion (2023)** for the optional already-present alternative. Take precision/state formats; ours is a native implementation and resume-compatible receipt. Search these titles.
- **Cheapest falsifier:** Q32 with GRAPA S=512, 8-bit moments versus FP32 AdamW. Predict ≥1.0 GiB persistent reduction once all states exist and ≤10% optimizer-inclusive time regression; require Q32 quality and correct state round-trip. Reject moment bias/overflow even if peak improves. **0.75 GPU-h**. Lion is a separate comparison, not an excuse to change Adam hyperparameters mid-gate.
- **Attach / site comment:** [optim.py:51][C22], native Adam step [kernels.cu:4219][C27] and its kernel at line 875; GRAPA state adapter [C10]. `// Prior art: Micikevicius et al. 2017/18 mixed precision; Dettmers et al. 2021/22 blockwise 8-bit moments; native state/resume adaptation. Unverified — lead to check.`

### 7. Distill a rank-64 bolt-on from captured activation pairs

- **Mechanism/savings:** pay teacher/student **forward** capture once, freeze the base completely, train the small adapter offline. Existing implementation is CPU `SiLU(h Aᵀ) Bᵀ`, not full-model fine-tuning. For d=4096, r=64, two matrices total 524,288 parameters and ~8 MiB of FP32 parameters+grads+Adam (D), before activation batches. Most base backward and optimizer state disappear by changing the training problem.
- **Cost/failure:** capture time/storage and replay distribution shift; representational MSE does not establish usable behavior. This cannot replace training GRAPA from birth or teaching knowledge unsupported by teacher signal. E2/E3 negatives are the relevant local failure cases.
- **Prior art / ours:** Hinton, Vinyals and Dean, **Distilling the Knowledge in a Neural Network (2015)**; Romero et al., **FitNets (2014/2015)**; Houlsby et al., **Parameter-Efficient Transfer Learning for NLP (2019)**; local **GraftRepository MOE-E1/E2 (2026)**. Take teacher/hidden-state matching and bottleneck adapters; ours would be new task/data and validated routing on this stack, not a new distillation method. Search those titles. Local lineage verified [C11], [R8].
- **Cheapest falsifier:** first reuse existing E2 rank-64 validation/captured-pair receipts, **0 GPU-h**: the proposition “low MSE already proves useful cheap learning” is already false in this receipt. For a new teacher, capture four disjoint 512-token windows on GRAPA-232M and replay a CPU adapter, then a fixed held-out forward probe: register ≥10% held-out NLL-gap recovery and ≤0.5% generic PPL increase, with a positive teacher gap as precondition. **0.15 GPU-h**, explicitly pilot-level and inconclusive if teacher gap absent. Full E2-style behavior gates are still owed.
- **Attach / site comment:** native adapter maps [nn.py:97][C26], frozen-base `no_grad` / activity guards [C21], local trainer [C11]. `// Prior art: Hinton et al. 2015, FitNets 2014/15, Houlsby et al. 2019, local MOE-E1/E2 2026; offline adapter fit from frozen-base captures. External leads unverified.`

### 8. LoRA/QLoRA for a frozen 7B base, when the objective allows it

- **Mechanism/savings:** train low-rank weight deltas with a frozen quantized base; reduce trainable gradients/moments and base weight residency. Unlike offline distillation, activations and dX through the base are still needed for internal adapters. Nominal 7B 4-bit payload is 3.260 GiB before metadata; this makes a 12 GB trial plausible, not guaranteed at S=2048.
- **Cost/failure:** capacity restriction, dequantization overhead, activation footprint and trainable adapters throughout the graph. Need rank coverage, quantized-linear dX support and item 3. Freezing all weights is insufficient if generic matmul still builds their gradients.
- **Prior art / ours:** Hu et al., **LoRA (2021/2022)**; Dettmers et al., **QLoRA (2023)**. Take frozen low-bit base plus low-rank trainable deltas; ours would be TensorCUDA integration and corpus-specific validation. Search those titles.
- **Cheapest falsifier:** first a GRAPA-sized frozen block with rank-64 projections, S=2048: require trainable-only moment/gradient allocation, correct dX at relL2≤1e-2 against dequantized BF16 reference, and ≥50% optimizer-state savings. **0.10 GPU-h**. Only after a separate local 7B artifact is pinned, a one-step 7B S=2048 B=1 memory admission gate predicts ≤10.5 GiB peak; **0.15 GPU-h**, currently artifact-dependent. Neither is a full-fine-tuning quality equivalence test.
- **Attach / site comment:** [ops.cpp:635][C13] existing quantized-linear VJP area, [nn.py:97][C26], [optim.py:18][C22]. `// Prior art: LoRA (Hu et al. 2021/22), QLoRA (Dettmers et al. 2023); native frozen quantized linear plus adapter gradients. Unverified — lead to check.`

### 9. Quantized saved activations, starting with wide MLP intermediates

- **Mechanism/savings:** store selected backward-needed activations at 8 bits, reconstruct only when consumed; blockwise scales and outlier exceptions. Ideal BF16→INT8 payload halves those saved buffers, not total memory. Target SiLU inputs/product or checkpoint boundaries, not logits/loss first. Compressing the 96 MiB checkpoint boundary inventory alone saves at most ~48 MiB before overhead at S=2048 (D).
- **Cost/failure:** quantization error in dW, especially outlier/rare-token activations; bandwidth/code overhead may exceed benefit with all-block checkpointing already enabled. Unbiased activation compression need not yield unbiased nonlinear/optimizer trajectories. Quantized Kq in APA is not a general activation-compression implementation.
- **Prior art / ours:** Chen et al., **ActNN (2021)**; Chakrabarti and Moseley, **Backprop with Approximate Activations for Memory-efficient Network Training (2019)**. Take compressed saved tensors and precision allocation; ours is per-op native GRAPA calibration/packing. Search titles plus `ActNN activation compression`.
- **Cheapest falsifier:** one replayed GRAPA MLP at S=2048, followed only if passed by Q32. Predict ≥35% reduction in the targeted saved-buffer bytes, no >10% time regression, initial dW relL2≤1e-2, then Q32. **0.50 GPU-h**. Reject if whole-step peak barely changes and memory was the sole motive; report that separately from successful payload compression.
- **Attach / site comment:** [ops.cpp:624][C13] operand captures, [ops.cpp:135][C16], [bindings.cpp:83][C3]. `// Prior art: ActNN (Chen et al. 2021), approximate activations (Chakrabarti/Moseley 2019); compress selected native saved tensors. Unverified — lead to check.`

### 10. Shorten saved-graph lifetime; cautiously stream optimizer updates

- **Mechanism/savings:** introduce an explicit destructive one-backward mode that releases consumed closure/data ownership where safe; separately evaluate parameter-ready optimizer updates to avoid retaining all leaf gradients until backward ends. Native op-gradient release already exists. Ideal maximum leaf-gradient saving is ~0.8665 GiB on GRAPA, not the whole 3.466 GiB state budget.
- **Cost/failure:** current engine supports repeated backward; clearing graph state changes that contract. Tied embedding receives contributions at both ends of the graph, and checkpoint replay may read a parameter again; updating weights early can make later gradients wrong. Global clipping and gradient accumulation are incompatible with naive per-leaf step hooks. Establish last-use readiness, not just first arrival.
- **Prior art / ours:** conventional reverse-mode liveness in **PyTorch autograd (2017/2019)**; **LOMO, Full Parameter Fine-tuning for Large Language Models with Limited Resources (Lv et al., 2023)**. Take graph lifetime management and gradient/update fusion; ours is a correct readiness contract for this engine's replay/ties. Search titles plus `optimizer step in backward tied weights checkpointing`.
- **Cheapest falsifier:** 2-block MLA with tied head and checkpointing, S=256, two identical optimizer steps against existing AdamW. Predict ≥25% leaf-gradient peak reduction in the streamed variant and FP32 parameter-update relL2≤1e-6; include shared/tied parameters, accumulated two-microbatch loss, and repeated-backward rejection or explicit supported semantics. **0.10 GPU-h**. Full 232M pilot only after this gate; no claim of total-memory reduction from this toy graph.
- **Attach / site comment:** [autograd.cpp:74][C21], [bindings.cpp:104][C3], [optim.py:54][C22]. `// Prior art: standard autograd liveness (PyTorch 2017/19), LOMO (Lv et al. 2023); native last-use-safe gradient/update lifetime, respecting ties and replay. Unverified — lead to check.`

### 11. GaLore-class low-rank optimizer-space gradients

- **Mechanism/savings:** project matrix gradients into a small basis and keep optimizer moments there, lift updates back to full parameter space. Savings target optimizer state, sometimes gradient storage with additional integration. Ordinary GaLore does **not** remove the dense dW GEMM or all full-gradient materialization. MLA's low-rank forward factorization does not prove its training gradients are low rank.
- **Cost/failure:** projection/SVD cost, stale basis, accumulated approximation and sensitivity to refresh/rank. Reducing trainable parameter rank (LoRA) and projecting full-parameter gradients are different methods.
- **Prior art / ours:** Zhao et al., **GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection (2024)**. Take projected moment/update state; ours would be native basis-refresh scheduling and GRAPA evidence. Search exact title.
- **Cheapest falsifier:** Q32 on GRAPA S=512, fixed rank=64 and refresh interval=8 for matrix weights (norms retain standard Adam). Predict ≥50% moment-byte reduction and ≤20% whole-step slowdown; require Q32. Count full dW and basis workspaces at peak. **0.75 GPU-h**. This short screen cannot certify long-run pretraining equivalence.
- **Attach / site comment:** [optim.py:51][C22], [ops.cpp:629][C13]. `// Prior art: GaLore (Zhao et al. 2024), gradient-subspace optimizer state; native matrix-shape/basis-refresh integration. Unverified — lead to check.`

### 12. Structured sparse updates / backward support selection

- **Mechanism/savings:** freeze or periodically update pre-registered channel/block groups, with an error-feedback residual if compressing updates. Optimizer work/state can shrink. Elementwise top-k after dense backward mostly saves update/communication traffic; on this single card it need not save a GEMM or wall time. Actual backward FLOP savings require structured support propagated into compute kernels, and may still require dense dX.
- **Cost/failure:** selection cost, residual memory, unsupported sparse matmul shapes, rare-feature starvation and delayed updates. Random or magnitude sparsity is not guaranteed to match hardware's structured sparsity modes. Tied output embeddings make sparse input-token updates alone ineffective for full-model training.
- **Prior art / ours:** Lin et al., **Deep Gradient Compression (2017/2018)**; Karimireddy et al., **Error Feedback Fixes SignSGD and other Gradient Compression Schemes (2019)**; Evci et al., **RigL (2020)** for dynamic sparse connectivity (a different intervention). Take sparse/error-feedback mechanisms with their actual scope; ours is native block-support experiments, not a claim that communication compression cuts backprop. Search titles.
- **Cheapest falsifier:** GRAPA S=512 Q32 with fixed 25% MLP output-channel update blocks, rotating coverage every four steps; compare to full update and charge residual state. Predict ≥10% full-step speed improvement and Q32 quality including rare-copy strings. **0.75 GPU-h**. If only update traffic shrinks, record a memory/traffic result and reject the FLOP/wall hypothesis.
- **Attach / site comment:** [ops.cpp:629][C13], [optim.py:54][C22], [embedding gradient][C18]. `// Prior art: DGC (Lin et al. 2017/18), error feedback (Karimireddy et al. 2019), RigL (Evci et al. 2020); hardware-compatible native update support, not generic top-k speedup. Unverified — lead to check.`

### 13. APA-keyed lower-precision or dropped tail backward — research candidate

- **Mechanism/savings:** retain selected keys at the current precision; try lower-precision tail dQ/dV products or omit them. Native dK is **already selected-only** because Kq is detached. Further dQ/dV omission changes the gradient, not just storage. Sparse contribution support still needs dense-shaped destination gradients unless the rest of the model changes. APA makes a selector available; it does not make the error free.
- **Cost/failure:** softmax coupling, cancellation in total gradients, rare-key learning, varying selected fraction, and irregular work that fails to accelerate. Full details and an adversarial witness are below. Do not conflate reducing arithmetic precision with deleting terms.
- **Prior art / ours:** local **GRAPA stop-gradient APA (2026)** already supplies selected dK; **BLASST (Yuan et al., 2025/2026; arXiv 2512.12087)** supplies a running-max omission comparator in a different block-sparse setting; **ThriftAttention (Sharratt et al., 2026, local SP-ledger attribution)** is a weight-sensitive precision lead. Take existing selector/precision motivation only. This exact dQ/dV-tail adaptation has **no prior art known to me beyond those leads**; that is not a novelty claim. Search: `BLASST running maximum backward gradients`, `ThriftAttention gradient selective precision`, `attention gradient sparsification tail softmax`.
- **Cheapest falsifier:** CPU analytical two-key witness below, **0 GPU-h**, already rejects “tails vanish” as a universal argument. For empirical promise: one GRAPA layer, S=256/2048, real dO from fixed held-out batches; decompose tail dQ/dV without changing forward, record selected fraction, omitted mass, norm error and per-row extremes. Predict ≥25% of backward arithmetic is avoidable while every layer's gradient relL2≤1e-2 and nonzero-gradient cosine≥.999. Failure rejects this operating point. Only then separate tail-drop and tail-low-precision kernels with ≥15% backward speed threshold and Q32 quality. **0.75 GPU-h** total. No sparse-speed claim from a dense zero-mask implementation.
- **Attach / site comment:** [kernels.cu:2594–2609][C6], [ops.cpp:744][C5]; **do not transplant** [apa.cpp:369–382][C24] without resolving semantics. `// Prior art: local GRAPA APA 2026 selected dK; BLASST (Yuan et al. 2025/26) comparator and ThriftAttention (2026) precision leads. New hypothesis is approximate tail dQ/dV; no exact-gradient guarantee. External leads unverified.`

### 14. Host offload with recomputation, if resident state blocks the task

- **Mechanism/savings:** offload selected saved activations or optimizer moments to host and bring them back at last responsible use. This can make a memory-infeasible task runnable; it saves device bytes, not backward FLOPs. On 7B the CPU-memory budget and PCIe transfer volume must be admitted before a run.
- **Cost/failure:** synchronous native execution may expose transfer latency; pinned memory and host RAM are not free. Offloading nominal 7B FP32 gradients and updated weights alone moves tens of GB/step; bandwidth ceiling, host allocation and CPU optimizer time must be charged. No invented PCIe bandwidth for this machine.
- **Prior art / ours:** Ren et al., **ZeRO-Offload (2021)**; Rajbhandari et al., **ZeRO-Infinity (2021)**. Take host/device partitioning and transfer scheduling; ours is native integration, not distributed scaling on one card. Search titles.
- **Cheapest falsifier:** GRAPA B=1 S=2048, offload only Adam moments, five same-state steps, measure end-to-end time/host RSS/peak device bytes. Predict ≥1.5 GiB reduction in device residency and ≤2× step time, correct FP32 update relL2≤1e-6. **0.15 GPU-h**. Failure of the wall threshold can still be a capacity tradeoff, not an optimization win.
- **Attach / site comment:** [optim.py:51][C22], [autograd.cpp:28][C21] saved-data ownership, [checkpoint state adapter][C10]. `// Prior art: ZeRO-Offload (Ren et al. 2021), ZeRO-Infinity (Rajbhandari et al. 2021); single-card native host/device residency scheduling. Unverified — lead to check.`

### 15. Forward-mode stochastic directional gradients

- **Mechanism/savings:** evaluate JVPs along random parameter directions and form a gradient estimator without a reverse activation tape. One directional derivative does not recover a full P-dimensional gradient. Exact coordinate reconstruction needs P directions; a random directional estimator can be unbiased for suitable isotropic directions but high variance. Dense tangent state and JVP operator coverage still cost memory/compute.
- **Cost/failure:** no native JVP API was established; implementing it is a new differentiation surface, and sample efficiency is the main risk. MLA/APA nondifferentiable selection must have the same fixed-router convention as the reverse baseline.
- **Prior art / ours:** Baydin et al., **Gradients without Backpropagation (2022)** and classical forward-mode AD. Take random-direction JVP gradients; ours would be native operator coverage and a deliberately narrow adapter trial. Search exact title.
- **Cheapest falsifier:** train only a rank-8 adapter on captured GRAPA S=512 activations, directions m=1/8/32 fixed beforehand. At equal measured time and data, require final held-out MSE≤1.05× reverse-mode baseline and ≥50% activation-tape reduction. **0.10 GPU-h** (CPU counterpart possible). Failure rejects this narrow use, not all forward-mode methods. Full GRAPA pretraining is unsupported extrapolation.
- **Attach / site comment:** [autograd.cpp:28][C21] needs a separate tangent API, [ops.cpp:622][C13] needs JVP rules; no backward-closure flag suffices. `// Prior art: forward-mode AD, Baydin et al. 2022 random-direction forward gradients; experimental native JVP adapter path. Unverified — lead to check.`

### 16. MeZO/SPSA zeroth-order updates

- **Mechanism/savings:** evaluate losses at θ±εz under no-grad, reconstruct perturbations from a seed, and update using their directional finite difference. No activation tape and potentially inference-like device memory. Per direction costs two forwards plus weight perturbation passes; this does not make learning require fewer examples or total FLOPs to target quality.
- **Cost/failure:** high estimator variance, step/ε sensitivity, noisy losses, non-smooth selector boundaries and catastrophic cancellation in low-precision weights. Need an affordable weight representation even for inference; BF16 7B alone still does not fit this card.
- **Prior art / ours:** Spall, **SPSA (1992)**; Malladi et al., **Fine-Tuning Language Models with Just Forward Passes / MeZO (2023)**. Take two-sided perturbations and seed regeneration; ours is native fit/quality accounting. Search exact titles.
- **Cheapest falsifier:** same captured-task rank-8 adapter as item 15, S=512, fixed m=1/8 directions, equal wall budget versus reverse-mode. Predict inference-like memory (≤1.2× forward-only peak) and held-out MSE≤1.05× baseline at equal time. **0.10 GPU-h**; if gradient-free updates need many more evaluations, reject the cheap-learning hypothesis even when memory passes. No 232M/7B pretraining conclusion.
- **Attach / site comment:** [optim.py:16][C22] separate optimizer, [autograd.cpp:10][C21] no-grad scope, [grapa/train.py:366][C2] future driver branch. `// Prior art: Spall 1992 SPSA, Malladi et al. 2023 MeZO; seeded two-forward estimator, native experimental driver. Unverified — lead to check.`

### 17. Reversible transformer blocks

- **Mechanism/savings:** replace ordinary residual blocks with invertible additive coupling so inputs can be reconstructed from outputs. Cuts depth-dependent saved hidden states; still retains local attention/MLP workspaces and recomputes activations. With checkpoint boundaries already only 96 MiB at S=2048, that particular saving is modest on GRAPA.
- **Cost/failure:** architecture/training change, reconstruction numerical drift, stochastic replay, and coupling constraints. Ordinary `x + attn(x)` is not generally invertible just because it is residual. Existing pretrained GRAPA weights cannot simply be relabeled reversible; optimizer/model state costs remain.
- **Prior art / ours:** Gomez et al., **The Reversible Residual Network (2017)**; Kitaev, Kaiser and Levskaya, **Reformer (2020)**. Take additive coupling/reconstruction, not Reformer's other attention approximations by default. Ours would be a new GRAPA architecture variant requiring retraining. Search titles.
- **Cheapest falsifier:** 2-block width-1024 MLA coupling prototype S=512, 8 sequential inversion/replay checks and a short matched toy-loss run. Predict reconstructed FP32 hidden relL2≤1e-5 and ≥40% saved-hidden reduction versus the same coupling blocks retaining all hidden states; report BF16 separately. **0.10 GPU-h**. Passing does not establish that it beats current whole-block checkpointing or preserves GRAPA's representational constraints.
- **Attach / site comment:** [bindings.cpp:83][C3] reconstruction VJP seam, [model_mla.py:72][C1] would require a new block type. `// Prior art: RevNet (Gomez et al. 2017), Reformer (Kitaev et al. 2020); new reversible-coupling model variant, not an invertible reinterpretation of current blocks. Unverified — lead to check.`

### Why APA tails do not simply disappear in backward

**D, algebra, not measured training evidence.** For one row with fixed selector m_j and detached Kq, define

    K_eff,j = m_j K_j + (1-m_j) Kq_j
    z_j = scale * Q·K_eff,j
    p = softmax(z),  O = sum_j p_j V_j
    g_j = dO·V_j,  Drow = sum_j p_j g_j = dO·O
    dz_j = p_j (g_j - Drow)
    dQ = scale * sum_j dz_j K_eff,j
    dK_j = scale * m_j dz_j Q
    dV_j = p_j dO

The softmax Jacobian is `J_ij = p_i(δ_ij−p_j)`. Small p_j can suppress an individual term, but neither all-tail probability mass nor value/upstream norms nor cancellation is controlled by a refine percentile. Kq is detached from K, **not from Q** in Q·Kq. Native code implements precisely this nonzero tail dQ and dV structure [C6]; this also corrects the overbroad codebook docstring “bulk path is never differentiated.” The current fixed selector ignores derivatives of the selection/quantizer itself; agreement with that composed reference proves that defined surrogate, not dense full-precision objective equivalence.

An analytical counterexample: p=(0.999,0.001), V=(0,1000), scalar dO=1. Then O=Drow=1 and dz=(-0.999,0.999). The 0.1%-mass key has a large score gradient (D). Realize z=(1+log 999,1) with Q=1, scale=1 and those effective keys. The tail contributes 0.999 to dQ; dropping only that term changes dQ by about 14.5% relative to the true dQ=−0.999·log 999 (D). Tail deletion also removes its value update; it is not justified by small mass alone. This is a symbolic witness, not a GPU test.

If tail mass ε=sum_T p_j, |g_j|≤G and ||K_eff,j||≤Kmax, triangle inequality gives

    ||omitted dQ|| <= 2 * scale * ε * G * Kmax
    sum_T ||omitted dV_j|| = ε * ||dO||

These are **absolute** bounds under stated norm assumptions. They do not guarantee small relative gradient error when the true dQ cancels toward zero, nor stable optimizer trajectories, nor a bound on an entire model's Jacobian. Across query rows and layers, omitted contributions accumulate. Ignoring the tail inside Drow also changes selected-key dz; dropping tail output terms or renormalizing selected probabilities changes more than masking gradient contributions. Keep these as distinct experiments.

**Prior art for this derivation:** standard softmax/vector-Jacobian calculus in attention (Vaswani et al., 2017), the FlashAttention output-dot identity (Dao et al., 2022), and elementary triangle inequality. Taken: those identities/bounds. Ours: applying them to this code's detached-Kq contract and this explicit witness; **no prior art known to me for this exact numeric witness**, no novelty claim. External metadata unverified — lead to check `softmax attention backward Jacobian FlashAttention`. Recommended comment at `kernels.cu:2580`: `// Standard softmax VJP (Vaswani et al. 2017; FlashAttention 2022): tail P remains in rowdot, dQ and dV; detached Kq removes only its dK path. External references unverified — lead to check.`

## 3. The honest map + the recommended first experiment

| Family | FLOPs | Device memory | Wall-clock | Generic / stack-specific | Status and characteristic failure |
|---|---|---|---|---|---|
| Output-dot row term (1) | Reduces redundant backward MACs | Slight lifetime increase possible | Plausible win, unmeasured | Standard identity; native APA seam specific | Standard technique, new local port; rounding/selector-replay seam |
| Streaming/tiled backward (2) | Recompute adds FLOPs; better scheduling | Large win for composed attention; native APA already linear | Depends on IO/atomics/occupancy | Generic; D≠Dv MLA adaptation specific | Standard algorithm family, no local speed result |
| Frozen-operand pruning (3) | Removes unneeded VJP GEMM | Removes unwanted gradient | Likely in frozen-base workloads | Generic autograd contract | Standard; no benefit when both operands train |
| Training fusion (4) | Mostly same math, less overhead | Fewer buffers | Plausible bandwidth/launch win | Generic; weighted NLL/ties specific | Standard; precision/loss-semantic bugs |
| Selective checkpointing (5) | Adds recompute | Reduces saved data | Usually cost, sometimes beats whole-block replay | Generic; current graph schedule specific | Already standard and partly implemented |
| BF16 / 8-bit optimizer (6) | Similar arithmetic count | Activations / moments respectively | Hardware/conversion dependent | Generic | BF16 implemented; optimizer quantization not established |
| Captured-pair distillation (7) | Avoids base backward entirely | Base training graph/state absent | Can be very cheap after capture | Generic method; bolt-on corpus/router specific | Implemented mechanism, behavioral negatives |
| LoRA/QLoRA (8) | Avoids base dW; retains dX | Trainable state + quantized base | Depends on dequant/backward | Generic | Standard, native fit/quality unmeasured |
| Activation compression (9) | Encode/decode extra | Targeted saved buffers | Can lose at small sequence | Generic | Standard research; gradient error/outliers |
| Lifetime / optimizer-in-backward (10) | Similar work | Leaf/saved-data lifetime | Scheduling dependent | Generic; replay/ties specific | Op-grad release already exists; other changes risky |
| GaLore (11) | Often adds projection work | Primarily moments | Can slow steps | Generic | Standard research; stale subspace, dense dW remains |
| Sparse updates (12) | Only with actual sparse compute | Conditional; residuals cost | Irregular sparse kernels can lose | Generic; tied embeddings relevant | Known rare-feature starvation/selection overhead |
| APA tail backward (13) | Potential reduction or lower precision | Not automatic with dense destinations | Open here | APA-specific selector; generic approximation idea | Open local hypothesis; softmax coupling, gradient bias |
| Host offload (14) | No reduction | Capacity relief | Transfer/CPU penalty | Generic single-card capacity route | Standard; host/PCIe bottleneck |
| Forward-mode / MeZO (15/16) | Per-step cost ≠ cost-to-quality | No reverse tape; tangent/state overhead remains | Often more evaluations | Generic | Known variance/sample-efficiency limits |
| Reversible blocks (17) | Reconstruction adds work | Hidden-state savings | Recompute overhead | Generic architecture change | Standard; drift and retraining cost |

**Recommended first experiment (one measurement, registered now, UNRUN):** make an operator-level **time-and-live-allocation census of the actual GRAPA training step**, on the pinned step-1720 checkpoint and its tokenizer, B=1/S=2048/refine=.15/BF16 with current whole-block checkpointing and FP32 AdamW. Use fixed token/weight arrays, two warmups and five measured steps, in an isolated successor harness; record initial forward, checkpoint replay, native APA backward, linear-map VJPs, pointwise/loss VJPs, Adam, and unique live/reserved allocation peaks. Run the same inputs without instrumentation to ensure profiling changes whole-step time by ≤5%; otherwise the timing attribution is INCONCLUSIVE. **Primary prediction:** native APA backward consumes ≥40% of non-replay backward device time. **Falsifier:** a clean measured share <40% rejects it as the first wall-time target and promotes the largest measured linear/loss/pointwise component; it does not refute the output-dot identity. The allocation census additionally decides whether loss/replay/lifetime rather than moments owns the peak. Register one ≤480-second work cell (≤0.134 GPU-h work, **0.15 GPU-h total lease budget**, far below 2 GPU-h); if the complete sample count cannot finish, stop INCONCLUSIVE, without shortening the sequence. No campaign overlap, model update persisted, network, process signal, or GPU work in this scout. Prior art: standard controlled operator profiling/CUDA event timing (NVIDIA CUDA system; exact documentation year **unverified — lead to check** `CUDA events elapsed time profiler overhead`), plus local SP3/SP4G provenance and interleaved comparison practice (2026); ours is the component attribution on this training graph. Future profiler comments belong at `autograd.cpp:76`, `bindings.cpp:97–104`, and `grapa/train.py:366–385`.

The census is recommended before implementation because existing receipts establish that training ran, but cannot tell whether removing a third of a particular kernel's arithmetic meaningfully lowers current end-to-end time. For exact improvements, success means matching the specified gradient objective and lower measured cost. For approximations, success means fewer resources **to the same held-out quality**, with per-family checks and longer convergence validation beyond these cheap rejection screens. A kernel result cannot establish that claim.

## 4. Process safety, execution ledger and residuals

The supplied BP-SCOUT-1 order is the immutable plan. This single requested report contains the idea ledger and synthesis; no separate plan/ledger files or implementation were necessary for the read-only scope. All successor thresholds above were written before any successor experiment; **none ran**. Amendments, if any, must be separate and cannot retroactively change these predictions.

| Scout action | Scope / result | Evidence class |
|---|---|---|
| Read HOUSE_RULES first, then supplied/local AGENTS | No git/subagents/GPU/network/background jobs/waits or signals | Command record |
| Lightweight memory lookup | Located older GRAPA D≠Dv/OOM context; current code and receipts superseded stale no-checkpoint premise. Requested `project_grapa_training` / `project_moe_bolton` keys were not found in the available registry. | Memory context, not current runtime evidence |
| Read native/legacy/PyTorch engine source and GRAPA docs/logs/evaluation metadata | No imports, builds, model loads or tests; code-site recommendations only | C/R |
| Read named checkpoint header and final 1024 bytes | 2,805,440,129-byte file exists; protocol-4 header; no full pickle deserialization, metadata corroborated through eval/log | C/R |
| Standard-library scalar arithmetic | Checked parameter counts, matrix FLOPs and bytes; no tensor library/GPU import | D calculation |
| Create writable target, write this report | Only intentional writes: `artifacts/bp_scout_1/REPORT.md` and its requested directory | Command record |
| Source digest comparison | Nine inspected native/PyTorch source files hashed before report writing; final verification recorded below | Scoped byte-integrity check, not repository cleanliness |

**Final document verification:** 40 source references resolve to existing files and in-range lines; no undefined C/R reference labels; all 9 listed source SHA-256 pins match; 17 ranked entries are present; the target directory contains only `REPORT.md`. Scalar rechecking corrected the rounded GRAPA attention-GEMM share to 36.0% and confirmed the analytical tail witness's 14.48% dQ error. These are document/source-integrity and arithmetic checks, not GPU, model-quality, or independent-review gates. No experimental threshold changed.

One source display (`nl ... | head`) printed a harmless `nl: write error: Broken pipe` when the reader closed after its requested lines. The relevant log was read; this was not a training/test failure or a process-control action. Broad searches occasionally returned more output than the display budget; load-bearing source sites were subsequently read directly. No `timeout`, `kill`, signal, GPU query, or background wait command was used. All Bash calls completed in seconds, below the 10-minute limit.

**Read-only scope confirmed by actions:** no code, configuration, checkpoint, log, model, service, or other seat's artifact was edited. **`git status` clean apart from `artifacts/bp_scout_1/`: NOT VERIFIED.** Running it would violate the explicit NO git order; concurrent seats also prevent inferring global tree cleanliness from our own write scope. No claim of a clean initial or final worktree is made. This is the explicit residual on Done item 4, not a fabricated pass. Source hashes cover only the listed files and the report-writing interval; they are not a substitute for git status or an audit of concurrent writers.

Pre-report SHA-256 pins (C; final comparison below):

```text
tensor_cuda/src/ops.cpp                 f08f8dec223f9555f95af3eb9fcac96d9730d366bc37f4a55c87bda25a215fbf
tensor_cuda/src/kernels.cu              93bd157bfa9ce58b9d9386a12b06d8a99a91542ea3a4022fffc7fe1b49c24b59
tensor_cuda/src/autograd.cpp            ad24a0a9853e9c173ac69e1c3bb0a1077cb96b88297f1b9b26c1ee58b7a36377
tensor_cuda/src/bindings.cpp            efbebeab32be240fa640e98de3f477ddd74ec937c12c519c6ccdaa843f70c1c1
tensor_cuda/tensor_cuda/nn.py           4659cdfa382c0166543a555ef5101e0e3b7ceb8d930bb822be95a0324263517d
tensor_cuda/tensor_cuda/optim.py        41693c324d5ea2f081ec7777ce8165fbc65d04a7b557fce27ff6d9b9a24457ee
tensor_cuda/tensor_cuda/functional.py   418ee285bdeefacbd9d5a1ccaf39700b8e83ea339377230c79f1d4480a1e3d9c
apa_cuda/csrc/apa.cpp                   6c2379506d1549371dcfb669efbd3e9d00d8d50d1ca9788ab27208dbfced642d
apa_cuda/apa_attention/ops.py           0d301773656ffb532a9606b08d7d02be1c595ea0d5db23c9907468c8d390dc91
```

## Done (verbatim from the order)

1. Cost model table (with evidence classes).
2. Ranked idea ledger (≥ 10 entries) with prior art and falsifying experiments.
3. The honest map + the recommended first experiment.
4. Process safety (read-only confirmed: `git status` clean apart from
   `artifacts/bp_scout_1/`), model id + effort.

Items 1–3 delivered above; item 2 has 17 ranked entries. Item 4 is delivered with the explicit NO-git/cleanliness limitation above, not affirmed verbatim as a measured result. Dispatch model/effort: `gpt-6-astra` / high; no independently observed runtime launch setting.

## Source index

Line anchors identify the inspected source sites; adjacent lines discussed in prose belong to the same function/receipt. Historical line references inside old logs can differ from current source. No linked file was modified by this scout.

[C1]: /mnt/ForgeRealm/GRAPA-Native-LLM/grapa/model_mla.py:24
[C2]: /mnt/ForgeRealm/GRAPA-Native-LLM/grapa/train.py:358
[C3]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/bindings.cpp:66
[C4]: /mnt/ForgeRealm/GRAPA-Native-LLM/grapa/attention.py:54
[C5]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/ops.cpp:734
[C6]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/kernels.cu:2517
[C7]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/functional.py:62
[C8]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/kernels.cu:2373
[C9]: /mnt/ForgeRealm/GRAPA-Native-LLM/grapa/codebooks.py:67
[C10]: /mnt/ForgeRealm/GRAPA-Native-LLM/grapa/checkpoint.py:26
[C11]: /mnt/ForgeRealm/GraftRepository/scripts/gpt_oss20b_expert_e1.py:3266
[C12]: /mnt/ForgeRealm/GRAPA-Native-LLM/grapa/attention_mla.py:49
[C13]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/ops.cpp:622
[C14]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/ops.cpp:796
[C15]: /mnt/ForgeRealm/GRAPA-Native-LLM/grapa/model.py:43
[C16]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/ops.cpp:133
[C17]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/nn.py:280
[C18]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/kernels.cu:4183
[C19]: /mnt/ForgeRealm/GRAPA-Native-LLM/grapa/loss.py:9
[C20]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/ops.cpp:999
[C21]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/autograd.cpp:14
[C22]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/optim.py:16
[C23]: /mnt/ForgeRealm/Project-Tensor/apa_cuda/apa_attention/ops.py:57
[C24]: /mnt/ForgeRealm/Project-Tensor/apa_cuda/csrc/apa.cpp:341
[C25]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/ops.cpp:435
[C26]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/tensor_cuda/nn.py:97
[C27]: /mnt/ForgeRealm/Project-Tensor/tensor_cuda/src/kernels.cu:4219
[R1]: /mnt/ForgeRealm/Project-Tensor/docs/APA.md:5
[R2]: /mnt/ForgeRealm/GRAPA-Native-LLM/logs/train_copy_rule_1700_to_1750_r015.log:3
[R3]: /mnt/ForgeRealm/GRAPA-Native-LLM/eval_runs/base_proc_1720_teacher/eval_20260623_201801_w12288_r0.15.summary.json:90
[R4]: /mnt/ForgeRealm/Project-Tensor/docs/APA_SP1_1_REPORT.md:9
[R5]: /mnt/ForgeRealm/Project-Tensor/docs/APA_SP2_DELTA_DERIVATION.md:13
[R6]: /mnt/ForgeRealm/Project-Tensor/docs/APA_SP3_LEDGER.md:39
[R7]: /mnt/ForgeRealm/Project-Tensor/docs/APA_SP5_LEDGER.md:208
[R8]: /mnt/ForgeRealm/GraftRepository/artifacts/moe_e2/MOE_E2_REPORT.md:7
[R9]: /mnt/ForgeRealm/GraftRepository/artifacts/moe_e3/MOE_E3_REPORT.md:5
[R10]: /mnt/ForgeRealm/GRAPA-Native-LLM/docs/IMPLEMENTATION_STATUS.md:71
[R11]: /mnt/ForgeRealm/GRAPA-Native-LLM/logs/grapa_mla_w20480_r015_bf16_train.log:1
[R12]: /mnt/ForgeRealm/Project-Tensor/docs/APA_SP4G_LEDGER.md:49
[R13]: /mnt/ForgeRealm/GRAPA-Native-LLM/docs/CORPUS_CONSOLIDATION_PLAN.md:18
