# APA-SP kernel optimization proposal

Status: **proposal**. No kernel has been changed. Evidence classes are marked
per claim: *receipt* (lead-run SPD1 GPU cells, `artifacts/apa_spd1/SPEED_CHAIN.md`),
*source* (reading the current kernels), *CPU emulation*
(`scripts/apa_sp_tile_occupancy.py`), *estimate* (arithmetic on published
hardware figures; RTX 4070 Super, sm_89, 56 SMs, ~504 GB/s, 48 MB L2).
Nothing here is a GPU measurement of a new kernel.

## 1. Bottom line

APA-SP is slow because of **how the kernels are built**, not because of the
selection rule. The prefill kernel is a FlashAttention-0: one warp per query
row, scalar FP32 FMAs, every key visited serially with two or three dependent
L2 round trips per key, no query tiling, no tensor cores. It runs 17–60× behind
torch flash and 4–19× behind the engine's *unfused* dense path. The decode
kernel is better (2–4× behind flash) but underfills the GPU, syncs every four
keys on the D≥128 path, does uncoalesced loads on the D=64 path, and re-reads
K/V once per query head under GQA.

The running-max rule tiles cleanly: `max` is exact and associative, so an
in-tile prefix-max scan with a carry across tiles gives the **same selection
as the serial loop for the same bulk scores**. That is the property that makes
a FlashAttention-2-structured tensor-core APA-SP possible.

Realistic targets, stated as ceilings first:

| Path | Today vs torch flash (BF16) | Physical floor | Realistic target |
|---|---|---|---|
| Prefill, floating `kq` | 17–60× slower | ≥1.5× flash (3 GEMMs vs 2) | 2–4× flash |
| Decode, floating `kq` | 2–4× slower | ≈ flash (same bytes ±15%) | ≈ flash |
| Decode, packed INT4 `kq` | n/a | ≈0.6–0.7× flash bytes | **faster than flash**, ~1.3× |

So: prefill can get ~10–20× faster but cannot beat flash while `kq` is a
full-width floating copy of K. Decode can match flash, and is the one place
APA-SP can beat it — once `kq` is stored packed, because decode is
bandwidth-bound and APA reads fewer bytes than flash only then.

## 2. Where the time goes

### 2.1 Receipts (SPD1, BF16 rows, CUDA-event median ms)

| Cell | APA-SP | APA two-pass | engine dense (unfused) | torch flash | SP / flash |
|---|---:|---:|---:|---:|---:|
| prefill S=2048 D=64 causal H4/KV4 | 2.58 | 4.10 | 0.335 | 0.147 | 17.5× |
| prefill S=8192 D=64 noncausal H4/KV4 | 68.6 | 115.2 | 17.7 | 1.145 | 60× |
| prefill S=8192 D=128 causal H4/KV4 | 47.0 | 71.7 | 5.84 | 1.241 | 38× |
| E1 prefill L=512 S=32768 D=128 H16/KV4 | 141.4 | 149.1 | 7.29 | 2.70 | 52× |
| decode S=2048 D=64 H4/KV4 | 0.162 | 0.232 | 0.072 | 0.081 | 2.0× |
| decode S=32768 D=64 H8/KV2 | 0.442 | 1.300 | 0.306 | 0.121 | 3.7× |
| decode S=32768 D=128 H4/KV4 | 0.718 | 3.970 | 0.233 | 0.177 | 4.1× |

Two tells in the receipts: BF16 SP is barely faster than FP32 SP (68.6 vs
72.5 ms; 47.0 vs 49.7 ms), and D=128 costs only ~1.5× D=64. A
bandwidth- or compute-bound kernel would move with both. This one is
**latency-bound**.

### 2.2 Prefill kernel — `apa_selective_sp_kernel` (`tensor_cuda/src/kernels.cu`, APA_SP1_ADDITION block)

*Source* + *estimate*:

- **Grid = one 32-thread CTA per query row.** Ada allows 24 resident CTAs per
  SM, so occupancy is capped at 24 warps/SM (50%) before registers matter.
- **No query tiling.** Each K/kq/V row is fetched once per query row that
  sees it. FlashAttention reuses each K tile across 64–128 queries from shared
  memory. Here everything streams from L2 (the S=8192 working set, ~24 MB,
  fits in the 48 MB L2, which is why it's latency- rather than DRAM-bound).
- **Serial per-key dependency chain.** Per key: load `kq` row → 5-step
  shuffle reduction → broadcast → running-max compare → (15%) load `k` row →
  5-step reduction → broadcast → two `__expf` → rescale all `acc[]` → load
  `v` row → FMA. Nothing for key j+1 is issued before key j finishes.
  Back-of-envelope on the S=8192 D=128 causal cell: 134M row-key visits over
  ~1,344 resident warps in 47 ms ≈ **~1,100 cycles per key visit**, i.e.
  three to four dependent L2 trips per key. That is the whole story.
- **Per-key rescale.** The online-softmax update rescales the accumulator on
  every key (Milakov & Gimelshein form) instead of once per tile (FA2 form).
- **CUDA cores only.** Ada BF16 tensor-core throughput (FP32 accumulate) is
  roughly 2× the FP32 CUDA-core peak, and this kernel reaches nowhere near
  either.

### 2.3 Decode kernel — `apa_selective_sp_splitk_kernel` (`tensor_cuda/src/apa_sp1_1.cuh`)

*Source*:

- **D≥128 (WCOOP) path:** each of 4 warps takes one key per iteration, so a
  "tile" is 4 keys and the loop pays **two `__syncthreads` per 4 keys** for
  the prefix carry.
- **D=64 path:** one key per thread, serial `d=0..63` loop. Adjacent lanes
  read rows `D` elements apart, which is the 25-sectors-where-4-would-do
  pathology A5 fixed for the base kernel. The V accumulation is the same
  shape with `acc[64]` per thread (95 registers).
- **Underfilled grid:** `B·H × ceil(S/2048)` CTAs of 4 warps. S=32768, H=4 →
  64 CTAs on 56 SMs, ~4–5 warps per SM. Decode needs many loads in flight to
  saturate DRAM; this has few.
- **No GQA sharing:** each query head re-reads its KV head's `kq`, `k`, `v`.
  At H8/KV2 that is 4× the necessary bytes (L2 absorbs part of it).
- **Partition-local prefix inflates refinement.** SP1.1's semantics reset the
  running max every 2048 keys. *CPU emulation*: refined fraction at frozen δ
  goes from **0.158 → 0.258** (S=8192) and **0.144 → 0.350** (S=32768) versus
  the whole-row prefix. SPD1's GPU masks agree (0.279 at S=8192 D=64). More
  refinement is more exact-K bytes, and it's why 21/24 decode classes came out
  fraction-unmatched in SPD1.

Side finding (*source*): the **two-pass** decode stats kernel is launched
`<<<rows, threads>>>` — one CTA per row over all S keys, not split-K. That is
why two-pass decode at 32K is 4 ms. It isn't the subject here, but in decode
the z-score rule doesn't need a second `kq` read at all: pass 1 can write the
bulk scores (4 B/key vs 256 B/key of `kq`) and pass 2 can read them back.

## 3. CPU emulation receipts that shape the design

`python3 scripts/apa_sp_tile_occupancy.py` (SP1 synthetic generator, SP1
frozen δ; synthetic iid scores — not real-model attention):

| Question | S=2048 D64 causal | S=2048 D64 noncausal | S=8192 D128 causal |
|---|---|---|---|
| Refined / visible | 0.145 | 0.160 | 0.149 |
| 16×16 tiles with **zero** refined keys | 0 / 16,512 | 0 / 32,768 | 0 / 131,328 |
| 16×64 / 64×64 / 128×64 empty tiles | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| 64-key row-tiles that raise the running max | 19.6% | 13.1% | 6.8% |
| 16-row warp tiles with any raising row | 78.5% | 68.9% | 43.0% |

Consequences:

1. **Skipping the exact GEMM per tile is dead.** Not one tile of any size is
   free of refined keys. Late in the row density is still 5–8%, and
   `0.95^256 ≈ 2e-6`. Compute exact QKᵀ densely on tensor cores and select
   per element. (Real attention with sinks and locality could differ; rerun
   this on dumped real Q/K before reviving the idea.)
2. **The prefix scan is cheap either way.** Even when every tile pays the full
   in-tile scan it is ~4% of the MMA time (see K1 in §4). A "tile doesn't raise the
   carry → skip the scan" fast path saves at most part of that, and warp
   lockstep means it fires on only 22–57% of warp tiles. Optional, low priority.

## 4. Proposals

Ordered by gate cost. **Tier 0** keeps every arithmetic operation in the same
order, so the gate is bitwise equality with today's kernel. **Tier 1**
reassociates fp32 sums (tensor-core accumulation order, tile-wise softmax), so
the gate is a selection flip-rate ceiling plus output tolerance, as BP-KERNEL-4
used (≤0.5% flips). **Tier 2** changes semantics and needs registration.

### T0-a. Prefill: break the per-key latency chain (bit-exact stopgap)

- Process keys in groups of 4–8 per warp: issue all `kq` and `v` loads for the
  group up front (16-byte vector loads where D allows), then run the 4–8 dot
  reductions interleaved (same per-lane FMA order, same shuffle tree), then
  walk the group serially for running-max / refine / online update in the
  original order. Issue the exact-`k` loads for all refined keys of the group
  together before reducing them.
- Pack 4 rows per 128-thread CTA (same warp-per-row work) to lift the
  24-CTA/SM cap to 48 warps/SM. Put GQA group heads for the same query
  position in the same CTA so their identical K/V loads hit L1.
- Every bulk, exact, `expf` and accumulator update happens in the same order
  on the same values → bitwise-equal output and mask. Verify with SASS that
  FMA contraction didn't change, and with an equality gate.
- *Estimate:* 3–5× (47 → ~10–15 ms on the S=8192 D128 causal cell). Still
  ~10× flash. Its value is that it ships in days with a trivial gate.

### T0-b. Decode: fix the SP1.1 kernel's obvious losses (bit-exact)

- WCOOP path: each warp computes and holds the bulk scores of 32 consecutive
  keys before the block exchanges warp maxima — one barrier pair per 128 keys
  instead of per 4. Same per-key dot order, same carry semantics.
- D=64 path: keep the per-thread serial FMA order (so bulk values are
  unchanged) but load each key row with 16-byte vector loads.
- *Estimate:* 1.5–2× at long S; does not fix grid fill or GQA (those need D1).

### K1. Prefill: FlashAttention-2-structured tensor-core APA-SP (Tier 1, the big one)

Do **not** extend BP-KERNEL-4's `bk4::forward`: it keeps scores and the O
accumulator in shared memory, does the softmax with 16 threads and runs MMA
on one warp, which is why it bought only 2×. Use the FA2 skeleton:

- CTA = (b, h, 64-query tile); 4 warps × 16 rows each; Q fragments loaded
  once into registers (`ldmatrix`). Key tiles of Bc=32–64, `cp.async`
  double-buffered `kq`/`k`, single-buffered `v` (sm_89 has no TMA/wgmma, so
  FA3 techniques don't apply). Causal: skip fully masked tiles, launch heavy
  query tiles first.
- Per key tile, three `mma.sync.m16n8k16` BF16→FP32 GEMMs:
  `S_b = Q·Kqᵀ`, `S_e = Q·Kᵀ`, later `O += P·V`. Scale after the dot
  (`bulk = acc*scale`, matching SP1's order).
- **Running max in the MMA accumulator layout:** each thread holds two rows,
  columns `8n + 2(lane%4) + {0,1}` for n-tiles `n`. Per n-tile: max of the
  pair, 2-step quad `__shfl_up_sync` scan for the exclusive prefix, carry
  update from lane 3 of the quad. Masked (causal) keys enter as −∞, which is
  exact because they only ever trail the visible keys. Then
  `refine = bulk >= prefix - delta`, `s = refine ? S_e : S_b`, no branch.
  *Estimate:* ~32 shuffles per row-half per 64-key tile ≈ 4% of the CTA's MMA
  cycles.
- Online softmax once per tile (FA2): row max/sum via quad shuffles, one O
  rescale per tile, `exp2f` with `log2e` folded in **after** the selection
  compare (so δ stays in natural-log units and the compare sees the same
  `bulk` value). P rounded to BF16 for the PV MMA — standard FA practice and
  already accepted by BP-KERNEL-4, but it is a change from SP1's fp32 weights
  and belongs in the tolerance gate.
- Sinks folded once per row at the end; diagnostic mask template stays off
  the timed path.
- Registers (D=128, Bc=64): Q 32 + S_b/S_e 64 + O 64 ≈ 180/thread at
  4 warps. Bc=32 halves the score registers if occupancy needs it.
- Why selection survives tiling: the prefix is per row in ascending key
  order, tiles preserve that order, and `fmaxf` is exact, so the mask equals
  the serial loop's mask *for the same bulk values*. The only selection
  drift comes from tensor-core accumulation order and rounding of the bulk
  dot (Fasi et al. show NVIDIA tensor-core fp32 accumulation is not IEEE
  round-to-nearest). That is SP1.1's "EMULATOR_ORDER_SENSITIVE" class again,
  gated by flip rate.
- FP32 inputs: TF32 would round both operands, which is a semantic change.
  Route FP32 to T0-a and make K1 the BF16/FP16 path; add a 3×TF32
  error-compensated variant only if FP32 is a production need.
- *Floor:* 3 GEMMs vs flash's 2 → ≥1.5× flash. Flash's 1.24 ms on the S=8192
  D128 causal cell is ~55 TFLOPS, so K1 at the same efficiency would be
  ~1.9 ms. *Estimate* for a first hand-written `mma.sync` kernel: 2.5–5 ms
  (10–20× faster than today).

**De-risk with a Triton prototype first** (1–2 days, BF16, D 64/128).
`tl.dot` gives the tiling, MMA and pipelining for free, and
`tl.associative_scan` with a `max` combine gives the in-tile prefix. It
answers "what does the tiled algorithm cost on this card" before anyone
writes `ldmatrix` code, and doubles as a second implementation for the parity
gate. Port to native `tensor_cuda` once it's within ~3× of flash.

### D1. Decode: stage bulk scores, pack GQA, fill the GPU (Tier 1)

Three launches, each fully parallel, CUDA-graph them in the decode loop:

1. **Bulk kernel**, grid `(B·KVH, S/256)`: one CTA per KV head and 256-key
   chunk, all G query heads of the group together (q for G heads in shared
   memory). 16-byte `cp.async` loads of `kq`, read **once** for all G heads.
   For G ≥ 4, use `mma` with M = G padded to 16 (FlashInfer-style GQA
   packing) so the hardware does the reduction; otherwise lane-split dots.
   Writes `bulk[g][j]` (4 B/key/head — vs 256 B/key of `kq` for D=128 BF16)
   and per-chunk maxima.
2. **Select + exact + PV kernel**, same grid: carry = max of the preceding
   chunk maxima (≤ 7 loads for 2048-key partitions), in-CTA prefix max over
   the chunk's staged scores (exact), refine mask, gather each `k` row once if
   **any** of the G heads refined it, chunk-local softmax with one max and one
   sum (all scores are known — no per-key rescale), PV with each `v` row read
   once for all G heads. Write `(m, l, acc)` partials.
3. **Merge**: the existing `apa_selective_merge_kernel` (sinks once), or fold
   it into kernel 2 with a last-block-done counter.

This is not the old two-pass problem. The old APA read `kq` twice (stats, then
recompute). Here `kq` is read once; the "second pass" reads 4-byte scores.
Bytes ≈ fused single pass.

The bulk dot order can be kept identical to SP1.1 (per-thread serial FMA for
D=64, lane-split `d = lane + 32t` with the same shuffle-down tree for D≥128),
in which case selection is **bit-identical** to SP1.1 and only the softmax
order changes. Lane-split/MMA dots are faster but move to Tier 1.

*Estimate:* bandwidth floor for decode S=32768 D=128 H4/KV4 with BF16 `kq` is
~72 MB → ~157 µs; flash is at 177 µs (floor ~146 µs). D1 target ≈ flash
(0.72 → ~0.2 ms). At H8/KV2 the GQA packing removes a 4× redundancy.

**D1-g (Tier 2, small): whole-row prefix in decode.** Once bulk scores are
staged, the carry can be the max over *all* preceding chunks rather than only
those in the same 2048-key partition, at zero cost. That restores SP1's
whole-row semantics in decode: refined fraction 0.35 → 0.14 at 32K (*CPU
emulation*), ~2.4× fewer exact-K bytes, and SP1's calibration then applies to
decode as registered instead of being transferred. Needs a registration
because it changes SP1.1's mask (back to SP1's).

### K3. Packed `kq` (Tier 2 — the only route to beating flash)

In every SPD1 cell `kq` is a floating reconstruction the same width as K
("kq remains floating; this does not measure compressed KV-cache residency").
APA-SP therefore reads `kq` **plus** 15–35% of K plus V, which is ≥ flash's
bytes. The speed thesis needs `kq` stored packed.

- **Decode:** symmetric INT4 codes + fp32 per-key scale (APAMQ F-A1 format)
  or TurboQuant codes (rotate q once per step, 16-entry LUT). Bytes per key
  for D=128 BF16: 64 (`kq`) + 0.15·256 (`k`, with D1-g) + 256 (`v`) ≈ 358 vs
  flash's 512 → **~0.7× flash bytes**. *Estimate:* ~110–130 µs at the 32K cell
  vs flash 177 µs.
- **Prefill, reopening APAMQ F-A2:** the F-A2 STOP was that staging
  reconstructed `code·scale` in BF16 rounds the operand. Don't stage the
  product — stage the **code**. Integers −7…7 are exact in BF16, the BF16 × BF16
  product is exact in fp32, so `S_b = (Q·Cᵀ)` on tensor cores followed by a
  per-key column scale in fp32 differs from F-A1's dequantize-then-dot only by
  where the scale multiplies (reassociation class — arguably more accurate,
  since F-A1's `code·scale` itself rounds). Register-level unpack of the
  nibbles into the B fragment (Marlin-style) avoids a shared-memory dequant
  pass. This applies to integer codebooks only; TurboQuant's Lloyd-Max
  centroids would have to be rounded to BF16 (a semantic change).
- Prefill FLOPs don't drop (exact is still a dense BF16 GEMM, per §3), so
  packed `kq` mainly buys memory and decode speed.

### Not recommended (with cause)

| Idea | Why not |
|---|---|
| Skip exact GEMM on tiles with no refined keys | 0 empty tiles at every size (§3) |
| Sparse/compacted exact on CUDA cores inside K1 | At ~15% density it's roughly a wash with a dense tensor-core GEMM on Ada once shared-memory operand bandwidth is counted, and adds divergence; A/B it only if K1 is GEMM-bound |
| Extend `bk4::forward` | Wrong skeleton (smem accumulators, single-warp MMA); measured 2× |
| Smaller decode partitions to fill the grid | Changes SP1.1's selection (more refinement); D1 gets parallelism without touching semantics |
| FA3 techniques (wgmma, TMA, warp specialization) | Hopper-only; sm_89 has `mma.sync` and `cp.async` |
| FP8/INT8 Q for the bulk GEMM (SageAttention-style) | Quantizes q — a new bulk contract — and exact is still dense BF16, so prefill wouldn't beat flash anyway |

## 5. Suggested order and falsifiable predictions

| Step | Work | Gate | Prediction (BF16, SPD1 cell) |
|---|---|---|---|
| 1 | T0-a + T0-b | bitwise equality vs current kernels, all 50 SPD1 cells | prefill S=8192 D128 causal ≤ 15 ms; decode 32K D128 ≤ 0.45 ms |
| 2 | Triton K1 prototype | flip rate ≤ 0.5% vs SP1 emulator; output tol vs SP1 FP32 | prefill S=8192 D128 causal ≤ 4× flash |
| 3 | D1 (SP1.1-identical dot order) | SP1.1 mask bit-identical; output tol | decode 32K D128 H4/KV4 ≤ 1.3× flash |
| 4 | Native K1 in `tensor_cuda` | as step 2 | prefill ≤ 3× flash on every S≥2048 cell |
| 5 | D1-g, K3 (registered) | new registration; quality gates per model | decode 32K packed-INT4 < flash |

Every gate needs the GPU box; nothing above has been run on one.

## Prior art

| Technique | Source | Taken vs ours |
|---|---|---|
| Query/key tiling, warp split-Q, per-tile online softmax, causal tile skip, heavy-first scheduling | FlashAttention (Dao et al., 2022); FlashAttention-2 (Dao, 2023) | Taken wholesale for K1 |
| Online softmax normalizer | Milakov & Gimelshein, 2018 | Taken (already in SP1) |
| Running-max selection criterion | BLASST (Yuan et al., arXiv 2512.12087) | Taken (already in SP1); BLASST also uses it per block to drop blocks — we refine instead of drop |
| Split-K decode with partial merge | Flash-Decoding (Dao, Haziza, Massa, Sizov, 2023) | Taken (already in SP1.1); D1 re-partitions it |
| GQA head packing into tensor-core tiles for decode | FlashInfer (Ye et al., 2024/2025) | Taken for D1 |
| Parallel prefix scan (warp max-scan) | Hillis & Steele, 1986; Blelloch, 1990 | Taken; applying it in the `mma` C-fragment layout for APA's running max is ours |
| Single-kernel scan with carry across blocks | Merrill & Garland, 2016 (decoupled look-back) | Considered for D1; the three-launch staged-score design is ours and avoids spin-waits |
| Last-block-done fused reduction | NVIDIA CUDA Samples `threadFenceReduction` (~2009); Stream-K fixup (Osama et al., 2023) | Optional merge fusion in D1 |
| INT4 × FP16 `mma.sync` with register-level unpack | Marlin (Frantar et al., 2024) | Taken for K3; exact-integer-code-in-BF16 + post-scale argument against the F-A2 STOP is ours |
| Error-compensated TF32 for FP32 GEMM | Ootomo & Yokota, 2022; CUTLASS 3xTF32 | Optional FP32 path |
| Tensor-core accumulation is not IEEE RN | Fasi, Higham, Mikaitis, Pranesh, 2021 | Cited for the Tier-1 gate rationale |
| Quantized QKᵀ on Ada tensor cores | SageAttention / SageAttention2 (Zhang et al., 2024–2025) | Context for the rejected INT8/FP8-Q idea |
| Triton tiling DSL, `associative_scan` | Tillet, Kung & Cox, 2019 | Prototype vehicle |
| Records of an iid sequence grow ~ ln n | Chandler, 1952; Rényi, 1962 | Motivates the carry fast path; data says it's minor |
| TurboQuant packing | Zandieh et al., 2025 | Format option for K3 |
| Bit-exact ILP restructure (T0), staged-score decode (D1), whole-row decode carry (D1-g) | no prior art known to me for these APA-specific forms; they're standard GPU techniques applied to APA | Ours |

Citations are from memory and unverified in this session — lead to check:
"FlashInfer GQA tensor cores decode", "Marlin mixed precision kernel 2024",
"Ootomo Yokota TF32 error correction 2022", "Numerical behavior of NVIDIA
tensor cores 2021", "decoupled look-back Merrill Garland 2016",
"SageAttention2 INT4", "Stream-K Osama 2023".
