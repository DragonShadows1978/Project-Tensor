# APA-SP5 Implementation Ledger — GPT-OSS-20B model test

Receipts, appended as they happened. Seat: Opus 5 (`claude-opus-5[1m]`),
`opus-max` profile (reasoning max). Card: RTX 4070 SUPER 12 GB, shared —
every GPU cell took `flock --exclusive` on `/tmp/forge-gpu.lock` through
`scripts/apa_sp5_lead_gpu.sh`, which bounds the worker with `timeout(1)` and
holds the registered 30 s cooldown inside the lease. No process this seat did
not start was ever signalled. No git was run. No subagents were spawned.

## Order of work (the order requires item 0 first; it was)

| # | what | result | receipt |
|---|---|---|---|
| 1 | Read order, HOUSE_RULES §8/§9, SP3 + SP4G reports, prior-art comparison | — | — |
| 2 | Read the READ-ONLY port `GraftRepository/core/gpt_oss20b_tc.py` | 24 layers, full-attention at odd indices 1..23, sliding 128 at even; `sink_attention_tc` materializes B×H×L×S with NO chunking | port :345-372, :622-820 |
| 3 | Host build | GREEN, 80 s, `git status tensor_cuda/` EMPTY | `scripts/apa_sp5_build.sh`; `artifacts/apa_sp5/build/` |
| 4 | Engine entry survey | `apa_selective_attention_sp` already dispatches `cap<=64`, arbitrary GQA (`KVH`, `H/KVH`) and a `sinks` tensor → **no launcher addition needed** | `kernels.cu:7441-7503`, `bindings.cpp:664-680` |
| 5 | **Kernel/shape pin** (GQA 64/8, D=64, real sinks) | **PASS**: fp32 relF 2.27e-7, max-abs 7.7e-7 vs dense fp64; sinks load-bearing (relF 0.0131 on-vs-zeroed); 0 causal leak; kernel fraction 0.8125 vs fp64-rule 0.8154 | `scripts/apa_sp5_shape_pin.py`; log `logs/apa_sp5_shape_pin_*.log` |
| 6 | Token stream, PROTOCOL-O | 287,210 tokens, o200k, no chat template | `artifacts/apa_sp5/tokens.npy`, `tokens_meta.json` |
| 7 | **Item 0 — cost of a window** | see below | `artifacts/apa_sp5/item0_*.json` |
| 8 | **Item 1 — sensitivity floor** | 58.76 ppl / 8.73 % (window 0) | `item1_sensitivity_W1024_win0.json` |
| 9 | Calibration (e_q, B fraction, C δ, sink mass), 12 full layers | see below | `calib_W1024_b4_all.json` |
| 10 | **Registration written (IMMUTABLE)** | sha256 `3cc3b3e1…58159` | `registration.json` |
| 11 | **Predictions written** (lead's + seat's) | sha256 `17427eb3…92e44` | `predictions.json` |
| 12 | **Item 3 — per-call exactness gate** | **PASS** on layers 11 and 19 | `item3_exactness_L11.json`, `_L19.json` |
| 13 | **Item 2 — sink on/off pin** | **CONFIRMED** | `item2_sinkpin_L19.json` |
| 14 | PROTOCOL-O ppl, arms A/B/C/D/E | see below | `ppl_{A..E}_W1024_N4_b4.json` |
| 15 | Realised fractions from the kernel's own mask | C 0.1681, E 0.99999 | `frac_d3.16_C.json`, `frac_d44.315_E.json` |
| 16 | Floor at protocol scale → **amendment 1** | 12.89 ppl / 2.14 % at N=4 | `amendment_001_sensitivity_floor_scale.json` |
| 17 | Determinism check on the outlier window | **bit-identical**, max-abs 0.0 | log `logs/apa_sp5_repeat_*.log` |

## Failures and what they taught (RED honesty)

1. **lm_head OOM, twice, then diagnosed.** `int4_linear` is documented
   two-stage: it dequantizes the whole weight to a (K,N) fp16 buffer. For this
   head that is 2880×201088×2 = **1,105 MiB** against **927 MiB** free, so it
   OOMs at *every* chunk size — chunk 128, 32, 8 and even M=1 all failed.
   Probing with raw `cudaMalloc` showed 512 MiB succeeded and 1024 MiB failed,
   which is what isolated the cause. `int4_linear_fused` never materializes
   that buffer and costs ~0.6 s for a whole W=1024 window.
   *Lesson: the allocator failing on a 26 MiB request meant the failing
   allocation was not the one I was chunking.*
2. **Arm A OOMs at W=2048.** Not a bug — the port's `sink_attention_tc` builds
   a dense B×H×L×S score tensor (64×2048×2048×2 = 512 MiB), adds a full mask,
   `cat`s an extra sink column, then softmaxes. This is the P6 answer and the
   reason W=2048 was rejected for PROTOCOL-O.
3. **Dense fp32 attention OOMs at W=1024** in the item-1 reference (256 MiB per
   score tensor, several live). Fixed by chunking over query positions, which
   is arithmetically identical because each row's softmax is independent.
4. **Two harness bugs of my own**, both caught by the failure and fixed:
   a NumPy broadcast error in the shape-pin eligibility mask, and a `core`
   import placed before `load_runtime()` set `sys.path`.

## Item 0 — the measurement everything was sized from

| quantity | measured |
|---|---:|
| resident load, cold page cache | 37.2 s |
| resident load, warm | 12.5 s |
| resident VRAM | 10,927 MiB |
| **free VRAM (`cudaMemGetInfo`)** | **927 MiB** |
| W=1024 forward, arm A | 33.9 s (full layers 16.2, sliding 16.5) |
| W=1024 forward, arm B | 34.9 s |
| W=1024 forward, arm D | 33.9 s |
| W=2048 forward, arm C | 67.5 s |
| **W=2048, arm A** | **OOM** |

→ **PROTOCOL-O = W 1024, N 4** (150 s against a 285 s rail). W=2048×N=4 was
284.6 s — 0.4 s of margin — *and* arm A cannot run it.

## Calibration on real keys (registration inputs, measured before registering)

12 full-attention layers, W=1024, bulk 4:

| layer | e_mean | e_p99 | e_max | B frac | B unrefined mass | sink mass (median) | C δ matched |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.891 | 4.90 | 15.06 | 0.167 | 0.483 | 0.253 | 3.247 |
| 3 | 0.650 | 3.19 | 10.49 | 0.169 | 0.484 | 0.304 | 2.533 |
| 5 | 0.614 | 2.31 | 7.14 | 0.169 | 0.466 | 0.391 | 2.461 |
| 7 | 0.698 | 4.75 | 15.80 | 0.168 | 0.533 | 0.067 | 4.103 |
| 9 | 0.569 | 2.43 | 14.39 | 0.170 | 0.502 | 0.108 | 3.068 |
| 11 | 0.564 | 2.26 | 8.71 | 0.169 | 0.344 | 0.013 | 3.287 |
| 13 | 0.638 | 2.55 | 9.84 | 0.171 | 0.403 | 0.013 | 3.662 |
| 15 | 0.714 | 2.83 | 8.86 | 0.169 | 0.658 | 0.113 | 3.649 |
| 17 | 0.728 | 2.99 | 11.26 | 0.168 | 0.677 | 0.238 | 3.447 |
| 19 | 0.890 | 4.33 | 18.65 | 0.165 | 0.400 | 0.636 | 3.034 |
| 21 | 0.832 | 4.35 | 19.86 | 0.167 | 0.440 | 0.575 | 2.936 |
| 23 | 0.714 | 3.79 | 16.54 | 0.166 | 0.556 | 0.403 | 2.826 |

e_q (registered rule: the MAX) = **19.855** → E's δ = ln(100) + 2·19.855 =
**44.315**. B's median fraction 0.1685 → C's single global δ = **3.16**
(per-layer matched δ ranged 2.46–4.10). Max relative weight of a skipped key
under B: **1.0 on every layer** — the z-score tail skips the top-weighted key
everywhere, exactly as on MiniCPM3 and Gemma.

## Results (see artifacts/apa_sp5/RESULTS.md for the full treatment)

ppl @1024×4, bulk 4: **A 601.80, B 570.82, C 685.02, D 607.02, E 606.99.**
D−A = +5.22, inside the 12.89 aggregate floor. C's realised fraction 0.1681 vs
B's 0.1685 (registered tolerance ±0.01) — the δ match is validated.

**The RED:** B and C sit outside the aggregate floor in opposite directions,
and one window of four carries both results (window 1: A 1529.6, B 804.3,
C 2341.8; spread 2.91× against 1.05–1.36× elsewhere). Dropping it inverts the
ranking (B becomes worst). The swing is deterministic — a two-repeat check in
one process gave **bit-identical** arrays. **P3 is not decidable at N=4 on this
protocol.** No threshold was moved to make it decidable.

## Prior art (Prior Art Directive, David 2026-09-06)

Annotated at each code site, and here:

- **Running-max selection criterion — BLASST**, Yuan et al., arXiv 2512.12087
  (MLSys 2026). *Taken:* the comparator (compare each candidate to the running
  max online softmax already maintains, act within a log-ratio). *Not theirs:*
  APA **refines** the selected key to full precision and keeps every key in the
  softmax; BLASST **drops** the block. Same comparator, opposite consequence.
  APA is David's design; the single pass is Astra's (SP1); the impossibility
  proof for APA's z-score rule is SP1's.
- **Weight-proportional quantization error — ThriftAttention**, Sharratt,
  arXiv 2605.23081 (2026). *Taken:* the reason important keys carry the error.
- **Online softmax with running max and rescale** — Milakov & Gimelshein
  (2018); **FlashAttention-2** — Dao (2023). Standard, used by everyone here.
- **Learned attention sinks** — GPT-OSS model card (OpenAI, 2025);
  **StreamingLLM**, Xiao et al., arXiv 2309.17453 (2023) for the phenomenon.
  *Taken:* the convention that the sink logit joins the softmax denominator and
  contributes no value vector — implemented by the port, not by me.
- **Bulk key quantizer** — TurboQuant-style rotation + codebook (Zandieh et
  al., 2025), as implemented in `tensor_cuda.quant`.
- **Protocol shape** — SP3 PROTOCOL-2 and SP4G PROTOCOL-G (this repo, 2026);
  **wikitext-2** — Merity et al. (2016).
- **Sensitivity-floor method** — SP4G amendment 6 (this repo, 2026).
- **`flock(2)` foreground lease, `timeout(1)` leash** — Unix; the pattern is
  SP3/SP4G's `lead_gpu.sh` in this repo.
- **Mine in this order:** the arm wiring and sink-ablation construction, the
  window-wall sizing argument, and the observation that the sensitivity floor
  is heteroscedastic across windows with inconsistent sign (amendment 1). I
  know of **no prior art** for that last point specifically.
- **UNVERIFIED — lead to check.** This seat had no network. Every arXiv number
  above comes from the lead's own
  `/mnt/Shared/APA_SP_Prior_Art_Comparison_2026-09-06.md` or from seat
  knowledge. Search terms: "BLASST block-sparse attention running max",
  "ThriftAttention mixed precision block FP4", "TurboQuant vector quantization
  KV cache", "StreamingLLM attention sink", "gpt-oss-20b attention sinks model
  card", "FLASH-D Alexandridis 2025".

## Deviations

1. The N=4 sensitivity-floor cell ran at `APA_SP5_WORKER_RAIL=560` because two
   4-window passes plus a load is 286 s and cannot fit a 285 s worker rail.
   The order's 590 s **outer** bound was respected; flock held throughout;
   30 s cooldown honoured. Splitting the pair across leases would have compared
   two different model loads. Recorded in amendment 1.
2. Arm wiring monkey-patches `GptOssAttentionTC.__call__` in-process; the port
   file was never edited and its sha256 is pinned in the registration.
3. `QuantLinearTC.USE_FUSED = True` for the lm_head — an existing opt-in path,
   documented "correctness-identical", required to fit 927 MiB.
4. Items 4, 5 and 6 were not implemented as cells. What was measured instead,
   and what is genuinely missing, is stated in RESULTS.md §8. Not claimed done.

## Registered successors (out of scope here, named not absorbed)

- A bitwise-replay margin harness for item 4 (SP4G a2 pattern) at W and 8192,
  which is what seat prediction S5 needs.
- A decode cell for item 5, after measuring the per-step re-quantization cost
  the port's `_quantize_keys(k, ...)` on the whole concatenated k implies.
- Ceiling rungs 4096/8192/16384 per arm; long-lease 24K–96K only on David's
  authorization.
- Bulk 8, with C's δ recalibrated at 8 bits first.
- **A protocol question for David:** N=4 is what the 285 s rail permits, and it
  is not enough to resolve the tails on this model. More windows need a longer
  lease. That is a scope decision, not a seat decision.

---

# Amendment 1 (2026-09-08) — N=16, items 4/5/6 as cells

Lead ruling accepted: **a window is a cell**. Each PROTOCOL-O window is one
~46 s cell, so the instrument was N-bound, not lease-bound. Same seat (Opus 5,
`opus-max`, reasoning max), same lease discipline, no git, no subagents.

Amendment order sha256 `06a5a1e742ae2474469f8d5177bc5b9d35b4d89e9553d928c9cac1d7a6753e33`.
Predictions (registered BEFORE any new window cell)
`artifacts/apa_sp5/predictions_a1.json` sha256
`cbcb534d52f048898912d231870a7c8d26e3ac8a0b512bec0d9f1f0a81618330`.
Fingerprint `artifacts/apa_sp5/amendment_001_fingerprint.json` sha256
`22c700687246218896852a4e42c2e9b3a496a377c5a8d9e02017f51744019681` —
**additive only, expected_invalidated = []**; the r1 registration keeps its
exact bytes (verified: still `3cc3b3e1…58159`).

## Cells

| kind | count | status | wall each |
|---|---:|---|---:|
| `ppl_{A,A32,B,C,D,E}_W1024_w{00..15}` | 96 | all PASS | ~46 s (77 s incl. cooldown) |
| `aggregate_a1_N16` | 1 | PASS | <1 s, CPU |
| `margins_{B,C}_W1024_w{01,02}` | 4 | all PASS, **replay bitwise** | ~95 s |
| `decode_{A,B,C}_2048` | 3 PASS + 1 receipted OOM | | ~140 s |
| `ceiling_{B,C}_{4096,8192}` | 4 + 1 receipted OOM | 2 FITS, 2 RAIL | 130–246 s |

Windows 0–3 were **seeded** from the r1 N=4 receipts (CPU, no lease) after the
per-window runner was verified to reproduce the r1 code path **to the last
digit**: `ppl_C_W1024_w01 = 2341.776919703325` recomputed on the card equals
the r1 aggregate's window-1 value exactly. Each seeded cell carries a
provenance block naming its source file and sha256. The other 71 ran on the
card in batches of ≤12, each cell taking and releasing its own lease.

## Result — the r1 RED is resolved, and the N=4 reading is reversed

Pooled over 16 windows / **8,192 targets**:

| arm | pooled ppl | Δ vs A | in floors | verdict |
|---|---:|---:|---:|---|
| A | 232.27 | — | — | baseline |
| A32 | 230.72 | −1.54 | — | **the floor** |
| B | 273.47 | +41.21 | 26.7× | RESOLVABLE |
| C | 244.21 | +11.94 | 7.8× | RESOLVABLE |
| D | 232.77 | +0.50 | 0.33× | not resolvable |
| E | 232.75 | +0.48 | 0.31× | not resolvable |

**Pooled floor 1.54 ppl (0.66 %). C − B = −29.26 = 19.0× the floor,
RESOLVABLE, C better in 14 of 16 windows.** At N=4 the same comparison read
+114.2 the *other* way; window 1 alone carried it. D and E sit above A in 9/16
windows — a coin flip, which is what exactness should look like.

Floor as a distribution: per-window relative median **1.10 %**, mean 1.71 %,
min 0.23 %, max **8.73 % (window 0 — the window the original order named)**.
Signs mixed 9+/7−, so pooling cancels.

## Item 4 — the lead's question answered

Bitwise replay PASSED on all four cells (kernel output and, for arm C, the
kernel's own diagnostic mask, both reproduced exactly).

| arm/win | unrefined mass | fraction | sink mass |
|---|---:|---:|---:|
| B / w1 | **0.4891** | 0.1825 | 0.2881 |
| C / w1 | **0.1098** | 0.2301 | 0.2777 |
| B / w2 | 0.4871 | 0.1818 | 0.2680 |
| C / w2 | 0.1192 | 0.2592 | 0.2591 |

**A MASS story, not a sink story**: mass ratio B/C = **4.45×**, sink ratio
**1.04×**. The sink is a learned per-head scalar and cannot know the text.
Further: **B's and C's margins barely move between window 1 and window 2**, so
window 1 is not a window where the tails behave differently — it is a window
where the *same* mass deficit is punished harder by the text. Max relative
weight of a skipped key is 1.0 for both arms on every layer.

## Item 5 — decode as an adapter cost

A **82.58**, B **84.02**, C **84.18** ms/token at S=2048 (32 synced steps);
APA/standard = **1.02×**.

Cited line, `/mnt/ForgeRealm/GraftRepository/core/gpt_oss20b_tc.py:741`:
`kq = _quantize_keys(k, R, CB, BND)` — `k` here is the whole concatenated key
history, so **every step re-quantizes the entire cache**; `kq` appears only at
:741 and its use sites :747/:763, so nothing caches it. **No fix applied**, per
the amendment.

**Honest correction to my own registered prediction A1S5** ("several times
slower"): the gap is 1.02×. The re-quantization is real in source but is not
the dominant decode cost on a 20B MoE — expert routing is. On MiniCPM3 the
same pattern cost 7–14×. *The architecture, not the pattern alone, decides
whether an adapter cost matters.*

## Item 6 — ceilings

B and C both **FIT 4,096** (130.4 / 127.8 s, 735 MiB free) and both **RAIL at
8,192** (reached 7,680 in ~246 s with ~544 MiB still free — **time wall, not
memory**). Arm A remains a registered non-fit at 2,048.

Two distinctions worth keeping:
1. At 4,096 the binding constraint is the **prefill chunk transient**, not the
   KV cache: chunk 1024 OOMs *with 829 MiB free*, chunk 512 completes with 735.
   Both receipts kept.
2. Measured consumption **53.6 MiB per 1K tokens** → memory exhaustion
   extrapolates to ~**17,800 tokens**, so 16,384 should fit memory but needs
   ~525 s at the measured ~32 ms/token. Registered as a long-lease cell for
   David's authorization; **not run**.

## Failures and corrections in this amendment (RED honesty)

1. **One batch exceeded the 10-minute foreground limit** and the harness moved
   it to the background — the order forbids background tasks. I did not kill it
   (that would violate the no-kill rule and it held the GPU lease); I waited for
   it to drain and then sized every subsequent batch to fit the foreground
   window. Reported, not hidden.
2. **My prediction A1S5 was wrong** about decode magnitude, and the mechanism
   reasoning behind it was wrong too. Recorded as a MISS with the corrected
   mechanism.
3. **A1S1 was a near miss** — I registered "C above A in ≥12 of 16" and got 11.
4. **The first ceiling_C_4096 OOMed** at chunk 1024; rather than treat it as
   the ceiling, I varied the chunk and found the transient was the constraint.
   Both receipts kept so the distinction survives.

## Prior art (amendment 1)

No new algorithm. Bitwise replay = SP4G a2; ceiling ascent = SP4G a7; decode
construction = SP3 a6 / SP4G a6; pooled-NLL perplexity is standard practice
(the perplexity of the concatenated target set). The selection rule remains
**BLASST**'s running-max criterion (Yuan et al., arXiv 2512.12087) applied to
**precision** rather than sparsity — APA is David's design, the single pass and
the z-score impossibility proof are APA-SP1's; **ThriftAttention** (Sharratt,
arXiv 2605.23081) for weight-proportional error; **FlashAttention-2** (Dao
2023) / online softmax (Milakov & Gimelshein 2018); **GPT-OSS attention sinks**
(OpenAI model card 2025) and **StreamingLLM** (Xiao et al., arXiv 2309.17453).
Mine here: the seeding-with-provenance construction and the mass-vs-sink
decomposition that answers the lead's question — **no prior art known to me**
for that specific decomposition. **UNVERIFIED — lead to check**: no network in
this seat; all arXiv numbers come from the lead's own prior-art comparison doc
or seat knowledge.

## Deviations (amendment 1)

1. The backgrounded batch described above. No new worker-rail deviation: every
   amendment-1 cell ran under the standard 285 s worker rail.
2. `--prefill-chunk 512` / `--chunk 512` are required at 2,048+ and 4,096+
   respectively; both the failing and the passing receipts are kept so the
   choice is visible rather than silently baked in.
3. `ceiling_{B,C}_16384` not run (RAIL by construction under a 285 s worker);
   registered as a long-lease cell. Bulk 8 still not run.
