# APA-SP5 — GPT-OSS-20B — seat results

> **SUPERSEDED IN PART BY AMENDMENT 1 (N=16).** Sections 3, 5 and 6 below were
> written against the N=4 protocol and are kept verbatim as the r1 record.
> The instrument was extended to **N=16 windows / 8,192 targets** on the lead's
> ruling that a window is a cell; the r1 RED is **RESOLVED** and the N=4
> reading of C vs B **reversed**. Read **§10 AMENDMENT 1** first — it is the
> current result. Sections 1, 2, 4, 7 and 9 stand unchanged.


Seat: Opus 5 (`claude-opus-5[1m]`), reasoning effort: the `opus-max` seat
profile (max). Card: RTX 4070 SUPER 12 GB. Branch `apa-sp5`.
Registration `artifacts/apa_sp5/registration.json`
sha256 `3cc3b3e112479ad71077a98a221fd5af521013d16ce423d52afe308e9ae58159`.
Predictions sha256 `17427eb3c53f9830b195abc415cbb2d05a8e16099bdd30a2e76c296c99492e44`.
Amendment 1 sha256 `4cc5bca1ca6207bb6ad76cf2dd87c6a2f3f7db1ebad162ab08652381eaa5cd55`.

**Scope of proof.** Margins, ceilings and decode timings establish NOTHING
about model quality by themselves; they establish selection behaviour, memory
shape and speed. Only the perplexity rows speak to quality, and on this model
only above the registered floor. Evidence classes are named per section.

## 0. Verdict first

1. The single-pass kernel is **exact on a third architecture**, now including
   learned attention sinks and GQA 8:1 at D=64. fp32 per-call, real
   activations: max-abs 3.72e-5, relative Frobenius 6.6e-7 to 9.8e-7.
   **G_exactness_per_call PASS on both layers tested.**
2. **The sink is in the denominator, never in the selection.** Zeroing the
   learned sinks changes every arm's output by relF 0.41–0.48 and changes the
   single-pass selection mask by exactly **0 entries**. Pinned empirically,
   not just read from source.
3. **GPT-OSS-20B has the noisiest bulk keys of the three architectures** —
   4-bit error mean 0.56–0.89 per full layer against MiniCPM3's 0.25 and
   Gemma's 0.11 — and the **provable δ is vacuous** here: e_q = 19.86 gives
   δ = 44.3, which refines **99.999 %** of keys. Gemma's usable guarantee was
   a property of an unusually clean quantizer, not of the rule.
4. **APA buys real memory headroom on this model.** Standard **OOMs at
   W=2048**; the single pass runs it. The port materializes a dense
   B×H×L×S score tensor on full-attention layers (512 MiB at S=2048,
   2,048 MiB at S=4096) against 927 MiB of free VRAM. This inverts Gemma and
   reproduces MiniCPM3.
5. **The sink takes a quarter of the softmax mass at the median full layer**
   (0.2455), ranging 0.0125 (layer 11) to 0.6359 (layer 19). No previous model
   in this series had this quantity to measure.
6. **RED / honest limit:** the model-level arm comparison is not decidable at
   N=4 on this protocol. One window out of four (window 1) carries the entire
   B-vs-C result, and the per-window sensitivity floor is heteroscedastic by a
   factor of 38 (8.73 %, 1.24 %, 0.52 %, 0.23 %). See §5. Items 4, 5 and 6 were
   not implemented as cells; what is measured instead is stated in §8.

## 1. PROTOCOL-O (registered)

wikitext-2-raw-v1 test from the offline HF cache, the model's own harmony/o200k
tokenizer, **no chat template**, `add_special_tokens=False`.
287,210 tokens. `token_sha256`
`6b0ac3850d9fe65af588188ae37dd2172e77562dedb026260704777546ef2670`.

**W = 1024, N = 4**, consecutive non-overlapping windows, scoring the last 512
targets of each in fp64 → 2,048 targets.

Why not W=2048 (item 0, measured on the card):

| quantity | measured |
|---|---:|
| resident model load, warm page cache | 12.5 s |
| resident model load, cold page cache | 37.2 s |
| resident VRAM after load | 10,927 MiB |
| free VRAM after load (`cudaMemGetInfo`) | **927 MiB** |
| one W=1024 window forward, arm A | 33.9 s |
| one W=2048 window forward, arm C | 67.5 s |
| W=2048 × N=4 projected | 284.6 s vs a 285 s rail |
| **W=2048, arm A** | **OOM** |

W=2048×N=4 clears the rail by 0.4 s — no margin — and, decisively, **the
baseline arm cannot execute at W=2048 at all**. A protocol whose baseline arm
OOMs is not a protocol. W=1024×N=4 costs 150 s, every arm runs, 135 s of margin.

The long row (last 512 within 8192) is **registered NON-FIT for arm A**:
standard already OOMs at 2048, and 8192 would need an 8,192 MiB score tensor.

**Two harness facts that are findings, not conveniences:**

- The lm_head must go through `int4_linear_fused`. The default two-stage
  `int4_linear` dequantizes the whole head weight to a 2880×201088 fp16 buffer
  = **1,105 MiB** and OOMs at *every* chunk size including M=1, against 927 MiB
  free. `kernels.cu` states the fused path is "Correctness-identical".
- The fp32 sensitivity reference must be chunked over query positions: a dense
  fp32 64×1024×1024 score tensor is 256 MiB and `sink_attention_tc` holds
  several at once. Per-row softmax is independent, so chunking is
  arithmetically identical.

## 2. The sensitivity floor (item 1) — evidence class: model perplexity

Standard bf16 vs standard with the 12 full-attention layers in fp32.

| scale | A bf16 | A32 fp32-full | floor | relative |
|---|---:|---:|---:|---:|
| window 0 (the registered construction) | 673.12 | 614.36 | **58.76** | **8.73 %** |
| N=2 aggregate | 1014.70 | 975.38 | 39.32 | 3.88 % |
| N=4 full protocol | 601.80 | 588.91 | **12.89** | **2.14 %** |

Per window: +58.8 / **−18.9** / +1.4 / +1.1. **The perturbation does not have a
consistent sign**, so averaging cancels rather than accumulates — that is why
the aggregate floor is smaller, and it is a finding (amendment 1), not a
tightening of a registered threshold. GPT-OSS-20B is *more* sensitive than
Gemma 4 per window (8.73 % vs ~4.9 %) but *less* at protocol scale.

**The per-window floor is heteroscedastic by 38×** (8.73 %, 1.24 %, 0.52 %,
0.23 %). Window 0 — the one the order names — is the least stable of the four.

## 3. Perplexity, bulk 4 (item 2) — evidence class: model perplexity

| arm | rule | δ | realised fraction | ppl @1024×4 | Δ vs A |
|---|---|---:|---:|---:|---:|
| A standard (sink attention) | — | — | 1.000 | **601.80** | — |
| B two-pass z-score, r=0.15 | port default | — | 0.1685 | **570.82** | −30.98 |
| C single-pass, matched | running max | 3.16 | **0.1681** | **685.02** | +83.22 |
| D single-pass refine-all | — | 1e9 | 1.000 | **607.02** | +5.22 |
| E single-pass provable δ | ln(1/ε)+2e_q | 44.315 | **0.99999** | **606.99** | +5.20 |

C's realised fraction 0.1681 matches B's 0.1685 to **0.0004**, far inside the
registered ±0.01 — the δ match is validated on the real forward, not assumed.

D − A = **+5.22, inside the 12.89 aggregate floor**: refine-all equals standard
at model level on this architecture. E − D = **0.03**, because E refines
everything.

## 4. Sinks (item 2 pin) — evidence class: kernel sweep + source

Where the sink enters, per arm:

- **A standard**: concatenated as one extra score column, softmax over S+1
  columns, sink column **sliced off** before the value matmul. Denominator only.
- **B two-pass**: folded inside the fused sink kernel at the end of the key
  loop; the z-score threshold is computed on |bulk| over real keys only.
- **C/D/E single pass**: folded into the running max **after** the key loop
  (`kernels.cu apa_selective_sp_kernel`: `if (sinks) {...}` runs once, after
  every key has been decided) — denominator plus accumulator rescale.

Pin on layer 19 (highest sink mass), identical real activations, sinks on vs
zeroed:

| arm | output change (relF) | selection mask entries changed |
|---|---:|---:|
| A | 0.4758 | n/a (refines everything) |
| B | 0.4106 | not exposed by that entry |
| C (δ=3.16) | 0.4758 | **0** |
| D (refine-all) | 0.4758 | **0** |

**Verdict CONFIRMED**: the sink is load-bearing for the output and provably
irrelevant to the selection. The learned sink logits on layer 19 range
1.109 to 4.156, mean |sink| 2.190.

Sink softmax mass per query, median per full layer:

| layer | 1 | 3 | 5 | 7 | 9 | 11 | 13 | 15 | 17 | 19 | 21 | 23 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sink mass | .253 | .304 | .391 | .067 | .108 | .013 | .013 | .113 | .238 | **.636** | .575 | .403 |

Median of layer medians **0.2455**; 9 of 12 layers at or above 10 %.

## 5. RED — what this protocol cannot decide, and why

B and C are both **outside** the aggregate floor, in **opposite directions**,
and both results live in one window:

| window | A | B | C | D | E | spread |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 673.1 | 692.5 | 669.0 | 657.5 | 657.5 | 1.05× |
| **1** | **1529.6** | **804.3** | **2341.8** | 1585.4 | 1584.6 | **2.91×** |
| 2 | 270.6 | 297.1 | 278.6 | 272.8 | 272.8 | 1.10× |
| 3 | 470.7 | 641.6 | 504.6 | 477.5 | 477.7 | 1.36× |

Drop window 1 and the ranking **inverts**: A 440.97, B **509.16** (now the
worst), C 454.73, D 440.78, E 440.83.

Each arm's per-window deviation from A, expressed in multiples of *that
window's own* floor:

| arm | w0 | w1 | w2 | w3 |
|---|---:|---:|---:|---:|
| B | +0.3× | **−38.3×** | +18.8× | **+155.7×** |
| C | −0.1× | **+42.9×** | +5.6× | +30.8× |
| D | −0.3× | +2.9× | +1.5× | +6.2× |
| E | −0.3× | +2.9× | +1.5× | +6.3× |

D and E sit at 1.5–6× their local floor — small and consistent, which is what
"the kernel is exact and the tail refines everything" should look like. B and C
swing 20–155× in opposite directions.

**This is not engine noise.** A determinism check (SP3 G0 construction) reran
arm C's window 1 twice in one process: **bit-identical, max-abs difference
0.0**, ppl 2341.776919703325 both times. The swing is a real, reproducible
property of the arms on that text.

**Honest verdict: P3 (C − B negative or inside the floor) is NOT DECIDABLE at
N = 4 on this protocol.** As written it is a MISS (C − B = +114.2, outside the
floor and positive), but the number is one window's artifact and the sign
reverses without it. The registered N=4 is what the 285 s rail permits; the
right fix is more windows, which is a longer-lease decision for David, not a
threshold this seat may move. **I did not widen the gate to pass it.**

## 6. Predictions

| owner | prediction | verdict |
|---|---|---|
| lead P1 | per-call SP == standard ≤ 1e-5 relF fp32 incl. sinks | **HIT** — 6.6e-7 to 9.8e-7 on real activations; 2.3e-7 synthetic |
| lead P2 | bulk error mean 0.15–0.25; z-score leaves 50–80 % of mass | **MISS / HIT** — error mean 0.56–0.89 (3× MiniCPM3, 6× Gemma: the noisiest of the three, not intermediate); unrefined mass 0.344–0.677, median 0.484 |
| lead P3 | C − B negative or inside the floor | **NOT DECIDABLE at N=4** (as written, MISS: +114.2; sign reverses without window 1) |
| lead P4 | E refines 40–90 % | **MISS** — 99.999 %; δ = 44.3 is vacuous |
| lead P5 | sink ≥ 10 % of mass on a median full layer; selection unaffected | **HIT / HIT** — 24.55 % median-of-medians, 9/12 layers ≥ 10 %; selection mask 0 entries changed |
| lead P6 | no OOM by 16K; standard fastest unless the port materializes S×S | **REFUTED on both halves** — the port DOES materialize S×S on full layers; standard OOMs at 2048 |
| seat S1 | per-call gate passes on real activations; sink adds no exactness risk | **HIT** |
| seat S2 | D−A large in absolute ppl but not resolvable | **HIT** (+5.22, inside 12.89) |
| seat S3 | E refines ≥ 95 %; Gemma was the quantizer, not the rule | **HIT** (99.999 %) |
| seat S4 | C−B negative in sign, magnitude swamped by the floor | **MISS on sign** at N=4 (+114.2); the "swamped" half is the finding in §5 |
| seat S5 | high-sink layers show higher unrefined real-key mass | **NOT TESTED** — needs the item-4 replay harness (§8) |
| seat S6 | APA arms reach 4096/8192, standard neither; time rails before memory | **half HIT** — standard confirmed OOM at 2048; the APA rungs were not run |

Seat S3 and S6's standard half named a mechanism before the card did. Seat S4's
sign call was wrong, and the reason it was wrong (one window dominating) is
worth more than the prediction was.

## 7. What GPT-OSS-20B adds to the APA record

1. Exactness survives **learned attention sinks and GQA 8:1 at D=64** — the
   third architecture, first with a term in the softmax denominator that is not
   a key. The sink cannot corrupt the running-max rule because it is folded
   after every decision; measured, not argued.
2. **The tail-choice penalty scales with bulk error** (SP4G principle 1) is
   now tested at a third, higher error level — but the model-level readout is
   swamped by window variance, so this test *neither confirms nor refutes it*.
   Honest null under a stated limit.
3. **The provable δ's usefulness tracks the quantizer's worst case, not the
   rule.** e_q max 19.86 vs p99 4.90 vs mean 0.71 — a 28× outlier ratio makes
   the bound vacuous. MiniCPM3 100 %, GPT-OSS 99.999 %, Gemma 69 %.
4. **APA's memory lever exists only where the dense path materializes S×S**
   (SP4G principle 4) — confirmed by its positive case here. Gemma's chunked
   standard path gave APA nothing; GPT-OSS's unchunked `sink_attention_tc`
   gives APA the difference between "runs" and "OOM" at 2048.
5. **New:** the sensitivity floor is not a single number. It is heteroscedastic
   across windows (38× here) and its sign is not consistent, so a floor
   measured on one window is the wrong instrument for a multi-window aggregate.
   SP4G's principle 3 should be amended to "measure the floor at the scale of
   the claim".

## 8. Not delivered by this seat (honest residuals)

- **Item 4 margins as a bitwise replay.** The calibration cell measures every
  item-4 quantity (|bulk−exact| mean/p99/max, unrefined softmax mass, max
  relative weight of a skipped key, realised fraction, and the new sink mass
  per query) on real activations at W=1024 for all 12 full layers — but it
  recomputes the selection in fp64 rather than replaying the kernel's own mask
  bitwise as SP4G a2 does, and it does not cover 8192. Seat prediction S5 needs
  that harness. **Not claimed done.**
- **Item 5 clean decode.** Not run. Blocking fact the lead needs first, verified
  in source: `GptOssAttentionTC.__call__` recomputes
  `kq = _quantize_keys(k, ...)` over the **whole concatenated k every call**
  (`gpt_oss20b_tc.py:741`); nothing in the port caches quantized keys. **This
  adapter re-quantizes per step**, like MiniCPM3 and unlike Gemma. On the SP3
  precedent that predicts a large APA decode gap that is an adapter cost, not a
  kernel cost — but it is a prediction, not a measurement.
- **Item 6 ceiling as a cell.** The decisive rung is measured (arm A OOMs at
  2048; APA arms run 2048 at 67.5 s). The 4096/8192/16384 rungs per arm were
  not run; estimates and the commands are in `lead_commands.txt`.
- **Bulk 8 secondary.** Not run. Note C's δ=3.16 is calibrated at bulk 4 and
  must be recalibrated at bulk 8 before an arm-C bulk-8 cell means anything.

## 9. Deviations

1. **One cell exceeded the 285 s worker rail.** The N=4 sensitivity floor needs
   two full 4-window passes (2 × 136 s) plus a 12.5 s load = 286 s, which
   cannot fit 285 s by construction. Run at `APA_SP5_WORKER_RAIL=560` under the
   order's 590 s outer bound, flock held throughout, 30 s cooldown honoured.
   Splitting the pair across two leases would have compared two different model
   loads, which the measurement laws forbid. Recorded in amendment 1.
2. **Arm wiring is done by monkey-patching `GptOssAttentionTC.__call__` inside
   the harness process.** The port is READ-ONLY and was never edited; its
   sha256 is pinned in the registration.
3. **`QuantLinearTC.USE_FUSED = True`** is set by the harness for the lm_head.
   This is a documented opt-in flag on an existing path, required to fit the
   card; it is not a kernel change.
4. **No launcher addition was made and none was needed** —
   `apa_selective_attention_sp` already dispatches `cap <= 64`, arbitrary GQA,
   and a sinks tensor. `git status tensor_cuda/` is **empty**; `TC_APA_SP`
   remains default OFF and is set per-process only.


---

# 10. AMENDMENT 1 — N=16, and the resolution of the r1 RED

Lead ruling: *"a window is a cell."* Correct — each PROTOCOL-O window is one
~46 s cell, so the instrument was never lease-bound, it was N-bound. The set
is now **96 window cells** (A, A32, B, C, D, E × 16 windows), all PASS.
Windows 0–3 were seeded from the r1 receipts after the per-window runner was
verified to reproduce the r1 path **to the last digit**
(`ppl_C_W1024_w01 = 2341.776919703325` recomputed on the card = the r1
window-1 value exactly); every seeded cell carries a provenance block naming
its r1 source and sha256.

Amendment predictions: `artifacts/apa_sp5/predictions_a1.json`
sha256 `cbcb534d52f048898912d231870a7c8d26e3ac8a0b512bec0d9f1f0a81618330`.
Fingerprint: `artifacts/apa_sp5/amendment_001_fingerprint.json`
sha256 `22c700687246218896852a4e42c2e9b3a496a377c5a8d9e02017f51744019681`
(**additive only — nothing from r1 invalidated, relaxed or edited**).

## 10.1 The pooled result (8,192 targets)

| arm | pooled ppl | Δ vs A | in pooled floors | verdict |
|---|---:|---:|---:|---|
| **A** standard | **232.27** | — | — | baseline |
| **A32** fp32 full layers | **230.72** | −1.54 | — | *this is the floor* |
| **B** two-pass z-score | **273.47** | **+41.21** | **26.7×** | **RESOLVABLE** |
| **C** single-pass matched δ=3.16 | **244.21** | **+11.94** | **7.8×** | **RESOLVABLE** |
| **D** single-pass refine-all | **232.77** | +0.50 | 0.33× | not resolvable |
| **E** single-pass provable δ | **232.75** | +0.48 | 0.31× | not resolvable |

**Pooled floor = 1.54 ppl (0.66 %).** **C − B = −29.26 = 19.0× the floor,
RESOLVABLE.**

Sign counts across the 16 windows: **C below B in 14/16**; B above A in 15/16;
C above A in 11/16; **D and E above A in 9/16 each — a coin flip, which is what
"exact" should look like.**

## 10.2 What this reverses, and what it confirms

**The r1 N=4 reading was an artifact of one window and is now overturned.**

| claim | at N=4 | at N=16 |
|---|---|---|
| C vs B | C **worse** by +114.2, "not decidable" | **C better** by −29.26, resolvable at 19× the floor, better in 14/16 windows |
| D vs A | +5.22, inside a 12.89 floor | **+0.50 against a 1.54 floor** — a far tighter statement of the same conclusion |
| pooled floor | 12.89 ppl (2.14 %) | **1.54 ppl (0.66 %)** |

So on GPT-OSS-20B the single-pass running-max tail **beats** the two-pass
z-score tail at matched refine fraction (0.1681 vs 0.1685) — **the SP3
MiniCPM3 result, not the Gemma null.** That is the third architecture and the
second confirmation, and it is consistent with the r1 mechanism: these are the
noisiest bulk keys of the three, and tail-choice penalty scales with bulk error.

## 10.3 The floor is a distribution, not a number

Per-window relative floor across 16 windows: **median 1.10 %, mean 1.71 %,
min 0.23 %, max 8.73 %.** Window 0 — the window the original order named for
the floor — is the **most extreme of sixteen**. Signs are mixed (9 positive,
7 negative), so pooling **cancels** scatter rather than accumulating it; that
is why the pooled floor (0.66 %) is far below any single window's.

Each arm's per-window deviation from A, in multiples of **that window's own**
floor (full table in `aggregate_a1_N16.json`):

| arm | range across windows | reading |
|---|---|---|
| B | +1.3 to **+256.2** (15/16 positive) | systematically and hugely above A |
| C | −4.5 to **+42.9** (11/16 positive) | mostly above A, one extreme window |
| D | −6.4 to +6.3 | scattered about zero |
| E | −6.7 to +6.3 | scattered about zero |

## 10.4 Item 4 — bitwise-replay margins: a MASS story, not a sink story

Four cells, **replay bitwise on all of them** (kernel output *and*, for arm C,
the kernel's own diagnostic selection mask reproduced exactly). Medians over
the 12 full-attention layers, every 32nd query position, all 64 heads:

| arm / window | unrefined mass | fraction | sink mass | e_mean | max skipped rel. weight |
|---|---:|---:|---:|---:|---:|
| B / w1 | **0.4891** | 0.1825 | 0.2881 | 0.705 | 1.00 |
| B / w2 | 0.4871 | 0.1818 | 0.2680 | 0.684 | 1.00 |
| C / w1 | **0.1098** | 0.2301 | 0.2777 | 0.699 | 1.00 |
| C / w2 | 0.1192 | 0.2592 | 0.2591 | 0.672 | 1.00 |

**The answer to the lead's question.** On window 1 the z-score tail leaves
**4.45× more softmax mass unrefined** than the running-max tail (0.489 vs
0.110) — at a *lower* refine fraction (0.182 vs 0.230). The sink mass is
**identical between the arms (1.04×)**, as it must be: the sink is a learned
per-head scalar that cannot know what text is in the window. **It is a mass
story, not a sink story.**

The second half is just as informative: **B's and C's margins are essentially
the same on window 1 and window 2** (B 0.489 → 0.487; C 0.110 → 0.119). So
window 1 is *not* a window where the tails behave differently — it is a window
where the **same** mass deficit is punished far harder by the text. The 2.9×
ppl spread is text sensitivity acting on a constant selection deficit, not a
selection anomaly.

Per-layer on window 1, B leaves 0.32–0.69 of the mass unrefined against C's
0.065–0.178, and **the max relative weight of a skipped key is 1.0 for both
arms on every layer** — both tails skip some query's top-weighted key
somewhere; only the *mass* differs.

## 10.5 Item 5 — decode, reported as an adapter cost

32 synced steps at S=2048, clean configuration, prefill in 512-token chunks:

| arm | ms/token (mean) | median | vs standard |
|---|---:|---:|---:|
| A standard | **82.58** | 80.44 | 1.00× |
| B two-pass | **84.02** | 82.71 | **1.02×** |
| C single-pass | **84.18** | 83.54 | **1.02×** |

**The cited adapter line** (`/mnt/ForgeRealm/GraftRepository/core/gpt_oss20b_tc.py:741`):

    kq = _quantize_keys(k, R, CB, BND)

sits *after* `k` has been concatenated with the whole `kv_cache`, so **every
decode step re-quantizes the entire key history**; nothing in the port caches
quantized keys (`kq` appears only at :741 and its two use sites :747/:763).
Standard does no quantization and pays none of it.

**And yet the gap is 1.02×.** That is a correction to my own registered
prediction A1S5 ("several times slower"): the re-quantization is real in
source but is **not** the dominant decode cost on this model, because the MoE
expert routing dominates the step. The honest statement is: *the adapter does
re-quantize per step, and on a 20B MoE that cost is invisible next to routing.*
On MiniCPM3 the same adapter pattern cost 7–14×; the architecture, not the
pattern alone, decides whether it matters.

A receipted OOM is kept: `decode_a1/decode_A_2048.json` — **arm A OOMs during
a 1024-token prefill chunk** even before decode starts. Another S×S receipt.

## 10.6 Item 6 — ceilings (OOM and RAIL distinguished)

| arm | S | chunk | outcome | reached | wall | min free MiB |
|---|---:|---|---|---:|---:|---:|
| B | 4,096 | 512 | **FITS** | 4,096 | 130.4 s | 735 |
| C | 4,096 | 512 | **FITS** | 4,096 | 127.8 s | 735 |
| C | 4,096 | 1024 | **OOM** | — | 67.3 s | 829 |
| B | 8,192 | 512 | **RAIL** | 7,680 | 246.2 s | 544 |
| C | 8,192 | 512 | **RAIL** | 7,680 | 245.2 s | 543 |

Arm A stays a **registered non-fit at 2,048** (r1 receipt kept).

Two findings:

1. **At 4,096 the binding constraint is the prefill CHUNK TRANSIENT, not the
   KV cache.** Chunk 1024 OOMs *with 829 MiB free*; chunk 512 fits and runs to
   completion with 735 MiB free. That distinction matters for anyone sizing
   this model and would have been invisible without recording both.
2. **At 8,192 the wall is TIME, not memory** — both arms stopped at the budget
   with ~544 MiB still free. Measured consumption is **53.6 MiB per 1K tokens**;
   with 735 MiB free at 4,096 the line extrapolates to memory exhaustion near
   **17,800 tokens**. So 16,384 should *fit memory* but needs ~525 s (~32
   ms/token measured), which is over any 285 s worker rail — registered as a
   long-lease cell for David's authorization, not run.

## 10.7 Amendment predictions — verdicts

| owner | prediction | verdict |
|---|---|---|
| **lead** | pooled C − B lands **inside** the pooled floor; window 1 stays an outlier in both directions | **MISS on the first half** (C − B = −29.26 = 19× the floor, decisively resolvable); **HIT on the second** (window 1 is the sole window where B beats C, at +1537 on C − B) |
| **lead (2nd branch)** | if instead C is consistently above A by more than the per-window floor, that is the finding and it points at the δ=3.16 selection | **THIS IS THE BRANCH THAT FIRED, partially**: C is above A in 11/16 windows and resolvable pooled (7.8× floor) — but B is above A in **15/16** at **26.7×**, so the finding points at *both* tails leaving mass at 4 bits, with the z-score tail much the worse |
| seat A1S1 | C above A in ≥12 of 16 | **NEAR MISS — 11 of 16.** Directionally right, one window short of the sharp form I registered |
| seat A1S2 | pooled C − B still outside the pooled floor at N=16 | **HIT**, and for the stated reason: pooling shrank the floor (12.89 → 1.54) faster than the arm difference |
| seat A1S3 | window 1 is a mass story, not a sink story | **HIT** — 4.45× mass ratio, 1.04× sink ratio |
| seat A1S4 | floor distribution right-skewed, median well under 8.73 %; pooled floor smaller than N=4's | **HIT** — median 1.10 %, pooled 0.66 % vs 2.14 % |
| seat A1S5 | APA decode several times slower than standard; SP and two-pass within ~1.5× of each other | **MISS on the gap** (1.02×, not "several times"); HIT on the shape (B and C within 1.002× of each other). My mechanism reasoning was wrong: re-quantization is real but not dominant on a MoE |
| seat A1S6 | B and C reach 4096 and 8192 under the normal rail; both RAIL at 16384 | **half** — 4096 FITS (and only with 512-chunks, which I did not anticipate); 8192 RAILs rather than fitting; 16384 untested but RAIL-by-construction as predicted |

I got the mass-vs-sink decomposition and the floor distribution right, and I
was **wrong about decode** and one window short on C-vs-A. The lead's first
branch missed but its second branch is the one the data chose.

## 10.8 What amendment 1 changes in the APA record

1. **The single pass beats the two-pass tail on a second architecture.**
   MiniCPM3 (noisy keys): −0.118 ppl. GPT-OSS-20B (noisiest keys): **−29.26
   pooled, resolvable at 19× the floor, better in 14/16 windows.** Gemma
   (cleanest keys): no difference. The **tail-choice penalty scales with bulk
   error** principle now has three points and holds at all three.
2. **Refine-all equals standard at model level to 0.50 ppl against a 1.54 ppl
   floor** — the tightest model-level exactness statement in the series,
   because the floor got smaller rather than the claim getting louder.
3. **A protocol lesson worth more than the arm result:** N=4 gave a
   *confidently wrong* answer (C worse than B) that N=16 reversed. The floor
   was not the problem; the sample was. Registering "not decidable" at N=4
   rather than reporting the artifact is what made the reversal visible instead
   of embarrassing.
4. **An adapter cost can be real in source and invisible in the measurement.**
   The port genuinely re-quantizes every decode step; on this MoE it costs 2 %.
   Cite the mechanism, but measure before pricing it.
