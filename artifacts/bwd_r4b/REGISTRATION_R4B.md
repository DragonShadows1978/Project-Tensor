# BWD-R4b registration (sealed before any GPU line is staged)

Orders: `orders/BWD_R4B_FULL_CHAIN.md` (c4164f43), Amendment 1 `orders/BWD_R4B_R2_AMENDMENT1.md` (506ec818),
Amendment 2 `orders/BWD_R4B_R3_AMENDMENT2.md` (709f3a69). `AMENDMENT1_R4B.md` sha256 7532d2c4b1005e606d1de4f2d1805bb40ff311cb48c19c88c354bcaab3af048e,
`AMENDMENT2_R4B.md` sha256 0dd5f5a24134146e394e0205fd537fe6096d00a6d978570fc133c621a3e65450 (both verbatim, folded in below). Start
records: LEDGER_R4B.md lines "START", "r2 start", "r3 start". L20 deviation (stated by both amendments): the
amendments were registered after fp64-only floor data and before any candidate data; no bar is widened.

## 1. Operands
- Crop: R4a's `get_cases('crop')` = BP1 `crop_pair` duration frames [86, 126], target half-frames
  [162, 242]; fp32 inputs and the fp32 source spectrum, converted to fp64 for the reference (DESIGN §6).
- Linear loss <wave, r>: r = bp1_run PROBE_SEED 41059 standard normal, unit L2, stored fp32 (sha <f4:[24000]:8d12082aea498d5a52a2fc019958fb575cd996897d7628d961bb824857c00a86).
- MFCC loss: Voice `mfcc.py` `mfcc_loss` / `mfcc_loss_vjp` with crop_pair's target and pairs, on the host in fp64 at
  the candidate's fp32 wave; dL/dwave returns as fp32. The fp64 loss is evaluated at the candidate's fp32 wave.
- 10 s (G5): R4a's `get_cases('10s')` rule; MFCC target = LJ001-0004 tiled to 240,000 samples, pairs = frame i with
  frame i over all 801 frames.
- Trained: 315 tensors (313 pre-tail + conv_post `onnx::Conv_8064`, `...conv_post.bias`) and s_d.
  Frozen: both iSTFT bases, har, the source spectrum, F0, N, asr. Flagged: 12 identity norms, the 29-row zero set.

## 2. Bars (unchanged from the order; none widened)
- G0: wave float32 bytes bitwise equal to Voice's KokoroTC forward (generator + fused tail) on the crop and the full
  LJ001-0004 input, harness grad off (no checkpoint) and grad on under both schedules (8 and 31 segments).
- G1: R4a's classes, `TOLERANCES_R4.json` sha256 27052e947b4704ec817ce5b962c94f7018d3143a2d802288fc888dfa002699be: style relative L2 <=
  0.0001; parameters relative L2 <= 0.001, or absolute L2 <=
  1e-06 x largest where the reference norm < 1e-06 x largest.
  Every compared array finite, non-empty, nonconstant (L19). Identity-norm, zero-set and conv_post rows reported
  separately. A1-3: zero-set rows must fall in the absolute-floor class on the candidate.
- G2: Amendment 3 (`artifacts/bwd_r4/AMENDMENT3_R5C.md`, b2605789...): |cand - fd| <= rho |fd| on every slope of the
  chosen triple, rho 1e-4 (style) / 1e-3 (parameters); strength floors (A2 0.1 L2(g) L2(v); A3-1 0.1 max|g| for
  coordinates); A3-2 triples; power: zeroing and negating each coordinate projection fails the bar.
- G3: checkpointed (8 and 31 segments) vs plain, per tensor, G1 classes, both losses; plain and checkpointed waves
  bitwise equal.
- G4: per-tensor gradient sha256 identical over 3 in-process runs and in a fresh process, both schedules, per loss.
- G5: 31 segments, R4a digests off, no allocator trim: NVML per-process usedGpuMemory peak through forward, MFCC loss
  and backward <= 4,000,000,000 bytes; sampler median interval <= 5 ms (poll period 2 ms); peak recomputed from the
  stored series (L44). Recorded: R4a's reserved-plus-non-pool basis, seconds per phase. No-checkpoint arm report only.
- G-API: `tc_signal/tests/test_bwd_r4b_public_conv_backward.py` on the GPU with the BP1/BP2 fixture roots: every
  test passes, no skip (R3's conv class, `TOLERANCES_R3.json` sha256 8f6c52e38d8a653aaaf9c1b51453fa8031b1f50f59444a3bf436262fa4845f2b,
  and bitwise equality with the tracked path); host refusal tests pass on CPU.

## 3. References (cotangent census, Amendment 2)
| Gate / line | Reference, oracle or selection | Cotangent | Governing rule | Gated |
|---|---|---|---|---|
| g1-linear G1 | fp64 chain fed fp64(r32) | shared r | order G1 | yes |
| g1-linear G2 | `host-fd-linear` oracle (sha e26dc56bc61b8e6b), directions npz bcb77e7cd76b7b63 | shared r | order G2, Amendment 3 | yes |
| g1-mfcc G1, r4bm03 | `ref-at-wave/reference/REF_MFCC_AT_WAVE_R4B.npz`: fp64 backward fed mfcc_loss_vjp(wave_c) in fp64 | at wave_c, computed by the reference | A1-1 | yes |
| g1-mfcc G2 | `ref-at-wave/merge` A2-1 oracle of <g_c, wave(theta)>, probes selected from the A1-1 gradient | g_c at wave_c, fixed | A2-1 | yes |
| g1-mfcc A1-2 | fp64 chain from the fp64 wave; in-situ floor from ref-at-wave | at the fp64 wave | A1-2 | report only |
| g1-mfcc A2-2 | round 1's `host-fd-mfcc` oracle (sha 6d632af3ef1d1e31) with ref-at-wave's shift | at the fp64 wave | A2-2 | report only |
| ref-at-wave | computes g_c from wave_c itself (never the candidate's upstream) | at wave_c | A1-1, A2-1 | producer |
| g3 (MFCC), g4-*-mfcc, g5-* | candidate against candidate; each run's cotangent from its own wave | candidate's | order G3/G4/G5 | yes (no fp64 reference) |
No gated comparison references the fp64-wave cotangent; the only ones that do are A1-2 and A2-2 (report only).
Operand census: `logs/bwd_r4b_cotangent_census.log`.

## 4. Probes and seeds
- Linear G2: 40 rows (8 slots: s_d, R4a's six G2 tensors, conv_post w; aligned + 3 largest-|g| coordinates each;
  8 s_d directions, PCG64 seed 0x593FB9A3, filter 0.125), fixed from the fp64 reference: `host-fd-linear`.
- MFCC G2 (A2-1): the same rule applied to the A1-1 reference gradient at wave_c, inside ref-at-wave; depends on
  wave_c only (G0 pins the forward bitwise), never on the candidate's backward.

## 5. Floors (measured before sealing)
- G1 linear: 0 (shared r). G1 MFCC under A1-1: cast floor on P2 = s_d 2.69e-03 of its bar,
  worst parameter 1.69e-03 (onnx::Conv_7610); rows within 10x:
  0 (`host-a1-floors/A1_FLOORS_R4B.json`).
- A1-3 zero set (29 rows; the tail adds none): linear:fp64_wave: max in-set 1.45e-17 of largest, gap 1.74e+12; linear:p2_wave_a1_reference: max in-set 1.45e-17 of largest, gap 1.74e+12; mfcc:fp64_wave: max in-set 1.09e-17 of largest, gap 4.9e+12; mfcc:p2_wave_a1_reference: max in-set 8.16e-18 of largest, gap 6.52e+12.
- G2 h vs h/2: linear worst 9.61e-04 of rho (kmodel.decoder.generator.noise_convs.1.weight); round-1 MFCC worst 1.01e-03 of rho (kmodel.decoder.generator.noise_convs.1.weight); A2-1 on the P2
  stand-in worst 9.64e-04 of rho (kmodel.decoder.generator.noise_convs.1.weight) (40/40 GREEN, `selftest-ref-at-wave-P2`).
- A2-1 rail (cast shift on each A2-1 probe at P2): worst 2.04e-03 of rho; rows within 10x:
  0 (`host-a2-floor/A2_FLOOR_R4B.json`).
- Report-only context: round 1's fp64-wave G1 floor (s_d 0.63/0.83 of its bar at P2/P2k, `FLOORS_G1_R4B.json`) and
  G2 forward floor (worst 0.58/0.77
  of rho) now apply only to A1-2 and A2-2.

## 6. Lines (STAGED_LINES_R4B.md; lead runs them)
| Line | Item | Depends on | Expected seconds | Min / max | Expected peak host bytes |
|---|---|---|---:|---:|---:|
| `g0-crop` | G0 crop, grad off/on, both schedules | - | 60 | 120 / 540 | 4,000,000,000 |
| `g0-full` | G0 full LJ001-0004, grad off/on, both schedules | - | 90 | 180 / 540 | 4,500,000,000 |
| `wave-capture` | A1-1 step 1: wave_c, both schedules byte-identical | - | 45 | 90 / 540 | 3,500,000,000 |
| `g1-linear` | G1 + G2 linear, both schedules | - | 120 | 240 / 540 | 5,000,000,000 |
| `g1-mfcc` | A1-1 G1 + A2-1 G2 MFCC, A1-2/A2-2 report | wave-capture, ref-at-wave | 150 | 300 / 540 | 6,500,000,000 |
| `g3` | G3 both losses, both schedules vs plain | - | 120 | 240 / 540 | 5,000,000,000 |
| `g4-session-linear` | G4 in-process x3, both schedules | - | 120 | 240 / 540 | 4,500,000,000 |
| `g4-session-mfcc` | G4 in-process x3, both schedules | - | 120 | 240 / 540 | 4,500,000,000 |
| `g4-fresh-linear` | G4 fresh process | g4-session-linear | 70 | 140 / 540 | 4,500,000,000 |
| `g4-fresh-mfcc` | G4 fresh process | g4-session-mfcc | 70 | 140 / 540 | 4,500,000,000 |
| `g5-checkpoint` | G5 10 s, 31 segments, NVML per-process peak | - | 220 | 440 / 540 | 4,500,000,000 |
| `g5-plain` | G5 no-checkpoint arm (report only) | - | 180 | 360 / 540 | 4,500,000,000 |
| `g-api` | G-API public conv backward, every BP1/BP2 geometry | - | 240 | 480 / 540 | 3,000,000,000 |
| `r4bm01_public_dw_transposed_kernel` | mutant, caught by G-API | g-api | 240 | 480 / 540 | 3,000,000,000 |
| `r4bm02_wave_upstream_dropped` | mutant, caught by G1 (linear) | - | 100 | 200 / 540 | 5,000,000,000 |
| `r4bm03_mfcc_vjp_sign` | mutant, caught by G1 (MFCC) | wave-capture, ref-at-wave | 120 | 240 / 540 | 6,500,000,000 |
| `r4bm04_conv_post_bias_grad_dropped` | mutant, caught by G1 (linear) | - | 100 | 200 / 540 | 5,000,000,000 |

ref-at-wave (lead CPU, A1-1 step 2): `scripts/bwd_r4b_refwave.py reference | fd --part 0/1 --parts 2 | merge
--root artifacts/bwd_r4b/ref-at-wave`, expected peak 6,000,000,000 host bytes each; stand-in timings
reference 17 s, fd parts about 6.3 min each in parallel, merge 10 s.

## 7. Mutants (each caught by a check that runs its edit, L41)
| Mutant | Edit | Caught by | Line |
|---|---|---|---|
| r4bm01_public_dw_transposed_kernel | `dw.flip([2])` in the public path only (mutant package; marker R4BM01_EDIT) | G-API: a test_gpu failure with the marker, after a clean g-api pass | `r4bm01_public_dw_transposed_kernel` |
| r4bm02_wave_upstream_dropped | upstream at the wave replaced by zeros (counter) | G1 (linear) fails after a clean pass | `r4bm02_wave_upstream_dropped` |
| r4bm03_mfcc_vjp_sign | dL/dwave negated (counter) | G1 (MFCC, A1-1 reference) fails after a clean pass | `r4bm03_mfcc_vjp_sign` |
| r4bm04_conv_post_bias_grad_dropped | Voice's tail_vjp wrapped to return db None (counter) | G1 (linear) fails on conv_post.bias after a clean pass | `r4bm04_conv_post_bias_grad_dropped` |
CPU half (self-test, 18/18 checks): each edit applied to the fp64 chain fails the same comparison code.

## 8. Recorded before sealing
- G6 (`G6_PER_ID_R4B.json`): verdict {'prod': True, 'r4a': True}; unexpected {'r4a': 0, 'prod': 0}.
- Ranking-eligible: 274 of 315 (`RANKING_ELIGIBLE_R4B.json`).
- Packages: `MODULES_R4B.json` sha256 8a41dd1944511ed3a136bb6b50fdb424dc97205c676e6d082a5ade8af8e5db9f.
