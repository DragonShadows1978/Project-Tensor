# Operand census and scope

Immutable real exports: artifacts/apa_ste1/INPUTS.json and inputs/*.npz.
Actual NPZ census: small1 10,924 bytes, small2 26,412 bytes, GRAPA
134,218,924 bytes; five FP32 fixture arrays q/k/kq/v/do, converted identically
to BF16 RN-even in every process. Inputs must match full BF16 words.

| shape | B | H | KVH | L | S | D | VD | causal | visible pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| small1 | 1 | 2 | 2 | 7 | 11 | 32 | 16 | no | 154 |
| small2 | 1 | 4 | 2 | 17 | 19 | 32 | 16 | yes | 748 |
| grapa | 1 | 16 | 16 | 4096 | 4096 | 128 | 64 | yes | 134250496 |

Every process explicitly sets and asserts backward g1 and forward h before
attention. Calls are the explicit h forward/g1 backward kernels reached by
GRAPA's training dispatch; source ops.cpp:789-795, 804-813, 1184-1188.
Saved-output rowdot and BF16 coefficient rounding in g1 remain unchanged;
float64 reference instead evaluates ideal forward and full VJP on BF16 inputs.
This is a finite numerical signal check, not an exact all-input g1 proof.
Z from refine .15 is float32(1.0364333894937898) = 1.0364333391189575;
all controls use -1e6. TC_DETERMINISTIC=1 and TC_DET_EMBED_BWD=1.
Base is 69b154cd; arm is unchanged af9ffce4. Three fresh processes/config/suite:
36 processes, 54 NPZ files; six tensors per file (plus five input operands).

| comparison | all operands per shape | fixed rule |
|---|---|---|
| G-G1 | 3 base/base pairs x six tensors | identical full bytes; global premise before all arm gates |
| G-G2 | 9 off/base + 9 unset/base pairs x six | identical full bytes |
| G-G3 | 9 on-all/off-all pairs x six | identical full bytes; actual threshold checks all visible selected |
| G-G4 | 9 on/off pairs x six | out/lse/thr/dq/dv identical bytes, dk unequal |
| G-G5 | all 9 on/off pairs, full dk elements | cosine >= .995; zero-intercept least-squares slope [.98,1.02]; on rel-L2 <= off rel-L2 + .01, each against its own rule |
| G-G6 | 3 on/on pairs x six | identical full bytes |
| G-G7 | all 9 on/off pairs | a whole key no visible query can select: on nonzero, off exactly zero; conservative dot envelope against identical saved FP32 threshold |
| disclosure | every on/off run dk | per-element BF16-ULP percentiles 0/50/90/95/99/99.9/100; no percentile gate |
| timing | off and on, two samples after warmup | CUDA event h-forward/g1-backward pair means and difference; no step projection |

252 byte rows/shape, 756 total; 27 signal rows and 27 witnesses. All dk elements
are included in G-G5, including near-threshold selection ties. Delta reference
is accumulated directly over ideal unselected visible pairs, not inferred from
captured gradients. No saved lse/threshold/gradient defines G-G5 reference.
Every float64 reference tensor must be finite, nonzero and nonconstant; extra
STE contribution is nonzero for selective shapes, exactly zero for refine-all.
Captured BF16 tensors use uint16, FP32 lse/thr use uint32: equal bytes at full
width including signed zeros. No mask, truncation, tolerance or self-comparison
for byte gates. Capture metadata, file hashes, fixture words, unique PIDs,
host/GPU identity and governor rows (one per label, ran, rc 0) are checked.

F4 census: real extension initializer rejects 2, empty, whitespace 1 and true;
CPU subprocesses exercise source and engine packages plus stray fallback module.
Real old small h_g1 exports and independent REV1 reference validate the CPU
oracle; they are not fresh STE5 GPU evidence.

## Prior art
Bengio, Leonard and Courville (2013), STE; Dao et al. (2022), attention VJP;
Higham (2002), dot rounding envelope; IEEE 754-2019, full widths/ULP spacing;
NIST FIPS 180-4 (2015), SHA-256; existing STE1/STE4/REV1 harness. Unverified -
lead to check these named works. New contribution is g1/h harness composition
and evidence accounting; no prior art known to me for the lead's exact gates.
