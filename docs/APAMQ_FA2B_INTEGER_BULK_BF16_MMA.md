# APAMQ-FA2B — Integer Bulk + BF16-MMA Refine

Status: **IMPLEMENTED; CPU/nvcc checked; GPU gates registered and pending the
lead's GPU run. No performance or runtime-correctness result is claimed from
this GPU-less sandbox.**

## Entry points and semantic boundary

- `apa_selective_attention_int4` is unchanged. It remains the F-A1
  symmetric-7 INT4-K, fp32-dequant bulk implementation.
- `apa_selective_attention_int4_bf16_mma` is lane 1. It accepts BF16 and
  `D % 16 == 0`. K packing, fp32-dequant bulk scores, fp32 population
  statistics, threshold, and selection predicate use the F-A1 helpers.
  Selected exact QK dots use BF16 WMMA with fp32 accumulation. Thus BF16
  products are exact and only accumulation association changes. The key-to-
  warp ordering is retained, so selection is identical by construction.
- `apa_selective_attention_int8q_int4` is lane 2 and a new operating point.
  Each query row gets `qscale=max(abs(q))/127` (zero-safe to 1) and signed
  codes `clamp(round_away(q/qscale), -127, 127)`. K retains EXP-APA-2's
  per-key `kscale=max(abs(k))/7`, signed `[-7,7]` convention. The bulk score
  is `int32_dot(qcode,kcode) * (qscale*kscale) * attention_scale`.

Both new entries keep threshold statistics in fp32. Non-selected keys retain
the integer-derived fp32 bulk score in lane 2. No BF16 dequantized K or global
`L*S` score/mask tensor is materialized.

## Implemented kernel shape

The integer bulk uses `__dp4a`, not IMMA. Four warps own interleaved 16-key
batches. Lanes 0–15 independently calculate one exact integer dot, each using
`D/4` dp4a instructions. At D=512 this is 128 dp4a instructions per live lane
to produce 512 signed products and additions. K stays packed until two bytes
are expanded into one four-lane signed register operand.

For comparison, the F-A1 packed loop consumes 256 K bytes per D=512 dot but
executes 256 iterations with two nibble decodes, two int-to-float conversions,
two `code*kscale` multiplies, and two fp32 Q multiply-adds: roughly eight
scalar arithmetic/conversion operations per packed byte. The dp4a loop issues
128 integer-dot instructions for the same 256 K bytes, or 0.5 dp4a instruction
per packed byte, plus one fp32 scale multiply after the exact sum. Q codes are
staged once per CTA and reused across its K walk.

Ada IMMA `s8` would require expanding the INT4 K tile to an INT8 matrix before
MMA, doubling K operand bytes from 256 to 512 per key and adding shared-memory
tile traffic. It also needs a sufficiently full M tile; decode `L=1` and
bottom-right causal query tails do not supply one. dp4a preserves the packed-K
traffic advantage, maps naturally to one-query and rect-causal rows, and gives
one implementation for prefill and decode. That is why dp4a is the implemented
bulk primitive. The sweep, not this static argument, decides performance.

Selected refinement batches up to 16 keys per warp. A BF16
`m16n16k16` WMMA sequence accumulates the query row against those 16 original
BF16 K rows in fp32; unused output rows are zero. Exact scores remain fused in
shared/register state and immediately enter the online-softmax pass.
Decode-shaped `L=1` calls reuse F-A1's automatic/forced split-K policy: one
global integer/F-A1 statistics pass, partitioned fused blend/softmax CTAs, and
the established fp32 online-softmax merge. This supplies long-S key-range
parallelism without changing either lane's bulk or selection contract.

## Registered correctness gates

1. Integer diagnostic: Q bytes, packed K bytes, and scales match the NumPy
   quantizers; integer sums match NumPy int32 with exact equality; unscaled
   fp32 `isum*qscale*kscale` uses `rtol=1e-6, atol=1e-6`.
2. Lane 2 end-to-end: NumPy mirrors BF16 input rounding, INT8-Q/INT4-K
   quantizers, exact integer bulk, fp32 mean plus `z*population_std`, original
   BF16 exact K refinement, bottom-right causal mask, and all-key softmax.
   Registered `rtol=2e-2, atol=2e-2` for BF16 output and reduction association.
3. Lane 1 versus unchanged F-A1 BF16: `rtol=8e-3, atol=8e-3` for fp32
   accumulation reassociation. Bulk/statistics helpers and predicate are
   shared, so selection is identical by construction.
4. MQA `KVH=1`, D=512, rectangular bottom-right causal `S>L` tests require
   both selected and non-selected keys at every tested row.
5. The existing `test_apa_selective_int4.py` suite remains the F-A1 regression
   gate (11 parametrized cases in the current collection).
6. Forced split-K versus monolithic decode is registered at `rtol=5e-3,
   atol=5e-3` for both new lanes; the only difference is partition merge
   association.

GPU command:

```bash
flock -w 7200 /tmp/forge-gpu.lock env PYTHONPATH=tensor_cuda \
  python3 -m pytest -q tensor_cuda/tests/test_apa_selective_int4.py \
  tensor_cuda/tests/test_apa_int8q_int4.py
```

## Performance legs

`scripts/apamq_e1_sweep.py` now records `int4_bf16_mma_apa` and
`int8q_int4_apa` beside `standard`, `fused_apa`, and `int4_apa`. It includes
D=128 and D=512, L=512 prefill and L=1 decode, and S=8K/16K/32K/64K (plus the
existing 4K breadth cell). The registered decision cell remains MQA KV=1,
D=512, L=512, S=16K: target at most 3x standard, stretch parity. Decode long-S
is measured without a predicted win.

GPU sweep command:

```bash
flock -w 7200 /tmp/forge-gpu.lock python3 scripts/apamq_e1_sweep.py
```
