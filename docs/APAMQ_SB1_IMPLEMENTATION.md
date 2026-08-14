# APAMQ-SB1 Implementation Receipt

> Historical SB1 receipt. The compaction, synchronization, and transient
> policy below was superseded by `APAMQ_SB2_IMPLEMENTATION.md`.

Evidence state: **IMPLEMENTED / CPU-CHECKED / GPU GATES PENDING LEAD**.
No GPU timing or parity result is claimed from the sandbox.

## Engine contract

`tensor_cuda.apa_gemm_selective_attention(q, k, v, scale, zthr,
is_causal=False, *, Lq=0, row0=0, window=0, k_codes=None,
k_scales=None)` accepts BF16 `q=[B,H,L,D]`, `k=[B,KVH,S,D]`, and
`v=[B,KVH,S,VD]`, with `H % KVH == 0` and `D % 4 == 0`.

- Q and K are independently quantized per final-axis row. The scale is
  `amax/127` (or `1` for an all-zero row). Codes are
  `clamp(roundf(x/scale), -127, 127)`. CUDA `roundf` is round-to-nearest with
  halfway cases away from zero. This is mirrored in NumPy with
  `copysign(floor(abs(x)+0.5), x)`, not `np.rint` (ties-to-even).
- GQA is flattened without KV expansion: each `(B,KVH)` cuBLASLt batch uses
  `A=[group*L,D]`, `B=[S,D]`, `C=[group*L,S]`.
- Bulk uses cuBLASLt row-major signed INT8 A/B, `TRANSB=T`,
  `CUBLAS_COMPUTE_32I`, and INT32 C/D. A/B leading dimensions are `D`; C is
  padded to an `S`-stride divisible by four. `D % 4 == 0` is enforced. The
  first successful heuristic under a 32 MiB workspace preference is used.
  Registered `D=128/512` and aligned prefill sizes satisfy tensor-core
  reduction/layout requirements; irregular gate sizes may use a compatible
  library fallback algorithm.
- INT32 sums are multiplied by the FP32 Q-scale, K-scale, and attention scale
  into an FP32 score matrix. Per-row selection statistics use valid keys only:
  `mean(abs(bulk)) + zthr*std(abs(bulk))`, population variance.
- Selected `(row,key)` pairs are prefix-compacted. Bounded chunks physically
  gather only selected BF16 Q/K rows, issue one strided-batched skinny BF16
  GEMM per chunk with FP32 accumulation/output, and scatter exact scores into
  the FP32 bulk matrix. No full exact rank matrix is formed.
- Softmax reads FP32 scores and writes BF16 probabilities only after
  normalization. Existing cuBLAS `matmul` performs `P@V`.
- `apa_gemm_selective_quantize_k(k)` returns cacheable signed-code bit patterns
  in a public uint8 tensor plus FP32 per-key scales. Supplying both through the
  optional entry arguments skips call-local K quantization.

Bottom-right causal bounds match the bounded blend convention. For a chunk,
`Lq` is the full query length, `row0` its absolute start, and `window=0` means
full causal history. `Lq=0` defaults to the passed chunk length.

## Registered gates

CPU and skip-safe native gates:

```bash
PYTHONPATH=tensor_cuda python3 -m pytest \
  tensor_cuda/tests/test_apamq_sb1.py -q -rs
```

Lead GPU gate (one command):

```bash
flock -w 7200 /tmp/forge-gpu.lock \
  bash tensor_cuda/run_apamq_sb1_gates.sh
```

The gate file registers:

- G-SB1-a: exact INT8 codes and integer sums, `rtol=atol=1e-6` scale/bulk,
  and `rtol=atol=2e-2` composed output;
- G-SB1-b: `S=121`, cached query `Lq=17`, chunk `L=11,row0=6`, MQA
  `KVH=1,D=512`, and GQA `KVH=4/8,D=128`;
- G-SB1-c: SB1 versus existing `apa_selective_attention` selected fractions
  and Jaccard overlap at matched `zthr`, printed as data with no tolerance;
- cached versus call-local K-code exact output parity.

Targeted G-SB1-d measurements (same generator/config, each one command):

```bash
flock -w 7200 /tmp/forge-gpu.lock \
  env PYTHONPATH=tensor_cuda python3 scripts/apamq_e1_sweep.py \
  --cell standard 1 512 prefill 16384

flock -w 7200 /tmp/forge-gpu.lock \
  env PYTHONPATH=tensor_cuda python3 scripts/apamq_e1_sweep.py \
  --cell gemm_apa 1 512 prefill 16384
```

The complete APAMQ-E1 matrix, now including `gemm_apa`, remains:

```bash
flock -w 7200 /tmp/forge-gpu.lock \
  env PYTHONPATH=tensor_cuda python3 scripts/apamq_e1_sweep.py
```

The port-level FC perplexity arm is pre-registered but not implemented or run
in SB1; its later bar remains `gemm_apa <= apa_blend +0.25% relative`.

## Historical SB1 transient and deviation receipt

- The speed mode intentionally materializes INT32 bulk and FP32 score matrices.
- Prefix compaction performs one device-to-host selected-count readback, which
  synchronizes before exact-size index allocation. It is included in wall time.
- To prevent selected Q/K duplication from becoming unbounded, refinement is
  chunked at 262,144 selected pairs. Therefore a call above that count issues
  multiple skinny BF16 GEMM batches, one per compact chunk, rather than one
  physically unbounded GEMM for the entire call. No unselected K row is
  gathered and no full exact rank matrix is computed.
- cuBLASLt descriptors and heuristic lookup are call-local in SB1. Persistent
  algorithm/workspace caching is left for a later optimization after the lead
  establishes parity and the RED/GREEN wall receipt.
