# PT-DET-1 training-path nondeterminism audit

**Embedding is the only gradient implementation changed.** This is a static
source audit, supported by the lead's CC46-B/C localization; it is not a new
GPU trace, sanitizer run, or proof that the whole engine is deterministic.

Scope: the registered v3 argv uses `g1_tf32/h_tf32` in blocks 0–10 and
`g1/h` elsewhere. The v2 BF16 route uses `g1/h` throughout. The duty wrapper
selects `g1/h` at
`/mnt/ForgeRealm/wt/grapa-cc46/scripts/run7/grapa_run7_duty_cycle.sh:211`;
`grapa/fwd_precision.py:384` applies/restores the block-specific variants.
The requested BF16 replay arm uses the **v3 corpus and checkpoint**, with
the precision-policy flags removed; it does not switch to the v2 corpus.

| Site (paths relative to this fork unless otherwise stated) | Does unordered arithmetic reach a gradient? | Registered v3 / v2 path |
|---|---|---|
| `tensor_cuda/src/kernels.cu:863`, `embed_bwd_kernel` | Yes. Repeated IDs contend on FP32 weight-gradient elements. | Both with the new switch OFF. ON routes only this VJP to the deterministic implementation. |
| `tensor_cuda/src/kernels.cu:2602`, `:2609`, backward variant `a` | Yes. Query blocks atomically accumulate dV and dK. | Neither registered route selects backward `a`. A caller reverting the backward selector to `a` remains exposed. |
| `tensor_cuda/src/kernels.cu:2811`, `:2819`, backward variants `b`/`d` | Yes when `WRITE_KV=true`; deletion ablation `c` removes these contributions and is not a valid full backward. | Neither registered route selects these variants. |
| `tensor_cuda/src/kernels.cu:2895`, `:2981`, backward `f`/`f_pass_a` | No float atomics in the selected instantiation. Query pass sets `WRITE_KV=false`; one key owner reduces dK/dV in fixed loops and warp/shared trees. | Not the live TF32 or BF16 route. Used by the separate `--fwd-precision-kernels fp32` fallback. Forward `a` in the phrase “a/f” must not be confused with backward `a`. Dispatch: `tensor_cuda/src/ops.cpp:1168`. |
| `tensor_cuda/src/kernels.cu:3015`, `:3179`, BF16 `g1/g2` | No float atomic accumulation found in these owner-tile backward implementations. | v3 blocks 11–23; v2 all blocks (`g1`). |
| `tensor_cuda/src/attention_tf32.cu:86`, `:203`, `:217` | Fixed query/key tile ownership; each dQ/dK/dV output is written once. No atomic in this file. | v3 blocks 0–10; no v2 hit. |
| `tensor_cuda/src/kernels.cu:4813`, gather/top-k backward via `tensor_cuda/src/ops.cpp:960`, `:977` | General gather with duplicate destinations is nondeterministic. | Both call it for loss gather, but `grapa/loss.py:22` selects one class per distinct `(batch, position)` row: destinations `(b*L+t)*V+target[b,t]` are injective, even for repeated class IDs. Therefore no contended reduction in this current use. No model top-k call found. |
| `tensor_cuda/src/conv.cu:56`, `:88`, `:118` | Yes: col2im, average-pool backward, and max-pool backward can have multiple writers. | Neither model path uses convolution/pooling backward. Left unchanged. |
| `tensor_cuda/tensor_cuda/nn.py:280` RMSNorm; `:124` LayerNorm; `tensor_cuda/src/kernels.cu:554`, `:590` reductions | Composed elementwise operations and fixed-tree or serial reductions; no float atomic found. | RMSNorm in both; LayerNorm unused in this MLA model. Fused RMSNorm is inference-only (`tensor_cuda/src/ops.cpp:460`) and rejects backward. |
| `/mnt/ForgeRealm/wt/grapa-cc46/grapa/grad_clip.py:27`; `tensor_cuda/src/kernels.cu:554` | Fixed per-tensor reduction, then Python accumulation in parameter order. Scaling is per-element. | Both. CC46-C's conclusion that the clip was not the first-step source is inherited evidence, not newly measured here. |
| `tensor_cuda/src/kernels.cu:877`; `tensor_cuda/tensor_cuda/optim.py:54` Adam(W) | One thread per element; fixed host parameter iteration; no atomic accumulation. | Both. |
| `tensor_cuda/src/autograd.cpp:15`, `:54` | Gradient merges use elementwise addition. DFS traverses the parents vector; the unordered set is only a membership test, not traversal order. | Both, including the embedding/tied-head gradient merge. |
| `tensor_cuda/src/matmul.cu:83`; `tensor_cuda/src/tf32_gemm.cu:281` | Vendor GEMM kernels cannot be cleared by a source-only atomic search. The ragged WMMA implementation has fixed output ownership. | BF16 cuBLAS in both; cuBLASLt TF32 in v3 promoted blocks. End-to-end replay remains the required empirical check. |
| `tensor_cuda/src/gemm_apa.cu:362`–`:366`; `tensor_cuda/src/kernels.cu:6695` | Integer pair queues/counters, not float gradient scatter. Queue order can matter to an independent future path; integer atomics alone are not a proof of full-path reproducibility. | Separate inference/instrumentation paths; not the selected training backward. |

`grapa/codebooks.py:80` uses `apa_quantize_gather` for quantizer lookup. Its
binding returns a non-differentiable tensor; the quantizer detaches both its
key input and returned reconstruction. It is not the general gather scatter-add
backward listed above. Seeded host RNG/checkpoint restoration and the loader
are also outside the atomic audit; the replay gate checks batch cursors and
the checkpoint schema, while CC46 performs the registered controller replay.

Successors, **not claimed fixed**: deterministic general scatter-add/top-k
backward with collisions, attention backward `a/b/d`, convolution/pooling
backward, and any vendor-kernel or environment issue that survives the four
replays. No speculative fix to these sites was included.

## Prior art

NVIDIA CUB stable radix sort and segmented reduction (Merrill/NVIDIA;
installed CUDA 12.6, 2024) supplies the grouping primitive and standard
scatter-to-segment transformation. The installed header
`/usr/local/cuda-12.6/include/cub/device/device_radix_sort.cuh:109` explicitly
documents stability. Its copyright identifies Duane Merrill (2011) and
NVIDIA (2011–2023); 2024 here identifies the toolkit used, not invention.

[PyTorch 1.9 (2021)](https://pytorch.org/blog/pytorch-1-9-released/) supplies
the precedent for opt-in deterministic indexing. [Demmel and Nguyen,
*Fast Reproducible Floating-Point Summation*, ARITH 2013](https://www.acsel-lab.com/arithmetic/arith21/papers/p54.pdf)
supplies the reproducible-summation context. This implementation does **not**
use their order-independent accumulator: it fixes position order and uses
ordinary FP64 accumulation followed by FP32 rounding. Cross-device,
cross-toolchain, arbitrary-input-order reproducibility is not claimed.
