# Project-Tensor

**Project-Tensor is a from-scratch CUDA tensor and inference engine for local
AI workloads.** Its current native runtime is a C++/CUDA extension with a
Python tensor/autograd layer and no PyTorch or TensorFlow dependency in the
inference path. It runs quantized LLMs and the complete Hunyuan3D image-to-3D
pipeline (its transformer replaced by TensorCUDA) on consumer NVIDIA GPUs.

**The named models below are validation points, not a compatibility
whitelist.** Support status is reported separately for TensorCUDA execution,
APA evaluation, and GRM end-to-end certification.

## Build and import the native engine

Requirements: Linux, Python 3.9+, NumPy, CMake 3.18+, a C++17 compiler, and an
NVIDIA CUDA toolkit. The first CMake configure may fetch pybind11 if it is not
already cached.

```bash
git clone <repo-url>
cd Project-Tensor

# Pass the CUDA SM for your card: 86 for RTX 3070, 89 for RTX 4070 SUPER.
./tensor_cuda/build.sh 89

# Extension-load smoke test; this does not allocate a GPU tensor.
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tensor_cuda python3 -c \
  "import tensor_cuda as tc; print('tensor_cuda OK:', tc.Tensor.__name__)"
```

The build is Release by default and writes `_tensor_cuda.*.so` into
`tensor_cuda/tensor_cuda/`. The repository-root `pyproject.toml` packages the
historical CuPy implementation; it is not the native-engine install path.

## What runs today

These are native-engine validation points, not an exhaustive model list. The
[evidence matrix](docs/SUPPORT_MATRIX.md) distinguishes local receipts from
historical reports and `unconfirmed` cells.

| Native validation point | Current public status | Evidence |
|---|---|---|
| MiniCPM3-4B | **Engine port**; **APA positive**; **single-pass positive** (SP3, 2026-09: −0.12 ppl vs two-pass at equal 10 % refine; exact to 0.0005) | [SP3 results](artifacts/apa_sp3/RESULTS.md) |
| Qwen3-4B | **Planned / unconfirmed** in local public receipts | [Matrix](docs/SUPPORT_MATRIX.md) |
| Qwen3.5-9B | **Engine port**; **APA positive** in the historical paper | [Matrix](docs/SUPPORT_MATRIX.md) |
| [Gemma-4 12B / MQA](docs/GEMMA4_MQA_ADJUDICATION.md) | **Engine port**, **Parity-gated**; **APA neutral** (SP4G, 2026-09: single pass exact per call at 1e-6; every tail inside the model's own ±2.56 ppl noise floor; no VRAM lever, 64K prefill on 12 GB for every arm) | [SP4G results](artifacts/apa_sp4g/RESULTS.md) |
| GPT-OSS-20B | **Engine port** workload; **APA positive**; **single-pass positive** (SP5, 2026-09: −29 ppl vs two-pass at 19× the floor over 16 windows; exact through attention sinks; standard OOMs at 2K where APA runs) | [SP5 results](artifacts/apa_sp5/RESULTS.md) |
| Trinity Nano | **Engine port**, **Parity-gated** | [Port ledger](docs/TRINITY_NANO_PORT_LEDGER.md) |
| Hunyuan3D-2.0 shape DiT | Complete **Engine port**; **APA positive** E2E visual receipt | [Visual receipt summary](docs/APA_VISUAL_TOKENS_PRIMER.md) |
| Hunyuan3D-2.1 visual DiT + MoE | **Planned / unconfirmed** in local public receipts | [Matrix](docs/SUPPORT_MATRIX.md) |

## Research systems built on it

| System | Relationship to Project-Tensor | Status source |
|---|---|---|
| Adaptive Precision Attention (APA) | Allocates score precision at the attention score/softmax boundary; it is neither sparse attention nor KV compression | [APA overview](docs/APA.md) |
| Graft Runtime Memory (GRM) | Captures, stores, routes, and mounts model-native attention state through model-specific dialects | [Matrix](docs/SUPPORT_MATRIX.md) |
| ColdCast (personal project; not publicly available) | Tests APA on Project-Tensor outside text LLMs by replacing Hunyuan3D's shape transformer with a torch-free TensorCUDA implementation; supplies the Hunyuan3D-2.0 non-causal visual APA E2E gate | [Visual receipt summary](docs/APA_VISUAL_TOKENS_PRIMER.md) |

## Single-pass selective APA (SP1–SP5, 2026-09)

APA's original selective kernel walks the keys twice: a bulk pass on
quantized keys to build a z-score threshold, then a refine pass. The
single-pass kernels (`apa_selective_sp_kernel`, split-K decode variant;
`TC_APA_SP` default **OFF**) walk once under the rule
`refine_j iff bulk_j ≥ max(bulk[0..j]) − δ`, which is monotone and therefore
exact in one pass; the z-score rule is provably not single-pass under
irrevocable streaming. Record, all lead-run on an RTX 4070 SUPER 12 GB:

| order | evidence class | result |
|---|---|---|
| [SP1 / SP1.1](docs/APA_SP1_LEDGER.md) | kernel sweep | single pass 1.1–1.9× faster on prefill, 2–6× on split-K decode, closer to dense at matched fraction; every pre-existing kernel body hash-pinned |
| [SP2](docs/APA_SP2_DELTA_DERIVATION.md) | kernel sweep | provable δ = ln(1/ε) + 2·e_q refines ~100 % at 2/4-bit keys: the guarantee is vacuous there; refine-all single pass still 1.4× faster than two-pass |
| [SPD1](artifacts/apa_spd1/SPEED_CHAIN.md) | kernel sweep | same-shape chain: single pass 2–4× behind engine dense on prefill, 20–50× behind torch flash (per-key scalar walk, no tensor cores); parity with dense at decode 32K; 4–16 MiB transient vs 2–4 GiB dense |
| [SP3 MiniCPM3-4B](artifacts/apa_sp3/RESULTS.md) | model perplexity | matched single pass 8.661 vs two-pass 8.779 vs standard 8.656 at 4-bit; the z-score tail leaves 92 % of softmax mass unrefined, the running-max tail 13 %; refine-all = standard to 0.0005 |
| [SP4G Gemma-4 12B](artifacts/apa_sp4g/RESULTS.md) | model perplexity | clean qk-normed MQA keys: every tail inside the model's bf16/fp32 noise floor (±2.56 ppl); provable δ usable (69 % refined); exactness per call only — the model amplifies rounding ~30× |
| [SP5 GPT-OSS-20B](artifacts/apa_sp5/RESULTS.md) | model perplexity | 16-window pooled: single pass 244.2 vs two-pass 273.5 vs standard 232.3 (floor 1.5); two-pass leaves 49 % of mass unrefined vs 11 %; exact through learned sinks; resident wall 12,288 tokens (fragmentation), streamed 16K at 1.03 GB |

Principles earned, each with three model points: the tail-choice penalty
scales with the 4-bit bulk error; the provable δ is model-dependent
(usable only where measured e_q is small); exactness is a per-call
property and model-level perplexity needs the model's own bf16/fp32
floor first; APA's memory lever exists only where the dense path
materializes S×S; a window is a cell, so N is sized from the floor's
spread. Production routing to the single pass is a separate, still-open
decision.

## Research and evaluation targets

This table names evaluation coverage, not engine or GRM certification. APA has
been studied across the attention families MHA,
[MQA](docs/GEMMA4_MQA_ADJUDICATION.md), GQA, and MLA, plus MoE,
NoPE/hybrid text, and bidirectional visual attention.

| Evidence class | Targets | Operative reading |
|---|---|---|
| **APA positive** | MiniCPM3-4B, GRAPA-232M, Qwen3.5-9B, GPT-OSS-20B, Hunyuan3D-2.0 | Per-target gates and provenance remain controlling |
| **APA boundary** | TinyLlama-1.1B, OLMoE-1B-7B, Trinity Nano | Quality, cost, or geometry limits are part of the result |
| **APA neutral** | [Gemma-4 12B / MQA](docs/GEMMA4_MQA_ADJUDICATION.md) (SP4G 2026-09 supersedes the earlier negative) | Exact per call; every tail inside the model's noise floor; no memory lever |
| **APA negative** | GPT-2 PoC | Abandoned or failed registered gates |
| **External receipt** | 26B GQA model with two KV heads | Collaborator-reported; not locally reproduced |
| **Planned / unconfirmed** | Mistral-7B; Hunyuan3D-2.1 APA | No closed in-repo gate |

## Representative performance receipts

- Single-pass APA (2026-09): MiniCPM3-4B −0.12 ppl and GPT-OSS-20B −29 ppl
  versus two-pass at equal refine fraction, Gemma-4 tie; see the
  [SP table above](#single-pass-selective-apa-sp1sp5-2026-09).
- MiniCPM3-4B's historical MLA report records 675 to 21.6 ms/token and a
  3K-to-32K context extension; see the [paper draft's evidence table](docs/APA_PAPER_DRAFT.md#41-the-agnosticism-table).
- The closed kernel program reports GPT-OSS-20B attention decode up 25–35%,
  expert GEMVs up 61%, and long-context prefill up 3.2×; see the
  [kernel synthesis](docs/KERNEL_OPT_SYNTHESIS.md).
- Trinity's matched-fp32 parity gate reproduced the reference top five for
  8/8 steps with overall max absolute error 6.5804e-05; see the
  [Trinity ledger](docs/TRINITY_NANO_PORT_LEDGER.md#2026-07-0809-t4-fp32-ab-disposition--semantic-parity-proven).
- Hunyuan3D-2.0's APA E2E gate reported 199.9 s versus 242.8 s for the standard
  path, with 14/14 meshes watertight; see the
  [visual receipt summary](docs/APA_VISUAL_TOKENS_PRIMER.md#3-the-modality-jump-hunyuan3d-transformer-replacement-2026-07-1314).

## Honest limits

- TensorCUDA is source-built for NVIDIA CUDA; this repository does not publish
  a portable CPU backend or a prebuilt wheel in the documented path.
- An extension import proves that the native module loads. GPU allocation,
  numerical parity, speed, context length, and memory ceilings require their
  own hardware-bound gates.
- Model loaders are explicit ports, not a generic Hugging Face bridge. A named
  engine port does not imply APA benefit or GRM certification.
- APA has an architecture-neutral insertion point, but useful operating points
  are model- and geometry-dependent. The three-model record (SP3/SP4G/SP5)
  says the quality lever is the tail choice and it scales with 4-bit bulk
  error, and the memory lever exists only where the dense path materializes
  S×S; [Gemma-4 12B / MQA](docs/GEMMA4_MQA_ADJUDICATION.md) has neither and is
  the adjudicated neutral.
- Model-level exactness gates are meaningless on numerically chaotic models
  (Gemma-4 amplifies rounding-level differences ~30× through its depth);
  exactness is gated per call on identical inputs, and any sub-percent
  perplexity claim needs the model's own bf16-vs-fp32 floor measured first.
- The single-pass kernels stay behind `TC_APA_SP` (default OFF); their
  prefill is 20–50× behind flash-class kernels because they walk keys
  per warp without tensor cores — a tiled single pass is the open kernel
  order.
- Reported performance is hardware- and protocol-specific. The principal local
  receipts use RTX 3070 8GB and RTX 4070 SUPER 12GB-class consumer cards.
- The Hunyuan3D-2.0 APA E2E receipt does not transfer to Hunyuan3D-2.1; the
  latter remains unconfirmed for APA until separately gated.

## Documentation

- [Master support and evidence matrix](docs/SUPPORT_MATRIX.md)
- [APA overview](docs/APA.md)
- [Gemma-4 12B / MQA adjudication](docs/GEMMA4_MQA_ADJUDICATION.md)
- [Kernel benchmark synthesis](docs/KERNEL_OPT_SYNTHESIS.md)
- [Quantization benchmark methodology and limits](docs/QUANT_SWEEP_SYNTHESIS.md)
- [Historical CuPy / `tensor_gpu_v2` API](docs/LEGACY_TENSOR_GPU_V2.md)
- [Full APA paper draft](docs/APA_PAPER_DRAFT.md) — historical research report;
  the Gemma verdict is superseded by the adjudication above
- [Failure and implementation ledgers](docs/KERNEL_OPT_IMPLEMENTATION_LEDGER.md)

## License

Copyright (C) 2026 David Perry.

This repository is licensed under the GNU Affero General Public License v3.0 —
see [LICENSE](LICENSE). Software derived from this code, including software
served over a network, must be released under the same terms. Commercial
licensing outside the AGPL terms is available from `dave@ai-storyforge.com`.

Associated research papers are licensed CC BY 4.0 via their Zenodo records.
