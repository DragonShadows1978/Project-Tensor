# APA-SPD1 implementation handoff

**G1 PASS; G2 BLOCKED by the dispatched no-GPU environment.** The benchmark,
leased one-cell runner and receipt-to-table writer are implemented. This is an
author-run CPU baseline, not blind red-team verification. Model id:
`gpt-6-astra`; reasoning effort: **high**.

Registration SHA-256:
`178a45cb99cb091b8514f05a1270baba53ae12786bba66054bd73c1c8e8b95d0`.
`registration.json` and `registration.sha256` are immutable. `ORDER.md` preserves
the dispatched order; interpretations and follow-up findings live in the ledger,
not edits to the order or registration.

## Contender registry and availability

| Contender | Entry point | Dtypes | CPU evidence / GPU status |
|---|---|---|---|
| Engine dense | `tensor_cuda.matmul(q_grouped,k,trans_b=True)` → `causal_softmax` or `Tensor.softmax(-1)` → grouped `matmul(weights,v)` | FP32 and BF16 | E1 standard composition; imported pinned runtime. GPU blocked. FP32 is the full-output accuracy reference. |
| Torch math | `F.scaled_dot_product_attention`, `sdpa_kernel([SDPBackend.MATH])` | FP32 and BF16 | Torch 2.11.0+cu130; CPU math verified on adversarial tiny cells; GPU blocked. |
| Torch efficient | Same API, forced `EFFICIENT_ATTENTION` | BF16 | Enum present; sm_89 support unknown until forced cell call. GQA K/V expansion included in measurement. |
| Torch flash | Same API, forced `FLASH_ATTENTION` | BF16 | Enum present; sm_89 support unknown until forced cell call; no backend fallback. |
| Optional flash_attn | `flash_attn.flash_attn_func` | BF16 | Not importable here; no installed version to report. If available on lead run, version is recorded; requires ≥2.1 for lower-right semantics. |
| APA two-pass | `tensor_cuda.apa_selective_attention` / existing `apa_selective_kernel` family | FP32 and BF16 | Entry present; registered r=.15 on SP1 grid, r=.10 on E1 extras. GPU blocked. |
| APA-SP | `tensor_cuda._C.apa_selective_attention_sp`, `TC_APA_SP=1` | FP32 and BF16 | Entry present. Existing prefill SP kernel for L>1; SP1.1 split-K dispatch for all L=1 cells. GPU blocked. |
| Optional SP2 | Actual SP2 launcher through pinned adapter, ε=1e-3 | FP32 | No frozen table found; row skipped. Runtime discovery and fail-closed adapter contract provided. |

Machine-readable registry and exact runtime hash: `CPU_INVENTORY.json`.
The runtime `.so` is reused from `artifacts/apa_sp1/build`, with its source and
binary hashes validated. Production files were not rebuilt or edited.

## Registration and predictions

Fifty cells: all 48 SP1 shapes plus causal E1 L=512, S=8192/32768, D=128,
H=16/KV=4 cells. Every shape has 12 registry slots; unavailable rows remain visible.
Nine CUDA-event samples after three warmups, rotating contender order, one process
and legacy stream 0. Separate three-call allocator peak passes; full output
relative Frobenius and max-absolute errors against engine FP32. Shared seeded
FP32 q/k/v are generated once; every element of the actual FP32/BF16 conversions
is checked across runtimes before measuring. GPU objects occupy separate runtime
allocations; values, shapes and masks match.

Lead predictions P1–P4 are preserved verbatim in `registration.json`: fp32 parity
and fused prefill ≥2×; two-pass slower through S=8192; SP half-gap / long-decode
within 2× / no flash wins; APA peak ≤1/50 dense. Seat predictions:

- A1: FP32 math/dense parity passes all 50 cells at atol=rtol=1e-3.
- A2: BF16 flash is faster than FP32 SP on every available S≥2048 prefill cell;
  explicitly a mixed-dtype comparison.
- A3: at least one split-K decode cell has >.02 fraction mismatch after delta transfer.
- A4: both primary APA variants use ≤.02 of FP32 dense peak on every square S=8192 prefill.
- A5: no exact-shape crossover comparison to paper §4.5 because its B=2 differs.

Only A5 is established by registered shape/source inspection. All GPU predictions
remain BLOCKED. Prediction misses will be results, not reasons to adjust thresholds.

## G1 receipts and G2 handoff

The final baseline passed **23 tests**. Coverage: exact grid/calibration reuse,
registry, seeded dtype conversion, bottom-right rectangular and decode masks,
GQA grouped engine composition against an independent loop oracle and real torch
CPU math, full-output comparison including zero/nonfinite cases, timing sample
minimum, table assembly, stale/tampered/missing receipts, and prevention of a
false G2 pass with RED numerics or a blocked SP2 row. Initial baseline: 22 tests;
the additional report-status guard is logged as final validation.

Registered mutation check: **5/5 non-error defects killed**, fraction **1.0 ≥ .80**.
Defects: top-left masking, wrong GQA mapping, zero-reference false pass, hidden
nonfinite output and acceptance of fewer than seven timing calls. These are
temporary copies; the real source is hash-checked unchanged. The final mutation
receipt, all logs and dry-run checks are indexed in `G1_RESULTS.json`.

`G1_DRY_RUN.jsonl` enumerates 50 unique cells without a device. A test also blocks
torch import entirely while dry-running. Bash syntax validation passes.
No GPU gate was attempted; the unavailable device is recorded, not represented as
successful CUDA execution. See `G2_BLOCKED.md` for exact `list`, `run`, `resume`,
`summary` commands and the SP2 extension contract. `SPEED_CHAIN.md` currently
shows every missing cell; its writer will render timings and interpretation from
lead receipts.

## Files, deviations, risks and RED

Implementation: `scripts/apa_spd1_common.py`, `scripts/apa_spd1_bench.py`,
`scripts/apa_spd1_report.py`, `scripts/apa_spd1_lead_gpu.sh`,
`scripts/apa_spd1_mutations.py`, `tensor_cuda/tests/test_apa_spd1.py`.
Documents/receipts: this directory and `docs/APA_SPD1_LEDGER.md`.
`DELIVERY_MANIFEST.json` hashes the delivered files; `G1_RESULTS.json` includes
the final source-preservation check and gate log paths.

**RED calibration premise:** SP1 deltas were calibrated with perturbed keys and
global-prefix selection. This grid uses TurboQuant4 MSE reconstruction; split-K
uses partition-local prefixes. No matched refine-budget assertion follows from
reusing δ. A CPU sample estimate of the two-pass z-score fraction is compared with
the GPU diagnostic SP mask on the same sampled queries; threshold-boundary
rounding means it is not an exact CUDA baseline mask. Mismatches are marked.
E1 extras transfer full-context decode δ values and remain UNCALIBRATED even if
the estimate agrees. No tuning was performed. This is a disclosed departure from
the order's presumed existing matched calibration, not a solved calibration gate.

**RED optional SP2:** absent now. If a frozen file appears without a real pinned
launcher adapter, only that row is blocked and G2 coverage remains blocked.
The extension has not been exercised against an actual SP2 interface.

Measurement scope: engine CUDA pool and torch native allocator count allocations
differently. Peaks include outputs and torch efficient GQA expansion, exclude
resident inputs/kq/quantizer setup, and do not cover library-private allocations
outside those allocators. They are allocator high-water deltas, not total VRAM.
Kq remains floating reconstruction, so no compressed-cache storage claim. E1
same-shape BF16 rows use its r=.10/INT4 family but different shared-input seed and
rounding staging; its old wall medians and this sweep's event medians are distinct
measurements. Paper §4.5 has no exact-shape match. GPU support, timings, memory,
numerical results, and blind review are **not claimed fixed/verified** by G1.

Process safety: no git, no subagents, no writes to main/other worktrees, no
production edits, no service changes, no background waits or GPU runs. Runner
uses a live advisory lease and busy-card probe, holds the lease during a bounded
30-second foreground cooldown and launches only one cell per call. GNU timeout
can terminate only its own spawned worker/process group; it never targets an
existing process. No process was killed during implementation. Advisory locking
cannot prevent an uncooperative client from starting later.

## Prior art

Existing implementations are reused: Vaswani et al. (2017) dense attention;
[PyTorch SDPA](https://docs.pytorch.org/docs/main/generated/torch.nn.attention.sdpa_kernel.html)
(contributors, 2023–2026); [FlashAttention-2](https://arxiv.org/abs/2307.08691)
(Dao, 2023); [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025,
repository MSE component); APA/E1/SP1/SP1.1 (David and Project-Tensor seats, 2026);
[BLASST](https://arxiv.org/abs/2512.12087) (Yuan et al., 2025/2026), running-max
criterion; online normalizer (Milakov & Gimelshein, 2018), Flash-Decoding (2023),
and related [ThriftAttention](https://arxiv.org/abs/2605.23081) (Sharratt, 2026).
Lloyd-Max, uniform scalar quantization, quantile summaries, counterbalancing,
SHA-256, flock/timeout and mutation testing are existing methods. This seat's
work is benchmark plumbing, test fixtures, and honest receipt assembly.
`PRIOR_ART.md` records precise reuse boundaries and unverified bibliographic leads.

kernel sweep; this establishes nothing about model quality
