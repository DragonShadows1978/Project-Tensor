# APA-MF1 audit: causal-mask constructions

Scope: `tensor_cuda/tensor_cuda/*.py`, `tensor_gpu_v2/*.py`, `apa_cuda/**` (with its C++/CUDA), the named tests,
and, for verification only, `tensor_cuda/src/*` (kernels). Search terms: `triu`, `tril`, `causal`, `s_max`,
`S - L` / `S-L`, `Lq`/`row0`, `arange` comparisons (`ci > ri`, `col > row`), `visible`, `count=causal`.
"before" line numbers are from the fd32b6d copies in `artifacts/apa_mf1/before/`; "now" line numbers are from
the A+B tree.

Classes: **bottom-right** (row i sees cols ≤ S−L+i) / **top-left** (row i sees cols ≤ i on an (L,S) rectangle) /
**square-only** (a top-left formula only ever used with S == L) / **not a mask** (passes the flag on).

## Fixed in this order (writable list)

| # | Site (before) | Quote (before) | Class | Now |
|---|---|---|---|---|
| 1 | `tensor_cuda/tensor_cuda/quant.py:174` | `cmask = np.triu(np.ones((L, S), np.float32), 1)` | top-left (stats mask) | fix A, `quant.py:181` `absr_f = absr_f.masked_fill(causal_add.float(), 0.0)`, derived from the shared `F._causal_mask` |
| 2 | `tensor_cuda/tensor_cuda/quant.py:177` | `thr = absr_f.mean([-1], True) + absr_f.std([-1], True) * z` | not a mask: diluted stats (masked zeros counted over S) | fix B, `quant.py:183-197` visible-count population mean/std (causal); the non-causal branch is unchanged |
| 3 | `tensor_gpu_v2/_core.py:1660` (sdpa) | `causal_mask = xp.triu(xp.ones((L, S), dtype=xp.bool_), k=1)` | top-left | `_core.py:1677` `_causal_mask_bool(xp, L, S)` |
| 4 | `tensor_gpu_v2/_core.py:1752-1755` (flash_attention fwd) | `causal_mask = col_indices > row_indices  # (1, 1, block, S)` | top-left | `_core.py:1771` `_causal_mask_bool(xp, L, S, i, end_i)` |
| 5 | `tensor_gpu_v2/_core.py:1821-1823` (flash_attention bwd) | `causal_mask = col_indices > row_indices` | top-left | `_core.py:1837` |
| 6 | `tensor_gpu_v2/_core.py:1974-1976` (APA refine ≥ 1 fwd) | `scores = xp.where(ci > ri, xp.float32(-1e9), scores)` | top-left | `_core.py:1988` |
| 7 | `tensor_gpu_v2/_core.py:2009-2011` (APA refine ≥ 1 bwd) | `cmask = ci > ri` | top-left | `_core.py:2022` |
| 8 | `tensor_gpu_v2/_core.py:2063` (APA non-tiled fwd; also the ranking and `abs_ranking` zeroing at :2081-2094) | `cmask = xp.triu(xp.ones((L, S), dtype=xp.bool_), k=1)` | top-left | `_core.py:2074` |
| 9 | `tensor_gpu_v2/_core.py:2142-2145` (APA `use_tiled` fwd; also `masked_positions` :2170-2171) | `causal_mask = ci > ri` | top-left | `_core.py:2154` |
| 10 | `tensor_gpu_v2/_core.py:2249` (APA bwd non-tiled) | `cmask = xp.triu(xp.ones((L, S), dtype=xp.bool_), k=1)` | top-left | `_core.py:2258` |
| 11 | `tensor_gpu_v2/_core.py:2276-2278` (APA bwd `use_tiled`) | `cmask = ci > ri` | top-left | `_core.py:2285` |
| 12 | `apa_cuda/tests/test_parity.py:109` (reference_forward) | `cmask = np.triu(np.ones((L, S), dtype=bool), k=1)` | top-left reference (used square-only by the existing tests) | `test_parity.py:110` `k=1 + (S - L)` |
| 13 | `apa_cuda/tests/test_parity.py:153` (reference_backward) | same | same | `test_parity.py:155` |

The single tensor_gpu_v2 helper is `tensor_gpu_v2/_core.py:1606` `def _causal_mask_bool(xp, L, S, row_start=0, row_end=None):`,
returning `cols > rows` with `rows = xp.arange(row_start, row_end).reshape(-1, 1) + (S - L)`. The order's table named 3
tensor_gpu_v2 lines (1660, 2063, 2249). The audit found 6 more top-left constructions in the same file (#4-#7, #9, #11).
All 9 are fixed, because they are causal-mask constructions inside the writable file.

## Already bottom-right (verify, not edited)

| Site | Quote | Class |
|---|---|---|
| `tensor_cuda/tensor_cuda/functional.py:38` | `bias = np.triu(np.full((L, S), -1e4, dtype=np.float32), k=1 + (S - L))` | bottom-right (63960a4) |
| `tensor_cuda/tensor_cuda/functional.py:88` | `scores = scores + _causal_mask(L, S, query.device.split(":")[0], query.dtype)` | bottom-right (uses :38) |
| `tensor_cuda/tensor_cuda/functional.py:86,99` | `tc.causal_softmax(...)` → `kernels.cu:7221` `int64_t visible = S - L + i + 1;      // bottom-right causal bound (S >= L)` | bottom-right; `kernels.cu:7257` throws on S < L |
| `tensor_cuda/tensor_cuda/quant.py:155` | `causal_add = F._causal_mask(L, S, dev, query.dtype)` | bottom-right (score mask) |
| `tensor_cuda/tensor_cuda/__init__.py:1003-1032` | `apa_gemm_selective_attention(..., Lq=0, row0=0, window=0)` → `gemm_apa.cu:253` `int64_t valid_hi = (int64_t)s.S - s.full_lq + row0 + i + 1;` | bottom-right (chunk-aware) |
| `tensor_cuda/tensor_cuda/__init__.py:1076-1093` | `apa_blend_softmax(bulk, rank, zthr, Lq=0, row0=0, window=0)` → `kernels.cu:3287` `int64_t valid_hi = (int64_t)S - Lq + abs_i + 1;  // bottom-right causal bound` (and `:3446` sink) | bottom-right for Lq > 0; with Lq = 0 the caller's baked mask is used (not a construction) |
| `kernels.cu:1183` apa_selective_kernel | `int s_max = is_causal ? ((S - L) + i + 1) : S;` | bottom-right |
| `kernels.cu:1451` apa_selective_stats_kernel, `:1525/:1528` apa_selective_split_kernel, `:2050` apa_selective_sink_kernel, `:2370` apa_selective_fwd_train_kernel, `:2553` apa_selective_bwd_kernel, `:2752` apa_selective_bwd_bk1_kernel | `int s_max = is_causal ? ((S - L) + i + 1) : S;` | bottom-right |
| `kernels.cu:2914` apa_selective_bwd_bk2_kv_kernel | `const int first = causal ? max(0, j-(S-L)) : 0;` | bottom-right (key-owner form) |
| `kernels.cu:3085-3086, 3118` BK3 backward | `bool visible=qi+i<L && kj+j<S && (!causal \|\| kj+j<=S-L+qi+i);` | bottom-right |
| `kernels.cu:5495, 5655, 5714, 5767, 5840` int4 kernels | `const int s_max = is_causal ? ((S - L) + i + 1) : S;` | bottom-right; `:6801` rejects S < L |
| `kernels.cu:7872` apa_selective_sp_kernel (SP: out of scope) | `const int smax = causal ? S - L + i + 1 : S;` | bottom-right; `:7971` rejects S < L |
| `kernels.cu:8046-8113` BK4 forward | `float count=float(causal?S-L+qi+tid+1:S),mean=sums[tid]/count;` | bottom-right, with visible-count stats (the fix B rule) |
| `kernels.cu:8151` bk4 a_mask (diagnostic) | `mask[(int64_t)row*S+j]=(!causal \|\| j<S-L+i+1) && fabsf(dot*scale)>=thr[row];` | bottom-right |
| `tensor_cuda/src/apa_sp2.cuh:98` (SP) | `(causal && kq.shape[2]<q.shape[2])` → throw | rejects S < L (SP, out of scope) |
| `tensor_cuda/src/ops.cpp:430` | `Tensor causal_softmax(const Tensor& scores)` → kernels.cu:7221 | bottom-right |

**Verified by test** (on real S=8192 > L): the shipped `apa_selective_fwd_train_kernel` (`kernels.cu:2370`, through
`bp_kernel_4_diagnostic` variant 'a' plus `a_mask`) and BK4 forward (variant 'h'). Their selection masks match the
fp64 bottom-right visible-count rule at ≥ 0.9999974 in all 36 cells (`artifacts/apa_mf1/real/*.json` `kernel_diag`).
The other kernel rows above are classified **by source reading only**. No new MF1 test runs them.

## Top-left or square-only, outside the writable list (listed, NOT fixed)

| Site | Quote | Class | Consequence |
|---|---|---|---|
| `apa_cuda/csrc/apa.cpp:186-191` (`apply_masks`, used by fwd :260-261/:296-297, bwd :367, and stats zeroing in `compute_threshold` :155-156) | `auto ri = at::arange(row_base, row_base + rows, ...` / `auto cmask = (ci > ri);  // (rows,S) bool` | **top-left** | The apa_cuda CUDA extension is still top-left on S > L. With the reference now fixed, a rectangular causal parity test would correctly FAIL against it. Minimal fix (stated, not done): `ri = arange(row_base, row_base+rows) + (S - L_total)`, which needs the full L passed into `apply_masks`. The extension is not built on this machine (`ModuleNotFoundError: apa_attention_cuda`). |
| `tensor_cuda/tests/test_apa_phase6.py:36` and `:47` | `cm = np.triu(np.full((q.shape[2], S), -1e9, np.float32), 1)` / `absr = np.where(np.triu(np.ones((q.shape[2], S), bool), 1), 0.0, absr)` | square-only (only `test_apa_causal`, L=S=48) | `:50` also encodes the diluted stats (`absr.mean(-1)` + `absr.std(-1)` over S). Under fix B this test goes pass→fail (the only G1 delta). The fix is a reference update to visible-count stats, which goes with David's B decision. |
| `tensor_cuda/tests/test_attention_phase5.py:19` | `s = s + np.triu(np.full((L, S), -1e9, np.float32), 1)` | square-only (`test_sdpa_causal`, L=S=6) | none today; it would mislead on a rectangle |

## Divergences noted (not masks; not in scope)

- `tensor_gpu_v2/_core.py:2101-2108, 2197-2204` (before): the tensor_gpu_v2 APA selects on `|ranking|` (exact scores).
  Its stats come from `abs_ranking.mean/std` over all S with masked zeros counted (diluted). The fused kernel and
  quant.py select on `|bulk|` with visible-count stats. Fix B was ordered for `tensor_cuda.quant.apa_quant_attention` only,
  so this function keeps its own documented rule. The same applies to apa_cuda `compute_threshold` (`apa.cpp:155-166`).

## Not a mask

`tensor_cuda/tensor_cuda/nn.py:306/314/334-335/750-751` (pass `is_causal` to `F.scaled_dot_product_attention`);
`tensor_cuda/tensor_cuda/apa_sp2.py:73/89` (passes `is_causal` to the SP entry); `tensor_gpu_v2/_ext.py:1542` and
`tensor_gpu_v2/_training.py:1914` (`is_causal=False`); `apa_cuda/apa_attention/ops.py:60-82,148,178` (passes the flag to
apa.cpp); `apa_cuda/benchmark.py:57,60`; `apa_cuda/csrc/apa_kernels.cu` (no causal logic);
`tensor_cuda/src/apa_sp1_1.cuh:19` (comment).

## S < L

Out of contract. The kernels reject it (`kernels.cu:6801`, `:7257`, `:7971`; bk4 launcher `S<L` → throw). After this
order, `quant.py` neither raises nor emits NaN: rows with no visible key get an all −1e4 score row (uniform softmax), as
`F._causal_mask` already did, and fix B clamps the count to ≥ 1. Receipt:
`tensor_cuda/tests/test_apa_mf1_quant.py::test_mf1_quant_s_less_than_l_is_finite` (passes before, A and A+B).
tensor_gpu_v2 rows with no visible key are all −1e9 (uniform). That is unchanged in kind, and is not asserted.
