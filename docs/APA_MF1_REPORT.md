# APA-MF1 report: rectangular causal masks (fix A) and visible-count stats (fix B)

The seat (Claude Opus 5.5 high, Agent subagent) could not write this file: the harness refuses report-file writes
from subagents. The lead wrote it from the seat's final message and verified it against disk (see "Lead
verification").
- Registration: `artifacts/apa_mf1/registration.json`, sha256 `8129cfbadc7daa7c3c31885720196b1cbfd8ddf5e7d6052cca43e29fca6cefec`.
- Verdict: `artifacts/apa_mf1/verdict.json`.
- Full tables: `artifacts/apa_mf1/tables.md`.
- Audit: `artifacts/apa_mf1/audit.md`.

## Verdict

- **Fix A (bottom-right alignment): GREEN.** 13 top-left constructions are fixed:
  - `quant.py` refine-stats mask;
  - 9 sites in `tensor_gpu_v2/_core.py`, behind one helper `_causal_mask_bool`. The order named 3; the audit found
    6 more: the flash forward and backward passes, APA refine ≥ 1, and the `use_tiled` forward and backward;
  - 2 test references in `apa_cuda/tests/test_parity.py`.

  Results against the fp64 definition:
  - rectangular cases match: fp32 ≤ 1.48e-6, bf16 ≤ 1.07e-2 relative;
  - every square case is byte-identical to `fd32b6d`;
  - the existing suite shows 0 of 789 baseline test ids changed.
- **Fix B (visible-count population stats in `apa_quant_attention`, matching `kernels.cu` BK4 and GRAPA): G3 GREEN.**
  - Selection agreement with the fused-kernel rule is 1.000000 on synthetic data and ≥ 0.99999985 on real captures.
    Before fix B it was 0.885–1.0 at state A, and 0.765–0.850 on the unchanged code on real data.
  - It changes square-case selection by design, and one existing test, `test_apa_phase6::test_apa_causal`, encodes
    the old diluted rule. It FAILS under A+B, while A alone passes.
  - Fix B cuts the refined fraction from about 17.5% to about 15.5% (the nominal 15%), so output error rises
    slightly. Promoting it is **David's call**.
- **Mutation: 13/13 killed.**

## Key real-data result (S=8192 captures, last L query rows, refine 0.15)

Before the fix, a query row at the end of a long prefix could refine only the first i+1 keys:

| L | Eligible fraction before the fix |
|---:|---:|
| 64 | 0.4% |
| 512 | 3.2% |
| 2048 | 14% |

After fix A it is 100%.

Output relative error against exact attention:

| Model, layer | bulk 4: before → A | bulk 8 (GRM's Qwen setting): before → A |
|---|---|---|
| Qwen3-1.7B L16 | 0.088 → 0.0094 | 0.015 → 0.0089 |
| Qwen3-4B L21 | 0.124 → 0.072 | 0.0185 → 0.0117 |

Layer 0 remains a high-error outlier at bulk 4 (relative error about 0.5), as before; that is unrelated to the mask.

## Live GRM path (Qwen3-1.7B, `attention_mode="apa"`, refine 0.15, bulk 8, wikitext-2)

| Prefill | Arm | KL vs standard (mean / max) | Top-1 |
|---|---|---|---|
| 8,192 tokens in 512-token chunks, then 32 steps | before | 0.004614 / 0.0296 | 31/32 |
| same | A | **0.003697** / 0.0255 | 31/32 |
| same | A+B | 0.004090 / 0.0312 | 32/32 |
| unchunked 4,096 (S == L control), then 32 steps | before = A | 0.005747 (logits byte-identical) | 31/32 |
| same | A+B | 0.007568 | 32/32 |

The control used 4,096 tokens: an unchunked 8,192 does not fit 12 GiB in the composed APA path.

## Still top-left or divergent (outside the writable list; open)

- `apa_cuda/csrc/apa.cpp:191`: `cmask = (ci > ri)` is top-left. That extension is not built on this machine.
- `tensor_gpu_v2` and `apa_cuda` APA select on |ranking| with diluted stats, a separate divergence.
- `tensor_cuda/tests/test_apa_phase6.py`: its square-only reference uses diluted stats. It must change if fix B is
  promoted.
- Not covered:
  - kernels other than the two verified by test;
  - the SP, packed and K1 entries;
  - `adaptive_heads`;
  - `attn_mask` combined with causal;
  - fp16;
  - B > 1.

## Predictions

- **P1 held.** Before the fix, exactly S−L visible columns were excluded per row; the per-row mean was 0.031983 at
  L=512.
- **P2 held, qualified.** The change is larger at bulk 4 in 18 of 18 groups, but at bulk 8 it is 5–46% of the before
  error, not uniformly "a little".
- **P3 held.** The GRM KL dropped 20% after A.
- **P4 held.** Every rectangular `tensor_gpu_v2` cell had relative error ≥ 0.91 before the fix.

## Lead verification (2026-10-07)

- **Scope:** `git status` shows only the 3 permitted files modified. The production repo is untouched: `fd32b6d`,
  with 0 tracked changes.
- **Before copies:** `artifacts/apa_mf1/before/*` are byte-identical to `fd32b6d`.
- **Patches:** `fixA.patch` then `fixB.patch`, applied to `fd32b6d`, reproduce all three delivered files
  byte-for-byte.
- **Tests:** the lead re-ran the three new suites on the GPU: `166 passed, 12 skipped in 49.42s`, which matches the
  seat.
- **Code review of the diffs:**
  - fix A derives the stats mask from the shared `F._causal_mask` bias, the 63960a4 convention;
  - the `tensor_gpu_v2` sites all go through one bottom-right helper;
  - fix B uses per-row visible counts `S−L+1 … S`.
- **Cross-model review:** pending (Codex).

## Prior art

- FlashAttention-2 (Dao, 2023): the bottom-right convention.
- Project-Tensor 63960a4 (2026-06-09).
- `kernels.cu` BK4 and GRAPA `attention.py` `_visible_counts`: the visible-count population stats.
- Mutation testing: DeMillo, Lipton & Sayward (1978).
- Teacher-forced KL: unverified, lead to check.
