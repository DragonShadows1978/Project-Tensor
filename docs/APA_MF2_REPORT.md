# APA-MF2 report (lead-written from the Opus 5.5 seat's final message; verified against disk)

- Order: `orders/APA_MF2_EXACT_RULE_MASK_LEAK_EXTENSION.md`.
- Registration: `artifacts/apa_mf2/registration.json`, sha256 `0f75c6d6…4ffc`.
- Verdict: `artifacts/apa_mf2/verdict.json`.
- Ledger: `docs/APA_MF2_LEDGER.md`.
- Commits:
  - item 1: eb81a46;
  - item 2: 66287b1;
  - item 3: 1b81120.

## Outcome

| Item | Result |
|---|---|
| 1. Fix B to the exact BK4 rule | B1/B2/B3 fixed (G2c GREEN). Agreement with the BK4 'h' diagnostic mask rose from 0.99868–0.99949 to **0.9999971–1.0** on 36 real-capture cells, with mismatches falling from 1,546,250 to 1,326. **Every real-data mismatch lies inside the 2e-5 tie band** (max gap 1.12e-5). Bit-exact parity (G2a) is RED: it would need an engine op that reproduces BK4's tensor-core dot bits and its FMA-fused statistics (the seat reproduced both bit for bit in an emulator). The seat stopped at the registered rail. |
| 2. The finite −1e4 causal leak | **Fixed.** `_causal_mask` is −inf and applied as a predicate, and `apa.cpp`'s `kNegInf` is −inf. The leak case gives exactly 0. **Byte-identical** on 72/72 real-capture cells, GRM standard logits, the GRAPA forward and GRM's additive pattern. No NaN for S ≥ L. |
| 3. The `apa.cpp` top-left mask | **Fixed.** The official build is impossible here (torch cu130 vs nvcc 12.x). The fix is verified from source and through a labeled stand-in build: the reviewer's probe gives `[3,3,3,3]`, against `[0,0,0,0]` before. |
| Mutation | 9/9 killed |

## Residual test failures (lead re-run: 8 failed, 315 passed, 12 skipped)

- **7 × `test_apa_mf2_rule.py::test_bf16_selection_equals_bk4_kernel_mask`** (registered exactness tests). The
  synthetic out-of-band cases are rows with ONE visible key. There, BK4's FMA-fused variance comes out slightly
  positive, so the kernel drops the row's only key, while the definition (and MF2) selects it. The output is
  unaffected: one visible key gets softmax weight 1 whether or not it is refined.
- **1 × `test_apa_mf1_quant.py::test_mf1_quant_s_less_than_l_is_finite`.** Out-of-contract S < L rows have no
  visible key, so with −inf masking they are NaN. They used to be a uniform average of masked keys. The CUDA kernels
  raise on S < L.

## Lead verification

- `git status` shows only the four permitted files modified, plus the new tests and receipts.
- Applying `item{1,2,3}.patch` in order to `apa-mf1` reproduces all seven files byte-for-byte.
- The lead re-ran the MF2, MF1 and phase6 tests: the same 8 failures.
- The production repo is untouched (`fd32b6d`).

## Open items

1. Accept the item-1 in-band residual, or build an engine op that matches BK4's dots and statistics.
2. S < L for the causal composed path and SDPA: raise (matching the kernels) or stay NaN.
3. The BK4 single-key quirk. It affects the selection only, never the output.
4. Finite `−1e9` fills remain in `tensor_gpu_v2` and the `apa_cuda` test reference. They leak only above 1e9.
5. The extension needs a CUDA 13 toolkit to build.
6. The GRM `"apa"` KL changed +6.7% against MF1 A+B (0.23 SE over 32 steps, which is noise).
