# APA-MF2 implementation ledger

Order: `orders/APA_MF2_EXACT_RULE_MASK_LEAK_EXTENSION.md`, sha256 `9272c014…dcb90f`. Seat: Claude Opus 5.5 high, run as an
Agent-tool subagent. Worktree: `WorkTree/tensor_fork_apa_wt_mf2` (branch `apa-mf2` = `apa-mf1` 42e19e6). Registration:
`artifacts/apa_mf2/registration.json`, sha256 `0f75c6d66b1be2c4caf424687db2903fc1ae33b4d37b5efb12ef70fc96e94ffc`.
Verdict: `artifacts/apa_mf2/verdict.json`. Tables: `artifacts/apa_mf2/tables.md`.

The house rules applied throughout:
- no git, subagents or deletes;
- native driver 595.91.07;
- `scripts/apa_mf2_env.sh` refuses the retired 595.84 override, and nothing set it;
- every GPU command ran as `flock -w 900 /tmp/forge-gpu.lock timeout 600 …`, synchronously.

## 1. Reading and the mechanism check (15:21–15:36)

**Read:** the order; `/mnt/ForgeRealm/HOUSE_RULES.md`; the MF1 review (`apa-mf1rev`): `docs/APA_MF1_REVIEW.md`,
`artifacts/apa_mf1rev/verdict.json` and `lead_gpu_results.md`; the mechanism doc `functional.py:20-43`; BK4
`kernels.cu:8021-8188`; and the train kernel `kernels.cu:2344-2420`.

**Found:**
- **No scope mismatch.** The exercised code paths are the ones the docs define.
- **BK4 takes BF16 operands only:** `kernels.cu:8163` "same CUDA BF16 required". The "fp32 item-1 cell" is therefore registered as the
  composed rule's fp32 arithmetic on the kernel's identical bf16-valued operands.
- **BK4 'h' accumulates its stats sequentially per row** (`kernels.cu:8065-8068`). The shipped 'a' kernel uses a lane-strided pass plus
  a tree reduction (`kernels.cu:2373-2399`).
- **Engine primitives:**
  - the fp32 matmul is `cublasSgemmStridedBatched` (`matmul.cu`);
  - `cumsum` is sequential (`kernels.cu:4774`);
  - `where`/`masked_fill` are true selects (`kernels.cu:762`, `ops.cpp:879-881`);
  - softmax initialises its max at -3.4e38 (`ops.cpp:841`), and `causal_softmax` at -3e38 (`kernels.cu:7216`);
  - divide is IEEE `x / y` (`kernels.cu:386`), and sqrt is `sqrtf` (`kernels.cu:411`).
- **`apa_cuda`:**
  - `apply_masks` is top-left (`apa.cpp:186-191`);
  - its fill `kNegInf = -1e9` is finite (`apa.cpp:34`).
- **Consumers of `_causal_mask`:**
  - GRM `gemma4_tc.py:740` and GRAPA `attention.py:82-84` add the bias;
  - GRM's Qwen3 standard path reaches it through `F.scaled_dot_product_attention` (`mistral7b_tc.py:491`).

## 2. Build (host only)

- `scripts/apa_mf2_build.sh` is a copy of `apa_mf1_build.sh`; only the paths changed. `scripts/apa_mf2_env.sh` is new: it uses the
  native driver and omits the MF1 env file's 595.84 override.
- `.so` sha256 `8ca4de6c…0908a`, which is **not** MF1's `8a13e5b3…cc437`. The investigation:
  - 19/19 sources are identical;
  - the g++ objects are byte-identical;
  - every nvcc `-dc` object differs;
  - after the nvcc module-id strings are normalized, the `kernels.o` SASS diff is 0 lines, and `matmul.o` also diffs to 0.
  - **Cause:** nvcc's per-compilation module ids. The device code is identical. Receipt: `logs/apa_mf2_build_provenance.log`.

## 3. Pre-registration design probe P0 (GPU, not a gate; disclosed in the registration)

**Question:** can any composition of engine ops be bit-identical to BK4?

**Method:**
- Extract BK4's own WMMA dots through its threshold output. With S=L=1, scale 1 and z 0, `th = |dot|` exactly.
- Rebuild BK4's thr and mask with an fp32 transcription of the compiled SASS: `scripts/apa_mf2_bk4emu.py`, using FFMA emulation with
  a TwoSum tie repair.

**Results** (`logs/apa_mf2_p0_probe_1p7b_L16_L2048_b8.log`, Qwen3-1.7B L16, L2048, bulk 8, 2 KV heads):
- The transcription on WMMA dots reproduces BK4's thr bits (0/4096 rows differ) and its mask (0/33,554,432 entries). The model of the
  kernel is confirmed.
- The engine's SGEMM dots differ bitwise from WMMA dots in 25,068,385 of 33,554,432 entries. Call this R1.
- Mask mismatches vs 'h' under each arithmetic:

  | Statistics arithmetic on the engine's SGEMM dots | Mask mismatches |
  |---|---|
  | BK4 transcription (R1 only) | 2 |
  | Engine tree sums | 15 |
  | Sequential `cumsum` plus host emulation | 2 |

- The two kernels themselves: 'h' vs 'a' disagree on 15 mask entries and on 3980/4096 thr bit patterns.

**Decision (registered rail):** exact parity requires WMMA-identical dots, which is a kernel or engine change and out of scope. Keep
the faithful rule components with plain engine ops. Do not add sequential-order or host-emulation knobs.

**SASS receipts** (`logs/apa_mf2_bk4_sass_excerpt.log`):
- `:5` "FFMA R4, |R4|, |R4|, R21" (squares);
- `:4` "FADD R5, |R4|.reuse, R33" (sums);
- `:21,:37` "FCHK" (IEEE div);
- `:47` "FFMA R3, -R32, R32, R3" (var);
- `:63` "FFMA R3, R4, c[0x0][0x1c4], R32" (th).

## 4. Registration (15:43)

`artifacts/apa_mf2/registration.json` and `.sha256` are create-only and chmod a-w. It covers:
- definitions, census and gates G1–G4;
- mutants M1–M9 and predictions P1–P3, S1–S4;
- the rails.

## 5. Tests first (failing receipt on the MF1 A+B tree)

**New test files:**
- `tensor_cuda/tests/test_apa_mf2_rule.py`;
- `tensor_cuda/tests/test_apa_mf2_mask.py`;
- `apa_cuda/tests/test_apa_mf2_extension.py`.

**Stub harness:**
- `scripts/apa_mf2_ext_stub_build.py` and `scripts/apa_mf2_ext_stub_kernels.cpp`.
- It compiles the shipped `apa.cpp` as a C++-only torch extension, with `apa_kernels.cu` replaced by an ATen transcription. It is
  labeled as a harness, not the shipped extension.

**Official extension build** (`apa_cuda/setup.py build_ext --inplace`) fails:
- `logs/apa_mf2_ext_build_before.log:29` "RuntimeError: ('The detected CUDA version (%s) mismatches the version that was used to
  compilePyTorch (%s)…', '12.0', '13.0')";
- `:33` gives the same with '12.6'.
- torch 2.11 is cu130, and no CUDA 13 nvcc exists on the machine.

**Receipts on the before tree:**
- `logs/apa_mf2_testsfirst_tensor_cuda.log:3919` "187 failed, 71 passed". The failures are the intended kind:
  - B1 `[[[[True, True]]]] vs BK4 [[[[False, False]]]]`;
  - B2 `[F,F] vs [F,T]`;
  - B3 27 vs 24;
  - N1 0.2797;
  - leaks down to 5.6e-45;
  - the -1e4 mask contract;
  - `test_apa_causal`.
- `logs/apa_mf2_testsfirst_apa_cuda_stub_before.log:287` "10 failed, 109 passed": the 10 are my rectangular tests (top-left).
- `logs/apa_mf2_testsfirst_apa_cuda_noext.log:164` "6 failed, 101 passed, 1 skipped". The 6 are `test_parity` ModuleNotFoundError,
  as in MF1's receipt.
- **Reviewer probes** (copies in `scripts/apa_mf2_reviewer/`, edits recorded in `SHA256SUMS.orig_mf1rev_3bee9fa` plus the diff in this
  ledger's section 9):
  - red-extension (stub before): `logs/apa_mf2_reviewer_before_extension.log:5` "actual [0. 0. 0. 0.]";
  - red-finite-bias: `:10` "actual [0.27970806 …]";
  - quant-zero: "expected 24 actual 27";
  - quant-rule: the reviewer's own `t.ge(2.)` tool error, after the B1 diagnostic printed BK4 thr 0.12549718 and selection `[0,0]`.

**Test bug, fixed:** my SDPA grad test read `.grad` of a bf16 non-leaf. `tc.tensor(dtype='bfloat16')` returns `astype` of an fp32 leaf
(`__init__.py:45`), so the test now differentiates fp32 leaves.

## 6. Item 1: `quant.py` selection (BK4 rule)

**`_bk4_selection`** (`quant.py:157-197`):
- `b` is the fp32 unrounded dot (SGEMM, `alpha=scale`);
- the `|b|` invisible entries are zeroed by the shared mask predicate (`:190`);
- `mean = sum / count` (IEEE div, `:191`);
- `var = (sum a² / count - mean²).relu()` (`:192`);
- `thr = mean + var.sqrt() * z` (`:193`);
- `sel = (a >= thr) * visible` (`:196`).

**Callers:** bf16/fp16 upcast the same operands (`:248`); fp32 reuses the unmasked bulk (`:245-246`).

**Reference:** `test_apa_phase6._ref` now uses the kernel rule (`test_apa_phase6.py:61` "mask = vis & (absr >= thr)"). Nothing else in
it changed.

**Receipts:**
- `logs/apa_mf2_item1_tests.log:151` "7 failed, 43 passed". All 7 are exactness cells with count-1 rows.
- `logs/apa_mf2_count1_witness.log:1` "RN(a^2)-a^2=+4.089e-12 kernel thr=… (thr-a)/a=+1.515e-04 kernel_sel=0 composed_sel=1". This is
  BK4's artifact.

**Snapshot:** `artifacts/apa_mf2/variants/quant_item1.py`. **Patch:** `artifacts/apa_mf2/item1.patch`.

**Softmax operand difference — reported, not changed** (it lies outside the selection block):
- BK4 softmaxes the unrounded fp32 bulk for unrefined keys and the fp32 exact dot for refined ones (`kernels.cu:8076,8082-8083`).
- The composed path uses the operand-dtype `tc.matmul` outputs.

**Prior art:**
- `kernels.cu` BK4 (Project-Tensor 2026): the rule matched.
- GRAPA `_visible_counts`.
- FlashAttention-2 bottom-right (Dao 2023) and 63960a4.
- The engine-op composition is mine.

## 7. Item 2: mask without a finite bias

**Changes:**
- `functional.py:50` "bias = np.triu(np.full((L, S), -np.inf, dtype=np.float32), k=1 + (S - L))";
- SDPA `functional.py:101-102` uses the predicate `masked_fill(..., -inf)`;
- `quant.py:226-227` uses the predicate `masked_fill(causal_add, -inf)`;
- `apa.cpp:40` "constexpr double kNegInf = -INFINITY;" (audit finding).

**Patch:** `artifacts/apa_mf2/item2.patch`. The `apa.cpp` hunk is built against the intermediate `variants/apa_item2.cpp`.

**Prior art:** FlashAttention-2 (Dao 2023) predicate masking with -inf; PyTorch SDPA boolean `attn_mask` semantics.

## 8. Item 3: `apa.cpp` bottom-right

**Change:** `apply_masks` takes the full L, and `ri = arange(row_base + (S - L), …)` (`apa.cpp:196,200`). All 5 call sites pass L.

**Patch:** `artifacts/apa_mf2/item3.patch`.

**Official build after:** `logs/apa_mf2_ext_build_after.log:2`, the same RuntimeError.

**Stub after:**
- `logs/apa_mf2_after_apa_cuda_stub.log` "119 passed";
- red-extension `logs/apa_mf2_reviewer_after_extension.log:5` "actual [3. 3. 3. 3.]".

**Prior art:** 63960a4 and the Dao 2023 v2.1 bottom-right causal.

## 9. Gate runs and results

**Patch check:** applying patches 1, 2 and 3 in order to the before copies reproduces all 7 files byte-for-byte.

**G2, real data** (`scripts/apa_mf2_item1_real.py`; `artifacts/apa_mf2/real/*.json`; `logs/apa_mf2_g2_real_*.log`):
- MF1 A+B agreement vs 'h' is 0.99867630–0.99949397 (1,546,250 mismatches).
- MF2 is 0.9999971390–1.0: exact in 5 of 36 cells, 1326 mismatches in total.
- 0 MF2 mismatches lie outside the band; the max relative gap is 1.12e-5.
- Plumbing holds in 36/36.
- 'h' vs 'a' disagree on 1430 entries in total; MF2 vs 'a' on 364.

**G2, synthetic** (`artifacts/apa_mf2/synth_item1.json`, `logs/apa_mf2_g2_synth.log`):
- MF2 is exact in 57/72 cells; MF1 A+B in 27/72.
- The bf16 out-of-band count is 4, all at count-1 rows (the BK4 artifact).
- The fp16 out-of-band count is 0.
- B1, B2 and B3 are RED on before and GREEN on after.

**G3** (`artifacts/apa_mf2/item2_gate.json`):
- N1: before row0 0.27970806; after 0.0.
- Leak sweep: 160/160 in-domain cells pass, i.e. exact zero and finite.
- Before arms: 144/168 cells leak.
- 8 out-of-domain cells hit quantizer overflow. Blind rows stay at 0, but visible rows go NaN; MF1 A+B was already NaN there.
- Real-capture byte identity: 72/72.

**GRM** (`artifacts/apa_mf2/grm/chunked.json`, `logs/apa_mf2_grm_chunked.log:12`):
- Standard logits are byte-identical before vs after.
- `"apa"` KL: MF1 A+B 0.0040903 (MF1's own number reproduced), MF2 0.0043640, i.e. +6.69%.
- Top-1: 32/32 vs 31/32.
- The paired difference is 0.23 standard errors.

**GRAPA:** `artifacts/apa_mf2/grapa.json`, 12/12 byte-equal.

**Additive pattern** (GRM `gemma4_tc.py:740`, GRAPA): `artifacts/apa_mf2/additive_consumer.json`, 9/9 byte-equal.

**Neither GRM nor GRAPA relies on the bias being finite:**
- **Source:** both add the bias, and then softmax (GRM) or `abs` + `masked_fill` (GRAPA, `attention.py:92-95`). The engine's
  `where`-based `masked_fill` is a select, and every S ≥ L row has a visible key.
- **Execution:** the byte-identical runs above.
- **Not executed:** GRAPA's composed refine < 1 causal branch. With `kq` present it is router-unreachable; source reading only.

**G1** (`logs/apa_mf2_g1_full.log:1979`, "45 failed, 1029 passed, 30 skipped, 2 warnings, 15 errors in 131.84s (0:02:11)"):
- 52 baseline ids are unchanged, and `test_apa_causal` passes.
- There are 8 new failing ids: 7 registered exactness tests, plus `test_apa_mf1_quant.py::test_mf1_quant_s_less_than_l_is_finite`
  (`logs/apa_mf2_g1_compare.log:4`).

**Root cause of the MF1 test failure:** with -inf, out-of-contract S < L rows have no visible key and softmax to NaN. The order protects
only S ≥ L. This is a semantics decision for the lead, and it is not changed (rail).

**`apa_cuda` tests:**
- with no extension: before = after = 6 ModuleNotFound / 101 passed / 1 skipped;
- through the stub: 10 failed → 0.

**Mutation** (`scripts/apa_mf2_mutation.py`; `artifacts/apa_mf2/mutation/*.json`): 9/9 killed, 0 errors.

| Mutant | Assertion kills |
|---|---:|
| M1 | 18 |
| M2 | 3 |
| M3 | 31 |
| M4 | 15 |
| M5 (stub) | 10 |
| M6 | 20 |
| M7 | 4 |
| M8 | 1 (B2) |
| M9 | 1 (B2) |

The rule: a mutant counts as killed when a baseline-passing test fails with an AssertionError.

**Prior art:** DeMillo, Lipton & Sayward 1978. Unverified — lead to check "Hints on Test Data Selection".

**Reviewer-probe copy edits:**
- `apa_mf1rev_runtime.py`: the source pin is replaced by printed hashes;
- `probe_quant.py`: the engine pin is the MF2 sha.

Both diffs were printed at copy time. Everything else is verbatim (`SHA256SUMS.orig_mf1rev_3bee9fa`).

## 10. Prior art (all sites)

- **`kernels.cu` BK4** (Project-Tensor 2026): the selection rule matched. I took the visible-count population one-pass moments, the
  clamp, the visible predicate and `>=`. The composition from engine ops is mine.
- **63960a4** (Project-Tensor, 2026-06-09) and **FlashAttention-2** (Dao 2023, v2.1): the bottom-right causal convention and -inf
  predicate masking. I used the convention; I copied no code.
- **PyTorch SDPA `attn_mask`:** boolean masked entries become -inf before softmax. Taken as the semantics.
- **GRAPA `attention.py` `_visible_counts`** (project-local): visible-count statistics.
- **IEEE-754 FMA**, **Knuth TwoSum** (TAOCP vol. 2) and **Figueroa 1995** ("When is double rounding innocuous?") underpin the fp32
  emulation in `apa_mf2_bk4emu.py`. Unverified — lead to check.
- **Fasi, Higham, Mikaitis & Pranesh 2021**, "Numerical behavior of NVIDIA tensor cores": relevant to R1, since tensor-core
  accumulation is not IEEE round-to-nearest. Unverified — lead to check.
- **Chan, Golub & LeVeque 1983** (variance algorithms), on cancellation in one-pass variance: this explains the BK4 count-1 artifact.
  Unverified — lead to check.
- **Extracting kernel dots through the threshold** at S=L=1, z=0: no prior art known to me.
- **Mutation testing:** DeMillo, Lipton & Sayward 1978. Unverified — lead to check.
- **Teacher-forced KL:** standard LM evaluation. Unverified — lead to check "KL divergence teacher forcing quantized model evaluation".
- **Stub harness and transcription:** mine; `torch.utils.cpp_extension` is standard tooling.

## 11. Scratch left behind

All of it is under `/tmp`, which is exempt:
- `/tmp/claude-1000/apa_mf2_mut/`;
- `/tmp/claude-1000/apa_mf2_patchcheck/`;
- the session scratchpad.

In the worktree, the stub build trees `artifacts/apa_mf2/ext_stub/{before,after}/{build_tmp,src}` are receipts. Nothing was moved to
DELETE_ME.
