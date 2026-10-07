# APA-MF2.2 implementation ledger

Order: `orders/APA_MF2_2_REVIEW_FIXES.md`, sha256 `dde5a35f…4b0a0`. Seat: Claude Opus 5.5 high (Agent-tool subagent).
Worktree: `WorkTree/tensor_fork_apa_wt_mf2` (branch `apa-mf2`, tree = MF2.1 c5c468d content at start).
Registration: `artifacts/apa_mf2_2/registration.json`, sha256 `478f19d35a5adb30111f7d90997c686fc5a1f8b3942a7978c35089927f5a307b`
(create-only, `chmod a-w`). Verdict: `artifacts/apa_mf2_2/verdict.json`. Tables: `artifacts/apa_mf2_2/tables.md`.

House rules as applied: no git (none run this order); no subagents; no deletes (scratch only under `/tmp/claude-1000/`);
every GPU command `flock -w 900 /tmp/forge-gpu.lock timeout 600 …`, foreground, native driver 595.91.07; the env scripts
refuse the 595.84 override and nothing set it; no process I did not start was touched.

## 1. Reading and mechanism check (18:40–18:47)

Read: the order; `apa-mf21rev` `docs/APA_MF2_1_REVIEW.md`, `artifacts/apa_mf21rev/verdict.json`, `tests/apa_mf21rev/gpu_probe.py`
(+ `oracle.py`, `provenance.py`), `scripts/apa_mf21rev_gpu.sh`, the lead's 8 GPU logs `logs/apa_mf21rev_gpu_*_20261007T2*.log`;
this worktree's `docs/APA_MF2_1_LEDGER.md`, `artifacts/apa_mf2_1/verdict.json`; `/mnt/ForgeRealm/HOUSE_RULES.md`; AGENTS.md.

Mechanism doc = the recurrence comments at `kernels.cu` `fused_sdpa_noncausal_kernel` (:5066–5101 now) and
`apa_int4_sdpa_noncausal_kernel` (:6196–6234 now), plus `quant._definition_selection`. The order's measures exercise exactly
these code paths (direct kernel calls with TC_ATTN_QTILE=0, and the composed selection). No scope mismatch.

Lead-log facts used: N1 18/18 NaN (`fused_overflow`), N2 6 RED (`definition`), bf16 'NoneType' 1+10 rows, training bf16 6 rows,
`masks geometry/2048x2048` 36/2048 off by ≤9.2e-4 (rel ≤2.03e-6), `identity_mf2` "identity arm failed: 048949d, exit 1".

## 2. Before copies, production engine, registration (18:46–18:49)

- `artifacts/apa_mf2_2/before/` = MF2.1 `kernels.cu` `13d2a2eb…`, `quant.py` `fa820b7a…` (SHA256SUMS, read-only). Tree engine
  sources == `artifacts/apa_mf2_1/build/manifest.json` (0 mismatches); tree package == reviewer c5c468d pins (8/8);
  `artifacts/apa_mf2_1/before/pkg` == reviewer 048949d pins (8/8); `tensor_cuda/__init__.py` is `2aa0435a…` in all three revisions.
- **fd32b6d engine** built by this seat: scratch root `/tmp/claude-1000/apa_mf2_2_fd32b6d_src` = tree `src/include` with the
  reviewer's fd32b6d `kernels.cu`/`ops.cpp`; all 19 source hashes == the reviewer's fd32b6d pins (asserted before the build) →
  `artifacts/apa_mf2_2/build_fd32b6d`, module sha256 `eeeaf8f6be6bebc07a771e13fcecbf861cf9dc2544e1bb81f38c06d1cf152907`
  (`logs/apa_mf2_2_build_fd32b6d.log`). Its sources equal MF2's (as the reviewer noted), but it is a separate build.
- Registration written and frozen at 18:48 before any GPU command. **Clerical error:** its `registered_at` text says 19:05.

## 3. Tests first (18:50–18:52)

New: `tensor_cuda/tests/test_apa_mf2_2_n1.py` (review witnesses, −inf-first sweep, bf16 rejection, fused F1 rows, ordinary
rows) and `tensor_cuda/tests/test_apa_mf2_2_n2.py` (isolated range sweep, review N2 witnesses, mixed-magnitude rows, zero rows,
full-API range). Oracle: `scripts/apa_mf2_2_oracle.py` (mine).

| Receipt | Engine | Result |
|---|---|---|
| `logs/apa_mf2_2_testsfirst_mf2_1.log` (v1) | MF2.1 | 143 failed — 4 of them my oracle bug (einsum broadcast in `softmax64`); fixed to a matmul |
| `logs/apa_mf2_2_testsfirst_mf2_1_v2.log:3613` | MF2.1 | "139 failed, 66 passed": all 18 witnesses, 72 sweep, 36 range, 6 full-API, 6 N2 witness, 1 mixed |
| `logs/apa_mf2_2_testsfirst_n1_on_mf2.log:221` | MF2 (048949d) | "8 failed, 95 passed": only the 8 fused F1 rows (old −3e38 start); the N1 sweep passes on MF2 |

## 4. Item 1 — N1 guard (`artifacts/apa_mf2_2/item1.patch`, 18:53)

Both recurrences, nothing else: `const float m_ref = m_new == -INFINITY ? 0.f : m_new;` then `corr = expf(m - m_ref)`,
`w = expf(score - m_ref)` (`kernels.cu:5098-5100`, `:6231-6233`). Start stays `float m = -INFINITY;` (`:5075`, `:6200`); the
merge is unchanged. Comments corrected: the old "Lk >= 1 … so gmax is finite" overstated the invariant (review :19).
Non-comment diff: +7 −5 lines (6 recurrence lines + the int4 start line whose trailing comment moved).

Why byte-identical on finite rows (reasoning, then measured in §7): for finite `m_new`, `m_ref == m_new`, the same
expressions as MF2.1; versus MF2/fd32b6d (start −3.0e38f) the first finite score s > −3e38 gives corr = expf(−3e38 − s) = 0
there and expf(−inf) = 0 here, so l and acc agree from the first key on.

Prior art: FlashAttention-2 (Dao 2023), `softmax.h` `Check_inf` rescale ("max == −inf ? 0 : max"); Milakov & Gimelshein
(2018) online normalizer. Unverified — lead to check ("flash-attention softmax.h Check_inf", "Online normalizer calculation
for softmax 2018"). Taken: the guard. Mine: its placement in this warp-streaming recurrence and the witnesses/sweep.

## 5. Item 2 — N2 range-robust statistics (`artifacts/apa_mf2_2/item2.patch`, 18:54)

`quant.py:216-225`: `amax = absb.max([-1], True)`; `amax = amax + (amax <= 0.0)` (all-zero row → 1); `u = absb / amax`;
mean/centered var/thr of `u` as before; `sel = u.ge(thr)` (scaled-domain comparison). No masked_fill or tc.where added (MF1
capture contract). Reasoning (docstring): u, mean, |d| ≤ 1 so no overflow; mean ≥ 1/cnt keeps nonzero deviations away from
underflow; the definition is degree-1 homogeneous in |b|, so the scaled predicate is the definition's; the one division per
entry adds ≤ 2^-24 relative rounding (inside the 2e-5 band).

Prior art: Blue (1978, ACM TOMS 4(1)) and reference BLAS `snrm2/dnrm2` (scale by the largest magnitude before squaring);
centered two-pass variance, Chan, Golub & LeVeque (1983). Unverified — lead to check ("Blue 1978 Euclidean norm", "BLAS dnrm2
scale ssq", "Chan Golub LeVeque 1983 sample variance"). Taken: scale-before-square. Mine: per-row visible-max scaling of this
selection, the zero-row guard, the scaled-domain comparison.

Build: `scripts/apa_mf2_2_build.sh` (copy of `apa_mf2_1_build.sh`, same flags) → `artifacts/apa_mf2_2/build`, module sha256
`bc7ed2a9fa68822114a69bfbe05ffc60af554b82e30e9fac7ba787e42d8c698d` (`logs/apa_mf2_2_build.log`).
`logs/apa_mf2_2_tests_after.log:4` "205 passed in 2.68s".
Patch chain: `logs/apa_mf2_2_patches.log:3` "PATCH CHAIN REPRODUCES TREE: True" (after: kernels.cu `17c89f3c…`, quant.py `3ff09e9b…`).

## 6. G1 (18:55–18:58; v2 19:14–19:17)

`logs/apa_mf2_2_g1_compare.log`: 37 failed + 15 errors = 52 ids == baseline, 0 new. After the post-mutation test addition (§11)
re-run: `logs/apa_mf2_2_g1_compare_v2.log:2-7` "37 failed, 1651 passed, 30 skipped, 2 warnings, 15 errors", "baseline ids 52 now
ids 52", "new failing ids: 0 []", "G1 PASS". Side effect: the suite regenerated the 17 `artifacts/wo*.ppm` (as in MF2.1).

## 7. G2 — N1 and byte identity (18:58; v3 19:18)

`scripts/apa_mf2_2_identity.{py,sh}`: one process per engine (fd32b6d / MF2 / MF2.1 / MF2.2), each with its own package,
`TC_ATTN_QTILE=0`; the arm asserts its engine path; compare refuses equal engine hashes.
v1 failed before any arm (my env export bug); v2 lacked the registered bf16-exact rows; v3 is the receipt:
`artifacts/apa_mf2_2/identity/compare_20261007T231830Z.json:390-392` `"G2a_mf2_2_n1_finite_and_correct": true`,
`"G2b_identity_vs_mf2_and_fd32b6d": true`, `"PN1_n1_bytes_equal_mf2_and_fd32b6d": true`.

- identity census (finite non-overflow, 882 keys): MF2.2 == fd32b6d 882/882, == MF2 882/882, == MF2.1 882/882.
- N1 (522 rows: 18 witnesses, 432 sweep, 72 bf16-exact via fp32): MF2.2 522/522 finite, max abs err 1.0e-6 (fp32) / 4.9e-4
  (fp16); bytes == MF2 and fd32b6d 522/522; MF2.1 0/522 finite.
- Observations: fused F1 rows fd32b6d/MF2 = 0, MF2.1/MF2.2 = 1; fully blocked rows fd32b6d/MF2 = 0, MF2.1/MF2.2 = NaN (policy
  unchanged from MF2.1, out of scope, open item); bf16 tensors are rejected by both kernels on every engine.
- My tables v1 bug: `glob arm_mf2_*` also matched `arm_mf2_2_*`, so the *table's* MF2 column showed the MF2.2 arm. The compare
  JSON used exact stamped names and was correct; tables re-rendered from stamped names with an engine-sha assertion.

## 8. G3 — N2 range sweep and real captures (19:00–19:05)

`scripts/apa_mf2_2_n2.{py,sh}`, MF2.1 and MF2.2 in separate processes. `logs/apa_mf2_2_n2_sweep.log:2` MF2.1
"isolated_out_of_band": 84888, "full_api_out_of_band": 16073, "witness_out_of_band": 6; `:4` MF2.2 all 0, zero and constant
rows OK. Real (36 cells, 1,849,617,408 visible entries): `artifacts/apa_mf2_2/tables.md:64-65` — MF2.1 102 in-band / 0
out-of-band vs the MF2.1 GEMM oracle (reproduces the MF2.1 receipt), MF2.2 87 in-band / 0 out; vs the fp32-bulk oracle 80 / 59
in-band, 0 out. 21/36 real cells have byte-equal selections MF2.1 vs MF2.2.

## 9. G4 — adjudication (19:06–19:09)

Probe copy `scripts/apa_mf2_2_probe/` (originals pinned `SHA256SUMS.orig_mf21rev_04b3e02`; edits
`logs/apa_mf2_2_probe_copy_edits.diff`): receipt/pin paths, an engine/package map incl. `mf2_2`, two default-OFF
adjudication switches (`APA_MF2_2_BF16_LEAF`, `APA_MF2_2_SKIP_BF16_POOL`), two observation cells (`masks_geometry_observe`,
`n2_full_api_before`). No check or tolerance of the original cells changed. Launcher `run.sh` (one process per item).

| Red rows (lead log) | MF2.1 rerun | MF2 | fd32b6d | MF2.2 | Class | Fix |
|---|---|---|---|---|---|---|
| fused_overflow N1 ×18 | RED (NaN) | PASS | PASS | PASS | new product regression (MF2.1 item 2) | item 1 |
| definition N2 ×6 | RED | n/a (no hook); full-API variant RED 5/6 | full-API RED 5/6 | PASS (+ full-API 6/6) | pre-existing product range defect | item 2 |
| sentinel bf16 grad ×1, reductions bf16 ×10, training bf16 ×6 | RED 'NoneType' / grad None | same error (reductions, training) | same | same | probe defect on a pre-existing API limitation (`tc.tensor(dtype='bfloat16', requires_grad=True)` = non-leaf) | not in scope; real bf16 leaf → PASS |
| masks geometry/2048x2048 | RED | RED | RED | RED | probe tolerance (abs 2e-4 at 1023.5); output byte-identical on all four (sha 49c33772…), rel 2.03e-6 | none |
| identity_mf2 exit 1 | exit 1 | arm exit 1 | arm exit 1 | arm exit 1 | probe defect ×2 (bf16 `.grad` None; latent bf16 `max_pool2d`) | corrected: 361/361 equal |

Receipts: `logs/apa_mf2_2_probe_summary_{mf2_2,before,leaf,identity_orig,identity_leaf}.log`, per-run logs
`logs/apa_mf2_2_probe_*`, JSON in `artifacts/apa_mf2_2/probe/`. Full per-row table: `artifacts/apa_mf2_2/tables.md` §G4.

## 10. N5 erratum (19:09)

`scripts/apa_mf2_2_n5_recount.py` → `artifacts/apa_mf2_2/n5_recount.json`: before vs after 529 keys; 526 equal; 3 differ (the
3 design keys); 419 finite on both arms (all equal); 110 non-finite on an arm (107 ±inf reduce rows + 3 design rows). The
per-class list in the MF2.1 sentence sums to 526. "481/481" (`artifacts/apa_mf2_1/verdict.json:35`,
`docs/APA_MF2_1_LEDGER.md:187`) is withdrawn. MF2.1 G3(c)'s universal "no non-finite … with a visible finite score" is withdrawn
(N1 rows are NaN on the MF2.1 engine).

## 11. Mutation (19:10–19:15)

`scripts/apa_mf2_2_mutation.py`; scratch mutants `/tmp/claude-1000/apa_mf2_2_mut/` (tree never edited); baseline
`logs/apa_mf2_2_mutation_baseline.log` "511 passed". Kernel mutants built with matching kernels.cu hashes
(`artifacts/apa_mf2_2/mutation/mutants.json`).

| Mutant | Result | First failing test |
|---|---|---|
| K1/K2 unguarded (fused/int4) | KILLED | review N1 witness |
| K3/K4 start −3.0e38f (F1 re-opened) | KILLED | fused F1 row −3.3e38 |
| K5/K6 guard on old max | KILLED | −inf-first sweep fp32 S5 key0 |
| P1 unscaled, P3 unscaled compare, P6 dev·vis dropped | KILLED | range sweep 1e-38 |
| P2 no zero guard | KILLED | all-zero rows |
| P4 per-head max | KILLED | mixed-magnitude rows |
| P7 `>` | KILLED | constant rows |
| P5 one-pass var (scaled) | **SURVIVED** | — |

Registered result `artifacts/apa_mf2_2/mutation/summary.json:2-5` killed 12 / non-error 13, rate 0.923 → PASS. P5 named as a
coverage gap: the "nearconst" spread (64·2^-20) sits inside the band. Added `test_near_tie_spread_centered_moments` (spacing
2^-17/2^-18, chosen by a probe of P5 vs MF2.2: P5 up to 2,272 out-of-band, MF2.2 0) → `artifacts/apa_mf2_2/mutation/P5_v2.json`
killed (13/13 follow-up). G1 re-run after this addition (§6).

Prior art: DeMillo, Lipton & Sayward (1978) — unverified, lead to check.

## 12. Deviations, scratch, open items

Deviations: registration timestamp text; tests-first v1 oracle bug; identity driver v1 env bug and v2 missing bf16-exact rows;
tables v1 glob bug; test added after the registered mutation run; probe copy edits; ppm side effect. All in `verdict.json`.

Scratch (all `/tmp`, exempt): `/tmp/claude-1000/apa_mf2_2_fd32b6d_src`, `/tmp/claude-1000/apa_mf2_2_mut` (mutant sources,
6 builds, P5 probe), `/tmp/claude-1000/apa_mf2_2_patchchk`.

Open items for David: fully blocked rows NaN in the two fused kernels (production 0); bf16 leaf API in `tensor_cuda/__init__.py`;
reviewer probe tolerance/bf16 defects; unmeasured z ≤ 0, Q-tile fp16 kernel with overflowing scores, ±inf/NaN scores; MF2.1
carry-overs (BK4 'h'/'a' statistics, −1e30 fused APA starts, argmax/topk starts, quantizer |k|² overflow, CUDA 13 extension).

## 13. Prior art (all sites)

- FlashAttention-2 (Dao 2023) `Check_inf` guarded online-softmax rescale; Milakov & Gimelshein (2018) online normalizer —
  kernels.cu item 1 comments, `test_apa_mf2_2_n1.py`. Unverified — lead to check.
- Blue (1978) scaled Euclidean norm; reference BLAS nrm2 scale/ssq; Chan, Golub & LeVeque (1983) centered two-pass variance —
  quant.py item 2 docstring, `test_apa_mf2_2_n2.py`, `apa_mf2_2_oracle.py`, `apa_mf2_2_n2.py`. Unverified — lead to check.
- McKeeman (1998) differential testing; SHA-256 (FIPS 180-4) content pinning — identity harness. Unverified.
- DeMillo, Lipton & Sayward (1978) mutation testing — mutation harness. Unverified.
- flock(1)/timeout(1) — launchers.
- Mine: the −inf-first sweep and its operand constructions, the range sweep patterns, mixed-magnitude/zero/near-tie-spread
  witnesses, the four-engine harnesses, the adjudication switches.
