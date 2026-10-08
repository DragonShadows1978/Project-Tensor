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

---

# Amendment 1 (`orders/APA_MF2_2_AMENDMENT_1.md`, sha256 `fda2447a…5238f`): fully blocked rows, special-row census

Base: apa-mf2 86f0a18 content (kernels.cu `17c89f3c…`, quant.py `3ff09e9b…`, checked). Registration
`artifacts/apa_mf2_2a/registration.json`, sha256 `1e4facbd51dcb8cd02375097c10d78845772db81c947bc4694918ab4a2e03d18`, hashed 19:25:47
before any GPU command (its `registered_at` text says 19:26). Verdict `artifacts/apa_mf2_2a/verdict.json`.

## A1. Pre-fix census (19:26–19:27)

`scripts/apa_mf2_2a_census.{py,sh}`: one process per engine, 2376 cells per engine (paths × {all_blocked, all_masked,
neg_inf_first, single_visible} × fp32/fp16/bf16 × D{64,96,128} × Lq{1,3} × S{1,2,33,1025}; score-input paths at D 64).
Blocked scores are made from finite operands by overflow (attention paths) or literal −inf (softmax paths).
`logs/apa_mf2_2a_census_compare_prefix.log`: fd32b6d finite / chain non-finite = 72 (MF2), 468 (MF2.1), 252 (MF2.2).
- MF2.2's 252 = 180 fused-kernel fully blocked cells (144 all_blocked + 36 S=1 neg_inf_first, fp32/fp16; bf16 is rejected by
  both kernels; fd32b6d gives 0 on all) + the 72 below. The 180 fused cells are item 3's target.
- **72 cells, a second class found by the census:** causal paths (`sdpa_infer_causal`, `sdpa_grad_causal`, `apa_quant_attention`
  p1.0/p0.15 causal), single_visible at Lq=3, S ∈ {33,1025}: rows 0–1 see only blocked keys. fd32b6d's finite output equals
  V[S-1], a FUTURE key (checked: `0.9362, -0.6765, -0.5241, 0.4423` == V[32][:4]) — its finite additive mask leaked the masked key
  (F4). NaN since MF2 66287b1 (predicate −inf mask). Outside the two fused kernels; reported, not fixed (registered rail).

## A2. Tests first (19:28)

`tensor_cuda/tests/test_apa_mf2_2a_blocked.py` (fully blocked rows = 0, a blocked row among ordinary rows,
large-magnitude finite rows ±150/±300). `logs/apa_mf2_2a_testsfirst_mf2_2.log:976` "39 failed, 8 passed, 12 skipped" on the
original MF2.2 engine (bc7ed2a9). 4 of the 39 were invalid: int4_partial with literal −inf keys is a NaN row (INT4 packer:
amax inf → NaN codes), NaN on fd32b6d and MF2 as well (`logs/apa_mf2_2a_literal_inf_check.log`); that construction is now
skipped with the receipt cited. v2 (MF2.2 rebuild): `logs/apa_mf2_2a_testsfirst_mf2_2_v2.log:1036` "35 failed, 8 passed, 16 skipped".

## A3. Item 3 (`artifacts/apa_mf2_2a/item3.patch`, 19:29)

Merge only, both kernels: `const float g_ref = gmax == -INFINITY ? 0.f : gmax; const float rescale = expf(m - g_ref);`
(`kernels.cu:5131-5132`, `:6263-6264`), non-comment +4 −2. The item-1 comment now points to the merge. The final normalizer was
already guarded (`inv_denom = red[0] > 0.f ? 1.f / red[0] : 0.f`). Start −inf and the N1 guard unchanged.
`logs/apa_mf2_2a_patch.log:2` "ITEM3 PATCH REPRODUCES TREE: True (base 17c89f3c -> d9d9851818f2b410)".
After engine `artifacts/apa_mf2_2a/build` `ccc9f8d2…` (`logs/apa_mf2_2a_build.log`).
`logs/apa_mf2_2a_tests_after_v2.log:4` "146 passed, 16 skipped" (item-3 tests + MF2.2 N1 tests).

Prior art: FlashAttention-2 (Dao 2023) split-KV combine / `Check_inf` guard (max == −inf → 0). Unverified — lead to check
("flash-attention combine lse -inf guard", "softmax.h Check_inf"). Taken: the guard. Mine: its placement in this warp merge.

## A4. SEAT ERROR: the MF2.2 engine binary was overwritten (19:29)

In the same command as the after build I ran a stray `bash scripts/apa_mf2_2_build.sh` with no `APA_MF2_2_BUILD_DIR`, which
rebuilt the default `artifacts/apa_mf2_2/build` from the item-3 sources (module `da897f7e…`), destroying the MF2.2 engine
`bc7ed2a9…`. That directory is outside this amendment's writable list. No copy survived (searched the fork and /tmp). Builds
here are not byte-reproducible (identical item-3 sources gave `da897f7e` and `ccc9f8d2`), so `bc7ed2a9` cannot be recreated.
- Receipts made with `bc7ed2a9` predate it: MF2.2 census arm (mtime 19:27:00), item-3 tests-first (19:28:10), and every MF2.2
  receipt of the main order.
- Remediation: the mis-built dir → `/mnt/ForgeRealm/DELETE_ME/20261007-193132_apa-mf2-2a_build` (delete-me, session apa-mf2-2a;
  manifest copy `logs/apa_mf2_2a_misbuild_manifest.json`); MF2.2 sources rebuilt from a scratch root (18 sources == MF2.1
  manifest, kernels.cu == `17c89f3c`, asserted) into `artifacts/apa_mf2_2/build` → `320932d9…` (`logs/apa_mf2_2a_rebuild_mf2_2.log`).
- Equivalence: the rebuild's census equals the original `bc7ed2a9` census on 2376/2376 cells; G3 compares after against BOTH
  the original's recorded identity arm and a fresh rebuild arm.
- `artifacts/apa_mf2_2/verdict.json` still names `bc7ed2a9` for that path; not edited (outside the writable list) — lead to annotate.

## A5. Census with after; G2 (19:33)

`logs/apa_mf2_2a_census_compare.log`; `artifacts/apa_mf2_2a/census/compare_20261007T2327Z.json:843-845` "fd32b6d_finite_after_nonfinite":
{"count": 72, …} → **G2 RED**; `:1077-1079` "fd32b6d_nonfinite_after_differs_from_mf2_2": {"count": 0} (forbidden-change check holds);
`:1083` "equal": 216 (fused all_blocked incl. the noncausal SDPA route, after == fd32b6d bytes).
`artifacts/apa_mf2_2a/census_table.md:1-2` "Fused fully-blocked cells (all_blocked + S=1 neg_inf_first): 180; after bytes ==
fd32b6d: 180", "after differs from MF2.2 on any other cell: 0 []". Condensed table: `artifacts/apa_mf2_2a/census_condensed.md`.
The 72 RED cells are exactly the causal class of §A1 (18 per path), unchanged since MF2.

## A6. G3 identity (19:34)

`artifacts/apa_mf2_2a/identity/g3_20261007T233445Z.json:25,30` "equal": 882 / "equal": 522 vs the original MF2.2 arm (bc7ed2a9),
`:35,40` the same vs the rebuild, `:44` "n1_witnesses_exactly_one": true, `:108` "G3_pass": true. Observations on after: F1 fused
rows 1.0, fully blocked overflow rows 0.0 (were NaN).

## A7. G1 (19:35–19:37)

`logs/apa_mf2_2a_g1_compare.log:2` "37 failed, 1694 passed, 46 skipped, 2 warnings, 15 errors", `:3` "baseline ids 52 now ids 52",
`:7` "G1 PASS". Side effect: 17 `artifacts/wo*.ppm` regenerated.

## A8. Mutation (19:38–19:40)

`scripts/apa_mf2_2a_mutation.py`, scratch `/tmp/claude-1000/apa_mf2_2a_mut`, baseline "146 passed, 16 skipped"
(`logs/apa_mf2_2a_mutation_baseline.log`). A1/A2 guard removed, A3/A4 constant reference 0, A5 N1 guard removed, A6 start
−3.0e38f: all KILLED; `artifacts/apa_mf2_2a/mutation/summary.json:5` "rate": 1.0. Prior art: DeMillo, Lipton & Sayward (1978) — unverified.

## A9. Open items for David (amendment)

- G2 RED class: causal rows whose every visible key is blocked — fd32b6d leaked a future key, the chain gives NaN since MF2.
  Choose (a) NaN as the documented policy or (b) 0 (FlashAttention-2 convention): zero the weights of rows whose visible max is
  −inf in `functional.scaled_dot_product_attention` (causal composed + grad) and `quant.apa_quant_attention` — outside this
  amendment's scope. Restoring fd32b6d's value would re-open F4.
- NaN rows where fd32b6d is also NaN are recorded and unchanged (eager/causal softmax, composed SDPA, apa_quant on all −inf rows;
  additive all-masked rows; int4_partial with −inf key components; +inf/NaN scores).

Scratch (all /tmp): `/tmp/claude-1000/apa_mf2_2_rebuild_src`, `/tmp/claude-1000/apa_mf2_2a_mut`, `/tmp/claude-1000/apa_mf2_2a_patchchk`,
`/tmp/claude-1000/apa_mf2_2a_literal_check.py`.

---

# Amendment 2 (`orders/APA_MF2_2_AMENDMENT_2.md`, sha256 `76918bea…874b1`): all -inf rows return 0 on composed paths

Base: apa-mf2 f3cde75 content (kernels.cu `d9d98518…`, functional.py `86df0db5…`, quant.py `3ff09e9b…`). Registration
`artifacts/apa_mf2_2b/registration.json`, sha256 `5e84edd44d7d8e1f70867efb75103c1fa2f1547614a4b94732ab739eb0edcb24`, hashed 19:47:32 before
any GPU command (text says 19:48). Verdict `artifacts/apa_mf2_2b/verdict.json`. **No engine build** in this amendment: after1 and
after2 both run `artifacts/apa_mf2_2a/build` (`ccc9f8d2…`); after1's package is the snapshot `artifacts/apa_mf2_2b/before/pkg`.
No build directory was written.

## B1. Tests first (19:49–19:51)

`tensor_cuda/tests/test_apa_mf2_2b_blocked.py` (56 tests: causal rows with all visible keys blocked on SDPA infer composed /
fused-softmax flag / grad; all-masked attn_mask rows infer/grad; non-causal composed; apa_quant p1/p.15 causal/non-causal and
backward; G4 left-padded batch, two-layer chain, apa_quant overflow padding; partially blocked rows vs oracle; NaN rows untouched).
v1 had 24 ValueErrors from my oracle indexing (`s[..., r, :]` dropped the row axis); fixed to row slices.
`logs/apa_mf2_2b_testsfirst_after1_v2.log:1957` "46 failed, 10 passed" on after1 — every failure "non-finite output" or a
blocked row not 0; the 10 passes are the untouched-row controls.

## B2. Item 4 (`artifacts/apa_mf2_2b/item4.patch`)

`functional.py`: `_blocked_rows` (every entry −inf: `(scores < -FLT_MAX).min(-1)`; NaN never blocked; invisible entries count as
blocked on the causal_softmax branch), `_guard_const`, `_guarded_softmax` (`softmax(where(blocked, 0, scores)) * (1 - blocked)`),
applied at the SDPA fused causal_softmax branch, the composed inference softmax and the composed grad softmax; `quant.py`:
`apa_quant_attention` softmax. The −inf predicate mask is unchanged (F4). Diff: functional.py +58 −5, quant.py +2 −1.

**Guard v1 → v2.** v1 used `Tensor.masked_fill`. `test_apa_mf1_quant.py` observes apa_quant_attention's selection as the LAST
`masked_fill` (stats mask) and LAST `tc.where` (refine) of the call; the guard runs after both, so G1 v1 FAILED with 60 new ids
(`logs/apa_mf2_2b_g1_compare.log:5`). v2 selects with the engine primitive `tc._C.where` (same `where_nd`, same backward —
gradient only where not selected; `logs/apa_mf2_2b_where_check.log`). No test assertion changed. Census v1 vs v2: 0 of 2376 cells
differ. Every gate below is the v2 run.

Prior art: FlashAttention-2 (Dao 2023) returns 0 for fully masked rows (Check_inf / lse guards); PyTorch SDPA returns NaN there.
Unverified — lead to check. Mine: the select-then-zero composition that keeps the backward finite.

## B3. G2 census (19:55, v2 20:00)

`scripts/apa_mf2_2b_census.sh` (Amendment-1 harness unchanged; arms into `artifacts/apa_mf2_2b/census/`),
`scripts/apa_mf2_2b_compare.py`. `artifacts/apa_mf2_2b/census/compare_v2_20261008T0000Z.json:38-39` "G2a_fd32b6d_finite_after2_nonfinite":
{"count": 0 (after1 had 72). Blocked-row cells non-finite on after2: 63 = 57 primitive score-input cells (eager_softmax 30,
causal_softmax 27; NaN on every engine since fd32b6d; ops.cpp / kernels.cu are outside the writable list — predicted in the
registration) + 6 `sdpa_grad_causal` fp16 S=1 cells whose forward is 0 but whose dq carries one NaN: the matmul backward at S=1
with alpha 1e34 gives NaN dq for an exactly zero upstream gradient, without any softmax, on fd32b6d and after2 alike
(`logs/apa_mf2_2b_fp16_alpha_backward_check.log:2` "float16 S=1 alpha=1e+34: dq finite=False"). `logs/apa_mf2_2b_blocked_values_check_v2.log:1`
"v2 attention blocked cells: 999; row-0 output exactly 0 and output finite: 999". **G2 as registered: RED** (those 63 cells);
attention-path forward: all 0. Table: `artifacts/apa_mf2_2b/census_condensed.md`.

## B4. G3 identity (20:00–20:01)

Census non-blocked cells `compare_v2…json:67-68` "differ_vs_recorded": 0, "differ_vs_fresh": 0 (1230 cells); fused set
`logs/apa_mf2_2b_identity_fused_v2.log:2-3` "v2 identity: 882/882 equal", "v2 n1: 522/522 equal"; probe identity_values
(`scripts/apa_mf2_2b_probe/`, edits `logs/apa_mf2_2b_probe_copy_edits.diff`; pins v2 after the guard change, v1 kept as
`artifacts/apa_mf2_2b/probe/provenance_pins_v1_superseded.json`) `logs/apa_mf2_2b_probe_identity_20261007T235709Z_mf2_2b_337027.log:4`
"matched_tensors": 361 → **PASS**.

## B5. G4 left-padded batch

22 G4 tests FAILED on after1 (`logs/apa_mf2_2b_testsfirst_after1_v2.log:1935` "FAILED …test_left_padded_batch[float32-1-infer_composed]");
all pass on after2 (`logs/apa_mf2_2b_tests_after2_v2.log:7` "371 passed, 12 skipped", with test_apa_mf1_quant and
test_apa_mf2_1_definition) → **PASS**.

## B6. G1 (19:57–19:59, v2)

`logs/apa_mf2_2b_g1_compare_v2.log:2-7` "37 failed, 1750 passed, 46 skipped, 2 warnings, 15 errors", "baseline ids 52 now ids 52",
"new failing ids: 0 []", "G1 PASS". 17 `artifacts/wo*.ppm` regenerated again.

## B7. Mutation (20:02)

`scripts/apa_mf2_2b_mutation.py` (scratch packages `/tmp/claude-1000/apa_mf2_2b_mut`), baseline "56 passed". B1–B4 guard removed
per site, B5 zeroing dropped, B6 'any −inf' detection, B7 invisible not counted: 7/7 KILLED, `artifacts/apa_mf2_2b/mutation/summary.json:5`
"rate": 1.0. Prior art: DeMillo, Lipton & Sayward (1978) — unverified.

## B8. Open items

Primitive `Tensor.softmax` / `tc.causal_softmax` on an all −inf row (NaN everywhere since fd32b6d): does the rule cover primitives?
Matmul backward alpha overflow at S=1 (pre-existing). 12 fp16 grad neg_inf_first cells non-finite on every engine (mechanism not
isolated). Blocked rows can be −0.0.

Scratch (/tmp): `/tmp/claude-1000/apa_mf2_2b_mut`, `/tmp/claude-1000/apa_mf2_2b_where_check.py`, `/tmp/claude-1000/apa_mf2_2b_alpha_check.py`.

---

# Amendment 3 (`orders/APA_MF2_2_AMENDMENT_3.md`, sha256 `cf8092cf…25e725`): V7 long-key statistics, last red probes, nits

Base: apa-mf2 a2e3ca1 (+ order aace603): functional.py `71f04556…`, quant.py `2020e5be…`, test_apa_phase6.py `c1e95bb2…`, kernels.cu
`d9d98518…` (checked; before copies `artifacts/apa_mf2_2c/before`). Registration `artifacts/apa_mf2_2c/registration.json`, sha256
`f0363a174099bb6fc4968f9afae5d4db414a8107b3cfb03fe772435fcfa71693`, hashed 20:50:47 before any GPU command (text says ~20:53).
Verdict `artifacts/apa_mf2_2c/verdict.json`. **No build**: a2e3ca1 and after3 run `artifacts/apa_mf2_2a/build` (`ccc9f8d2…`); every
build directory's .so predates this amendment and `artifacts/apa_mf2_2c/build` does not exist.

## C1. Reading and the V7 mechanism

Read the order, `apa-mf22rev` `docs/APA_MF2_2_REVIEW.md`, `artifacts/apa_mf22rev/verdict.json`, its probes and the lead's
`logs/apa_mf22rev_gpu_*_20261008T*.log`. V7 mechanism (kernels.cu:567-582, 652): the row sum is 256 threads × S/256 sequential fp32
adds + an 8-level tree, depth S/256 + 7 (4103 at S = 2^20). The engine has no float64 dtype (`core.h:30`), and kernels.cu is not
writable here, so the lever is the summation ORDER composed from engine ops.

## C2. Tests first

`tensor_cuda/tests/test_apa_mf2_2c_longkeys.py` (review witness exact + public-quantizer operand; sweep S {2^10, 2^14, 2^17, 2^17+3,
2^20} × widths {2^-20, 2^-16, 2^-12, 2^-10, 1} × 3 patterns × causal; full API fp32/bf16; `_row_sum` accuracy/coverage) and
`test_apa_mf2_2c_nits.py`. The order's four widths do not reproduce V7 at all on a2e3ca1 (v1 receipt: only the witness and full-API
fp32 S=2^20 failed); I added the witness width 2^-12 to the sweep. `logs/apa_mf2_2c_testsfirst_a2e3ca1_v2.log:327` "14 failed, 158 passed";
nits on the before copies `logs/apa_mf2_2c_nits_testsfirst_before.log:46` "3 failed".

## C3. Item 5 (`artifacts/apa_mf2_2c/item5.patch`; quant.py +44 −4, functional.py +5 −2, test_apa_phase6.py +17 −11)

- **V7:** `quant._row_sum` — S ≤ 8192: engine single-pass sum (unchanged bytes, depth ≤ 39); S > 8192: 4096-key blocks summed, partials
  summed recursively, the < 4096 remainder summed natively and added once (depth ≤ 32 at S ≤ 2^20; first-order bound depth × 2^-24 ×
  Σ|x|). Both moments use it; the definition is unchanged. Memory at S = 2^20: views + 256 partials per row per moment (1 KiB/row);
  for S not a multiple of 4096, one transient slice copy per moment. Time at S = 2^20, 2 rows: 0.034 ms vs 0.554 ms native.
  Prior art: blocked/pairwise summation error analysis, Higham (1993, SIAM J. Sci. Comput. 14(4); 2002 ch. 4); Chan, Golub &
  LeVeque (1983) on pairwise accumulation for the variance — unverified, lead to check. Mine: the two-level engine-op composition
  and the 8192/4096 split.
- **V6:** the docstring's per-entry "≤ 2^-24 relative" division claim corrected (underflow; row-relative bound).
- **V3:** `_guarded_softmax` docstring now states the measured behaviour: torch 2.11.0 SDPA returns 0 for a fully masked row (additive
  and boolean masks), `torch.softmax` returns NaN (`logs/apa_mf2_2c_v3_torch_check.log:2` "SDPA fully masked additive mask -> first values [0.0, 0.0, 0.0]").
- **V5:** `test_apa_phase6._ref` uses a −inf predicate mask, centered fp64 statistics and 0 weight for all −inf rows; the review witness
  gives [1,0,0] (was [0,0,0]). `logs/apa_mf2_2c_nits_after3.log:2` "8 passed" (nits + the 5 phase6 tests).
- **V4:** `artifacts/apa_mf2_2c/v4_recount.json`: of the 1230 non-blocked Amendment-2 census records, 1092 are finite tensor records
  byte-equal, 12 pre-existing non-finite gradient records, 126 the same unsupported-operand error with no tensor bytes. The corrected
  wording replaces "bytes on all 1230 census cells" in `artifacts/apa_mf2_2b/verdict.json:19` (that file is outside this order's list).

`logs/apa_mf2_2c_tests_after3.log:4` "172 passed"; `logs/apa_mf2_2c_patch.log:2` "ITEM5 PATCH REPRODUCES TREE: True".

## C4. G2 (V7)

Probe copy `scripts/apa_mf2_2c_probe/` (one edit: receipt/pin dir, `logs/apa_mf2_2c_probe_copy_edits.diff`; pins add arm `after3`;
witness copied, sha `ce8cbe6c…` equal). `logs/apa_mf2_2c_probe_summary.log:1-2`: long_keys a2e3ca1 "failed=2", after3 "failed=0".
`logs/apa_mf2_2c_v7_table.log:1-2`: a2e3ca1 163840 out-of-band (isolated; all at S=2^20, w=2^-12, sorted) and 163840 (full API fp32
S=2^20), after3 0 and 0 under both bands. Table `artifacts/apa_mf2_2c/v7_table.txt`. **PASS.**

## C5. G3 (no regression)

N2 sweep: `logs/apa_mf2_2c_n2_sweep.log:1` all 0 out-of-band, every table == the MF2.2 recording. Real: `logs/apa_mf2_2c_real_compare.log:1`
"real cells 36; out-of-band (both oracles) 0; selection bytes == MF2.2 recording: 36/36". Review identity corpus:
`logs/apa_mf2_2c_identity_corpus.log:3` "702 arrays, after3 vs a2e3ca1 byte-different: 0". Census and fused identity:
`logs/apa_mf2_2c_g3_census_identity.log:1-3` (2376/2376, 882/882, 522/522). **PASS** — no precision-only difference to list.

## C6. G4 adjudication

| Red row (lead log) | a2e3ca1 | after3 | Class | Evidence |
|---|---|---|---|---|
| guards/{fp32,fp16,bf16}/NaN_softmax_policy | RED | RED | probe defect | `np.array_equal` of shape (1,1,9) vs (9,) is False for equal values (numpy alone: `logs/apa_mf2_2c_nanpolicy_shape_check.log:1`); guard values correct in all dtypes (`logs/apa_mf2_2c_nanpolicy_diag_a2e3ca1.log`) |
| real/Qwen3-1.7B/L16/bfloat16/L3 (2/384) | RED | RED | pre-existing bf16 precision; probe tolerance | output sha `ccace60b…` identical on fd32b6d / a2e3ca1 / after3; bf16-storage oracle within 0.0156 of the engine, itself 0.132 from fp64 (`artifacts/apa_mf2_2c/real/real_diag_*.json`) |

No in-scope product defect; nothing fixed for G4. **PASS (adjudicated).**

## C7. G1 and mutation

`logs/apa_mf2_2c_g1_compare.log:2-7` "37 failed, 1925 passed, 46 skipped, 2 warnings, 15 errors", "baseline ids 52 now ids 52",
"new failing ids: 0 []", "G1 PASS". Mutation (`scripts/apa_mf2_2c_mutation.py`, scratch `/tmp/claude-1000/apa_mf2_2c_mut`, baseline
"537 passed"): M1 native restored, M2 remainder dropped, M3 mean native, M5 one-pass KILLED; M4 variance-only native SURVIVED —
`artifacts/apa_mf2_2c/mutation/summary.json:5` "rate": 0.8 → PASS at the threshold. M4: a CPU search (90 patterns, S = 2^20) bounds
the variance-only drift at 5.4e-6 relative, below the band (`logs/apa_mf2_2c_m4_search.log:1`): not distinguishable in the measured
domain; named coverage gap. Prior art: DeMillo, Lipton & Sayward (1978) — unverified.

## C8. Open items

V1 (GRAPA attention.py:114; GRM Gemma4/DeepSeek/MiniCPM3 direct primitive softmax) — read-only, reported. M4 gap. Review-probe fixes
(`.ravel()` in the guards cell; bf16-aware tolerance in the real cell). `_row_sum` bound beyond S = 2^20 is reasoning only.
Scratch (/tmp): `/tmp/claude-1000/apa_mf2_2c_mut`, `/tmp/claude-1000/apa_mf2_2c_patchchk`, `/tmp/claude-1000/apa_mf2_2c_before_tests`,
`/tmp/claude-1000/apa_mf2_2c_m4_search.py`. G1 regenerated the 17 `artifacts/wo*.ppm` again.

## Promotion record (lead)

- **2026-10-07 ~22:10 EDT, David's verbatim approval of the promotion ticket:** "I approve the thing".
- **Ticket scope, as presented:**
  - **Promoted ref:** `apa-mf2` at 7813b48 (production fd32b6d plus the MF1–MF2.2 chain).
  - **Not covered:** BK4/'a' kernel statistics; GRAPA/GRM direct primitive softmax; the apa_cuda CUDA 13 build; the
    quantizer overflow; S beyond about 2^20.
- **Review receipt:** `artifacts/apa_mf_promotion/review_receipt.json` (final Codex verdict APPROVE-WITH-NITS, no
  blocking findings).
- **Condition stated to David:** the final review's GPU probes must pass before the merge. They are pending because
  APA-SPM2 holds the GPU.
