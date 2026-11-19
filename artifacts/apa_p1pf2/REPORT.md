# APA-P1-PF2 report (the seat's final message, filed verbatim by the lead; the harness refused the seat's own write)

# APA-P1-PF2 report (r1, r2 and resume under Amendments 1 and 2): ACCEPT by the registered rule — G1 to G8 all pass

Every gate passes, which makes the round ACCEPT under Amendment 1. Next are your on-disk check and the Codex review. One thing to weigh first: the informative speed line shows packed prefill 1.12 to 1.55 times slower than dense SP prefill, and both are 44 to 77 times the flash reference on this card.

- **Total GPU:** 2.52 h of the 5 h cap. Every line finished by 03:05, before the 04:45 GPU-sharing cutoff, so that note changed nothing.
- **Harness refusal:** the harness refused my write of `REPORT.md` in r1, so this message is the report; please file it verbatim.
- **Paths:** all relative paths below are under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2/artifacts/apa_p1pf2/`.

## Gates
- **G1, identity: PASS** (PF-I3 under Amendment 1).
  - PF-I1 PASS, PF-I1n PASS, PF-I2 PASS (r1).
  - PF-I3 PASS under A1.1, with τ set per dtype and no cap.
  - A1.2 holds: m12 fails PF-I3 on 132 of 132 rows.
  - r1's verdict under the original 1e-5 cap stays FAIL, as registered.
- **G2, attention quality: PASS.** All four layers pass PF-A1 and PF-A2.
- **G3, model quality: PASS.** Both models, both populations, all three clauses.
- **G4, refusals: PASS.** 26 of 26 refusals and 3 of 3 acceptance rows.
- **G5, memory: PASS.** 24 of 24 cases within the bound.
- **G6, regression: PASS.** LF1, P1-D L2/L4, LF6, SASS and the pins all pass.
- **G7, mutants: PASS.** 16 of 16 killed, 16 of 16 controls clean.
- **G8, scope: PASS.**
  - Writes stayed in the writable set.
  - Only the three edits Amendment 2 names were made after r2.
  - GPU use was only through `forge-gpu-run`, each line after a dry run that returned 78.
  - No git writes. Nothing deleted: three failed or stale files were renamed aside, not removed.

## Diff summary
Product, unchanged since r1: 4 files, +324/−11.
- **New `tensor_cuda/src/apa_spo/packed_prefill.cuh`:**
  - The kernel `spq_pf::prefill<T,CAP,DIAG,SEQ>` is the dense prefill kernel's source with one substitution: each key's bulk score comes from its 4-bit codes, using packed decode's per-pair arithmetic.
  - The query row is rotated in the kernel. The carry is the whole-row running maximum.
  - The host entry raises every refusal before any allocation or launch. A bridge feeds the binding.
- **`packed_launcher.cuh`:** one include line (+6).
- **`bindings.cpp`:** the `packed_prefill` binding in `_tensor_cuda_spo2` (+19).
- **`apa_packed.py`:** the L > 1 branch replaces R-E's refusal, with route record backend `spq_packed_prefill_tq4_row` and the pinned delta; the docstring is updated (+44/−11).

## Module sha256s and SASS
- **Builds:**
  - Build F3: `2e557a23eaf228512a2f1a2f0629401d91d02d05705087661be5c48849c0f3e9`
  - Build PF: `c3c27f8651d82390aac89d62ef704e20dade16d38e9ac8f6cf0e3fb456f4f831`
  - Build PFH (hooks): `e54b39c355ff584ed1f20b27f410819641c933fbf1b4c1c6a35e4dbe6dd0d4ee`
- **Mutant builds** (incremental, one object rebuilt each, base Build PF; m13 on Build PFH):

  | Mutant | Module sha256 |
  |---|---|
  | m01 | `b081a6d3bdb35b8f13ad843d3daf849a19da8586743a6aa082069a9a42133a92` |
  | m02 | `807fbbf03861c18a0f232a8c1cfc6878262dfcd07d0c301335ea0d599313ee17` |
  | m04 | `ca6c5e8d572ca16de2ec6b279167253d1e95547bb3464bf45cb4dc485cf64589` |
  | m05 | `00d9d6a05813a9501e7578c443e6ede1f9336f17da11049119cd020f7e8b4fb7` |
  | m06 | `382e0461ff1bc3e2946e45eaa5186c97e831271d7a32af065fe81390c8fa12eb` |
  | m07 | `ad9b884243144483e4eb8ec74f35d0381181030a56635a3b08598930070b3164` |
  | m08 | `e400683d306ae73114a97232eb9181877baccdf58c45c1e406649b93a5a7cc23` |
  | m09 | `c99e48ae5ca834aa57a270166d0b46a3e9f1cc52cca223b72b1eaaa9274ddc55` |
  | m10 | `30bc7486d5b57b4e1a0d0ab114bac501b3b5435e0a3903011ddb92d28df661f3` |
  | m11 | `839d1f9e1d1fdbf2d4553c1b82d084c8b57ee9dcb1b732292d0fac092bdce55f` |
  | m12 | `c72c61230b78e1198694618e1ff4481d892ef21e4171f263620933118e35dfd6` |
  | m13 | `bbe65b626278299c83b616918adef14c850698b604eaebd4bb489e952e4a0278` |
  | m14 | `539b46f8c16331662758843d15b313ee5491b0c9c333991ee5d84033b582f89b` |
  | m16 | `9deeac9bcc032043ca9e1fcd806f92b1c23939b5291bc3508fb3511c7318c101` |
  | m03, m15 | Python mutants on Build PF's module |

- **SASS: PASS** (`SASS_COMPARISON_PF.json`, sha256 `781b2a3a…`).
  - 813 kernels: 781 equal, 0 changed, 0 removed, 32 added. The 32 added are exactly the registered `spq_pf::prefill` instantiations.
  - Mappings, applied in names and bodies:
    - path-derived names: `<letters><digits>_GLOBAL__N__<8hex>_`, `_GLOBAL__N__<8hex>_`, `__nv_static_<digits>__<8hex>_`;
    - the build-directory suffix `_cu_<8hex>[_<n>]`;
    - mapped keys asserted unique.
  - A fresh F3 dump equals F3's own dump.
- **Registrations:**
  - `EXPECTED.json` sha256 `ef2aa16a…`, written once, before any PF build.
  - `REGISTRATION.json` sha256 `3a201384…`.
  - `mutants/MUTANTS.json` sha256 `4771bd3a…`.

## G1's per-case counts
- **Grid:** 4,848 cases, 1,398,320 oracle rows, 21,912,774,758 decisions. Masks unequal: 0. Bulk bits unequal on Build PFH: 0. Nonzero mask tails: 0.
- **Extras:** exact-tie rows 32 of 32. Store-prefix premise holds.
- **PF-I2:** 4,848 of 4,848 cases bitwise, 3,749,744,640 elements.
- **PF-I1n:** 192 of 192 cases; 128 stored norms NaN; 7,032 non-finite rows on each side, the same row sets.
- **PF-I3 under A1.1** (`a1/PF_I3_A1.json`, sha256 `b83f8bf2…`):

  | dtype | τ | Worst row at the registered delta | Rows above τ | Rows |
  |---|---:|---:|---:|---:|
  | fp32 | 1.564e-4 | 4.11e-5 | 0 | 699,162 |
  | bf16 | 1.351e-2 | 3.17e-3 | 0 | 699,158 |

  A1.2: m12 fails on 132 of 132 rows (worst 3.96e-2); the Build PF control has 0 rows above τ (worst 2.9e-6).
- **Per class** (oracle rows / decisions; 0 unequal in every class):

  | Class | fp32 | bf16 |
  |---|---|---|
  | grid D64 | 329,406 / 5,280,751,968 | 329,401 / 5,279,632,534 |
  | grid D128 | 329,404 / 5,281,100,158 | 329,405 / 5,279,640,778 |
  | D64/VD128 | 38,016 / 393,491,400 | 38,016 / 393,491,400 |
  | D128/VD256 | 1,168 / 1,166,630 | 1,168 / 1,166,630 |
  | D128/VD512 | 1,168 / 1,166,630 | 1,168 / 1,166,630 |

## G2's per-layer table
`gpu/G2/RESULT.json`, sha256 `1ab3a3d0…`.

| Layer | Flips / N (bar) | Rate | Refined (dense, packed) | Pooled d / e_SP (≤ 0.02) | Pooled e_packed / e_SP (≤ 1.01) | Rows with d > e_SP (bar) | Floor rows (violations) | Row ratio median / p99 / max | Device / emulated flips |
|---|---|---:|---|---:|---:|---|---|---|---|
| 1.7B L11 | 37 / 536,936,448 (536) | 6.9e-8 | 5.41%, 5.41% | 6.10e-4 | 1.000004 | 0 (13) | 10,018 (0) | 2.0e-5 / 4.2e-4 / 0.938 | 37 / 63 |
| 1.7B L22 | 18 / 536,936,448 (536) | 3.4e-8 | 2.78%, 2.78% | 3.48e-4 | 1.000001 | 0 (13) | 222 (0) | 1.1e-5 / 6.8e-5 / 0.089 | 18 / 22 |
| 4B L14 | 81 / 1,073,872,896 (1,073) | 7.5e-8 | 4.73%, 4.73% | 1.75e-3 | 1.000000 | 0 (26) | 2,411 (0) | 2.2e-5 / 6.5e-4 / 0.468 | 81 / 91 |
| 4B L28 | 21 / 1,073,872,896 (1,073) | 2.0e-8 | 1.74%, 1.74% | 8.55e-4 | 1.000001 | 0 (26) | 435 (0) | 1.3e-5 / 8.3e-5 / 0.249 | 21 / 165 |

On every layer the packed route record carries the pinned delta, all outputs and masks are informative, and both masks have 0 entries above the diagonal.

## G3's per-model, per-population table
`gpu/G3/RESULT.json`, sha256 `5060a474…`.

| Model | Population | N | Mean KL dense / packed (bound) | Top-1 dense / packed (slack) | dNLL packed − dense, 95% CI | Verdict |
|---|---|---:|---|---|---|---|
| Qwen3-1.7B | prefill | 10,240 | 1.5998e-3 / 1.6332e-3 (1.7798e-3) | 10,071 / 10,063 (30) | +0.00074 [−0.00046, +0.00184] | PASS |
| Qwen3-1.7B | decode | 2,560 | 1.2631e-3 / 1.2872e-3 (1.4094e-3) | 2,509 / 2,510 (7) | −0.00040 [−0.00197, +0.00112] | PASS |
| Qwen3-4B | prefill | 10,240 | 8.7354e-4 / 8.9052e-4 (9.8090e-4) | 10,134 / 10,136 (30) | +0.00101 [−0.00014, +0.00212] | PASS |
| Qwen3-4B | decode | 2,560 | 8.4547e-4 / 8.3988e-4 (9.5002e-4) | 2,532 / 2,532 (7) | −0.00157 [−0.00404, +0.00065] | PASS |

- **Routes:** every SP prefill call took the intended route (108 per 1.7B window, 140 per 4B window), and so did every decode call (layer 0 FLASH; 69,120 and 89,600 SP decode calls per arm). Packed delta mismatches: 0.
- **Timing (L42):** the timed window took 174 s, projecting 2.71 h total against the 5 h cap (`G3_PROJECTION.json`). Measured windows: 1.7B 169–176 s; 4B 220–281 s.
- **Top-1 boundary self-test (L104):** an arm one event past the slack fails; one exactly at the slack passes.

## G4's refusal rows
`gpu/G4_r3/RESULT.json` sha256 `c3c6194b…`; `BOUNDARY.json` sha256 `16935ab9…`.
- **Evidence form:** Python refusals show 0 spied calls. Native refusals show 1 spied call, with 0 kernel launches and 0 allocations during the call (LD_PRELOAD counter).
- **Python refusals (0 spied calls):** TC_APA_SP unset; dense table on packed keys; unpinned packed table; query rank three; bulk_bits mismatch; check_source mismatch; q, v and sinks each on another device; D not 64/128; bf16 query on fp32 packed keys; non-finite scale; non-positive scale; epsilon out of domain; non-finite epsilon; visible rows ≠ k rows.
- **Native refusals (1 spied call, 0 launches, 0 allocations):** causal with S < L; VD above the cap; mixed dtypes; sinks of the wrong shape; and on the raw entry: fmt int4, negative delta, non-finite delta, dump in the default build, launch range B·H·L > INT_MAX (fake extents), fp16 K.
- **Acceptance rows** (the one intended change of P1-D L2, `RE_prefill_on_packed_refused`, is now these):
  - prefill on packed keys runs;
  - the route record carries the prefill backend with the pinned packed delta and e_q;
  - the public route's output and mask equal the raw entry's bitwise.
- **BOUNDARY.json** records Build F3 on the same inputs:
  - the same refusal where the check predates this round;
  - R-E's `prefill (L > 1) on packed keys is not supported` for causal S < L, VD above the cap, mixed dtypes and wrong sinks shape;
  - no raw entry at all: F3 has no `packed_prefill`, so each raw row raises TypeError (`'NoneType' object is not callable`).
- **Caller census (L90, `CALLER_CENSUS.json`):** 1 product caller, `apa_sp2.apa_selective_attention_sp`, which reaches the L > 1 route. 26 test and 91 script sites are listed only; none changed.

## G5's memory figures
`gpu/G5_r2/RESULT.json`, sha256 `e5104b34…`. High-water above resident memory during one packed prefill call (bytes). The bound is out + mask + 2 MiB. Every row is identical with allocator pooling off and on, and every value equals out (+ mask) exactly: 1 allocation without the mask, 2 with it. No key-shaped transient anywhere; 24 of 24 within the bound.

| Geometry | S, L | Without mask | With mask |
|---|---|---:|---:|
| 1.7B (H16) | 2048, 2048 | 16,777,216 | 83,886,080 |
| 1.7B | 8192, 8192 | 67,108,864 | 1,140,850,688 |
| 1.7B | 32768, 2048 | 16,777,216 | 1,090,519,040 |
| 4B (H32) | 2048, 2048 | 33,554,432 | 167,772,160 |
| 4B | 8192, 8192 | 134,217,728 | 2,281,701,376 |
| 4B | 32768, 2048 | 33,554,432 | 2,181,038,080 |

## G6, regression against Build F3
- **LF1 (P1's finite identity):** 6,922 of 6,922 arrays bitwise equal, all informative, key sets equal.
- **P1-D L2:** Build F3 30 of 30, Build PF 29 of 30. The only differing row is the intended change, `RE_prefill_on_packed_refused`.
- **P1-D L4:** 68 B per key per KV head on both builds; every check passes in all 6 cases per build.
- **LF6:** 2,078 IDs (1,376 + 702), 0 differences, same collections, module shas checked.
- **SASS:** as above.
- **Pins** (`PINS.json`):
  - The packed table, the margin registration, the dense table and the three pinned constants are unchanged.
  - `git diff 2945c3e9` over the dense SP goldens and their sources is empty.

## Mutant table
`MUTANT_RESULTS.json`, sha256 `105e5466…`. Each mutant was run on its own build with the named subset, and on a clean control (Build PF; Build PFH for m13).

| # | Mutant | Expected failing assertion | Assertions that failed on the mutant | Killed | Control clean |
|---|---|---|---|---|---|
| 1 | running maximum restarts every 2,048 keys | pf_i1_mask_equal | mask, pf_i3 | yes | yes |
| 2 | `>` for `>=` | pf_i1_ties | ties | yes | yes |
| 3 | dense-table delta (Python) | g4_route_record_packed_delta | route record | yes | yes |
| 4 | causal bound S−L+i | pf_i1_mask_equal | mask, pf_i2, pf_i3 | yes | yes |
| 5 | KV head as h % KVH | pf_i1_mask_equal | mask, pf_i2, pf_i3 | yes | yes |
| 6 | norm multiply dropped | pf_i1_mask_equal | mask, pf_i3 | yes | yes |
| 7 | nibble order swapped | pf_i1_mask_equal | mask, pf_i3 | yes | yes |
| 8 | rotation indexed R[i,d] | pf_i1_mask_equal | mask, pf_i3 | yes | yes |
| 9 | codes addressed with an S stride | pf_i1_mask_equal | mask, pf_i3 | yes | yes |
| 10 | refined key keeps its bulk score | pf_i2_equal | pf_i2, pf_i3 | yes | yes |
| 11 | exact score from codes, not k | pf_i2_equal | pf_i2, pf_i3 | yes | yes |
| 12 | unrefined key dropped | pf_i3_rows_within_tau | pf_i3 | yes | yes |
| 13 | rotation reduction order (Build PFH) | pf_i1_bulk_equal | bulk bits | yes | yes |
| 14 | sinks omitted | pf_i2_equal | pf_i2, pf_i3 | yes | yes |
| 15 | silent fallback to reconstruction plus dense kernel (Python) | g5_no_key_shaped_transient | G5 census | yes | yes |
| 16 | NaN norm treated as a skipped key | pf_i1n_ok | pf_i1n | yes | yes |

On m15 the high-water exceeds out by a key-sized block: 25,755,648 B against an 18,874,368 B bound at S 2048.

## Speed line
Informative, not gated (`gpu/SPEED/RESULT.json`, sha256 `6329c6e6…`). Median of 5 calls after 2 warm-ups, fp32 SP routes against bf16 flash, on an RTX 4070 SUPER (12 GB).

| Geometry | S, L | Packed ms | Dense ms | Flash ms | Packed × flash | Dense × flash | Packed / dense |
|---|---|---:|---:|---:|---:|---:|---:|
| H16 / KV8 | 8192, 8192 | 269.35 | 175.92 | 3.99 | 67.4 | 44.0 | 1.53 |
| H16 / KV8 | 32768, 2048 | 620.04 | 554.64 | 8.02 | 77.3 | 69.1 | 1.12 |
| H32 / KV8 | 8192, 8192 | 541.31 | 349.78 | 7.80 | 69.4 | 44.8 | 1.55 |
| H32 / KV8 | 32768, 2048 | 1102.11 | 975.60 | 15.53 | 71.0 | 62.8 | 1.13 |

- **Against dense SP prefill:** packed is slower, inside the design's 0.85–1.6× estimate but at its slow end (reasoning: per-key nibble decode and table lookups in a one-warp-per-row kernel).
- **Against flash:** both SP prefill routes are 44–77× slower. The one-warp-per-row structure that route (a-S) inherits from dense prefill is the bottleneck, not the packed read (reasoning).
- Grouped reuse for G in {4, 8} was out of scope.

## Each GPU line's result
Receipts are in `gpu/`, each with `.start`, `.log`, `.exit` and a dry run that returned 78. No line exited 124 or 76.

| Line | Exit | Wall | What it did |
|---|---|---:|---|
| G1a_r1 | 0 | 249 s | G1 part complete |
| G1b_r1 | 1 | 450 s | out of memory, repaired |
| G1b_r2 | 0 | 1,471 s | G1 part complete |
| G1c_r1 | 0 | 416 s | G1 part complete |
| G1Ha_r1 | 0 | 1,110 s | hooks bulk bits complete |
| G7m12_r1 | 0 | 5 s | A1.2 holds |
| G7a_r1 | 0 | 48 s | 8 of 8 killed |
| G7b_r1 | 0 | 62 s | m15 crashed in the shim |
| G7c_r1 | 0 | 21 s | m15 rerun, killed |
| G5_r1 | 0 | 5 s | shim crash |
| G5_r2 | 0 | 23 s | PASS |
| G4_r1 | 0 | 2 s | name-shadow fault |
| G4_r2 | 1 | 2 s | RESULT PASS; BOUNDARY.json collided with a stale file |
| G4_r3 | 0 | 2 s | PASS |
| G6LF1_r1 | 0 | 14 s | PASS |
| G6L2L4_r1 | 0 | 19 s | PASS |
| G6LF6a_r1 | 0 | 306 s | PASS |
| G6LF6b_r1 | 0 | 526 s | PASS |
| G2_r1 | 1 | 19 s | 1.7B units saved; 4B ran out of memory |
| G2_r2 | 0 | 15 s | 4B units; PASS |
| G3t_r1 | 1 | — | hook not registered |
| G3t_r2 | 0 | 194 s | timing window |
| G3a_r1 | 0 | 1,562 s | 1.7B windows complete |
| G3b_r1 | 0 | 1,571 s | 4B windows |
| G3c_r1 | 0 | 915 s | 4B windows complete |
| SPEED_r1 | 0 | 35 s | done |

Total: 2.52 GPU hours.

## Deviations
1. **Amendment 1 (lead):** PF-I3's τ is per dtype with no cap. r1's FAIL under the 1e-5 cap stands for r1.
2. **L56b repairs, 3 used:**
   - r1: the SASS dump's missing TMPDIR, and G1's out-of-memory.
   - r2: the allocation shim resolved CUDA symbols through RTLD_NEXT, which crashed; it now resolves them through the loaded libcudart, as F1's shim does.
3. **Amendment 2 (lead):** authorized exactly three edits outside the count, and I made exactly those three, nothing else:
   - G4: rename the local `fake` to `fake_table`.
   - G2: count the mask tails head by head.
   - G3: call `H.register()` in setup.
   - The fresh allowance of 1 was not used.
4. **Own-tool slip (L56a):** G4_r2 computed and saved a PASS RESULT, then its BOUNDARY.json write collided with r1's stale empty file. I renamed that file to `BOUNDARY.G4_r1_empty.json` and reran (G4_r3) with no code change.
5. **Edits made before any run, from static review (no exception had occurred):**
   - G4's raw rows go through the spy, with their operands built outside each row.
   - The F3 child of G4 records Build F3's refusal of L > 1.
   - A fault handler was added to the shim re-exec.
6. **Other script edits:** the m13 mutant is built on Build PFH; the `MUTANTS.json` path in the GPU script is fixed.
7. **G7 merge:** m15's result comes from G7c, because its G5 children crashed in G7b's shim. The merged file records this.
8. **Park:** I stopped my own mutant build driver at the 16:30 usage stop and resumed it at m07 after the reset.
9. **Registered case choices:**
   - fp16 rows are refusals, not grid cases, because the packed store refuses fp16 K.
   - Two VD sub-grids were added to the G1 grid.
   - Tie rows use threshold ties built by an fp32 delta search.
   - The hooks bulk check runs each oracle row as an L = 1 causal prefill where the full bulk dump would exceed 2 GiB.
   - LF6 runs both builds on this tree's tests, which equal 2945c3e9's (`diff -rq`).
   - G4's native-refusal evidence is 0 launches and 0 allocations.
10. **GPU sharing (04:45 rule):** no effect; the last line ended at 03:05.
11. **Own-tool waits:** several foreground waits ran up to about 590 s on lines that had already finished.

## Every file written
All under the worktree root `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1pf2`.
- **Product:**
  - `tensor_cuda/src/apa_spo/packed_prefill.cuh` (new)
  - `tensor_cuda/src/apa_spo/packed_launcher.cuh`
  - `tensor_cuda/src/bindings.cpp`
  - `tensor_cuda/tensor_cuda/apa_packed.py`
- **Scripts** (run as `python3 -B -P`; no name clashes with the standard library):
  - `scripts/apa_p1pf2_build.py`
  - `scripts/apa_p1pf2_gpu.py`
  - `scripts/apa_p1pf2_line.py`
  - `scripts/apa_p1pf2_sass.py`
  - `scripts/apa_p1pf2_mutants.py`
  - `scripts/apa_p1pf2_a11.py`
  - `scripts/apa_p1pf2_g2.py`
  - `scripts/apa_p1pf2_q1.py`
  - `scripts/apa_p1pf2_g45.py`
  - `scripts/apa_p1pf2_g6.py`
  - `scripts/apa_p1pf2_mut.py`
  - `scripts/apa_p1pf2_speed.py`
- **No test file was written** (`tensor_cuda/tests/test_apa_p1pf2_*.py` does not exist).
- **`artifacts/apa_p1pf2/`:**
  - Registrations and inputs: `INSTRUMENTS.log`, `EXPECTED.json` (+ `.sha256`), `REGISTRATION.json` (+ `.sha256`), `mutants/MUTANTS.json`.
  - Builds and arms: `build_PF/`, `build_PFH/`, `arms/{PF,PFH}/`, `mutants/<m>/` (tree, build, arm), `mutants/BUILD_DRIVER*.log`.
  - SASS: `sass/{F3,PF}.sass`, `SASS_COMPARISON_PF.json`.
  - Gate results: `a1/PF_I3_A1.json`, `MUTANT_RESULTS.json`, `BOUNDARY.json`, `CALLER_CENSUS.json`, `PINS.json`, `G3_PROJECTION.json`.
  - State notes: `PARK_STATE.md`, `STATE_r2.md`.
  - `gpu/` directories: `G1`, `G1H`, `G7`, `G7a`, `G7b`, `G7c`, `G5`, `G5_r2`, `G4`, `G4_r2`, `G4_r3`, `G6_LF1`, `G6_L2L4`, `G6_LF6a`, `G6_LF6b`, `G2`, `G3`, `SPEED`, plus each line's receipts.
  - `logs/`.
  - Renamed aside (left, not deleted): `sass/F3.sass.failed_r1_no_tmpdir`, `BOUNDARY.G4_r1_empty.json`, `/tmp/apap1pf2_shim/libpf2count.so.r1_segv`.
- **Scratch left in place:** `/tmp/apap1pf2_{main,build,sass,gpu,tmp,shim,tree0,l6}`.

## Prior art
- **TurboQuant** (Zandieh et al., 2025, arXiv 2504.19874): the rotated 4-bit codes, taken as the stored form.
- **Asymmetric distance computation in product quantization** (Jégou, Douze and Schmid, 2011): scoring a query against codes without reconstructing, taken as the idea; here the query is rotated rather than scored through a per-subspace table.
- **BLASST** (Yuan et al., 2025, arXiv 2512.12087): the running-maximum refine threshold, taken unchanged.
- **Online softmax** (Milakov and Gimelshein, 2018, arXiv 1805.02867) and **FlashAttention** (Dao et al., 2022; FA-2 as the speed reference): the dense prefill kernel's streaming softmax, inherited unchanged.
- **Mutation testing** (DeMillo, Lipton and Sayward, 1978): the 16 mutants.
- **KL divergence** (Kullback and Leibler, 1951) and the **paired percentile bootstrap** (Efron, 1979): P1-Q and AB-1's metrics, imported.
- **LD_PRELOAD interposition** (POSIX `dlsym`/`dlopen`) and F1's launch-count shim: the G4/G5 counter.
- **IEEE 754 round-to-nearest subtraction:** the tie construction.
- **P1-D, F1, F2, F3, P1-Q and AB-1 code:** imported.
- **Ours, no prior art known to me:**
  - packed decode's per-pair bulk arithmetic inside the dense prefill structure, which makes decisions bitwise equal to packed decode;
  - the 32-keys-per-warp form for the sequential D64 order;
  - the threshold-tie construction;
  - using the all-refined regime as a bitwise oracle against the dense kernel.
- **Unverified — lead to check.** Search terms: "attention scores from quantized keys without dequantization rotated query"; "TurboQuant 2504.19874"; "BLASST 2512.12087".

G1 PASS (Amendment 1; A1.2 holds) · G2 PASS · G3 PASS · G4 PASS · G5 PASS · G6 PASS · G7 PASS (16 of 16) · G8 PASS — ACCEPT by the registered rule, pending the lead's on-disk check and the Codex review.
