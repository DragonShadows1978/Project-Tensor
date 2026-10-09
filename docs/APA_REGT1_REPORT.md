# APA-REGT1 report: three detectors ported to production as regression tests

Order: `orders/APA_REGT1_REGRESSION_TESTS.md` (4c62298). Seat: Claude Opus 5.5 high (Agent-tool subagent),
2026-10-09. The harness refused report files from the subagent, so the lead transcribed this report from the seat's
final message and added the lead's verification at the end.

David, 2026-10-09, verbatim: "On things like regression tests going into production, that's fine unless it really
significantly hurts performance."

## Verdict

| gate | result | evidence |
|---|---|---|
| G1 port | **PASS** | Only the six new files, in authorized paths. "Fork" and "WorkTree" appear in none of them (grep exit 1). With `CUDA_VISIBLE_DEVICES=""`: 20 passed, 11 skipped, 0 errors. |
| G2 main | **PASS** | 31/31 pass on the pinned main engine d8fda9a8. Its build manifest matches this worktree's a28e28e sources: 19/19 files, and the .so matches the pin. Full suite: FAILED/ERROR set equals the 52 baseline ids, with 0 new. |
| G3 teeth | **PASS per detector** | ZWS fails on PYSTAT and CU7. The PDWL cut check and the stagger cuts fail on CU5; the PDWL label check fails on LD1's CU1. IMM, textbook and gradcheck each fail on MU2–MU6, with identical alarm sets. |
| G4 cost | **PASS** | Measured suite overhead: +1.30 s mean over two with/without pairs (+2.46 s and +0.13 s), within noise. The 31 new tests' summed test durations are 0.484 s: 0.089 s for the 20 CPU-only tests and 0.395 s for the 11 GPU tests. That sum is per-test time, not the suite's wall-time delta. Nothing gated or dropped. |

**Engine runtime performance is unaffected:** no engine source changed. The six files are three test modules and the
three script modules they import.

## Files

| file | lines | what it is |
|---|---|---|
| `scripts/apa_ld1_pdwl.py` | 561 | LD1 PDWL detector (lines 1–355 byte-identical to cb2fac1) plus the engine adapter, as a verbatim line subset of `apa_ld1_entries.py` |
| `scripts/apa_ld2_pdwl.py` | 447 | LD2 stagger cuts and ZWS, plus the B04 ghost-block fix, `Attn2` (verbatim subset of `apa_ld2_entries.py`) and `zws_cell` (byte-identical to `apa_ld2_run.py`) |
| `scripts/apa_om135_core.py` | 804 | OM135 IMM, textbook and gradcheck checks (lines 1–765 byte-identical to 2bbee82) plus `prepare()` (byte-identical to `apa_om135_prep.py`) |
| `tensor_cuda/tests/test_apa_ld1_pdwl.py` | 124 | PDWL CPU self-tests; engine SDPA and apa_selective pass on main, and the in-test mutants alarm |
| `tensor_cuda/tests/test_apa_ld2_pdwl.py` | 197 | frontier cuts, stagger cuts, ZWS (numpy and engine), and the B04 square cell |
| `tensor_cuda/tests/test_apa_om135_imm.py` | 171 | IMM reference, exact-integer operands, the three methods passing on main, and a wrong-orientation positive control |

## Port differences from the fork sources

1. **`scripts/apa_ld1_pdwl.py`.** The engine adapter (`up`, `down`, `expand_heads`, `batch_stride_off1`,
   `class Attn`) is appended from `apa_ld1_entries.py` as a verbatim line subset: 164 code lines, 0 non-verbatim.
   - **Removed:**
     - `load_engine`, because production tests use the importable engine;
     - the GRM and GRAPA callers, which had hardcoded external paths;
     - `_synthetic_margins`, which wrote under `artifacts/`;
     - the code the tests do not exercise.
   - **Kept verbatim:** the `P.` lines, through the alias `P = sys.modules[__name__]`.
2. **`scripts/apa_ld2_pdwl.py`: the B04 fix in `zws_cuts`.**
   - When t_g < 1, it uses t_g = (S−L) + L//2 and cuts {t_g, t_g + L//4}. This is the ghost block from LD2's
     exploratory rerun, and the one retune the order allows.
   - Every registered cell with t_g ≥ 1 is unchanged; the test asserts `zws_cuts(3,33) == (30,[30,31])`.
   - `Attn2` and `zws_cell` are appended, with the aliases `EN = P` and `P2 = self`.
3. **`scripts/apa_om135_core.py`.** `prepare()` is appended. `load_engine`, `driver_check` and `ROOT`/`ART` are kept as
   dead code so the prefix stays byte-identical. `driver_check` refuses 595.84 and never sets it.
4. **The tests.**
   - **LD1 and LD2 engine tests:** they `importorskip("tensor_cuda")`, then probe CUDA by allocating one tensor, as
     `test_terrain_render.py`'s `_require_cuda` does, and skip if the probe fails.
   - **OM135 test:** `MAIN_BUILD` (the fork path) and `C.load_engine` are removed. It binds `C.tc`, `C.F` and `C.nn`
     from the importable engine.
   - **B04:** added as apa_quant fp32 at L = S = 33, with two `zws_cuts` geometry assertions.
   - **Unchanged:** detector logic, thresholds, comparisons and every existing cell.
   - Full diffs: `artifacts/apa_regt1/port_diffs_vs_fork.diff`.

## Known-bad runs (one engine per process; all three test files on each)

| detector | known-bad build | sha256 (prefix) | result |
|---|---|---|---|
| ZWS | PYSTAT: main .so + `apa-ld2 artifacts/apa_ld2/mutants/PYSTAT_fd32b6d/pkg` (quant.py differs from a28e28e only at :254) | .so d8fda9a8, quant.py 9425e47e | FAIL (ALARM, 40 blind rows, apa_quant fp32 (3,33)) |
| ZWS | `apa_ld2/mutants/CU7_stats_range/build` | 939787e4 | FAIL (ALARM, 34 blind rows, apa_sel bf16 (3,33)) |
| PDWL cut check, stagger cuts, ZWS | `apa_ld2/mutants/CU5_offby1/build` | 0769fa05 | FAIL on all 5 LD engine tests |
| PDWL label check, stagger cuts | `apa_ld1/mutants/CU1_gqa_tile/build` (supplementary) | 6902a615 | FAIL (4 tests) |
| LD1, stagger, ZWS | `apa_ld1/mutants/CU2_topleft/build` (supplementary) | 8632ff51 | FAIL (5 tests) |
| IMM / textbook / gradcheck | `apa_om135/mutants/MU2_swapped_dB_operands/build` | 2e166353 | FAIL ×3 dtypes |
| same | `MU3a_missing_transpose_trans_b_dB` | 51bbb68a | FAIL ×3 |
| same | `MU4_broadcast_keep_first_slice` | 55483b6b | FAIL ×3 |
| same | `MU5_alpha_dropped_trans_b_dB` | a02662ca | FAIL ×3 |
| same | `MU6_apa_dv_query_row` | 5635286f | FAIL ×3 |
| same | `MU1_dropped_not_trans_b` | 7cfb8e67 | FAIL ×3, by `RuntimeError: matmul inner dim mismatch` (a crash, not a detection) |
| same | `MU3b_missing_transpose_dB` | 0330b1ea | FAIL ×3, by RuntimeError (a crash) |
| none | `apa_ld2/mutants/CU6_tile_edge/build` | 5f9abe1e | **31 PASS.** CU6 never fires at the ported shapes: the frontiers are 17–24 and 31–33, never a multiple of 64. |

The seat's evidence script also ran each OM135 method separately. On every MU build the IMM, textbook and gradcheck
alarm sets are identical and non-empty. On main, all three pass on all 24 cell-dtypes.

### Coverage notes

- **CU6 (tile-edge) is not caught.** The source tests have no cell where a frontier lands on a multiple of 64. This
  is a coverage gap inherited from the source tests, not a port regression. A tile-edge engine cell is a follow-up
  item; the order allowed no new cells beyond B04.
- **`test_wrong_orientation_fixture_is_flagged_gpu` passes on every engine.** It is a positive control: every run, it
  asserts that all three methods alarm on OM135's Python-level wrapper mutant. Its detectors are the same three
  methods, and those fail on MU2–MU6.
  - Lead ruling: this is a detector self-test, not an engine-level test, so Measures (b) does not apply to it. It is
    kept.
- **B04 (the square control):**
  - With the corrected ghost block it PASSes on main (187 must-change rows).
  - Under the registered rule it is still REFUSED-operand.
  - It also PASSes on PYSTAT and CU7. At L = S, PYSTAT's mask equals main's as a nonzero predicate (0 entries
    differ), so the bug is a no-op there, and the rectangular cell carries the PYSTAT alarm.

## Per-test durations

The standalone run gives 31 passed in 1.67 s (`logs/apa_regt1_main_new_tests_r1.log`). In-suite times come from
`artifacts/apa_regt1/suite_with_r1.xml`.
- **Largest in-suite costs:** the LD1 engine test, 0.081 s (fp32) and 0.129 s (bf16), and the OM135
  three-methods-on-main test, 0.038–0.052 s per dtype.
- **Everything else:** under 0.03 s each.

## Full suite on main

| run | new files | summary | pytest s |
|---|---|---|---|
| r1 | with | 37 failed, 1956 passed, 46 skipped, 15 errors | 147.70 |
| r2 | without | 37 failed, 1925 passed, 46 skipped, 15 errors | 145.24 |
| r3 | with | 37 failed, 1956 passed, 46 skipped, 15 errors | 144.22 |
| r4 | without | 37 failed, 1925 passed, 46 skipped, 15 errors | 144.09 |

The FAILED/ERROR set equals the 52 baseline ids in every run.

## Deviations (seat)

1. **No report or ledger file:** the subagent harness refused them. This file is the lead's transcription.
2. **Read-only git:** the seat ran `git show`, `git diff --stat`, `git log` and `git ls-tree`, plus one plain
   `git status`, which may have refreshed the index stat cache. Nothing changed state.
3. **Helper code moved into the three authorized scripts:** the code of the four fork helper modules went there as
   verbatim line subsets.
4. **B04:** run in fp32 only, in the test's small geometry.
5. **Extra known-bad builds:** LD1's CU1 and CU2 were used as supplementary evidence.
6. **The evidence script:** its source is embedded in `artifacts/apa_regt1/evidence_zws_main.log`.

## Prior art

The sources' annotations are kept at each code site.
- **Detectors:**
  - openai/math #107 Prop 3.1 and #135, via the TX2 dossiers;
  - Goldberg (1991);
  - Chen, Cheung & Yiu (1998), metamorphic testing;
  - DeMillo, Lipton & Sayward (1978), mutation testing;
  - Dao (2023);
  - Dorfman (1943);
  - Nisan (1991) and Nisan & Wigderson (1997);
  - Baur & Strassen (1983);
  - Griewank & Walther (2008);
  - PyTorch gradcheck fast_mode;
  - Bridle (1990);
  - Freivalds (1977), for contrast.
- **This order's own pieces:**
  - the B04 ghost-block correction is APA-LD2's exploratory rerun;
  - the CUDA-skip probe is from `test_terrain_render.py`;
  - for the verbatim line-subset port, no prior art is known to me.

## Lead verification (2026-10-09)

Each check below was re-run by the lead, independently of the seat:
- **Forbidden strings:** `grep -nE "Fork|WorkTree"` over the six files exits 1.
- **Byte-identical prefixes:** `apa_ld1_pdwl.py` lines 1–355 equal cb2fac1, and `apa_om135_core.py` lines 1–765
  equal 2bbee82 (cmp). `apa_ld2_pdwl.py` differs from cb2fac1 only by the B04 block in `zws_cuts` and the appended
  adapter.
- **Driver:** "Open Kernel Module for x86_64  595.99.02".
- **Engine runs** (each under the GPU lock, one engine per process):

| engine (sha256 prefix) | result |
|---|---|
| main (d8fda9a8) | 31 passed in 1.66 s |
| PYSTAT | 1 failed, 30 passed |
| CU7 (939787e4) | 1 failed, 30 passed |
| CU5 (0769fa05) | 5 failed, 26 passed |
| MU2 (2e166353) | 3 failed, 28 passed |
| MU6 (5635286f) | 3 failed, 28 passed |
| main with `CUDA_VISIBLE_DEVICES=""` | 20 passed, 11 skipped |

## Cross-model review and promotion (2026-10-09)

### The review

- **Reviewer:** Codex gpt-6.1-sol, reasoning high.
- **Order:** `orders/APA_REGT1_REVIEW.md` on branch `apa-regt1rev`.
- **Run:** sentinel run 20261009T200915Z-1624941 ended COMPLETED with SHIM-DONE exit=0.
- **Receipts:** `apa-regt1rev` ffa99e4 (`docs/APA_REGT1_REVIEW.md`, `artifacts/apa_regt1rev/verdict.json`).

| gate | verdict |
|---|---|
| R1 faithful port | **FAITHFUL**: no non-declared logic differences |
| R2 sound as production tests | **SOUND-WITH-NITS** |
| R3 B04 | **CORRECT**: only B04 changes; all 17 rectangular cells keep their original cuts |
| R4 promotion of a28e28e..e4023c1 | **APPROVE-WITH-NITS**, no blocking findings |

### Open nits (none blocking)

- **F01 (SUSPICIOUS): the CUDA-availability guard is too broad.** It catches any exception from the one-tensor probe
  and skips. A programming defect in that probe would therefore show up as 11 skips, not failures.
  - Follow-up: narrow the guard to no-device errors, or have a GPU CI lane require zero GPU skips.
  - Production's `test_terrain_render.py` uses the same pattern.
- **F03 (BENIGN):** the r3/r4 summaries were abbreviated, and the duration sum was worded as overhead. Both are
  corrected above.
- **F04 (BENIGN):** the registered exact-integer APA operands make atomic accumulation order-independent. Future
  nondeterministic paths are not covered.
- **F05 (BENIGN):** the CU6 tile-edge gap is inherited from the source tests.
- **F06 (BENIGN):** the dormant `load_engine`, `driver_check` and `ROOT`/`ART` helpers in `apa_om135_core.py` are
  unreachable from the shipped tests.
- **F07 (BENIGN):** the wrong-orientation test is an active positive control, not an engine kill.

### The reviewer's GPU probes, run by the lead

The lead ran these on 2026-10-09 at about 20:31Z, on driver 595.99.02 with main d8fda9a8, using
`scripts/apa_regt1rev_gpu.sh`. All 3 passed:

| probe | result |
|---|---|
| repeat | 352 shipped GPU item calls, each repeated 32 times: zero skips, refusals, noise or assertion failures |
| zws | 96/96 valid ZWS runs PASS; the original B04 rule is still REFUSED-operand, with 264 ghost V-tag changes |
| om135 | 2304 certified method/cell/dtype checks, all non-alarm |

### What is promoted

- **Scope:** six new files, three test modules and the three script modules they import, plus their receipts. No
  engine source, existing test or runtime path changes.
- **Covers:**
  - the ZWS statistics tag, the PDWL/frontier-first stagger cuts and the OM135 IMM/textbook/gradcheck backward checks,
    at their registered cells, on the engine as built from main;
  - the B04 square cell, with the corrected ghost block.
- **Does not cover:**
  - CU6 tile-edge leaks;
  - nondeterministic backward paths;
  - engine speed;
  - GRM or GRAPA consumers.
- **David's approval** (2026-10-09, verbatim): "On things like regression tests going into production, that's fine
  unless it really significantly hurts performance."
- **The performance condition:** the measured suite overhead is about 1 s on a 146 s suite, and engine runtime is
  untouched.
