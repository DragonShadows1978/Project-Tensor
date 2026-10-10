# APA-REGT2 report: the packed 0/1 identity probe (pIdV) ported to production as regression tests

Order: `orders/APA_REGT2_PIDV_REGRESSION.md` (d2c6606). Seat: Claude Opus 5.5 high (Agent-tool subagent), 2026-10-09.
The harness refused report files from the subagent, so the lead transcribed this report from the seat's final
message and added the lead's own verification at the end. The full per-test, per-build and port-diff detail is in
`artifacts/apa_regt2/` (ledger, port diff, guard demo, suite compare) and `logs/apa_regt2_*`.

David, 2026-10-09, verbatim:
- the standing rule: "On things like regression tests going into production, that's fine unless it really
  significantly hurts performance."
- on porting the 0/1 probe: "Yes, yes, and yes."

## Verdict

| gate | result | evidence |
|---|---|---|
| G1 port | **PASS** | Two shipped files, plus evidence. No `Fork`/`WorkTree`/absolute paths in shipped files. With `CUDA_VISIBLE_DEVICES=""`: 8 passed, 27 skipped. An injected non-device exception in the availability probe makes 26 tests FAIL; they do not skip. This closes REGT1 review F01 for this module. |
| G2 main | **PASS** | 34 passed and 1 skipped (`TC_SLOW_TESTS`) on main d8fda9a8; the slow cell passes when enabled (20.8 s). Five full-suite runs: FAILED/ERROR = the 52 baseline ids every time, with 0 new. |
| G3 teeth | **PASS, 8/8 builds** | CU1 5 failed, CU3 8, CU4 8, CU5 8, **CU6 4**, I1 5, I2 2 (split-K only), I3 5. Every failure is on an apa_selective cell, the mutated kernel. Every other kernel's test passes on every mutant. |
| G4 cost | **PASS** | The new tests' summed in-suite time is 11.9 s, against a 30 s budget. Suite wall deltas were +0.8 s and +6.1 s (run-to-run spread is about 6.7 s). The S = 98,304 cell (26.9 s, 10.75 GB peak RSS) is behind `TC_SLOW_TESTS=1`. |

**What it adds over REGT1.** It catches **CU6, the tile-edge leak that REGT1's tests miss**. It also catches
wrong-head, wrong-batch and interior-tile reads (CU1, CU3, CU4, I1, I2, I3) at the production layout (B 2, Hq 16,
KVH 8, D 128, S 8192), with one run per cell and an exact zero/nonzero oracle.

**Engine runtime is unaffected**: no engine source changed.

## Files

| file | lines | contents |
|---|---|---|
| `scripts/apa_pidv.py` | 327 | See the list below. |
| `tensor_cuda/tests/test_apa_pidv.py` | 331 | 8 CPU tests, plus 19 edge cells and 7 production-layout cells on the GPU. Each GPU cell asserts PASS on main and ALARM on an in-test positive control (offby1 or bstride). One S = 98,304 cell is behind `TC_SLOW_TESTS=1`. |

The script contains:
- the pIdV detector: `spread`, `_empty_layout`, `_cut_nt`, `pidv_layouts`, `pidv_operand`, `pidv_check`,
  `pidv_cell`;
- `PidvAttn`, which re-adds sdpa_fused_cs, apa_int4 and apa_sp to REGT1's adapter;
- `synthetic_margins`, `make_cell` and `bind`.

## Port differences from LD3 (24f8b11)

1. **The detector is unchanged.** All seven functions are byte-identical; the lead checked them by AST source
   segment. The alias `V2 = sys.modules[__name__]` keeps `pidv_cell`'s `V2.` calls verbatim.
2. **`PidvAttn(P.Attn)`** re-adds the sdpa_fused_cs, apa_int4 and apa_sp branches from `apa_ld1_entries.py`. These
   are needed because REGT1's adapter port dropped them.
   - `TC_APA_SP=1` is set and restored around each apa_sp call, where LD1 set it process-wide.
   - apa_sp's kq is computed before the parent's `_prep` (same values).
3. **`synthetic_margins(dirpath)`** writes to pytest's tmp area, never into the repo.
4. **Not ported:**
   - v2, labels and twins, `column_check`, the campaigns, and the idV and dense baselines;
   - the GRM and sdpa_mask cells (out of scope);
   - the non-production-geometry long-S cells.
5. **Dead code:** one unused 5-line helper, `operand_footprint_bytes`. It is shipped, to keep the recorded hashes
   valid. Nit.

## Availability guard demonstration (`artifacts/apa_regt2/g1_guard_demo.log`)

| case | outcome |
|---|---|
| real no-device | skip (27) |
| extension not importable | skip (27) |
| injected non-device RuntimeError | **26 FAILED** |
| injected no-device RuntimeError | skip (27) |
| injected non-extension ImportError | **26 FAILED** |

## Seat deviations

- A suite-run filename glob mislabelled one with-run. It was kept as an extra with-run and renamed, and the
  without-runs were redone.
- A guard-demo log was overwritten by a re-run; the outcomes were identical.
- Two exploratory timing probes on main were used to size the budget.
- Scratch files remain under `/tmp`, which is exempt.

## Prior art

These annotations are at the code site, `scripts/apa_pidv.py:31-48`:
- Dorfman (1943) and non-adaptive combinatorial group testing (Du & Hwang): the set-indicator columns are group
  tests;
- identity-V (APA-LD1/LD2): the exact zero/nonzero oracle;
- DeMillo, Lipton & Sayward (1978): mutation testing;
- Dao (2023): bottom-right causal;
- the pIdV design itself is APA-LD3's.

The narrow availability guard has no prior art known to us. The external items are unverified; lead to check.

## Lead verification (2026-10-09)

- **Paths:** only the two shipped files plus `artifacts/apa_regt2/` and `logs/apa_regt2_*` are new. No tracked file
  is modified. No `Fork`/`WorkTree`/`/mnt/`/`/home/` strings in the shipped files.
- **Source:** LD3 sources on disk equal commit 24f8b11, and the seven detector functions are byte-identical (AST
  source segments).
- **Re-runs** (each under the GPU lock, on driver 595.99.02):

| engine | result |
|---|---|
| main d8fda9a8 | 34 passed, 1 skipped |
| CU6 5f9abe1e | 4 failed |
| I2 add3c3ec | 2 failed |
| CU1 6902a615 | 5 failed |
| I3 c22c290e | 5 failed |
| no CUDA | 8 passed, 27 skipped |
| **production package with its installed engine** (PYTHONPATH `tensor_cuda/tensor_cuda:tensor_cuda` of `/mnt/ForgeRealm/Project-Tensor`) | 34 passed, 1 skipped |
