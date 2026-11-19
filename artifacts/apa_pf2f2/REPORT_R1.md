# APA-PF2-F2 report

**RED: SCOPE MISMATCH.** An edit at the sole authorized load,
`packed.cuh:203`, cannot fix clause 3 for the required grouped decode
cases. House Rules §3 stops the order before implementation or gates.
**Not claimed fixed.** Full acceptance is pending a scope amendment and the
registered gates.

## Evidence

- **Contract.** `docs/APA.md:131` says “A key whose stored norm is non-finite,
  for any reason, makes every row that reads it non-finite.”
- **Source inspection.** The public decode entry uses `gqa_path="r2"`
  (`tensor_cuda/tensor_cuda/apa_packed.py:547–549`), as does the G2 norm
  witness (`scripts/apa_pf2f1_nonfinite.py:41–47`). The binding and r2
  wrappers pass the store scales into `apa_spq1_core`
  (`bindings.cpp:2785–2789`, `spq4.cuh:460`, `packed_launcher.cuh:102`).
  The production dispatcher sends G=4 and G=8 to `spq1::bulk_shared`
  (`launcher.cuh:266–278`). That kernel reads the stored value into scalar
  `n` at `packed.cuh:149`, from lane 0 with a broadcast for D=128, and
  directly for D=64. The TQ4 multiply reads it at line 178. The separate
  `spq1::bulk` kernel owns the line 203 load. A `norms[]` search misses
  the scalar load.
- **Inherited GPU receipt.** All 16 lead `G2_norm` cases failed
  `"failed": ["raw_inf_decode_row_sets"]`, with
  `"inf_decode_nan_elements": 0` and `"nan_decode_all_nan": true`.
  Eight cases used G=4 (both D, both dtypes and both prefill diagnostics
  settings). Prefill had 128–1,024 NaN elements. The log at
  `../tensor_fork_apa_wt_p1pf2/artifacts/apa_pf2f1/gpu/G2_norm_real.log` hashes to
  `30e54788ecbd485eac2bb38827301ac1d82671c20226f932eaf48ed8851ca1e5`,
  matching `G2_norm_real.exit.json` (exit 1). The three local CUDA headers
  in the source trace matched their F1 build source hashes.
- **Reasoning from the dispatch.** Changing line 203 leaves all eight G=4
  witnesses on a grouped kernel that still multiplies the negative dot by
  raw +Inf. The correction requires the same finite check after the grouped
  norm load at line 149. A separate order amendment can authorize that
  second site and register SASS and mutant expectations for both kernels.

## Diff

Empty. No product or test edits.

## Module SHA-256s

| Build | SHA-256 |
|---|---|
| F1, lead build and staged module, both verified | `641a3139181cdee3e800f62afad08c0881b3018c26e57d953fa6c13413f7cbe7` |
| F2 | None. Build held. |
| `pf2m01_decode_norm_finite_only` | None. Build held. |

## CPU results

Tool preflight only. CMake 3.28.3, nvcc 12.0.140 (PATH) and 12.6.85 (F1
compiler), pytest 9.0.2, Python 3.12.3. G1–G4 were not executed or staged.
G5 scope check found only the three authorized evidence files written, with no
git commands, deletions or GPU use. Formal gate registration was held.

## Mutant

| Mutant | CPU result | GPU killing test |
|---|---|---|
| `pf2m01_decode_norm_finite_only` | NOT BUILT | NOT RUN. G2 norm witnesses held at the scope check. |

## STAGED_LINES.md

~~~markdown
# STAGED_LINES.md

**HOLD — RED: SCOPE MISMATCH.** Zero GPU lines staged and zero dry runs. The
required G2 norm reproduction includes eight G=4 cases that dispatch to
`spq1::bulk_shared` and bypass the sole authorized edit at `packed.cuh:203`.
This seat stopped at the pre-build mechanism check. There are no launcher
commands to run. See `REPORT.md` for the dispatch evidence.
~~~

## Deviations

The required mechanism check halted implementation, registration, builds,
CPU tests and GPU staging. No script repairs, tool slips or threshold changes.

## Prior art

IEEE 754-2008 quiet-NaN propagation and Project-Tensor PF2-F1 (2026).
The proposed two-site decode change reuses PF2-F1’s finite check and
quiet-NaN sentinel (`packed_prefill.cuh:101–104,125–127`). The additional
site follows from this worktree’s dispatch. The external clause lookup is
unverified — lead to check `IEEE 754-2008 6.2 quiet NaN multiplication`.

## Every file written

- `artifacts/apa_pf2f2/LEDGER.md`
- `artifacts/apa_pf2f2/REPORT.md`
- `artifacts/apa_pf2f2/STAGED_LINES.md`
