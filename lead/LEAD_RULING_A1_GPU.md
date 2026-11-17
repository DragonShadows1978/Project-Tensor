# APA-P1 A1 GPU run (2026-11-03, standing window): lead ruling — P149, P150 and P151 HELD

**All three registered predictions hold.**
- **The run:** the 58 staged lines of `artifacts/apa_p1_a1/STAGED_GPU_LINES.sh` (sha256 `6b0e8bf1…`, registration
  `d51c0f48…`), launched by `lead/run_p1_a1_gpu.sh` (06668fa8) from 06:04 to 11:04, in 29 duty-cycle legs, every line
  through `forge-gpu-run --min 120 --max 600`.
- **The result:**
  - every line exited 0, and the staged verdict is `A1 verdict exit=0`;
  - the wrapper logged `staged script rc=0`.
- **The seal:** the run root was sealed with `post_run_manifest.sh`: 6,765 files, manifest sha256 `00a0afef…`. The lead
  re-checked it against disk from the run root with `sha256sum -c --strict`: rc 0.

**The lead scored the predictions from the per-line comparison records, not from the exit codes:**

| Prediction | Result | Evidence (tabulated by the lead) |
|---|---|---|
| **P149** (the fixed port is bitwise identical to the fixed lab build on every staged bitwise case, new shapes included) | **HELD** | 13 cases, 858 fields compared, **0 unequal**. Every reference is nonzero and non-constant. The new odd shapes are included (B1_H2_KV1_S255/S256/S257, B1_H4_KV2_S257, B2_H2_KV1_S257, B2_H4_KV2_S255/S257). |
| **P150** (no production test outcome changes under the fixed port; zero-test files compared as ID sets) | **HELD** | 44 regression files, 2,078 test IDs, **0 outcome differences**, every collected ID set equal. `test_apa_tiled.py` collects zero tests on both arms and is scored as equal empty sets: the scorer fix from GPU run 1's ruling. |
| **P151** (bytes and layout per S follow the fixed recurrence; no layout assertion on any staged shape, diagnostics on or off) | **HELD** | Bytes: 6 model shapes (Qwen3-1.7B and Qwen3-4B, S = 2,048, 8,192 and 32,768), every layer's measured bytes equal the recurrence (580 B per key per KV head on packed layers, 512 on layer 0), with 0 mismatches. Layout: `bytes/arena_comparison.json` PASS on 13 cases. The bitwise cases carry the `eb3_nondiag` and `eb4_nondiag` fields beside the diagnostic fields, and all ran without an assertion. That includes shapes where B·KV·⌈S/64⌉ is not a multiple of 8, which tripped EB4's assertion before the fix. |

**On P151's layout clause:** the non-diagnostic arena layout is checked against the source recurrence, together with
successful non-diagnostic calls. The diagnostic arenas are measured. That is how the A1 order registered it.

## What this establishes

**The production port of APA-2P carries both defects' fixes:**
- DS1's 16-byte arena alignment, and the zero-size skip;
- EB4's layout assertion.

On the GPU it is bitwise equal to the fixed lab build, and it changes no production test outcome. Its KV memory is the
2P level: 580 B per key per KV head on packed layers.

## Next for P1

These are promotion items, not further measurement:
- the S3 tickets and their tests;
- a blind red team;
- the default switch to packed (L18);
- the caller check;
- the owner's verbatim approval.

## Receipts (this commit)

- **The records:** the 117 comparison and exit records under the run root: 13 bitwise, 44 regression, bytes, the arena
  comparison, and the 58 `*.exit.json`.
- **The wrapper's log, line output and manifest:** in `lead/a1_gpu/`.
- **Held back under the manifest:** the bulk arrays, per-arm builds, pytest temp trees and per-line logs (2.0 GB). They
  stay in the worktree.

## Prior art

No new algorithm. CUDA natural alignment for 16-byte vector loads, aligned bump allocation, and pytest outcome
comparison.
