# PAINT-CUDA-3 frozen engine gate register

Status: frozen before implementation on branch `paint-cuda-kernels` at
`7f9aef701a2b3c764673c34bf9f519db3f670c04`.

This leg is engine-only. It does not import or modify ColdCast. The CPU oracle
is ColdCast `hy3d_tc/paint/bake.py:926-1131`: one global pass over the original
uncolored face/corner occurrence stream, stable CSR neighbor order, in-place
float32 updates, and the WO-30b global pass/island cap receipt semantics.

## Public contracts

```python
build_inpaint_island_csr(
    faces_i64,
    vertex_islands_i64,
    uncolored_occurrences_i64,
) -> tuple[
    neighbor_offsets_i64,
    neighbors_i64,
    active_island_ids_i64,
    island_offsets_i64,
    island_occurrences_i64,
]

inpaint_island_passes(
    positions_f32,
    vertex_colors_f32,
    vertex_mask_f32,
    neighbor_offsets_i64,
    neighbors_i64,
    island_offsets_i64,
    island_occurrences_i64,
    pass_count_cap,
    *,
    threads=128,
) -> tuple[vertex_colors_f32, vertex_mask_f32, island_uncolored_i64]
```

`build_inpaint_island_csr` is a host-only NumPy builder. Neighbor CSR is
stable-sorted by source vertex exactly like the oracle. Active island IDs are
in ascending label order; occurrences are grouped by island with a stable
sort, preserving their original order within every island.

`inpaint_island_passes` is detached, CUDA-only, and float32/int64-only where
named. It launches one active island per warp; lane zero serially visits that
island's occurrence segment and stable neighbor rows, while islands execute
concurrently. Each call performs exactly `pass_count_cap` complete global
passes and returns cloned updated state plus the final uncolored-occurrence
count for every active island. `threads` is a warp multiple in `[32,1024]`
and changes packing of island-warps into CTAs only.

The shipping mode is ordered in-place smoothing only. No Jacobi or
per-island-convergence arm is registered or built.

## Arithmetic and order contract

For each occurrence, neighbor order is fixed. For each currently colored
neighbor, the kernel performs separately rounded float32 subtract, square,
ordered three-term sum, square root, floor at `1e-4`, reciprocal, weight
square, color multiply/add in channel order, and weight add. A positive total
weight promotes the vertex immediately using separately rounded division, so
later occurrences in the same island/pass observe that update. No FMA,
floating atomic, cross-island update, or occurrence reordering is permitted.

Disconnected islands share no vertex state, so interleaving whole island
segments is non-semantic. Global convergence and deadlines remain host policy:
an exact oracle driver launches one pass at a time, consumes the summed
uncolored count, updates the oracle's `smooth_count`, checks the material soft
deadline between launches, and stops at the same global island-iteration cap.
On a cap, every `active_island_ids_i64` row is emitted with the same completed
global pass count and the existing WO-30b reason/schema. A multi-pass launch is
only valid when the caller already intends to execute that exact bounded pass
count; it does not introduce per-island convergence.

## Registered gates

| Gate | Frozen pass condition |
|---|---|
| G-ORDER | Chain, star, ring, single-texel, 10,000-island swarm, and one giant island are byte/bit-exact against the vendored CPU oracle at equal global pass counts. If any arithmetic operation prevents exactness, stop the exact claim there and report mismatch count, maximum ULP, and maximum absolute spread; no threshold is inferred. |
| G-CAP | Host-driven global pass cap and island-iteration cap identify the same active islands, reasons, and completed pass counts as the CPU oracle. Deadline checks occur between one-pass launches; capped still means every active island is reported under the existing WO-30b schema, not merely islands that remain uncolored. |
| G-DET | Every output buffer is byte-equal across five reruns at each of two `threads` configurations and byte-equal across configurations. |
| G-SUITE | The full engine suite adds no failure relative to the PAINT-CUDA-2 baseline. The strict blend xfail becomes a pass under the lead-registered 2026-07-16 gate: trust at most 4 ULP, texture at most 8 ULP, and absolute spread at most `5e-7`. The standing GroupNorm failure and selector CLI collection exclusion remain recorded. |

## Performance registration

Report synchronized resident-state wall time at fixture scale and synthetic
production scale with 110,386 active islands and occurrence counts sampled
from the dragon receipt distribution. Report peak GPU memory against the
registered 3.0 GiB rail. This leg registers no performance threshold.

## Scope boundary

Navier-Stokes fallback, Canny, nearest-valid fallback fill, material encoding,
wall-clock measurement, receipt serialization, graph construction, component
labelling, and convergence policy remain on CPU. This engine primitive only
executes bounded ordered island passes over caller-built CSR.

## Execution receipt

Build: PASS, CUDA 12.6.85, Release sm_89. Generated flags:

```text
CUDA: -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_89,code=[compute_89,sm_89] -Xcompiler=-fPIC -Xcompiler=-fvisibility=hidden
CXX:  -O3 -DNDEBUG -std=gnu++17 -fPIC -fvisibility=hidden
```

| Gate | Result | Receipt |
|---|---|---|
| G-ORDER | PASS | Chain, star, ring, single-texel, 10,000-island swarm, and one 8,192-vertex giant island all matched CPU color, mask, and per-island uncolored-count buffers byte-for-byte at equal pass counts. Across every fixture: 0 mismatched color elements, 0 ULP, 0 absolute spread, 0 mask mismatches, 0 count mismatches. |
| G-CAP | PASS | Mixed active islands `[0,4,9]`: iteration budget 2 capped all three at pass 2 with reason `island_iteration`; deterministic deadline budget 1 capped the same three at pass 1 with reason `wall_clock`; uncapped execution completed exactly at pass 6 with no capped rows. CPU and CUDA state plus WO-30b receipt rows were identical, including original occurrence counts `[29,2,1]` and `uncolored_occurrences_last_complete_pass: null`. |
| G-DET | PASS | Every color, mask, and island-count buffer was byte-equal over five reruns at 32 threads, five reruns at 256 threads, and across both configurations. |
| G-SUITE | PASS relative to standing failure | Before (PAINT-CUDA-2 accepted): 299 passed, 18 skipped, 1 strict xfailed, 1 failed. After: 311 passed, 18 skipped, 1 failed. The only failure before/after is `test_ext_phase7.py::test_norms_and_conv1d` at the standing GroupNorm reshape. The selector CLI script remains excluded. The strict blend xfail is now a pass at observed trust 2 ULP, texture 4 ULP, and maximum absolute spread `2.384185791015625e-7`, inside the lead-registered 4/8 ULP and `5e-7` bounds. |

The G-ORDER arithmetic operation was the inverse-square neighbor-weighted,
in-place float32 add/average described above. Bit-exactness did not fail, so
there is no nonzero spread or inferred tolerance for island smoothing.

Performance uses the compact receipt scale (228 islands, 2,487 occurrences,
8 passes) and the full dragon histogram duplicated across both material
instances (110,386 islands, 2,895,698 occurrences, 11 passes). The production
synthetic has 1,092,370 vertices and four stable chain neighbors per vertex.

| Workload | One bounded pass | Fixed known pass count, one launch | Oracle driver, one-pass launches + host count/deadline boundary |
|---|---:|---:|---:|
| Fixture | median 0.028579 ms | 8 passes: 0.088343 ms | 8 launches: 0.316399 ms |
| Synthetic production | median 65.279598 ms | 11 passes: 642.947513 ms | 11 launches: 715.190057 ms |

Production resident inputs were 93.775 MiB by array size. Process memory was
182 MiB at context baseline, 286 MiB with inputs resident, 320 MiB at one-pass
peak, and 352 MiB at the oracle-driver peak. Whole-GPU peak increase over
baseline was 170 MiB: PASS against the 3,072 MiB rail. After all runs the GPU
reported 197 MiB used and 0% utilization.

Raw performance values are in
`docs/briefs/PAINT_CUDA_3_PERF_RECEIPT.json.txt`; the extracted dragon
histogram and artifact hashes are in
`tensor_cuda/tests/paint_inpaint_dragon_distribution.json.txt`.
