# APA-SP3 amendment 5 (lead, 2026-09-06) — G3 decode with the allocation pool ON; 32K captures OFF by default

Same worktree as amendment 4 (`/mnt/ForgeRealm/Project-Tensor-wt-apa-sp3-a4`,
branch `apa-sp3-a4`, your a4 work committed by the lead as a0276ed). Same
boundaries and rules. The lead merges into `apa-sp3` after the running
chain there ends.

## Finding on the card (bits 4, S=2048, PROTOCOL-2)

`decode_b4_A_2048` 1.84 tok/s, `B` 0.98, `C` 0.78, with the receipt stating
`raw pool OFF`. The June receipt for this model on this engine is 21.6
ms/token (46 tok/s) with `tc.set_alloc_pooling(True)`, which is the
adapter's production setting (see `GraftRepository/tests/minicpm3_bulkbits_floor.py`).
Pool-off decode pays raw cudaMalloc/cudaFree per step through your
LD_PRELOAD interposer; those numbers are memory-shape evidence, not speed.
`decode_b4_B_32768` then hit WORKER_TIMEOUT for the same reason; `A_8192`
and `A_32768` are the registered dense non-fit.

## Mission

1. Register a new kind `decode_pool` (ids `decode_pool_b{bits}_{arm}_{S}`,
   same arms/lengths/steps as `decode`) that runs decode with
   `set_alloc_pooling(True)` exactly as production, CUDA-event or
   synced-wall per-token timing as before, ≥ 32 steps, and reports
   tokens/s and ms/token. Peak memory in this kind is whatever the pool
   reports (state the source); the pool-off `decode` receipts remain the
   exact-peak evidence and are NOT re-run. Prediction P5 ("SP decode at
   32K ≥ 2× two-pass tokens/s in-model") is evaluated on `decode_pool`.
2. Rails: 290 s worker TERM for `decode_pool` at S ≤ 8192; for S=32768
   register the planning estimate from the 8192 measurement and say if it
   cannot fit the rail (then it is a non-fit, not a retry).
3. **32K captures OFF by default**: the `capture_*_32768` ranges and
   aggregations (your a4 addition) stay registered but are gated behind
   an explicit lead flag/cell-list inclusion; they are not in the
   default `lead_commands.txt` order. Disk: 366 GB free on the NVMe; one
   32K arm is ~93 GB by your layout. The lead decides whether to run
   them; you do not.
4. Fingerprint amendment as in a4 (new amendment JSON, immutable; per-kind
   closure). Existing receipts — including your a4-registered kinds —
   must stay valid; if this change touches a legacy kind's closure, say
   which and why.
5. CPU gates (new kind, rails, pool flag actually set — pin it with a
   test that asserts the pool state inside the worker), `--dry-run`,
   refreshed `lead_commands.txt` (dependency order; `decode_pool` after
   `freeze_b{bits}` like `decode`), blocked-report, ledger. No git, no
   subagents, foreground only, < 10 min per call, never kill anything.

## Done (verbatim)

1. New cell ids/estimates; the pool-state pin test name; what the peak
   field means under the pool.
2. Fingerprint amendment path; receipts affected (expected: none).
3. CPU gate results; blocked-report; exact lead commands.
4. Prior art; deviations; RED; process safety; model id and effort.
