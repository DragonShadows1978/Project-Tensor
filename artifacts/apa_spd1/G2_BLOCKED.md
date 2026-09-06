# G2 BLOCKED — lead GPU execution required

Implementation seat: `gpt-6-astra`, high effort. Evidence class: CPU import probe
and source inspection. **No SPD1 GPU timing, GPU numerical, or peak-memory result
has been obtained.** This is the dispatch's expected GPU boundary.

`CPU_INVENTORY.json` reports torch `2.11.0+cu130`, CUDA build `13.0`, and
`cuda_available: false`. The inherited engine binary imports and matches its
source manifest. The orientation allocation probe failed verbatim:

```text
RuntimeError: CUDA error at from_host: no CUDA-capable device is detected
```

Torch math/efficient/flash enum/API presence is confirmed. Their **sm_89 execution
availability remains unknown**; the leased worker forces each backend separately
on the actual cell. `flash_attn` is absent. No SP2 frozen table was found in
`/mnt/ForgeRealm/Project-Tensor/artifacts/apa_sp2`; that optional row is skipped
unless the runtime discovery finds one. The inventory contains these facts, not
inferred flash support from the GPU architecture name.

## Exact lead commands

Run in the dispatched worktree with the inherited SP1 build present. Nothing
needs to be installed, compiled, or changed in main for the existing contenders.
Each `run` or `resume` invocation processes at most **one cell**. Repeat `resume`
as separate foreground invocations until it says all cells were attempted; inspect
RED exits and explicitly rerun a failed cell only after recording its disposition.

```bash
cd /mnt/ForgeRealm/Project-Tensor-wt-apa-spd1
timeout 55s bash scripts/apa_spd1_lead_gpu.sh list
timeout 55s bash scripts/apa_spd1_lead_gpu.sh --dry-run
timeout 55s bash scripts/apa_spd1_lead_gpu.sh inventory
timeout --signal=TERM --kill-after=5s 590s bash scripts/apa_spd1_lead_gpu.sh run prefill_s512_d64_c0_h4_kv4
timeout --signal=TERM --kill-after=5s 590s bash scripts/apa_spd1_lead_gpu.sh resume
timeout 55s bash scripts/apa_spd1_lead_gpu.sh summary
```

Device defaults to physical index 0; set `APA_SPD1_GPU` to another single numeric
index only if it is an sm_89 card. This order is registered for sm_89; another
architecture requires a separate amendment.

The runner holds `/tmp/forge-gpu.lock` through the 30-second foreground cooldown.
Lease wait ≤5 s; failed/busy `nvidia-smi` aborts before worker launch. Worker timeout
510 s, with up to 5 s to terminate **only its own spawned process**; outer envelope
580+5 s. Bound calculation: lease 5 + next-cell selection 5 + GPU probe 5 + stamp 3
+ worker 515 + exit receipt 5 + cooldown 35 = 573 s, before the hard outer envelope.
No background wait, GPU sampler thread, retry loop, `kill` command, service action,
git command or subagent. The lock is advisory; noncooperating clients can still
race after the busy probe. The lead must keep other heavy card work sequential.

Receipts and transcripts: `artifacts/apa_spd1/gpu/CELL.STAMP.receipt.json`,
`.exit.json`, `.log`. Attempts are never overwritten. `resume` skips previously
attempted cells at the same fingerprint, including failures; `run CELL` is the
explicit retry. Source/runtime/harness/SP2 pins prevent stale results from being
silently accepted. A changed optional SP2 table makes prior fingerprints stale.

`summary` writes `SPEED_CHAIN.md` and `speed_chain.json` from current receipts,
including explicit missing/unavailable/RED rows, fixed-universe prediction counts,
same-dtype speed ratios, IQR-overlap qualifications and E1/paper reconciliation.
It cannot promote a numerical RED or blocked SP2 interface to G2 PASS.

## Conditional SP2 integration

No SP2 table/launcher interface exists at implementation time. At runtime:

1. No `*FROZEN*` file under the specified main artifact directory: `UNAVAILABLE`.
2. A frozen candidate exists but no actual launcher adapter is provided:
   `BLOCKED_SP2_INTERFACE`; the other rows run, G2 coverage remains blocked.
3. For an actual available launcher, a **separate amendment** can supply
   `artifacts/apa_spd1/sp2_adapter.json` with `frozen_table`, `table_sha256`,
   `launcher`, `launcher_sha256`, and `entry_point`. The launcher must be under
   main and is only read/imported. Its entry accepts keyword arguments
   `tc, shape, tensors, epsilon, frozen_path` and returns a dictionary containing
   `call` (zero-argument actual SP2 attention callable) and `metadata`. `tensors`
   is `(q,k,kq,v)`, FP32, same objects as the other engine rows; epsilon is fixed
   at `1e-3`. Setup must use frozen calibration only and is outside measurement.

This is a documented extension contract, **not a claim that an absent SP2 API was
implemented or GPU-tested**. No replacement δ is inferred or tuned. The lead must
validate the real adapter when SP2 is delivered; this seat cannot change main.

kernel sweep; this establishes nothing about model quality
