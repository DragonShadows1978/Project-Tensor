# APA-SP4G amendment 1 (lead, 2026-09-07) — first model cell RED on the card; coalesce the 8192 margin bands

Same worktree (`/mnt/ForgeRealm/Project-Tensor-wt-apa-sp4g`, your r1
committed as d1a04a8), same boundaries and rules. `kernel512` PASSED on
the card (D=512/MQA pins, all dtypes). Then:

**`ppl_A_2048_w0` RED** (receipt `artifacts/apa_sp4g/jobs/ppl_A_2048_w0.json`,
traceback stored):

```
AttributeError: /usr/local/cuda-12.6/lib64/libcudart.so.12: undefined symbol: cudaGetDeviceDefaultMemPool
```

The CUDA runtime API is `cudaDeviceGetDefaultMemPool` (and
`cudaMemPoolGetAttribute` for the high-water counters). Fix the symbol
(and any sibling you spelled from memory), add a CPU test that resolves
every ctypes symbol your worker uses against the installed
`libcudart.so.12` at import time so a misspelling REDs in the sandbox
rather than on the card, and re-run your CPU gates. Fingerprint
amendment as SP3 a4 did; `kernel512` stays valid if its closure is
untouched, else say so.

**Second item, registered now:** your G2 split the S=8192 margins into
128-query bands, 64 per layer × 8 layers × 2 arms = 1,024 cells, ~10.7
hours of mandatory cooldown for what is, with ONE KV head, 67 M pairs per
layer (MiniCPM3 did 40 heads × 8192² per layer in one job under the same
rail). Re-register the 8192 margins as ONE cell per (arm, layer) with a
registered planning estimate under 285 s, keeping bitwise capture replay
and the all-pairs population percentiles; keep the bands only as a
fallback kind the lead can opt into if a whole-layer job hits the rail.
Same for the 2048 margins if they are banded. Refresh `lead_commands.txt`
and the blocked-report; `cells.json` gains an amendment record, the
registration itself stays immutable.

No git, no subagents, foreground only, < 10 min per call, never kill
anything. Prior Art Directive applies (nothing new expected; say so).

## Done (verbatim)

1. The fix (file/lines), the symbol-resolution test name, CPU gate
   results.
2. New margin cell ids/estimates; receipts affected by the fingerprint
   amendment (expected: none or `kernel512` only, with reason).
3. Refreshed lead commands; blocked-report; RED; process safety; model
   id and effort.
