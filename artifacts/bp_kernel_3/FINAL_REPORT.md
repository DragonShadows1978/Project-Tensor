# BP-KERNEL-3 — CPU preparation complete; native gates untested

Both g1 and g2 built successfully, SM 89 only. No GPU work was performed.
57 CPU author tests passed; both final dry-runs completed. One scope exception
is disclosed below. Predictions have not been evaluated.

## Implementation

16 query x 16 key tiles, padded width 128, four warps / 128 threads.
WMMA 16x16x16 BF16 inputs, FP32 accumulation for bulk scores, dO V^T and all
three gradient outer products. g1 selected exact scores use masked shared
scalar loops; g2 uses full exact MMA. p/dS round to BF16 for MMA; the imported
BP-KERNEL-2 FP64 gate remains decisive. D/VD 1..128, BF16 only, explicit guards.
Group heads accumulate within key ownership; every gradient element is
written once, no atomics.

Code: kernels.cu 3017–3218; WMMA helpers 3035–3047; tiled owners 3050–3169;
launcher 3179–3218; f half instrumentation 2884–2885 and 2967–3013.
Explicit dispatch: ops.cpp 20, 1094–1097, 1117–1144; bindings.cpp 16–17, 932–934.
Diff: kernels.cu +220/-0, ops.cpp +9/-2, bindings.cpp +5/-0; new driver/test
lines 296/244/230. `final_diff_stat_revision_2.json` is the current diff index.

Original a bytes [107888,111834) unchanged:
`e7adc1e3442732b2fa221513ad75e2bdf665f62703f74b8b9c7410e0fb090a95`.

CPU cuobjdump confirms `HMMA.16816.F32.BF16` in all four compiled kernels.
g1 query/key: 76/72 registers, 32,320 shared bytes; g2: 78/80 registers,
33,344 shared bytes. All report zero stack/local memory. This is binary
inspection, not native correctness or performance validation.

## Validation

```
57 passed in 0.85s
[100%] Built target _tensor_cuda
```

Build exit 0. Combined-output final line is the retained unused-variable
warning remark: `Remark: The warnings can be suppressed with "-diag-suppress <warning-number>"`.
No suppression was added. Full build output and binary hash are in
`engine_build_receipt.json` / `build.log`.

```
DRY_RUN /mnt/ForgeRealm/wt/pt-bk3/artifacts/bp_kernel_3/dry_kernel_a16e08ec77dd4ecbba01ef6988b5ff5a/receipt.json
DRY_RUN /mnt/ForgeRealm/wt/pt-bk3/artifacts/bp_kernel_3/census/dry_c159fc2623cf4b4789ac2c0b105078f8/receipt.json
```

Registration sha256:
`abac971ebca5d8e93324077b5b8634cc69a24863964ed01ecfbb589a79e23356`.
Current source-only amendment chain sha256:
`62beefb80710b8b7aeb3deeff8292aa9021067fe0de935849ba5a426db51af4b`.
Census registration sha256:
`6c8b473d9188b9cb0e93ab08132058db336a6dac909e28f312cd04881dff53f5`.

Predictions, verbatim:
- best green g ≤ 0.35 × f (≤ ~46 ms at f ≈ 131).
- whole step ≤ 2.5 s with g (f-arm reproduces ≈ 5.0 s within 10 %).
Secondary: g1 ≤ g2 (selection still pays on tensor cores); if g2 < g1, say so plainly.

Both routes built. Interleaved 3+10 whole-op launches; separate interleaved
3+10 f/g half diagnostics. RED timings do not count. Census f then best green
g, 2+5 each; no eligible g => g1 TIMING-ONLY. Two foreground cells, 300 s work
cap / 590 s lease, no shortened samples; incomplete => INCONCLUSIVE.
Append `--dry-run` or `--run` to each line in `lead_commands.txt`.

Canonical GRAPA has four source differences from the historical parent;
old/current hashes are recorded in census registration. Checkpoint, tokens,
tokenizer and config are pinned. The f reproduction gate remains mandatory.
Amendment 001 fixes canonical symlink normalization; amendment 002 retains
best-green g even if f is RED, with TIMING-ONLY eligibility.

## Prior art

FlashAttention-2 / Tri Dao (2023): tiling, ownership and backward structure
taken; APA selective coefficient tiles/integration ours. FlashAttention /
Dao et al. (2022): softmax VJP and output-dot identity taken. NVIDIA CUDA WMMA
(2017; BF16 2020), CUDA events (2007+): fragment API and timing taken.
NumPy / Harris et al. (2020), pytest / Krekel (2004), SHA-256 / NIST (2001),
hash chaining / Haber and Stornetta (1991), POSIX/Python supervision and
BP-KERNEL-1/2 + BP-CENSUS-1/2 (2026): infrastructure taken; experiment-specific
integration ours. The exact gate/decision rule is user-specified; no prior art
known to me for that particular combination. All literature attributions are
**unverified — lead to check**; search those authors/titles plus "CUDA WMMA
BF16 16x16x16" and "Haber Stornetta 1991 hash chain". No network used.

## Scope and honest residuals

Model/effort: Codex Astra (`gpt-6-astra`), high, order-specified. No GPU, no git,
no subagents, nothing killed/signalled, no network or checkpoint writes.
All persistent project edits are within grants. Scope exception: two temporary
helpers were created in `/tmp` outside the named target, then moved into
`preparation_helpers/`. A literal no-outside-writes confirmation is not made.

GPU correctness, performance, real step and training quality are UNTESTED.
The half timings identify the slower half, and g1/g2 compares selective scalar
exact with dense exact MMA. Without further measurement, tiling-versus-MMA
cost attribution remains unresolved; no unsupported causal claim is made.
