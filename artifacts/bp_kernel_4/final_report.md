# BP-KERNEL-4 — CPU preparation complete; GPU verdict unmeasured

The lead's two GPU cells remain unclaimed. Build, FP64 reference, 48 CPU tests, both dry runs, and 11/11 mutation checks passed. No CUDA parity, speed or training-quality claim.

Source finding: checkpointed initial forward executes under NoGradGuard and canonical GRAPA routes it to the unchanged inference kernel; replay uses training forward. Thus h targets replay, and the order's assumption that both initial/replay use the training kernel is not supported by current source. Predictions remain unchanged. Census APA subscopes measure training forward only; a zero initial training-forward share does not imply zero inference APA cost. See census/source_route_receipt.json and docs/BP_KERNEL_4_LEDGER.md.

1. Variant h: 16x16 query/key tiles, 128 padded width, 128 threads. WMMA 16x16x16 BF16 matrix_a row-major, bulk matrix_b col-major (KQ transpose), P V matrix_b row-major, FP32 accumulator. Statistics pass accumulates visible |bulk| and |bulk| squared over ALL key tiles; threshold is mean + zthr*sqrt(max(second_moment-mean^2,0)). Second pass recomputes bulk, selects abs(bulk)>=threshold, computes only selected exact dots by scalar shared-memory loops, and updates online softmax and P V MMA. O BF16/lse FP32/thr FP32 layouts unchanged. h2 not built.

Code sites: tensor_cuda/src/kernels.cu:8027 (h), :8155 (guarded launcher/diagnostics), :8182 (explicit variant); tensor_cuda/src/ops.cpp:32 (forward switch), :760 (training integration/scopes); tensor_cuda/src/bindings.cpp:941 (bindings). CPU reference/gates scripts/bp_kernel_4.py:82/:138/:200; census scripts/bp_census_4.py:107/:184.

Diff stat from difflib: kernels.cu +177/-0, ops.cpp +21/-0, bindings.cpp +18/-0; scripts/bp_kernel_4.py +294, scripts/bp_census_4.py +254, tests/test_bp_kernel_4.py +139. Existing backward switch/closure preserved. Default forward kernel and launcher remain byte-identical:

- Forward kernel: 980f4ea882d728e7fa4af87ae6b6b31ea9c74107ab9ead0d0619e52a7e0c2517
- Forward launcher: 219c2bf1a1b01bedac235dd2b4b2bc6d482a0933ca5ee8420b3ab702e979eb85
- BP-KERNEL-1 fixed byte range: e7adc1e3442732b2fa221513ad75e2bdf665f62703f74b8b9c7410e0fb090a95

2. Last output lines (commands and exit codes also in cpu_validation_receipt.json):

```
48 passed in 3.90s
[100%] Built target _tensor_cuda
BUILD_RC=0
FP64_REFERENCE /mnt/ForgeRealm/wt/pt-bk4/artifacts/bp_kernel_4/reference.npz sha256=b46c7767b4c9c2e1ef9b45a92c6b4cf125636e5bdd05b6f0783a8628cbd503a3
DRY_RUN /mnt/ForgeRealm/wt/pt-bk4/artifacts/bp_kernel_4/dry_kernel_7ef6ed587d314f59a9256db1aa484b31/receipt.json
DRY_RUN /mnt/ForgeRealm/wt/pt-bk4/artifacts/bp_kernel_4/census/dry_da0c7c465fdf4898a0ce6d9150796718/receipt.json
```

Pytest used `-p no:cacheprovider` to avoid unrelated cache writes. Both lead_commands.txt lines were tested with appended --dry-run, overriding --run. Offline SM89 build took 61.216 s; only existing bk3 unused-lane warning. SASS confirms 28 HMMA instructions in each h specialization. CPU references/dry runs do not validate GPU execution.

3. Registration SHA256: 763db7c30f0b02d735a92f830e2e55fb884e7bd560bcbaf860d1ca7b095c04e4. No amendments. Predictions verbatim:

- h ≤ 0.20 × a-forward with all gates green.
- initial forward + checkpoint replay ≤ 0.25 s combined (from 0.90) and whole step ≤ 1.3 s.

h2 was not built. Registered falsifiers, .005 flip ceiling, downstream BP-KERNEL-2 tolerance, 3+10 microbenchmark, 2+5 census and two 300 s work / 590 s lease cells are unchanged. RED stops micro-timing; incomplete is INCONCLUSIVE.

## Prior art

4. FlashAttention (Dao et al., 2022) / FlashAttention-2 (Dao, 2023) forward: tiled ownership, recomputation and online-softmax rescaling taken. NVIDIA WMMA (2017; BF16 2020): fragment API taken through bk3. APA threshold integration into those tiles is ours; the threshold rule itself is shipped behavior. CUDA events/clock64 (2007+) and warp shuffle reduction (2013), NumPy (Harris 2020), pytest (Krekel 2004), SHA256 (NIST 2001), gprof (Graham/Kessler/McKusick 1982), pybind11 (Jakob 2015), mutation testing (DeMillo/Lipton/Sayward 1978), and BP-KERNEL-1/2/3 plus BP-CENSUS-1/3 harnesses (2026) taken. AdamW (Loshchilov/Hutter 2019) and checkpointing/PyTorch (Paszke 2019) inherited through unchanged GRAPA. The exact gate rule belongs to the order; no prior art known to me for that rule. External bibliography unverified — lead to check those titles/authors/years and "CUDA WMMA BF16 16x16x16"; no network used. Code comments and ledger carry the same provenance.

5. Assigned model/effort: Codex Astra (gpt-6-astra), reasoning high. Confirmations: no GPU, no git, no subagents, nothing killed or signalled, no network, no checkpoint writes, no edits outside grants. Foreground commands stayed below ten minutes. Original order/registration unchanged. Lead owns native gates and blind verification.
