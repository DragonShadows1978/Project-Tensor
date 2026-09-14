# BP-KERNEL-3 implementation ledger

2026-09-13 — Codex Astra (`gpt-6-astra`), reasoning high, order-specified.
Immutable plan: `orders/BP_KERNEL_3_TILED_TENSOR_CORE.md`. No edits to plan.
Status: CPU preparation in progress; native correctness and performance UNTESTED.

## Preparation and implementation

- Read HOUSE_RULES.md, local AGENTS.md, immutable order, parent result and live
  BP-KERNEL-1/2 source and registrations. No git, subagents, network or GPU probes.
- Saved byte baselines of kernels.cu, ops.cpp and bindings.cpp under
  `artifacts/bp_kernel_3/baseline/`. Copied the three expressly authorized NPZ
  files from canonical Project-Tensor. Their hashes are frozen in registration.
- g1 and g2: 16 query x 16 key tiles, padded D/VD stride 128, 128 threads / four
  warps. One block owns 16 query rows or 16 key rows. Four warps own disjoint
  gradient columns; warp 0 forms dense score products. Shared memory is 33,344
  bytes per block (reasoning from declared arrays). Full precision bounds are
  BF16 input, FP32 accumulators, BF16 output. D and VD in 1..128; other shapes
  fail explicitly. Nonmultiples of 16 zero-pad safely. Grouped heads accumulate
  inside a key owner; each gradient element has one final writer, no atomics.
- g1 exact scores: masked scalar FP32 loop using shared Q/K, only selected
  visible pairs. g2: full Q K^T WMMA tile, then selection. Both compute bulk
  Q KQ^T, dO V^T, selected and bulk dQ products, dK and dV on tensor cores.
  p and dS round to BF16 for outer-product inputs. These extra rounding sites
  are explicit; the unchanged imported BP-KERNEL-2 FP64 gate decides.
- Saved lse/thr, >= absolute bulk selection, bottom-right causal mask and
  output-dot row-dot retained. No fallback search or modified tolerance.
- Opt-in CUDA event diagnostics added to f and g, disabled for ordinary calls
  and whole-op interleaving. Separate 3+10 interleaved f/g1/g2 diagnostic
  launches report query and key halves. RED timings do not count.
- CPU model mirrors owner loops, padding, causality and grouped heads; tests
  compare both traversals to independent scalar dense math. This is author
  CPU logic evidence, NOT CUDA validation or a blind verification.

## Canonical GRAPA provenance finding / implementation amendment

The explicitly ordered canonical GRAPA harness exists. Its historical
registration names a pruned worktree, and its receipt is now `receipt.json.gz`.
Four canonical files differ from historical hashes: `corpus/factory.py`,
`grapa/evaluate.py`, `grapa/loss.py`, `grapa/train.py`. The census registers
both old and current hashes before gates; all live dependencies are pinned.
The existing checkpoint, tokens, tokenizer and model configuration remain
required and independently checked. Current loss adds optional weighting;
this harness calls the unweighted path. It does not run train.main, corpus
generation or evaluation. No historical-source equivalence is claimed. The
registered f reproduction within 10% remains mandatory. This implements the
order's explicit canonical-checkout direction without altering its predictions.
No read-only canonical files were edited or recreated.

## Prior art

- FlashAttention-2 / Tri Dao (2023): taken owner tiling, shared reuse,
  recomputation and backward accumulation structure. APA mixed-score selection
  belongs to Project-Tensor; selective coefficient tiles and this integration
  are ours. No novelty claim for ownership or tiling.
- FlashAttention / Dao et al. (2022): taken softmax VJP and output-dot identity
  through f and imported FP64 reference.
- NVIDIA CUDA WMMA (2017; BF16 2020), CUDA events (2007+): taken 16x16x16
  row/column-major fragment API, BF16 inputs / FP32 accumulation and event
  timing. Ours: g1/g2 routing and opt-in half receipt plumbing.
- NumPy / Harris et al. (2020), pytest / Krekel (2004), SHA-256 / NIST (2001),
  POSIX flock, Python subprocess timeout, BP-KERNEL-1/2 and BP-CENSUS-1/2
  (2026): taken CPU algebra/test infrastructure, create-only pinned artifacts,
  reference/gate and bounded foreground execution. Exact gate and selection
  verdict thresholds are the user's rule; no prior art known to me for that
  particular experimental rule.
- All literature dates/attributions: unverified — lead to check. Search terms:
  "FlashAttention-2 Dao 2023 backward", "FlashAttention 2022 Di dO O",
  "NVIDIA CUDA WMMA BF16 16x16x16", "NumPy Harris 2020", "pytest Krekel",
  "NIST SHA-256 2001". No network used in this seat.

## CPU build

`python artifacts/bp_kernel_3/build_driver.py`: CMake Release, CUDA 12.6,
SM 89 only, disconnected local pybind11, build-bk3, -j4. Exit 0, 60.036 s.
Stdout ended `[100%] Built target _tensor_cuda`. Stderr reported one unused
`lane` variable warning at kernels.cu:3056; it is retained verbatim in build.log.
`engine_build_receipt.json` records the combined-output last line (warning
remark), successful return codes and binary sha. No extension import or GPU
execution was performed by the build. Both template routes compiled.

## Amendment 001 and final CPU evidence

- Initial immutable registration sha256:
  `abac971ebca5d8e93324077b5b8634cc69a24863964ed01ecfbb589a79e23356`.
  Initial author baseline was **55 passed in 0.85s**; first kernel dry-run
  `dry_kernel_737075b5eb6a475aa4dabe20e1415671/receipt.json` completed.
- Census registration initially failed before creating its file because
  GRAPA's canonical path is a symlink; parent `Path.resolve()` returned a
  physical cold-storage path. Exact exception is retained in
  `artifacts/bp_kernel_3/amendment_001_note.md`. Fix uses canonical root plus
  relative checkpoint path. Original registration remains unchanged.
- Source-only amendment 001 sha256:
  `c9d16cb4c0a65ab42bbe79937eaab3ca3fbf3eda6d623dec6803158aaa248cd3`.
  Prediction, tolerance and protocol keys cannot be changed by this schema.
  Receipts bind both original and effective chain hashes. Prior art for this
  addition: hash chaining, Haber and Stornetta (1991), SHA-256 NIST (2001),
  taken; source-only amendment records ours. **unverified — lead to check**
  search "Haber Stornetta 1991 hash chain". No novelty claim.
- Census registration sha256:
  `6c8b473d9188b9cb0e93ab08132058db336a6dac909e28f312cd04881dff53f5`.
- Final author baseline: **56 passed in 0.86s**, command
  `PYTHONDONTWRITEBYTECODE=1 pytest -q tests/test_bp_kernel_3.py -p no:cacheprovider`.
  No receipt marks, skips or suppression directives were added. Eight schema
  defect cases and gate/decision boundary checks reject fabricated bad inputs;
  this is not a full source mutation campaign or blind verification.
- Kernel dry-run: `artifacts/bp_kernel_3/dry_kernel_c3b4dbbbc080499fadf4ebb9ded32397/receipt.json`.
- Census dry-run: `artifacts/bp_kernel_3/census/dry_c73e455ce17a44e9b3f1d65653a2f84c/receipt.json`.
  Both say DRY_RUN / CPU plumbing only. Their schemas validate. Native timing
  eligibility is false. The CPU f half diagnostics are placeholders generated
  with the tiled CPU model; they make no claim about f's native halves.
- Reverified all effective kernel pins, canonical census pins, checkpoint
  metadata, source baselines, binary and variant-a range. No GPU cell claims
  exist. The 129 MB FP64 reference hash is
  `43f6d06a21090adefe09da676e24cc1b785f7be07d9aae44b6e3fa06b19ad4d3`.
- CPU `cuobjdump` inspection confirms four g owner/route instantiations, each
  containing `HMMA.16816.F32.BF16`. See `mma_build_inspection.json`, `g_sass.txt`
  and `resource_usage.txt`. g1 query/key use 76/72 registers and 32,320 bytes
  shared; g2 query/key use 78/80 registers and 33,344 bytes shared. All four
  report zero stack and local memory. Compiler removes unused exact-score
  shared storage in g1; this refines the earlier declared-storage estimate.
  Instruction counts are static binary counts, not dynamic execution counts.
- Final source diff: kernels.cu +220/-0, ops.cpp +9/-2, bindings.cpp +5/-0.
  New drivers/tests: 296 / 237 / 217 lines. See `final_diff_stat.json`.
  Code sites: kernels.cu 3017–3218 (g), 3035–3047 (WMMA helpers), 3050–3169
  (owners), 3179–3218 (launcher); f diagnostics 2884–2885 and 2967–3013.
  ops.cpp 20, 1094–1097, 1117–1144; bindings.cpp 16–17, 932–934.
- Variant a bytes [107888,111834) unchanged, sha256
  `e7adc1e3442732b2fa221513ad75e2bdf665f62703f74b8b9c7410e0fb090a95`.
  Full prefix through the original launcher and original default VJP body
  also remain byte-identical, checked by the CPU test.

## Handoff and limits

Append `--dry-run` or `--run` to each line in
`artifacts/bp_kernel_3/lead_commands.txt`. Cell 1 runs first; cell 2 consumes
its complete gate and chooses the fastest green g, or g1 TIMING-ONLY if none
is eligible. Every whole-op arm uses 3+10 interleaved launches; f/g half
measurements use a separate 3+10 interleaving. Cell 2 is f then chosen g,
2+5 steps each, within the inherited foreground 300 s child cap and 590 s
lease budget. A timeout kills only the supervisor's own child. No timeouts
or process termination happened in this seat. No automatic background waits.

Both predictions remain UNTESTED:
- best green g ≤ 0.35 × f (≤ ~46 ms at f ≈ 131).
- whole step ≤ 2.5 s with g (f-arm reproduces ≈ 5.0 s within 10 %).
Secondary: g1 ≤ g2 (selection still pays on tensor cores); if g2 < g1, say so plainly.

No native correctness, speedup, real-step result or training-quality claim.
Half timings plus g1/g2 identify the slower half and compare selective scalar
exact to dense exact MMA. They do not causally separate tiling from MMA cost;
if the prediction fails without a clear selection difference, that requested
attribution remains unresolved rather than invented. Not claimed fixed.

Confirmations: no GPU, no git, no subagents, nothing killed/signalled, no
network, no checkpoint writes. All persistent project edits are within the
express grants. **Scope exception:** two temporary helpers were initially
created in `/tmp` (`bk3_kernel.txt`, `bk3_census_edit.py`), outside the user's
named target. They have been moved into the authorized artifact directory
`preparation_helpers/`. Thus a literal claim of no writes outside the grant
throughout the turn would be false; no such claim is made.

## Amendment 002 — final handoff revision

Subsequent source inspection identified a real route-selection edge case:
if f was RED but g2 was GREEN, census chose fallback g1 because route choice
was tied to eligibility. The corrected harness retains the fastest green g
from any complete cell while keeping f-RED comparisons TIMING-ONLY. A
synthetic regression test now covers it. Missing/incomplete/no-green cells
still use g1. No GPU gate has run; protocol/predictions unchanged.

`registration_amendment_002.json` binds the correction and test. Current
chain sha256 is `62beefb80710b8b7aeb3deeff8292aa9021067fe0de935849ba5a426db51af4b`.
The census's original immutable registration references an ancestor in this
verified source-only chain; each new receipt also binds the effective chain.
See `amendment_002_note.md` for rationale and prior-art annotation. All
historical registrations, amendments and CPU receipts remain unchanged.

Final revision author baseline: **57 passed in 0.85s**.
Final revision dry-run last lines:

```
DRY_RUN /mnt/ForgeRealm/wt/pt-bk3/artifacts/bp_kernel_3/dry_kernel_a16e08ec77dd4ecbba01ef6988b5ff5a/receipt.json
DRY_RUN /mnt/ForgeRealm/wt/pt-bk3/artifacts/bp_kernel_3/census/dry_c159fc2623cf4b4789ac2c0b105078f8/receipt.json
```

C++ sources and built binary have not changed since the successful build.
New driver/test line counts now 296 / 244 / 230; source diff remains
kernels.cu +220/-0, ops.cpp +9/-2, bindings.cpp +5/-0. Current evidence index:
`preparation_validation_revision_2.json`; current diff index:
`final_diff_stat_revision_2.json`. Earlier "final" entries above are preserved
as the preceding preparation revision, not overwritten. No further test
broadening or GPU activity was performed.
