# BP-KERNEL-2 ledger — 2026-09-13

CPU preparation COMPLETE; native correctness, a's distance from truth, f speed,
and the whole-step predictions are PENDING the lead's two GPU cells. No native
GREEN or speedup is claimed. This is an author baseline, not blind verification.
Order: `orders/BP_KERNEL_2_ATOMICS_FREE_DKDV.md`, unchanged and pinned in the
registration. Model/effort: Codex Astra (`gpt-6-astra`), reasoning high (assigned).

## Entry 1 — inspection and immutable source snapshots

Read `/mnt/Shared/HOUSE_RULES.md`, local AGENTS.md, the order, the parent result,
BP-KERNEL-1 source/registration/forward-state receipts, and the read-only
BP-CENSUS-1 driver and timing-hook artifacts. No git or hardware probe was used.
Original four edited source files were copied to `artifacts/bp_kernel_2/baseline/`
before edits. The source patch and diff stat are ordinary Python difflib outputs.
No checkpoints were written; no read-only GRAPA files were changed.

Evidence: `registration.json` sources and pins; `source.patch`; `diff_stat.json`.
All paths below without an absolute prefix are relative to
`/mnt/ForgeRealm/wt/pt-bk2/artifacts/bp_kernel_2/` unless explicitly identified.

## Entry 2 — native implementation (source inspection / CPU compilation)

`tensor_cuda/src/kernels.cu`:
- Lines 2721–2839: reuse BP-KERNEL-1 c/d query kernel. Add optional FP32 rowdot
  destination (2728, 2787), preserving its existing dQ math. The output-dot
  route uses its existing OUTPUT_DOT branch; fallback uses existing Pass A.
- Lines 2881–2955: f's key-owner pass. One 128-thread block per batch/KV-head/key.
  Threads stride visible queries and grouped query heads, recompute mixed scores
  from frozen lse/thr, accumulate dK/dV in FP32 register arrays, reduce through
  warp/shared memory, and cast/write each output once. Causal lower query bound
  is `max(0,j-(S-L))`; unselected Kq remains detached.
- Lines 2957–2998: f launcher. Reused query pass has WRITE_KV=false (both atomic
  sites compile out), followed by the key-owner kernel. Only O(BHL) FP32 rowdot
  scratch; no global dK/dV reduction scratch and no atomics on either f route.
  Pre-pass is fused into the query launch, avoiding a duplicate Pass A.

`tensor_cuda/src/ops.cpp`: lines 15–28 explicit thread-local setter/getter and
forward declaration; 758–768 opt-in VJP branch captures saved O and chosen
variant; 1094–1138 validated f/f_pass_a dispatch. Original default-a VJP body
remains present verbatim. No environment-controlled implicit dispatch was added.
`tensor_cuda/src/bindings.cpp`: 929–931 explicit setter/getter bindings.

Original kernel and launcher prefix, including the pinned range, is unchanged.
Pinned bytes [107888,111834), lines 2523–2629, sha256:
`e7adc1e3442732b2fa221513ad75e2bdf665f62703f74b8b9c7410e0fb090a95`.
This is a source-byte claim; no GPU or binary-equivalence claim is made.

Native source diff: kernels.cu +120/-1; ops.cpp +32/-2; bindings.cpp +30/-0;
autograd.cpp +9/-0, total +191/-3. The copied timing header adds 88 lines.
`timing_hook_copy.json` records hashes of the read-only BP-CENSUS-1 artifact
copies, used because the branch-named worktree was absent. Header copy:
`fcdcd19bcbb5e2935c8cbb898bd47a802456d1da65f1d0748fa955124932b5da`;
autograd.cpp copy:
`833bbf8c6aa9b308b655dbc54e6908a79fe5db5702c5bb7f207f9f860487e188`.
Bindings timing hunks were copied into the current BP-KERNEL-1 bindings, retaining
its existing variant API. These are the explicitly authorized timing-hook edits.

## Entry 3 — reference, registration, and rowdot decision (CPU evidence)

`python3 scripts/bp_kernel_2.py --reference` ran on CPU after create-only
`reference_registration.json`. Inputs and saved state were checked against the
parent pins before computation. FP64 reference uses the C++ float scale promoted
to FP64; frozen saved lse/thr; materialized FP64 bulk-selection mask (`>=`, signed
bulk magnitude); exact selected score; no probability renormalization; bottom-right
causal mask; full dQ/dK/dV with grouped heads and detached Kq. Row chunks bound
intermediate storage, without changing the registered full geometry or samples.

Reference: `reference.npz`, sha256
`43f6d06a21090adefe09da676e24cc1b785f7be07d9aae44b6e3fa06b19ad4d3`.
Receipt: `reference_receipt.json`; 2.281081500928849 seconds CPU; mask contains
5,203,755 selected entries. It stores full FP64 dQ/dK/dV/rowdot and the Boolean mask.

**a's GPU distance from the FP64 reference: NOT MEASURED IN THIS SEAT.**
Cell 1 records it prominently as `correctness.a_distance_from_reference`, with
max_abs and relative_L2 for each array. Every tolerance is exactly twice that
fresh a distance. Zero means zero; no epsilon. Nonfinite metrics are RED.

Registration written before tests/dry-run gates:
`registration.json`, sha256
`18adcb9fa30aeba70e3e170f91135fd5f26013e3dfde75d4b5935d6264765854`.
It pins reference, input/state, before/after sources, binary, order, scripts,
tests, build receipt and lead commands. Runtime validation fails closed on drift.
Census registration: `census/registration.json`, sha256
`3d9eb6c78ff748f54b381d46cc39ac8aaac7999437fe608fa360f2254a075b80`.
All registrations and receipts remain immutable; future changes need amendments.

f route is a registered gate decision, not selected here from CPU numbers:
cell 1 probes output-dot f and gates dQ; if dQ is RED, switches to `f_pass_a`
and re-gates all three gradients before timing. Otherwise retains output-dot.
If final f is RED, no f micro-timing runs (registered falsifier); b/d raw timing
is ineligible when RED. A RED f still proceeds to census as TIMING-ONLY / NOT A
VALID STEP. Census pins the cell-1 receipt in create-only
`census/gate_input_registration.json` before its child starts. Incomplete cell 1
uses Pass A and the same invalid-step label. No route is selected by speed.

## Entry 4 — CPU compilation and author validation

Build recipe: `python3 artifacts/bp_kernel_2/build_driver.py`, CMake Release,
sm_89, CUDA 12.6 nvcc, local pybind11 source with FetchContent fully disconnected,
`tensor_cuda/build-bk2`, `-j4`. CPU compilation only, 59.41800695192069 seconds.
Receipt: `engine_build_receipt.json`; full output: `build.log`.
Binary sha256: `6fd610a50af29bac17ad2f4407d1301dc855c6354639eff4852bce9dff536cb4`.
Last line: `[100%] Built target _tensor_cuda`.

Tests command:
`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 python3 -m pytest -q -o cache_dir=artifacts/bp_kernel_2/pytest_cache tests/test_bp_kernel_2.py`
Last line: `23 passed in 2.54s`.
Coverage: independent scalar scores and explicit dense softmax Jacobian oracle;
causal/noncausal, square/non-square, grouped heads, multiple batches, selection
all/none/equality/negative-bulk edge, frozen lse without renormalization; fabricated
gate boundary, zero and nonfinite distances; original source bytes/default body;
f's no-atomics structure; source drift rejection; create-only NPZ/JSON; receipt
schemas and incomplete/RED census semantics. No CUDA import/execution in tests.

Kernel dry-run last line:
`DRY_RUN /mnt/ForgeRealm/wt/pt-bk2/artifacts/bp_kernel_2/dry_kernel_373db6aa220d4f43a87b0a26442ad8ca/receipt.json`
Census dry-run last line:
`DRY_RUN /mnt/ForgeRealm/wt/pt-bk2/artifacts/bp_kernel_2/census/dry_16cdc647400644b492430b548e0a618c/receipt.json`
Both used `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=2 python3`.
Tiny kernel simulation retained output-dot after dQ passed, then correctly marked
f RED on dK and skipped its timing. This is CPU plumbing evidence only, not a
prediction about CUDA f. Census executed 2 warm-ups + 5 measured CPU steps in each
arm. Its CPU adapter does not execute either native variant; no native validity,
performance, or training-quality claim follows.

## Entry 5 — lead handoff / real-step status

Part (iii) RUNNABLE, pending native execution. No minimal GRAPA diff is needed.
The new thin driver imports the unchanged parent profiler/CPU adapter and GRAPA
objects; explicitly preloads the pinned pt-bk2 engine; verifies the engine path,
binary and timing fingerprint; loads the same checkpoint independently per arm;
uses the parent's fixed tokens/config/optimizer metadata; instruments both arms
in a then f order. Neither harness writes a checkpoint.

`lead_commands.txt` contains the two absolute commands, one per line; append
`--dry-run` to either for the CPU path. Each native command owns a separate
create-only cell claim and receipt, acquires nonblocking `/tmp/forge-gpu.lock`,
and starts one foreground child with a 300-second work timeout. The parent holds
the lock through child completion/termination; lease budget is 590 seconds.
Only that newly started child can be terminated by the harness on timeout.
The dispatch seat ran neither native command and killed/signalled nothing.
Incomplete samples are INCONCLUSIVE; shapes and sample counts are never shortened.

OFF control is not repeated; the order and BP-CENSUS-1 receipt report 1.3% overhead.
Census reports a reproduction within 10% of 11.8 s separately from f <=5 s, and
its full component table names the remaining time. Seeded kernel inputs do not
certify checkpoint gradients; the census is step timing, not model-quality evidence.

## Prior art

- FlashAttention-2 backward / Tri Dao (2023): key-owned dK/dV and query
  recomputation taken. Warp/shared reductions use NVIDIA CUDA conventions
  (2007 onward). APA mixed-score selection belongs to the existing project;
  our change integrates it with key ownership, existing c dQ, and saved rowdot.
  No claim of inventing atomics-free attention backward or tensor-core tiling.
- FlashAttention / Dao et al. (2022): output-dot identity `D_i=dO_i dot O_i`,
  recomputation and softmax VJP taken. Our contribution is this integration and
  explicit BF16-output reference gate/fallback, not the identity. Standard
  `diag(p)-p p^T` Jacobian is taken; no prior art known to me for its first
  author/year, so no priority attribution for that first derivation is asserted.
- NumPy / Harris et al. (2020): dense FP64 products taken; BP-KERNEL-1 (2026)
  BF16 fixtures/metrics/events/supervision and BP-CENSUS-1 (2026) profiler,
  unchanged GRAPA driver operations and CPU adapter taken. Checkpoint/autograd
  adapter uses PyTorch / Paszke et al. (2019), AdamW / Loshchilov and Hutter (2019).
- gprof / Graham, Kessler, McKusick (1982): exclusive call-tree accounting;
  NVIDIA CUDA events (2007 onward), pytest / Krekel (2004), SHA-256 / NIST (2001),
  Python difflib and POSIX flock/subprocess deadlines: existing infrastructure
  taken. Ours is the order-specific plumbing, schema and evidence checks.
- No prior art known to me for the exact `2 * |a-reference|` rule or this exact
  registration schema. They implement the user's registered policy and are
  not presented as a new error estimator or methodology.

All external attributions are **unverified — lead to check**. Search terms:
“FlashAttention-2 backward Dao 2023 dK dV”; “FlashAttention 2022 Di dO O”;
“softmax Jacobian diag p outer p”; the author/system/year names above. No network
was used. Prior-art comments are present at kernel, reference, gate, driver,
profiler-copy and oracle code sites.

## Execution attestations and residuals

No GPU; no git; no subagents; no network; no checkpoint writes; nothing killed or
signalled. All writes stayed in the explicitly granted source/harness/test/artifact/
ledger/build paths or `/tmp` (temporary edit driver). The timing header/autograd
copy and rebuilt extension are covered by the order's explicit grants.
Remaining work belongs to the lead: both bounded GPU cells, blind verification,
interpretation, and commits. No prediction has yet been confirmed or falsified
on CUDA. Not claimed fixed: native correctness, native speed or training quality.
