# BP-KERNEL-1 implementation ledger — 2026-09-13

**CPU BUILD / AUTHOR UNIT TESTS GREEN. GPU CENSUS UNRUN; atomics prediction INCONCLUSIVE.**
Seat: Codex Astra (`gpt-6-astra`), reasoning high, as specified by dispatch; independent launch metadata is not exposed. Lead: Fable 5.1. Immutable plan: user's BP-KERNEL-1 order dated 2026-09-13, including its copied prediction/falsifier. This ledger records implementation and limitations; it does not rewrite that order.

## Entry 1 — read-only diagnosis and baseline preservation

Evidence class: source inspection and historical receipts, no new GPU measurement.

Read `/mnt/Shared/HOUSE_RULES.md`, worktree `AGENTS.md`, BP-SCOUT-1's “What native APA actually computes” and idea #1, `/mnt/Shared/BP_Census_1_Result_2026-09-13.md`, the parent census receipt, and `grapa/model_mla.py`, `attention_mla.py`, `attention.py`. Read the local `gpu_lease` implementation in GraftRepository. A lightweight memory lookup supplied only the reminder to verify current source and keep a kernel result separate from model quality; all geometry/source facts here were read live.

Verified the three pre-edit source files against `/mnt/ForgeRealm/Project-Tensor/tensor_cuda/src`: all byte-equal. Saved immutable copies under `artifacts/bp_kernel_1/baseline/` and before hashes in `baseline_pins.json`. No git operation was used; the branch/base commit identity is the order's attestation, not a new git verification.

The protected original kernel region is **bytes [107888,111834)**, zero-based, end-exclusive; **lines 2523–2629** including template and trailing whitespace. SHA256:

`e7adc1e3442732b2fa221513ad75e2bdf665f62703f74b8b9c7410e0fb090a95`

The whole prefix through both original training launchers remains identical. The complete original ops.cpp is an unchanged prefix, preserving the training closure. The original Python backward binding's region is unchanged; its line numbers moved because local declarations were added above it. Tests verify these properties against the saved main source, not by invoking git.

Geometry: **B=1; H=KVH=16; L=S=2048; D=96; VD=64; BF16; causal**. D is 64 non-RoPE + 32 RoPE. MLA expands K/V to all 16 heads; latent compression does not imply KVH=1 in this op. Scale is `1/sqrt(96)`. Refine .15 maps to `_norm_ppf(.85) = 1.0364333894937898`; threshold is visible-key `mean(abs(bulk)) + zthr * sqrt(max(mean(abs(bulk)^2)-mean(abs(bulk))^2,0))`. The parent's effective config records checkpointing, BF16, d_model=1024, 24 layers, vocab=8192, window=2048; the attention files establish the remaining dimensions.

## Entry 2 — implementation

Evidence class: source implementation; compiled but not GPU executed.

| Variant | Change and sites in current source | Gradient status |
|---|---|---|
| a | Original kernel `kernels.cu:2523–2629`, launcher `2670–2712`; original default binding and autograd closure untouched | Control |
| b | Separate `apa_selective_bwd_bk1_kernel` instantiation, `kernels.cu:2722–2837`; FP32 dK/dV atomic updates at 2805–2818; scratch/one final cast at 2840–2879 | Candidate, must pass gate |
| c | Separate `WRITE_KV=false, OUTPUT_DOT=false` instantiation at 2868; both KV atomic paths compile out; dQ math remains; same output allocation/zeroing dtype as a at 2847–2850 | TIMING ONLY, zero returned dK/dV are placeholders |
| d | Separate `WRITE_KV=true, OUTPUT_DOT=true` instantiation at 2869; replaces Pass A with saved BF16 `dO dot O` at 2759–2764, on top of b | Candidate, must pass gate |

The variants share a source template but compile as separately selectable CUDA kernels. Selection requires the explicit `variant` string in new binding `apa_selective_bwd_variant` (`bindings.cpp:857–871`); validation/dispatch is appended at `ops.cpp:1066–1108`. No default variant argument was added to the original binding, no default training graph is changed, and no new header was needed. Variant e was optional and is not implemented.

Source-only diff: `kernels.cu +167/-0`, `ops.cpp +44/-0`, `bindings.cpp +24/-0`, total **+235/-0**. Reviewable patch: `artifacts/bp_kernel_1/source.patch`; machine stat: `diff_stat.json`. New harness: 407 lines; new author tests: 195 lines. Artifact binaries/NPZ and baseline snapshots are separate from this source-only stat.

The harness loads the built extension by its exact pinned absolute filename, rather than resolving a global installed engine. It uses actual CUDA runtime events around the bare backward binding, synchronizes before/after, and keeps outputs alive through the end event. Timing includes the op's output allocation/zeroing and b/d final casts; it excludes forward, correctness host copies and fixture generation. The c ablation removes both atomic stores and their associated arithmetic/register effects, so the a–c difference is an ablation bound, not a hardware instruction counter.

Prior art at implementation sites: NVIDIA CUDA FP32 atomic accumulation; Dao et al., FlashAttention (2022), recomputed softmax VJP/output-dot identity; existing Project-Tensor NDArray/pybind conventions (2026). Taken: those primitives and identities; ours: adaptation to the detached-Kq APA contract and explicit micro-census selection. External references **unverified — lead to check**; search terms appear at code sites and below.

## Entry 3 — build, before registration/gates

Evidence class: CPU compilation only. Executed the exact authorized configure flags and `cmake --build tensor_cuda/build-bk1 -j4`, foreground, disconnected, no device execution. Both commands rc=0; elapsed 53.170744695235044 seconds. Command arrays, return codes, binary size/hash are in `artifacts/bp_kernel_1/engine_build_receipt.json`; complete output is `build.log`. Build helper is `build_driver.py` in the authorized artifact directory.

Last CMake build line: `[100%] Built target _tensor_cuda`

Built binary: `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`, 15,070,624 bytes, SHA256:

`bf6e25bd8d31bf9253b696dbde6d8d2a556c4cacd214cef748267431d955b16d`

Tool transport yielded a session while the foreground build ran; it was immediately followed to completion. No shell job was backgrounded, no background sleep/wait was launched, and no other work was run concurrently with the build.

## Entry 4 — frozen registration and explicit two-stage forward-state pinning

Evidence class: CPU fixture production and create-only registration, before tests or dry-run gates.

`artifacts/bp_kernel_1/registration.json` SHA256:

`fbddb4422d3932f21b82ad54328bc2049094c77a6711d8c59b39b8f63a2ed643`

`registration.sha256` pins the exact registration bytes. The harness verifies registration bytes, source before/after hashes, baseline pins, its own code, tests, engine receipt/binary, lead command, input NPZ and external geometry sources. Any drift fails closed. Registration was created once and has not been amended or overwritten.

Registered input fixture: seed 20260913; NumPy PCG64 normal FP32 draws, rounded ties-to-even to BF16 and stored losslessly as FP32 in `inputs.npz`. q/k/kq shapes `(1,16,2048,96)`; v/dO `(1,16,2048,64)`. Kq is seeded `BF16(k + .125 * normal)`, a synthetic correlated detached approximation, not GRAPA's real quantizer or checkpoint activations. This is a kernel-shape census as ordered; no real-model gradient distribution is claimed.

**Explicit pending item / order interpretation:** this seat is forbidden GPU execution and therefore cannot precompute real CUDA-forward lse/thr/O. `inputs.npz` currently contains q/k/kq/v/dO only. In the lead's only cell, the real training forward generates out/lse/thr once; the harness writes create-only `forward_state.npz` and `forward_state_registration.json` with its SHA256 BEFORE any backward correctness gate. Original registration/inputs never change. This is two-stage pinning, not a claim that a pre-run all-in-one NPZ already contains real forward state. The registration names this deviation for lead review. No CPU-generated statistics are presented as the real CUDA statistics.

Tolerance implemented verbatim from registration:

> For each of dQ/dK/dV, max_abs and relative_L2(candidate,a1) must each be <= 2 * the same metric(a2,a1). No additional absolute or relative tolerance. a1/a2 and their spreads must be finite; zero spread gives zero tolerance. relative_L2 = ||x-ref||_2 / ||ref||_2; zero/zero=0, nonzero/zero is nonfinite and RED.

The two a runs precede b/d gates, all warmups and all measurements. Any failing gradient or metric marks that candidate RED; its raw timing can be retained but `timing_counts=false`, and the secondary d claim is ineligible. The a/c timing-only prediction remains decidable if all samples and finite control checks complete. Zero dQ nondeterminism can make d fail despite small output-rounding error; no epsilon floor was introduced.

Immutable prediction: **time(a) − time(c) ≥ 50 % of time(a) — the atomics own the kernel.**

Immutable falsifier: **< 50 % promotes the scalar dot loops (tiling / tensor cores, variant e) as the first target.**

Secondary: **d ≤ 0.6 × a if the prediction holds and d is green.**

Timing is 3 complete warmup rounds then 10 complete measured rounds in a,b,c,d order. No shortening, skipping RED variants, retry, or tuning. Nonblocking `/tmp/forge-gpu.lock` flock, single visible device, create-only cell claim, foreground synchronous subprocess with 300 s timeout; parent retains lock. Only its own child may be terminated on timeout; incomplete results are INCONCLUSIVE. The 300 s worker timeout also bounds ordinary execution below the 590 s lease cap. A second attempt after claiming the cell fails closed even if the first did not finish. Second cell requires a lead order.

## Entry 5 — CPU verification

Evidence class: author unit tests, not a blind red-team or GPU gate. After registration, ran:

`PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS='-p no:cacheprovider' python3 -m pytest -q tests/test_bp_kernel_1.py`

rc=0; exact last line: **`22 passed in 0.76s`**.

Coverage includes registration overwrite/drift refusal, source pins, protected a/default regions, finite/nonfinite/zero spread rules, individual bad gradient, zero denominator, exact 50%/60% boundaries, RED secondary exclusion, incomplete/invalid timings, create-only JSON/NPZ, shared CPU orchestration, all 52 scheduled launches, dry-run receipt schema and CLI precedence, deterministic BF16 fixtures and budget expiry. Tests count all three gradients; an empty zip cannot pass vacuously. No test asserted a candidate must be GREEN.

Ran:

`PYTHONDONTWRITEBYTECODE=1 python3 scripts/bp_kernel_1.py --dry-run --output-dir artifacts/bp_kernel_1/dry_run_final`

rc=0; exact last line:

`DRY_RUN receipt=artifacts/bp_kernel_1/dry_run_final/receipt.json`

The deterministic CPU simulator produced finite zero a run-to-run spread, so b and d are **RED in the CPU simulation** under the unchanged zero tolerance. This demonstrates gate behavior, not CUDA failure: b dK max_abs .001953125 and dV .015625; d dQ .00390625, dK .0078125, dV .015625. All CPU timings are marked ineligible; overall verdict DRY_RUN. It uses tiny GQA/unequal-width shapes through the same orchestration, not the registered full-size GPU shapes. The real root `receipt.json`, `cell_claim.json` and real forward-state files remain absent: zero GPU cells consumed.

## Entry 6 — handoff and residuals

One lead command is in `artifacts/bp_kernel_1/lead_commands.txt`. Appending `--dry-run` overrides `--run` before any lock, extension import or GPU work. The only live command is:

`PYTHONDONTWRITEBYTECODE=1 python3 /mnt/ForgeRealm/wt/pt-bk1/scripts/bp_kernel_1.py --run`

Lead reviews the explicit two-stage forward-state pinning, verifies the artifacts, runs the sole GPU cell, then owns commit/interpretation. The author has not dispatched any verifier. GPU kernel numerics, CUDA event runtime behavior, lock/timeout execution on a real GPU and the atomics prediction are untested in this seat. **Not claimed fixed:** training throughput, model quality, or native gradient precision. No optimization is promoted by these CPU results.

No GPU, git, subagents, network, service changes, or process kills/signals were used. Product edits are only the three authorized source files. Deliverable writes are only the named harness/test/ledger/artifact paths, plus authorized CPU build output and temporary scripting/test files under the environment's writable `/tmp` grant. No edits to the read-only GRAPA/main/source trees; no memory updates.

## Prior art

All external bibliographic attribution is **unverified — lead to check** because this seat used no network.

- **Dao et al., FlashAttention (2022)**: take backward probability recomputation, softmax VJP and `D_i = dO_i dot O_i`. Ours is the explicit adaptation to native mixed-score APA with detached Kq, D=96/VD=64 and a saved BF16 output; no new identity claim. Search `FlashAttention backward Di sum dO O 2022`.
- **Atomics-free dK/dV via key-parallel backward passes**, FlashAttention (Dao et al., 2022) and related attention work partitioning: related prior art, **not implemented here**. c deletes dK/dV writes and cannot be called a valid key-parallel algorithm. Search `FlashAttention backward key parallel dK dV atomics`.
- **NVIDIA CUDA system** (exact documentation year unverified): take FP32 `atomicAdd`, wider accumulation with one final cast, and CUDA-event timing. Ours is experiment plumbing/selection and the deletion ablation. Search `CUDA BF16 FP32 atomic accumulation`, `CUDA events elapsed time`.
- **NumPy PCG64 (2019); O'Neill PCG (2014); IEEE round-to-nearest-even**: take seeded RNG and standard rounding for reproducible fixture bits. **No prior art known to me for this exact synthetic Kq fixture choice**; no novelty claim. Search `NumPy PCG64 Generator 2019`, `ONeill PCG 2014`, `bfloat16 round nearest even`.
- **pytest, Krekel et al. (2004; year unverified)** and boundary/adversarial testing: take conventional test infrastructure; ours is the registered metric/verdict/create-only invariant set. Search `pytest Holger Krekel history`.
- **Local GraftRepository gpu_lease convention and Project-Tensor NDArray/pybind dispatch (2026 source)**: take flock exclusion and existing wrapper structure; ours is one-shot census wiring with a synchronous child timeout. These local source attributions were read, not fetched externally.
