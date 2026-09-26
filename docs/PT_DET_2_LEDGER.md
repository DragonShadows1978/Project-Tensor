# PT-DET-2 implementation ledger

Status: registered; implementation pending. GPU certification BLOCKED by order.

Immutable plan: `orders/PT_DET_2_GATHER_BWD.md`. House rules read. No git/subagents/GPU/production lock operations. All compiler scratch, caches and outputs are confined to this fork. Branch/HEAD are operator-supplied identity, not verified with git.

## Registration before gates

`artifacts/pt_det_2/REGISTRATION.json`, SHA256 `f39756c6bf261d4be5595d255797ad81cb7f2b21c329afe3600a6f68e1d49675`. No gates run before registration. Baselines copied before editing; exact hashes are in the registration. PT-DET-1 Amendment 004 is separate; prior orders, registrations and manifests remain intact.

## Premise correction

Evidence class: static source plus address reasoning. `grapa/loss.py:22` gathers one class per distinct row. Destination `(b*L+t)*V+target[b,t]` is injective across rows even when class IDs repeat. This agrees with `docs/PT_DET_1_AUDIT.md`; the order's collision premise for the current loss is incorrect. General gather collisions remain a real unordered-atomic site. Registered cases cover true K=1 loss geometry and K=4 deliberate same-row collisions, separately labeled. Full-step success remains unmeasured.

## Prior art

CUB/Merrill/NVIDIA (2011 onward; CUDA 12.6 used, 2024): stable sorting and sorted segmented scatter reduction taken. Installed `device_radix_sort.cuh:109` documents stability. PyTorch 1.9 (2021) opt-in deterministic indexing policy taken; primary release page verified. Demmel and Nguyen (ARITH 2013), Fast Reproducible Floating-Point Summation, primary paper verified for context; no order-independent accumulator used. PT-DET-1/CC46/PT-TF32 (2026), NIST SHA256 (2001), POSIX process/deadline mechanisms reused. This change supplies general flat-destination mapping and shared family integration, not a new sorting or summation algorithm. Code comments and final report will carry these boundaries.

## Registered slot

Embedding 80 s + gather 80 s + all eight 30-step replay processes 1320 s + finalization reserve 20 s = 1500 s maximum. Stop after first non-GREEN lane. The workload fitting this cap is unmeasured; timeout remains BLOCKED and no step count is reduced. CPU tests/compilation are author baselines, not blind verification.

## Implementation and build 02

Evidence class: source inspection and CPU-only compilation. Added general stable destination sorting and a single fixed-order FP64 owner per segment, sharing PT-DET-1 scalar arithmetic. `TC_DETERMINISTIC` takes precedence when present, with the legacy environment fallback; both API names share TLS state. Added top-level Python aliases, raw full-dispatch gather hook, CMake source, CPU contracts and one sequential slot. The original gather/embedding atomic paths are preserved.

Build receipt: `artifacts/pt_det_1/build_02/receipt.json`. Verbatim: `BUILD_RC=0 SOURCE_PINS_UNCHANGED=True RECEIPT=/mnt/ForgeRealm/wt/pt-tf32/artifacts/pt_det_1/build_02/receipt.json`. Existing unused-lane and nvlink incompatible archive warnings remain, unsuppressed. Both host contract executables linked against the rebuilt objects. No GPU use.

## Broad CPU baseline: retained historical-pin failures

Evidence class: author CPU tests, `artifacts/pt_det_2/cpu_tests_01/pytest.log`. Verbatim: `2 failed, 173 passed in 5.34s`; `PT_DET_2 CPU_TESTS_RC=1`. All PT-DET-1/PT-DET-2 checks passed. Two older PT-TF32-4 certificate tests reject stale source seals: `ValueError: source drift: /mnt/ForgeRealm/wt/pt-tf32/tensor_cuda/tensor_cuda/__init__.py` (optional model registration) and `ValueError: source drift: scripts/pt_tf32_4.py` (historical manifest with the additive certification adapter not enabled). These historical registrations will not be rewritten or assertions weakened. The new additive seal will serve the intended combined certification route. The optional old model registration requires a separately authorized fresh registration; not claimed fixed here.

## Amendment 001: bitwise loss evidence

Static finding: inherited CC46 probe has full-precision gnorm but only rounded trainer text for loss. The new order asks for bitwise losses. Before testing the new capture, separate `AMENDMENT_001.json` registers 30 finite scalar dtype/shape/byte digests and float.hex values through an in-memory loss wrapper. This adds a common synchronization before backward in all arms. No trainer file changes. Existing row/norm/checkpoint gates remain mandatory, thresholds unchanged. Prior art: reuse CC46 full-precision scalar probes and SHA256 receipts; local addition is loss capture.

## Amendment 002: nested timeout ownership

Evidence class: static harness review, not an observed leaked process. An outer slot timeout could kill a replay script while its trainer owned a separate session. Before safety tests, Amendment 002 registers a shared outer-owned process group for nested training and an inherited absolute deadline. Inner cleanup signals only its created child; the outer owner can terminate its whole child group. Standalone replay retains independent child groups. No production lock actions or outside processes are involved. Prior art: POSIX process groups and monotonic deadlines, PT-TF32/PT-DET-1 ownership discipline.

## CPU implementation checks and full model-shape arithmetic

Evidence class: author CPU baseline. `cpu_tests_02/pytest.log`: `108 passed in 3.80s`, `PT_DET_2 CPU_TESTS_RC=0`. Full registered shape arithmetic: `cpu_model_receipt/gather.log`, `cpu_model_shapes/summary.json`. Six cases x five calls all byte-identical; maximum FP64 relative L2 `2.2841852151174615e-08`. K=1 errors are zero; K=4 errors are nonzero FP32 rounding within the registered <=1e-6 bar. This is CPU evidence, not CUDA timing or model replay.

| Batch | True loss K=1 rel-L2 | Colliding K=4 rel-L2 | Five CPU calls each |
|---|---:|---:|---|
| x_32055 | 0.0 | 2.280637833220592e-08 | byte-identical |
| x_32083 | 0.0 | 2.284185215117462e-08 | byte-identical |
| x_32110 | 0.0 | 2.179560985602261e-08 | byte-identical |

## Final seal and verification

Active additive seal: `artifacts/pt_det_1/SOURCE_MANIFEST_003.json`, SHA256 `792ed96fd492b80d314e695aaccaeb027d612cf12a93dd80909224e83b90d5ab`. PT-DET-1 old seals remain unchanged; registration SHA256 remains `0b1af9f61695c5cf14ae6b0ecedac255dfebc4f18ac9821ed250751d79d48033`. PT-DET-2 order SHA256 remains `51cf309c7b497f8d7af575e4392216300d0898b5c197ae9ce0498d00380cda6f`.

Evidence class: author CPU suite plus active provenance verification. `final_cpu/pytest.log`: `116 passed in 4.60s`; `PT_DET_2 FINAL_CPU_RC=0`; `PT_DET_2 FINAL_MANIFEST_OK 792ed96fd492b80d314e695aaccaeb027d612cf12a93dd80909224e83b90d5ab`. Exact argv and environment are saved in `final_cpu/receipt.json`. This includes the same eight affected PT-TF32 certification checks used in the prior PT-DET-1 final suite, now against the new additive seal through `PT_DET_1_CERTIFICATION=1`. No assertion is skipped or relaxed; the unrelated historical model-pin failure remains named above.

Final handoff receipts: gather, embedding, repro, slot each returned code 2 as expected with CUDA hidden. Verbatim guard reason: `BLOCKED: lead GPU slot required; CUDA_VISIBLE_DEVICES is empty or --lead-gpu absent`. Sequence and plan each returned 0. `final_handoff/receipt.json` contains exact argv/results; guard ordering prevents any production descriptor inspection here.

## Done

**Implemented, rebuilt, and CPU checked in the fork; GPU certification BLOCKED.** Original training divergence and <=2x GPU cost are **not claimed fixed**. All source pins match build 02. Binary: `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`, 10457288 bytes, SHA256 `69b154cd2e9a5a7cbe288e1419d963697e5add57d2b4ee827c4c78a273ff2c5c`. Build elapsed 67.968 s; inherited compiler warnings were retained.

Current blocked report: `artifacts/pt_det_2/BLOCKED_REPORT.json`. Static audit: `artifacts/pt_det_2/SOURCE_AUDIT.json`. Shared narrative continued in `docs/PT_DET_1_REPORT.md` with the old report preserved below the new update. No GPU, live-engine edit/build/import, git, subagents, background waits, or production-lock operation. No process outside the task's own children was signaled. Compiler temporaries/caches/pytest outputs were explicitly confined to this fork.

The lead runs this only in an authorized exclusive GPU slot, with the already-held
lock inherited as FD 9 and a fresh output directory. The harness does not acquire,
wait on, or release the lock. This seat never accessed it.

```bash
cd /mnt/ForgeRealm/wt/pt-tf32
CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  python -B scripts/pt_det_1_slot.py --lead-gpu --lock-fd 9 \
  --out artifacts/pt_det_1/lead_slot_det2_01
```

Sequential budgets: embedding **80 s**, gather **80 s**, all four paired 30-step
replays **1320 s**, finalization reserve **20 s**; global cap **1500 s**. The
first non-GREEN lane blocks every later lane. Nested trainers share the outer
child's process group and inherit an earlier absolute deadline, so an outer
timeout can clean up its descendants. No step-count reduction is permitted.
This is a cap, not a measured completion estimate. The printed sequence is
`artifacts/pt_det_2/final_handoff/sequence.log`.

## Final prior art record

- **CUB / Duane Merrill / NVIDIA (2011 onward; CUDA 12.6 used, 2024):** stable
  radix sorting and sorted segmented scatter-add are taken. The installed primary
  source `/usr/local/cuda-12.6/include/cub/device/device_radix_sort.cuh:109`
  documents stability; 2024 identifies the toolkit, not invention.
- **[PyTorch 1.9 (2021)](https://pytorch.org/blog/pytorch-1-9-released/):** opt-in
  deterministic indexing policy is taken; the primary release page was verified.
- **[Demmel and Nguyen, ARITH 2013](https://www.acsel-lab.com/arithmetic/arith21/papers/p54.pdf):**
  reproducible-summation context only. Their order-independent accumulator is not
  implemented. This code fixes original position order and uses ordinary FP64
  addition followed by FP32 rounding; cross-toolchain/device/order invariance is
  not claimed. The primary paper was verified.
- **PT-DET-1 / CC46 / PT-TF32 (2026), SHA256 / NIST (2001), POSIX:** existing
  shared scalar arithmetic, replay/digest/receipt and process-deadline mechanics
  are reused. This change adds n-D full-destination grouping, one shared switch,
  loss-byte capture and combined-slot integration. No new sorting or summation
  algorithm is claimed. Code-site annotations identify these same boundaries.

Code sites: `deterministic_gather.h:8`, `gather_deterministic.cu:15` and `:29`, `embed_deterministic.cu:15`, `kernels.cu` deterministic scatter dispatch, bindings/Python aliases, and harness function/module comments. Kernel claims are source reasoning until lead measurement; CPU baselines are not blind review.
