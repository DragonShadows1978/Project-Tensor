# TensorCUDA Engine Split — Recommended Course of Action

Status: **PROPOSAL** — a recommendation for the lead, not an order. Nothing in
this document has been executed. If adopted, the lead cuts orders from it and
the adopted text becomes immutable per house rules; execution detail then goes
to `ENGINE_SPLIT_LEDGER.md`.

Tree inspected: `main` at `fd32b6d` (2026-10-02). Every factual claim below is
**code inspection** of that tree unless marked otherwise. Line numbers refer to
that commit.

## Objective

Get project-specific kernels out of the TensorCUDA engine **without breaking
anyone who uses them**. Today the engine is a single pybind11 module
(`_tensor_cuda`) built from a single CMake source list, and about 70 % of its
CUDA is payload for specific consumer projects. The end state:

- a **core engine** that knows nothing about APA, GRM, ColdCast or
  Project-Scorch;
- **extensions** that those projects own. They build against the core's public
  headers, register their ops through a defined contract, and still show up as
  `tc.<op>` for existing callers;
- **governance pins scoped per component**, so a change to one project's
  kernels stops invalidating every other project's receipts.

## Invariants (hold through every phase)

1. **No numerics change.** Every kernel body moves byte-for-byte. A move is
   accepted only on bit-identical outputs over the relocation oracle (P0), with
   per-kernel SASS identity as the tiebreaker.
2. **APA is load-bearing.** No phase changes APA selection, refinement,
   dispatch or defaults. APA relocates; it is not redesigned.
3. **The public Python surface is preserved.** Every name in the P0 API
   manifest still resolves as `tensor_cuda.<name>` after every phase, with the
   same signature.
4. **The default build stays complete** until P7. Consumers opt *out* of
   extensions; nobody has to opt in to keep working.
5. **Campaigns are paused across P4–P5.** Relocating code breaks every open
   registration's source pins. Run those phases between campaigns.

## 1. What is in the engine today

| Bucket | Location | ~Lines | Consumer |
|---|---|---:|---|
| Engine core | `kernels.cu` 1–1010, 4758–5190, 7167–7568; `matmul.cu`, `conv.cu`, `autograd.cpp`, most of `ops.cpp` | ~3,000 | everyone |
| Weight / KV quant | `kernels.cu` 3578–4757 (int4, intn, w8a16, MXFP4, KV int4) | ~1,180 | LLM ports (Gemma-4 port, GPT-OSS, MiniCPM3 …) |
| **APA** | `kernels.cu` 1011–3577, 5191–7166 (APA arms), 7857–8190; `gemm_apa.cu`; `apa_sp1_1.cuh`; `apa_sp2.cuh` | **~5,800** | APA research, GRAPA training (BP-KERNEL-*) |
| **GRM row ops** | `kernels.cu` ~7569–7744 (`export_rope_rows`, `export_row_pair(s)`, `swap_row_pairs_with_rope`, `evict_row_pairs`, `arena_row_pair_transaction`) | ~180 | Graft Runtime Memory |
| **ColdCast paint** | `paint_raster.cu`, `paint_bake.cu`, `paint_inpaint.cu` | 1,196 | ColdCast (PAINT-CUDA-1/2/3) |
| **Project-Scorch voxel** | `terrain.cu`, `terrain_objects.cu`, `dda.cu` | 2,979 | Project-Scorch (WO-T1, WO-7B … WO-12G) |
| Model-arch op | `kernels.cu` 7745–7856 `gated_delta_step` | 112 | Qwen3.5 port |

Python layer (`tensor_cuda/tensor_cuda/__init__.py`, 1,288 lines):
- paint wrappers and `build_inpaint_island_csr`: lines 137–398;
- terrain validation and wrapper: lines 399–746;
- `dda_raycast`: lines 125–136;
- GRM row-pair wrappers: lines 789–830;
- APA: lines 96–124, 941–1111, 1237 (`apa_quant_attention` via `quant.py`), and 1268–end (training plus the `apa_sp2` re-exports).

Tests (`tensor_cuda/tests/`):

| Bucket | Files | Lines |
|---|---:|---:|
| APA | 30 | 6,769 |
| Voxel | 2 | 4,625 |
| Paint | 6 | 2,272 |
| Row ops | 3 | 342 |
| Core | 22 | 2,493 |

## 2. Coupling findings (why it is one piece)

**F1 — No extension path exists, and the record says so.** The WO-T1 order
(`docs/orders/WO-T1_dda_raycast_op.md`) reads: *"Engine assessment confirmed
tensor_cuda has no external kernel-launch path, so the op lands as a
first-class engine op."* Every later consumer followed that precedent
(WO-7B: *"Follow the WO-T1 pattern"*). The monolith is the default outcome of
having no plugin contract, not a design choice.

**F2 — Device helpers are private to `kernels.cu`.** `ld<T>`, `st<T>`,
`DISPATCH_FLOAT`, `nblk` and `kT` live in `kernels.cu`'s anonymous namespace
(lines ~204–240). APA cannot leave that file without them. The paint files
work around it with their own copies of `same_device`,
`validate_launch_threads` and `require_same_cuda_device` (duplicated in
`paint_bake.cu` and `paint_inpaint.cu`, with `same_device` also in
`paint_raster.cu`).

**F3 — `core.h` is a registry of consumers.** It declares
`TerrainRenderCamera`, `TerrainRenderLight`, `TerrainRenderConstants`,
`TerrainRenderObject`, the paint entry points, `ApaInt4Workspace` and about
20 APA functions, next to `Storage` and `NDArray`. `ops.h` opens with the
APA-SP1 block.

**F4 — Scorch state lives inside the autograd `Variable`.**
`include/tc/autograd.h:24-46`: every `Variable` in the engine carries
`mutable TerrainDensityCacheEntry terrain_density_cache[2]`, read only by
`ops.cpp:396` (`terrain_render`). This is the deepest leak in the tree: a game
renderer's cache in the core object of the autograd graph.

**F5 — Core attention is an APA template instantiation.** Plain
`fused_sdpa_noncausal` on fp16 with D ≤ 64 runs
`qtile::attn_noncausal_f16_kernel<false, false>` (`kernels.cu:6736`). That is
the same template APA instantiates `<true, true>` / `<true, false>`
(`kernels.cu:7113`, `7123`). The APA arms are behind `if constexpr (kAPA)`,
so the non-APA instantiation compiles them out. This seam is cuttable, but it
is the one place where core and APA share a kernel body.

**F6 — Each op is spread across four files.** Each op is touched in:
- its `.cu` file;
- a Tensor wrapper in `ops.cpp` (paint and voxel at 291–428, GRM row ops at
  523–645, APA at 16–40, 447–458 and 752–798);
- an `m.def` in `bindings.cpp` (paint and voxel at 401–583, APA at roughly
  585–607 and 687–976);
- a wrapper in `__init__.py`.

**F7 — Governance pins couple unrelated projects.** Campaign registrations pin
the entire engine source set by whole-file SHA-256. For example,
`artifacts/apa_sp4g/registration.json` and
`artifacts/bp_kernel_3/registration.json` pin `terrain.cu`, `paint_*.cu` and
`dda.cu`. So:
- a Scorch or ColdCast change invalidates APA receipts, and the reverse;
- new work is appended to `kernels.cu` to keep historical byte offsets fixed
  (`kernels.cu:8014`: *"BP-KERNEL-4 BEGIN: appended so all historical byte
  offsets stay fixed"*), which grows the monolith on purpose;
- `tests/conftest.py` defines *sha-bound* as a pin "that a LATER campaign
  moved", which is exactly this cross-project churn.

**F8 — Hidden singletons and unwritten contracts.** These must stay
exactly-once in the process:
- the allocator pooling flag (`set_alloc_pooling`, `kernels.cu:152`);
- `Storage::revision`, which `ApaInt4Workspace` depends on;
- the cuBLAS handle (`matmul.cu`, file-static `g_handle`);
- the cuBLASLt handle and workspace (`gemm_apa.cu:61-70`);
- the global grad mode.

The rule that everything runs on the legacy default stream is assumed
everywhere and written down nowhere. `bp_op_timing.h` also carries a
`source_pin` constant.

**Precedent.** `apa_cuda/` is already APA as a separate extension (a PyTorch
`CUDAExtension`, `apa_cuda/setup.py`). Splitting APA out is not a new idea in
this repo; it was never applied to the native engine.

## 3. Target layout

```
tensor_cuda/                      CORE — no consumer names anywhere
  include/tc/
    core.h  autograd.h  ops.h     (consumer declarations removed)
    device.cuh                    NEW: ld/st/DISPATCH_FLOAT/nblk/launch checks
    extension.h                   NEW: registration contract + derived-state cache
    attn/qtile_skeleton.cuh       NEW (P6): shared Q-tile attention template
  src/  kernels.cu (core regions) matmul.cu conv.cu autograd.cpp ops.cpp bindings.cpp
        quant_linear.cu  kv_quant.cu   (split out of kernels.cu, still core)
  tensor_cuda/                    Python core + extension loader

extensions/                       first-party extensions, separate targets
  apa/   csrc/ (selective, int4, gemm_apa, train fwd/bwd, bk1-4, sp1/sp2,
               qtile APA policy, blend_softmax, quantize_gather)
         python/tc_apa/ (wrappers, quant.py tables, apa_sp2.py)  tests/
  grm/   csrc/ (row-pair ops)     python/tc_grm/   tests/

(moved to consumer repos at P8, consumed via find_package(TensorCUDA))
  Project-Scorch/tc_voxel/        dda.cu terrain.cu terrain_objects.cu + tests
  ColdCast/tc_paint/              paint_raster/bake/inpaint.cu + tests
```

### What stays in core, and why

| Item | Decision | Reason |
|---|---|---|
| int4 / intn / w8a16 / MXFP4 linears | **Core** (own file `quant_linear.cu`) | Generic LLM inference primitives with several model consumers. MXFP4 is the OCP Microscaling format, not a GPT-OSS-only format. Excising them would put every LLM port behind an extension. |
| `kv_int4_pack` / `kv_int4_unpack` | **Core** (`kv_quant.cu`) | Consumed by the Gemma-4 port (`external/kimi-swarm/gemma4_tc.py`, `core/kv_manager.py`); generic KV-cache storage. |
| `write_rows`, `export_rows`, `splice_rows`, `evict_rows`, `rope_apply` | **Core** | Generic KV-cache and RoPE manipulation. |
| `export_rope_rows`, row-pair family, `arena_row_pair_transaction` | **GRM extension** | Mount and arena semantics (`sink_tokens`, `current_mount_tokens`, `arena_width`) are GRM's protocol, not a tensor primitive. |
| `gated_delta_step` | **Core for now** | A single model-architecture op. Open a `tc_models` extension when a second one appears; one op does not justify the machinery. |
| `fused_sdpa_noncausal` + Q-tile skeleton | **Core** | Generic attention. The APA arms leave at P6. |
| `apa_quantize_gather`, `apa_blend_softmax*` | **APA** | APA-specific despite generic-looking shapes. |
| `bp::` timing (`bp_op_timing.h`) | **Core**, as an opt-in profiler hook | Generic nested CUDA-event timing. The `bp_kernel_N_*` variant switches go to APA. |
| `dda_raycast` | **Voxel**, not core | Generic Amanatides–Woo traversal, but Scorch is its only consumer and `terrain.cu` re-implements it anyway (`terrain.cu:3-4`). |

## 4. Mechanism (staged)

**Stage A — source-level extensions, one binary (P1–P6).** Each extension is
a CMake subdirectory or static object library that the core build pulls in
through `-DTC_EXTENSIONS=apa;grm;voxel;paint` (default: all). Each extension
provides one `void register_<name>(pybind11::module_&)` function, and the
core's `PYBIND11_MODULE` calls the enabled ones. Properties:
- still one `_tensor_cuda.so`, so no new ABI risk;
- binary-bound receipts remain meaningful;
- the code physically leaves core directories, and core headers stop naming
  consumers.

**Stage B — shared core plus per-extension modules (P7).**
`libtensorcuda_core.so` exports `Storage`, `NDArray`, `Tensor` and autograd,
plus a CMake package (`find_package(TensorCUDA)`). `_tensor_cuda` and each
`tc_<ext>/_C` link that library. Rules this stage has to follow:

- **Never link core statically into an extension.** Doing so duplicates every
  F8 singleton (two allocator pools, two grad modes, two cuBLAS handles), and
  it fails quietly.
- **Device code does not cross `.so` boundaries.** Shared device helpers must
  be header-only `__forceinline__` code in `tc/device.cuh`. Each extension
  stays self-contained at device-link time; `CUDA_SEPARABLE_COMPILATION` does
  not change this.
- **One pybind11 for everything.** Keep v2.12.0 as already pinned and the same
  compiler, so `tc.Tensor` passes between modules through shared pybind11
  internals.
- **Keep LTO off.** `CMakeLists.txt` already disables it for the
  nvcc/gcc LTO version mismatch; that applies to every extension too.

**Python discovery.** Core `tensor_cuda/__init__.py` gains a module-level
`__getattr__` that resolves unknown names against registered extensions.
Extensions are discovered through the `importlib.metadata` entry-point group
`tensor_cuda.extensions`, with in-tree extensions listed statically. Existing
`tc.apa_selective_attention(...)` and `tc.terrain_render(...)` calls keep
working without edits, and new code can import `tc_apa` directly.

**Not recommended now: a C-ABI plugin boundary (DLPack-shaped).** It is the
most decoupled option and would allow extensions built with a different nvcc.
The costs:
- an allocator callback in the ABI, so outputs still go through the core pool
  and `Storage::revision`;
- autograd wired in Python. `apa_selective_train` (`ops.cpp:752`) builds its
  graph in C++ today, so it would have to be rewritten.
- `Tensor` exposes no `data_ptr`, `__dlpack__` or `__cuda_array_interface__`
  yet.

Revisit only if third parties need to build extensions.

## 5. Phases

Each phase ends at a gate. A gate failure stops the line; it is never waived
by re-reading.

### P0 — Relocation oracle and API manifest (registration, before any edit)

- Build the current tree. Record the `.so` digest and run the full
  `tensor_cuda/tests` and `tests/` suites, saving the pass/skip/fail set.
- **API manifest:** `sorted(dir(tensor_cuda))` plus `sorted(dir(_C))`, each
  name with its pybind11 signature string.
- **Relocation oracle:** a fixed-seed input set per public op, where
  "per public op" means every `m.def` except the debug and profile entries,
  which are listed explicitly. Store SHA-256 digests of the outputs (fp64-cast
  values hashed alongside raw bytes).
- **Per-kernel SASS digest:** `cuobjdump -sass` of the built `.so`, split per
  `__global__` symbol and hashed.
- **Gate:** all four artifacts written create-only, with their own SHA-256 in
  a registration file.

### P1 — Headers and helpers (no kernel bytes move)

- Create `tc/device.cuh` from the `kernels.cu` helpers. Leave the originals in
  place, guarded so there is a single definition, then switch `kernels.cu` to
  include the header in the same commit. Delete the paint duplicates.
- Split consumer declarations out of `core.h` and `ops.h` into `tc/apa.h`,
  `tc/grm.h`, `tc/voxel.h` and `tc/paint.h`. `core.h` keeps
  `#include`-forwarding them until P5 so call sites do not change.
- **Gate:** API manifest identical, oracle bit-identical, SASS digests
  identical, test pass set identical.

### P2 — Evict consumer state from `Variable` (F4)

- Add to `tc/extension.h` a generic **derived-state cache**: a side table
  keyed by `(Storage*, slot id)`. Each entry stores a `weak_ptr<Storage>`, the
  source `revision` and an opaque `shared_ptr<void>` payload.
- Move `terrain_density_cache` into it and remove the field from `Variable`.
  `ApaInt4Workspace` keeps its explicit handle API, but can later use the same
  cache for revision-checked invalidation.
- **Gate:** terrain tests and oracle bit-identical; `sizeof(Variable)`
  shrinks; no other source changes.

### P3 — Bindings and Python split (no kernel bytes move)

- Split `bindings.cpp` into `register_core`, `register_apa`, `register_grm`,
  `register_voxel` and `register_paint`, each in its own `.cpp`.
- Move the domain Python into `tensor_cuda/_ext/{apa,grm,voxel,paint}.py`,
  re-exported by `__init__.py`.
- Move each domain's `ops.cpp` Tensor wrappers next to its bindings.
- **Gate:** manifest identical (names and signatures); oracle bit-identical.

### P4 — Pin migration (governance; must precede any kernel move)

- **Relocation registration.** For every pinned source region in every live
  registration, record a mapping from
  `(old file, byte range or whole file, sha)` to `(new file, sha)`. A
  verifier checks that each region's bytes appear verbatim at the new
  location. Under that verifier a pure move is provable, and does not count
  as a numerics change.
- **Content pins from now on.** Pin regions by marker-delimited content
  (`// <ID>_BEGIN` … `// <ID>_END`, already used for APA-SP1) and hash the
  bytes between the markers, not file offsets. Registrations then pin only the
  core plus the extension under test, not the whole engine.
- Classify the pins this phase retires as `campaign_receipt` in
  `tests/conftest.py`, following the existing H4 and BP-H1 rules. No
  assertion changes.
- **Gate:** the verifier passes on a no-op relocation (identity mapping), and
  on a deliberately mutated copy it fails, naming the region.

### P5 — Relocate kernels, still one `.so` (Stage A)

Order, cleanest first:

1. **voxel**: already separate files; zero `kernels.cu` bytes.
2. **paint**: separate files.
3. **GRM**: one contiguous `kernels.cu` region.
4. **quant / KV-quant** into core's own files.
5. **APA**: largest. Moves as whole marker-delimited regions, in source
   order.

Every move is a pure cut-and-paste of bodies plus `#include "tc/device.cuh"`.

- **Gate per move:** relocation verifier passes; oracle bit-identical; SASS
  digest identical per kernel. If SASS differs, the move is reverted and the
  cause recorded; nothing is accepted on output equality alone. Test pass set
  identical.

### P6 — The attention seam (F5)

- **Step 1 (pure move):** move the `qtile` namespace verbatim into
  `tc/attn/qtile_skeleton.cuh`. Core instantiates `<false, false>`; APA
  includes the header and instantiates its two arms in its own translation
  unit.
- **Step 2 (optional refactor, own order):** replace the `bool kAPA,
  bool kLadder` parameters with a score-policy type. Core gets a null policy;
  APA supplies the bulk-stage, selection-statistics and refine hooks, so APA
  logic no longer lives in a core header.
- **Gate:** `<false, false>` SASS identical to P0 (the `if constexpr` arms
  make this attainable); APA instantiations' SASS identical to P0; oracle
  bit-identical.

### P7 — Shared core library (Stage B)

- Build `libtensorcuda_core.so` plus per-extension pybind modules, and export
  the CMake package.
- `-DTC_EXTENSIONS=` (empty) builds a core with no consumer code; that build
  is the real proof of the split.
- **Gates:**
  - a **singleton test**: an extension module and core report the same
    allocator-pool state, grad mode and `Storage` revision counter;
  - an extension accepts a core `tc.Tensor` and returns one that core
    autograd can backpropagate through;
  - the empty-extensions build passes the core test bucket;
  - the full build gives an oracle bit-identical to P0.

### P8 — Externalize consumer extensions

- Move `voxel` to Project-Scorch and `paint` to ColdCast, each with its
  tests. They consume `find_package(TensorCUDA)`; the shims in this repo are
  deleted after their consumers switch.
- APA and GRM stay in `extensions/` in this repo (see D1).
- **Gate:** each consumer repo builds its extension against an installed core
  and passes its moved tests. The Project-Tensor core suite has no voxel or
  paint tests left.

## 6. Decisions for the lead

- **D1 — Where APA lives.** The recommendation is an in-repo first-party
  extension (`extensions/apa`), not a separate repo. The campaign
  infrastructure, registrations, artifacts and ledgers live here and are
  APA-centric, so moving APA out of the repo would orphan them. Out of core,
  in the repo.
- **D2 — GRM's home.** `extensions/grm` here, or GraftRepository. It turns on
  whether GRM's protocol evolves with the engine or with GRM.
- **D3 — Whether the generic row ops stay core.** The recommendation is yes;
  only the row-pair, arena and mount family moves.
- **D4 — Timing.** P4–P5 invalidate the source pins of any open registration.
  Schedule them between campaigns.
- **D5 — Whether P6 step 2 happens at all.** Step 1 alone already gets APA
  out of core translation units; step 2 only matters if core attention will
  grow other policies.

## 7. Risks

| Risk | Mitigation |
|---|---|
| Moving a kernel between translation units changes codegen | Per-kernel SASS digest gate (P5). Nothing is accepted on output equality alone. |
| Singletons duplicated across `.so` files (F8) | Never link core statically; singleton test at P7. |
| Cross-`.so` device symbol use | Header-only `device.cuh`; no exported `__device__` symbols. |
| pybind11 internals mismatch between modules | One pinned pybind11 version, one compiler, one build-type policy. |
| Campaign receipts break | Expected and handled: relocation registration plus `campaign_receipt` classification (P4). |
| Silent Python API drift | API manifest gate at every phase. |
| Unwritten default-stream contract broken by an extension | Write it into `tc/extension.h`: all launches go on the legacy default stream until the core grows a stream API. |

## 8. Prior art

Per the AGENTS.md Prior Art Directive. Nothing below was verified against the
literature in this session; every entry is **unverified — lead to check**, with
search terms given.

- **Shared core library plus separately compiled extension modules.** Taken
  from PyTorch C++/CUDA extensions (`torch.utils.cpp_extension`, ~2018) and
  pybind11 cross-module type sharing through its internals ID. Ours: applying
  it to a torch-free engine, and the singleton-test gate. Search: "pybind11
  cross-module types internals", "torch cpp_extension".
- **Static op registration and loadable op libraries.** PyTorch
  `TORCH_LIBRARY` dispatcher registration (~2020); TensorFlow `REGISTER_OP`
  and `tf.load_op_library` (~2016). Taken: a per-extension registration
  function called by the host module. Search: "TORCH_LIBRARY dispatcher",
  "tf.load_op_library custom op".
- **C-ABI tensor exchange (considered, not recommended).** DLPack (~2017);
  Numba `__cuda_array_interface__` (~2018); XLA custom calls. Search:
  "DLPack spec", "cuda_array_interface Numba".
- **Score-policy attention template (P6 step 2).** Essentially PyTorch
  FlexAttention's `score_mod`/`mask_mod` (He et al., 2024) on a
  FlashAttention-style tiled skeleton (Dao 2022; Dao 2023, FA-2). CUTLASS
  epilogue/visitor templates are the C++ analogue. Ours: using it to separate
  APA's bulk/refine selection from a shared skeleton. Search: "FlexAttention
  score_mod 2024", "CUTLASS epilogue visitor tree".
- **Entry-point plugin discovery.** setuptools `entry_points` /
  `importlib.metadata`. Taken verbatim as an idiom.
- **Proving a move with machine code (per-kernel SASS digests).** Close to
  reproducible-build practice (reproducible-builds.org, ~2013+). Search:
  "reproducible builds binary identical verification", "cuobjdump sass diff".
  Ours: using it as the acceptance gate for pure code relocation.
- **Content-addressed, marker-delimited source pins instead of byte offsets.**
  No prior art known to me specific to experiment governance. The general
  idea is content addressing (for example, git objects). Search:
  "content-addressed code region pinning", "semantic anchors source
  provenance".
- **Revision-keyed derived-state cache (P2).** Weak-key side tables with a
  version-counter invalidation. Closest known analogue is PyTorch's tensor
  `_version` counter used for autograd saved-tensor checks. Search: "PyTorch
  tensor _version counter saved tensors". Ours: the slot-keyed cache API.
- **Amanatides–Woo voxel traversal (1987)** for `dda_raycast`, already cited
  at the code site. It is only relocated here.
