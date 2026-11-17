# ENG3 ledger (seat: Claude Opus 5.5, Agent tool; order orders/ENG3_HOST_TENSOR_CORE.md at 114d6269)

Times EST (UTC-5), 2026-11-16, unless marked UTC. No git writes; read-only git only.

## Setup
- 07:05 read the order, HOUSE_RULES.md, AGENTS.md; read kernels.cu 1-800 (Storage, NDArray core), core.h, __init__.py
  1-200, bindings.cpp tensor_from_numpy / tensor_to_numpy / Tensor class / empty; autograd.cpp; ops.cpp cast, sum,
  where, masked_fill, pow backward. Read ENG2's LEDGER.md and build receipts (eng2 worktree, read-only) for the build
  commands.
- 07:09 L01 (`preflight_L01.txt`): cmake 3.28.3, nvcc 12.6.85, pytest 9.0.2, Python 3.12.3, numpy 2.4.2,
  `forge-gpu-run --status --min 30 --max 600` rc 0 ("state: no run"). Slip (L56(a)): the first numpy check ran with
  `python3 -I`, which hides the user site where numpy lives; re-ran with `-B -P` (ok). Disk: /mnt/ForgeRealm
  109,946,040,320 B free (stop rule 40 GB). MemAvailable 33.5 GB.

## Build B
- configure_B 12:10:08-12:10:13 UTC exit 0; build_B 12:10:19-12:12:08 UTC exit 0 (`build/build_receipts.json`, logs
  beside it). Command: `systemd-run --user --scope -q -p MemoryMax=12G -p OOMPolicy=continue -p CPUWeight=50
  --setenv=TMPDIR=/tmp/eng3_tmp nice -n 19 cmake ...`; configure flags as ENG2: Release, CUDA arch 89, production
  pybind11 source dir, FETCHCONTENT_FULLY_DISCONNECTED, CUDACXX cuda-12.6; build `-j6`, run detached and waited on
  with `timeout 590 tail --pid`.
- Module sha256 9d8be988977d7a197805d7daa9c347008e34e487f30c140687cb5fbbe10d0af6, copied to
  tensor_cuda/build-eng3/B/module/; source map (28 files) sha256 3ebf1edc75d1ef0fd12284360a6b349caaf83d3af14e60405288dd970d492093
  (`build/BUILD_B.json`).

## L99 seat check (`SEAT_CHECK_B_cpu.txt`, CUDA_VISIBLE_DEVICES empty, Build B)
- `tc.tensor([1.0], device="cpu")` -> RuntimeError "CUDA error at from_host: no CUDA-capable device is detected".
  The defect reproduces. `tc.zeros(3, device="cpu")` the same. An empty host tensor and `tc.empty(3, device="cpu")`
  work (from_host returns before cuda_check_last when bytes == 0; empty does no CUDA call for "cpu").
- Introspection of the built module: `tc.full`, `Tensor.clone`, `Tensor.to`, `Tensor.cpu`, `Tensor.cuda` do not exist;
  no tc or Tensor name contains full/clone/copy.

## E1 census (`HOST_CENSUS.json`, `core_rows.json`, `scripts/eng3_census.py`)
- Core rows (hand-written from the L82 grep): E2 = zeros (cudaMemset 577), full (fill_kernel 585, check 587),
  from_host (check 685), to_host (check 693), clone (cudaMemcpy D2D 698), to (via clone 702 / cudaMemcpy 706),
  astype (kernels 734/742/750, checks 737/745/754). fill_ refuses already (643). Storage/empty/reshape/data_ptr make no
  CUDA call for a host tensor. No other NDArray core member calls CUDA.
- Python rows (regex over bindings.cpp's module block, 217 entries): 7 E2 data path, 4 refuse a host tensor already,
  34 metadata/config, 172 compute entries that hand the tensor's pointer to engine kernels (14 of them have a device
  check one call level down; deeper checks are not traced).
- Python reach of the E2 members: from_host via tc.tensor and all factories; to_host via Tensor.numpy; astype via
  Tensor.astype/half/float and bfloat16 creation; full and clone only via Tensor.backward (seed full(shape, 1.0),
  accumulate_grad -> reduce_to -> clone); NDArray::zeros only inside compute-op backward closures (tc.zeros is NumPy +
  from_host); NDArray::to has no caller in tensor_cuda/src and no binding.

## STOP (L35): three clauses cannot hold as written
1. G1 names `.to("cpu")`, `full` and `clone` "through the public Python API". `.to` does not exist on the public API and
   NDArray::to has no caller anywhere in the engine; no Python entry takes a fill value; clone is reached only through
   backward. bindings.cpp is outside the writable set, and __init__.py cannot call NDArray::to/clone/full without a
   binding. A Python-only `.to` would test a stand-in, not NDArray::to.
2. E4 + G1 "E4's members refuse on a host tensor": E1's census (which the order defines to include every public Python
   entry) puts up to 172 compute entries in E4. Their code is in bindings.cpp, ops.cpp, matmul.cu, conv.cu, the paint/
   terrain/dda files and kernels.cu outside the core section: all read-only for this round. The only writable route is
   a blanket wrapper in __init__.py, which cannot reach C++-internal paths (for example a second gradient
   accumulation in backward -> ew_binary on host memory).
3. G1 ".numpy() ... equal to a numpy reference on every supported dtype": bindings.cpp:103-109 has no Bool branch; a
   bool tensor's numpy() allocates a float32 array and to_host copies numel bytes into it (reasoning from source; not
   measured; same on a device tensor). The fix is in read-only bindings.cpp.
- Per L35 and House Rules section 3 (stop at the registered rail), the seat stopped before any product edit. Nothing in
  kernels.cu, core.h or __init__.py was changed. No GPU line ran. No mutant was registered or built.

## Prior art (seat's list)
- Census by call-site inventory: no prior art known to me beyond grep/regex inventories; ENG2's caller_census.py
  (this repo) used Python's ast for the same purpose.
- Build receipts and source sha256 maps: content-hash manifests (reproducible-builds practice); ENG2's receipts.
- The planned E3 work (not started) rests on IEEE 754-2008 conversions and round-to-nearest-even; the bfloat16 format
  (Google Brain); CUDA Math API `__float2half_rn` / `__float2bfloat16_rn`; mutation testing (DeMillo, Lipton and
  Sayward, 1978). All unverified in this sandbox -- lead to check.

# r2 (resume under Amendment 1, then Amendment 2) -- 2026-11-16

## Setup and probe
- 07:18 read orders/ENG3_R2_RESUME.md (cba230b1). 07:20 L01 r2 (`preflight_L01_r2.txt`): cmake 3.28.3, nvcc 12.6.85,
  pytest 9.0.2, Python 3.12.3 + numpy 2.4.2, forge-gpu-run --status rc 0. torch 2.11 present (used by G2 for bfloat16
  NaN-payload inputs via DLPack).
- 07:21 probe line `probe_B` (exploration, not a gate) launched; its wrapper sat in the 76-retry sleep with no forge
  output; stopped by me (my own process) at 07:34 to allow Build E, since the wrapper checked the module only once
  (receipts moved to gpu/probe_B_attempt1_stopped). Wrapper fixed: module sha256 re-checked before EVERY attempt
  (L103), ATTEMPTS.json written progressively. 07:41 probe_B ran: forge exit 0, inner 0
  (gpu/probe_B/RESULT.json). Device NaN outputs: every half NaN -> float 0x7FFFFFFF; every float NaN -> half 0x7FFF,
  bfloat16 0x7FFF; half -> float non-NaN equals NumPy's widening on all 65,536 patterns; uint8 -> float exact.
## Product edits (diff: artifacts/eng3/before vs artifacts/eng3/E_src)
- kernels.cu: ENG3 host block (host_f32_to_f16, host_f16_to_f32, host_f32_to_bf16, host_bf16_to_f32, host_ld/st,
  host_rintf, host_to_u8, host_astype_into); zeros/full/from_host/to_host/clone/astype host branches. Device branches
  unchanged statement for statement. Prior art in the code comment (IEEE 754-2008 RNE; bfloat16; CUDA Math API / PTX
  cvt; c10::BFloat16 RNE formulation, unverified).
- bindings.cpp: A1.1 private hooks _core_full, _core_zeros, _core_clone, _core_to (no public name; __init__.py
  untouched); A2.1 Bool branch in tensor_to_numpy. An edit hook flagged TX3-F1B's earlier, unmerged fix of the same
  defect (fork worktree tensor_fork_apa_wt_tx3f1, tc/numpy_export.h); cited as in-repo prior art in the comment.
- core.h: unchanged (no new declarations needed).
## Builds (build/build_receipts.json, MODULES.txt)
- E 9f070b6f... (first E build had no Bool branch; rebuilt after A2.1; a comment-only edit after the probe rebuilt to
  the same sha256). Bh 17f305eb... = B's kernels.cu/core.h + the A1.1 hook block only (device-path reference for
  zeros/full/clone/to, which Build B cannot reach). Mutants from /tmp/eng3_src_<id> trees: m01 1cbc36ae, m02 5521073d,
  m03 9c3530d7, m04 72a72c1a, m05 4e7d9fc9. First mutant trees (before the comment edit) moved to /tmp/eng3_old and
  artifacts/eng3/mutants_superseded_v1 (never built).
## Instruments
- artifacts/eng3/tools/cudart_count.c (.so): LD_PRELOAD counter of the 32 libcudart functions the module imports.
  Repair 1 of 3 (L56(b)): dlsym(RTLD_NEXT) cannot see libcudart (a dependency of an RTLD_LOCAL extension) -> segfault;
  fixed with dlopen(RTLD_NOLOAD). Verified on Build B: import 0 calls, empty host tensor 0, nonempty host tensor 2.
  (A /proc/self/maps libcuda check was tried first and rejected: libcuda is mapped at import.)
- scripts/eng3_suite.py repair (L56(b) 2 of 3): collection errors interrupted the session -> --continue-on-collection-
  errors (both builds); try 1 kept as g3/cpu_B_try1_interrupted.
## Registrations (before the runs they govern)
- MUTANTS.json 12:49:16 UTC (sha cd41f97d...), A14_PAIRS.json, G3_INTENDED.json 13:05:45 UTC (sha 0f8224e0...),
  g3/SPLIT.json (GPU halves).
## Deviation: first CPU before-run read E sources
- g3/cpu_B (07:51-08:01) ran Build B's module with E's kernels.cu/bindings.cpp in the worktree, so the suite's
  source-pin tests read E bytes. Kept as g3/cpu_B_srcE; G3 uses reruns with each build's own sources
  (scripts/eng3_srcswap.py B|E).
## Gate runs (times EST)
- 08:25 G1 CPU on E (CUDA hidden): g3/G1_cpu_E 76/76 passed (incl. 8 G1b cpu rows; no-CUDA counter host_calls 0,
  control_calls 4).
- 08:27-08:28 mutants on the CPU rows (all five) + named-only runs for m01/m03/m05 (markers in junit).
- 08:29-08:39 CPU suite E (E sources) g3/cpu_E; 08:40-08:51 CPU suite B (B sources) g3/cpu_B; compare
  g3/G3_cpu_compare.json: 3368 IDs, 0 differences.
- GPU lines (each: L103 sha check before every attempt, dry run 78, forge exit, inner exit, log sha256 in
  gpu/<line>/RESULT.json): G3_B1, G3_B2 (+devpath B_public), devpath_Bh (attempt 0 exit 76, retried after 300 s),
  G3_E (own tests G1/G1b/G2 on GPU 121/121, suite halves, devpath E_hooks/E_public), mut_m02, mut_m04, mut_m05.
  Lease waits behind another seat's line (p1qr) are inside the first attempts of G3_B1/G3_B2.
- G3 GPU compare g3/G3_gpu_compare.json: 3368 IDs, 0 differences. Devpath: B vs Bh public 24/24 identical (Bh is a
  faithful B reference); B vs E public 24/24; Bh vs E hooks 91/93 identical, the 2 others are the A2.2 waived bool
  .numpy() rows (Bh: 60-byte float32 array whose first 15 bytes are the data; E: the 15-byte bool array); every raw
  bool byte row (from_host_raw, clone, to_*) identical.
- Tool label slip (L56(a)): v1 of eng3_g3_compare.py waived only key bool_numpy_waived/bool, while hooks mode also
  records the same bool .numpy() under from_host+numpy/bool; v1 output kept as devpath/CMP_Bh_vs_E_hooks_v1_keylabel.json
  (that key "bytes differ"); v2 waives exactly the two keys that read bool through .numpy(), as A2.2 states.
- G4: MUTANT_RESULTS.json, 5/5 killed on their named assertions, marker in each named run's junit; m03 counter
  {'cudaMemcpy': 13}.
- L60: BOUNDARY.json (fill_ refusal measured on B and E with CUDA hidden; the 14 checked compute entries listed).
## G5 event (scope)
- scripts/eng3_build_arms.sh and scripts/eng3_line_steps.sh were written in scripts/, but the writable set there is
  "new scripts/eng3_*.py". Found at 10:04 by the scope check; moved (not deleted) to artifacts/eng3/tools/
  (sha256 7ea42ee8... and 26975818...). GPU line argv in gpu/*/START.json name the old path. G5 reported FAIL (2
  events), corrected.
