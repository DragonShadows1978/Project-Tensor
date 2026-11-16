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
