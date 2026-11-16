<!-- Filed by the lead verbatim from the r1 seat's final message (the harness refused the seat's own write). -->

# ENG3 report: RED. I stopped at L35 before editing any product file.

The write of `artifacts/eng3/REPORT.md` was refused ("Subagents should return findings as text, not write report files"). Per the order, this message is the report; please file it verbatim. The ledger write succeeded: `artifacts/eng3/LEDGER.md`.

**Verdict:** three clauses of the order cannot hold as written inside the writable set. I ran the L99 seat check (the defect reproduces) and the E1 census, then stopped (L35; House Rules §3). Nothing in `kernels.cu`, `core.h` or `__init__.py` was changed. No GPU line ran and no mutant was registered.

## Gates
- **G1: FAIL / NOT RUN.** RED: the clause cannot hold as written (items 1–3 below).
- **G2: NOT RUN.** I stopped before E3.
- **G3: NOT RUN.** Only Build B exists.
- **G4: NOT RUN.** `MUTANTS.json` was not written; no mutant was registered or built.
- **G5: PASS.**
  - Writes are only inside the writable set (list at the end).
  - `git diff --quiet` shows no tracked file modified.
  - No git writes, nothing deleted, no GPU use.

## Seat check (L99) [measured, `artifacts/eng3/SEAT_CHECK_B_cpu.txt`, Build B, CUDA_VISIBLE_DEVICES empty]
- `tc.tensor([1.0], device="cpu")` raises `RuntimeError: CUDA error at from_host: no CUDA-capable device is detected`. The defect reproduces.
- `tc.zeros(3, device="cpu")` raises the same error.
- An empty host tensor works, and so does `tc.empty(3, device="cpu")`.
- Introspection of the built module: `tc.full`, `Tensor.clone`, `Tensor.to`, `Tensor.cpu` and `Tensor.cuda` do not exist. No tc or Tensor name contains full, clone or copy.

## Census summary (E1) [reasoning from source; `artifacts/eng3/HOST_CENSUS.json`, sha256 604632d7…f646]
**Core members that call CUDA on a host tensor's storage (the E2 set):**

| Member | kernels.cu lines | CUDA use on host storage |
|---|---|---|
| `zeros` | 575-579 | `cudaMemset` (577) |
| `full` | 580-589 | `fill_kernel` launch (585), `cuda_check_last` (587) |
| `from_host` | 679-687 | `cuda_check_last` (685) |
| `to_host` | 688-694 | `cuda_check_last` (693) |
| `clone` | 695-700 | `cudaMemcpy` device-to-device (698) |
| `to` | 701-708 | host to host goes through `clone` (702) |
| `astype` | 727-756 | kernel launches at 734/742/750, checks at 737/745/754 |

**Other core members:**
- `fill_` already refuses a host tensor (643).
- `Storage`, `empty`, `reshape`, `data_ptr` and `cuda_storage_device_mismatch` make no CUDA call for a host tensor.
- No other NDArray core member calls CUDA, so there is no core-level E4 member.

**Python entries:** 217, from a regex over the module block in `bindings.cpp`.

| Class | Count |
|---|---|
| E2 data path | 7 |
| Already refuse a host tensor (`fill_`, `__dlpack__`, `__dlpack_device__`, `_from_dlpack_capsule`) | 4 |
| Metadata or config | 34 |
| Compute entries that pass the tensor's pointer to engine kernels | 172 |

Of the 172 compute entries, 14 have a device check one call level down. Deeper checks were not traced.

**How the public Python API reaches each E2 member:**

| Member | Reached through |
|---|---|
| `from_host` | `tc.tensor` and every factory |
| `to_host` | `Tensor.numpy` |
| `astype` | `Tensor.astype`, `half`, `float`, bfloat16 creation, and `numpy()` of a bfloat16 tensor |
| `full`, `clone` | Only `Tensor.backward`: the seed is `full(shape, 1.0)`, then `accumulate_grad` → `reduce_to` → `clone` |
| `NDArray::zeros` | Only compute-op backward closures; `tc.zeros` is NumPy plus `from_host` |
| `NDArray::to` | No caller in `tensor_cuda/src` and no binding |

**E4 refusals implemented: none.** I stopped first.

## Why I stopped (L35)
1. **G1 names entries the public API does not have.** G1 requires `full`, `clone` and `.to("cpu")` "through the public Python API".
   - `.to` does not exist, and nothing in the engine calls `NDArray::to`.
   - No Python entry takes a fill value.
   - `bindings.cpp` is read-only, and `__init__.py` cannot reach these members without a binding. A Python-only `.to` would test a stand-in, not `NDArray::to`.
2. **E4's population is outside the writable set.** E1's census covers every public Python entry, so E4 covers up to 172 compute entries.
   - Their code is in `bindings.cpp`, `ops.cpp`, `matmul.cu`, `conv.cu`, the paint, terrain and dda files, and `kernels.cu` outside the core section. All of these are read-only.
   - A blanket wrapper in `__init__.py` cannot reach paths that start inside C++. Example: a second gradient accumulation in backward calls `ew_binary` on host memory.
3. **`.numpy()` cannot match a NumPy reference for bool, on any device.** `tensor_to_numpy` (`bindings.cpp:103-109`) has no Bool branch.
   - A bool tensor gets a float32 output array, and `to_host` copies only numel bytes into it.
   - This is reasoning from the source, not measured.

## Proposed amendments (stated, not implemented)
- **A1, the G1 surface.** Pick one:
  - (a) Add `bindings.cpp` to the writable set for public `Tensor.clone()`, `Tensor.to(device)` and `tc.full`. This changes the public API of infrastructure.
  - (b) Add a private `_C._core_host_*` test hook in `bindings.cpp`.
  - (c) Restate G1: reach `full` and `clone` through `backward`, and `to` and `NDArray::zeros` through a C++ driver built under `build-eng3/`.
- **A2, the scope of E4.** Pick one:
  - (i) Limit E4 to core members, which leaves none beyond the E2 set and `fill_`. Host compute stays dependent on HMM and is registered as a successor.
  - (ii) Add the compute files to the writable set, with a `require_cuda(a, "<member>")` helper in `core.h` called at each entry.
  - (iii) Rule that a wrapper in `__init__.py` is E4's mechanism, and list the C++-internal gap as not covered.
- **A3, bool `.numpy()`.** Either exclude bool from G1's `.numpy()` check, or allow a Bool branch in `bindings.cpp`. A Bool branch changes a device result, so it also needs E5 waived for that row.
- **What carries over:** Build B is built and receipted. The census is ready. The E2/E3 design is unaffected by any of these amendments.

## Diff summary
No product diff. No tracked file was modified. Only new untracked helpers and receipts exist.

## Module sha256s
- **Build B module:** `9d8be988977d7a197805d7daa9c347008e34e487f30c140687cb5fbbe10d0af6`.
  - It is installed in the worktree package and copied to `tensor_cuda/build-eng3/B/module/`.
  - Source map: 28 files, sha256 `3ebf1edc75d1ef0fd12284360a6b349caaf83d3af14e60405288dd970d492093` (`artifacts/eng3/build/BUILD_B.json`).
- **Build E:** none.
- **Mutant builds:** none.

## G2 per-pair result
Not run: no pair was measured.

## G3 per-ID result and intended changes
Not run. No intended-change list was registered.

## Mutant table
- `m01_from_host_checks_cuda`: not registered, not run.
- `m02_host_astype_truncates`: not registered, not run.
- `m03_host_clone_cudamemcpy`: not registered, not run.
- `m04_u8_no_clamp`: not registered, not run.

## GPU lines
None ran: no dry run, no lease, no `RESULT.json`.

## Deviations
- **L56(a):** the first L01 numpy check used `python3 -I`, which hides the user site-packages. I re-ran it with `-B -P` and it passed.
- **Build niceness:** ENG2 dropped `nice -n 19` after its builds hit the time bound. I kept `nice -n 19` with `-j6`, and the build took 110 s.
  - It ran in its own systemd scope (MemoryMax=12G, OOMPolicy=continue, CPUWeight=50).
  - It started with MemAvailable at 33.5 GB.
  - Receipts: `artifacts/eng3/build/build_receipts.json`.
- **REPORT.md:** the write was refused, so this message is the report.

## Prior art
- **Census:** a grep/regex call-site inventory. No prior art known to me beyond ENG2's `caller_census.py` in this repo, which used Python's ast.
- **Build receipts:** content-hash manifests (reproducible-builds practice; ENG2's receipts).
- **Planned E3 and G4 (not started):**
  - IEEE 754-2008 conversions with round-to-nearest-even;
  - the bfloat16 format (Google Brain);
  - CUDA Math API `__float2half_rn` and `__float2bfloat16_rn`;
  - mutation testing (DeMillo, Lipton and Sayward, 1978).

  None of these were verified in the sandbox (no network). Lead to check.

## L109
Helpers run as `python3 -B -P`: `eng3_run.py`, `eng3_shamap.py`, `eng3_census.py`, `eng3_seat_check.py`. None of these names is a standard-library module name.

## Files written
All paths are under `/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_eng3/`.
- `artifacts/eng3/LEDGER.md`
- `artifacts/eng3/HOST_CENSUS.json`
- `artifacts/eng3/core_rows.json`
- `artifacts/eng3/preflight_L01.txt`
- `artifacts/eng3/SEAT_CHECK_B_cpu.txt`
- `artifacts/eng3/build/BUILD_B.json`
- `artifacts/eng3/build/build_receipts.json`
- `artifacts/eng3/build/build_receipts_configure_B.log`
- `artifacts/eng3/build/build_receipts_build_B.log`
- `scripts/eng3_run.py`
- `scripts/eng3_shamap.py`
- `scripts/eng3_census.py`
- `scripts/eng3_seat_check.py`
- `logs/eng3_build_B.out`
- `tensor_cuda/build-eng3/B/`: the CMake build tree, 27 MB, including `module/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`
- `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`: the installed Build B module, git-ignored

Scratch left in place: `/tmp/eng3_tmp/` (empty).
