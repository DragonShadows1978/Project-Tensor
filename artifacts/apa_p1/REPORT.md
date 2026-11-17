# APA-P1 report

ACCEPT under the registered CPU/staging rule: **G1 PASS; G2 PASS; G3 PASS; G4 staged YES.** GPU verdicts are pending the lead's run. P142 is unscored; P143's CPU part matched and its GPU part is unscored.

## Gate receipts

| Gate | Verdict | Evidence class and receipts |
|---|---|---|
| G1 scope | PASS | Static source identity: [g1.json](g1.json), [g1_tracked.diff](g1_tracked.diff), [g1_diff_stat.txt](g1_diff_stat.txt), [hunk_map.json](hunk_map.json) |
| G2 build + SASS | PASS | CUDA build: [build.json](build.json), [configure.log](configure.log), [build.log](build.log), [module.sha256](module.sha256); instruction census: [sass_comparison.json](sass_comparison.json), [sass_counts.tsv](sass_counts.tsv) |
| G3 CPU outcome parity | PASS | Exact per-ID comparison: [g3_comparison.json](g3_comparison.json), [g3_ids.tsv](g3_ids.tsv), [g3_production.json](g3_production.json), [g3_port.json](g3_port.json), both arm logs |
| G4 staging | YES, not executed | Static/CPU preparation: [STAGED_GPU_LINES.sh](STAGED_GPU_LINES.sh), [EXPECTED.json](EXPECTED.json), [EXPECTED.sha256](EXPECTED.sha256), [g4_staging.json](g4_staging.json), [INPUT_CENSUS.json](INPUT_CENSUS.json) |

Registration: immutable committed `orders/APA_P1_PORT_PACKED.md`, [REGISTRATION.md](REGISTRATION.md), [G4_REGISTRATION.md](G4_REGISTRATION.md). No changes to the order or registrations. This is the additive port slice only; acceptance does not switch a default or promote the path.

## Mechanism and precision

The port retains exact bf16 K and V. Its bulk is **stored** as packed TQ4 4-bit centroid codes plus one fp32 norm per key: 64+4 bytes at D128, or 4.25 stored bits/element. It is **computed** through the inherited variant-B split-bf16 hi/lo tensor-core products, with fp32 accumulation, norm and score scaling. The tail reads the exact bf16 K, uses bf16 MMA with fp32 accumulation, and all keys remain in the denominator. P rounds to bf16 before the inherited PV accumulation. No header arithmetic was edited.

SP token storage is `256 K + 256 V + 64 codes + 4 norm = 580 B/key/KV head`. SP-tk8 adds 128 B of int8 V and 4 B of V scale: **712 B/key/KV head**. Layer 0 remains exact bf16 K/V, 512 B/key/KV head. The 580-byte claim applies to SP token state, excludes shared tables and decode scratch, and makes no NVML-residency or performance claim. The symmetric-7 2P-INT4 comparator has equal token bytes but different codes/arithmetic.

Preflight read `docs/APA.md`, the paper's sections 2–3, and pinned survey S2/S5 at `e5538604`. These gates exercise packed bulk, exact tail, append and the additive production API seams. They do not evaluate model quality, performance, S3 hardening/promotion, the DS1 pitch ABI or a blind red team. Those are not claimed fixed or completed.

## Hunk map

| Seam | Added lines | D2 source | Production bytes |
|---|---:|---|---|
| tensor_cuda/src/kernels.cu | 8662–8673 | aaa9b4ea:8191-8197 | unchanged |
| tensor_cuda/include/tc/ops.h | 305–350 | aaa9b4ea:302-342 | unchanged |
| tensor_cuda/src/bindings.cpp | 2738–2821 | aaa9b4ea:986-1064 (companion initializer only) | unchanged |
| tensor_cuda/tensor_cuda/__init__.py | 1538–1674 | aaa9b4ea:1290-1422 (loader/workspace/wrappers) | unchanged |
| tensor_cuda/CMakeLists.txt | 70–80 | survey S5 step 3 + aaa9b4ea:scripts/apa_ab1_build.sh main defines | unchanged |


The 16 new headers and five seams are the complete engine-source delta. All five production files retain their entire original byte prefix; **zero removed lines**. `git diff --stat 0e1dea3 -- tensor_cuda/` records 290 insertions in five tracked files; it omits the 16 untracked new headers, which are enumerated separately below. The only pre-existing non-engine committed delta from `0e1dea3` is the order file. The feature is explicitly default-off (`TC_APA_PACKED`); the companion initializer adds opt-in entries and does not redirect the existing SP/SP2 APIs. K1 AOT source/initializer/export is absent.

## Header identity table

Every reference is read with `git show aaa9b4ea:<path>`, never from lab working source files. All headers are under `tensor_cuda/src/apa_spo/`.

| Header | Lab/port common SHA-256 | Identity |
|---|---|---|
| d1.cuh | `e676bfdce86479abe9c213d87886a0c229659d28e8ab287bad0ec6f1713c8dc3` | PASS |
| parallel.cuh | `a42ee95b045b8dab6361a36d4672e3594c9e61b8de818dcffb920651b4324fa8` | PASS |
| tensor.cuh | `aa1f87c2d4b4cb0a8d86201cc765afe2c9b6c61bc22719b160ddb793005cbdbc` | PASS |
| packed.cuh | `62cc1d76e32256e57c1cc9099ddf0e052bbb45660542a2fe4726fe90f620905d` | PASS |
| launcher.cuh | `9111f312eaa39da8558f77c63d505bc4173cd1d844bcee647049eca5fb4df353` | PASS |
| baseline_dump.cuh | `9b7859de3e472624e63dcbca5e280f6ef6488bd6810970e4604252c52357ffea` | PASS |
| merge.cuh | `aa29e7ab057ce4c99735c6357b8c95b2673579abe2fd37bc3267de8db752c611` | PASS |
| packed_launcher.cuh | `2589204db7e9ea195bc9e9dd4db6895932123a9126153d1a28cfd5f8e927a0d5` | PASS |
| spq4.cuh | `c1da6e22f3ae3c1e32c6f604fc5a68e3b63ec61664bf102e52f37fd4c4fc99cb` | PASS |
| spq6.cuh | `f0f12741d3fd6ff34b5b18d6acb073bfb93cba961923fa4a6c6623396b3f3c4e` | PASS |
| eb2b_values.cuh | `93676e50290b02e6c4e5b21c9cf7646a4f56d7780fe6c0c52f08984a7be14afb` | PASS |
| eb2b.cuh | `046316cc424ee6384bb31c45547abcd84bd2ce701f6ea77ef2c9e7ab55afb7f7` | PASS |
| eb3_tuning.cuh | `87a24dccab5b928ccbb4a7b297af7965d8fd984e3630baac8e01a2bc7f45f64f` | PASS |
| eb3.cuh | `f030b8ab777c6522ab6a63d446164a1855bde8b44b4646a1e4ae1ee29d2622ff` | PASS |
| eb4_tuning.cuh | `4f7d5554b04f5169fe6d5480c145c0aa0e2c4b843cb6f2bfb5a79b51278db766` | PASS |
| eb4.cuh | `661778924e45d674db1f459ae97cb0eb29392b2187e30c64ffec1697acdafba5` | PASS |


## Build and SASS

CUDA **12.6 V12.6.85**, sm89, Release, CMake equivalent of `tensor_cuda/build.sh 89`, output `tensor_cuda/build-p1/`. CMake and nvcc commands/exits are in build.json; the existing production output location is preserved. The build uses offline read-only pybind11 sources, so no dependency git operation occurs.

Defines applied consistently to CUDA and C++: `TC_APA_PACKED=1`, `TC_APA_EB2B=1`, `TC_APA_EB3=1`, `TC_APA_EB4=1`, `APA_EB2B_MUTANT=0`, `APA_FXTC1_VARIANT_DEFAULT=2`. The inherited EB3/EB4 non-mutant defaults are unchanged.

| Module | SHA-256 |
|---|---|
| Port | `ae8e86bc6943732725cfd3bcfacc5e279e5f925da466e518c36c8b362a562ee5` |
| Production reference | `d6b1b7ad1d87e0d7d5eed70cff94b33e469e6f18b857d78679110e6b2dce3c24` |
| AB-1 main reference | `daa0b10c87222775a16b52a6811962b682319394c0f31675a924e68f5192933f` |

The lab survey worktree's main-build receipt points to the actual read-only AB-1 build in `tensor_fork_apa_wt_ab1/artifacts/apa_ab1/build/main/`. That module's SHA matches the pinned `aaa9b4ea` G0 receipt. Both baseline modules remain untouched. The port module is restored in the package after G3 and matches the saved build-p1 copy.

**SASS: 213/213 ported kernels match. 428,344 instructions in each arm. Zero instruction-count differences, zero opcode-histogram differences, zero instruction-text differences.** The census covers all new-header namespaces (spo2/spo4/spo5, spq1/spq4/spq6, EB2b/EB3/EB4) plus baseline_dump, including predecessor/ablation instantiations. cuobjdump 12.6 selects sm89 and counts every address-prefixed instruction including alignment NOPs. Matched mangled symbols require no path normalization. Difference/cause list: **empty**. Raw disassemblies: sass_port.txt and sass_lab.txt. Source paths differ; emitted instruction text does not.

## G3 CPU outcomes

**689/689 IDs matched exactly**, including every setup/call/teardown outcome string; zero collection errors and zero outcome differences. Threads were 1; CUDA_VISIBLE_DEVICES was empty. No production test was edited, deselected, suppressed or given an extra skip. Per-arm module SHA provenance is recorded in each JSON. This is an outcome-parity PASS, **not a passing-suite claim**.

| Outcome | Production | Port |
|---|---:|---:|
| PASSED | 410 | 410 |
| SKIPPED | 139 | 139 |
| FAILED | 125 | 125 |
| ERROR (setup) | 15 | 15 |
| Total IDs | 689 | 689 |

| Production test file | IDs | Production P/S/F/E | Port P/S/F/E | Differences |
|---|---:|---|---|---:|
| test_apa_sp1_1.py | 15 | 12/0/3/0 | 12/0/3/0 | 0 |
| test_apa_sp1_harness.py | 1 | 1/0/0/0 | 1/0/0/0 | 0 |
| test_apa_sp1_host.py | 15 | 0/0/0/15 | 0/0/0/15 | 0 |
| test_apa_sp1_reference.py | 21 | 20/0/1/0 | 20/0/1/0 | 0 |
| test_apa_sp2.py | 9 | 1/0/8/0 | 1/0/8/0 | 0 |
| test_apa_sp3.py | 59 | 56/0/3/0 | 56/0/3/0 | 0 |
| test_apa_sp3_a4.py | 30 | 30/0/0/0 | 30/0/0/0 | 0 |
| test_apa_sp3_a5.py | 24 | 22/0/2/0 | 22/0/2/0 | 0 |
| test_apa_sp3_a6.py | 44 | 43/0/1/0 | 43/0/1/0 | 0 |
| test_apa_sp4g.py | 48 | 43/0/5/0 | 43/0/5/0 | 0 |
| test_apa_sp4g_a2.py | 19 | 18/0/1/0 | 18/0/1/0 | 0 |
| test_apa_sp4g_a3.py | 21 | 20/0/1/0 | 20/0/1/0 | 0 |
| test_apa_sp4g_a4.py | 13 | 11/0/2/0 | 11/0/2/0 | 0 |
| test_apa_sp4g_a5.py | 28 | 26/0/2/0 | 26/0/2/0 | 0 |
| test_apa_sp4g_a6.py | 19 | 18/0/1/0 | 18/0/1/0 | 0 |
| test_apa_sp4g_a7.py | 30 | 29/0/1/0 | 29/0/1/0 | 0 |
| test_apa_spd1.py | 23 | 18/0/5/0 | 18/0/5/0 | 0 |
| test_dlpack.py | 257 | 42/139/76/0 | 42/139/76/0 | 0 |
| test_engine.py | 7 | 0/0/7/0 | 0/0/7/0 | 0 |
| test_ops_phase2.py | 6 | 0/0/6/0 | 0/0/6/0 | 0 |


P/S/F/E means passed/skipped/failed/setup error. Exact failure messages are retained in the per-arm JSONs, including unchanged missing historical snapshots/manifests and CUDA-no-device errors. Representative literal failures:

```text
FileNotFoundError: [Errno 2] No such file or directory: '<worktree>/artifacts/apa_sp1/build/manifest.json'
RuntimeError: cudaMalloc failed: no CUDA-capable device is detected
RuntimeError: empty: the current-device query cudaGetDevice failed (cudaErrorNoDevice: no CUDA-capable device is detected); nothing was allocated
```

The first path is shortened here only; the receipt retains its full path. Historical `load_runtime` pins are not bypassed, so some host guard tests stop before reaching an imported module. Existing no-device failures and missing fixtures are **not claimed fixed**. All module imports are independently provenance-checked by the pytest receipt plugin.

## Staged GPU lines and expectations

The lead runs from the worktree root:

```bash
bash artifacts/apa_p1/STAGED_GPU_LINES.sh
```

There are **51 literal lease-wrapped lines**. Every one has `forge-gpu-run --max 600`, an inner `timeout 540s`, and an outer `timeout 1750s`. GPU visibility is enabled only inside the lease; the seat never executes any of them. The script stops below the registered free-space rails and writes exit/argv/log-hash receipts only after each process exits. A window/tool/timeout failure remains RED. Each new run has a separate timestamped directory and create-only child receipts.

| Lines | Registered operands | Expected |
|---|---|---|
| 6 bitwise | Qwen3-1.7B L5 (G2), Qwen3-4B L7 (G4), KVH8, D128; S=2048,8192,32768; both EB3/SP and EB4/SP-tk8 | Every saved integer-view element equal to the lab main build; no tolerance |
| Same six lines: append | Prefix lengths 1,63,64,65,S-4,S-3,S-2,S-1,S; compact K/V; last four decode outputs | Codes, fp32 norm bits, valid-row history, outputs, masks, flags, int8 V and scale bits equal |
| 1 bytes | All 28/36 model layers, KVH8, each S | Actual unique Storage.nbytes = `8*S*(512+(NL-1)*580)` |
| 44 regression | 43 original APA/APAMQ files + DLPack, 2078 frozen collected IDs | Same exact per-ID phase/outcome strings under paired production/port modules; every difference includes messages |

The numerical workers read real SPQ3 post-norm/post-RoPE captures for 2K/8K prefixes and EB4 P0 captures at 32K. SHA pins come from `aaa9b4ea` receipts. Each bf16 upload is checked against the captured integer bits. Every saved reference array must be nonzero and nonconstant; no vacuous equality is accepted. Diagnostic and nondiagnostic outputs must also agree within each worker. Delta float32(8.1), row carry, EB3 variant 0 and registered EB4 fractions .02/.05 are unchanged. Shared rotation/codebook state is also compared. Source/module/capture drift fails closed.

| S | Expected total SP token bytes, 1.7B | Expected total SP token bytes, 4B |
|---:|---:|---:|
| 2048 | 264,962,048 | 340,983,808 |
| 8192 | 1,059,848,192 | 1,363,935,232 |
| 32768 | 4,239,392,768 | 5,455,740,928 |

Reference arithmetic: pinned AB-1 G2 allocation records in reference_g2.json. This check covers actual Storage payload allocations, excludes shared tables/transient scratch, and does not assert the older AB-1 overall G2/NVML gate passed. Expected GPU test outcomes are generated by the paired production baseline in the same staged line; CPU skip/failure strings are not substituted for a GPU reference. G3's own CPU baseline remains g3_production.json.

EXPECTED.json SHA-256: `461a6a9c5abcd81c0fa7f6386639b5bdae2352ad6d0b67f264b7fc176f4fe050`. Full frozen test IDs, all command lines and all source/module pins are in that file. Literal staged commands follow:

```bash
run_line bitwise_17b_2048 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-bitwise_17b_2048 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py bitwise --model Qwen3-1.7B --S 2048 --out "$RUN/bitwise_17b_2048" || exit 2
run_line bitwise_17b_8192 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-bitwise_17b_8192 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py bitwise --model Qwen3-1.7B --S 8192 --out "$RUN/bitwise_17b_8192" || exit 2
run_line bitwise_17b_32768 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-bitwise_17b_32768 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py bitwise --model Qwen3-1.7B --S 32768 --out "$RUN/bitwise_17b_32768" || exit 2
run_line bitwise_4b_2048 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-bitwise_4b_2048 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py bitwise --model Qwen3-4B --S 2048 --out "$RUN/bitwise_4b_2048" || exit 2
run_line bitwise_4b_8192 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-bitwise_4b_8192 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py bitwise --model Qwen3-4B --S 8192 --out "$RUN/bitwise_4b_8192" || exit 2
run_line bitwise_4b_32768 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-bitwise_4b_32768 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py bitwise --model Qwen3-4B --S 32768 --out "$RUN/bitwise_4b_32768" || exit 2
run_line bytes timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-bytes -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py bytes --out "$RUN/bytes" || exit 2
run_line reg_apa_int4_sdpa_noncausal timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_int4_sdpa_noncausal -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_int4_sdpa_noncausal.py --out "$RUN/reg_apa_int4_sdpa_noncausal" || exit 2
run_line reg_apa_ld1_pdwl timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_ld1_pdwl -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_ld1_pdwl.py --out "$RUN/reg_apa_ld1_pdwl" || exit 2
run_line reg_apa_ld2_pdwl timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_ld2_pdwl -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_ld2_pdwl.py --out "$RUN/reg_apa_ld2_pdwl" || exit 2
run_line reg_apa_mf1_quant timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf1_quant -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf1_quant.py --out "$RUN/reg_apa_mf1_quant" || exit 2
run_line reg_apa_mf2_1_definition timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf2_1_definition -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf2_1_definition.py --out "$RUN/reg_apa_mf2_1_definition" || exit 2
run_line reg_apa_mf2_1_gaps timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf2_1_gaps -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf2_1_gaps.py --out "$RUN/reg_apa_mf2_1_gaps" || exit 2
run_line reg_apa_mf2_1_sentinel timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf2_1_sentinel -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf2_1_sentinel.py --out "$RUN/reg_apa_mf2_1_sentinel" || exit 2
run_line reg_apa_mf2_1_short timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf2_1_short -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf2_1_short.py --out "$RUN/reg_apa_mf2_1_short" || exit 2
run_line reg_apa_mf2_2_n1 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf2_2_n1 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf2_2_n1.py --out "$RUN/reg_apa_mf2_2_n1" || exit 2
run_line reg_apa_mf2_2_n2 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf2_2_n2 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf2_2_n2.py --out "$RUN/reg_apa_mf2_2_n2" || exit 2
run_line reg_apa_mf2_2a_blocked timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf2_2a_blocked -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf2_2a_blocked.py --out "$RUN/reg_apa_mf2_2a_blocked" || exit 2
run_line reg_apa_mf2_2b_blocked timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf2_2b_blocked -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf2_2b_blocked.py --out "$RUN/reg_apa_mf2_2b_blocked" || exit 2
run_line reg_apa_mf2_2c_longkeys timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf2_2c_longkeys -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf2_2c_longkeys.py --out "$RUN/reg_apa_mf2_2c_longkeys" || exit 2
run_line reg_apa_mf2_2c_nits timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf2_2c_nits -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf2_2c_nits.py --out "$RUN/reg_apa_mf2_2c_nits" || exit 2
run_line reg_apa_mf2_mask timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf2_mask -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf2_mask.py --out "$RUN/reg_apa_mf2_mask" || exit 2
run_line reg_apa_mf2_rule timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_mf2_rule -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_mf2_rule.py --out "$RUN/reg_apa_mf2_rule" || exit 2
run_line reg_apa_om135_imm timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_om135_imm -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_om135_imm.py --out "$RUN/reg_apa_om135_imm" || exit 2
run_line reg_apa_phase6 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_phase6 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_phase6.py --out "$RUN/reg_apa_phase6" || exit 2
run_line reg_apa_pidv timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_pidv -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_pidv.py --out "$RUN/reg_apa_pidv" || exit 2
run_line reg_apa_selective timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_selective -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_selective.py --out "$RUN/reg_apa_selective" || exit 2
run_line reg_apa_selective_int4 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_selective_int4 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_selective_int4.py --out "$RUN/reg_apa_selective_int4" || exit 2
run_line reg_apa_selective_splitk timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_selective_splitk -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_selective_splitk.py --out "$RUN/reg_apa_selective_splitk" || exit 2
run_line reg_apa_sp1_1 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp1_1 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp1_1.py --out "$RUN/reg_apa_sp1_1" || exit 2
run_line reg_apa_sp1_harness timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp1_harness -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp1_harness.py --out "$RUN/reg_apa_sp1_harness" || exit 2
run_line reg_apa_sp1_host timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp1_host -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp1_host.py --out "$RUN/reg_apa_sp1_host" || exit 2
run_line reg_apa_sp1_reference timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp1_reference -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp1_reference.py --out "$RUN/reg_apa_sp1_reference" || exit 2
run_line reg_apa_sp2 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp2 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp2.py --out "$RUN/reg_apa_sp2" || exit 2
run_line reg_apa_sp3 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp3 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp3.py --out "$RUN/reg_apa_sp3" || exit 2
run_line reg_apa_sp3_a4 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp3_a4 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp3_a4.py --out "$RUN/reg_apa_sp3_a4" || exit 2
run_line reg_apa_sp3_a5 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp3_a5 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp3_a5.py --out "$RUN/reg_apa_sp3_a5" || exit 2
run_line reg_apa_sp3_a6 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp3_a6 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp3_a6.py --out "$RUN/reg_apa_sp3_a6" || exit 2
run_line reg_apa_sp4g timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp4g -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp4g.py --out "$RUN/reg_apa_sp4g" || exit 2
run_line reg_apa_sp4g_a2 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp4g_a2 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp4g_a2.py --out "$RUN/reg_apa_sp4g_a2" || exit 2
run_line reg_apa_sp4g_a3 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp4g_a3 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp4g_a3.py --out "$RUN/reg_apa_sp4g_a3" || exit 2
run_line reg_apa_sp4g_a4 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp4g_a4 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp4g_a4.py --out "$RUN/reg_apa_sp4g_a4" || exit 2
run_line reg_apa_sp4g_a5 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp4g_a5 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp4g_a5.py --out "$RUN/reg_apa_sp4g_a5" || exit 2
run_line reg_apa_sp4g_a6 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp4g_a6 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp4g_a6.py --out "$RUN/reg_apa_sp4g_a6" || exit 2
run_line reg_apa_sp4g_a7 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_sp4g_a7 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_sp4g_a7.py --out "$RUN/reg_apa_sp4g_a7" || exit 2
run_line reg_apa_spd1 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_spd1 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_spd1.py --out "$RUN/reg_apa_spd1" || exit 2
run_line reg_apa_tiled timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_tiled -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_tiled.py --out "$RUN/reg_apa_tiled" || exit 2
run_line reg_apa_value_dim timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apa_value_dim -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apa_value_dim.py --out "$RUN/reg_apa_value_dim" || exit 2
run_line reg_apamq_df1 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apamq_df1 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apamq_df1.py --out "$RUN/reg_apamq_df1" || exit 2
run_line reg_apamq_sb1 timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_apamq_sb1 -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_apamq_sb1.py --out "$RUN/reg_apamq_sb1" || exit 2
run_line reg_dlpack timeout 1750s forge-gpu-run --min 120 --max 600 --label APA-P1-reg_dlpack -- timeout 540s env CUDA_VISIBLE_DEVICES=0 APA_P1_LEAD_GPU=1 python artifacts/apa_p1/gpu_driver.py regression --test-file tensor_cuda/tests/test_dlpack.py --out "$RUN/reg_dlpack" || exit 2
```

## Prior art

Taken unchanged: Project-Tensor SPQ1/SPQ4/SPQ6/EB2b/EB3/EB4 (2026). Git (2005) three-way merge supplies survey S5's dry-run method, cited at every seam; this order uses additive assembly and does not perform a merge. Header annotations retain TurboQuant (Zandieh et al., 2025), BLASST (Yuan et al., 2025, running-max criterion), Flash-Decoding (Dao et al., 2023), FlashInfer (Ye et al., 2025), KIVI (Liu et al., 2024), ThriftAttention (2026), NVIDIA threadFenceReduction, split-bf16 and prefetch lineage. External metadata is inherited and **unverified — lead to check**; search leads already at code sites include “CUTLASS swap AB small M”, “swapAB GEMM tensor core decode”, “TurboQuant 2504.19874”, “BLASST 2512.12087”, “ThriftAttention block precision”, “NVIDIA threadFenceReduction”. No new literature search is claimed.

Staging takes AB-1 reference-pair/raw-integer checks and SPQ3/EB4 captures (2026), pybind11 cross-module access (2016), CUDA memcpy/storage metadata, SHA-256 (NIST 2015) and pytest's report API. The production accessor is the exact AB-1 source rebuilt against production headers; its read-only CPU pointer/byte/alias smoke passed. No numerical or selection algorithm is introduced. **No prior art known to me for this exact hunk-selection/port assembly; no novelty claimed.** Code-site seam notes, original header annotations and LEDGER.md carry the lineage.

## Files written and leave-and-list

The complete per-file inventory is [FILES_WRITTEN.md](FILES_WRITTEN.md), with machine-readable inventory [FILES_WRITTEN.json](FILES_WRITTEN.json). It enumerates all 16 headers, five appended seams, package module, every build-p1 output, every apa_p1 artifact and retained /tmp/apa_p1_seat scratch entry. Nothing was moved or deleted; no DELETE_ME entry is needed. No direct writes to logs/. No git writes, subagents, background shell jobs, GPU calls with a visible device, service changes or baseline writes. Source tests and every other engine file remain untouched.

Budget: started 2026-11-02T18:06Z; report synthesized 2026-11-02T18:32:34.026010+00:00, below the 3-hour rail. Every seat command ran with empty CUDA_VISIBLE_DEVICES and an explicit timeout below 30 minutes; running foreground commands were polled only at the requested maximum wait. Build/scratch remain in the authorized locations. The final default switch, promotion requirements, DS1 pitch ABI, red team, model quality and performance remain for later orders.
