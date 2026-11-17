# Recorded check commands

All child commands inherited the registered environment: TMPDIR=/tmp/p1_rev2_review, empty CUDA_VISIBLE_DEVICES, PYTHONDONTWRITEBYTECODE=1, OMP/OPENBLAS/MKL threads=1, under nice19 and an outer explicit timeout<=1200s.

Here-doc Python command input is retained in the session tool transcript. Decisive arithmetic/SASS/symbol/array procedures have standalone source files here; receipts retain their actual output.

## 001_diff_stat

```sh
git diff --stat e2fb85a1 710cc0d4 -- tensor_cuda
```

Output: receipts/001_diff_stat.log

## 002_revision_log

```sh
git log --oneline e2fb85a1..710cc0d4
```

Output: receipts/002_revision_log.log

## 003_instruments

```sh
bash -c 'git --version; python3 --version; python3 -c '"'"'import numpy; print("NumPy",numpy.__version__)'"'"'; python3 -m pytest --version; nm --version; cuobjdump --version; c++filt --version'
```

Output: receipts/003_instruments.log

## 004_blind_diff

```sh
git diff e2fb85a1 710cc0d4 -- tensor_cuda/CMakeLists.txt tensor_cuda/src tensor_cuda/tensor_cuda
```

Output: receipts/004_blind_diff.log

## 005_python_routes

```sh
bash -c 'nl -ba tensor_cuda/tensor_cuda/apa_packed.py; nl -ba tensor_cuda/tensor_cuda/apa_sp2.py; git diff e2fb85a1 710cc0d4 -- tensor_cuda/tensor_cuda/__init__.py'
```

Output: receipts/005_python_routes.log

## 006_launcher_diff

```sh
git diff e2fb85a1 710cc0d4 -- tensor_cuda/src/apa_spo/launcher.cuh tensor_cuda/src/apa_spo/packed_launcher.cuh tensor_cuda/src/apa_spo/packed.cuh tensor_cuda/src/apa_spo/merge.cuh tensor_cuda/src/apa_spo/spq4.cuh tensor_cuda/src/apa_spo/spq6.cuh
```

Output: receipts/006_launcher_diff.log

## 007_eb_diff

```sh
git diff e2fb85a1 710cc0d4 -- tensor_cuda/src/apa_spo/eb3.cuh tensor_cuda/src/apa_spo/eb4.cuh tensor_cuda/src/bindings.cpp
```

Output: receipts/007_eb_diff.log

## 008_entries_vectors

```sh
bash -c 'rg -n "uint[24]|float[24]|int[24]|reinterpret_cast" tensor_cuda/src/apa_spo/; nl -ba tensor_cuda/src/apa_spo/packed_launcher.cuh; sed -n "55,155p" tensor_cuda/src/apa_spo/launcher.cuh; rg -n "apa_selective_attention_sp|partition_layout|_tables|revision|struct NDArray|data_ptr\(|contiguous" tensor_cuda/src/tc/ndarray.h tensor_cuda/src/tc/tensor.h tensor_cuda/tensor_cuda/quant.py tensor_cuda/src/apa_sp*.cuh tensor_cuda/src/kernels.cu'
```

Output: receipts/008_entries_vectors.log

## 009_core_files

```sh
bash -c 'nl -ba tensor_cuda/src/apa_spo/packed_launcher.cuh; nl -ba tensor_cuda/src/apa_spo/launcher.cuh | head -n 165; nl -ba tensor_cuda/src/apa_spo/packed.cuh | tail -n 130; nl -ba tensor_cuda/src/kernels.cu | sed -n "8400,8525p"; rg --files tensor_cuda | rg "(ndarray|tensor.h|ops.h|apa_sp.*cuh)$"'
```

Output: receipts/009_core_files.log

## 010_selection

```sh
bash -c 'nl -ba tensor_cuda/src/apa_spo/d1.cuh; nl -ba tensor_cuda/src/apa_spo/parallel.cuh; nl -ba tensor_cuda/src/apa_sp1_1.cuh | sed -n "1,120p"; nl -ba tensor_cuda/src/apa_spo/launcher.cuh | sed -n "160,280p"; nl -ba tensor_cuda/tensor_cuda/quant.py | sed -n "1,170p"'
```

Output: receipts/010_selection.log

## 011_eb_launchers

```sh
bash -c 'nl -ba tensor_cuda/src/apa_spo/eb2b.cuh | sed -n "98,232p"; nl -ba tensor_cuda/src/apa_spo/eb3.cuh | sed -n "790,860p"; nl -ba tensor_cuda/src/apa_spo/eb3.cuh | sed -n "906,995p"; nl -ba tensor_cuda/src/apa_spo/eb4.cuh | sed -n "679,795p"'
```

Output: receipts/011_eb_launchers.log

## 012_dispatch_headers

```sh
bash -c 'nl -ba tensor_cuda/src/apa_spo/eb3.cuh | sed -n "996,1110p"; nl -ba tensor_cuda/src/apa_spo/eb4.cuh | sed -n "795,834p"; nl -ba tensor_cuda/src/apa_spo/spq6.cuh | sed -n "260,302p"; nl -ba tensor_cuda/src/apa_spo/spq4.cuh | sed -n "359,465p"; nl -ba tensor_cuda/src/bindings.cpp | sed -n "2720,2802p"; rg -n "apa_eb2b_run|apa_spq6_tq4_tc|apa_spq1" tensor_cuda/include/tc/ops.h; rg --files tensor_cuda/include/tc | head -n 20'
```

Output: receipts/012_dispatch_headers.log

## 013_core_identity

```sh
bash -c 'rg -n "struct NDArray|struct Storage|revision|data_ptr|NDArray\(|CheckedDeviceGuard" tensor_cuda/include/tc/core.h; rg -n "NDArray::NDArray|Storage::|revision|CheckedDeviceGuard" tensor_cuda/src/core.cpp; nl -ba tensor_cuda/src/apa_spo/merge.cuh; nl -ba tensor_cuda/tensor_cuda/apa_packed.py | sed -n "354,437p"; rg -n "revision|mark_modified|bump" tensor_cuda/src/bindings.cpp tensor_cuda/include/tc/autograd.h tensor_cuda/src/autograd.cpp; nl -ba tensor_cuda/src/apa_spo/eb3.cuh | sed -n "866,906p"'
```

Output: receipts/013_core_identity.log

## 014_storage_source

```sh
bash -c 'rg --files tensor_cuda/src | head -n 38; rg -n "NDArray::NDArray|Storage::Storage|void\* NDArray::data_ptr|struct CheckedDeviceGuard|class CheckedDeviceGuard" tensor_cuda/src tensor_cuda/include; nl -ba tensor_cuda/include/tc/core.h | sed -n "45,140p"; nl -ba tensor_cuda/src/apa_sp1_1.cuh | sed -n "140,200p"; rg -n "apa_int4_fixed_partition_layout" tensor_cuda/src/kernels.cu tensor_cuda/src/apa_sp1_1.cuh'
```

Output: receipts/014_storage_source.log

## 015_merge_storage

```sh
bash -c 'nl -ba tensor_cuda/src/kernels.cu | sed -n "174,198p"; nl -ba tensor_cuda/src/kernels.cu | sed -n "531,545p"; rg -n "__global__ void apa_selective_merge_kernel" tensor_cuda/src/kernels.cu; nl -ba tensor_cuda/src/kernels.cu | sed -n "2080,2150p"; nl -ba tensor_cuda/src/kernels.cu | sed -n "2260,2280p"; nl -ba tensor_cuda/src/apa_spo/eb2b_values.cuh; nl -ba tensor_cuda/src/apa_spo/eb3.cuh | sed -n "568,610p"; nl -ba tensor_cuda/src/apa_spo/eb4.cuh | sed -n "282,312p"'
```

Output: receipts/015_merge_storage.log

## 016_blind_host

```sh
python3 artifacts/apa_p1_rev2/blind_host_checks.py
```

Output: receipts/016_blind_host.log

## 017_claims_orders

```sh
bash -c 'cat orders/APA_P1_REV2_KNOWN_ITEMS.md; cat orders/APA_P1D_PACKED_DEFAULT.md; cat orders/APA_P1F1_REV1_FIXES.md'
```

Output: receipts/017_claims_orders.log

## 018_prior_findings

```sh
python3 -
```

Output: receipts/018_prior_findings.log

## 019_hashes_reports

```sh
python3 -
```

Output: receipts/019_hashes_reports.log

## 020_dump_structure

```sh
python3 -
```

Output: receipts/020_dump_structure.log

## 021_test_sources

```sh
bash -c 'nl -ba tensor_cuda/tests/test_apa_p1f1_cpu.py | sed -n "1,210p"; nl -ba tensor_cuda/tests/test_apa_p1d_policy.py | sed -n "1,135p"; sed -n "1,145p" scripts/apa_p1f1_gpu.py; sed -n "1,170p" scripts/apa_p1d_sass.py'
```

Output: receipts/021_test_sources.log

## 022_test_tail

```sh
bash -c 'nl -ba tensor_cuda/tests/test_apa_p1f1_cpu.py | sed -n "210,356p"; nl -ba tensor_cuda/tests/test_apa_p1d_policy.py | sed -n "132,160p"; rg -n "^def test_|skip|expected_sha|tamper|sha256" tensor_cuda/tests/test_apa_p1d_policy.py; rg -n "def load|spec_from|import tensor|_tensor_cuda" scripts/apa_p1f1_gpu.py | head -n 32; nl -ba tensor_cuda/tests/conftest.py | head -n 100; sed -n "1,180p" scripts/apa_p1f1_sass.py'
```

Output: receipts/022_test_tail.log

## 023_sass_audit

```sh
python3 artifacts/apa_p1_rev2/sass_audit.py
```

Output: receipts/023_sass_audit.log

## 024_symbols

```sh
python3 artifacts/apa_p1_rev2/symbol_audit.py
```

Output: receipts/024_symbols.log

## 025_cpu_pytest

```sh
python3 -
```

Output: receipts/025_cpu_pytest.log

## 026_ptds_headers

```sh
bash -c 'rg -n "CUDA_API_PER_THREAD_DEFAULT_STREAM|__CUDA_API_PER_THREAD_DEFAULT_STREAM" /usr/local/cuda-12.6/include/cuda_runtime_api.h /usr/local/cuda-12.6/include/cuda_runtime.h /usr/local/cuda-12.6/include/crt/host_runtime.h; /usr/local/cuda-12.6/bin/nvcc --version; rg -n "default.stream|cuda_runtime" tensor_cuda/CMakeLists.txt; nl -ba tensor_cuda/src/kernels.cu | head -n 72; rg -n "^def lf[1-6]|np.save|save\(|nonfinite|nan|inf|zero|int4_stats|align|slots" scripts/apa_p1f1_gpu.py'
```

Output: receipts/026_ptds_headers.log

## 027_nvcc_preprocess

```sh
python3 -
```

Output: receipts/027_nvcc_preprocess.log

## 028_tickets_tests_source

```sh
bash -c 'nl -ba tensor_cuda/src/apa_spo/eb4.cuh | sed -n "82,118p"; nl -ba tensor_cuda/src/apa_spo/eb4.cuh | sed -n "182,246p"; nl -ba tensor_cuda/src/apa_spo/eb3.cuh | sed -n "86,165p"; rg -n "CHOOSE_SITE|COUNTER_MIN|REG_|VARIANT_DEFAULT" tensor_cuda/src/apa_spo/eb*_tuning.cuh tensor_cuda/src/apa_spo/spq6.cuh; nl -ba tensor_cuda/src/kernels.cu | sed -n "2149,2194p"; nl -ba scripts/apa_p1f1_gpu.py | sed -n "453,512p"; nl -ba scripts/apa_p1f1_gpu.py | sed -n "899,958p"'
```

Output: receipts/028_tickets_tests_source.log

## 029_gpu_receipt_inventory

```sh
python3 -
```

Output: receipts/029_gpu_receipt_inventory.log

## 030_receipt_structures

```sh
python3 -
```

Output: receipts/030_receipt_structures.log

## 031_receipt_structures_repair1

```sh
python3 -
```

Output: receipts/031_receipt_structures_repair1.log

## 032_nonfinite_row_diffs

```sh
python3 -
```

Output: receipts/032_nonfinite_row_diffs.log

## 033_hook_loader_binary

```sh
python3 -
```

Output: receipts/033_hook_loader_binary.log

## 034_vector_cast_census

```sh
python3 -
```

Output: receipts/034_vector_cast_census.log

## 035_receipt_audit

```sh
python3 artifacts/apa_p1_rev2/receipt_audit.py
```

Output: receipts/035_receipt_audit.log

## 036_more_source_receipts

```sh
python3 -
```

Output: receipts/036_more_source_receipts.log

## 037_device_coverage

```sh
bash -c 'rg -n "RTX|NVIDIA|UUID|GPU-|device|cuda_visible_devices" /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d/artifacts/apa_p1f1/gpu/LF1_r2.log /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d/artifacts/apa_p1f1/gpu/LF1_r2.start /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_p1d/artifacts/apa_p1f1/gpu/LF6a/test_apa_selective_F.json | head -n 20; nl -ba scripts/apa_p1f1_gpu.py | sed -n "313,376p"; nl -ba scripts/apa_p1f1_gpu.py | sed -n "749,872p"'
```

Output: receipts/037_device_coverage.log

## 038_ops_cmake_tests

```sh
bash -c 'nl -ba tensor_cuda/include/tc/ops.h | sed -n "288,348p"; nl -ba tensor_cuda/CMakeLists.txt | sed -n "1,105p"; git diff e2fb85a1 710cc0d4 -- tensor_cuda/tests/test_apa_p1_exact_tie.py; nl -ba scripts/apa_p1f1_gpu.py | sed -n "164,184p"; nl -ba tensor_cuda/src/apa_sp2.cuh | head -n 65; rg -n "cudaDevice|init_default|set_device" tensor_cuda/src/bindings.cpp | head -n 22'
```

Output: receipts/038_ops_cmake_tests.log

## 039_numeric_boundaries

```sh
python3 artifacts/apa_p1_rev2/numeric_boundaries.py
```

Output: receipts/039_numeric_boundaries.log

## 040_repro_compile

```sh
python3 -m py_compile artifacts/apa_p1_rev2/repro/r01_inf_key_default.py artifacts/apa_p1_rev2/repro/r02_finite_overflow.py
```

Output: receipts/040_repro_compile.log

## 041_regression_recompute

```sh
python3 -
```

Output: receipts/041_regression_recompute.log

## 042_repro_compile_final

```sh
python3 -m py_compile artifacts/apa_p1_rev2/repro/r01_inf_key_default.py artifacts/apa_p1_rev2/repro/r02_finite_overflow.py artifacts/apa_p1_rev2/repro/r03_empty_helpers.py
```

Output: receipts/042_repro_compile_final.log

## 043_policy_test_remainder

```sh
bash -c 'nl -ba tensor_cuda/tests/test_apa_p1d_policy.py | sed -n "158,456p"; nl -ba tensor_cuda/tests/test_apa_p1d_policy.py | sed -n "457,570p"'
```

Output: receipts/043_policy_test_remainder.log

## 044_instrument_help_scope

```sh
python3 -
```

Output: receipts/044_instrument_help_scope.log

## 045_public_route_census

```sh
bash -c 'rg -n "quantize_sp2_keys|apa_selective_attention_sp|storage=None|PackedSpKeys\(" tensor_cuda/tensor_cuda; rg -n "apa_p1f1_eb2b_run|apa_p1f1_merge_tickets|bulk_mma_tq<" tensor_cuda/src/apa_spo tensor_cuda/src/bindings.cpp; nl -ba tensor_cuda/src/bindings.cpp | sed -n "1180,1232p"'
```

Output: receipts/045_public_route_census.log

## 046_import_alignment_scope

```sh
bash -c 'rg -n "align|uintptr_t|byte_offset" tensor_cuda/src/bindings.cpp | head -n 32; nl -ba tensor_cuda/src/bindings.cpp | sed -n "1110,1182p"; rg --files tensor_cuda | rg "dlpack.*\\.(h|hpp)$"'
```

Output: receipts/046_import_alignment_scope.log

## 047_dlpack_alignment_contract

```sh
bash -c 'rg -n -A 24 -B 5 "align|void\\* data" tensor_cuda/include/dlpack/dlpack.h; nl -ba tensor_cuda/src/bindings.cpp | sed -n "1305,1322p"'
```

Output: receipts/047_dlpack_alignment_contract.log

## 048_q_inf_extremes

```sh
python3 -
```

Output: receipts/048_q_inf_extremes.log

## 049_final_source_slices

```sh
bash -c 'nl -ba scripts/apa_p1f1_gpu.py | sed -n "420,453p"; nl -ba tensor_cuda/src/apa_spo/spq4.cuh | sed -n "174,215p"; nl -ba tensor_cuda/src/apa_spo/eb4.cuh | sed -n "350,400p"; nl -ba tensor_cuda/src/apa_spo/eb3.cuh | sed -n "438,478p"; nl -ba tensor_cuda/src/kernels.cu | sed -n "2191,2199p"; nl -ba tensor_cuda/src/apa_spo/eb4.cuh | sed -n "43,66p"'
```

Output: receipts/049_final_source_slices.log

## 050_nonfinite_extremes_script

```sh
python3 artifacts/apa_p1_rev2/nonfinite_extremes.py
```

Output: receipts/050_nonfinite_extremes_script.log

## 051_final_compile

```sh
python3 -m py_compile artifacts/apa_p1_rev2/repro/r01_inf_key_default.py artifacts/apa_p1_rev2/repro/r02_finite_overflow.py artifacts/apa_p1_rev2/repro/r03_empty_helpers.py artifacts/apa_p1_rev2/write_report.py
```

Output: receipts/051_final_compile.log

## 052_write_report

```sh
python3 artifacts/apa_p1_rev2/write_report.py
```

Output: receipts/052_write_report.log

## 053_consistency_reads

```sh
bash -c 'nl -ba scripts/apa_p1f1_gpu.py | sed -n "259,312p"; rg -n "def apa_eb[234]|def _apa_spq4|check_source|_workspace_tuple" tensor_cuda/tensor_cuda/__init__.py; rg -n "_tensor_cuda_spo2|Register|timeout|script" artifacts/apa_p1_rev2/REPORT.md | tail -n 10'
```

Output: receipts/053_consistency_reads.log

## 054_timing_array_census

```sh
python3 -
```

Output: receipts/054_timing_array_census.log

## 055_eb2b_tick_lifetime

```sh
bash -c 'rg -n -A 3 -B 3 "ticks|stamp|sel_ticks|pv_ticks" tensor_cuda/src/apa_spo/spq4.cuh; nl -ba tensor_cuda/src/apa_spo/eb2b.cuh | sed -n "187,211p"'
```

Output: receipts/055_eb2b_tick_lifetime.log

## 056_receipt_audit_no_timing

```sh
python3 artifacts/apa_p1_rev2/receipt_audit.py
```

Output: receipts/056_receipt_audit_no_timing.log

## 057_final_report

```sh
python3 artifacts/apa_p1_rev2/write_report.py
```

Output: receipts/057_final_report.log

## 058_final_integrity

```sh
python3 -
```

Output: receipts/058_final_integrity.log
