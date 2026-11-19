# APA-PF2-F3 report

**G3 CPU, G4 CPU, G5 SASS and G6 passed. ACCEPT is pending the lead's real GPU lines.**
The shipped test uses the import-path engine and compiles the diagnostic
index helper in pytest's temporary directory. The norm witness includes
−Inf. Both raw `tc` kernels canonicalize non-finite stored norms to the
PF2-F1 quiet NaN on the default and mutation-hook loads.

## Gate evidence

| Gate | Result | Receipt |
|---|---|---|
| G1 finite identity | Staged, 5,600 inherited and 224 new cases | `LINES.json` |
| G2 non-finite | Staged, 16 sink, 16 norm, 44 partition and eight raw `tc` configurations | `LINES.json` |
| G3 mutants | CPU build, import, and 360 host guard checks passed. Device kills staged. | `build_pf3m*/BUILD.json`, `CPU_IMPORTS.log`, `HOST_GUARD.json` |
| G4 shipped tests | CPU passed, 77 collected. GPU staged, requires 77 passes. | `CPU_ARCHIVE.json`, `CPU_PYTEST.json` |
| G5 regression | Static SM89 comparison passed, 803 identical blocks and exactly ten changed `tc` blocks. Each prefill mutant changed 32 prefill blocks. P1 regression staged. | `SASS_COMPARISON.json`, `LINES.json` |
| G6 scope | PASS, three authorized product edits, zero missing paths, zero GPU leases | `SCOPE.json`, `FILES_WRITTEN.txt`, `DRY_RUNS.json` |

The 224 finite `tc` cases cover every runtime-reachable specialization in the
hooks-off builds, with int4 and tq4, both D and VD widths, both diagnostics
settings, both carry values and G from 2 through 8. The F2/F3 SASS comparison
also includes the emitted TQ4 variants 0, 1 and 3. Source and host checks
cover both hook load indices. The GPU evidence is admission dry runs.

## Module SHA-256s

| Arm | SHA-256 |
|---|---|
| `F2` | `aefe2513a4d33d7310421f65e6619f0706de2889410836bda42193053f8ec97a` |
| `F3` | `1036ec930a1e15d6c21213d977be22fd7c3846814d783f8fea40949ebb8f2347` |
| `pf3m01_prefill_norm_revert` | `6d856546adf37b27f4ff82d457243f1f8ffc926aab9c33ec7f5944db8b5d9c1f` |
| `pf3m02_prefill_norm_pinf_only` | `3d550c8bd4c4f5d29fd5ef4db7f928df3ecd6f24ff31587b5be4ebb07212f850` |
| `pf3m03_tc_norm_revert` | `905d451556ae3b7e18bdc560a9cb8a9a369cfad8f9c13a0928503aa6e87dfbcb` |

## Fresh-tree CPU run

Archive HEAD was `66359102d5f01ac1f6ba853131a9aebabbe99722`. Its docs
and three original edit targets matched `1cb328d227cdc4ca3ee691444576d3ba4a187020`
byte for byte. The candidate test and fixture were overlaid in scratch.
The archive contained the old F2 arm manifest without a module.
`CUDA_VISIBLE_DEVICES` was unset. Pytest exited 0 and the JUnit check
confirmed the sole pass was the diagnostic tail index test.

```text
.sssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss [ 93%]
sssss                                                                    [100%]
=========================== short test summary info ============================
SKIPPED [16] tensor_cuda/tests/test_apa_pf2_nonfinite.py:95: CUDA unavailable
SKIPPED [16] tensor_cuda/tests/test_apa_pf2_nonfinite.py:101: CUDA unavailable
SKIPPED [44] tensor_cuda/tests/test_apa_pf2_nonfinite.py:107: CUDA unavailable
1 passed, 76 skipped in 1.76s
```

## Mutants

| Mutant | CPU result | Killing test staged |
|---|---|---|
| `pf3m01_prefill_norm_revert` | PASS. Two prefill guards reverted. One CUDA object rebuilt and linked. | `G3_pf3m01`, +Inf and −Inf fail all 16 norm cases |
| `pf3m02_prefill_norm_pinf_only` | PASS. Two prefill guards only canonicalize +Inf. One CUDA object rebuilt and linked. | `G3_pf3m02`, −Inf fails all 16 norm cases |
| `pf3m03_tc_norm_revert` | PASS. Both shared `tc` guards reverted. One CUDA object rebuilt and linked. | `G3_pf3m03`, both infinities fail all eight raw `tc` cases |

The staged mutant lines require the clean F3 controls and the expected
failure predicates. Device execution is pending.

## Diff

```diff
--- a/tensor_cuda/src/apa_spo/spq4.cuh
+++ b/tensor_cuda/src/apa_spo/spq4.cuh
@@ -106,11 +106,15 @@
     #pragma unroll
     for(int i=0;i<2;++i){
       int j=(warp+4*t)*8+2*tig+i;ks[t][i]=0.f;
+      if(j<count){
 #if APA_SPQ4_MUTANT==3
-      if(j<count)ks[t][i]=scales[(int64_t)bh*capacity+start+j+1];   // scale from the neighbour key
-#else
-      if(j<count)ks[t][i]=scales[(int64_t)bh*capacity+start+j];
-#endif
+        const float norm=scales[(int64_t)bh*capacity+start+j+1];   // scale from the neighbour key
+#else
+        const float norm=scales[(int64_t)bh*capacity+start+j];
+#endif
+        // PF2-F1 (Project-Tensor, 2026) and IEEE 754-2008: reuse the quiet-NaN norm guard on tc.
+        ks[t][i]=isfinite(norm)?norm:__int_as_float(0x7fc00000);
+      }
     }
   }
   #pragma unroll
--- a/tensor_cuda/src/apa_spo/spq6.cuh
+++ b/tensor_cuda/src/apa_spo/spq6.cuh
@@ -137,11 +137,15 @@
     #pragma unroll
     for(int i=0;i<2;++i){
       int j=(warp+4*t)*8+2*tig+i;ks[t][i]=0.f;
+      if(j<count){
 #if APA_SPQ6_MUTANT==3
-      if(j<count)ks[t][i]=scales[(int64_t)bh*capacity+start+j+1];   // norm from the neighbour key
-#else
-      if(j<count)ks[t][i]=scales[(int64_t)bh*capacity+start+j];
-#endif
+        const float norm=scales[(int64_t)bh*capacity+start+j+1];   // norm from the neighbour key
+#else
+        const float norm=scales[(int64_t)bh*capacity+start+j];
+#endif
+        // PF2-F1 (Project-Tensor, 2026) and IEEE 754-2008: reuse the quiet-NaN norm guard on tc.
+        ks[t][i]=isfinite(norm)?norm:__int_as_float(0x7fc00000);
+      }
     }
   }
   __syncthreads();   // ctab ready
--- a/tensor_cuda/tests/test_apa_pf2_nonfinite.py
+++ b/tensor_cuda/tests/test_apa_pf2_nonfinite.py
@@ -1,42 +1,95 @@
-"""PF2-F2: REV1's sink/norm witnesses and partition-carry decode coverage.
+"""PF2: REV1's sink/norm witnesses and partition-carry decode coverage.
 Prior art: IEEE 754-2008; Project-Tensor REV1 reproductions (2026), preserved
 operands and extended decode coverage by kernel and dtype. ISO C++17/CERT INT32-C boundary widening
 (unverified - lead to check). No new numerical algorithm.
 """
+import os
 from pathlib import Path
-import os
+import shutil
+import subprocess
 import sys
 import pytest
 
-ROOT=Path(__file__).resolve().parents[2]
-sys.path.insert(0,str(ROOT/'scripts'))
-import apa_pf2f2_nonfinite as N
+ROOT = Path(__file__).resolve().parents[2]
+sys.path.insert(0, str(ROOT / "scripts"))
+import apa_pf2f3_nonfinite as N
 
 
-def test_actual_diagnostic_tail_helper():
-    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
-        pytest.skip('CPU helper check requires CUDA_VISIBLE_DEVICES empty')
-    import apa_pf2f2_index as I
-    r=I.check(ROOT/'tensor_cuda/src/apa_spo/packed_prefill.cuh','pytest_F2')
-    assert r['status']=='PASS',r
+# PF2-F1/F2 (Project-Tensor, 2026): compile the actual index helper under
+# UBSan and exercise its C++17 int64_t additions across the INT_MAX boundary.
+HARNESS = r'''
+#include <climits>
+#include <iostream>
+#include <type_traits>
+int main() {
+  static_assert(std::is_same<decltype(diagnostic_tail_index(0,0)), int64_t>::value);
+  int tests=0;
+  for (const int S : {INT_MAX, INT_MAX-31}) {
+    for (int lane=0; lane<32; ++lane) {
+      const int64_t expected=int64_t(S)+int64_t(lane);
+      const int64_t got=diagnostic_tail_index(S,lane);
+      if(got!=expected || got<0) return 10;
+      const int64_t next=diagnostic_tail_index(got,32);
+      if(next!=expected+int64_t(32) || next<=got) return 11;
+      ++tests;
+    }
+  }
+  for(const int S : {1,17,31,32,33,257}) {
+    for(int smax=0;smax<=S;++smax) {
+      int counts[258]={};
+      for(int lane=0;lane<32;++lane)
+        for(int64_t j=diagnostic_tail_index(smax,lane);j<S;j=diagnostic_tail_index(j,32)) {
+          if(j<smax || j<0 || j>=S) return 12;
+          ++counts[j];
+        }
+      for(int j=0;j<S;++j) if(counts[j]!=(j>=smax ? 1 : 0)) return 13;
+      ++tests;
+    }
+  }
+  std::cout << "INDEX_PASS checks=" << tests << " boundary_lanes=64 additions=128\n";
+}
+'''
 
 
-@pytest.fixture(scope='module')
+def test_actual_diagnostic_tail_helper(tmp_path):
+    compiler = next(
+        (path for name in ("c++", "g++", "clang++") if (path := shutil.which(name))),
+        None,
+    )
+    if compiler is None:
+        pytest.skip("no host C++ compiler")
+
+    header = (ROOT / "tensor_cuda/src/apa_spo/packed_prefill.cuh").read_text()
+    assert "int64_t j = diagnostic_tail_index(smax, lane)" in header
+    assert "j = diagnostic_tail_index(j, 32)" in header
+    helper = header.split("// APA_PF2F1_INDEX_BEGIN\n", 1)[1].split("// APA_PF2F1_INDEX_END", 1)[0]
+    source = tmp_path / "index_check.cpp"
+    binary = tmp_path / "index_check"
+    source.write_text("#include <cstdint>\n#define __host__\n#define __device__\n" + helper + HARNESS)
+    compile_result = subprocess.run(
+        [compiler, "-std=c++17", "-O1", "-fsanitize=undefined",
+         "-fno-sanitize-recover=all", str(source), "-o", str(binary)],
+        capture_output=True, text=True, timeout=60, cwd=tmp_path,
+    )
+    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
+    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30, cwd=tmp_path)
+    assert result.returncode == 0, result.stdout + result.stderr
+    assert "INDEX_PASS checks=441 boundary_lanes=64 additions=128" in result.stdout
+
+
+@pytest.fixture(scope="module")
 def runtime():
-    if os.environ.get('CUDA_VISIBLE_DEVICES')=='':
-        pytest.skip('no GPU visible')
+    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
+        pytest.skip("no GPU visible")
     import torch
+
     if not torch.cuda.is_available():
-        pytest.skip('CUDA unavailable')
-    torch.set_num_threads(1)
-    torch.set_num_interop_threads(1)
-    os.environ['TC_APA_SP']='1'
-    label = os.environ.get('APA_PF2F2_ARM')
-    if label or (ROOT/'artifacts/apa_pf2f2/arms/F2/ARM.json').is_file():
-        import apa_pf2f2_cpu as C
-        return C.load_arm(label or 'F2')
+        pytest.skip("CUDA unavailable")
     import tensor_cuda
-    return tensor_cuda
+
+    with pytest.MonkeyPatch.context() as patch:
+        patch.setenv("TC_APA_SP", "1")
+        yield tensor_cuda
 
 
 @pytest.mark.parametrize('case',N.sink_cases())
```

`NORM_FIXTURE.diff` records changes to the copied non-finite fixture.
The host index test changes its scratch location and compiler skip.
All 76 GPU cases change engine selection. The 16 stored-norm cases also
add −Inf and check row parity for all three non-finite classes.

## STAGED_LINES.md in full

~~~markdown
# STAGED_LINES.md

69 GPU lines staged. All 69 dry runs returned 78 with
`would_start: true` and zero lease attempts. G2_sink runs first.
Run from the worktree root as the GPU lead.

```bash
set -eu
cd /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_pf2f3
export CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
line_names=$(timeout 15s python3 -B -P -c 'import json; print(" ".join(json.load(open("artifacts/apa_pf2f3/LINES.json"))))')
for line in $line_names; do
  FORGE_GPU_DRY_RUN=0 timeout 660s python3 -B -P scripts/apa_pf2f3_line.py "$line" --lead
done
```

| Lines in order | Count | Min seconds | Coverage |
|---|---:|---:|---|
| G2_sink, G2_norm, G2_partition, G2_tc | 4 | 30, 60, 90, 60 | 16 sink, 16 norm, 44 partition, eight `tc` configurations |
| G1_public_00–03, G1_raw_00–03, G1_extra_00–01 | 10 | 120 | 752 inherited finite decode cases |
| G1_prefill_000–045 | 46 | 500 | 4,848 inherited finite prefill cases |
| G1_tc_00–03 | 4 | 120 | 224 finite BF16 `tc` cases across the reachable instantiations |
| G3_pf3m01, G3_pf3m02, G3_pf3m03 | 3 | 120 | Clean controls and targeted device failures for each mutant |
| G4_pytest | 1 | 120 | All 77 shipped tests with the F3 package on PYTHONPATH |
| G5_regression | 1 | 120 | 643 inherited P1 packed decode regression predicates |

Each launcher verifies the module, Python package, and registered input
SHA-256s before invoking `forge-gpu-run --min <n> --max 540 --margin 30`.
The launcher sets the CUDA runtime path, thread environment, scratch,
and device visibility. Full dry argv and environment are in
`gpu/*_dry.start.json`. Admission, exit 78 and log hashes are in
`gpu/*_dry.exit.json` and `DRY_RUNS.json`.

The launchers record the lead's exits and log SHA-256s in
`gpu/*_real.exit.json`. Test verdicts go in `results/<line>/`. G3
requires exit 1 from the inner mutant worker and exit 0 from the clean
control. The enclosing line exits successfully only when the complete
registered failure and control predicates match. The loop stops at the
first failed line.
~~~

## Deviations

- The first archive provenance check raised
  `AssertionError: 66359102d5f01ac1f6ba853131a9aebabbe99722` because it
  expected the order's earlier doc commit. A second read-only archive
  verified identical original product and doc bytes. The fresh-tree
  run used actual branch HEAD and the candidate pytest overlay.
- Script repair 1 of 3. The first host guard extraction matched two
  multiplies and raised
  `ValueError: too many values to unpack (expected 1)`. Anchoring the
  expression to the `kn[jk]` load yielded 360 passing checks. No gate
  thresholds changed. The partial source is retained in scratch.
- The seat kept scratch and ran the GPU launchers in dry-run mode only.
  The release CUDA build and the three incremental clones use the pinned
  F2 flags. No commits or git writes.

## Prior art

- IEEE 754-2008 quiet NaN and Project-Tensor PF2-F1 (2026). The `tc`
  loads reuse the existing `isfinite` selection and `0x7fc00000`
  sentinel. The extension is the application to both `tc` branches.
  External clause wording unverified — lead to check
  “IEEE 754-2008 6.2 quiet NaN multiplication”.
- Project-Tensor PF2-REV2 (2026) supplied the −Inf negative-dot witness.
  This round applies it to the shipped prefill test and extends the
  same sign construction to raw int4 and tq4 `tc`.
- DeMillo, Lipton and Sayward (1978) supplied mutation analysis. The
  exact reverts and +Inf-only treatment are the registered F3 mutants.
  Exact unsigned comparisons and normalized kernel blocks reuse
  Project-Tensor PF2-F1/F2 (2026).
- The diagnostic boundary cases and UBSan check come from PF2-F1/F2
  (2026). ISO C++17 integer promotion and CERT INT32-C wording
  unverified — lead to check “C++17 integral conversion signed integer
  overflow CERT INT32-C”.

## Every file written

`FILES_WRITTEN.txt` lists all 760 retained worktree files
and 25536 retained scratch files by absolute path. It includes
the report and itself. The worktree entries are the three product files
in the diff, 12 scripts, and 745 files under
`artifacts/apa_pf2f3/`.

- `scripts/apa_pf2f3_build.py`
- `scripts/apa_pf2f3_cpu.py`
- `scripts/apa_pf2f3_gpu.py`
- `scripts/apa_pf2f3_identity_reference.py`
- `scripts/apa_pf2f3_line.py`
- `scripts/apa_pf2f3_nonfinite.py`
- `scripts/apa_pf2f3_p1_inputs.py`
- `scripts/apa_pf2f3_p1_regression.py`
- `scripts/apa_pf2f3_pytest.py`
- `scripts/apa_pf2f3_sass_reference.py`
- `scripts/apa_pf2f3_scope.py`
- `scripts/apa_pf2f3_stage.py`

Top-level artifact files are `CHANGES.diff`, `COPY_DEPS.json`, `CPU_ARCHIVE.json`, `CPU_IMPORTS.log`, `CPU_PYTEST.json`, `CPU_PYTEST.log`, `CPU_PYTEST.xml`, `DRY_RUNS.json`, `DRY_RUNS.log`, `FILES_WRITTEN.txt`, `HOST_GUARD.json`, `LEDGER.md`, `LINES.json`, `LINE_PINS.json`, `NORM_FIXTURE.diff`, `OPERAND_CENSUS.txt`, `PREFLIGHT_DRY.log`, `REGISTRATION.md`, `REGISTRATION.sha256`, `REPORT.md`, `SASS_COMPARISON.json`, `SCOPE.json`, `SCOPE_BEFORE.json`, `STAGED_LINES.md`, `TOOLS.log`.

Generated directories contain 55 arm files,
458
build files, and 207 dry-run files. Scratch is retained at
`/tmp/pf2f3_seat` and `/tmp/pf2f3_gpu`. No paths were deleted by the seat.
