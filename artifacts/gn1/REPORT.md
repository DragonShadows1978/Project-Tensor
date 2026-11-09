# GN1 report

ACCEPT for this round: **G1 PASS; G2 PASS; G3 PASS; G4 PASS.** G5 is NOT RUN, owned by the lead after clean cross-model review. Evidence class: source inspection, CPU unit tests, CPU engine build and CPU staging audit. Actual GPU forward/backward correctness, existing-suite outcomes and mutant kills are not claimed fixed or verified by this round.

## Gates and receipts

- **G1 PASS** — 7424 other tracked files byte-identical; only GroupNorm changed; new test untracked, separate addition diff. Receipts: `G1_scope.json`, `tracked_before.json`, `tracked_after.json`, `nn.diff`, `test_new.diff`, `changes.patch`.
- **G2 PASS** — two exact reference examples, three float64 finite-difference checks, refusal on CPU, both-arm hidden-CUDA import/collection; 299 new IDs, 306 focus IDs, 362 regression IDs. Receipts: `reference_checks.log`, `cuda126_cpu_reference_and_refusal.log`, `cuda126_cpu_import_collection.json`, `ids_candidate.txt`, `test_id_tolerances.json`, `conditioning_census.json`.
- **G3 PASS** — CUDA 12.6 Release sm89 build; module pinned; 72 unchanged source inputs. Receipts: `configure_cuda126.json`, `build_cuda126.json`, `engine_pin.json`, `engine_sources_before.json`, `engine_sources_after.json`, `symbols_nm.txt`, `symbols_strings.txt`, `module_ldd.txt`, `nvcc_cuda126_version.txt`.
- **G4 PASS** — 18 bounded lines with expected child/parent exits; 14 one-edit mutants; CPU stand-in audit child=0, timeout=false, parent=0. Receipts: `STAGED_GPU_COMMANDS.sh`, `gpu_lines.json`, `MUTANTS.md`, `mutants.json`, `mutant_validation.json`, `staging_seal.json`, `audit_complete_seal/START.json`, `audit_complete_seal/candidate_focus/parent_result.json`.

G2's hand examples have group means (1,5), variances (1,1), eps=3, exact output (0,4,0,8); and means (1/2,5/2), variances (1/4,1/4), eps=3/4, exact output (((0,2),(-3,1))). All equalities use array_equal. Three central-difference tests independently validate dx, dweight and dbias. Both arms import tensor_cuda.nn without a visible device. The candidate refusal test also passes before device allocation.

## Fix and source evidence

Root cause: baseline nn.py:520 passes -1 to reshape; kernels.cu:757-760 checks literal element counts and does not infer it. Source evidence and historical TX3 W5 are in mechanism_evidence.txt. TX3's broader run was explicitly VOID and is not adopted as clean suite evidence.

The fix refuses GroupNorm(4,6) with a ValueError naming 4 and 6, and replaces -1 with (C//G)*math.prod(spatial). Constructor signature, parameter initialization, eps and shape are unchanged. ops.cpp:915-918 already defines biased population variance as mean((x-mean(x))**2); no correction is needed. Only GroupNorm's bytes changed in nn.py.

The new test is untracked (no git writes allowed), so tracked_scope.diff contains only nn.py; test_new.diff and changes.patch include its full addition. Other 7424 tracked files are byte-identical before/after.

### nn.py diff (verbatim)

```diff
diff --git a/tensor_cuda/tensor_cuda/nn.py b/tensor_cuda/tensor_cuda/nn.py
index 673cc988..781b488b 100644
--- a/tensor_cuda/tensor_cuda/nn.py
+++ b/tensor_cuda/tensor_cuda/nn.py
@@ -510,6 +510,13 @@ class BatchNorm1D(Module):
 class GroupNorm(Module):
     def __init__(self, num_groups, num_channels, eps=1e-5):
         super().__init__()
+        # Prior art: Wu and He (2018), Group Normalization; PyTorch GroupNorm
+        # divisibility refusal (docs unverified — lead to check: "GroupNorm").
+        if num_channels % num_groups != 0:
+            raise ValueError(
+                f"num_channels ({num_channels}) must be divisible by "
+                f"num_groups ({num_groups})"
+            )
         self.G, self.C, self.eps = num_groups, num_channels, eps
         self.weight = parameter(np.ones(num_channels))
         self.bias = parameter(np.zeros(num_channels))
@@ -517,7 +524,10 @@ class GroupNorm(Module):
     def forward(self, x):
         N, C = x.shape[0], x.shape[1]
         spatial = list(x.shape[2:])
-        g = x.reshape([N, self.G, -1])
+        # Explicit extent: NDArray::reshape does not infer -1.
+        g = x.reshape([N, self.G, (C // self.G) * math.prod(spatial)])
+        # ops.cpp::var computes mean((x - mean(x))**2), already biased (ddof=0),
+        # as required by Wu and He (2018) and PyTorch GroupNorm.
         mean = g.mean([2], True); var = g.var([2], True)
         g = (g - mean) * (var + self.eps).pow(-0.5)
         g = g.reshape([N, C] + spatial)
```

## Tests, exact IDs and tolerances

299 new IDs: 144 ordinary forward cases, 142 ordinary backward cases, 6 near-eps cases, 2 binary-exact hand examples, 3 finite-difference operand checks, constructor defaults and divisibility refusal. Ranks 2-5; N=1/2; C=6; G=1/2/6 plus G=3 near-eps; default/nondefault epsilon; nontrivial signed weights and biases; fixed seeded random output projection for backward. Both focus files collect 306 IDs; all ten nn-importing test files collect 362 on both arms.

Reference statistics and VJPs use NumPy float64 on represented source operands, independently of TensorCUDA's computed outputs. All three reached leaves must have correct gradients; no parameter is silently omitted.

# GN1 numerical budgets, registered before GPU

REGISTRATION.md is the immutable original; AMENDMENT_01.md states the sole representability amendment. No tolerance was changed. Every numeric comparison uses abs(a-b) <= atol + rtol*abs(reference), via numpy.allclose. NaN never passes.

| Dtype / lane | atol | rtol | Derivation |
|---|---:|---:|---|
| float32 forward | 3e-5 | 3e-5 | n<=72; FP32 sum gamma_71<4.24e-6; ~7x reduction budget includes centering and inverse std |
| float32 backward (all x,w,b) | 3e-4 | 3e-4 | ~70x gamma_71 covers graph paths, broadcasts and cancellation; near-eps inverse std <=46 |
| float16 forward | 0.02 | 0.02 | u=2^-11; about 40u for rounded mean/sub/mul/var/pow/affine intermediates |
| float16 backward (all x,w,b) | 0.06 | 0.06 | about 120u for VJP intermediates and accumulated paths; overflow domain excluded by AMENDMENT_01 |
| bfloat16 forward | 0.12 | 0.12 | u=2^-8; about 30u for rounded intermediates |
| bfloat16 backward (all x,w,b) | 0.35 | 0.35 | about 90u for VJP cancellation and path accumulation |
| CPU hand-computed reference | 0 | 0 | Means 1,5 and 1/2,5/2; variances 1,1 and 1/4,1/4; sqrt(var+eps)=2 or 1 exactly in binary |
| CPU analytic vs central differences | 2e-7 | 2e-7 | h=2^-18; O(h^2) truncation and O(eps64/h) subtraction, bounded small fixture with eps=0.125 |

The finite test domain has |x|<=2.25, |weight|<=1.5, |projection|<=1, non-singleton ordinary variance>=1/64, exact-zero singleton centers and separately constructed variance=eps cases. conditioning_census.json verifies operand conditioning without a GPU (not engine correctness). References use represented source operands, so input quantization is not counted as kernel error. Absolute floors cover near-zero cancellation. These are conservative engineering budgets for this census, not a universal forward-error proof. FP32 witnesses pin the mutants independently of loose low-precision budgets.

Prior art: standard floating-point gamma_n error bound (Higham, Accuracy and Stability of Numerical Algorithms, 1996/2002; unverified — lead to check), IEEE 754 roundoff (1985/2019), and classical central differences (Fornberg 1988 lead). Ours: finite-domain budgets chosen before device data.

Exact per-ID tolerance assignment is test_id_tolerances.json. The complete ID list follows verbatim (the same bytes as ids_candidate.txt and ids_baseline.txt):

```text
tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_hand_values[two_groups_rank2]
tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_hand_values[per_channel_rank3]
tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_central_differences[x]
tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_central_differences[weight]
tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_central_differences[bias]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r4-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r4-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r4-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r4-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r4-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r4-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r4-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r4-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r4-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r4-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r4-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r4-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r5-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r5-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r5-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r5-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r5-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r5-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r5-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r5-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r5-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r5-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r5-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r5-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r2-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r2-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r2-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r2-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r2-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r2-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r2-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r2-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r2-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r2-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r2-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r2-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r3-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r3-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r3-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r3-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r3-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r3-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r3-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r3-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r3-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r3-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r3-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r3-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r4-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r4-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r4-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r4-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r4-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r4-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r4-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r4-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r4-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r4-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r4-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r4-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r5-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r5-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r5-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r5-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r5-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r5-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r5-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r5-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r5-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r5-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r5-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float32-r5-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g1-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g1-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g2-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g2-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g6-default]
tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g6-eps0125]
tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-float32]
tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-float16]
tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-bfloat16]
tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[backward-float32]
tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[backward-float16]
tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[backward-bfloat16]
tensor_cuda/tests/test_groupnorm_gn1.py::test_constructor_defaults
tensor_cuda/tests/test_groupnorm_gn1.py::test_nondivisible_refusal
```

## Dtype census

# GN1 dtype census (source evidence; no GPU execution)

| Dtype | mean/var | pow and elementwise | GN1 coverage |
|---|---|---|---|
| float32 | Supported; FP32 sum/output | Supported | All forward/backward, default and nondefault eps |
| float16 | FP32 reduction accumulator, stored as FP16 before scalar divide; var is composed mean/sub/mul/mean | FP32 arithmetic, store each op as FP16; same-dtype binary operands required | Same-dtype detached leaf affine parameters; all forward, backward except the two default-eps rank-2 G=C cases, recorded before GPU in AMENDMENT_01 |
| bfloat16 | FP32 reduction accumulator, stored as BF16 | FP32 arithmetic, store each op as BF16 | Same-dtype detached leaf affine parameters; all forward/backward |
| float64 | Not an engine DType | Not supported | NumPy reference only. Public tensor(dtype="float64") silently falls back to FP32; it does not test float64 compute |
| int64, uint8, bool | Floating reduction dispatch refuses nonempty tensors | Floating pow/elementwise dispatch refuses | Not a supported normalization path; no numeric comparison claimed |

Citations: kernels.cu:568-574 DISPATCH_FLOAT; :896-913 ew_binary checks identical dtypes; :915-924 ew_scalar; :978-1017 reduce_lastdim_kernel uses float accumulation and st<T> output; :1068-1096 reduction dispatcher; :1200-1204 ew_pow uses DISPATCH_FLOAT. ops.cpp:817-822 mean=sum/count; :915-918 var=mean((x-mean)**2), explicitly population variance ddof=0; :854-859 pow backward materializes x**(p-1). bindings.cpp:1808-1811 requires_grad setter/grad accessor; :100-106 BF16 numpy export casts to FP32. __init__.py:30-38 dtype mapping, :54-61 BF16 cast construction.

Constructor affine tensors remain FP32 as required; mixed FP16/BF16 x with uncast default affine parameters is not supported by ew_binary (dtype mismatch). Tests explicitly supply matching typed parameters and check output dtype. Detach makes BF16 cast outputs genuine leaves, so backward clears no parameter gradient as an intermediate. Autograd reshape, reductions, pow and affine operations reach all three leaves; no parameter gradient is omitted.

The two omitted half backward operands are not claimed fixed or supported: eps**(-1.5) at singleton default eps is ~3.16e7, beyond FP16 max 65504; the class-only repair does not change the engine's pow backward intermediate. Half backward uses eps=0.125 there, and near-eps uses variance=eps=0.25. No GPU results motivated this source-based representability boundary.

## Engine pin and isolation

Final toolchain: CUDA 12.6; CMake Release, architecture 89, disconnected pybind11 source as specified. Both commands have the exact D2 arguments and exited 0; environment selected CUDACXX=/usr/local/cuda-12.6/bin/nvcc and prepended that bin directory to PATH.

- Commit: `26790d64c146bdbacc0b626bfb7ffec46cdc8fae` (immutable order commit on gn1; engine source bytes match baseline).
- Baseline: `0e1dea3c958a4a3a88e69835bc407bd59cbc77b1`.
- Module: `tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so`.
- Module sha256: `c9b7dc610421010a8b11e8e3af74d1a9ad99928dabb98835aa6c265754daf8ce`.
- Source-map sha256: `98ef59cd48200fb4af38caf9dd017022f0ad032b7f42082dfd1574506b470a5c`; 72 source inputs, before=after.

Both Python arms are baseline package files, with only candidate nn.py replaced; arm_isolation.json records the single differing file. All arms/mutants symlink to the same final module. The seal checks every package/test/module byte before the child imports. nm exports PyInit__tensor_cuda; strings contains the reshape diagnostic and expected operation names. ldd confirms CUDA 12.6 runtime/cuBLAS locations. These are CPU-only build/import facts.

## Mutation list (no executions this round)

Prior art: DeMillo, Lipton and Sayward (1978). Each mutant is the fixed nn.py plus exactly one literal replacement inside GroupNorm. mutant_validation.json checks this and parses all 14. Each predicted witness is an existing exact ID; it is not an observed killer. Every future parent receipt must list actual call-phase failing IDs; timeout, setup, collection and unavailable-device failures cannot count as kills.

| ID | One decision changed | Expected pytest exit | Predicted witness |
|---|---|---:|---|
| M01 | unbiased_variance | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g2-default]` |
| M02 | epsilon_outside_sqrt | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-float32]` |
| M03 | epsilon_ignored | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-float32]` |
| M04 | wrong_reduction_axis | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M05 | wrong_group_extent | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g2-default]` |
| M06 | interleaved_channels | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M07 | weight_omitted | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M08 | bias_omitted | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M09 | affine_swapped | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M10 | refusal_removed | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_nondivisible_refusal` |
| M11 | broadcast_extra_dimension | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M12 | normalization_omitted | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M13 | wrong_power | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M14 | default_epsilon_changed | 1 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_constructor_defaults` |

No mutant survivors or equivalence proofs exist yet because no mutant executed. Candidate focus must pass in the same RUNS before mutation. The lead/reviewer own survivor adjudication.

## Staged GPU lines and expected exits

RUNS must be an explicit absolute fresh run root; no default. One label per invocation, in the order below, each externally bounded below 20 minutes. The tooling directory is artifacts/gn1. Every actual GPU child command goes through forge-gpu-run with --max and an inner timeout. Every parent writes child exit and timeout after the child exits (L45). All parent exits are expected 0; raw child exits below are preserved, never converted into child PASS.

# GPU expectations, before execution

Every run_line parent must exit 0, meaning its child's registered raw exit and detailed test checks matched. The L45 receipt is parent_result.json, emitted after subprocess.run returns, with child exit, timeout flag and timestamps. No raw failure is relabeled as a child PASS. Unknown collection/device/setup errors never count as kills.

One label per shell invocation, each externally bounded below 20 minutes. Run labels in gpu_lines.json order under a single explicit RUNS. Each actual forge invocation starts in artifacts/gn1 (the tooling directory), has --max, and wraps an inner GNU timeout. RUNS has no default; duplicate labels refuse to overwrite. START.json records date output, order git log and content seal before the first line. Parent enforces 25 GiB free on ForgeRealm and 20 GiB on root. TMPDIR, pytest basetemp, forge log, stdout, JUnit, child and parent receipts all reside under RUNS. Threads=1.

Baseline focus expected pytest=1: 294 failures, 12 passes, 306 IDs. The 292 new numerical GroupNorm IDs and the existing phase7 norms ID fail with reshape: element count mismatch. The new refusal ID also fails because the baseline lacks D1 (DID NOT RAISE); this is an explicit order-consistency clarification, not an unrelated failure. Five CPU reference IDs and constructor defaults pass, as do the six other phase7 tests. Candidate focus expected 306 passes, pytest=0.

Regression: both arms have exactly 362 IDs across ten test files importing nn. Baseline expected pytest=1 from the registered GroupNorm failures; candidate expected pytest=0, with existing full-shape opt-in skips allowed only in regression. Every baseline-passing exact node ID must still pass; missing tests, setup errors and changed census invalidate the lane. Existing optional skip IDs remain visible.

Mutants: each expected pytest=1, 299 collected IDs, no setup/collection failure. Parent records exact call-phase failing IDs as actual_killers; predicted witnesses in MUTANTS.md are not evidence of a kill. A passing candidate_focus parent receipt in the same RUNS is mandatory before any mutant. All 14 must die or the lead and reviewer must provide an output-equivalence proof. score_gpu.py aggregates the 18 completed real-GPU parent receipts; no audit can satisfy G5.

CPU --audit substitutes forge_gpu_run_standin.py for forge-gpu-run, preserves --max and the inner timeout and runs only the five CPU reference tests through the SAME candidate focus parent/child plumbing. It uses hidden CUDA and imports but never allocates a device tensor. It is plumbing evidence only.

| Label | Arm / files | Expected child exit | Expected parent exit | inner / max / outer seconds |
|---|---|---:|---:|---|
| baseline_focus | baseline / focus | 1 | 0 | 240 / 300 / 330 |
| candidate_focus | candidate / focus | 0 | 0 | 240 / 300 / 330 |
| baseline_regression | baseline / regression | 1 | 0 | 840 / 900 / 930 |
| candidate_regression | candidate / regression | 0 | 0 | 840 / 900 / 930 |
| M01 | M01 / mutation | 1 | 0 | 180 / 240 / 270 |
| M02 | M02 / mutation | 1 | 0 | 180 / 240 / 270 |
| M03 | M03 / mutation | 1 | 0 | 180 / 240 / 270 |
| M04 | M04 / mutation | 1 | 0 | 180 / 240 / 270 |
| M05 | M05 / mutation | 1 | 0 | 180 / 240 / 270 |
| M06 | M06 / mutation | 1 | 0 | 180 / 240 / 270 |
| M07 | M07 / mutation | 1 | 0 | 180 / 240 / 270 |
| M08 | M08 / mutation | 1 | 0 | 180 / 240 / 270 |
| M09 | M09 / mutation | 1 | 0 | 180 / 240 / 270 |
| M10 | M10 / mutation | 1 | 0 | 180 / 240 / 270 |
| M11 | M11 / mutation | 1 | 0 | 180 / 240 / 270 |
| M12 | M12 / mutation | 1 | 0 | 180 / 240 / 270 |
| M13 | M13 / mutation | 1 | 0 | 180 / 240 / 270 |
| M14 | M14 / mutation | 1 | 0 | 180 / 240 / 270 |

### STAGED_GPU_COMMANDS.sh (verbatim)

```bash
#!/usr/bin/env bash
# Prior art: GNU timeout, POSIX shell; GN1 L45 completed-child supervision.
# Expected parent exit for EVERY run_line is 0. Child exits/bounds below.
# Usage: RUNS=/explicit/fresh/run/root timeout 1180s bash STAGED_GPU_COMMANDS.sh LABEL
# Repeat with the SAME RUNS for each label (each command stays below 20 minutes).
# CPU plumbing: RUNS=/explicit/audit/root timeout 120s bash STAGED_GPU_COMMANDS.sh --audit
# No RUNS default. Full execution is the lead's G5 after cross-model review.
set -euo pipefail
: "${RUNS:?RUNS is required; use a fresh run root and reuse it across the 18 labels}"
[[ "$RUNS" == /* ]] || { echo "RUNS must be absolute" >&2; exit 64; }
TOOL="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$TOOL"
export RUNS
export TMPDIR="$RUNS/tmp" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
GPU_RUN=forge-gpu-run
label="${1:?select one registered label, or --audit}"
if [[ "$label" == --audit ]]; then
    export GN1_AUDIT=1 CUDA_VISIBLE_DEVICES=
    GPU_RUN="$TOOL/forge_gpu_run_standin.py"
    label=candidate_focus
else
    unset GN1_AUDIT
fi
mkdir -p -- "$TMPDIR"
timeout 20s python3 start_gpu_run.py
case "$label" in
  baseline_focus) # expected child 1, parent 0; inner 240s, --max 300s, outer 330s
    timeout 350s python3 run_gpu_line.py baseline_focus 1 330 "$GPU_RUN" --min 250 --max 300 --label gn1-baseline_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py baseline focus
    ;;
  candidate_focus) # expected child 0, parent 0; inner 240s, --max 300s, outer 330s
    timeout 350s python3 run_gpu_line.py candidate_focus 0 330 "$GPU_RUN" --min 250 --max 300 --label gn1-candidate_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py candidate focus
    ;;
  baseline_regression) # expected child 1, parent 0; inner 840s, --max 900s, outer 930s
    timeout 950s python3 run_gpu_line.py baseline_regression 1 930 "$GPU_RUN" --min 850 --max 900 --label gn1-baseline_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py baseline regression
    ;;
  candidate_regression) # expected child 0, parent 0; inner 840s, --max 900s, outer 930s
    timeout 950s python3 run_gpu_line.py candidate_regression 0 930 "$GPU_RUN" --min 850 --max 900 --label gn1-candidate_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py candidate regression
    ;;
  M01) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M01 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M01 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M01 mutation
    ;;
  M02) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M02 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M02 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M02 mutation
    ;;
  M03) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M03 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M03 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M03 mutation
    ;;
  M04) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M04 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M04 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M04 mutation
    ;;
  M05) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M05 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M05 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M05 mutation
    ;;
  M06) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M06 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M06 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M06 mutation
    ;;
  M07) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M07 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M07 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M07 mutation
    ;;
  M08) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M08 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M08 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M08 mutation
    ;;
  M09) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M09 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M09 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M09 mutation
    ;;
  M10) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M10 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M10 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M10 mutation
    ;;
  M11) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M11 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M11 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M11 mutation
    ;;
  M12) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M12 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M12 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M12 mutation
    ;;
  M13) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M13 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M13 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M13 mutation
    ;;
  M14) # expected child 1, parent 0; inner 180s, --max 240s, outer 270s
    timeout 290s python3 run_gpu_line.py M14 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M14 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M14 mutation
    ;;
  *) echo "unknown GN1 label: $label" >&2; exit 64 ;;
esac
# After all 18 labels: RUNS=/same/root timeout 20s python3 score_gpu.py (expected 0).
```

Audit receipt: audit_complete_seal/candidate_focus/parent_result.json: child exit=0, timeout=false, passed=true; actual enclosing command exited 0. The stand-in preserved --max=300 and inner timeout=240, ran in tooling cwd and executed exactly five CPU reference IDs. This does not establish GPU correctness.

## Predictions

- **P81 NOT HELD.** Source ops.cpp:915-918 is population ddof=0, contrary to the registered prediction of unbiased variance.
- **P82 NOT HELD as an empirical claim in this round (not established, not refuted).** Mutants were not executed. Its specific coverage concern is addressed: exact variance=eps cases are present, where ignoring eps changes normalization by sqrt(2); expected kill remains a reasoning prediction for G5.
- P83 NOT RUN: scored by the lead in G5.

## Prior art

Wu and He, Group Normalization (ECCV 2018): normalization formula, contiguous channel grouping and biased variance; PyTorch nn.GroupNorm documentation: divisibility refusal and biased variance; DeMillo, Lipton and Sayward, Hints on Test Data Selection: Help for the Practicing Programmer (1978): mutation testing; classical central finite differences (Fornberg 1988 is a lead for finite-difference formulas): oracle gradient validation; Higham (1996/2002): gamma_n floating-point error budget; IEEE 754 (1985/2019): roundoff and BF16 input-census rounding. pytest report hooks, Python subprocess.run and GNU timeout provide standard synchronous test/receipt supervision; SHA-256 (NIST 2001) provides source/module pins.

External bibliographic citations are **unverified — lead to check**. Search terms: Wu He Group Normalization ECCV 2018; PyTorch GroupNorm unbiased False; DeMillo Lipton Sayward 1978; Fornberg 1988 finite difference formulas; Higham gamma_n 1996 2002; IEEE 754 bfloat16 round to nearest; pytest hooks history; GNU timeout; NIST FIPS 180 SHA-256. Local semantics and source definitions are verified in mechanism_evidence.txt. No novel algorithm is claimed. GN1 contribution is the scoped explicit extent/refusal repair, concrete fixtures/error budgets and auditable arm/mutation staging.

## Honest residuals and retained mistakes

- G5 and blind cross-model review are pending. No GPU used. No performance claims.
- The first configure/build selected system CUDA 12.0.140 despite 12.6 being installed. AMENDMENT_02 records the correction. The final module and G3/G4 receipts use 12.6. Original build/module/pin/receipts/audits are retained, not silently discarded.
- Expanded pytest collection initially exited 2 with `ImportError: cannot import name 'nn' from 'tensor_cuda' (unknown location)`: importlib constructed a namespace before lazy test imports. Preloading the pinned package, exactly as the staged child does, fixed the harness census. collect_baseline_focus.log preserves the failure; *_preload and cuda126_* logs preserve the successful corrections.
- AMENDMENT_01 records FP16 default-eps singleton backward intermediate overflow. Those two cases are not claimed fixed/supported; their forward cases and nondefault-eps backward cases remain tested. No tolerance was tuned to GPU data.
- Follow-up: -1 inference in NDArray::reshape is a separate feature. Also ticket the FP16 pow-backward representability boundary. No other normalization layer or op was edited.

## Packet verification

FINAL_VERIFICATION.json records the final source/test/tool/package seal checks, 9 pinned files in every arm (including nested quantization files), exact ID censuses, retained unchanged source map and corrected CUDA 12.6 pin. Packet assembly initially encountered a directory symlink while building the file-hash inventory; it is now recorded as a directory link, with its target, rather than read as a regular file. The earlier root-files-only package seal is retained as staging_seal_root_files_only.json. No product/tests/tolerances changed in this correction.

## Done

Every written file is listed verbatim below and in files_written.txt; written_file_sha256.json records hashes/links (excluding its own self-digest). No deletes, git writes, subagents or device runs occurred. The initial build was moved only within authorized paths.

Leave for the lead to bin if desired: artifacts/gn1/build_cuda120_retained/ and artifacts/gn1/engine_cuda120_retained.so. Historical audit_cpu/ and audit_cpu_final/ are retained receipts using the earlier module; audit_complete_seal/ is governing. All baseline/candidate/mutant arms are deliverables. Scratch directories /tmp/gn1_bootstrap/ and /tmp/gn1_cpu/ are exempt, left in place; no persistent scratch outside the authorized roots.

```text
artifacts/gn1/AMENDMENT_01.md
artifacts/gn1/AMENDMENT_02.md
artifacts/gn1/DTYPE_CENSUS.md
artifacts/gn1/FINAL_VERIFICATION.json
artifacts/gn1/G1_scope.json
artifacts/gn1/GATES.json
artifacts/gn1/GPU_EXPECTATIONS.md
artifacts/gn1/LEDGER.md
artifacts/gn1/MUTANTS.md
artifacts/gn1/REGISTRATION.md
artifacts/gn1/REPORT.md
artifacts/gn1/STAGED_GPU_COMMANDS.sh
artifacts/gn1/TOLERANCES.md
artifacts/gn1/arm_isolation.json
artifacts/gn1/arm_isolation_cuda120_retained.json
artifacts/gn1/arms/M01/tensor_cuda/__init__.py
artifacts/gn1/arms/M01/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M01/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M01/tensor_cuda/functional.py
artifacts/gn1/arms/M01/tensor_cuda/nn.py
artifacts/gn1/arms/M01/tensor_cuda/optim.py
artifacts/gn1/arms/M01/tensor_cuda/quant.py
artifacts/gn1/arms/M01/tensor_cuda/quantization
artifacts/gn1/arms/M02/tensor_cuda/__init__.py
artifacts/gn1/arms/M02/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M02/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M02/tensor_cuda/functional.py
artifacts/gn1/arms/M02/tensor_cuda/nn.py
artifacts/gn1/arms/M02/tensor_cuda/optim.py
artifacts/gn1/arms/M02/tensor_cuda/quant.py
artifacts/gn1/arms/M02/tensor_cuda/quantization
artifacts/gn1/arms/M03/tensor_cuda/__init__.py
artifacts/gn1/arms/M03/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M03/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M03/tensor_cuda/functional.py
artifacts/gn1/arms/M03/tensor_cuda/nn.py
artifacts/gn1/arms/M03/tensor_cuda/optim.py
artifacts/gn1/arms/M03/tensor_cuda/quant.py
artifacts/gn1/arms/M03/tensor_cuda/quantization
artifacts/gn1/arms/M04/tensor_cuda/__init__.py
artifacts/gn1/arms/M04/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M04/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M04/tensor_cuda/functional.py
artifacts/gn1/arms/M04/tensor_cuda/nn.py
artifacts/gn1/arms/M04/tensor_cuda/optim.py
artifacts/gn1/arms/M04/tensor_cuda/quant.py
artifacts/gn1/arms/M04/tensor_cuda/quantization
artifacts/gn1/arms/M05/tensor_cuda/__init__.py
artifacts/gn1/arms/M05/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M05/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M05/tensor_cuda/functional.py
artifacts/gn1/arms/M05/tensor_cuda/nn.py
artifacts/gn1/arms/M05/tensor_cuda/optim.py
artifacts/gn1/arms/M05/tensor_cuda/quant.py
artifacts/gn1/arms/M05/tensor_cuda/quantization
artifacts/gn1/arms/M06/tensor_cuda/__init__.py
artifacts/gn1/arms/M06/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M06/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M06/tensor_cuda/functional.py
artifacts/gn1/arms/M06/tensor_cuda/nn.py
artifacts/gn1/arms/M06/tensor_cuda/optim.py
artifacts/gn1/arms/M06/tensor_cuda/quant.py
artifacts/gn1/arms/M06/tensor_cuda/quantization
artifacts/gn1/arms/M07/tensor_cuda/__init__.py
artifacts/gn1/arms/M07/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M07/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M07/tensor_cuda/functional.py
artifacts/gn1/arms/M07/tensor_cuda/nn.py
artifacts/gn1/arms/M07/tensor_cuda/optim.py
artifacts/gn1/arms/M07/tensor_cuda/quant.py
artifacts/gn1/arms/M07/tensor_cuda/quantization
artifacts/gn1/arms/M08/tensor_cuda/__init__.py
artifacts/gn1/arms/M08/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M08/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M08/tensor_cuda/functional.py
artifacts/gn1/arms/M08/tensor_cuda/nn.py
artifacts/gn1/arms/M08/tensor_cuda/optim.py
artifacts/gn1/arms/M08/tensor_cuda/quant.py
artifacts/gn1/arms/M08/tensor_cuda/quantization
artifacts/gn1/arms/M09/tensor_cuda/__init__.py
artifacts/gn1/arms/M09/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M09/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M09/tensor_cuda/functional.py
artifacts/gn1/arms/M09/tensor_cuda/nn.py
artifacts/gn1/arms/M09/tensor_cuda/optim.py
artifacts/gn1/arms/M09/tensor_cuda/quant.py
artifacts/gn1/arms/M09/tensor_cuda/quantization
artifacts/gn1/arms/M10/tensor_cuda/__init__.py
artifacts/gn1/arms/M10/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M10/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M10/tensor_cuda/functional.py
artifacts/gn1/arms/M10/tensor_cuda/nn.py
artifacts/gn1/arms/M10/tensor_cuda/optim.py
artifacts/gn1/arms/M10/tensor_cuda/quant.py
artifacts/gn1/arms/M10/tensor_cuda/quantization
artifacts/gn1/arms/M11/tensor_cuda/__init__.py
artifacts/gn1/arms/M11/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M11/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M11/tensor_cuda/functional.py
artifacts/gn1/arms/M11/tensor_cuda/nn.py
artifacts/gn1/arms/M11/tensor_cuda/optim.py
artifacts/gn1/arms/M11/tensor_cuda/quant.py
artifacts/gn1/arms/M11/tensor_cuda/quantization
artifacts/gn1/arms/M12/tensor_cuda/__init__.py
artifacts/gn1/arms/M12/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M12/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M12/tensor_cuda/functional.py
artifacts/gn1/arms/M12/tensor_cuda/nn.py
artifacts/gn1/arms/M12/tensor_cuda/optim.py
artifacts/gn1/arms/M12/tensor_cuda/quant.py
artifacts/gn1/arms/M12/tensor_cuda/quantization
artifacts/gn1/arms/M13/tensor_cuda/__init__.py
artifacts/gn1/arms/M13/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M13/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M13/tensor_cuda/functional.py
artifacts/gn1/arms/M13/tensor_cuda/nn.py
artifacts/gn1/arms/M13/tensor_cuda/optim.py
artifacts/gn1/arms/M13/tensor_cuda/quant.py
artifacts/gn1/arms/M13/tensor_cuda/quantization
artifacts/gn1/arms/M14/tensor_cuda/__init__.py
artifacts/gn1/arms/M14/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/M14/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/M14/tensor_cuda/functional.py
artifacts/gn1/arms/M14/tensor_cuda/nn.py
artifacts/gn1/arms/M14/tensor_cuda/optim.py
artifacts/gn1/arms/M14/tensor_cuda/quant.py
artifacts/gn1/arms/M14/tensor_cuda/quantization
artifacts/gn1/arms/baseline/tensor_cuda/__init__.py
artifacts/gn1/arms/baseline/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/baseline/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/baseline/tensor_cuda/functional.py
artifacts/gn1/arms/baseline/tensor_cuda/nn.py
artifacts/gn1/arms/baseline/tensor_cuda/optim.py
artifacts/gn1/arms/baseline/tensor_cuda/quant.py
artifacts/gn1/arms/baseline/tensor_cuda/quantization/__init__.py
artifacts/gn1/arms/baseline/tensor_cuda/quantization/affine.py
artifacts/gn1/arms/candidate/tensor_cuda/__init__.py
artifacts/gn1/arms/candidate/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/arms/candidate/tensor_cuda/apa_sp2.py
artifacts/gn1/arms/candidate/tensor_cuda/functional.py
artifacts/gn1/arms/candidate/tensor_cuda/nn.py
artifacts/gn1/arms/candidate/tensor_cuda/optim.py
artifacts/gn1/arms/candidate/tensor_cuda/quant.py
artifacts/gn1/arms/candidate/tensor_cuda/quantization/__init__.py
artifacts/gn1/arms/candidate/tensor_cuda/quantization/affine.py
artifacts/gn1/audit_complete_seal/START.json
artifacts/gn1/audit_complete_seal/candidate_focus/child_result.json
artifacts/gn1/audit_complete_seal/candidate_focus/invocation.json
artifacts/gn1/audit_complete_seal/candidate_focus/junit.xml
artifacts/gn1/audit_complete_seal/candidate_focus/parent_result.json
artifacts/gn1/audit_complete_seal/candidate_focus/stdout.log
artifacts/gn1/audit_cpu/START.json
artifacts/gn1/audit_cpu/candidate_focus/child_result.json
artifacts/gn1/audit_cpu/candidate_focus/invocation.json
artifacts/gn1/audit_cpu/candidate_focus/junit.xml
artifacts/gn1/audit_cpu/candidate_focus/parent_result.json
artifacts/gn1/audit_cpu/candidate_focus/stdout.log
artifacts/gn1/audit_cpu_final/START.json
artifacts/gn1/audit_cpu_final/candidate_focus/child_result.json
artifacts/gn1/audit_cpu_final/candidate_focus/invocation.json
artifacts/gn1/audit_cpu_final/candidate_focus/junit.xml
artifacts/gn1/audit_cpu_final/candidate_focus/parent_result.json
artifacts/gn1/audit_cpu_final/candidate_focus/stdout.log
artifacts/gn1/audit_cuda126/START.json
artifacts/gn1/audit_cuda126/candidate_focus/child_result.json
artifacts/gn1/audit_cuda126/candidate_focus/invocation.json
artifacts/gn1/audit_cuda126/candidate_focus/junit.xml
artifacts/gn1/audit_cuda126/candidate_focus/parent_result.json
artifacts/gn1/audit_cuda126/candidate_focus/stdout.log
artifacts/gn1/baseline_expected_failures.json
artifacts/gn1/build.json
artifacts/gn1/build.log
artifacts/gn1/build_cuda120_retained/CMakeCache.txt
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CMakeCUDACompiler.cmake
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CMakeCXXCompiler.cmake
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CMakeDetermineCompilerABI_CUDA.bin
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CMakeDetermineCompilerABI_CXX.bin
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CMakeSystem.cmake
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/CMakeCUDACompilerId.cu
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/a.out
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cpp1.ii
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cpp4.ii
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.c
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.cpp
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.gpu
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.stub.c
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.fatbin
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.fatbin.c
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.module_id
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.ptx
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.sm_52.cubin
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.fatbin
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.fatbin.c
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.reg.c
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.sm_52.cubin
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCXX/CMakeCXXCompilerId.cpp
artifacts/gn1/build_cuda120_retained/CMakeFiles/3.28.3/CompilerIdCXX/a.out
artifacts/gn1/build_cuda120_retained/CMakeFiles/CMakeConfigureLog.yaml
artifacts/gn1/build_cuda120_retained/CMakeFiles/CMakeDirectoryInformation.cmake
artifacts/gn1/build_cuda120_retained/CMakeFiles/Makefile.cmake
artifacts/gn1/build_cuda120_retained/CMakeFiles/Makefile2
artifacts/gn1/build_cuda120_retained/CMakeFiles/TargetDirectories.txt
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/DependInfo.cmake
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/build.make
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/cmake_clean.cmake
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/cmake_device_link.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/compiler_depend.make
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/compiler_depend.ts
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/depend.make
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/deviceLinkLibs.rsp
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/deviceObjects1.rsp
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/dlink.txt
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/flags.make
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/includes_CUDA.rsp
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/link.txt
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/progress.make
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/autograd.cpp.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/autograd.cpp.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/bindings.cpp.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/bindings.cpp.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/conv.cu.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/conv.cu.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/dda.cu.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/dda.cu.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/gemm_apa.cu.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/gemm_apa.cu.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/kernels.cu.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/kernels.cu.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/matmul.cu.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/matmul.cu.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/ops.cpp.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/ops.cpp.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/paint_bake.cu.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/paint_bake.cu.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/paint_inpaint.cu.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/paint_inpaint.cu.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/paint_raster.cu.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/paint_raster.cu.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/terrain.cu.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/terrain.cu.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/terrain_objects.cu.o
artifacts/gn1/build_cuda120_retained/CMakeFiles/_tensor_cuda.dir/src/terrain_objects.cu.o.d
artifacts/gn1/build_cuda120_retained/CMakeFiles/cmake.check_cache
artifacts/gn1/build_cuda120_retained/CMakeFiles/progress.marks
artifacts/gn1/build_cuda120_retained/Makefile
artifacts/gn1/build_cuda120_retained/_deps/pybind11-build/CMakeFiles/CMakeDirectoryInformation.cmake
artifacts/gn1/build_cuda120_retained/_deps/pybind11-build/CMakeFiles/progress.marks
artifacts/gn1/build_cuda120_retained/_deps/pybind11-build/Makefile
artifacts/gn1/build_cuda120_retained/_deps/pybind11-build/cmake_install.cmake
artifacts/gn1/build_cuda120_retained/cmake_install.cmake
artifacts/gn1/build_cuda126.json
artifacts/gn1/build_cuda126.log
artifacts/gn1/changes.patch
artifacts/gn1/cmake_version.txt
artifacts/gn1/collect_baseline.json
artifacts/gn1/collect_baseline.log
artifacts/gn1/collect_baseline_focus.json
artifacts/gn1/collect_baseline_focus.log
artifacts/gn1/collect_baseline_focus_preload.json
artifacts/gn1/collect_baseline_focus_preload.log
artifacts/gn1/collect_baseline_regression_preload.json
artifacts/gn1/collect_baseline_regression_preload.log
artifacts/gn1/collect_candidate.json
artifacts/gn1/collect_candidate.log
artifacts/gn1/collect_candidate_focus_preload.json
artifacts/gn1/collect_candidate_focus_preload.log
artifacts/gn1/collect_candidate_regression_preload.json
artifacts/gn1/collect_candidate_regression_preload.log
artifacts/gn1/conditioning_census.json
artifacts/gn1/configure.json
artifacts/gn1/configure.log
artifacts/gn1/configure_cuda126.json
artifacts/gn1/configure_cuda126.log
artifacts/gn1/cuda126_baseline_collection.log
artifacts/gn1/cuda126_baseline_import.log
artifacts/gn1/cuda126_candidate_collection.log
artifacts/gn1/cuda126_candidate_import.log
artifacts/gn1/cuda126_cpu_import_collection.json
artifacts/gn1/cuda126_cpu_reference_and_refusal.json
artifacts/gn1/cuda126_cpu_reference_and_refusal.log
artifacts/gn1/engine_cuda120_retained.so
artifacts/gn1/engine_pin.json
artifacts/gn1/engine_pin_cuda120_retained.json
artifacts/gn1/engine_sources_after.json
artifacts/gn1/engine_sources_before.json
artifacts/gn1/files_written.txt
artifacts/gn1/final_cpu_reference_and_refusal.json
artifacts/gn1/final_cpu_reference_and_refusal.log
artifacts/gn1/forge_gpu_run_standin.py
artifacts/gn1/gpu_lines.json
artifacts/gn1/ids_baseline.txt
artifacts/gn1/ids_baseline_focus.txt
artifacts/gn1/ids_baseline_regression.txt
artifacts/gn1/ids_candidate.txt
artifacts/gn1/ids_candidate_focus.txt
artifacts/gn1/ids_candidate_regression.txt
artifacts/gn1/ids_focus.txt
artifacts/gn1/ids_mutation.txt
artifacts/gn1/ids_regression.txt
artifacts/gn1/import_baseline.json
artifacts/gn1/import_baseline.log
artifacts/gn1/import_candidate.json
artifacts/gn1/import_candidate.log
artifacts/gn1/mechanism_evidence.txt
artifacts/gn1/module_ldd.txt
artifacts/gn1/mutant_validation.json
artifacts/gn1/mutants.json
artifacts/gn1/mutants/M01/PRIOR_ART.md
artifacts/gn1/mutants/M01/nn.diff
artifacts/gn1/mutants/M01/nn.py
artifacts/gn1/mutants/M02/PRIOR_ART.md
artifacts/gn1/mutants/M02/nn.diff
artifacts/gn1/mutants/M02/nn.py
artifacts/gn1/mutants/M03/PRIOR_ART.md
artifacts/gn1/mutants/M03/nn.diff
artifacts/gn1/mutants/M03/nn.py
artifacts/gn1/mutants/M04/PRIOR_ART.md
artifacts/gn1/mutants/M04/nn.diff
artifacts/gn1/mutants/M04/nn.py
artifacts/gn1/mutants/M05/PRIOR_ART.md
artifacts/gn1/mutants/M05/nn.diff
artifacts/gn1/mutants/M05/nn.py
artifacts/gn1/mutants/M06/PRIOR_ART.md
artifacts/gn1/mutants/M06/nn.diff
artifacts/gn1/mutants/M06/nn.py
artifacts/gn1/mutants/M07/PRIOR_ART.md
artifacts/gn1/mutants/M07/nn.diff
artifacts/gn1/mutants/M07/nn.py
artifacts/gn1/mutants/M08/PRIOR_ART.md
artifacts/gn1/mutants/M08/nn.diff
artifacts/gn1/mutants/M08/nn.py
artifacts/gn1/mutants/M09/PRIOR_ART.md
artifacts/gn1/mutants/M09/nn.diff
artifacts/gn1/mutants/M09/nn.py
artifacts/gn1/mutants/M10/PRIOR_ART.md
artifacts/gn1/mutants/M10/nn.diff
artifacts/gn1/mutants/M10/nn.py
artifacts/gn1/mutants/M11/PRIOR_ART.md
artifacts/gn1/mutants/M11/nn.diff
artifacts/gn1/mutants/M11/nn.py
artifacts/gn1/mutants/M12/PRIOR_ART.md
artifacts/gn1/mutants/M12/nn.diff
artifacts/gn1/mutants/M12/nn.py
artifacts/gn1/mutants/M13/PRIOR_ART.md
artifacts/gn1/mutants/M13/nn.diff
artifacts/gn1/mutants/M13/nn.py
artifacts/gn1/mutants/M14/PRIOR_ART.md
artifacts/gn1/mutants/M14/nn.diff
artifacts/gn1/mutants/M14/nn.py
artifacts/gn1/nn.baseline.py
artifacts/gn1/nn.diff
artifacts/gn1/nn_import_test_files.json
artifacts/gn1/nvcc_cuda126_version.txt
artifacts/gn1/nvcc_version.txt
artifacts/gn1/order.sha256
artifacts/gn1/python_version.txt
artifacts/gn1/reference_checks.json
artifacts/gn1/reference_checks.log
artifacts/gn1/run_cpu_gate.py
artifacts/gn1/run_gpu_line.py
artifacts/gn1/run_gpu_tests.py
artifacts/gn1/score_gpu.py
artifacts/gn1/staging_seal.json
artifacts/gn1/staging_seal_cuda120_retained.json
artifacts/gn1/staging_seal_initial.json
artifacts/gn1/staging_seal_root_files_only.json
artifacts/gn1/start_gpu_run.py
artifacts/gn1/symbols_nm.txt
artifacts/gn1/symbols_nm_cuda120_retained.txt
artifacts/gn1/symbols_strings.txt
artifacts/gn1/symbols_strings_cuda120_retained.txt
artifacts/gn1/test_id_tolerances.json
artifacts/gn1/test_new.diff
artifacts/gn1/tracked_after.json
artifacts/gn1/tracked_before.json
artifacts/gn1/tracked_scope.diff
artifacts/gn1/written_file_sha256.json
tensor_cuda/build-gn1/CMakeCache.txt
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CMakeCUDACompiler.cmake
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CMakeCXXCompiler.cmake
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CMakeDetermineCompilerABI_CUDA.bin
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CMakeDetermineCompilerABI_CXX.bin
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CMakeSystem.cmake
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/CMakeCUDACompilerId.cu
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/a.out
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cpp1.ii
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cpp4.ii
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.c
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.cpp
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.gpu
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.cudafe1.stub.c
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.fatbin
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.fatbin.c
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.module_id
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.o
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.ptx
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/CMakeCUDACompilerId.sm_52.cubin
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.fatbin
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.fatbin.c
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.o
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.reg.c
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCUDA/tmp/a_dlink.sm_52.cubin
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCXX/CMakeCXXCompilerId.cpp
tensor_cuda/build-gn1/CMakeFiles/3.28.3/CompilerIdCXX/a.out
tensor_cuda/build-gn1/CMakeFiles/CMakeConfigureLog.yaml
tensor_cuda/build-gn1/CMakeFiles/CMakeDirectoryInformation.cmake
tensor_cuda/build-gn1/CMakeFiles/Makefile.cmake
tensor_cuda/build-gn1/CMakeFiles/Makefile2
tensor_cuda/build-gn1/CMakeFiles/TargetDirectories.txt
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/DependInfo.cmake
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/build.make
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/cmake_clean.cmake
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/cmake_device_link.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/compiler_depend.make
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/compiler_depend.ts
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/depend.make
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/deviceLinkLibs.rsp
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/deviceObjects1.rsp
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/dlink.txt
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/flags.make
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/includes_CUDA.rsp
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/link.txt
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/progress.make
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/autograd.cpp.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/autograd.cpp.o.d
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/bindings.cpp.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/bindings.cpp.o.d
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/conv.cu.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/conv.cu.o.d
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/dda.cu.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/dda.cu.o.d
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/gemm_apa.cu.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/gemm_apa.cu.o.d
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/kernels.cu.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/kernels.cu.o.d
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/matmul.cu.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/matmul.cu.o.d
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/ops.cpp.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/ops.cpp.o.d
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/paint_bake.cu.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/paint_bake.cu.o.d
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/paint_inpaint.cu.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/paint_inpaint.cu.o.d
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/paint_raster.cu.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/paint_raster.cu.o.d
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/terrain.cu.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/terrain.cu.o.d
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/terrain_objects.cu.o
tensor_cuda/build-gn1/CMakeFiles/_tensor_cuda.dir/src/terrain_objects.cu.o.d
tensor_cuda/build-gn1/CMakeFiles/cmake.check_cache
tensor_cuda/build-gn1/CMakeFiles/progress.marks
tensor_cuda/build-gn1/Makefile
tensor_cuda/build-gn1/_deps/pybind11-build/CMakeFiles/CMakeDirectoryInformation.cmake
tensor_cuda/build-gn1/_deps/pybind11-build/CMakeFiles/progress.marks
tensor_cuda/build-gn1/_deps/pybind11-build/Makefile
tensor_cuda/build-gn1/_deps/pybind11-build/cmake_install.cmake
tensor_cuda/build-gn1/cmake_install.cmake
tensor_cuda/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
tensor_cuda/tensor_cuda/nn.py
tensor_cuda/tests/test_groupnorm_gn1.py
```
