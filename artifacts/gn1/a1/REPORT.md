# GN1 amendment A1 report

ACCEPT under registered G1-G4 acceptance. G5 NOT RUN: actual device forward/backward, regression outcomes and mutant kills are not claimed. Evidence classes: source/byte identity, CPU NumPy unit tests and central differences, arithmetic, hidden-CUDA imports/collection, and one CPU staging audit. No engine rebuild, C++/CUDA source change, GPU use, git write, deletion or production edit.

## Gates

- G1 PASS — only GroupNorm in nn.py and the designated test changed. 7,593 other current tracked files and 445 existing round-one/build files are byte-identical. Relative to 0e1dea3, common product files differ only in GroupNorm plus the new test; relative to 50b3418f, GroupNorm changes are D1 only. Orders/receipt metadata are scoped separately and preserved. Receipts: G1_scope.json, scope_pin_checks.json/log, tracked_before.json, tracked_after.json, base_sha256.json, round1_sha256.json, readonly_before.json, readonly_after.json, changes_vs_base.patch, changes_vs_round1.patch, float32_identity.json.
- G2 PASS — 29 CPU oracle checks (two exact examples, three original differences, 24 new low-variance differences) plus refusal pass. Both arms import the pinned module and collect identical 351 GN1 / 358 focus / 414 regression IDs with CUDA hidden. 98 original FP32/FP32 IDs and tolerances unchanged. Recomputed all new low-variance range comparisons, with eight actual FP16 M15-overflow operands. Receipts: G2_cpu.json, reference_checks_preloaded.json/log, cpu_candidate_mutation_references.json, collect_baseline_focus.json/log, collect_candidate_focus.json/log, collect_baseline_regression.json/log, collect_candidate_regression.json/log, ids_baseline_focus.txt, ids_candidate_focus.txt, ids_baseline_regression.txt, ids_candidate_regression.txt, arithmetic_checks.json/log, overflow_arithmetic.json, test_id_tolerances.json.
- G3 PASS — pinned module c9b7dc610421010a8b11e8e3af74d1a9ad99928dabb98835aa6c265754daf8ce; source-map SHA256 98ef59cd48200fb4af38caf9dd017022f0ad032b7f42082dfd1574506b470a5c, 72 unchanged source inputs, unchanged existing build tree. All 20 arms use symlinks to that same module. Baseline Python bytes exactly match 0e1dea3. No rebuild. Receipts: G3_pin.json, engine_pin.json, engine_sources_original.json, engine_sources_after.json, scope_pin_checks.json/log.
- G4 PASS — 22 bounded staged lines, each expected child/parent exit registered; M01-M18 re-derived on D1, no equivalence or actual kill claimed; CPU audit child=0, parent=0, timeout=false, 29 reference calls passed. Receipts: G4_staging.json, staging_checks.json/log, STAGED_GPU_COMMANDS.sh, gpu_lines.json, mutants.json, MUTANTS.md, mutant_validation.json, staging_seal.json, audit_cpu/START.json, audit_cpu/candidate_focus/parent_result.json, audit_cpu/candidate_focus/child_result.json, audit_cpu/candidate_focus/stdout.log.

## D1 and float32 identity

The input is differentiably cast to FP32 for half/BF16; all statistics, normalization and affine operations compute in FP32. Non-FP32 parameter casts retain each leaf's source dtype in backward; final output returns to the original input dtype. Original FP32/FP32 inputs return through the original affine statement before all parameter/output casts. float32_identity.json compares the reached math statement source segments byte-for-byte, ordered SHA256 d1c87a381d0a74e4b48fe8e03ed28c124b89e8bb061ec3619ae6b17f6e241991. Indentation/control guards are added, not tensor arithmetic. The constructor is unchanged from round one.

### nn.py diff against round one (verbatim)

```diff
--- round1/tensor_cuda/tensor_cuda/nn.py
+++ candidate/tensor_cuda/tensor_cuda/nn.py
@@ -522,6 +522,13 @@
         self.bias = parameter(np.zeros(num_channels))
 
     def forward(self, x):
+        # Prior art: PyTorch half normalization uses float32 accumulation;
+        # Micikevicius et al. (2018), Mixed Precision Training (both unverified
+        # — lead to check: "PyTorch group_norm acc_type half"). Round 1's
+        # equations stay unchanged; A1 keeps pow backward intermediates in FP32.
+        dtype = x.dtype
+        if dtype in ("float16", "bfloat16"):
+            x = x.astype("float32")
         N, C = x.shape[0], x.shape[1]
         spatial = list(x.shape[2:])
         # Explicit extent: NDArray::reshape does not infer -1.
@@ -532,7 +539,13 @@
         g = (g - mean) * (var + self.eps).pow(-0.5)
         g = g.reshape([N, C] + spatial)
         wshape = [1, C] + [1] * len(spatial)
-        return g * self.weight.reshape(wshape) + self.bias.reshape(wshape)
+        if (dtype == "float32" and self.weight.dtype == "float32"
+                and self.bias.dtype == "float32"):
+            return g * self.weight.reshape(wshape) + self.bias.reshape(wshape)
+        weight = self.weight if self.weight.dtype == "float32" else self.weight.astype("float32")
+        bias = self.bias if self.bias.dtype == "float32" else self.bias.astype("float32")
+        output = g * weight.reshape(wshape) + bias.reshape(wshape)
+        return output if output.dtype == dtype else output.astype(dtype)
 
 
 class InstanceNorm2D(Module):
```

# Registered A1 tolerances

Evidence class: pre-device arithmetic/finite-census reasoning. REGISTRATION.md and tolerances_registered.json are immutable and precede every gate. No GPU data was collected. For represented operands and float32 compute, use round one's |z-r|<=A+R|r| with A=R=3e-5 forward, A=R=3e-4 backward. One destination rounding yields atol=(1+u)A+s, rtol=(1+u)R+u, including subnormals. Here u16=2^-11, s16=2^-25; ubf=2^-8, sbf=2^-134. FP32 has no extra rounding allowance. Each parameter gradient uses its own leaf dtype's budget, not x's.

| Destination | Forward atol | Forward rtol | Backward atol | Backward rtol |
|---|---:|---:|---:|---:|
| float32 | 3e-05 | 3e-05 | 0.0003 | 0.0003 |
| float16 | 3.0044450759887697e-05 | 0.0005182958984375 | 0.00030017628669738765 | 0.000788427734375 |
| bfloat16 | 3.01171875e-05 | 0.0039363671875 | 0.00030117187499999995 | 0.004207421875 |

All low-precision budgets are no looser than round one. The original 98 FP32 IDs keep their original tolerances. The ordinary census uses round one's conditioning. New low-variance fixtures have extent eight and exact zero represented centers, |projection|<=1/2, |w|<=1.5, |q|<=.75 and invstd<=317; 20*u32*.75*317<2.84e-4 is the conservative finite-fixture FP32 backward floor calculation (within 3e-4), with relative allowance for nonzero results. It is an engineering budget for the census, not a universal error proof.

CPU oracle comparisons: binary-exact hand values use (0,0); original float64 central differences use (2e-7,2e-7), h=2^-18; new low-variance differences use (2e-7,2e-7), h=2^-25. The small h squared relative to var+eps (<9e-11) and float64 subtraction support that budget. These checks all passed before the staged CPU audit. Dtype strings, IDs and SHA256 use exact equality. NaN/inf always fail.

Prior art: IEEE 754 (1985/2019), Higham (1996/2002), classical central differences (Fornberg 1988 lead), round-one GN1 budgets. Literature unverified — lead to check named titles/terms. New contribution: this finite-census rounding derivation and fixture registration.

# Dtype and parameter configurations (source, collection and reasoning evidence)

| Input | Parameter leaves | Output | x gradient | weight/bias gradients | Coverage |
|---|---|---|---|---|---|
| float32 | float32 | float32 | float32 | float32 | Original 98 IDs, byte-identical math statements, no casts |
| float16 | float16 | float16 | float16 | float16 | Original census, including restored singleton backward IDs |
| bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | Original census |
| float16 | module's untouched float32 | float16 | float16 | float32 | Mixed and low-variance, forward/backward |
| float16 | float16 via Module.half() | float16 | float16 | float16 | Mixed and low-variance, forward/backward |
| bfloat16 | module's untouched float32 | bfloat16 | bfloat16 | float32 | Mixed and low-variance, forward/backward |
| bfloat16 | float16 via Module.half() | bfloat16 | bfloat16 | float16 | Mixed and low-variance, forward/backward |
| float32 | float16 via Module.half() | float32 | float32 | float16 | Two new M17 witnesses, outside original all-FP32 lane |

Assertions are staged, not observed GPU results. Every gradient dtype must equal its own leaf's dtype, even when it differs from x. Cast backward is source-verified at ops.cpp:1060-1063. Integer/bool normalization and float64 engine computation are not supported, as established in round one's source census; NumPy float64 is reference only. Actual collected input/parameter counts are dtype_configuration_census.json. There are 100 engine IDs with FP32 inputs: 98 original FP32/FP32 IDs plus two new FP32/FP16 mixed witnesses. P85 refers to preservation of the original all-FP32 lane, not a claim that those new mixed witnesses do not exist.

# Overflow arithmetic (CPU float64 and represented-storage reasoning)

The engine's pow backward calls ew_pow(a.data(), p-1), ops.cpp:854-859; p=-.5 materializes base^-1.5 in a's dtype before multiplication. FP16 max=65504. Therefore base<65504^(-2/3)=0.0006153961657226473 exceeds its finite range. With default eps=1e-5, variance<0.0006053961657226473 is in this domain.

Nominal variance 1e-4: (1.1e-4)^(-1.5)=866784.1720414475 > 65504.
Nominal variance 1e-6: (1.1e-5)^(-1.5)=27410122.234342143 > 65504.
Singleton variance 0: (1e-5)^(-1.5)=31622776.60168379 > 65504.

The represented variances differ slightly from the decimal targets. Every new low-variance ID, its represented variance, and M15's stepwise stored var/base/power are in overflow_arithmetic.json. Default and .half() affine configuration and forward/backward do not change variance. Unique dtype/target rows follow; every group has this same variance.

| Input dtype | Target variance | Represented variance | Float64 inverse cube | M15 stored var+eps | M15 inverse cube before storage | Actual M15 FP16 overflow |
|---|---:|---:|---:|---:|---:|---|
| float16 | 0.0001 | 0.00010004272917285562 | 866279.36756096117 | 0.00011003017425537109 | 866427.64110906946 | True |
| float16 | 1e-06 | 1.000808879325632e-06 | 27407099.128423598 | 1.1026859283447266e-05 | 27310034.658887811 | True |
| bfloat16 | 0.0001 | 0.00010019540786743164 | 864479.61135957914 | 0.00011014938354492188 | 865021.48335623718 | False |
| bfloat16 | 1e-06 | 9.9890166893601418e-07 | 27414228.027062975 | 1.1026859283447266e-05 | 27310034.658887811 | False |

BF16 itself has sufficient exponent range: the >65504 comparison for BF16-represented operands is a hypothetical FP16 range statement, not a claim that BF16 overflows. All eight new FP16 forward/backward/configuration IDs have an FP16-overflowing M15 backward intermediate. Arithmetic proves that intermediate range failure; an actual test kill remains G5. D1's FP32 storage keeps these powers representable. Prior art: IEEE 754 range/rounding, round one's AMENDMENT_01 and ops.cpp mechanism; literature unverified — lead to check.

## Test IDs

52 new IDs include two restored singleton backward IDs, eight mixed-parameter IDs, 16 low-variance device IDs, two FP32-input/half-parameter witnesses, and 24 CPU low-variance finite-difference IDs. The 194 existing low-precision IDs now use the pre-registered tighter budgets. All 299 old IDs are preserved; none is skipped or xfailed. Original FP32/FP32 numeric tolerances/IDs remain unchanged. Exact strings and per-leaf tolerances follow.

# Exact new/changed IDs and tolerances

All IDs below are verbatim pytest node IDs. The prefix is tensor_cuda/tests/test_groupnorm_gn1.py::. F32/F16/B16 mean the output dtype, with each gradient using its own leaf's matching dtype entry. Entries give (atol,rtol). Forward checks also execute inside backward IDs. Exact dtype strings and finite arrays are required for every engine comparison.

| Key | Forward (atol,rtol) | Backward (atol,rtol) | Derivation |
|---|---|---|---|
| F32 | (3e-5,3e-5) | (3e-4,3e-4) | Original FP32 budget, unchanged |
| F16 | (3.0044450759887697e-5,0.0005182958984375) | (0.00030017628669738765,0.000788427734375) | Represented input; FP32 budget; one rounding u=2^-11,s=2^-25 |
| B16 | (3.01171875e-5,0.0039363671875) | (0.00030117187499999995,0.004207421875) | Represented input; FP32 budget; one rounding u=2^-8,s=2^-134 |
| FD | (2e-7,2e-7) | n/a | Float64 central differences h=2^-25; truncation/subtraction derivation in REGISTRATION.md |

For F16/B16: A'= (1+u)A+s, R'=(1+u)R+u, A=R=3e-5 forward / 3e-4 backward. Table rows identify output / x-gradient / parameter-gradient keys. A dash means no gradient assertion in that forward ID. All IDs and leaf-specific numeric tolerances also appear in test_id_tolerances.json. The original 98 float32/float32 IDs are unchanged and are not listed here.

| ID | Status | Output / x grad / parameter grads |
|---|---|---|
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n1-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r2-n2-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n1-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r3-n2-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n1-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r4-n2-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n1-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g1-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g1-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g2-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g2-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g6-default]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float16-r5-n2-g6-eps0125]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n1-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r2-n2-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n1-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r3-n2-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n1-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r4-n2-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n1-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g1-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g1-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g2-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g2-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g6-default]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[bfloat16-r5-n2-g6-eps0125]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g6-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n1-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g6-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r3-n2-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g6-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n1-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g6-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r4-n2-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g6-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n1-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g1-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g1-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g2-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g2-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g6-default]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r5-n2-g6-eps0125]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n1-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r2-n2-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n1-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r3-n2-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n1-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r4-n2-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n1-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g1-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g1-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g2-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g2-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g6-default]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[bfloat16-r5-n2-g6-eps0125]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-float16]` | changed | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-bfloat16]` | changed | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[backward-float16]` | changed | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[backward-bfloat16]` | changed | B16 / B16 / B16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n1-g6-default]` | new | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_backward_values[float16-r2-n2-g6-default]` | new | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[forward-default-float16]` | new | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[forward-default-bfloat16]` | new | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[forward-half-float16]` | new | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[forward-half-bfloat16]` | new | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[backward-default-float16]` | new | F16 / F16 / F32 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[backward-default-bfloat16]` | new | B16 / B16 / F32 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[backward-half-float16]` | new | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[backward-half-bfloat16]` | new | B16 / B16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-4-default-float16]` | new | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-4-default-bfloat16]` | new | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-4-half-float16]` | new | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-4-half-bfloat16]` | new | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-6-default-float16]` | new | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-6-default-bfloat16]` | new | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-6-half-float16]` | new | F16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[forward-var1e-6-half-bfloat16]` | new | B16 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-4-default-float16]` | new | F16 / F16 / F32 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-4-default-bfloat16]` | new | B16 / B16 / F32 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-4-half-float16]` | new | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-4-half-bfloat16]` | new | B16 / B16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-6-default-float16]` | new | F16 / F16 / F32 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-6-default-bfloat16]` | new | B16 / B16 / F32 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-6-half-float16]` | new | F16 / F16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-6-half-bfloat16]` | new | B16 / B16 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_fp32_input_half_parameters[forward]` | new | F32 / - / - |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_fp32_input_half_parameters[backward]` | new | F32 / F32 / F16 |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-4-default-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-4-default-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-4-half-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-4-half-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-6-default-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-6-default-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-6-half-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[x-var1e-6-half-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-4-default-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-4-default-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-4-half-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-4-half-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-6-default-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-6-default-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-6-half-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[weight-var1e-6-half-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-4-default-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-4-default-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-4-half-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-4-half-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-6-default-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-6-default-bfloat16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-6-half-float16]` | new | FD |
| `tensor_cuda/tests/test_groupnorm_gn1.py::test_reference_low_variance_central_differences[bias-var1e-6-half-bfloat16]` | new | FD |

# A1 mutant registration

Prior art: DeMillo, Lipton and Sayward (1978), unverified — lead to check. Re-derived on D1. One semantic decision per mutant, including both affine branches or dtype plumbing where necessary. No GPU executions or actual kills claimed. Shape/dtype errors during a test call are valid structural kills; device, collection or setup failures and timeouts are invalid. Every actual killer must be recorded by the parent after child exit. No equivalences claimed.

| ID | Decision | Child / parent expected | Predicted witness (not observed) |
|---|---|---|---|
| M01 | unbiased_variance | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g2-default]` |
| M02 | epsilon_outside_sqrt | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-float32]` |
| M03 | epsilon_ignored | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_near_epsilon[forward-float32]` |
| M04 | wrong_reduction_axis | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M05 | wrong_group_extent | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r2-n1-g2-default]` |
| M06 | interleaved_channels | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M07 | weight_omitted | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M08 | bias_omitted | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M09 | affine_swapped | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M10 | refusal_removed | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_nondivisible_refusal` |
| M11 | broadcast_extra_dimension | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M12 | normalization_omitted | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M13 | wrong_power | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_forward_values[float32-r3-n2-g2-default]` |
| M14 | default_epsilon_changed | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_constructor_defaults` |
| M15 | input_dtype_compute | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_low_variance[backward-var1e-6-half-float16]` |
| M16 | output_not_cast_back | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[forward-default-float16]` |
| M17 | parameters_not_cast_for_fp32_input | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_fp32_input_half_parameters[forward]` |
| M18 | input_upcast_without_parameter_casts | 1 / 0 | `tensor_cuda/tests/test_groupnorm_gn1.py::test_mixed_parameters[forward-half-float16]` |

# Staged lines (verbatim)

Run from artifacts/gn1/a1 with RUNS an explicit absolute fresh subdirectory of a1, reused across labels; TMPDIR=/tmp/gn1_a1_staged. No RUNS default. The shell writes/checks START.json before each line and changes into its tooling directory. GPU_RUN=forge-gpu-run for real G5; no device command ran this round. Every parent exit is expected 0, with child expected 0 on candidate focus/regression, child expected 1 on baseline focus/regression and all mutants. The second numeric argument below is the expected child exit. Each parent writes its receipt after child exit; collection, device, setup failures and timeouts cannot count as mutant kills.

```bash
timeout 350s python3 run_gpu_line.py baseline_focus 1 330 "$GPU_RUN" --min 250 --max 300 --label gn1-baseline_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py baseline focus
timeout 350s python3 run_gpu_line.py candidate_focus 0 330 "$GPU_RUN" --min 250 --max 300 --label gn1-candidate_focus -- timeout --kill-after=10s 240s python3 run_gpu_tests.py candidate focus
timeout 950s python3 run_gpu_line.py baseline_regression 1 930 "$GPU_RUN" --min 850 --max 900 --label gn1-baseline_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py baseline regression
timeout 950s python3 run_gpu_line.py candidate_regression 0 930 "$GPU_RUN" --min 850 --max 900 --label gn1-candidate_regression -- timeout --kill-after=10s 840s python3 run_gpu_tests.py candidate regression
timeout 290s python3 run_gpu_line.py M01 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M01 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M01 mutation
timeout 290s python3 run_gpu_line.py M02 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M02 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M02 mutation
timeout 290s python3 run_gpu_line.py M03 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M03 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M03 mutation
timeout 290s python3 run_gpu_line.py M04 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M04 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M04 mutation
timeout 290s python3 run_gpu_line.py M05 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M05 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M05 mutation
timeout 290s python3 run_gpu_line.py M06 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M06 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M06 mutation
timeout 290s python3 run_gpu_line.py M07 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M07 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M07 mutation
timeout 290s python3 run_gpu_line.py M08 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M08 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M08 mutation
timeout 290s python3 run_gpu_line.py M09 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M09 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M09 mutation
timeout 290s python3 run_gpu_line.py M10 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M10 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M10 mutation
timeout 290s python3 run_gpu_line.py M11 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M11 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M11 mutation
timeout 290s python3 run_gpu_line.py M12 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M12 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M12 mutation
timeout 290s python3 run_gpu_line.py M13 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M13 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M13 mutation
timeout 290s python3 run_gpu_line.py M14 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M14 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M14 mutation
timeout 290s python3 run_gpu_line.py M15 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M15 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M15 mutation
timeout 290s python3 run_gpu_line.py M16 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M16 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M16 mutation
timeout 290s python3 run_gpu_line.py M17 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M17 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M17 mutation
timeout 290s python3 run_gpu_line.py M18 1 270 "$GPU_RUN" --min 190 --max 240 --label gn1-M18 -- timeout --kill-after=10s 180s python3 run_gpu_tests.py M18 mutation
```

CPU audit line actually run (exit 0):

```bash
TMPDIR=/tmp/gn1_a1_boot CUDA_VISIBLE_DEVICES= RUNS=/mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1/artifacts/gn1/a1/audit_cpu timeout 120s bash /mnt/ForgeRealm/Project-Tensor/Fork/tensor_fork_apa/WorkTree/tensor_fork_apa_wt_gn1/artifacts/gn1/a1/STAGED_GPU_COMMANDS.sh --audit
```

The child ran 29 reference IDs, all passed; parent recorded child exit=0, timeout=false, audit=true, passed=true, after child exit. score_gpu.py is staged only; running it after all 22 actual device lines is expected exit 0.

## Predictions

- P84 NOT HELD — not established, not refuted. This is a G5 prediction: actual FP16/BF16 pass outcomes and M15's observed killer require GPU execution, which this round forbids. CPU arithmetic and strictly tighter registered budgets support the mechanism but cannot substitute for that outcome.
- P85 HELD — 98 original FP32/FP32 IDs, original tolerances, byte-identical reached tensor-math statements and no casts. Two newly required FP32-input/half-parameter witnesses are separately counted; total FP32-input engine IDs are 100. They exercise different parameter dtypes from the unchanged original lane.

## Failures retained and residuals

Preparation attempt 1 failed with AssertionError: ('M02', '(var + self.eps).pow(-0.5)', 5), because the generator counted matching text across nn.py instead of just GroupNorm. The generator was restricted to the class and partial artifacts completed; no change outside the class was applied. The first CPU smoke recorded 29 passed and one refusal import failure: ImportError: cannot import name 'nn' from 'tensor_cuda' (unknown location). inspect_cpu.py now preloads/asserts the real staged package before pytest, matching round one's GPU child. The corrected smoke passes; original failure receipt reference_checks.json/log is retained. No tolerance or reference was changed in either correction.

Not claimed fixed outside GroupNorm: engine FP16 pow-backward materialization; BatchNorm1D/InstanceNorm2D; -1 reshape inference; GPU correctness, regression results and mutant kills; performance. G5 and independent cross-model review remain the lead's next steps. Actual mutant killers are null until valid completed device receipts exist. No equivalence proofs are claimed.

## Prior art

PyTorch GroupNorm accumulation in FP32 for half normalization (living implementation/version unverified — lead to check: PyTorch group_norm acc_type half); Micikevicius et al. (2018), Mixed Precision Training (unverified — lead to check title), for FP32 compute with low-precision storage. Wu and He (2018), Group Normalization, for population statistics and contiguous grouping; PyTorch GroupNorm for refusal and biased variance; round-one GN1 implementation/reference/scaffolding reused. IEEE 754 (1985/2019) for range and one-rounding bounds; Higham (1996/2002), Accuracy and Stability of Numerical Algorithms, for gamma_n; classical central differences (Fornberg 1988, unverified lead to check); DeMillo, Lipton and Sayward (1978), Hints on Test Data Selection, for mutation testing. SHA256 (NIST 2001), pytest hooks (Krekel et al. 2004+), GNU timeout and Python subprocess for pinning/synchronous receipt plumbing. Literature not checked through a network; named leads are for lead verification. A1's contribution is D1's dtype wiring, these finite fixture/error-budget registrations, and extending the existing mutation/staging instruments.

## Done

No deletes, no moves to DELETE_ME, and nothing for the lead to bin. Existing round-one artifacts, arms and CUDA-12.0 retained build remain read-only historical evidence. Partial A1 staging was completed in place; first failed receipts remain for audit. Scratch directories /tmp/gn1_a1_boot and /tmp/gn1_a1_cpu remain; Python cache/scratch only is exempt but nothing was deleted. Every file written, including new module symlinks (not module copies), is listed below and in files_written.txt/json.

```text
tensor_cuda/tensor_cuda/nn.py
tensor_cuda/tests/test_groupnorm_gn1.py
artifacts/gn1/a1/DONE_VERBATIM.md
artifacts/gn1/a1/DTYPE_CENSUS.md
artifacts/gn1/a1/G1_scope.json
artifacts/gn1/a1/G2_arithmetic_summary.json
artifacts/gn1/a1/G2_cpu.json
artifacts/gn1/a1/G3_pin.json
artifacts/gn1/a1/G4_staging.json
artifacts/gn1/a1/GATES.json
artifacts/gn1/a1/IDS_VERBATIM.md
artifacts/gn1/a1/LEDGER.md
artifacts/gn1/a1/MUTANTS.md
artifacts/gn1/a1/NEW_CHANGED_IDS.md
artifacts/gn1/a1/OVERFLOW.md
artifacts/gn1/a1/REGISTRATION.md
artifacts/gn1/a1/REPORT.md
artifacts/gn1/a1/STAGED_GPU_COMMANDS.sh
artifacts/gn1/a1/STAGED_LINES.md
artifacts/gn1/a1/TOLERANCES.md
artifacts/gn1/a1/arithmetic_checks.json
artifacts/gn1/a1/arithmetic_checks.log
artifacts/gn1/a1/arms/M01/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M01/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M01/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M01/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M01/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M01/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M01/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M01/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M01/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M02/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M02/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M02/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M02/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M02/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M02/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M02/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M02/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M02/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M03/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M03/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M03/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M03/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M03/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M03/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M03/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M03/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M03/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M04/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M04/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M04/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M04/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M04/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M04/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M04/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M04/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M04/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M05/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M05/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M05/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M05/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M05/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M05/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M05/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M05/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M05/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M06/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M06/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M06/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M06/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M06/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M06/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M06/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M06/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M06/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M07/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M07/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M07/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M07/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M07/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M07/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M07/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M07/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M07/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M08/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M08/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M08/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M08/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M08/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M08/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M08/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M08/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M08/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M09/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M09/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M09/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M09/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M09/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M09/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M09/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M09/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M09/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M10/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M10/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M10/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M10/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M10/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M10/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M10/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M10/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M10/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M11/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M11/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M11/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M11/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M11/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M11/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M11/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M11/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M11/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M12/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M12/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M12/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M12/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M12/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M12/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M12/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M12/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M12/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M13/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M13/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M13/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M13/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M13/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M13/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M13/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M13/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M13/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M14/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M14/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M14/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M14/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M14/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M14/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M14/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M14/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M14/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M15/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M15/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M15/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M15/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M15/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M15/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M15/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M15/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M15/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M16/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M16/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M16/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M16/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M16/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M16/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M16/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M16/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M16/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M17/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M17/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M17/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M17/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M17/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M17/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M17/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M17/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M17/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/M18/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/M18/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/M18/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/M18/tensor_cuda/functional.py
artifacts/gn1/a1/arms/M18/tensor_cuda/nn.py
artifacts/gn1/a1/arms/M18/tensor_cuda/optim.py
artifacts/gn1/a1/arms/M18/tensor_cuda/quant.py
artifacts/gn1/a1/arms/M18/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/M18/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/baseline/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/baseline/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/baseline/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/baseline/tensor_cuda/functional.py
artifacts/gn1/a1/arms/baseline/tensor_cuda/nn.py
artifacts/gn1/a1/arms/baseline/tensor_cuda/optim.py
artifacts/gn1/a1/arms/baseline/tensor_cuda/quant.py
artifacts/gn1/a1/arms/baseline/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/baseline/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/arms/candidate/tensor_cuda/__init__.py
artifacts/gn1/a1/arms/candidate/tensor_cuda/_tensor_cuda.cpython-312-x86_64-linux-gnu.so
artifacts/gn1/a1/arms/candidate/tensor_cuda/apa_sp2.py
artifacts/gn1/a1/arms/candidate/tensor_cuda/functional.py
artifacts/gn1/a1/arms/candidate/tensor_cuda/nn.py
artifacts/gn1/a1/arms/candidate/tensor_cuda/optim.py
artifacts/gn1/a1/arms/candidate/tensor_cuda/quant.py
artifacts/gn1/a1/arms/candidate/tensor_cuda/quantization/__init__.py
artifacts/gn1/a1/arms/candidate/tensor_cuda/quantization/affine.py
artifacts/gn1/a1/audit_cpu/START.json
artifacts/gn1/a1/audit_cpu/candidate_focus/child_result.json
artifacts/gn1/a1/audit_cpu/candidate_focus/invocation.json
artifacts/gn1/a1/audit_cpu/candidate_focus/junit.xml
artifacts/gn1/a1/audit_cpu/candidate_focus/parent_result.json
artifacts/gn1/a1/audit_cpu/candidate_focus/stdout.log
artifacts/gn1/a1/base_sha256.json
artifacts/gn1/a1/baseline_expected_failures.json
artifacts/gn1/a1/build_report.py
artifacts/gn1/a1/changes_vs_base.patch
artifacts/gn1/a1/changes_vs_round1.patch
artifacts/gn1/a1/collect_baseline_focus.json
artifacts/gn1/a1/collect_baseline_focus.log
artifacts/gn1/a1/collect_baseline_regression.json
artifacts/gn1/a1/collect_baseline_regression.log
artifacts/gn1/a1/collect_candidate_focus.json
artifacts/gn1/a1/collect_candidate_focus.log
artifacts/gn1/a1/collect_candidate_regression.json
artifacts/gn1/a1/collect_candidate_regression.log
artifacts/gn1/a1/cpu_arithmetic.py
artifacts/gn1/a1/cpu_baseline_focus_collect.json
artifacts/gn1/a1/cpu_baseline_regression_collect.json
artifacts/gn1/a1/cpu_candidate_focus_collect.json
artifacts/gn1/a1/cpu_candidate_mutation_references.json
artifacts/gn1/a1/cpu_candidate_regression_collect.json
artifacts/gn1/a1/dtype_configuration_census.json
artifacts/gn1/a1/engine_pin.json
artifacts/gn1/a1/engine_sources_after.json
artifacts/gn1/a1/engine_sources_original.json
artifacts/gn1/a1/files_written.json
artifacts/gn1/a1/files_written.txt
artifacts/gn1/a1/float32_identity.json
artifacts/gn1/a1/forge_gpu_run_standin.py
artifacts/gn1/a1/gpu_lines.json
artifacts/gn1/a1/ids_audit.txt
artifacts/gn1/a1/ids_baseline_focus.txt
artifacts/gn1/a1/ids_baseline_regression.txt
artifacts/gn1/a1/ids_candidate_focus.txt
artifacts/gn1/a1/ids_candidate_regression.txt
artifacts/gn1/a1/ids_focus.txt
artifacts/gn1/a1/ids_mutation.txt
artifacts/gn1/a1/ids_regression.txt
artifacts/gn1/a1/inspect_cpu.py
artifacts/gn1/a1/mutant_validation.json
artifacts/gn1/a1/mutants/M01/nn.diff
artifacts/gn1/a1/mutants/M01/nn.py
artifacts/gn1/a1/mutants/M02/nn.diff
artifacts/gn1/a1/mutants/M02/nn.py
artifacts/gn1/a1/mutants/M03/nn.diff
artifacts/gn1/a1/mutants/M03/nn.py
artifacts/gn1/a1/mutants/M04/nn.diff
artifacts/gn1/a1/mutants/M04/nn.py
artifacts/gn1/a1/mutants/M05/nn.diff
artifacts/gn1/a1/mutants/M05/nn.py
artifacts/gn1/a1/mutants/M06/nn.diff
artifacts/gn1/a1/mutants/M06/nn.py
artifacts/gn1/a1/mutants/M07/nn.diff
artifacts/gn1/a1/mutants/M07/nn.py
artifacts/gn1/a1/mutants/M08/nn.diff
artifacts/gn1/a1/mutants/M08/nn.py
artifacts/gn1/a1/mutants/M09/nn.diff
artifacts/gn1/a1/mutants/M09/nn.py
artifacts/gn1/a1/mutants/M10/nn.diff
artifacts/gn1/a1/mutants/M10/nn.py
artifacts/gn1/a1/mutants/M11/nn.diff
artifacts/gn1/a1/mutants/M11/nn.py
artifacts/gn1/a1/mutants/M12/nn.diff
artifacts/gn1/a1/mutants/M12/nn.py
artifacts/gn1/a1/mutants/M13/nn.diff
artifacts/gn1/a1/mutants/M13/nn.py
artifacts/gn1/a1/mutants/M14/nn.diff
artifacts/gn1/a1/mutants/M14/nn.py
artifacts/gn1/a1/mutants/M15/nn.diff
artifacts/gn1/a1/mutants/M15/nn.py
artifacts/gn1/a1/mutants/M16/nn.diff
artifacts/gn1/a1/mutants/M16/nn.py
artifacts/gn1/a1/mutants/M17/nn.diff
artifacts/gn1/a1/mutants/M17/nn.py
artifacts/gn1/a1/mutants/M18/nn.diff
artifacts/gn1/a1/mutants/M18/nn.py
artifacts/gn1/a1/mutants.json
artifacts/gn1/a1/nn.base.py
artifacts/gn1/a1/nn.round1.py
artifacts/gn1/a1/nn_import_test_files.json
artifacts/gn1/a1/nn_vs_base.diff
artifacts/gn1/a1/nn_vs_round1.diff
artifacts/gn1/a1/overflow_arithmetic.json
artifacts/gn1/a1/prepare_a1.py
artifacts/gn1/a1/readonly_after.json
artifacts/gn1/a1/readonly_before.json
artifacts/gn1/a1/reference_checks.json
artifacts/gn1/a1/reference_checks.log
artifacts/gn1/a1/reference_checks_preloaded.json
artifacts/gn1/a1/reference_checks_preloaded.log
artifacts/gn1/a1/round1_sha256.json
artifacts/gn1/a1/run_cpu_gate.py
artifacts/gn1/a1/run_gpu_line.py
artifacts/gn1/a1/run_gpu_tests.py
artifacts/gn1/a1/scope_pin_checks.json
artifacts/gn1/a1/scope_pin_checks.log
artifacts/gn1/a1/scope_vs_base.json
artifacts/gn1/a1/scope_vs_round1.json
artifacts/gn1/a1/score_gpu.py
artifacts/gn1/a1/staging_checks.json
artifacts/gn1/a1/staging_checks.log
artifacts/gn1/a1/staging_seal.json
artifacts/gn1/a1/start_gpu_run.py
artifacts/gn1/a1/started.json
artifacts/gn1/a1/test.round1.py
artifacts/gn1/a1/test_id_tolerances.json
artifacts/gn1/a1/tolerances_registered.json
artifacts/gn1/a1/tracked_after.json
artifacts/gn1/a1/tracked_before.json
artifacts/gn1/a1/verify_scope.py
artifacts/gn1/a1/verify_staging.py
```
