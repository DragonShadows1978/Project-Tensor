# GN1 A1 registration (immutable)

Registered before implementation, CPU gates, or device data. Governing plan is the user amendment and committed A1 order at c96beb03ae61f846817360a59f2bbf5a26bdbba3. ACCEPT iff G1-G4 pass; G5 belongs to the lead after clean cross-model review. No GPU runs or rebuilds.

## Numerical comparisons

Represent x, parameters and projection first. Reference is round-one NumPy float64 formula and VJP on those represented operands. D1 computes every normalization/affine op in float32. Compare unrounded float64 reference to actual output/gradient. For a float32 budget |z-r| <= A+R|r| and one final rounding with unit roundoff u and subnormal half-step s, |round(z)-r| <= (1+u)A+s + ((1+u)R+u)|r|. Use A=R=3e-5 forward and 3e-4 backward from round 1. FP32 leaves use the original budget without rounding. FP16 u=2^-11, s=2^-25; BF16 u=2^-8, s=2^-134. No budget may be changed after registration. All budgets are tighter than round 1's low-precision budgets; FP32 IDs/tolerances remain unchanged. Output and each leaf gradient dtype must match exactly; all actual/reference arrays must be finite.

## Added operands

Restore test_backward_values[float16-r2-n1-g6-default] and [float16-r2-n2-g6-default]. Add test_mixed_parameters: input float16/bfloat16 x default (untouched module FP32 parameters) / half (nontrivial FP32 parameters followed by actual Module.half()), forward/backward, shape (2,6,4), G=3, default eps. Add test_low_variance on the same parameter configurations, forward/backward, variance targets 1e-4 and 1e-6, eps=1e-5. Use symmetric +/-sqrt(target) about zero, group extent 8, so represented means are exactly zero and represented variance is measured (not assumed equal to decimal target). New projections are binary exact, |projection|<=1/2, |w|<=1.5 and |q|<=.75; all grad leaves are checked. Add float32 x with .half() parameters, forward/backward, to pin M17 without adding or changing any ID containing float32. Existing near-epsilon and ordinary tests remain, with new low-precision budgets.

Low-variance FP32 reasoning: symmetric centers are exact; extent-8 reductions give gamma_7<4.18e-7; |q|<=.75, invstd<=317. Propagated centering/variance/pow/VJP error is bounded by a conservative 20*u32*.75*317 < 2.84e-4 absolute, inside the existing 3e-4 FP32 backward floor; relative budget covers nonzero results. Ordinary forward/backward use round one's finite-domain engineering budgets. This is finite-census reasoning, not universal numerical certification.

## CPU oracle/overflow checks

Keep two binary-exact examples and the three original finite differences. Check each represented low-variance dtype/configuration against float64 central differences for x, weight and bias: h=2^-25, atol=rtol=2e-7. Smallest sqrt(var+eps)>0.003; (h/sqrt(var+eps))^2<9e-11; O(eps64/h) subtraction below the registered absolute floor for this bounded output fixture. Both arms import/collect with CUDA_VISIBLE_DEVICES empty, all IDs listed. Recompute threshold 65504^(-2/3) and every represented (var+eps)^(-1.5)>65504 (FP16 overflow claim, including BF16-represented operands hypothetically stored in FP16; BF16 itself has enough exponent range).

## Mutants and staging

Re-derive M01-M14 from round one on D1. Add M15: input-dtype compute with parameters in input dtype; M16: output not cast back; M17: no parameter casts when original x is FP32; M18: x upcast but no parameter casts. Every mutant is one registered semantic decision, in an isolated arm; no actual kills claimed on CPU. Expected child exit 1 each; parent 0 iff valid kill. Candidate focus and regression expected child 0; baseline focus and regression expected child 1. Expected parent exit 0 for every line. Same pinned module in every arm. RUNS has no default, fresh receipts, start seal before first line. Focus inner/max/outer 240/300/330 seconds; regression 840/900/930; mutations 180/240/270. CPU audit runs only reference IDs through stand-in, child/parent 0, CUDA hidden. No baseline/device/setup error can count as mutation kill.

## Scope and stop rules

Only GroupNorm in nn.py, the designated test, a1 artifacts, logs and /tmp/gn1_a1_* are writable. Preserve all other tracked bytes and round-one/build receipts by before/after SHA256 maps. Compare product scope against 0e1dea3 and 50b3418f. No git writes or deletes. All commands bounded below 1200s, total ceiling 5400s; no background waits. Every CPU run hides CUDA, TMPDIR under /tmp/gn1_a1_*, threads 1. P84 is a device prediction and is not established by CPU checks; P85 is exact-ID/tolerance/operation-sequence evidence.

## Prior art

Wu and He (2018), Group Normalization, equations/contiguous population statistics; PyTorch GroupNorm FP32 half normalization accumulation (unverified — lead to check: PyTorch group_norm acc_type half); Micikevicius et al. (2018), Mixed Precision Training, FP32 computation with low-precision storage (unverified — lead to check); IEEE 754 (1985/2019) round-to-nearest error; Higham (1996/2002) gamma_n; classical central differences, Fornberg (1988) lead; DeMillo, Lipton and Sayward (1978) mutation testing. Literature unverified — lead to check named titles. Round-one reference/scaffolding is reused, new work is D1 wiring and this finite operand/tolerance registration.
