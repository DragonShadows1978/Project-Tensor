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
