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
