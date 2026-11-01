# Equivalence proofs

## V3

Reasoning, not a device observation. Before the upcast predicate, D5 refuses
all x.dtype strings outside {float32, float16, bfloat16}. For each of the three
remaining strings, membership in {float16, bfloat16} is exactly equivalent to
inequality to float32. Refused inputs reach neither predicate. V3 changes
nothing else, so the same tensors, casts, operations, outputs and gradients
are reached for every accepted input, and the same exceptions for every
refused input. No test can kill this mutant after D5. The former int64 counterexample
now stops at D5. This is the explicit proof required by D4; not a GPU kill.
Prior art: elementary case analysis (no specific prior art known to me);
mutation equivalence classification, DeMillo/Lipton/Sayward (1978), unverified
— lead to check: equivalent mutants.

## numpy-bool-g

`E_npbool_g`: delete the numpy.bool_ clause at nn.py:518 (replace it by False).

## numpy-bool-c

`E_npbool_c`: delete the numpy.bool_ clause at nn.py:526 (replace it by False).

Both proofs use PEP 357 (2006): built-in bool is still rejected; for a numpy.bool_ operand, NumPy 2.4.2 operator.index raises TypeError, caught by the following validation path. Every other operand follows the unchanged path. Therefore each single-site deletion has identical acceptance and normalized storage under this NumPy version. This is a version-dependent equivalence, not an unconditional proof for future NumPy releases. Every equivalence line records the arm interpreter's NumPy version and the live operator.index(numpy.bool_(True)) result. Acceptance invalidates the equivalence and fails the line. Expected exit 0 for both. The two inverted clauses D5_g_npbool/D5_c_npbool remain reach checks.

Prior art: PEP 357 (2006), equivalent mutation analysis (Offutt/Voas, year unverified - lead to check). A4 adds the version-bound live obligation and GN1 receipt binding.
