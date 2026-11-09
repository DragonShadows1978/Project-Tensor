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
