# ENG2-REV3 registration (immutable)

Order: orders/ENG2_REV3_A3_REVIEW.md; mechanism: orders/ENG2_A3_ENG1_ROWS_AND_CONSUMERS.md. Scope matches: review A3 rows, recorded G5 kill evidence, consumer refusal boundaries, staging seal.

Plan registered before independent checks: Q1 exact git source diff and fresh pytest collections of both commit files in isolated scratch, compare normalized IDs and L38 pins; Q2 own fnmatch-based killer check over both registries, verify every G5 input against its own MANIFEST.sha256 before parsing; Q3 own AST pass over tracked Python source via git only, all seven changed entry types, all three repositories, every role and unresolved site retained; eight or more source spot checks across all repositories; compare unchanged L60 rows; Q4 rerun copied seal verifier without writes outside target, inspect manifest-verified dry_4 labels. No GPU or registered gate is executed.

Expected exact counts: ten IDs removed and none added on each arm; 31 mutants each retains at least one registered failing killer. No percentage coverage bar. Consumer sites reported, not gated.

Controls: TMPDIR=/tmp/eng2_rev3_20261114; CUDA_VISIBLE_DEVICES empty; nice 10; OMP/OPENBLAS/MKL/NUMEXPR thread counts 1; foreground timeout <=600s; <=2 CPU-busy processes; Python bytecode and pytest cache disabled or scratch-directed. Planned jobs stay below 1GB by streaming source files. Repairs limited to three per own script, each ledgered. No writes outside rev3 and scratch.
