# APA-SP1: scope of the single-pass result

Evidence class: reasoning, checked by CPU unit tests. Numerical receipts are
in `cpu_summary.json`, `mutations.json` and the G1 logs. This is an author-run
baseline; the lead owns the independent blind verification required by house
rule 8. No GPU equivalence or speed claim is made by this document.

## Q1: the dependency forbids irrevocable selective streaming

Define the economical streaming model precisely: a key is visited in ascending
index order, its exact dot is computed only if selected, its chosen score is
committed to the online-softmax accumulator, and past contributions cannot be
recovered individually or corrected after seeing the suffix. There is no stored
per-key state and no speculative exact dot on every key. In that model, exact
z-score selection for arbitrary inputs is impossible.

The registered witness has q=[1], bulk scores [1,0,0,t], exact scores
[2,0,0,-10], values [1,0,0,0], scale=1 and z=Phi^-1(0.85).
Take t=0 and t=10. The complete information available at the first key is
identical. At t=0 the first key is above the global magnitude threshold; at
t=10 it is below. Its exact and bulk dots differ. Thus either irrevocable
choice is wrong for one suffix. The first value is nonzero and the last
exact score is negative, making the wrong early choice observable in output
well beyond the existing tolerance, rather than hiding it behind a dominant
last-key softmax. `counterexample()` records both thresholds, refine masks and
outputs; `test_q1_counterexample_and_buffer_loophole` pins the witness.

This is **not** a general lower bound on every algorithm described as “one
pass.” In particular, the literal wording without a memory/work restriction
admits a one-read construction: at each key read Kq, K and V, compute both
dots, and store (bulk, exact, V). Compute the threshold after that traversal,
then select scores and reduce softmax from the records. It reads the original
keys once, computes every exact dot, and stores O(S*(VD+2)) scalars per query.
It subsequently traverses buffered records; it is not a single streaming
softmax pass. `row_buffered_one_read()` constructs and checks this loophole.
With the same cached dot bits, statistics reduction and softmax reduction order
as the old path, storing operands/results does not change those bits. Our
NumPy implementation is checked at the existing tolerance, not presented as a
bit-level CUDA receipt or an all-input floating-point guarantee.

Even forbidding a second linear record traversal does not yield a universal
impossibility theorem if unbounded state and all-exact work are allowed. In
real arithmetic, maintain a balanced tree keyed by |bulk|, carrying separate
stable softmax aggregate states for bulk and exact scores in each subtree.
Insertion consumes the key/value once, with logarithmic tree work. Once the
global threshold is known, a range query joins the below-threshold bulk state
and above-threshold exact state. This observation is a reasoning construction,
not an implemented/gated kernel, and has different floating reduction order.

Verdict: lead P1 is a hit for irrevocable selective streaming and a miss if
read literally as an unrestricted impossibility claim. No global lower bound
is claimed. The buffered construction is a CPU witness only; no profitable
Q1 streaming CUDA candidate was found or entered into the GPU sweep. Q3 is
therefore implemented for the actual Q2 streaming kernel. This is an explicit
scope limitation, not a hidden equivalence claim.

## Q2: a safe exclusion rule, with conservative extra refinement

For finite bulk scores b_j and fixed delta>=0, let M_j=max(b_0,...,b_j).
The implemented decision is b_j>=M_j-delta, including equality. Keys visit in
ascending absolute index within the valid causal range. The bulk maximum is
separate from the online-softmax maximum of the already chosen mixed scores.
Sinks are excluded from selection and enter the denominator once at the end.

If key j is skipped, then b_j<M_j-delta. Every completion has M_S>=M_j,
therefore b_j<M_S-delta. Conversely, every key satisfying the terminal rule
b_j>=M_S-delta was refined when visited. Prefix decisions cannot change after
a suffix perturbation; early refinements may become unnecessary, but no
skipped key becomes required by that terminal **bulk-max** rule.

This characterizes the general condition: a skip is sound exactly when no
admissible completion requires that key. A conservative selector contains
the union of required sets over all completions consistent with its prefix.
For scalar threshold rules b_j>=T_final, a prefix lower bound T_j<=T_final
for every completion suffices. Nondecreasing thresholds with a consistent
terminal definition supply that bound: fixed cutoffs, running maxima minus
fixed deltas, and nondecreasing functions of such maxima. Running means,
variances and their unrestricted z-score combination do not generally supply
it. With additional known bounds on future inputs, other certified lower
bounds could work; none is assumed here.

Important limits follow directly from the definition:

- The output equals **this prefix rule's** mixed-score attention, not the
  z-score rule, not dense exact attention, and not final-max-only attention.
- Order matters: a permutation can alter the amount and identity of extra
  early refinement. “Order-safe” is not permutation-invariant.
- An exact score can be arbitrarily larger than its bulk score. This proof
  supplies no no-false-negative guarantee for dense-exact dominance, nor any
  approximation-error bound. A constructed CPU test checks this distinction.
- Every new record is selected; at delta=0, equality also selects tied records.
  Mandatory early refinement can prevent fraction matching at short prefixes.
- Using an exclusive prefix maximum yields the same decisions for delta>=0:
  new records pass either rule, and non-records see the same maximum. The
  registered mutation suite correctly retains this equivalent survivor.

Once the choice at j is made, the chosen score never changes. Standard stable
online-softmax maintains the partial normalizer and weighted value sum, so
joining the next selected/bulk score is exactly the offline softmax in real
arithmetic. No threshold prepass is needed. NumPy forward recurrence is checked
against an independent materialized prefix-scan specification on every random
draw; the literal row emulator also computes the dot and refinement during
each visit. CUDA FMA, shuffle reduction, expf and output rounding require G2;
CPU proof does not substitute for that receipt.

## Q3: why one walk may still lose

The new warp computes each bulk dot once, computes exact only when selected,
and loads every value once. Lane ownership makes dimensions coalesced, and
the production entry allocates only its output: O(B*H*L*VD) global bytes and
O(D+VD) per-row register state, independent of S. Optional diagnostic masks
use O(B*H*L*S) bytes and are excluded from timings and production memory claims.

The existing prefill block has multiple independent key streams and decode
can use independent split-K partitions after a global threshold stage. Our
single prefix chain loses that parallelism. Splitting it into independently
reset chains would be another selection rule; computing incoming prefix
maxima restores a scan or ordered carry. No split-K variant is included, and
decode underfill is unresolved. Eliminating one dot walk is not a speed proof.

Calibration uses only registered realized refine fractions and a fixed delta
grid. Holdout fractions must agree within the registered absolute tolerance
to support matched comparisons. Reports retain unmatched rows as RED; they
cannot count as hits. Overlap is intersection divided by z-score selected
count, with Jaccard and both fractions also reported. The signed positive-tail
rule competes against a magnitude rule that selects both signs, so high overlap
is not implied by equal budgets. Synthetic CPU tail/deviation evidence and GPU
kernel-sweep evidence remain separate. **this establishes nothing about model
quality**.
