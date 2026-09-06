# APA-SP2: a relative-weight floor determines the refine margin

Evidence class: **reasoning**, with CPU property tests in
`tensor_cuda/tests/test_apa_sp2.py`. Empirical margins and performance are a
separate **kernel sweep**: **this establishes nothing about model quality**.
Registration: `artifacts/apa_sp2/registration.json` (immutable). No epsilon is
selected or recommended by this seat.

Let the eligible keys be the bottom-right causal prefix (or all keys for
noncausal attention). Assume nonempty, finite logits, `0 < epsilon <= 1`,
and a finite, nonnegative constant `e_q` that bounds **every** eligible key's
bulk error for this pass. Write `s_j = scale * q dot k_j`, `b_j = bulk_j`,
`|b_j-s_j| <= e_q`, and `s_star = max_j s_j`. The proof first uses real
arithmetic. CUDA arithmetic qualifications appear below.

With `Z = sum_j exp(s_j)`, define `w_j = exp(s_j)/Z` and
`w_star = exp(s_star)/Z`. An unchanged sink, if present, simply adds its
exponential to Z and has zero value numerator; all ratios below still hold.
Set `a = ln(1/epsilon) = -ln(epsilon) >= 0`. Since exponentiation is strictly
increasing, `s_j < s_star-a` implies
`w_j/w_star = exp(s_j-s_star) < exp(-a) = epsilon`.

The inclusive running bulk maximum at key j is `m_j = max_{i<=j} b_i` over
eligible visited keys. There is a visited key t attaining m_j. The error
assumption gives `s_j <= b_j+e_q` and `s_t >= b_t-e_q = m_j-e_q`.
The full-set exact maximum is at least s_t, hence `s_star >= m_j-e_q`.
Consequently,

```
b_j < m_j - (a + 2 e_q)
=> s_j <= b_j + e_q < m_j - e_q - a <= s_star - a
=> w_j < epsilon * w_star.
```

Every inequality either uses the assumed absolute-error bound, the strict
skip comparison, or the maximum over a superset. The **algorithm** skips iff
`b_j < m_j-delta`, refining otherwise, with
`delta(epsilon,e_q) = ln(1/epsilon) + 2 e_q`, in scaled-logit units.
This is a sufficient certificate of being below the floor, not an iff
characterization of all keys whose exact weight is below it. Equality at
the cutoff refines. Epsilon is dimensionless (relative weight); e_q and
delta have the unit of the scaled logits. Changing scale changes e_q too.

The prefix subtlety helps safety: the full exact maximum can only be larger
than the exact logit of the key setting the current prefix maximum. A future
large key tightens the weight ratio bound for an earlier skip; it cannot
invalidate that skip. It can make an earlier refinement unnecessary.

For fixed delta, `m_l >= m_j` for any later l, and therefore
`b_j < m_j-delta <= m_l-delta`: a skip stays a skip under larger prefix or
final maxima. Under ordered key partitions, the local prefix maximum
`m_p(j) <= m_j`, because it ranges over a subset. Thus a local skip satisfies
`b_j < m_p(j)-delta <= m_j-delta`; the same exact-weight proof applies using
the key that set m_p(j). Partition refinement contains global-prefix
refinement. With a shared frozen e_q, decreasing epsilon increases delta,
so the refine sets nest. None of these statements equates the resulting
mixed-score output to global-prefix output or proves that extra refinement
monotonically reduces output error. SP1.1's counterexample still applies.

The skipped-mass and perturbation bounds need separate quantities. Let U
be the skipped set, N its size, and `M = sum_{j in U} w_j`. For N>0,
`M < N epsilon w_star`; also `M <= 1`. For N=0, M=0. This bounds the exact
probability mass of skipped keys, not their output error in isolation.

For the ideal mixed logits `t_j = b_j` on U and `s_j` elsewhere, let
`Z_tilde = sum exp(t_j)` (plus the same sink). The selection margin yields
an additional useful inequality:

```
b_j < m_j-a-2 e_q <= s_star-a-e_q,
```

where the second inequality follows by rearranging `m_j-e_q <= s_star`.
Thus both `exp(b_j)` and `exp(s_j)` are below `epsilon exp(s_star)`.
Using `|s_j-b_j| <= e_q` more precisely,

```
|exp(b_j)-exp(s_j)|
 = exp(b_j) * |1-exp(s_j-b_j)|
 <= exp(b_j) * (exp(e_q)-1)
 <= epsilon * exp(s_star) * (1-exp(-e_q)).
```

Here `|1-exp(x)| <= exp(e_q)-1` for `x in [-e_q,e_q]` because
`1-exp(-e_q) <= exp(e_q)-1`. The final inequality uses the stronger bound
on b_j; at e_q=0 the error is exactly zero. A skipped key's **unnormalized
mass error divided by the original Z** is therefore at most
`epsilon w_star (1-exp(-e_q)) <= epsilon w_star`. This last claim would
not follow from the exact-weight floor alone with arbitrary perturbations;
the extra e_q of protection on b_j is essential.

Let `T = sum_j |exp(t_j)-exp(s_j)| / Z`, including the unchanged sink with
zero difference. Then `|Z_tilde-Z|/Z <= T` by the triangle inequality, and
`T <= N epsilon w_star (1-exp(-e_q))`. With vectors `u_j=exp(t_j)` and
`v_j=exp(s_j)`, insert u/Z between the two normalized probability vectors:

```
||u/Z_tilde - v/Z||_1
 <= ||u/Z_tilde-u/Z||_1 + ||u/Z-v/Z||_1
 = |Z_tilde-Z|/Z + T <= 2T.
```

Both normalized vectors are probabilities, so their L1 distance also is
at most 2. Hence the three requested bounds, each in one line, are:

1. **Individual skip:** `skip_j => w_j < epsilon w_star`, conditional on the full-key error bound.
2. **Aggregate skipped mass:** `M <= min(1, N epsilon w_star)` (strict against the second term when N>0).
3. **Softmax perturbation:** `||w_tilde-w||_1 <= min(2, 2 N epsilon w_star (1-exp(-e_q)))`; also `|Z_tilde-Z|/Z <= N epsilon w_star (1-exp(-e_q))`.

For values with `||V_j|| <= V_max`, the ideal output's absolute norm error
is at most `V_max * ||w_tilde-w||_1` by the triangle inequality. There is no
general relative-Frobenius guarantee: the dense output can be arbitrarily
close to zero. Negative/positive value cancellations also mean output
deviation need not decrease at every epsilon. The GPU curve measures it.

RED: replacing the unknown w_star by its worst-case upper bound 1 gives
`N epsilon = 32.768` for N=32768 at epsilon=1e-3; even epsilon=1e-4 gives
3.2768. The capped mass bound is just 1 in both cases: useless as a
worst-case probability guarantee. For N=8192 at epsilon=1e-4 it is 0.8192,
still loose. These are reasoning examples at an upper limit for N, not
measured skip counts (at least a prefix record is refined in each partition).
The curve records actual maximum N epsilon, observed skipped mass, and
error-bound exceedances without changing the registered margin.

The measurement protocol uses the existing rotated Lloyd-Max TurboQuant
implementation (`tensor_cuda/tensor_cuda/quant.py:_tables,_quantize_keys`).
The widths are 2 and 4 bits (`test_apa_phase6.py:test_apa_bits`); the packed
INT4 and INT8 GEMM families are different paths. Kq is reconstructed fp32,
not packed low-bit storage in the SP kernel. GQA uses one rotation per KV
head, shared by its query heads. Baseline and SP consume the same Kq.

G2 reports max, pooled p99.9, mean and count of absolute error, using GPU
bulk dots in the matching scalar/warp arithmetic order and FP64 exact dots
of the fp32 inputs at the actual fp32 scale. Each registered shape has two
independent seeded draws, 32 evenly spaced prefill queries (or the single
decode query), and all eligible keys. The chosen statistic is the maximum
across all these valid calibration logits, pooled by (bits,D), rounded
upward to fp32. A high percentile would knowingly exclude observed errors
and cannot serve as their bound. The maximum is still **not a universal
bound**, nor even a claim about unmeasured prefill queries. Scaling Q alone
can break any fixed finite table. The independent G3 draw checks all
eligible logits, records violations, and never retunes the table.

No cheap exact per-call e_q computation is claimed: computing every exact
dot would defeat selective refinement. The new public SP launcher therefore
loads a SHA-pinned, create-only G2 table once and looks up e_q by (bits,D),
with explicit epsilon and no default. The registered table covers fp32,
rotation enabled, scale=1/sqrt(D), and the registered calibration regime.
Missing tables, unsupported widths/dimensions/dtypes/scales fail closed.
Callers are responsible for matching Kq and the calibration regime; this
entry does not certify arbitrary tensors or inspect their norms.
The low-level epsilon+e_q entry and old direct-delta entry are distinct.

The host computes delta in double and rounds upward when converting to
float32, avoiding a smaller-than-derived fp32 margin at conversion. With
finite fp32 b_j, m_j and delta, rounding the subtraction cutoff to nearest
cannot make a representable b_j strictly between that real cutoff and its
adjacent representable upper neighbor; strict skip therefore preserves the
inequality against the real cutoff. Transcendental library rounding and the
empirical margin are not formal interval certification. Refined dot
arithmetic, approximate expf, softmax accumulation, and split-K merge add
numerical errors outside the ideal t_j proof. G1 tests the conditional
mathematics; G3 checks actual masks, epsilon/direct-delta parity and full
outputs at the inherited atol=rtol=0.001. No tolerance is widened.

All old attention kernel bodies remain byte-identical; TC_APA_SP stays
default OFF and must be exactly `1`. SP2 does not redirect an existing
two-pass API, does not change partition counts, and does not perform a
model evaluation. Blind verification under House Rules section 8 belongs
to the lead; author tests are baseline evidence only.

## Prior art (added by the lead, 2026-09-06; the seat ran before the Prior Art Directive existed)

- **BLASST** — Yuan, Shinn et al., "Dynamic Blocked Attention Sparsity
  via Softmax Thresholding", arXiv 2512.12087 (MLSys 2026). Skips key
  BLOCKS whose block max is more than ln(λ) below the running row max
  maintained by online softmax. The running-max comparator and the
  ln(1/ε) ↔ ln(λ) identity above are theirs. Taken: the criterion and
  its three-step justification. Ours: applying it to PRECISION
  refinement (every key stays in the softmax) at per-KEY granularity on
  the quantized bulk score, the 2·e_q margin (both sides of the
  comparison are quantized in APA; BLASST compares exact tile maxima),
  the exactness/monotone proof, and the split-K containment result.
  BLASST also reports λ ≈ a/L (inverse in context length) from
  calibration to a target sparsity — the first hypothesis to test if
  the ε curve drifts with S, noting APA refines rather than drops.
- **ThriftAttention** — Sharratt, arXiv 2605.23081 (May 2026).
  Selective mixed precision: top-k key blocks by a mean-of-tokens proxy
  computed before the pass are done in FP16, the rest in FP4, merged
  via online softmax. Taken: the observation that quantization error
  matters in proportion to softmax weight (their eq. 4), which is why
  the max-relative tail is the accurate tail. Ours: selection inside
  the pass on the quantized score itself, no pre-pass, per key.
- **FLASH-D** (Alexandridis et al., 2025), cited by BLASST as using
  online-softmax properties similarly for numerical stability on custom
  hardware — unverified by the lead; to check before any paper claim.
- Standard components, no claim: online softmax (Milakov & Gimelshein
  2018; FlashAttention), flash-decoding split-K merge, warp reductions,
  learned attention sinks.
Full comparison: /mnt/Shared/APA_SP_Prior_Art_Comparison_2026-09-06.md.
