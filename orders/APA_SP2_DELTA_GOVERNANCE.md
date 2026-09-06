# APA-SP2 — δ governance: derive the single-pass refine margin from a relative-weight floor

David, 2026-09-06: "I'm fine with the recommendation." The
recommendation (lead, Option C): the single-pass rule's δ is not a
fitted number per shape; it is DERIVED from one house-wide constant
with a unit, ε = the relative softmax weight below which a key is not
worth refining, plus MEASURED quantization margins. This order builds
the derivation, measures the margins, sweeps ε, and hands David the
curve. The seat registers nothing but the grid; ε is David's pick.

YOUR WRITABLE TARGET is the worktree you were dispatched into
(`/mnt/ForgeRealm/Project-Tensor-wt-apa-sp2`, branch `apa-sp2` forked
from `main` @ 62a982e, which contains your SP1/SP1.1 work merged). Same
boundary and laws as SP1/SP1.1: `apa_selective_sp_*` kernels are yours
to extend; every pre-existing kernel body byte-identical and hash-
pinned; `TC_APA_SP` stays default OFF. Your sandbox has NO GPU: build,
CPU-verify, deliver leased bounded scripts + blocked-reports; the lead
runs the card.

## Context

1. `orders/APA_SP1_SINGLE_PASS_SELECTIVE.md`, `orders/APA_SP1_1_SPLITK_SINGLE_PASS.md`,
   `docs/APA_SP1_LEDGER.md`, `docs/APA_SP1_1_REPORT.md`, and the
   receipts under `artifacts/apa_sp1/` (registration: δ was grid-
   searched per class to match the z-score refine fraction — e.g.
   δ=2.25 on `prefill_s8192_d64_c1_h4_kv4`; that is the governance gap).
2. The rule: `refine_j iff bulk_j ≥ m_j − δ`, where `bulk_j = q·k̂_j·scale`
   with k̂ the quantized key, and `m_j` the running max of bulk scores.
3. `docs/APA.md`, `docs/KERNEL_OPT_IMPLEMENTATION_PLAN.md` (evidence
   classes), `/mnt/Shared/HOUSE_RULES.md` §8/§9.

## The derivation (write it up in `docs/APA_SP2_DELTA_DERIVATION.md`
with every inequality justified)

Let `s_j = q·k_j·scale` be the exact logit and `bulk_j` its quantized
estimate; let `e_q = max_j |bulk_j − s_j|` over the key set (the bulk
quantization error bound). Let `s* = max_j s_j`. A key with
`s_j < s* − ln(1/ε)` has softmax weight `< ε · w*` where `w*` is the
max key's weight. Using only quantities the single pass can see:
`s_j ≤ bulk_j + e_q` and `s* ≥ m_j − e_q` (since the true max is at
least the true logit of the key that set the running max). So a key
is provably below the floor when `bulk_j + e_q < m_j − e_q − ln(1/ε)`,
i.e. **skip iff `bulk_j < m_j − (ln(1/ε) + 2·e_q)`; refine otherwise.**
Hence **`δ(ε, e_q) = ln(1/ε) + 2·e_q`** (in the same scaled-logit
units the kernel already uses). Prove the monotone property still
holds with this δ (it is a constant per pass, so it does). Note the
prefix-max subtlety: at key j the running max is over keys ≤ j, so
`s*` over the full set can only be larger, which only tightens the
skip condition — state this. The guarantee is one-sided: refined keys
may be unnecessary; skipped keys have weight < ε·w* up to the
unrefined-key softmax perturbation — bound that too (a skipped key's
error contributes at most `ε·w*` relative mass; summed over skipped
keys give the aggregate bound in terms of the number skipped and ε,
and state honestly that the bound is loose).

## Mission

1. **Measure `e_q`** (kernel-sweep evidence): for the registered
   shapes and both bulk bit widths the repo supports for the selective
   path (name them from the code), the distribution of `|bulk_j − s_j|`
   over seeded (q, K) draws: max, 99.9th percentile, mean. `e_q` is a
   REGISTERED-BEFORE-USE value per (bit width, D); pick the statistic
   (max over draws vs a high percentile) and say why. Deliver as a
   small table + JSON.
2. **Implement `δ(ε, e_q)`** as the production path for the SP kernels
   (prefill and split-K): the launcher takes `ε` (and looks up `e_q`
   from a registered table or computes it per call if cheap — say
   which) and derives δ; the old direct-δ entry stays for
   reproduction of SP1 receipts. Flag semantics unchanged (`TC_APA_SP`
   default OFF).
3. **Sweep ε** ∈ {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4} on the
   registered shapes (prefill S∈{2048,8192}, decode S∈{8192,32768},
   D∈{64,128}, causal/non-causal, the two GQA mappings) and both bit
   widths: refine fraction, output deviation vs dense fp32 (relative
   Frobenius, max abs), speed vs the two-pass kernel at ITS registered
   percentile (and vs dense if cheap). Also the old kernel's own
   fraction/deviation as the reference row.
4. **The curve for David** (`artifacts/apa_sp2/EPSILON_CURVE.md`):
   per shape class, fraction and deviation and speed vs ε; the ε at
   which the SP deviation first drops below the two-pass kernel's; the
   ε at which SP speed drops below the two-pass kernel's; and a
   one-paragraph reading. NO recommendation of ε from the seat; the
   pick is David's.

## Gates

Register first (`artifacts/apa_sp2/registration.json`, IMMUTABLE,
citing SP1/SP1.1 registration shas): the ε grid, the `e_q` statistic
rule, the shapes, and lead predictions: (P1) `e_q` at the coarser bit
width is ≥ 0.05 scaled-logit units on D=128; (P2) at ε=1e-3 the SP
refine fraction is below the z-score fraction on peaked (causal,
long-S) shapes and above it on non-causal short ones; (P3) SP
deviation beats the two-pass kernel at ε ≤ 1e-2 on ≥ 75% of shapes;
(P4) SP speed stays ≥ 1.0× the two-pass kernel down to ε=1e-3 on
prefill S=8192 and decode S=32768.
G1 (CPU, you): derivation tests (skip condition ⇒ weight < ε·w*
on random draws with a known e_q; monotone property; both entry
points agree when δ is passed directly); hash pins; existing SP tests.
G2 (GPU, lead): `e_q` measurement; G3 (GPU, lead): the ε sweep.
Deliver `scripts/apa_sp2_lead_gpu.sh` (leased, ≤10 min per class,
`list|run|resume|summary`, same conventions as SP1's) and the exact
commands file. Evidence class: kernel sweep; say "this establishes
nothing about model quality" where the numbers are.

## Rules (binding)

NO git. NO subagents. NO background waits (foreground with explicit
`timeout`, every call under 10 minutes; `python3 -c 'import time;
time.sleep(n)'` for gaps). Never kill a process you did not start.
Registration immutable; amendments separate. Existing kernel bodies
byte-identical. RED honesty: if the bound is loose enough to be
useless at practical ε, say so with the number. Ledger in
`docs/APA_SP1_LEDGER.md` (continue it).

## Done (verbatim)

1. The derivation doc path and the three bounds stated in one line
   each.
2. `e_q` table (registered rule + numbers, or blocked-report).
3. Kernel/launcher changes (files/lines), flag semantics, hash pins.
4. G1 results; G2/G3 blocked-reports with exact lead commands.
5. Registration sha; predictions hit/miss on what is measurable without
   a GPU.
6. Deviations; residual risks; RED; process safety; model id and effort.
