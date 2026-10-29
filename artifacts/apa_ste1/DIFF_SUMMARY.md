# Diff summary
Four existing source files changed, one header added; three missing baseline
support files restored unchanged from the base engine's read-only source.

- `kernels.cu`: annotate mechanism; add a compile-time DK_STE boolean to base,
  bk1, bk2, bk3 backwards; launch false/true specializations on the host. Key
  contribution is admitted for unselected keys when on. bk2 query prepass and
  bk3 query half explicitly remain false. The diagnostic c keeps selected dK
  and dV zero and adds only unselected dK when on.
- `attention_tf32.cu`: equivalent KEY-only STE coefficient and host dispatch;
  query half remains false. Includes the otherwise easy-to-miss TF32 path.
- `ops.cpp` + `include/tc/apa_dk_ste.h`: strict cached process opt-in, unset/0 off,
  1 on, every other string errors. Prints exactly
  `tensor_cuda: TC_APA_DK_STE=<0|1>` once to stderr.
- `bindings.cpp`: call the helper at extension initialization, before exposing
  any tensor use. No Python package behavior changed.
- Restored `embed_deterministic.cu`, `gather_deterministic.cu`, and
  `deterministic_gather.h`: hashes exactly match base build source pins.

No forward body, selection, threshold, lse, p, dScore, dQ or dV arithmetic was
changed. The false specialization recovers original guards, pointer choices,
allocations, casts, launches and fixed reduction orders. CENSUS.md contains
per-operand proofs; SOURCE_DIFF.patch contains the complete source diff.
Default base backward has 12/12 exact instruction encodings vs 69b154cd.
Across all families, literal SASS equality is RED for 7 address-immediate-only
mismatches (3 off, 4 forward); 75/78 off and 15/19 forward match exactly.
Runtime byte equality remains a lead gate. No byte-identical runtime claim is
made from these CPU/static checks, particularly for atomic accumulation order.

Prior art: Bengio, Leonard and Courville (2013) STE; Dao et al. (2022)
FlashAttention saved-state recomputation already present. Unverified - lead to
check. Integration is key-only; no new estimator or attention algorithm claim.
