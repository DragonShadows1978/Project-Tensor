# APA-PF2-F3 gate registration

The immutable order is orders/APA_PF2_F3_REV2_FIXES.md. The mechanism is
the stored-norm contract in docs/APA.md §8 and the findings in PF2-REV2.
Review findings SHA-256 is
20d78a4d79bd2039b565098dec47fe3ea990e36216edc25b2f031f2d61874e71.
The F2 read-only module from the F2 worktree is pinned to
aefe2513a4d33d7310421f65e6619f0706de2889410836bda42193053f8ec97a,
with build receipt b23fa3e93bd5866a34e9e61eca2933990ee3bacefadac9001ced0410c58df827.
The exact inherited F2 case splits are taken from LINES.json with SHA-256
6207c4c7122a4e5eec5bb4660249eea2725634f0ac08ad473dea7d5dac093d04.

- G1. Build F3 with the F2 release flags (CUDA 12.6, sm_89, hooks off).
  Compare F2 and F3 on the 752 inherited decode and 4,848 inherited
  prefill cases at the same indices, with the same seeds, options and
  line splits. Add 224 raw tc BF16 cases, the Cartesian product of
  format {int4,tq4}, D and VD {64,128}, diagnostics {off,on}, carry
  {partition,row} and G {2,3,4,5,6,7,8}, with B=1, KVH=2 and S=257.
  Split these by format and D (56 cases per line). Record the output,
  and for diagnostics the mask, bulk and prefix. Equal dtype and shape,
  nonempty finite and nonconstant operands, same-width unsigned views
  and zero unequal elements are required. This exercises both spq4 bulk
  specializations, both reachable spq6 variant-2 bulk specializations,
  all eight select_pv specializations and BF16 merge. Variants 0, 1 and
  3 are emitted but unreachable in both registered hooks-off builds.
- G2. Run the 16 REV1 sink cases, 16 REV1 norm configurations and 44
  F2 partition norm configurations. Test +Inf, -Inf and NaN norms on
  each of the 16 norm configurations for prefill and row decode.
  Check every reading row for all-NaN and require equal non-finite
  row sets. Retain finite nonconstant controls. Run all 44 partition
  cases with their F2 decode predicate and additionally compare it with
  noncausal prefill on the same visible inputs for both query rows.
  Run eight raw tc cases (format, D and diagnostics crossed, G=4,
  two KV groups, S=2, VD=D, partition carry). Test both infinities
  and NaN with a negative code dot, checking all reading rows,
  no unrelated rows, and the non-finite bulk scores when dumping.
  Count the structurally constant all-NaN rows.
- G3. Build three distinct incremental clones of F3, changing only
  kernels.cu.o plus device/module linking. CPU requires exact isolated
  edits, distinct module SHA-256s, arm imports, and a host bit-pattern
  check of the extracted guards including the tc hook load branches.
  Each GPU mutant line first runs its clean F3 suite in a separate
  process. pf3m01 reverts both prefill expressions and must fail for
  both infinities on all 16 norm configurations. pf3m02 replaces both
  expressions with a +Inf-only guard and must fail for -Inf on all 16.
  For both mutants NaN and decode controls must pass. pf3m03 removes
  both tc guards, including hooks, and must fail for both infinities on
  all eight tc cases, while the NaN and unrelated-group controls pass.
  Require the full set of rows and the expected assertion failures.
  Missing output, import/launch error, exception, or timeout is RED.
- G4. CPU smoke uses git archive HEAD and overlays the candidate test
  and the new fixture module in scratch, since the seat cannot commit.
  Everything else is from the archive, including the tracked F2
  ARM.json, without its binary. Use the default test environment with
  CUDA_VISIBLE_DEVICES unset, with order-required scratch and CPU
  resource limits. Require one host-index pass and 76 GPU skips out of
  77 collected tests. GPU runs the same file with the F3 package site on
  PYTHONPATH, checks the extension SHA again in the test process and
  requires all 77 passes. Archive SHA or embedded commit ID, overlay
  hashes, raw log and JUnit tallies are recorded.
- G5. Compare normalized sm_89 Function blocks from F2 and F3, with
  identical key sets and exactly 10 changed bulk tc kernels (two spq4
  D instantiations and eight spq6 D/variant instantiations, including
  those emitted but unreachable). Compare F3 with each prefill mutant
  for exactly the 32 prefill instantiations and with the tc mutant for
  exactly the 10 bulk tc instantiations. Run all 643 inherited P1
  packed-decode regression predicates on F3.
- G6. Hash the worktree before and after and retain a file inventory.
  Only the named three product files and authorized new outputs may
  differ. Treat the dispatcher-owned apa_pf2f3_r1.log as a live log.
  Nothing is deleted. No GPU run, git mutation, or other worktree write.
- CPU bounds. Empty CUDA_VISIBLE_DEVICES except G4 smoke. Single-thread
  env and 4,000,000,000-byte RLIMIT_AS. Builds use nice 19, at most -j4
  and fewer than 60 minutes per command. Disk rail is 40,000,000,000
  bytes free on /mnt/ForgeRealm. Scratch prefix /tmp/pf2f3_.
- GPU launchers use forge-gpu-run with max 540, margin 30, min at most
  510 and an explicit environment. Pin each participating module before
  lease. Every staged line requires dry exit 78 with would_start true
  and no lease attempt. Record each lead exit and log SHA.
- Acceptance requires G3 CPU, G4 CPU and G6 to pass, and the lead's real
  G1-G5 lines to pass. SASS is a CPU comparison. First real line is G2
  sink. At most three repairs of own scripts, with receipts and no
  weakening of gates.
- Prior art. Reuse PF2-F1's guard at the tc sites (Project-Tensor, 2026),
  IEEE 754-2008 quiet NaN, and REV2's negative-dot -Inf witness (2026).
  External clause unverified — lead to check "IEEE 754-2008 6.2 quiet
  NaN multiplication". Mutation testing follows DeMillo, Lipton and
  Sayward (1978); exact bits and SASS reuse PF2-F2 (2026). The guard's
  placement at the tc loads and device coverage of the new mutants are
  this round's extensions.
