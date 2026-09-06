DISPATCH CONTEXT (lead, 2026-09-06): You are the implementation seat for APA-SP1.1, continuing your own SP1 work. YOUR WRITABLE TARGET is this worktree: /mnt/ForgeRealm/Project-Tensor-wt-apa-sp1 (branch apa-sp1, HEAD 759c07c = your SP1 work committed by the lead + this order). Never touch /mnt/ForgeRealm/Project-Tensor (main) or other worktrees. Your sandbox has NO GPU: build and CPU-verify; deliver every GPU gate as a leased, bounded, runnable script plus a blocked-report for the lead. The SP1 GPU receipts you did not have are now under artifacts/apa_sp1/gpu/ in this worktree (51 classes, lead-run). NO git. NO subagents. NO background waits (foreground with explicit timeout, each call under 10 minutes; python3 -c 'import time;time.sleep(n)' for gaps). Never kill a process you did not start. Registration immutable; amendments separate. State your model id and effort in the final report. The order follows verbatim.

# APA-SP1.1 — Split-K single pass for decode; adjudicate the baseline gate; finalize the scoring

Lead-authored 2026-09-06 from the APA-SP1 GPU sweep (lead-run; all 51
classes worker exit 0; receipts under `artifacts/apa_sp1/gpu/`,
summary via `scripts/apa_sp1_lead_gpu.sh summary`). SP1's results:
prefill 1.09–1.90× with deviation-vs-dense ratio 0.50–0.90 of the
two-pass kernel at matched refine fraction; **decode 0.06–0.41×** —
a one-block single pass loses to the existing split-K decode path;
one G2 FAIL where the BASELINE two-pass kernel misses its own emulator
(0.0016 > 0.001) on `prefill_s2048_d64_c1_h4_kv4` while SP matches to
5e-7; several decode rows `matched=False` (refine-fraction matching
failed); the summary's prediction fields still read PENDING.

YOUR WRITABLE TARGET is this worktree
(`/mnt/ForgeRealm/Project-Tensor-wt-apa-sp1`, branch `apa-sp1`, HEAD
= the SP1 commit). Same boundary as SP1. Your sandbox has NO GPU:
build, CPU-verify, deliver leased bounded scripts + blocked-reports;
the lead runs the card.

## Part A — Split-K single pass (the decode fix)

Claim to prove first, then build: the running-max rule is monotone
under key-range partitioning. Each partition p walks its keys with its
own prefix max `m_p` and decides `refine iff bulk_j ≥ m_p − δ`; the
global max `m ≥ m_p`, so a key skipped in p (bulk < m_p − δ ≤ m − δ)
is still skipped under the global rule — decisions never become wrong
on merge; refined-but-unnecessary keys cost work only. Write the
argument, pin it with a CPU test (random partitions of random keys:
partition-wise selection ⊇ global-rule selection; outputs equal after
the online-softmax merge within tolerance). Then implement
`apa_selective_sp_splitk_kernel` + merge, mirroring the existing
split-K structure for decode shapes (L=1) and the sink variant, behind
the same `TC_APA_SP` flag, existing kernels still hash-pinned. Register
the partition count rule (derive from the existing split-K launcher;
no new constant).

## Part B — Adjudicate the baseline gate failure

On `prefill_s2048_d64_c1_h4_kv4` the existing two-pass kernel differs
from its CPU emulator by max_abs 0.0016. Decide which is right and
why: fp32 accumulation order in the stats pass, the z-score threshold
sitting on a bulk-score boundary (a key exactly at `thr` flipping
between kernel and emulator), causal-bound handling, or GQA head
mapping. Show the diagnosis with the specific key(s) whose refine
decision differs, from receipts you can reconstruct on CPU with the
seeded inputs. Do NOT change the existing kernel; do NOT widen the
tolerance. If the emulator is wrong, fix the emulator and re-pin; if
the kernel's arithmetic order is the cause, say so and register the
class as EMULATOR_ORDER_SENSITIVE with the evidence.

## Part C — Refine-fraction matching on decode and the scoring

Explain every `matched=False` row (decode S=2048/8192/32768 rows) —
why the SP fraction at the registered δ did not land within the
matching tolerance of the z fraction — and whether that invalidates the
speed/deviation comparison on those rows (state it plainly either
way). Then finalize `summary` so every registered prediction (lead's
P2/P3 and your A3/A4/A5) reads HIT/MISS from the receipts instead of
PENDING, with the receipt path per verdict.

## Gates

Register first (`artifacts/apa_sp1/registration_sp1_1.json`,
IMMUTABLE, citing SP1's registration sha
`12059d49f39abe9450b34989f06ac5ca8a736d80e3127c9d41c1ed97e650c9fe`):
Part A partition rule + predictions (lead: split-K SP reaches ≥ 0.8×
the two-pass decode speed on S ≥ 8192 and ≥ 1.0× on S=32768, with
deviation ratio ≤ 1.0 at matched fraction), Part B hypothesis ranking,
Part C expectations.
G1 (CPU, you): existing SP1 tests + Part A monotone/merge test +
Part B diagnosis test green; hash pins hold.
G2/G3 (GPU, lead): equivalence of the split-K SP kernel vs its
emulator on the decode classes; re-sweep of the 24 decode classes
(reuse `scripts/apa_sp1_lead_gpu.sh`, add a `splitk` target); Part B
class re-run with the corrected emulator if that is the verdict.
Evidence class: kernel sweep. Say "this establishes nothing about
model quality" where the numbers are.

## Rules (binding)

NO git. NO subagents. NO background waits (foreground with explicit
`timeout`, every call under 10 minutes; `python3 -c 'import time;
time.sleep(n)'` for gaps). Never kill a process you did not start.
Thresholds/δ/tolerances registered, never adjusted after; the
existing kernel bodies byte-identical. Ledger as you go in
`docs/APA_SP1_LEDGER.md`. RED honesty.

## Done (verbatim)

1. Part A: the monotone-under-partition argument, the CPU pin, the
   kernel files/lines, the registered partition rule.
2. Part B: the diagnosis with the specific keys, the verdict, what
   changed (emulator only) or the registered sensitivity.
3. Part C: the `matched=False` explanation per row and the finalized
   prediction table with receipt paths.
4. G1 results; G2/G3 blocked-reports with the exact lead commands.
5. Registration sha; deviations; residual risks; RED; process safety;
   model id and effort.
