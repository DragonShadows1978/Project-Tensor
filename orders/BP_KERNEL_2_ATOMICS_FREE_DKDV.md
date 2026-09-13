# BP-KERNEL-2 — a true reference gate, an atomics-free dK/dV pass, and the real step (2026-09-13)

Seat: Codex Astra (`gpt-6-astra`, reasoning high). Lead: Fable 5.1 (plans, runs the GPU cells, verifies, commits).
David's order (2026-09-13 ~10:00 EDT): "run BP-KERNEL-2, thats the obvious next one."
Parent: `/mnt/Shared/BP_Kernel_1_Result_2026-09-13.md` — `apa_selective_bwd` a = 411 ms, c (no dK/dV writes) = 86 ms, so 79 % of the
kernel is BF16 `atomicAdd` contention on dK/dV; b (FP32 atomic scratch) = 193 ms and d (b + output-dot) = 148 ms were RED under a
gate that took the noisy BF16-atomic kernel as truth. The successor registered there is this order; its predictions are immutable.

## Writable target (explicit grant)
YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/pt-bk2` (Project-Tensor worktree, branch `bp-kernel-2`, forked from `bp-kernel-1`
b7beddb — the a/b/c/d variants, harness `scripts/bp_kernel_1.py`, tests and receipts are already there). Edits AUTHORIZED: new kernel
variant(s) in `tensor_cuda/src/kernels.cu` + `ops.cpp` + `bindings.cpp` behind the existing explicit variant dispatch (**the default
path — variant a — stays byte-identical**; keep the BP-KERNEL-1 byte-range pin valid), `scripts/bp_kernel_2.py`, `scripts/bp_census_2.py`,
`tests/test_bp_kernel_2.py`, `artifacts/bp_kernel_2/`, `docs/BP_KERNEL_2_LEDGER.md`. CPU build AUTHORIZED (same recipe as BP-KERNEL-1,
build dir `tensor_cuda/build-bk2`). You MAY cherry-pick the timing-hook files from branch `pt-bp-census-1` (commit cd0c723:
`include/tc/bp_op_timing.h`, the `autograd.cpp` and `bindings.cpp` hunks) by copying them in — no git; record their sha256.
READ-ONLY: `/mnt/ForgeRealm/wt/grapa-bp1` (the BP-CENSUS-1 harness `scripts/bp_census_1.py`, its checkpoint symlink, receipts),
`/mnt/ForgeRealm/GRAPA-Native-LLM`, everything else. No git (lead commits). No subagents. **No GPU in this seat.** Foreground only,
every Bash call < 10 min. Never kill or signal a process you did not start. No network. Do not write checkpoints.

## Part (i) — a true reference gate (registered)
On the SAME pinned inputs and forward state as BP-KERNEL-1 (`artifacts/bp_kernel_1/inputs.npz` sha 53c38919…, `forward_state.npz`
lse/thr/O), compute an **FP64 CPU reference** of exactly the math the shipped kernel implements (read `kernels.cu:2523–2629` Pass B:
bulk score from kq, exact score from k when |bulk·scale| ≥ thr, p = exp(score − lse), rowdot = Σ p·(dO·v), dscore = p·(dov − rowdot),
dV_j += p·dO_i, dQ_i += dscore·scale·(k_j if selected else kq_j), dK_j += dscore·scale·q_i for selected only; causal mask as the
kernel). numpy FP64 with the selection mask materialised is fine at these shapes (H=16, 2048², D=96). Save `reference.npz` (sha pinned).
**Gate rule (registered):** for each of dQ/dK/dV and each metric (max-abs, relative-L2), tolerance = **2 × |a − reference|**, where a's
distance is measured from a fresh GPU run of variant a in the cell. A candidate farther than that from the reference is RED and its
timing does not count. Report a's own distance from the reference prominently — it is the first measurement of how far the shipped
kernel is from truth. (If a's distance is 0 on some metric the tolerance is 0 there; say so, do not add an epsilon.)

## Part (ii) — candidates (registered)
- **b** (already built): FP32 atomic scratch — re-judged against the reference.
- **f** (new): **atomics-free dK/dV**. Query-parallel kernel for dQ only (variant c's dQ path — c already computes a full valid dQ; reuse
  it, do not re-derive) plus a **key-parallel** kernel: one block per key row j (per b, kv-head), threads stride over queries i in the
  causal range, recompute score_ij from q_i and k_j/kq_j with the saved lse_i/thr_i, dov = dO_i·v_j, rowdot_i from a cheap
  query-parallel pre-pass (the output-dot identity dO_i·O_i, saved O — as d does; report whether that rowdot passes the gate on dQ, and
  if it does not, fall back to a Pass-A rowdot pre-pass and say so), then accumulate dV_j += p·dO_i and dK_j += dscore·scale·q_i in
  registers/shared memory and write each once. No atomics anywhere. FlashAttention-2 backward structure (Dao 2023; taken), APA
  selection is ours.
- **Prediction (immutable):** f ≤ 0.35 × a (≤ ~144 ms at a ≈ 411) **with dQ, dK, dV all inside the reference tolerance.**
  **Falsifier:** f green but > 0.35 × a → the scalar dot loops are the next target (tiling / tensor cores); f RED → fix before any timing.
- Timing exactly as BP-KERNEL-1: CUDA events around the bare op, 3 warm-ups + 10 measured, interleaved a b f (and d, for the record).

## Part (iii) — the real step (registered)
`scripts/bp_census_2.py`: re-run the BP-CENSUS-1 step census with the engine from `pt-bk2` (variant selectable by an explicit engine
call or env `TC_APA_BWD_VARIANT` read ONLY in the explicit dispatch — default a unchanged) — two instrumented arms in ONE cell,
**variant a then variant f**, 2 warm-ups + 5 measured each, same checkpoint/tokens/config as BP-CENSUS-1 (`/mnt/ForgeRealm/wt/grapa-bp1`
is read-only: import its harness module and override its ART/ENGINE/REG at import time, or re-implement the thin driver; the receipt goes
to `artifacts/bp_kernel_2/census/receipt.json`, create-only, with its own `registration.json` written before the run). The OFF control
is NOT repeated (overhead was 1.3 % in BP-CENSUS-1; say so). If variant f fails the Part (ii) gate, Part (iii) runs anyway but its
f-arm is labelled TIMING-ONLY / NOT A VALID STEP. **Prediction (immutable):** whole step with f ≤ 5.0 s (a-arm reproduces ≈ 11.8 s
within 10 %). **Falsifier:** > 5.0 s → the census component table says what remains.
If a clean Part (iii) is impossible without editing the read-only GRAPA harness, deliver Parts (i)+(ii) in full, and for (iii) a
BLOCKED report with the exact minimal diff the lead should route.

## Budget
Two GPU cells, lead-run, each ≤ 300 s work / ≤ 590 s lease under `flock /tmp/forge-gpu.lock`, single card, foreground, self-terminating
(inside David's standing bounded-run clearance). Incomplete → INCONCLUSIVE, never shorten shapes or samples. Create-only receipts.

## Deliverables
`scripts/bp_kernel_2.py` (`--reference` builds reference.npz on CPU — you run this; `--dry-run` CPU tiny-shape path; `--run` = cell 1),
`scripts/bp_census_2.py` (`--dry-run`, `--run` = cell 2), `artifacts/bp_kernel_2/registration.json` (pins: sources before/after,
reference.npz, inputs, predictions/falsifiers verbatim, budget; harness fails closed on drift), `tests/test_bp_kernel_2.py` (CPU:
reference math vs a tiny dense oracle on tiny shapes incl. causal + selection edge cases; gate rule on fabricated distances; variant
a byte-range still pinned; schema; create-only), `artifacts/bp_kernel_2/lead_commands.txt` (one per line, each accepting `--dry-run`),
`artifacts/bp_kernel_2/engine_build_receipt.json`, `docs/BP_KERNEL_2_LEDGER.md`.

## Done — paste verbatim in your final message
1. Variant f: kernel structure, code sites (line ranges), diff stat; confirmation variant a's pinned range is unchanged (sha).
2. Last line of `python3 -m pytest -q tests/test_bp_kernel_2.py`, of the build, of `--reference` (ends in reference.npz + sha), and of
   both `--dry-run` commands (each ends in a receipt path).
3. Registration sha256; the rowdot route f uses and why; whether Part (iii) is runnable or BLOCKED (with the minimal diff).
4. Prior art (FlashAttention-2 backward / Dao 2023; output-dot identity; taken vs ours; "unverified — lead to check" where you cannot verify).
5. Model + effort; confirmation: no GPU, no git, no subagents, nothing killed, no network, no edits outside the grants.
