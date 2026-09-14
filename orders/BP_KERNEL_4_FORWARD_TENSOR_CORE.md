# BP-KERNEL-4 — the APA forward train kernel on tensor cores (2026-09-13)

Seat: Codex Astra (`gpt-6-astra`, reasoning high). Lead: Fable 5.1 (plans, runs the GPU cells, verifies, commits).
David's order (2026-09-13 ~16:20 EDT): "run BP-KERNEL-4, Lets see what happens."
Parent: `/mnt/Shared/BP_Kernel_3_Result_2026-09-13.md` — backward g1 = 10.9 ms; real step 2.02 s at S=2048 of which initial forward
0.46 s + checkpoint replay 0.44 s (both through the still-scalar `apa_selective_fwd_train_kernel`, kernels.cu:2344) and
intermediate gradient accumulation 0.47 s. Real train.py at window 12288: 41.2 s/step. The successor registered there is this order.

## Writable target (explicit grant)
YOUR WRITABLE TARGET is `/mnt/ForgeRealm/wt/pt-bk4` (Project-Tensor worktree, branch `bp-kernel-4`, forked from `bp-kernel-3`
8d7e027 — g1/g2 and the BP-KERNEL-2/3 harnesses are there). Edits AUTHORIZED: a new forward-train variant in `tensor_cuda/src/kernels.cu`
+ `ops.cpp` + `bindings.cpp` behind an explicit variant switch (**the default forward path stays byte-identical; the existing backward
variant switch and the BP-KERNEL-1 byte-range pin must still hold**), `scripts/bp_kernel_4.py`, `scripts/bp_census_4.py`,
`tests/test_bp_kernel_4.py`, `artifacts/bp_kernel_4/`, `docs/BP_KERNEL_4_LEDGER.md`. CPU build AUTHORIZED (same recipe, `build-bk4`,
SM 89). Copy the gitignored pinned arrays from the canonical checkout as before (`/mnt/ForgeRealm/Project-Tensor/artifacts/bp_kernel_1/
{inputs,forward_state}.npz`, `/mnt/ForgeRealm/Project-Tensor/artifacts/bp_kernel_2/reference.npz`). READ-ONLY: `/mnt/ForgeRealm/GRAPA-Native-LLM`
(canonical; reference it repo-relative as `bp_census_3.py` does), everything else. No git, no subagents, **no GPU in this seat**, foreground,
every Bash call < 10 min, never kill or signal a process you did not start, no network, no checkpoint writes.

## The kernel
`apa_selective_fwd_train_kernel<T,DMAX,WCOOP>` (kernels.cu:2344, launched :2654): per query row it computes the bulk score against KQ for
every key, derives the per-row threshold `thr` from the refine percentile, computes the exact score against K for selected keys, the
online softmax, the output O, and saves `lse` and `thr` for the backward. Read it fully, including how `thr` is derived (the
percentile/selection rule is the APA contract — it must be preserved exactly in semantics; only the arithmetic route changes).
The inference kernel `apa_selective_kernel` (:1146) is OUT of scope (successor).

## Variant h (registered)
The forward, tiled and on tensor cores the way g1 is: query/key tiles through shared memory; WMMA 16×16×16 BF16 in / FP32 accumulate for
the bulk scores Q·KQᵀ and for P·V; the exact score Q·Kᵀ only for selected pairs (masked scalar path, as g1) — a dense-exact sibling h2 is
optional; the online softmax and the threshold derivation kept semantically identical. Output O, `lse`, `thr` written exactly as today so the
unchanged g1 backward consumes them. Prior art: FlashAttention / FlashAttention-2 forward (Dao 2022/2023) — taken; the APA selective
threshold tile is ours.

## Gates and predictions (immutable)
1. **Forward parity gate first.** FP64 CPU reference of the forward's exact math (bulk, threshold from the same percentile rule, exact
   for selected, softmax, O, lse) on the pinned inputs. Tolerance per array (O, lse) and metric = 2 × |a-forward − reference|. `thr`:
   report max-abs and the **selection flip rate** (fraction of (query,key) pairs whose selected/unselected status differs from the
   shipped forward); registered ceiling **0.5 %** of pairs. Over tolerance or over the flip ceiling → RED, timing does not count.
   **Also gate the downstream:** run the unchanged g1 backward on h's saved (lse, thr, O) and compare dQ/dK/dV against the FP64
   backward reference with the BP-KERNEL-2 tolerance — the forward is only green if the backward it feeds is still green.
2. Timing: CUDA events around the bare forward-train op, 3 warm-ups + 10 measured, interleaved a h (h2).
   **Prediction:** h ≤ **0.20 × a-forward** with all gates green. **Falsifier:** > 0.20 × → report which of tiling / MMA / threshold
   derivation is the residual (per-phase timing inside the kernel or per-half diagnostics REQUIRED in the receipt).
3. **The real step (cell 2):** `scripts/bp_census_4.py` — same protocol (two arms in one cell: **g1-backward + a-forward** then
   **g1-backward + h-forward**, 2 + 5 steps each, same checkpoint/tokens/config), receipts to `artifacts/bp_kernel_4/census/`.
   **Prediction:** initial forward + checkpoint replay ≤ **0.25 s** combined (from 0.90) and whole step ≤ **1.3 s**. **Falsifier:**
   the census component table says what remains (if the APA forward turns out to be less than half of the forward, say so with the numbers —
   that is a legitimate outcome, not a failure of the seat).
4. Budget: two lead-run cells, each ≤ 300 s work / ≤ 590 s lease, single card, self-terminating; incomplete → INCONCLUSIVE.

## Deliverables
`scripts/bp_kernel_4.py` (`--reference` builds the forward FP64 reference — you run it on CPU; `--dry-run`; `--run` = cell 1),
`scripts/bp_census_4.py` (`--dry-run`, `--run` = cell 2), `artifacts/bp_kernel_4/registration.json` (pins, predictions verbatim, budget;
fail-closed on drift), `tests/test_bp_kernel_4.py` (CPU: forward reference vs a tiny dense oracle incl. the percentile-threshold rule,
causal and tile-boundary cases; flip-rate and gate logic on fabricated data; default forward region pinned unchanged; schema; create-only),
`artifacts/bp_kernel_4/lead_commands.txt` (each line accepts `--dry-run`), `artifacts/bp_kernel_4/engine_build_receipt.json`,
`docs/BP_KERNEL_4_LEDGER.md`.

## Done — paste verbatim in your final message
1. Variant h: tile shapes, fragments, how the threshold is derived inside the tile without changing its semantics, code sites, diff stat;
   confirmation the default forward path and the BP-KERNEL-1 pinned range are unchanged (shas).
2. Last lines of `pytest -q tests/test_bp_kernel_4.py`, the build, `--reference`, both `--dry-run` commands.
3. Registration sha256; the predictions verbatim; whether h2 was built.
4. Prior art (FlashAttention 2022 / FA-2 2023 forward; WMMA; taken vs ours; "unverified — lead to check" where you cannot verify).
5. Model + effort; confirmations: no GPU, no git, no subagents, nothing killed, no network, no edits outside the grants.
